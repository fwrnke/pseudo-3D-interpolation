"""
Utility function to apply **forward** FFT along specified axis of 3D cube (netCDF).

"""
import os
import sys
import yaml
import argparse
import warnings
import datetime

import numpy as np
import xarray as xr
import xrft
import dask
from dask.diagnostics import ProgressBar

from pseudo_3D_interpolation.functions.utils import xprint, show_progressbar

xr.set_options(keep_attrs=True)  # preserve metadata during computations

# %% FUNCTIONS
# fmt: off
def define_input_args():  # noqa
    parser = argparse.ArgumentParser(
        description='Apply FFT along time axis of (pseudo-)3D cube.')
    parser.add_argument('path_cube', type=str,
                        help='Input path of 3D cube')
    parser.add_argument('--params_netcdf', type=str, required=True,
                        help='Path of netCDF parameter file (YAML format).')
    parser.add_argument('--prefix', type=str, default='freq',
                        help='Prefix for new netCDF variable and coordinate.')
    parser.add_argument('--compute_real', action='store_true',
                        help='Compute FFT assuming real input and thus discarting redundant negative frequencies.')
    parser.add_argument('--upsampling-factor', type=int, default=1,
                        help='Increase resolution of FFT by `upsampling-factor`.')
    # filter options
    parser.add_argument('--filter', type=str, default=None, choices=['lowpass', 'highpass', 'bandpass'],
                        help='Optional filter to apply prior to FFT computation.')
    parser.add_argument('--filter_freqs', type=int, nargs='+', help='Filter corner frequencies (in Hz).')
    parser.add_argument('--drop-filtered-freq', action='store_true', help='Drop filtered frequency samples.')
    #
    parser.add_argument('--verbose', '-V', type=int, nargs='?', default=0, const=1, choices=[0, 1, 2],
                        help='Level of output verbosity (default: 0)')
    return parser
# fmt: on


def _get_stopband(nstopband: int, kind: str):
    stopband_size = nstopband * 2
    stopband_size += 1 if stopband_size % 2 == 0 else 0
    _slice = (
        slice(1, stopband_size // 2 + 1)  # highpass
        if kind == 'highpass'
        else slice(stopband_size // 2, -1)  # lowpass
    )

    return np.hanning(stopband_size)[_slice]


def _get_const_values(kind: str):
    if kind == 'highpass':
        c = (0, 1)
    elif kind == 'lowpass':
        c = (1, 0)
    elif kind == 'bandpass':
        c = (0, 0)

    return c


def get_freq_filter_win(
    filter_freqs: list,
    frequencies: xr.DataArray,
    dim: str = 'freq_twt',
    filter_type: str = 'lowpass',
) -> xr.DataArray:
    """
    Calculate filter window (`highpass`, `lowpass`, or `bandpass`) in frequency domain.

    Parameters
    ----------
    filter_freqs : list
        Cut-off frequencies as [fmin, fmax] for `highpass` or `lowpass`
        and [f1, f2, f3, f4] for `bandpass`. Must be same units as `frequencies` (e.g., kHz).
    frequencies : xr.DataArray
        DataArray coordinate of frequencies. Must be same units as `filter_freqs`(e.g., kHz).
    filter_type : str, optional
        Filter type in frequency domain (default: `lowpass`)

    Returns
    -------
    xr.DataArray
        Filter window with values in range [0, 1] that can be used for multiplication.

    """
    if filter_type in ['lowpass', 'highpass']:
        fmin = min(filter_freqs)
        fmax = max(filter_freqs)

        constant_values = _get_const_values(kind=filter_type)

        # get sample numbers for each interval
        n_lower = np.count_nonzero(frequencies < fmin)
        n_stopband = np.count_nonzero((frequencies >= fmin) & (frequencies <= fmax))
        n_higher = np.count_nonzero(frequencies > fmax)
        # print(n_lower, n_stopband, n_higher)

        # create full bandpass stopband
        stopband = _get_stopband(n_stopband, kind=filter_type)

    elif filter_type == 'bandpass':
        filter_freqs.sort()
        f1, f2, f3, f4 = filter_freqs

        constant_values = _get_const_values(filter_type)

        # get sample numbers for each interval
        n_lower = np.count_nonzero(frequencies < f1)
        n_stopband_low = np.count_nonzero((frequencies >= f1) & (frequencies <= f2))
        n_stopband = np.count_nonzero((frequencies > f2) & (frequencies < f3))
        n_stopband_high = np.count_nonzero((frequencies >= f3) & (frequencies <= f4))
        n_higher = np.count_nonzero(frequencies > f4)
        # print(n_lower, n_stopband_low, n_stopband, n_stopband_high, n_higher)

        # get stopband for lower freqs
        stopband_low = _get_stopband(n_stopband_low, kind='highpass')
        # print('stopband_low', stopband_low.shape)

        # get stopband for higher freqs
        stopband_high = _get_stopband(n_stopband_high, kind='lowpass')
        # print('stopband_high', stopband_high.shape)

        # create full bandpass stopband
        stopband = np.hstack((stopband_low, np.ones((n_stopband,)), stopband_high))
        # print('stopband', stopband.shape)

    # create full filter window
    filter_window = np.pad(
        stopband, pad_width=(n_lower, n_higher), mode='constant', constant_values=(constant_values,)
    )

    return xr.DataArray(filter_window, dims=[dim], coords={dim: frequencies.data})


def get_freq_filter_mask(
        da: xr.DataArray,
        dim: str,
        freqs: list,
        filter_type: str = 'lowpass'
) -> xr.DataArray:
    """
    Return mask of filtered frequency samples.

    Parameters
    ----------
    da : xr.DataArray, xr.Dataset
        Input data (in frequency domin).
    dim : str
        Name of frequency dimension (default: 'freq_{NAME}').
    freqs : list
        List of filter frequencies as [fmin, fmax] (lowpass, highpass) or [f1, f2, f3, f4] (bandpass).
    filter_type : str, optional
        Filter type (default: 'lowpass'). Available options are 'lowpass', 'higpass' and 'bandpass'.

    Returns
    -------
    xr.DataArray
        Boolean mask DataArray (for multiplication with input data).

    """
    filter_freqs = sorted(freqs)
    if filter_type == 'lowpass':
        assert len(freqs) == 2, 'Please provide filter frequencies as [fmin, fmax]'
        return da[dim] <= filter_freqs[-1]
    elif filter_type == 'highpass':
        assert len(freqs) == 2, 'Please provide filter frequencies as [fmin, fmax]'
        return da[dim] >= filter_freqs[0]
    elif filter_type == 'bandpass':
        assert len(freqs) == 4, 'Please provide filter frequencies as [f1, f2, f3, f4]'
        return np.logical_and(da[dim] >= filter_freqs[0], da[dim] <= filter_freqs[-1])


def main(argv=sys.argv, return_dataset=False):  # noqa
    """Apply FFT along _time_ axis wrapper function."""
    TODAY = datetime.date.today().strftime('%Y-%m-%d')
    SCRIPT = os.path.splitext(os.path.basename(__file__))[0]

    parser = define_input_args()
    args = parser.parse_args(argv[1:])  # exclude filename parameter at position 0
    
    path_cube = args.path_cube
    dir_work, filename = os.path.split(path_cube)
    basename, suffix = os.path.splitext(filename)
    fout = basename.replace('twt', f'{args.prefix}')
    fout += f'_up-{args.upsampling_factor}' if args.upsampling_factor > 1 else ''
    fout += '-trunc' if args.drop_filtered_freq else ''
    path_cube_freq = os.path.join(dir_work, fout + suffix)

    prefix = f'{args.prefix}_'

    # load netCDF metadata
    with open(args.params_netcdf, 'r') as f_attrs:
        kwargs_nc = yaml.safe_load(f_attrs)

    # (1) Open cube dataset
    cube_time2freq = xr.open_dataset(path_cube, chunks='auto', engine='h5netcdf')

    # get parameter names
    dim = [d for d in list(cube_time2freq.dims) if d not in ['iline', 'xline']][0]
    var = [v for v in list(cube_time2freq.data_vars) if v not in ['fold', 'amp_ref']][0]
    var_new = f'{prefix}{var}'  # 'freq'
    dim_new = f'{prefix}{dim}'

    # rechunk
    chunks = {dim: -1, 'iline': 15, 'xline': -1}
    cube_time2freq = cube_time2freq.chunk(chunks)

    attrs_var = {}

    # (2) Compute FFT along time axis (twt, il, xl) --> (freq, il, xl)
    xprint('Compute FFT along time axis', kind='info', verbosity=args.verbose)
    dim_size = cube_time2freq[dim].size
    if dim_size % 2 != 0:
        warnings.warn(
            (
                f'Selected dim `{dim}` has odd length ({dim_size}), ',
                'which causes issues for inverse FFT. Last slice will be removed!',
            )
        )
        dim_slice = slice(0, dim_size - 1)
    else:
        dim_slice = slice(None)
    
    # get shape of FFT output (optionally: higher resolution determined by `upsampling-factor`)
    shape = {dim: args.upsampling_factor * cube_time2freq[var][dim_slice][dim].size}
    history_reso = f' FACTOR x{args.upsampling_factor}' if args.upsampling_factor > 1 else ''
    
    # apply FFT
    cube_freq = (
        xrft.fft(  # don't use window function --> incorrect amplitude preservation
            cube_time2freq[var][dim_slice],
            dim=dim,
            real_dim=dim if args.compute_real else None,
            shift=False,
            true_phase=True,
            true_amplitude=True,
            prefix=prefix,
            chunks_to_segments=False,
            shape=shape,
        )
        .astype('complex64')
        .to_dataset(name=var_new)
    )

    # add fold DataArray
    cube_freq = cube_freq.assign(fold=cube_time2freq.fold)

    # (3) apply frequency domain filter
    if args.filter is not None:
        if args.filter_freqs is None:
            raise ValueError('Filter frequencies must be specified!')

        # convert filter_freqs if neccessary
        units = cube_time2freq[dim].attrs.get('units')
        divisor = 1000 if units == 'ms' else 1
        filter_freqs = [f / divisor for f in args.filter_freqs]

        xprint(
            f'Apply > {args.filter} < filter ({"/".join([str(round(f * divisor)) for f in filter_freqs])} Hz) in frequency domain',
            kind='info', verbosity=args.verbose,
        )
        filter_window = get_freq_filter_win(
            filter_freqs, frequencies=cube_freq[dim_new], dim=dim_new, filter_type=args.filter
        )

        # cube_freq[var_new] *= filter_window  # TODO: remove
        cube_freq[var_new] = (cube_freq[var_new] * filter_window).astype('complex64')
        
        # OPTIONAL: drop filtered frequency samples (i.e. slices) to reduce disk space
        if args.drop_filtered_freq and args.filter == 'lowpass':
            cube_freq[dim_new].attrs['nfft'] = cube_freq[dim_new].size  # store original data size for later reconstruction
            mask_filtered = get_freq_filter_mask(cube_freq, dim_new, freqs=filter_freqs, filter_type=args.filter)
            cube_freq = cube_freq.drop_dims('freq_twt').assign(
                {f'{var_new}': cube_freq[var_new].where(mask_filtered, drop=True).astype('complex64')}
            )
        elif args.drop_filtered_freq and args.filter != 'lowpass':
            warnings.warn(f'Filter type `{args.filter}` does not support dropping of frequency slices')

        # update metadata
        _filter_freq_str = '/'.join(str(f) for f in args.filter_freqs)
        attrs_var = {'filter': args.filter, 'filter_freq_Hz': _filter_freq_str}
        history_filter = f' {args.filter.upper()} ({_filter_freq_str} Hz)'

    # (4) update & assign attributes
    cube_freq.attrs = cube_time2freq.attrs
    cube_freq.attrs.update(
        {
            'long_name': cube_time2freq.attrs["long_name"] + ' (frequency domain)',
            'description': cube_time2freq.attrs["description"] + ' (frequency domain)',
            'history': cube_time2freq.attrs["history"]
            + f'{SCRIPT}: FFT({var}){history_reso}{history_filter if args.filter is not None else ""};',
            'text': cube_time2freq.attrs.get('text', '')
            + f'\n{TODAY}: FFT(TIME){history_reso}{history_filter if args.filter is not None else ""}'
        }
    )
    cube_freq[var_new].attrs.update({'original_var': var})
    if kwargs_nc is not None:
        cube_freq[var_new].attrs.update(kwargs_nc['attrs_freq'].get('data', {}))  # data attributes
        cube_freq[var_new].attrs.update(attrs_var)  # add filter attributes
        cube_freq[dim_new].attrs.update(
            kwargs_nc['attrs_freq'].get('new_dim', {})
        )  # new dim attributes
    
    # (5) Write full volume to netCDF (il, xl, freq)
    with dask.config.set(scheduler='threads'), show_progressbar(ProgressBar(), verbose=args.verbose):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            cube_freq.to_netcdf(path=path_cube_freq, engine='h5netcdf', invalid_netcdf=True)

    if return_dataset:
        return cube_freq


#%% MAIN
if __name__ == '__main__':

    main()
