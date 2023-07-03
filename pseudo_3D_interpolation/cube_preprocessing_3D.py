"""
Utility script to apply various processing functions to 3D cube.
These include:
    
  - amplitude **gain** (time-variant)
  - amplitude **balancing** (time-invariant)
  - **frequency filtering** (bandpass, lowpass, highpass)
  - **resampling** (up, down)
  - trace **envelope** calculation

"""
import os
import sys
import yaml
import argparse
import re
import datetime

import numpy as np
import xarray as xr
from scipy.signal import resample, resample_poly
import dask
from dask.diagnostics import ProgressBar

from pseudo_3D_interpolation.functions.utils import xprint, show_progressbar, ffloat, convert_twt
from pseudo_3D_interpolation.functions.filter import bandpass_filter, lowpass_filter, highpass_filter
from pseudo_3D_interpolation.functions.signal import calc_reference_amplitude, gain, get_resampled_twt, envelope

xr.set_options(keep_attrs=True)  # preserve metadata during computations

FILTER = {
    'bandpass': bandpass_filter,
    'lowpass': lowpass_filter,
    'highpass': highpass_filter,
}

#%% CLASS

class ParseGainArguments(argparse.Action):
    """Parse gain arguments into dict."""
    
    def __call__(self, parser, namespace, args, option_string=None):  # noqa
        setattr(namespace, self.dest, dict())
        for arg in args:
            key, val = arg.split('=')
            if key.lower() == 'linear':
                from ast import literal_eval
                val = literal_eval(val)
            elif key.lower() == 'pgc':
                val = val.strip('((').strip('))').split('),(')
                val = {float(s.split(',')[0]): float(s.split(',')[1]) for s in val}
            else:
                val = float(val)
            getattr(namespace, self.dest)[key] = val

#%% FUNCTIONS
# fmt: off
def define_input_args():  # noqa
    parser = argparse.ArgumentParser(
        description='Apply pre-processing alogrithms to (pseudo-)3D cube.')
    parser.add_argument('path_cube', type=str,
                        help='Input path of 3D cube')
    parser.add_argument('--path_out', type=str,
                        help='Output path of pre-processed 3D cube')
    parser.add_argument('--fsuffix', type=str, default='preproc', help='Optional filename suffix')
    parser.add_argument('--params_netcdf', type=str, required=True,
                        help='Path of netCDF parameter file (YAML format).')
    # gain options
    parser.add_argument('--gain', nargs='*', default=None, action=ParseGainArguments,
                        help='Parameter for (time-variant) gain function(s).')
    parser.add_argument('--use_samples', action='store_true',
                        help='Use samples instead of TWT for (time-variant) gain function(s).')
    parser.add_argument('--balance', type=str, nargs='?', const='rms', choices=['rms', 'max'],
                        help='Method to define reference amplitude for (time-invariant) scaling.')
    parser.add_argument('--store_ref_amp', action='store_true',
                        help='Store reference amplitude used for trace balancing as netCDF variable.')
    # filter options
    parser.add_argument('--filter', type=str, default=None, choices=['lowpass', 'highpass', 'bandpass'],
                        help='Optional filter to apply prior to FFT computation.')
    parser.add_argument('--filter_freqs', type=int, nargs='+',
                        help='Filter corner frequencies (in Hz).')
    # resampling options
    parser.add_argument('--resampling_function', type=str, default='resample_poly',
                        choices=['resample', 'resample_poly'],
                        help='Resampling function from scipy.signal.')
    parser.add_argument('--resampling_interval', '-dt', type=float,
                        help='Output sampling interval/period of signal (in ms).')
    parser.add_argument('--resampling_frequency', '-fs', type=float,
                        help='Output sampling frequency/rate of signal (in Hz).')
    parser.add_argument('--resampling_factor', '-f', type=float,
                        help='Resampling factor (<1: upsampling, >1: downsampling).')
    parser.add_argument('--window_resample', type=str, default='hann',
                        help='Window function for resampling (from scipy.signal.windows).')
    # envelope
    parser.add_argument('--envelope', action='store_true',
                        help='Calculate trace envelope.')
    #
    parser.add_argument('--verbose', '-V', type=int, nargs='?', default=0, const=1, choices=[0, 1, 2],
                        help='Level of output verbosity (default: 0)')
    return parser
# fmt: on

def main(argv=sys.argv, return_dataset=False):  # noqa
    """Preprocess 3D cube wrapper function."""
    TODAY = datetime.date.today().strftime('%Y-%m-%d')
    SCRIPT = os.path.splitext(os.path.basename(__file__))[0]

    parser = define_input_args()
    args = parser.parse_args(argv[1:])  # exclude filename parameter at position 0
    xprint(args, kind='debug', verbosity=args.verbose)
    
    path_cube = args.path_cube
    dir_work, filename = os.path.split(path_cube)
    basename, suffix = os.path.splitext(filename)
    
    fsuffix = 'AGC' if args.gain and 'agc' in '\t'.join(args.gain) else args.fsuffix
    path_cube_proc = os.path.join(dir_work, f'{basename}_{fsuffix}.nc')
    
    # sanity checks
    if args.resampling_interval is not None:
        xprint('Using `resampling_interval` for resampling', kind='info', verbosity=args.verbose)
        resampling_interval = args.resampling_interval
        resampling_factor = None
    elif args.resampling_frequency is not None:
        xprint('Using `resampling_frequency` for resampling', kind='info', verbosity=args.verbose)
        resampling_interval = (1 / args.resampling_frequency) * 1000
        resampling_factor = None
    elif args.resampling_factor is not None:
        xprint('Using `resampling_factor` for resampling', kind='info', verbosity=args.verbose)
        resampling_interval = None
        resampling_factor = args.resampling_factor

    # (0) Open cube dataset
    cube = xr.open_dataset(path_cube, chunks='auto', engine='h5netcdf')

    # get parameter names
    dim = [d for d in list(cube.dims) if d not in ['iline', 'xline']][0]
    core_dims = (['xline', dim], )  # if 'agc' in '\t'.join(args.gain) else (['xline', dim], )
    var = [v for v in list(cube.data_vars) if v != 'fold'][0]
    var_ref = f'{var}_ref'
    xprint(f'dim:       {dim}', kind='debug', verbosity=args.verbose)
    xprint(f'core_dims: {core_dims}', kind='debug', verbosity=args.verbose)
    xprint(f'var:       {var}', kind='debug', verbosity=args.verbose)
    
    # rechunk
    # chunks = {dim: -1, 'iline': 1, 'xline': 10} if 'agc' in '\t'.join(args.gain) else {dim: -1, 'iline': 15, 'xline': -1}
    chunks = {dim: -1, 'iline': 15, 'xline': -1}
    cube = cube.chunk(chunks)
    xprint(f'chunks:    {chunks}', kind='debug', verbosity=args.verbose)
    
    twt_s = convert_twt(cube[var].twt.values, cube.twt.attrs.get('units', 'ms'), 's')
    dt = cube[dim].attrs.get('dt', np.median(np.diff(cube[dim].data)))  # sampling rate (ms)
    n_samples = cube[dim].size

    # init output cube
    cube_proc = cube.copy()
    
    # load netCDF metadata
    if args.params_netcdf is not None:
        with open(args.params_netcdf, 'r') as f_attrs:
            kwargs_nc = yaml.safe_load(f_attrs)
    else:
        kwargs_nc = None
    
    var_new = 'env' if args.envelope else var
    attrs_var = kwargs_nc['attrs_time'][var_new] if kwargs_nc is not None else cube[var].attrs
    _history = f'{SCRIPT}:'
    _text = f'{TODAY}: '
    
    # (1) balance traces (time-invariant)
    if args.balance is not None:
        balance = args.balance
        xprint(f'Balance traces using < {balance} > amplitude for scaling (time-invariant)',
               kind='info', verbosity=args.verbose)
        
        # compute reference amplitude per trace
        kwargs_func = dict(scale=balance)
        ref_amplitudes = cube[var].reduce(
            calc_reference_amplitude, dim='twt', keep_attrs=True, **kwargs_func
        )
        cube_proc[var] = (cube[var] / ref_amplitudes)
        
        # update metadata
        attrs_var.update({'balanced': f'{balance} amplitude'})
        _history += f' amplitude balancing ({balance}),'
        _text += 'BALANCE.'
        
        if args.store_ref_amp:
            cube_proc[var_ref] = ref_amplitudes
            cube_proc[var_ref].attrs = {
                'description': 'Reference amplitudes used to scale traces',
                'method': f'{balance} scaling',
                'units': cube[var].attrs.get('units', '-'),
            }
            
    # (2) apply time-variant gain function
    if args.gain is not None:
        xprint(f'Apply time-variant gain function using {"samples" if args.use_samples else "TWT"} ({", ".join(args.gain)})',
               kind='info', verbosity=args.verbose)
        
        kwargs_gain = args.gain
        xprint(args.gain, kind='debug', verbosity=args.verbose)
        # kwargs_gain = dict([
        #     (
        #         p.split('=')[0],
        #         # p.split('=')[1]
        #         bool(p.split('=')[1]) if p.split('=')[1].lower() in ['true', 'false']
        #         else (
        #             tuple(p.split('=')[1]) if p.split('=')[0].lower() == 'linear'
        #             else (
        #                 dict(p.split('=')[1]) if p.split('=')[0].lower() == 'pgc'
        #                 else float(p.split('=')[1])
        #             )
        #         )
        #     ) for p in kwargs_gain
        # ])
        # xprint(args.gain, kind='debug', verbosity=args.verbose)
        kwargs_gain['twt'] = np.arange(twt_s.size) if args.use_samples else twt_s
        
        # apply gain function(s)
        cube_proc[var] = xr.apply_ufunc(
            gain,
            cube_proc[var],
            input_core_dims=core_dims,  # (['twt', 'xline'], ),
            output_core_dims=core_dims,  # (['twt', 'xline'], ),
            keep_attrs=True,
            kwargs=kwargs_gain,
            dask='parallelized',
            output_dtypes=cube_proc[var].dtype,
        )
        
        # update metadata
        kwargs_gain_str = ' '.join([
            f'{key}={val}' for key, val in kwargs_gain.items() if key != 'twt'
        ])
        kwargs_gain_str += ' (sample-based)' if args.use_samples else ' (TWT-based)'
        attrs_var.update({'gain': kwargs_gain_str})
        _history += f' amplitude gain ({kwargs_gain_str}),'
        _text += 'GAIN.'

    # (3) FILTER (FREQUENCY)
    if args.filter is not None:
        if args.filter_freqs is None:
            raise ValueError('Filter frequencies must be specified!')
        else:
            filter_freqs_str = [str(f) for f in args.filter_freqs]
        
        kwargs_filter = dict(
            freqs=args.filter_freqs,  # cutoff frequencies (f_low, f_high)
            fs=1 / (dt / 1000),       # sampling frequency (Hz)
            axis=-1,                  # time axis
        )
        xprint('kwargs_filter:', kwargs_filter, kind='debug', verbosity=args.verbose)
        
        xprint(f'Apply > {args.filter} < filter ({"/".join(filter_freqs_str)} Hz)', kind='info', verbosity=args.verbose)
        cube_proc[var] = xr.apply_ufunc(
            FILTER.get(args.filter),
            cube_proc[var],
            input_core_dims=core_dims,  # (['xline', dim], ),  # time axis has to be last
            output_core_dims=core_dims,  # (['xline', dim], ),
            keep_attrs=True,
            kwargs=kwargs_filter,
            dask='parallelized',
            output_dtypes=cube_proc[var].dtype,
        )
        
        # update metadata
        _filter_freq_str = '/'.join(str(f) for f in args.filter_freqs)
        attrs_var.update({'filter': args.filter,
                          'filter_freq_Hz': _filter_freq_str})
        _history += f' {args.filter} ({_filter_freq_str} Hz),'
        _text += f'{args.filter.upper()} ({_filter_freq_str} Hz).'
    
    # (4) RESAMPLING
    if any([a is not None for a in [
            args.resampling_interval, args.resampling_frequency, args.resampling_factor
    ]]):
        # setup resampling parameter
        if resampling_interval is not None:
            resampling_factor = resampling_interval / dt
        elif resampling_factor is not None:
            resampling_interval = resampling_factor * dt
        n_resamples = int(np.ceil(n_samples / resampling_factor))
        
        # setup resampling function
        if args.resampling_function == 'resample':
            resample_func = resample
            kwargs_resample = dict(num=n_resamples, t=None, window=args.window_resample, axis=-1)
        elif args.resampling_function == 'resample_poly':
            up = 1 / resampling_factor if resampling_factor < 1 else 1
            down = resampling_factor if resampling_factor > 1 else 1
            resample_func = resample_poly
            kwargs_resample = dict(up=up, down=down, window=args.window_resample, axis=-1)
        xprint('kwargs_resample:', kwargs_resample, kind='debug', verbosity=args.verbose)
    
        xprint(f'Resample > {dim} < from > {dt} < to > {resampling_interval} < ms', kind='info', verbosity=args.verbose)
        
        # init resampled cube
        cube_resampled = cube_proc.drop_vars(var)
        cube_resampled = cube_resampled.assign_coords(
            {dim: get_resampled_twt(cube_proc[dim].data, n_resamples, n_samples)}
        )  # assign resampled dimension
        cube_resampled[dim].attrs = cube_proc[dim].attrs  # copy metadata
        cube_resampled[dim].attrs.update(dt=resampling_interval)
        
        # resample traces
        cube_resampled[var] = xr.apply_ufunc(
            resample_func,
            cube_proc[var],
            input_core_dims=core_dims,  # [['xline', dim], ],  # time axis has to be last
            output_core_dims=core_dims,  # [['xline', dim], ],
            exclude_dims=set((dim,)),            # allow dimension to change size
            keep_attrs=True,
            kwargs=kwargs_resample,
            dask='parallelized',
            dask_gufunc_kwargs={
                'output_sizes': {dim: n_resamples, 'xline': cube_proc['xline'].size}
            },
            output_dtypes=cube_proc[var].dtype,
        )
        cube_resampled[dim] = np.around(cube_resampled[dim].astype('float64'), 3)
        
        # update metadata
        cube_resampled[dim].attrs.update({'resampled': 'True',
                                          'dt_original': dt})
        _history += f' resampling (factor: {resampling_factor}),'
        _text += 'RESAMPLE.'
        
        # update output filename
        path_cube_proc = re.sub(
            r'\d{1,2}\+\d{0,3}(ms)',  # noqa
            ffloat(resampling_interval).replace('.', '+') + 'ms',
            path_cube_proc
        )
        
        cube_out = cube_resampled
    else:
        cube_out = cube_proc
        
    # (5) calculate trace envelope
    if args.envelope:
        xprint('Calculate trace envelope (instantaneous amplitude)', kind='info', verbosity=args.verbose)
                
        cube_out[var] = xr.apply_ufunc(
            envelope,
            cube_out[var],
            input_core_dims=core_dims,  # [['xline', dim], ],
            output_core_dims=core_dims,  # [['xline', dim], ],
            keep_attrs=True,
            dask='parallelized',
            output_dtypes=cube_out[var].dtype,
        )
        cube_out = cube_out.rename_vars({var: var_new})  # update variable name in Dataset
        
        # update history
        _history += ' trace envelope,'
        _text += 'ENV.'
        
        # update output filename
        path_cube_proc = path_cube_proc.replace(var, var_new)
    
    # (6) add/update metadata
    xprint('Update netCDF metadata attributes', kind='info', verbosity=args.verbose)
    cube_out.attrs.update({
        'history': cube_out.attrs["history"] + f'{_history[:-1]};',  # remove trailing comma
        'text': cube_out.attrs.get("text") + f'\n{_text[:-1]}',  # remove trailing period
    })
    cube_out[var_new].attrs = attrs_var
    
    # (7) write balanced cube to disk
    xprint('Write output data to file', kind='info', verbosity=args.verbose)
    path_cube_proc = (args.path_out if args.path_out is not None else path_cube_proc)
    with dask.config.set(scheduler='threads'), show_progressbar(ProgressBar(), verbose=args.verbose):
        cube_out.transpose(dim, 'iline', 'xline').to_netcdf(
            path=path_cube_proc,
            engine='h5netcdf',
        )
    
    if return_dataset:
        return cube_out.transpose(dim, 'iline', 'xline'), cube

# %% MAIN
if __name__ == '__main__':
    
    main()
