"""
Utility script to apply **inverse** FFT (IFFT) along specified axis of 3D cube (netCDF).

"""
import os
import sys
import yaml
import argparse
import datetime

import xarray as xr
import xrft
import dask
from dask.diagnostics import ProgressBar

from pseudo_3D_interpolation.functions.utils import xprint, show_progressbar, rescale_dask

# %% FUNCTIONS
# fmt: off
def define_input_args():  # noqa
    parser = argparse.ArgumentParser(
        description='Apply inverse FFT along frequency axis of (pseudo-)3D cube.')
    parser.add_argument('path_cube', type=str,
                        help='Input path of 3D cube.')
    parser.add_argument('--params_netcdf', type=str, required=True,
                        help='Path of netCDF parameter file (*.yaml).')
    parser.add_argument('--compute_real', action='store_true',
                        help='Compute IFFT assuming real input was used for previously applied FFT.')
    parser.add_argument('--rescale-envelope', action='store_true', help='Rescale envelope data to [0-1].')
    parser.add_argument('--verbose', '-V', type=int, nargs='?', default=0, const=1, choices=[0, 1, 2],
                        help='Level of output verbosity (default: 0)')
    return parser
# fmt: on


def main(argv=sys.argv, return_dataset=False):  # noqa
    """Apply inverse FFT along _frequency_ axis wrapper function."""
    xr.set_options(keep_attrs=True)
    TODAY = datetime.date.today().strftime('%Y-%m-%d')
    SCRIPT = os.path.splitext(os.path.basename(__file__))[0]

    parser = define_input_args()
    args = parser.parse_args(argv[1:])  # exclude filename parameter at position 0

    path_cube_freq_interp = args.path_cube
    dir_work, file = os.path.split(path_cube_freq_interp)

    # load netCDF metadata
    with open(args.params_netcdf, 'r') as f_attrs:
        kwargs_nc = yaml.safe_load(f_attrs)

    # (1) Open cube dataset
    cube_freq2time = xr.open_dataset(
        path_cube_freq_interp,
        chunks='auto',
        engine='h5netcdf',
    )

    # extract dimension for IFFT
    dim = [d for d in list(cube_freq2time.dims) if d not in ['iline', 'xline']][0]
    # extract prefix
    prefix = dim.split('_')[0]
    # use ``original`` variable name if stored as metadata
    var = [v for v in list(cube_freq2time.data_vars) if prefix in v][0]
    var = cube_freq2time[var].attrs.get(
        'original_var', f'{"_".join(var.split(".")[0].split("_")[1:])}'
    )

    # rechunk using detected dimension
    chunks = {dim: -1, 'iline': 20, 'xline': -1}
    cube_freq2time = cube_freq2time.chunk(chunks)

    # restore complex array (from merged float32 netCDF variables)
    var_names = list(cube_freq2time.data_vars)
    var_real = [v for v in var_names if 'real' in v]
    var_imag = [v for v in var_names if 'imag' in v]
    if len(var_real) > 0 and len(var_imag) > 0:
        cube_freq2time[var] = cube_freq2time[var_real[0]] + cube_freq2time[var_imag[0]] * 1j
        cube_freq2time = cube_freq2time.drop_vars((var_real[0], var_imag[0]))

    # (2) Compute IFFT along frequency axis (il, xl, freq) --> (il, xl, twt)
    # compute IFFT along time axis
    cube_time = (
        xrft.ifft(
            cube_freq2time[var],
            dim=dim,
            real_dim=dim if args.compute_real else None,
            shift=True,
            true_phase=True,
            true_amplitude=True,
        )
        .astype('float32')
        .to_dataset(name=var)
    )

    # add fold DataArray
    # cube_time = cube_time.assign(fold=cube_freq2time.fold)
    cube_time['fold'] = cube_freq2time['fold']

    # fix rounding errors..
    cube_time = cube_time.assign_coords(
        {'twt': ('twt', cube_time.coords['twt'].data.astype('float32'), cube_time.coords['twt'].attrs)}
    )

    # update & assign attributes
    cube_time.attrs = cube_freq2time.attrs
    cube_time.attrs.update(
        {
            'long_name': cube_freq2time.attrs["long_name"].split(' (')[0] + ' (interpolated)',
            'history': cube_freq2time.attrs["history"] + f'{SCRIPT}: IFFT({var});',
            'text': cube_freq2time.attrs.get('text', '') + f'\n{TODAY}: INVERSE FFT(FREQ -> TIME)'
        }
    )
    if kwargs_nc is not None:
        cube_time[var].attrs.update(kwargs_nc['attrs_time'].get(var.split('_')[0], {}))
        cube_time['twt'].attrs.update(kwargs_nc['attrs_time'].get('twt', {}))
        cube_time['twt'].attrs['dt'] = float(f"{cube_time['twt'].attrs['spacing']:g}")
        cube_time['twt'].attrs.pop('spacing')
        
    # OPTIONAL: rescale to [0, 1]
    if args.rescale_envelope:
        # clip minimum values < 0
        cube_time[var] = (cube_time[var].dims, dask.array.where(cube_time[var] < 0, 0, cube_time[var]))
        
        # compute global min/max values
        amin, amax = dask.compute(cube_time[var].min(), cube_time[var].max())
        amin, amax = amin.values, amax.values
        xprint(f'amin: {amin}', kind='debug', verbosity=args.verbose)
        xprint(f'amax: {amax}', kind='debug', verbosity=args.verbose)
        
        cube_time[var] = xr.apply_ufunc(
            rescale_dask,
            cube_time[var],
            input_core_dims=[['twt'], ],
            output_core_dims=[['twt'], ],
            keep_attrs=True,
            dask='parallelized',
            output_dtypes=cube_time[var].dtype,
            kwargs={'amin': amin, 'amax': amax},  # use global min/max values
        ).transpose('twt', ...)

    # (3) Write full volume (il, xl, twt) to netCDF
    tsuffix = '_rescale-env' if args.rescale_envelope else ''
    basename, fsuffix = os.path.splitext(file)
    path_cube_time_interp = os.path.join(
        dir_work, basename.replace(prefix, 'twt') + f'_interp-freq{tsuffix}{fsuffix}'
    )
    xprint('Compute inverse FFT along time axis', kind='info', verbosity=args.verbose)
    with dask.config.set(scheduler='threads'), show_progressbar(
        ProgressBar(), verbose=args.verbose
    ):
        cube_time.to_netcdf(path=path_cube_time_interp, engine='h5netcdf')

    if return_dataset:
        return cube_time


# %% MAIN
if __name__ == '__main__':
    
    main()
