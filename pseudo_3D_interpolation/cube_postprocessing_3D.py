"""
Utility script to apply various (post-)processing functions to 3D cube.
These include:
    
  - iline/xline upsampling (to equal bin size after interpolation)
  - acquisition footprint removal (inline, crossline, or both)
  - smoothing of frequency/time slices (noise removal)
  - AGC (Automatic Gain Control)

"""
import os
import sys
import re
import argparse
import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import xarray as xr
import dask
from dask.diagnostics import ProgressBar
import scipy.signal as signal
from scipy.ndimage import gaussian_filter, median_filter

from pseudo_3D_interpolation.functions.utils import xprint, rescale, convert_twt, show_progressbar
from pseudo_3D_interpolation.functions.signal import get_AGC_samples, AGC

#%% FUNCTIONS

# fmt: off
def define_input_args():  # noqa
    parser = argparse.ArgumentParser(description='Apply post-processing algorithm to (pseudo-)3D cube.')
    parser.add_argument('path_cube', type=str, help='Input path of 3D cube')
    parser.add_argument('--path_out', type=str, help='Output path of pre-processed 3D cube')
    # UPSAMPLING
    parser.add_argument(
        '--upsample', nargs='?', type=str, const='linear', choices=['linear', 'nearest', 'slinear', 'cubic', 'polynomial'],
        help='Upsample `xarray.Dataset` to equal bin size along ilines and xlines.'
    )
    parser.add_argument(
        '--spatial-dealiasing', action='store_true', help='Whether to apply filter in kx-ky domain to remove spatial aliasing.'
    )
    # FOOTPRINT
    parser.add_argument(
        '--remove-footprint', nargs='?', const='slice', choices=['slice', 'profile', 'profile-iline', 'profile-xline'],
        help='Remove acquisition footprint.'
    )
    parser.add_argument(
        '--direction', choices=['both', 'iline', 'xline', 'twt'],
        help="Direction of acquisition footprint removal filter (default: `'both'`)."
    )
    parser.add_argument(
        '--footprint-sigma', type=int, default=7, help='Standard deviation for smoothing Gaussian filter (default: `7`) to remove footprint.'
    )
    parser.add_argument(
        '--buffer-center', type=float, default=0.20, help='Percentual buffer (0-1) around center in kx-ky domain (default: `0.20`).'
    )
    parser.add_argument(
        '--buffer-filter', type=int, default=3, help='Footprint filter buffer size (in grid cells).'
    )
    # FILTER
    parser.add_argument(
        '--smooth', nargs='?', choices=['gaussian', 'median'], help='Smooth slices (frequency or time domain).'
    )
    parser.add_argument(
        '--smooth-sigma', type=int, default=1, help='Standard deviation for Gaussian kernel.'
    )
    parser.add_argument(
        '--smooth-size', type=int, default=3, help='Shape of Median kernel (identical for iline and xline).'
    )
    parser.add_argument(
        '--rescale', nargs='*', default=None, type=float,
        help='Rescale smoothed slices to given percentile range (without arguments: [0.01, 99.99]).'
    )
    # AGC
    parser.add_argument('--agc', action='store_true', help='Apply Automatic Gain Control (AGC).')
    parser.add_argument('--agc-win', type=float, help='AGC window length (in seconds).')
    parser.add_argument('--agc-kind', type=str, default='rms', choices=['rms', 'mean', 'median'], help='AGC kind.')
    parser.add_argument('--agc-sqrt', action='store_true', help='Whether to compute squared AGC (enhances strong amplitudes).')
    #
    parser.add_argument('--verbose', '-V', type=int, nargs='?', default=0, const=1, choices=[0, 1, 2],
                        help='Level of output verbosity (default: 0)')
    return parser
# fmt: on


def smoothing_filter(
        x: np.ndarray,
        filter_name: str = None,
        kwargs_filter: dict = None,
        rescale_slice: bool = False,
        kwargs_rescale: dict = None
) -> np.ndarray:
    """
    Wrap `scipy.ndimage` filter.

    Parameters
    ----------
    x : np.ndarray
        Input 2D slice.
    filter_name : str, optional
        Name of 2D filter function, either `gaussian_filter` (default) or `median_filter`.
    kwargs_filter : dict, optional
        Filter kwargs.
    rescale_slice : bool, optional
        Whether to rescale slice to range (0, 1).
    kwargs_rescale : dict, optional
        Rescale kwargs.

    Returns
    -------
    np.ndarray
        Filtered (and optionally rescaled) input slice.

    """
    FILTER = {'gaussian': gaussian_filter, 'median': median_filter}
    func = FILTER.get(filter_name)

    if rescale_slice:
        vmin, vmax = np.percentile(x, sorted(kwargs_rescale['vminmax']))
        return rescale(func(x, **kwargs_filter), vmin=vmin, vmax=vmax)
    else:
        return func(x, **kwargs_filter)


def gaussian_kernel_2d(
        sigma: int = 7,
        n: int = None,
        normalized: bool = True,
        orientation: str = 'equal'
):
    """
    Generate 2D Gaussian kernel (n x n) using standard deviation `sigma`.

    Parameters
    ----------
    sigma : int, optional
        Standard deviation (default: 7).
    n : int, tuple(int, int), optional
        Kernel size (in grid cells).
        If `int`, single size for both iline and xline direction.
        if `tuple(ny, nx)`, different sizes for iline and xline direction.
    normalized : bool, optional
        Whether to normalize the output kernel (default: True).

    Returns
    -------
    kernel_2D : np.ndarray
        2D kernel (optionally normalized).
    
    References
    ----------
    [^1]: [https://gist.github.com/thomasaarholt/267ec4fff40ca9dff1106490ea3b7567](https://gist.github.com/thomasaarholt/267ec4fff40ca9dff1106490ea3b7567)

    """
    if isinstance(n, tuple):
        ny, nx = n
    else:
        ny = nx = n
    
    factor = {'equal': (8, 8), 'iline': (2, 8), 'xline': (8, 2)}
    
    ny = sigma * factor[orientation][0] + 1 if ny is None else ny
    ny = ny + 1 if ny % 2 == 0 else ny
    
    nx = sigma * factor[orientation][1] + 1 if nx is None else nx
    nx = nx + 1 if nx % 2 == 0 else nx
    
    kernel_y = signal.windows.gaussian(ny, sigma)
    kernel_x = signal.windows.gaussian(nx, sigma)
    kernel_2D = np.outer(kernel_y, kernel_x)
    # kernel_2D = np.outer(signal.windows.gaussian(sigma * 2 + 1, sigma), signal.windows.gaussian(n, sigma))
    if normalized:
        kernel_2D /= 2 * np.pi * (sigma**2)
    return kernel_2D


def remove_acquisition_footprint(
        data: np.ndarray,
        sigma: int = 7,
        direction: str = 'both',
        buffer_center: float = 0.25,
        buffer_filter: int = 3,
        return_filter: bool = False,
        dims: tuple = ('iline', 'xline'),
        verbose: int = 1,
):
    """
    Remove acquisition footprint from data (iline/xline slice).

    Parameters
    ----------
    data : np.ndarray
        Input data slice (frequency or time).
    sigma : int, optional
        Standard deviation to create smoothing Gaussian filter (default: `7`).
    direction : str, optional
        Direction of filter (default: `'both'`). One of 'both', 'iline', 'xline', or 'twt'.
    buffer_center : float, optional
        Percentual buffer around center in kx-ky domain (default: `0.25`).
    buffer_filter : int, optional
        Buffer size (in grid cells).
    return_filter : bool, optional
        Whether to return filter grid (default: `False`).
    verbose : int, optional
        Level of verbosity (default: `1`).

    Returns
    -------
    data_filt : np.ndarray
        Filtered input data slice.

    """
    ny, nx = data.shape
    npad = sigma * 5
    ny_pad = ny + npad
    nx_pad = nx + npad
    xprint('data:', data.shape, kind='debug', verbosity=verbose)
    
    # init 2D Gaussian kernel (for smooting filter)
    kernel = gaussian_kernel_2d(sigma=sigma)
    xprint('kernel:', kernel.shape, kind='debug', verbosity=verbose)
    
    # create padded filter array
    filter_shape = np.zeros((ny_pad, nx_pad), dtype='int8')
    xprint('filter_shape:', filter_shape.shape, kind='debug', verbosity=verbose)
    
    if direction == 'iline':
        direction = 'horizontal' if dims[0] == 'iline' else 'vertical'
    elif direction == 'xline':
        direction = 'vertical' if dims[1] == 'xline' else 'horizontal'
    elif direction == 'twt':
        direction = 'vertical' if ny > nx else 'horizontal'
    
    xprint(f'direction: >>> {direction} <<<', kind='debug', verbosity=verbose)
    if direction in ['both', 'horizontal']:  # iline
        cidx = nx_pad // 2 + 1
        fwidth = round(ny_pad * (1 - buffer_center) + .5) // 2
        filter_shape[ :fwidth, cidx - buffer_filter: cidx + buffer_filter + 1] = 1  # noqa
        filter_shape[-fwidth:, cidx - buffer_filter: cidx + buffer_filter + 1] = 1  # noqa
    if direction in ['both', 'vertical']:  # xline
        cidx = ny_pad // 2 + 1
        fwidth = round(nx_pad * (1 - buffer_center) + .5) // 2
        filter_shape[cidx - buffer_filter: cidx + buffer_filter + 1,  :fwidth] = 1  # noqa
        filter_shape[cidx - buffer_filter: cidx + buffer_filter + 1, -fwidth:] = 1  # noqa
    xprint('fwidth:', fwidth, kind='debug', verbosity=verbose)
    
    # convolve filter with Gaussian kernel
    ffilter = signal.fftconvolve(filter_shape, kernel, mode='same')
    # unpad, rescale and invert filter
    ffilter = 1 - rescale(ffilter[npad // 2 : -npad // 2, npad // 2 : -npad // 2])
    
    data_filt = np.fft.ifft2(np.multiply(np.fft.ifftshift(ffilter), np.fft.fft2(data))).real
    xprint('data_filt:', data_filt.shape, kind='debug', verbosity=verbose)
    
    if return_filter:
        return data_filt, ffilter
    
    return data_filt


def spatial_antialiasing(
        data: np.ndarray,
        direction: str,
        factors_upsampling: dict,
        sigma: int = 7,
        dims: tuple = ('iline', 'xline'),
        return_filter: bool = False,
        verbose: int = 1,
):
    """
    Apply spatial de-aliasing in kx-ky domain following iline/xline upsamling.

    Parameters
    ----------
    data : np.ndarray | xr.DataArray
        Input data slice (frequency or time).
    direction : str
        Direction of filter. One of ['iline', 'xline'].
    factors_upsampling : dict
        Dictionary of upsampling factors for 'iline' and 'xline'.
    sigma : int, optional
        Standard deviation to create smoothing Gaussian filter (default: `7`).
    dims : tuple, optional
        Iterable of iline/xline coordinate names. Must be identical to `factors_upsampling` keys.
        The default is ('iline', 'xline').
    return_filter : bool, optional
        Whether to return filter grid (default: `False`).
    verbose : int, optional
        Level of verbosity (default: `1`).

    Returns
    -------
    data_filt: np.ndarray | xr.DataArray
        De-aliased input data.

    """
    il, xl = dims
    
    if not sorted(dims) == sorted(factors_upsampling.keys()):
        raise ValueError(f'Coordinates {dims} not found in `factors_upsampling` {factors_upsampling.keys()}')
    
    ny, nx = data.shape
    npad = sigma * 5
    ny_pad = ny + npad
    nx_pad = nx + npad
    p = 0.98  # reduce `half_width` to 98% of actual width
    
    # init 2D Gaussian kernel (for smooting filter)
    kernel = gaussian_kernel_2d(sigma=sigma)
    xprint('kernel:', kernel.shape, kind='debug', verbosity=verbose)
    
    # create padded filter array
    filter_shape = np.zeros((ny_pad, nx_pad), dtype='int8')
    xprint('filter_shape:', filter_shape.shape, kind='debug', verbosity=verbose)
    
    if direction == 'iline':
        direction = 'horizontal' if dims[0] == 'iline' else 'vertical'
    elif direction == 'xline':
        direction = 'vertical' if dims[1] == 'xline' else 'horizontal'
    
    xprint('direction:', direction, kind='debug', verbosity=verbose)
    if direction == 'horizontal':
        perc = 1 - factors_upsampling.get(xl, 1) / factors_upsampling.get(il, 1)
        half_width = round(ny * perc * p) // 2 + npad
        filter_shape[half_width : -half_width, :] = 1
    elif direction == 'vertical':
        perc = 1 - factors_upsampling.get(il, 1) / factors_upsampling.get(xl, 1)
        half_width = round(nx * perc * p) // 2 + npad
        filter_shape[:, half_width : -half_width] = 1
    xprint('perc:', perc, kind='debug', verbosity=verbose)
    xprint('half_width:', half_width, kind='debug', verbosity=verbose)
        
    # convolve filter with Gaussian kernel
    ffilter = signal.fftconvolve(filter_shape, kernel, mode='same')
    xprint('ffilter.shape:', ffilter.shape, kind='debug', verbosity=verbose)
    # unpad, rescale and invert filter
    ffilter = rescale(ffilter[npad // 2 : -npad // 2, npad // 2 : -npad // 2], vmin=1e-3, vmax=1)
    xprint('ffilter.shape:', ffilter.shape, kind='debug', verbosity=verbose)
    
    data_filt = np.fft.ifft2(np.multiply(np.fft.ifftshift(ffilter), np.fft.fft2(data))).real
    
    if return_filter:
        return data_filt, ffilter
    
    return data_filt


def upsample_ilxl(
        ds,
        coords: tuple = ('iline', 'xline'),
        method: str = 'linear',
        update_attrs: bool = True,
        spatial_dealiasing: bool = True,
        return_factor: bool = False,
        verbose: int = 1
):
    """
    Upsample xarray.Dataset to equal bin size along ilines and xlines.
    Requires empty (i.e. previously omitted) iline/xline indices!

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    coords : tuple, optional
        Iterable of iline/xline coordinate names. The default is ('iline', 'xline').
    method: str, optional
        The method used to interpolate (default: `'linear'`).
        Check `xarray.Dataset.interp_like` for all available options.
    update_attrs : bool, optional
        Whether to update the attribute metadata (default: `True`).
    spatial_dealiasing: bool, optional
        Whether to apply filter in kx-ky domain to remove spatial aliasing.
    return_factor: bool, optional
        Whether to return upsampling factor dictionary (default: False).
    verbose: int, optional
        Level of verbosity (default: `1`).

    Raises
    ------
    ValueError
        Raise error when metadata is missing in input dataset.

    Returns
    -------
    ds_interp : xr.Dataset
        Copy of upsampled input dataset with equal bin sizes.

    """
    il, xl = coords
    
    diff_ilines = ds.coords[il].diff(il).data[0]  # get INLINE step
    interp_ilines = (diff_ilines != 1)  # whether to interpolate missing INLINES
    
    diff_xlines = ds.coords[xl].diff(xl).data[0]  # get CROSSLINE step
    interp_xlines = (diff_xlines != 1)  # whether to interpolate missing CROSSLINES
    
    if not interp_ilines and not interp_xlines:
        xprint(
            f'No missing/omitted `{coords[0]}` or `{coords[1]}` indices. Returning input dataset.',
            kind='warning', verbosity=verbose
        )
        return ds, dict(iline=diff_ilines, xline=diff_xlines)
    
    # init new ilines/xlines
    _ilines = np.arange(ds[il][0], ds[il][-1] + 1, 1) if interp_ilines else ds[il]
    _xlines = np.arange(ds[xl][0], ds[xl][-1] + 1, 1) if interp_xlines else ds[xl]
    
    # create upsampled coordinate dataset
    ds_other = xr.Dataset(coords={il: _ilines, xl: _xlines})
    
    # copy attributes
    ds_other[il].attrs = ds[il].attrs
    ds_other[xl].attrs = ds[xl].attrs

    if update_attrs:
        bin_il = ds_other[il].attrs.get('bin_il')
        bin_xl = ds_other[xl].attrs.get('bin_xl')
        
        if all([bin_il, bin_xl]):
            # coordinate attributes
            (
                ds_other[il].attrs.update({'bin_il': type(bin_xl)(bin_il / diff_xlines)})  # ensure indentical dtypes
                if interp_xlines else ds_other[il].attrs
            )
            (
                ds_other[xl].attrs.update({'bin_xl': type(bin_il)(bin_xl / diff_ilines)})  # ensure indentical dtypes
                if interp_ilines else ds_other[xl].attrs
            )
            # global attributes
            attrs_update = {
                'bin_size_iline': type(bin_xl)(bin_il / diff_xlines),
                'bin_size_xline': type(bin_il)(bin_xl / diff_ilines)
            }
        else:
            raise ValueError('Could not find metadata `bin_il` and/or `bin_xl`')
    
    # upsample resolution
    ds_interp = ds.interp_like(ds_other, method=method, assume_sorted=True)
    if update_attrs:
        ds_interp.attrs.update(attrs_update)  # update global attributes
        
    if spatial_dealiasing:
        data_vars = [var for var in list(ds_interp.data_vars) if ds_interp[var].ndim == 3]
        # init filtered dataset
        filtered = xr.Dataset(coords=ds_interp.coords)
        
        kwargs = dict(
            direction='iline' if interp_ilines else 'xline',
            factors_upsampling=dict(iline=diff_ilines, xline=diff_xlines),
            sigma=7,
            verbose=verbose,
        )
        
        # de-aliase data variable(s)
        for var in data_vars:
            filtered[var] = xr.apply_ufunc(
                spatial_antialiasing,
                ds_interp[var],
                input_core_dims=[coords],
                output_core_dims=[coords],
                kwargs=kwargs,
                keep_attrs=update_attrs,
                vectorize=True,
                dask='parallelized',
                output_dtypes=[ds_interp[var].dtype]
            )
        
        # copy non-data variable(s)
        nodata_vars = set(ds_interp.data_vars).symmetric_difference(
            set([var for var in ds_interp.data_vars if ds_interp[var].ndim == 3])
        )
        for var in nodata_vars:
            filtered[var] = ds_interp[var]
        
        # copy global attributes
        filtered.attrs = ds_interp.attrs
        
        ds_out = filtered
    else:
        ds_out = ds_interp
        
    if return_factor:
        return ds_out, dict(iline=diff_ilines, xline=diff_xlines)
    
    return ds_out


def main(argv=sys.argv, return_dataset=False):  # noqa
    """Preprocess 3D cube wrapper function."""
    TODAY = datetime.date.today().strftime('%Y-%m-%d')
    SCRIPT = os.path.splitext(os.path.basename(__file__))[0]

    parser = define_input_args()
    args = parser.parse_args(argv[1:])
    args.rescale = [0.01, 99.99] if args.rescale == [] else args.rescale
    xprint(args, kind='debug', verbosity=args.verbose)
    
    if (args.agc or args.remove_footprint == 'profile') and (args.remove_footprint == 'slice' or args.upsample or args.smooth):
        xprint(
            (
                'The option `--agc`/`--remove_footprint profile` and `--remove_footprint slice`/`--upsampling`/`--filter`'
                ' are mutually exclusive as they require different chunk sizes. Please run this script twice instead.'
            ), kind='error', verbosity=args.verbose
        )
        return
      
    path_cube = args.path_cube
    dir_work, filename = os.path.split(path_cube)
    basename, suffix = os.path.splitext(filename)
      
    # (0) Open cube dataset
    cube = xr.open_dataset(path_cube, chunks='auto', engine='h5netcdf')

    # get parameter names
    dim = [d for d in list(cube.dims) if d not in ['iline', 'xline']][0]
    data_vars = [var for var in list(cube.data_vars) if cube[var].ndim == 3]
    nodata_vars = list(set(cube.data_vars).symmetric_difference(
        set([var for var in cube.data_vars if cube[var].ndim == 3])
    ))
    core_dims = (['iline', 'xline'],)
    xprint(f'dim:         {dim}', kind='debug', verbosity=args.verbose)
    xprint(f'data_vars:   {data_vars}', kind='debug', verbosity=args.verbose)
    xprint(f'nodata_vars: {nodata_vars}', kind='debug', verbosity=args.verbose)
    
    # open with appropriate chunks
    if args.agc:
        chunks = {'iline': 1, 'xline': 100, dim: -1}
    # elif args.remove_footprint == 'slice':
    #     chunks = {'iline': -1, 'xline': -1, dim: 1}
    # elif args.remove_footprint == 'profile':
    elif (args.remove_footprint is not None) and ('profile' in args.remove_footprint):
        if 'iline' in args.remove_footprint:
            chunks = {'iline': 1, 'xline': -1, dim: -1}
        elif 'xline' in args.remove_footprint:
            chunks = {'iline': -1, 'xline': 1, dim: -1}
        else:
            SEL_ILINES = cube['iline'].size < cube['xline'].size
            chunks = {
                'iline': 1 if SEL_ILINES else -1,
                'xline': -1 if SEL_ILINES else 1,
                dim: -1
            }
    else:
        chunks = {'iline': -1, 'xline': -1, dim: 1}
    cube.close()  # close file before re-opening with determined chunk sizes
        
    cube = xr.open_dataset(path_cube, chunks=chunks, engine='h5netcdf')
    xprint(f'chunks: {chunks}', kind='debug', verbosity=args.verbose)
    # print(cube)
    # print(cube.__dask_graph__())
    
    cube_proc = cube
    _history = f'{SCRIPT}:'
    _text = f'{TODAY}: '
    text_suffix = ''
           
    # ========== inline/crossline upsampling (to equal bin sizes) ==========
    if args.upsample is not None:
        xprint('Upsample iline/xline bins to equal size', kind='info', verbosity=args.verbose)
        cube_proc, factor = upsample_ilxl(
            cube_proc, method=args.upsample, spatial_dealiasing=args.spatial_dealiasing,
            return_factor=True, verbose=args.verbose
        )
        if factor['iline'] != factor['xline']:
            _history += ' iline/xline bin size upsampling,'
            _text += 'UPSAMPLING.'
            text_suffix += '_upsampled'
            bin_size_iline = cube.attrs['bin_size_iline'] / factor['xline']
            bin_size_iline = f"{bin_size_iline:.0f}" if bin_size_iline % 1 == 0 else f"{bin_size_iline}"
            bin_size_xline = cube.attrs['bin_size_xline'] / factor['iline']
            bin_size_xline = f"{bin_size_xline:.0f}" if bin_size_xline % 1 == 0 else f"{bin_size_xline}"
            bin_size_str = f'{bin_size_iline.replace(".","+")}x{bin_size_xline.replace(".","+")}m'
            basename = re.sub(r'_\d{1}\+?\d{0,2}x\d{1}\+?\d{0,2}m_', f'_{bin_size_str}_', basename)
    
    
    # ========== acquisition footprint removal ==========  # noqa
    if args.remove_footprint:
        xprint('Remove acquisition footprint', kind='info', verbosity=args.verbose)
        # cube_proc = xr.Dataset(coords=cube.coords, attrs=cube.attrs)
        
        if args.direction is None:
            if args.remove_footprint == 'slice':
                ratio = cube['iline'].attrs.get('bin_il') / cube['xline'].attrs.get('bin_xl')
                direction = 'both' if ratio == 1 else 'iline' if ratio < 1 else 'xline'
            elif 'profile' in args.remove_footprint:
                direction = 'twt'
                dim_direction = 'iline' if chunks['iline'] < chunks['xline'] else 'xline'
                xprint('dim_direction:', dim_direction, kind='debug', verbosity=args.verbose)
            xprint(f'Detected footprint direction: `{direction}`', kind='info', verbosity=args.verbose)
        else:
            direction = args.direction
        
        core_dims = (['iline', 'xline'],) if args.remove_footprint == 'slice' else ([dim_direction, dim],)
        xprint('core_dims:', core_dims, kind='debug', verbosity=args.verbose)
        
        kwargs = dict(
            direction=direction,
            sigma=args.footprint_sigma,
            buffer_center=args.buffer_center,
            buffer_filter=args.buffer_filter,
            dims=core_dims[0] if isinstance(core_dims[0], (list, tuple)) else core_dims,
            verbose=args.verbose,
        )
        xprint(kwargs, kind='debug', verbosity=args.verbose)
        
        for var in data_vars:
            # kwargs.update(dims=tuple(d for d in cube[var].dims if d != dim))
            # xprint(kwargs, kind='debug', verbosity=args.verbose)
            
            cube_proc[var] = xr.apply_ufunc(
                remove_acquisition_footprint,
                cube_proc[var],  # cube[var],
                input_core_dims=core_dims,
                output_core_dims=core_dims,
                kwargs=kwargs,
                keep_attrs=True,
                vectorize=True,
                dask='parallelized',
                output_dtypes=[cube[var].dtype]
            ).transpose(dim, 'iline', 'xline')
        _history += f' footprint removal ({args.remove_footprint}: {direction}),'
        _text += 'FOOTPRINT REMOVAL.'
        text_suffix += '_footprint-profile' if 'profile' in args.remove_footprint else '_footprint'
        text_suffix += '-il' if 'iline' in args.remove_footprint else '-xl' if 'xline' in args.remove_footprint else ''
        
    
    # ========== smoothing filter (frequency/time slice) ==========  # noqa
    if args.smooth:
        xprint(f'args.rescale:  {args.rescale}', kind='debug', verbosity=args.verbose)
        
        if args.smooth == 'gaussian':
            kwargs_smooth_str = 'sigma={args.smooth_sigma}'
            kwargs_smooth = dict(filter_name=args.smooth, kwargs_filter=dict(sigma=args.smooth_sigma))
        elif args.smooth == 'median':
            kwargs_smooth_str = 'size={args.smooth_size}'
            kwargs_smooth = dict(filter_name=args.smooth, kwargs_filter=dict(size=args.smooth_size))
            
        if args.rescale:
            kwargs_smooth.update(rescale_slice=True, kwargs_rescale=dict(vminmax=args.rescale))
        xprint(f'kwargs_smooth:  {kwargs_smooth}', kind='debug', verbosity=args.verbose)
        
        for var in data_vars:
            cube_proc[var] = xr.apply_ufunc(
                smoothing_filter,
                cube_proc[var],
                input_core_dims=core_dims,
                output_core_dims=core_dims,
                vectorize=True,
                dask='parallelized',
                output_dtypes=[cube_proc[var].dtype],
                keep_attrs=True,
                kwargs=kwargs_smooth,
            )
        _history += f' {args.smooth} filter ({kwargs_smooth_str}),'
        _text += f'{args.smooth.upper()} FILTER.'
        text_suffix += f'_{args.smooth}'
        text_suffix += f'-{args.smooth_sigma}' if args.smooth == 'gaussian' else f'-{args.smooth_size}'
        text_suffix += f'_rescale-{"-".join([str(i) for i in args.rescale])}' if args.rescale is not None else ''
    
    
    # ========== AGC ==========  # noqa
    if args.agc:
        if dim != 'twt':
            xprint(f"Input data must be in time domain (dim='twt') and not dim={dim}", kind='error', verbosity=args.verbose)
            return
        
        if args.agc_win is None:
            xprint('AGC window length (`--agc-win`) is required!', kind='error', verbosity=args.verbose)
            return
        
        cube_proc = xr.Dataset(coords=cube_proc.coords, attrs=cube.attrs)
        dt = cube_proc[dim].attrs.get('dt', np.median(np.diff(cube[dim].values)))  # sampling interval (ms)
        dt = convert_twt(dt, cube_proc[dim].attrs.get('units', 'ms'), 's')
        win_samples = get_AGC_samples(args.agc_win, dt=dt)
        xprint(f'Apply AGC with window length of >{args.agc_win}< sec', kind='info', verbosity=args.verbose)
        xprint(f'win_samples: {win_samples}', kind='debug', verbosity=args.verbose)
        
        kwargs_agc = dict(win=win_samples, kind=args.agc_kind, squared=args.agc_sqrt)
        xprint(kwargs_agc, kind='debug', verbosity=args.verbose)
        
        for var in data_vars:
            cube_proc[var] = (('iline', 'xline', 'twt'), AGC(cube[var].data, **kwargs_agc))
    
        _history += f' AGC (win={args.agc_win:g} kind={args.agc_kind}, squared={args.agc_sqrt}),'
        _text += 'AGC ({args.agc_win:g} s).'
        text_suffix += '_AGC'
    
    # print(cube_proc.__dask_graph__())
    # cube_proc[var].data.visualize(optimize_graph=True)

    # add/update metadata
    xprint('Update netCDF metadata attributes', kind='info', verbosity=args.verbose)
    for v in nodata_vars:
        cube_proc[v] = cube[v]
    cube_proc.attrs.update({
        'history': cube_proc.attrs["history"] + f'{_history[:-1]};',  # remove trailing comma
        'text': cube_proc.attrs.get("text") + f'\n{_text[:-1]}',  # remove trailing period
    })
    
    # (4) write balanced cube to disk
    xprint(f'Write output data to file > {basename}{text_suffix}{suffix} <', kind='info', verbosity=args.verbose)
    path_cube_proc = os.path.join(dir_work, f'{basename}{text_suffix}{suffix}')
    xprint('path_cube_proc:', path_cube_proc, kind='debug', verbosity=args.verbose)
    if args.agc:
        from dask.distributed import Client  # better resource management
    
        with Client():  # LocalCluster(n_workers=12, threads_per_worker=1)
            cube_proc.to_netcdf(path=path_cube_proc, engine='h5netcdf')
    else:
        with dask.config.set(scheduler='threads'), show_progressbar(ProgressBar(), verbose=args.verbose):
            cube_proc.to_netcdf(path=path_cube_proc, engine='h5netcdf')
        
    if return_dataset:
        return cube_proc, cube

#%% MAIN

if __name__ == '__main__':
    
    main()
