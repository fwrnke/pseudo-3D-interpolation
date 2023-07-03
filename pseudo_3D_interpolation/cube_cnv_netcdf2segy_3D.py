"""
Utility script to convert 3D cube from netCDF to SEG-Y format.

"""
import os
import sys
import yaml
import argparse
import datetime

import numpy as np
import xarray as xr
import pyproj
import segyio

from segysak.segy._segy_core import check_tracefield
from segysak.segy._segy_writer import (
    _check_dimension, output_byte_locs,
    _ncdf2segy_2d, _ncdf2segy_2d_gath,
    _ncdf2segy_3d, _ncdf2segy_3dgath
)
from segysak.segy._segy_text import _clean_texthead

from pseudo_3D_interpolation.functions.utils import xprint
from pseudo_3D_interpolation.functions.header import check_coordinate_scalar

xr.set_options(keep_attrs=True)
    
# %% FUNCTIONS

# fmt: off
def define_input_args():  # noqa
    parser = argparse.ArgumentParser(
        description='Convert 3D cube from netCDF to SEG-Y file format.'
        )
    parser.add_argument('path_cube', type=str, help='Input path of 3D cube')
    parser.add_argument(
        '--params_netcdf', type=str, required=True, help='Path of netCDF parameter file (YAML format).'
    )
    parser.add_argument('--path_segy', type=str, help='Output SEG-Y file path.')
    parser.add_argument(
        '--scalar_coords', type=str, default='auto',
        choices=[-1000, -100, -10, 0, 10, 100, 1000, 'auto'],
        help='Coordinate scalar for SEG-Y trace header.'
    )
    parser.add_argument(
        "--verbose", "-V", type=int, nargs="?", default=0, const=1, choices=[0, 1, 2],
        help="Level of output verbosity (default: 0)",
    )
    return parser
# fmt: on


def segy_writer_custom(
    seisnc,
    segyfile,
    trace_header_map=None,
    il_chunks=None,
    dimension=None,
    silent=False,
    use_text=False,
):  # noqa
    """
    Custom implementation of `segysak.segy.segy_writer` to change some hardcoded defaults.

    Parameters
    ----------
    seisnc : xr.Dataset
        Input datset to write.
    segyfile : str
        Output SEG-Y file path.
    trace_header_map : dict, optional
        A dictionary of seisnc variables and byte locations. The variable will be written
        to the trace headers in the assigned byte location (default: None).
        By default: `CMP=23`, `cdp_x=181`, `cdp_y=185`, `iline=189`, `xline=193`.
    il_chunks : int, optional
        The size of data to work on - if you have memory limitations (default: `10`).
        This is primarily used for large 3D and ignored for 2D data.
    dimension : str, optional
        Data dimension to output, defaults to `twt` or `depth` whichever is present.
    silent : bool, optional
        Turn off progress reporting (defaults: `False`).
    use_text : bool, optional
        Use the seisnc text for the EBCIDC output. This text usally comes from
        the loaded SEG-Y file and may not match the segysak SEG-Y output (default: `False`).
    
    References
    ----------
    [^1]: SEGY-SAK, [https://segysak.readthedocs.io/en/latest/generated/segysak.segy.segy_writer.html](https://segysak.readthedocs.io/en/latest/generated/segysak.segy.segy_writer.html)

    """
    if trace_header_map:
        check_tracefield(trace_header_map.values())
        
    dimension = _check_dimension(seisnc, dimension)

    # ensure there is a coord_scalar
    coord_scalar = seisnc.coord_scalar
    if coord_scalar is None:
        coord_scalar = 0
        seisnc.attrs["coord_scalar"] = coord_scalar
    coord_scalar_mult = np.power(abs(coord_scalar), np.sign(coord_scalar) * -1)
    seisnc.attrs["coord_scalar_mult"] = coord_scalar_mult

    # create empty trace header map if necessary
    if trace_header_map is None:
        trace_header_map = dict()
    else:  # check that values of thm in seisnc
        for key in trace_header_map:
            try:
                _ = seisnc[key]
            except KeyError:
                raise ValueError("keys of trace_header_map must be in seisnc")

    # transfrom text if requested
    if use_text:
        if isinstance(seisnc.text, dict):  # allow for parsing custom dict
            text = _clean_texthead(seisnc.text, 80)
        else:
            text = {i + 1: line for i, line in enumerate(seisnc.text.split("\n"))}
            text = _clean_texthead(text, 80)

    common_kwargs = dict(silent=silent, dimension=dimension, text=text)

    if seisnc.seis.is_2d():
        thm = output_byte_locs("standard_2d")
        thm.unfreeze()
        thm.update(trace_header_map)
        _ncdf2segy_2d(seisnc, segyfile, **common_kwargs, **thm)
    elif seisnc.seis.is_2dgath():
        thm = output_byte_locs("standard_2d_gath")
        thm.unfreeze()
        thm.update(trace_header_map)
        _ncdf2segy_2d_gath(seisnc, segyfile, **common_kwargs, **thm)
    elif seisnc.seis.is_3d():
        thm = output_byte_locs("standard_3d")
        thm.unfreeze()
        thm.update(trace_header_map)
        _ncdf2segy_3d(
            seisnc,
            segyfile,
            **common_kwargs,
            il_chunks=il_chunks,
            **thm,
        )
    elif seisnc.seis.is_3dgath():
        thm = output_byte_locs("standard_3d_gath")
        thm.unfreeze()
        thm.update(trace_header_map)
        _ncdf2segy_3dgath(
            seisnc,
            segyfile,
            **common_kwargs,
            il_chunks=il_chunks,
            **thm,
        )
    else:
        # cannot determine type of data writing a continuous traces
        raise NotImplementedError()


def main(argv=sys.argv):  # noqa
    """Convert `netCDF`cube to `SEG_Y` format."""
    TIMESTAMP = datetime.datetime.now().isoformat(timespec="seconds")
    
    parser = define_input_args()
    args = parser.parse_args(argv[1:])
    
    path_cube = args.path_cube
    path_segy = args.path_segy if args.path_segy is not None else path_cube.replace('.nc', '.sgy')
    params_netcdf = args.params_netcdf
    verbose = args.verbose

    # Load netCDF metadata
    if params_netcdf is not None:
        with open(params_netcdf, 'r') as f_attrs:  # args.params_netcdf
            kwargs_nc = yaml.safe_load(f_attrs)
        NO_DATA_VARS = kwargs_nc.get('var_aux', ['fold', 'ref_amp'])
    else:
        kwargs_nc = None
        NO_DATA_VARS = ['fold', 'ref_amp']
    
    # open netCDF file
    xprint('Open 3D cube', kind='info', verbosity=verbose)
    chunks = {'twt' : -1, 'iline' : 15, 'xline' : -1}
    cube = xr.open_dataset(path_cube, chunks=chunks, engine='h5netcdf')
    # cube['iline'] = cube['iline'] * 2  # FIXME
    var = [v for v in list(cube.data_vars) if v not in NO_DATA_VARS][0]
        
    # Load coordinates
    cube.coords['x'] = cube.coords['x'].compute()
    cube.coords['y'] = cube.coords['y'].compute()
    
    # Setup coordinates
    coord_scalar, coord_scalar_mult = check_coordinate_scalar(
        args.scalar_coords, xcoords=cube.coords['x'].values, ycoords=cube.coords['y'].values)

    # Add missing attributes
    cube.attrs['sample_rate'] = cube['twt'].attrs.get(
        'dt',
        int(float(cube['twt'].diff(dim='twt').mean()) * 1000) / 1000
        )
    crs = pyproj.CRS(cube.attrs.get('spatial_ref'))
    if cube.attrs.get('measurement_system') is None:
        cube.attrs['measurement_system'] = 'm' if crs.is_projected else 'deg'
    if cube.attrs.get('epsg') is None:
        cube.attrs['epsg'] = crs.to_epsg() if crs is not None else None
    cube.attrs['coord_scalar'] = coord_scalar
    cube.attrs['coord_scalar_mult'] = coord_scalar_mult
    cube.attrs['source_file'] = os.path.basename(path_cube)
    
    # rename for segysak
    cube = cube.rename({
        var: 'data',
        'x': 'cdp_x',
        'y': 'cdp_y',
        })
    
    # change dtype of fold array
    shape = cube['cdp_x'].shape
    ncdps = np.prod(shape)
    cube['cdp'] = (('iline', 'xline'), np.arange(1, ncdps + 1).reshape(shape, order='C'))
    cube = cube.set_coords(['fold', 'cdp'])
    
    # specify output byte locations for trace header values
    trace_header_map = {
        'cdp': segyio.TraceField.CDP,
        'fold': segyio.TraceField.NStackedTraces,
        'cdp_x': segyio.TraceField.CDP_X,
        'cdp_y': segyio.TraceField.CDP_Y,
        'iline': segyio.TraceField.INLINE_3D,
        'xline': segyio.TraceField.CROSSLINE_3D,
        }
    
    # Textual header
    # set trace header defaults
    textual_header = {
        1: '3D SEG-Y CONVERTED FROM NETCDF USING SEGYSAK',
        3: f'CREATION: {TIMESTAMP}',
        4: f'EVOKER: {os.getlogin()}',
        10: '*** PROCESSING STEPS ***',
        35: '*** BYTE LOCATION OF KEY HEADERS ***',
        36: f'CDP: {trace_header_map.get("cdp","")}  FOLD: {trace_header_map.get("fold","")}',
        37: f'CDP UTM-X: {trace_header_map.get("cdp_x","")} CDP UTM-Y: {trace_header_map.get("cdp_x","")} '
        + f'ALL COORDS SCALED BY: {coord_scalar_mult}',
        38: f'INLINE: {trace_header_map.get("iline","")}, XLINE: {trace_header_map.get("xline","")}',
        40: "END TEXTUAL HEADER",
    }
    
    # add processing steps
    # cube.attrs['text'] = cube.attrs['text'] + '\n=== 3D PROCESSING ===' + f"\n{TODAY}: 3D BINNING {stacking_method} ILINE:{bin_size_iline:g} " + f"XLINE:{bin_size_xline} UNIT: METER" + f'\n{_text[:-1]}'  + f'\n{TODAY}: LOWPASS (5000/5200 Hz) ' + 'FFT(TIME -> FREQ)'  + f'\n{TODAY}: FFT (FREQUENCY DOMAIN)' + f'\n{TODAY}: INVERSE FFT(FREQ -> TIME)'
    proc_steps = cube.attrs['text'].split('\n')
    textual_header.update(dict(zip(range(11, 11 + len(proc_steps)), proc_steps)))
    textual_header_str = ''
    for k in range(1, 41):
        line = textual_header.get(k)
        if line is None:
            textual_header_str += f'C{k:02d} ' + ' ' * 75 + '\n'
        else:
            textual_header_str += f'C{k:02d} {line[:75]:<75}\n'
    cube.attrs['text'] = textual_header_str[:-2]  # remove trailing line break

    # Write cube to SEG-Y
    # UserWarning from SEGYSAK due to incorrect sanity check in `segysak.segy._segy_text.put_segy_texthead`
    # -> raise issue on GitHub
    segy_writer_custom(
        cube,
        path_segy,
        trace_header_map=trace_header_map,
        il_chunks=chunks.get('iline'),
        dimension='twt',
        silent=False,
        use_text=True,
    )
    
    # update binary header
    dto = cube['twt'].attrs.get('dt_original', None)
    with segyio.open(path_segy, 'r+') as segyf:
        segyf.bin.update(
            dto=int(dto * 1000) if dto is not None else 0,
            tsort=segyio.TraceSortingFormat.INLINE_SORTING,
        )

# %% MAIN
if __name__ == '__main__':

    main()
