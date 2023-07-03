"""
Utility script to reproject/transform SEG-Y header coordinates between coordinate reference systems (CRS).

"""
import os
import sys
import glob
import argparse
import datetime
from shutil import copy2
from contextlib import redirect_stdout

import segyio
import pyproj
from tqdm import tqdm

from pseudo_3D_interpolation.functions.header import (
    scale_coordinates,
    set_coordinates,
    get_textual_header,
    add_processing_info_header,
    write_textual_header,
)
from pseudo_3D_interpolation.functions.utils import TRACE_HEADER_COORDS, xprint, clean_log_file
from pseudo_3D_interpolation.functions.filter import smooth

#%% FUNCTIONS
# fmt: off
def define_input_args():  # noqa
    parser = argparse.ArgumentParser(
        description='Coordinate transformation for SEG-Y file(s).')
    parser.add_argument('input_path', type=str, help='Input file or directory.')
    parser.add_argument('--crs_src', type=str, required=True,
                        help='Source CRS of SEG-Y file(s). Indicate using EPSG code or PROJ.4 string.')
    parser.add_argument('--crs_dst', type=str, required=True,
                        help='Destination CRS of SEG-Y file(s). Indicate using EPSG code or PROJ.4 string.')
    parser.add_argument('--output_dir', '-o', type=str,
                        help='Output directory for reprojected SEG-Y file(s).')
    parser.add_argument('--inplace', '-i', action='store_true',
                        help='Edit SEG-Y file(s) inplace.')
    parser.add_argument('--filename_suffix', '-fns', type=str,
                        help='Filename suffix for guided selection (e.g. "env" or "despk"). Only used when "input_path" is a directory.')
    parser.add_argument('--suffix', '-s', type=str,
                        help='File suffix. Only used when "input_path" is a directory.')
    parser.add_argument('--txt_suffix', type=str,
                        help='Additional text to append to output filename.')
    parser.add_argument('--scalar_coords', '-sc', type=int,
                        default=-100, choices=[-1000, -100, -10, 0, 10, 100, 1000],
                        help='''Output coordinate scalar.
                            Negative: division by absolute value,
                            positive: multiplication by absolute value.''')
    parser.add_argument('--src_coords', type=str, choices=['source', 'CDP', 'group'], default='source',
                        help='Byte position of input coordinates in SEG-Y file(s).')
    parser.add_argument('--dst_coords', type=str, choices=['source', 'CDP', 'group'], default='source',
                        help='Byte position of output coordinates in SEG-Y file(s).')
    parser.add_argument('--smooth', type=int, nargs='?', default=None, const=11,
                        help='Smooth coordinates using window of size `k` traces (default: 11).')
    parser.add_argument('--verbose', '-V', type=int, nargs='?', default=0, const=1, choices=[0, 1, 2],
                        help='Level of output verbosity (default: 0).')
    return parser
# fmt: on


def dms2dd(dms):
    """Convert DMS (Degrees, Minutes, Seconds) coordinate string to DD (Decimal Degrees)."""
    dms_str = str(dms)
    degrees, minutes, seconds = int(dms_str[:3]), int(dms_str[3:5]), int(dms_str[5:])
    if len(str(seconds)) > 2:
        seconds = float(str(seconds)[:2] + '.' + str(seconds)[2:])
    return float(degrees) + float(minutes) / 60 + float(seconds) / (60 * 60)


def wrapper_reproject_segy(in_path, src_coords_bytes, dst_coords_bytes, args):
    """
    Reproject SEG-Y header coordinates for single SEG-Y file.

    Parameters
    ----------
    in_path : str
        SEG-Y input path.
    src_coords_bytes : tuple
        Input byte position of coordinates in trace header.
    dst_coords_bytes : tuple
        Output byte position of coordinates in trace header.
    args : argparse.Namespace
        Input parameter.

    """
    basepath, filename = os.path.split(in_path)
    basename, suffix = os.path.splitext(filename)
    xprint(f'Processing file < {filename} >', kind='info', verbosity=args.verbose)

    default_txt_suffix = 'reproj'
    
    if args.inplace is True:  # `inplace` parameter supersedes any `output_dir`
        xprint('Updating SEG-Y inplace', kind='warning', verbosity=args.verbose)
        path = in_path
    else:
        if args.output_dir is None:  # default behavior
            xprint('Creating copy of file in INPUT directory:\n', basepath, kind='info', verbosity=args.verbose)
            out_dir = basepath
        elif args.output_dir is not None and os.path.isdir(args.output_dir):
            xprint('Creating copy of file in OUTPUT directory:\n', args.output_dir, kind='info', verbosity=args.verbose)
            out_dir = args.output_dir
        else:
            raise FileNotFoundError(f'The output directory > {args.output_dir} < does not exist')
            
        if args.txt_suffix is not None:
            out_name = f'{basename}_{args.txt_suffix}'
        else:
            out_name = f'{basename}_{default_txt_suffix}'
        out_path = os.path.join(out_dir, f'{out_name}{suffix}')
    
        # sanity check
        if os.path.isfile(out_path):
            xprint('Output file already exists and will be removed!', kind='warning', verbosity=args.verbose)
            os.remove(out_path)

        copy2(in_path, out_path)
        path = out_path

    # read SEGY file
    with segyio.open(path, 'r+', strict=False, ignore_geometry=True) as file:
        # get scaled coordinates
        xcoords, ycoords, coordinate_units = scale_coordinates(file, src_coords_bytes)

        # get source & destination CRS
        crs_src = pyproj.crs.CRS(args.crs_src)
        if coordinate_units != 1 and crs_src.is_projected:
            xprint(
                'Forced source CRS to be geographic (WGS84 - EPSG:4326)!',
                kind='warning',
                verbosity=args.verbose,
            )
            crs_src = pyproj.crs.CRS('epsg:4326')
        crs_dst = pyproj.crs.CRS(args.crs_dst)

        # create CRS transformer
        transformer = pyproj.transformer.Transformer.from_crs(crs_src, crs_dst, always_xy=True)

        # convert coordinates
        xcoords_t, ycoords_t = transformer.transform(xcoords, ycoords, errcheck=True)
        
        # [OPTIONAL] smooth coordinates
        if args.smooth is not None:
            xcoords_t = smooth(xcoords_t, args.smooth)
            ycoords_t = smooth(ycoords_t, args.smooth)

        # update coordinates in trace header
        set_coordinates(
            file,
            xcoords_t,
            ycoords_t,
            crs_dst,
            dst_coords_bytes,
            coordinate_units=1,
            scaler=args.scalar_coords,
        )

    # update textual header
    text = get_textual_header(path)
    ## add info about new CRS
    info = f'EPSG:{crs_dst.to_epsg()}'
    text_updated = add_processing_info_header(text, info, prefix='CRS (PROJECTED)')
    # add info about byte position of reprojected coords
    info = f'REPROJECT (BYTES:{dst_coords_bytes[0]} {dst_coords_bytes[1]})'
    info += ' SMOOTHED' if args.smooth is not None else ''
    text_updated = add_processing_info_header(text_updated, info, prefix='_TODAY_')
    write_textual_header(path, text_updated)


def main(argv=sys.argv):  # noqa
    """Reproject trace header coordinats of SEG-Y file(s)."""
    TIMESTAMP = datetime.datetime.now().isoformat(timespec='seconds').replace(':', '')
    SCRIPT = os.path.basename(__file__).split(".")[0]  # FIXME

    parser = define_input_args()
    args = parser.parse_args(argv[1:])  # exclude filename parameter at position 0
    xprint(args, kind='debug', verbosity=args.verbose)
    
    # input and output coordinate byte positions
    src_coords_bytes = TRACE_HEADER_COORDS.get(args.src_coords)
    dst_coords_bytes = TRACE_HEADER_COORDS.get(args.dst_coords)
    
    # check input file(s)
    in_path = args.input_path
    basepath, filename = os.path.split(in_path)
    basename, suffix = os.path.splitext(filename)
    if suffix == '':
        basepath = in_path
        basename, suffix = None, None  # noqa

    # (1) single input file
    if os.path.isfile(in_path) and (suffix != '.txt'):
        # perform coordinate transformation
        wrapper_reproject_segy(in_path, src_coords_bytes, dst_coords_bytes, args)
        sys.exit()

    # (2) input directory (multiple files)
    elif os.path.isdir(in_path):
        pattern = '*'
        pattern += f'{args.filename_suffix}' if args.filename_suffix is not None else pattern
        pattern += f'.{args.suffix}' if args.suffix is not None else '.sgy'
        file_list = glob.glob(os.path.join(in_path, pattern))

    # (3) file input is datalist (multiple files)
    elif os.path.isfile(in_path) and (suffix == '.txt'):
        with open(in_path, 'r') as datalist:
            file_list = datalist.readlines()
            file_list = [
                os.path.join(basepath, line.rstrip())
                if os.path.split(line.rstrip()) not in ['', '.']
                else line.rstrip()
                for line in file_list
            ]
    else:
        raise FileNotFoundError('Invalid input file')

    # compute files from options (2) or (3)
    if len(file_list) > 0:
        # redirect stdout to logfile
        logfile = os.path.join(basepath, f'{TIMESTAMP}_{SCRIPT}.log')
        with open(logfile, 'w', newline='\n') as f:
            with redirect_stdout(f):
                xprint(f'Processing total of < {len(file_list)} > files', kind='info', verbosity=args.verbose)
                for file_path in tqdm(
                    file_list,
                    desc='Reprojecting SEG-Y',
                    ncols=80,
                    total=len(file_list),
                    unit_scale=True,
                    unit=' files',
                ):
                    # perform coordinate transformation
                    wrapper_reproject_segy(file_path, src_coords_bytes, dst_coords_bytes, args)
        clean_log_file(logfile)
    else:
        sys.exit('No input files to process. Exit process.')


#%% MAIN
if __name__ == '__main__':

    main()
