"""Utility script to convert SEG-Y files to netCDF format using `segysak` in parallel."""

import os
import sys
import glob
import argparse
import multiprocessing as mp
from itertools import repeat

from segysak.segy import segy_converter

from pseudo_3D_interpolation.functions.utils import xprint

#%% FUNCTIONS
# fmt: off
def define_input_args():  # noqa
    parser = argparse.ArgumentParser(
        description='Convert SEG-Y files to netCDF format using `segysak`.')
    parser.add_argument('path_input', type=str, help='Input datalist or directory.')
    parser.add_argument('--output_dir', '-o', type=str,
                        help='Output directory for created netCDF file(s)')
    parser.add_argument('--suffix', '-s', type=str, default='sgy',
                        help='File suffix. Only used when "path_input" is a directory.')
    parser.add_argument('--filename_suffix', '-fns', type=str,
                        help='Filename suffix for guided selection (e.g. "env" or "despk"). Only used when "path_input" is a directory.')
    parser.add_argument('--nprocesses', type=int, default=4,
                        help='Number of parallel conversions (<= number of CPUs).')
    parser.add_argument('--verbose', '-V', type=int, nargs='?', default=0, choices=[0, 1, 2],
                        help='Level of output verbosity (default: 0)')
    return parser
# fmt: on


def _converter(file, file_dir, **kwargs_segysak) -> None:
    """Wrap function to convert SEG-Y to netCDF."""
    name, suffix = os.path.splitext(file)
    out_nc = os.path.join(file_dir, name + '.seisnc')

    if kwargs_segysak == {}:
        kwargs_segysak = dict(cdp=5, cdpx=73, cdpy=77, silent=True)
    kwargs_segysak['strict'] = False

    # convert SEG-Y --> netCDF
    segy_converter(os.path.join(file_dir, file), out_nc, **kwargs_segysak)


def main(argv=sys.argv):  # noqa
    parser = define_input_args()
    args = parser.parse_args(argv[1:])  # exclude filename parameter at position 0

    basepath, filename = os.path.split(args.path_input)
    basename, suffix = os.path.splitext(filename)
    if suffix == '':
        basepath = args.path_input
        basename, suffix = None, None  # noqa

    if args.output_dir is None:
        dir_out = basepath
    elif os.path.isdir(args.output_dir):
        dir_out = args.output_dir
    else:
        raise ValueError('``--output_dir`` must be an existing directory!')
    
    # (1) single input file
    if os.path.isfile(args.path_input) and (suffix in ['.sgy', '.segy']):
        xprint('[INFO]    Converting SEG-Y file to netCDF format', kind='info', verbosity=args.verbose)
        _converter(args.path_input, dir_out)
        sys.exit()

    # (2) "path_input" is directory
    elif os.path.isdir(args.path_input):
        pattern = '*'
        pattern += f'{args.filename_suffix}' if args.filename_suffix is not None else pattern
        pattern += f'.{args.suffix}'
        files = glob.glob(os.path.join(args.path_input, pattern))

    # (3) file input is datalist (multiple files)
    elif os.path.isfile(args.path_input) and (suffix == '.txt'):
        # read file names from datalist
        with open(args.path_input, 'r') as f:
            files = f.read().splitlines()
    else:
        raise FileNotFoundError('Invalid input file')

    xprint(f'[INFO]    Converting > {len(files)} < SEG-Y files to netCDF format', kind='info', verbosity=args.verbose)

    # parallel file conversions
    with mp.Pool(processes=args.nprocesses) as pool:
        _ = pool.starmap(_converter, zip(files, repeat(dir_out)))

    xprint(f'[SUCCESS]    Finished conversion of > {len(files)} < SEG-Y files!', kind='info', verbosity=args.verbose)


#%% MAIN

if __name__ == '__main__':

    main()
