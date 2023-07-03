"""
Compensate tidal effect for SEG-Y file(s).
Using the OSU TPXO9 atlas tide model with the `tpxo-tide-prediction` package as an interface.

References
----------
[^1]: Oregon State University (OSU), [https://www.tpxo.net/home](https://www.tpxo.net/home)
[^2]: TPXO9-atlas models, [https://www.tpxo.net/global/tpxo9-atlas](https://www.tpxo.net/global/tpxo9-atlas)
[^3]: `tpxo-tide-prediction`, [https://github.com/fwrnke/tpxo-tide-prediction](https://github.com/fwrnke/tpxo-tide-prediction)

"""
import os
import sys
import glob
import argparse
import datetime
from shutil import copy2
from contextlib import redirect_stdout

import numpy as np
import segyio
import pyproj
from tqdm import tqdm

from pseudo_3D_interpolation.functions.utils import xprint, clean_log_file, depth2twt, depth2samples, twt2samples
from pseudo_3D_interpolation.functions.utils import TRACE_HEADER_COORDS
from pseudo_3D_interpolation.functions.header import (
    scale_coordinates,
    get_textual_header,
    add_processing_info_header,
    write_textual_header,
)
from pseudo_3D_interpolation.functions.backends import tpxo_tide_prediction_enabled

if tpxo_tide_prediction_enabled:
    from tpxo_tide_prediction import tide_predict
else:
    sys.exit('Module > tpxo-tide-prediction < not found. Please install before running this script.')


#%% FUNCTIONS
# fmt: off
def define_input_args():  # noqa
    parser = argparse.ArgumentParser(
        description='Compensate tidal effect for SEG-Y file(s) using TPXO9-atlas-v4 tide model.')
    parser.add_argument('input_path', type=str, help='Input file or directory.')
    parser.add_argument('model_dir', type=str, help='Input directory of tidal model files.')
    parser.add_argument('--output_dir', '-o', type=str,
                        help='Output directory for compensated SEG-Y file(s).')
    parser.add_argument('--inplace', '-i', action='store_true',
                        help='Edit SEG-Y file(s) inplace.')
    parser.add_argument('--suffix', '-s', type=str,
                        help='File suffix. Only used when "input_path" is a directory.')
    parser.add_argument('--filename_suffix', '-fns', type=str,
                        help='Filename suffix for guided selection (e.g. "env" or "despk"). Only used when "input_path" is a directory.')
    parser.add_argument('--txt_suffix', type=str,
                        help='Additional text to append to output filename.')
    parser.add_argument('--constituents', '-c', nargs='+',
                        choices=['m2', 's2', 'n2', 'k2', 'k1', 'o1', 'p1', 'q1',
                                 'm4', 'mf', '2n2', 'mm', 'mn4', 'ms4'],
                        default=['m2', 's2', 'n2', 'k2', 'k1', 'o1', 'p1', 'q1'],
                        help='Available tidal constituents supported by TPXO9 atlas model.')
    parser.add_argument('--correct_minor', action='store_true',
                        help='Correct for minor tidal constituents.')
    parser.add_argument('--src_coords', type=str, choices=['source', 'CDP', 'group'], default='source',
                        help='Byte position of input coordinates in SEG-Y file(s).')
    parser.add_argument('--crs_src', type=str, default='epsg:32760',
                        help='Source CRS of SEG-Y file(s). Indicate using EPSG code or PROJ.4 string.')
    parser.add_argument('--write_aux', action='store_true',
                        help='Write times and tide predictions to auxiliary file (*.tid).')
    parser.add_argument('--verbose', '-V', type=int, nargs='?', default=0, choices=[0, 1, 2],
                        help='Level of output verbosity (default: 0).')
    return parser
# fmt: on


def compensate_tide(
    data, tide, dt: float, tide_units: str = 'meter', units: str = 'ms', v: int = 1500, verbosity=1
):
    """
    Apply predicted tide offset to seismic traces.

    Parameters
    ----------
    data : np.ndarray
        Seismic section (samples x traces).
    tide : np.ndarray
        Predicted tide for each trace location (1D array).
    dt : float
        Sampling interval in specified units (default: `ms`).
    tide_units : str, optional
        Units of predicted tide values. Either elevation (`meter`, _default_),
        two-way travel time (`s`, `ms`) or `samples`.
    units : str, optional
        Time unit (for dt) (default: `ms`).
    v : int, optional
        Acoustic velocity used for depth-time conversion (default: `1500` m/s).

    Returns
    -------
    data_comp : np.ndarray(s)
        Compensated seismic section.

    """
    # convert dt to seconds
    if units == 's':
        pass
    elif units == 'ms':
        dt = dt / 1000
    elif units == 'ns':
        dt = dt / 1e-6

    if tide_units == 'meter':
        tide_samples = depth2samples(tide, dt, v=v, units='s')
    elif tide_units in ['s', 'ms']:
        tide_samples = twt2samples(tide, dt, units='s')
    elif tide_units == 'samples':
        tide_samples = tide
    else:
        raise ValueError(f'Provided unknown unit < {tide_units} > for tide values.')
    tide_samples = np.around(tide_samples, 0).astype('int32')

    # create (transposed) copy of original data
    data_comp = data.T.copy()

    for i, col in enumerate(data_comp):
        offset = tide_samples[i]
        if offset > 0:
            col[:] = np.hstack((col[abs(offset) :], np.zeros(abs(offset))))
            xprint(
                f'trace #{i}:{offset:>5}   ->   up: {col.shape}', kind='debug', verbosity=verbosity
            )
        elif offset < 0:
            col[:] = np.hstack((np.zeros(abs(offset)), col[: -abs(offset)]))
            xprint(
                f'trace #{i}:{offset:>5}   ->   down: {col.shape}',
                kind='debug',
                verbosity=verbosity,
            )
        else:
            pass  # no static correction

    return data_comp.T


def wrapper_tide_compensation(in_path, args):
    """Compensate tidal effect for single SEG-Y."""
    basepath, filename = os.path.split(in_path)
    basename, suffix = os.path.splitext(filename)
    xprint(f'Processing file < {filename} >', kind='info', verbosity=args.verbose)

    default_txt_suffix = 'tide'
    
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

    # get coordinate byte positions
    src_coords_bytes = TRACE_HEADER_COORDS.get(args.src_coords)

    # read SEGY file
    with segyio.open(path, 'r+', strict=False, ignore_geometry=True) as src:
        n_traces = src.tracecount  # total number of traces
        dt = segyio.tools.dt(src) / 1000  # sample rate [ms]
        n_samples = src.samples.size  # total number of samples
        # twt = src.samples                   # two way travel time (TWTT) [ms]

        xprint(f'n_traces:  {n_traces}', kind='debug', verbosity=args.verbose)
        xprint(f'n_samples: {n_samples}', kind='debug', verbosity=args.verbose)
        xprint(f'dt:        {dt}', kind='debug', verbosity=args.verbose)

        tracl = src.attributes(segyio.TraceField.TRACE_SEQUENCE_LINE)[:]
        tracr = src.attributes(segyio.TraceField.TRACE_SEQUENCE_FILE)[:]
        fldr = src.attributes(segyio.TraceField.FieldRecord)[:]

        # get seismic data [amplitude]; transpose to fit numpy data structure
        data_src = src.trace.raw[:].T  # eager version (completely read into memory)

        # get scaled coordinates
        xprint('Reading coordinates from SEG-Y file', kind='debug', verbosity=args.verbose)
        xcoords, ycoords, coordinate_units = scale_coordinates(src, src_coords_bytes)

        # get source & destination CRS
        crs_src = pyproj.crs.CRS(args.crs_src)
        crs_dst = pyproj.crs.CRS('epsg:4326')

        # transform coordinate to geographical coordinates
        transformer = pyproj.transformer.Transformer.from_crs(crs_src, crs_dst, always_xy=True)
        lon, lat = transformer.transform(xcoords, ycoords, errcheck=True)

        # filter duplicate lat/lon locations
        latlon = np.vstack((lat, lon)).T
        latlon_unique = np.unique(
            latlon, axis=0, return_index=True, return_inverse=True, return_counts=True
        )
        latlon_uniq, latlon_uniq_idx, latlon_uniq_inv, latlon_uniq_cnts = latlon_unique
        # get unique lat/lons locations
        lat, lon = latlon_uniq[:, 0], latlon_uniq[:, 1]

        # get datetime elements from trace headers
        xprint('Reading timestamps from SEG-Y file', kind='debug', verbosity=args.verbose)
        times = []
        for h in src.header:
            year = h[segyio.TraceField.YearDataRecorded]
            day_of_year = h[segyio.TraceField.DayOfYear]
            hour = h[segyio.TraceField.HourOfDay]
            minute = h[segyio.TraceField.MinuteOfHour]
            second = h[segyio.TraceField.SecondOfMinute]
            dt_string = datetime.datetime.strptime(
                f'{year}-{day_of_year} {hour}:{minute}:{second}', '%Y-%j %H:%M:%S'
            ).strftime('%Y-%m-%dT%H:%M:%S')
            times.append(np.datetime64(dt_string, 's'))
        times = np.asarray(times)
        # get times masked by unique lat/lons locations
        times = times[latlon_uniq_idx]

        # predict tides along SEG-Y file at given times
        xprint('Predicting tidal elevation along profile', kind='debug', verbosity=args.verbose)
        tides_track = tide_predict(
            args.model_dir,
            lat,
            lon,
            times,
            args.constituents,
            correct_minor=args.correct_minor,
            mode='track',
        )[
            latlon_uniq_inv
        ]  # set tide for original lat/lon locations (INCLUDING duplicates)

        # reset times (INCLUDING duplicates) -> for output
        times = times[latlon_uniq_inv]

        # apply tidal compensation to seismic traces
        xprint('Compensating tide', kind='debug', verbosity=args.verbose)
        data_comp = compensate_tide(
            data_src, tides_track, dt, tide_units='meter', units='ms', verbosity=args.verbose
        )

        # set output amplitudes (transpose to fit SEG-Y format)
        xprint('Writing compensated data to disk', kind='debug', verbosity=args.verbose)
        src.trace = np.ascontiguousarray(data_comp.T, dtype=data_src.dtype)

    # update textual header
    text = get_textual_header(path)
    text_updated = add_processing_info_header(
        text, 'TIDE COMPENSATION', prefix='_TODAY_', newline=True
    )
    write_textual_header(path, text_updated)

    if args.write_aux:
        tides_twt = depth2twt(tides_track)
        tides_samples = np.around(depth2samples(tides_track, dt=dt, units='ms'), 0)
        aux_path = os.path.join(out_dir, f'{out_name}.tid')
        xprint(f'Creating auxiliary file < {out_name}.tid >', kind='debug', verbosity=args.verbose)

        with open(aux_path, 'w', newline='\n') as fout:
            fout.write('tracl,tracr,fldr,time,tide_m,tide_ms,tide_samples\n')
            for i in range(tides_track.size):
                line = (
                    f'{tracl[i]},{tracr[i]},{fldr[i]},'
                    + f'{np.datetime_as_string(times[i],"s")},'
                    + f'{tides_track[i]:.6f},{tides_twt[i]*1000:.3f},'
                    + f'{tides_samples[i]:.0f}\n'
                )
                fout.write(line)


def main(argv=sys.argv):  # noqa
    TIMESTAMP = datetime.datetime.now().isoformat(timespec='seconds').replace(':', '')
    SCRIPT = os.path.basename(__file__).split(".")[0]

    parser = define_input_args()
    args = parser.parse_args(argv[1:])  # exclude filename parameter at position 0

    # check input file(s)
    in_path = args.input_path
    basepath, filename = os.path.split(in_path)
    basename, suffix = os.path.splitext(filename)
    if suffix == '':
        basepath = in_path
        basename, suffix = None, None  # noqa

    # (1) single input file
    if os.path.isfile(in_path) and (suffix != '.txt'):
        # compensate tidal effect
        wrapper_tide_compensation(in_path, args)
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
        with open(logfile, mode='w', newline='\n') as f:
            with redirect_stdout(f):
                xprint(f'Processing total of < {len(file_list)} > files', kind='info', verbosity=args.verbose)
                for file_path in tqdm(
                    file_list,
                    desc='Compensate tides',
                    ncols=80,
                    total=len(file_list),
                    unit_scale=True,
                    unit=' files',
                ):
                    # compensate tidal effect
                    wrapper_tide_compensation(file_path, args)
        clean_log_file(logfile)
    else:
        sys.exit('No input files to process. Exit process.')


#%% MAIN

if __name__ == '__main__':

    main()
