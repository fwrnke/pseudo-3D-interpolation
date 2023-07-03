"""
Check for vertical offsets in SEG-Y file(s) and apply zero padding.
For this, the script uses the trace header keyword _DelayRecordingTime_
and pads seismic traces with zeros to compensate variable recording starting times.

"""
import os
import sys
import glob
import argparse
import datetime
from contextlib import redirect_stdout

import numpy as np
import segyio
from tqdm import tqdm

from pseudo_3D_interpolation.functions.header import (
    get_textual_header,
    add_processing_info_header,
    write_textual_header,
)
from pseudo_3D_interpolation.functions.utils import xprint, clean_log_file

#%% FUNCTIONS
# fmt: off
def define_input_args():  # noqa
    parser = argparse.ArgumentParser(
        description='Pad time delays in SEG-Y file(s) using "DelayRecordingTime".')
    parser.add_argument('input_path', type=str, help='Input file or directory.')
    parser.add_argument('--output_dir', '-o', type=str,
                        help='Output directory for padded SEG-Y file(s).')
    parser.add_argument('--suffix', '-s', type=str,
                        help='File suffix. Only used when "input_path" is a directory.')
    parser.add_argument('--filename_suffix', '-fns', type=str,
                        help='Filename suffix for guided selection (e.g. "env" or "despk"). Only used when "input_path" is a directory.')
    parser.add_argument('--txt_suffix', type=str,
                        help='Additional text to append to output filename.')
    parser.add_argument('--byte_delay', type=int, default=109,
                        help='Byte position of input delay times in SEG-Y file(s) (default: 109, "DelayRecordingTime")')
    parser.add_argument('--verbose', '-V', type=int, nargs='?', default=0, choices=[0, 1, 2],
                        help='Level of output verbosity (default: 0).')
    return parser
# fmt: on


def pad_trace_data(data, recording_delays, n_traces, dt, twt, verbosity=0):
    """
    Pad seismic trace data recorded in window mode with a fixed length and variable `DelayRecordingTimes`.
    This function reads the DelayRecordingTime from the trace headers and pads the data at top and bottom with zeros.

    Parameters
    ----------
    data : numpy.ndarray
        Amplitude array with samples (rows) and traces (columns).
        Transposed array read by segyio for numpy compatibility.
    recording_delays : numpy.ndarray
        Array of DelayRecordingTimes for each trace in SEG-Y file [ms].
    n_traces : int
        Number of traces in SEG-Y file.
    dt : float
        Sampling interval [ms].
    twt : numpy.ndarray
        Two way travel times (TWTT) of every sample in a trace.

    Returns
    -------
    data_padded : numpy.ndarray
        Padded input data array in time dimension.
    twt_padded : numpy.ndarray
        Updated TWTT that fit the expanded time dimension.
    n_samples_padded : int
        New number of traces in SEG-Y file.
    idx_delay : numpy.ndarray
        Trace indices where DelayRecordingTime changes.
    min_delay : int
        Minimum DelayRecordingTime in SEG-Y file [ms].
    max_delay : int
        Maximum DelayRecordingTime in SEG-Y file [ms].

    """
    # find indices where DelayRecordingTime changes
    idx_delay = np.where(np.roll(recording_delays, 1) != recording_delays)[0]
    # sanity check: in case first and last trace of file have same DelayRecordingTime!
    if idx_delay[0] != 0:
        idx_delay = np.insert(idx_delay, 0, 0, axis=0)

    # minimum and maximum DelayRecordingTime in SEG-Y
    min_delay = recording_delays.min()
    max_delay = recording_delays.max()

    # pad twt array from minimum delay to maximum delay + data window size
    twt_padded = np.arange(min_delay, max_delay + (twt[-1] - twt[0]) + dt, dt)
    n_samples_padded = len(twt_padded)

    # initialize padded data array
    data_padded = np.ndarray((len(twt_padded), n_traces))
    # make sure array is contiguous in memory (C order)
    data_padded = np.ascontiguousarray(data_padded, dtype=data.dtype)

    for i, delay in enumerate(idx_delay):
        xprint(f'{i}: {recording_delays[delay]} ms', kind='debug', verbosity=verbosity)
        # ---------- first data slices ----------
        if idx_delay[i] != idx_delay[-1]:
            # ----- get data slice -----
            data_tmp = data[:, idx_delay[i] : idx_delay[i + 1]]
            data_len = data_tmp.shape[0]
            xprint(f'data_tmp.shape: {data_tmp.shape}', kind='debug', verbosity=verbosity)

            # create array with samples to pad at the data top
            pad_top = np.zeros(
                [
                    np.arange(min_delay, recording_delays[delay], dt).size,
                    idx_delay[i + 1] - idx_delay[i],
                ]
            )
            xprint(f'pad_top:    {pad_top.shape}', kind='debug', verbosity=verbosity)
            data_tmp = np.insert(data_tmp, 0, pad_top, axis=0)

            # create array with samples to pad at the data bottom
            pad_bottom = np.zeros(
                [twt_padded.size - data_len - pad_top.shape[0], idx_delay[i + 1] - idx_delay[i]]
            )
            xprint(f'pad_bottom:    {pad_bottom.shape}', kind='debug', verbosity=verbosity)
            data_tmp = np.insert(data_tmp, data_tmp.shape[0], pad_bottom, axis=0)

            # ----- insert padded data into new ndarray -----
            data_padded[:, idx_delay[i] : idx_delay[i + 1]] = data_tmp

        # ---------- last data slice ----------
        elif idx_delay[i] == idx_delay[-1]:
            xprint('*** last slice ***', kind='debug', verbosity=verbosity)
            # ----- get data slice -----
            data_tmp = data[:, idx_delay[i] : n_traces]
            xprint(f'data_tmp.shape: {data_tmp.shape}', kind='debug', verbosity=verbosity)

            # create array with samples to pad at the data top
            pad_top = np.zeros(
                [np.arange(min_delay, recording_delays[delay], dt).size, n_traces - idx_delay[i]]
            )
            xprint(f'pad_top:    {pad_top.shape}', kind='debug', verbosity=verbosity)
            data_tmp = np.insert(data_tmp, 0, pad_top, axis=0)

            # create array with samples to pad at the data bottom
            pad_bottom = np.zeros(
                [twt_padded.size - twt.size - pad_top.shape[0], n_traces - idx_delay[i]]
            )
            xprint(f'pad_bottom:    {pad_bottom.shape}', kind='debug', verbosity=verbosity)
            data_tmp = np.insert(data_tmp, data_tmp.shape[0], pad_bottom, axis=0)

            # ----- insert padded data into new ndarray -----
            data_padded[:, idx_delay[i] : n_traces] = data_tmp

    return data_padded, twt_padded, n_samples_padded, (idx_delay, min_delay, max_delay)


def wrapper_delrt_padding_segy(in_path, args):  # noqa
    """Pad DelayRecordingTime `delrt` for single SEG-Y file."""
    basepath, filename = os.path.split(in_path)
    basename, suffix = os.path.splitext(filename)
    xprint(f'Processing file < {filename} >', kind='info', verbosity=args.verbose)

    default_txt_suffix = 'pad'
    
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

    # read SEGY file
    with segyio.open(in_path, 'r', strict=False, ignore_geometry=True) as src:
        n_traces = src.tracecount  # total number of traces
        dt = segyio.tools.dt(src) / 1000  # sample rate [ms]
        # n_samples = src.samples.size        # total number of samples
        twt = src.samples  # two way travel time (TWTT) [ms]

        # get "DelayRecordingTime" value for each trace
        recording_delays = src.attributes(args.byte_delay)[:]
        # find unique DelayRecordingTimes and corresponding trace index
        recording_delays_uniq, recording_delays_idx = np.unique(
            recording_delays[:], return_index=True
        )

        # check if padding is needed
        if len(recording_delays_uniq) > 1:
            xprint(
                f'Found < {len(recording_delays_uniq)} > different "DelayRecordingTimes" for file < {filename} >',
                kind='info',
                verbosity=args.verbose,
            )
        else:
            return False

        # get seismic data [amplitude]; transpose to fit numpy data structure
        data = src.trace.raw[:].T  # eager version (completely read into memory)
        
        hns = src.bin[segyio.BinField.Samples]  # samples per trace

        # pad source data to generate continuous array without vertical offsets
        data_padded, twt_padded, n_samples_padded, delay_attrs = pad_trace_data(
            data, recording_delays, n_traces, dt, twt, args.verbose
        )
        (idx_delay, min_delay, max_delay) = delay_attrs

        # get source metadata
        spec = segyio.tools.metadata(src)
        spec.samples = twt_padded  # set padded samples

        # create new output SEG-Y file with updated header information
        xprint(f'Writing padded output file < {out_name}{suffix} >', kind='info', verbosity=args.verbose)
        with segyio.create(out_path, spec) as dst:
            dst.text[0] = src.text[0]  # copy textual header
            dst.bin = src.bin  # copy binary header
            dst.header = src.header  # copy trace headers
            dst.trace = np.ascontiguousarray(
                data_padded.T, dtype=data.dtype
            )  # set padded trace data

            # update binary header with padded sample count
            dst.bin[segyio.BinField.Samples] = n_samples_padded
            dst.bin[segyio.BinField.SamplesOriginal] = hns

            # update trace headers with padded sample count & new DelayRecordingTime
            for h in dst.header[:]:
                h.update(
                    {
                        segyio.TraceField.TRACE_SAMPLE_COUNT: n_samples_padded,
                        segyio.TraceField.DelayRecordingTime: min_delay,
                    }
                )

        # update textual header
        text = get_textual_header(out_path)
        info = f'PAD DELRT (byte:{segyio.TraceField.DelayRecordingTime})'
        text_updated = add_processing_info_header(text, info, prefix='_TODAY_')
        write_textual_header(out_path, text_updated)


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
    if os.path.isfile(in_path) and suffix != '.txt':
        # pad traces of SEG-Y file
        check_DelayRecordingTime = wrapper_delrt_padding_segy(in_path, args)
        if check_DelayRecordingTime is False:
            xprint(
                'Continuous "DelayRecordingTime" for whole SEG-Y file --> skipped!',
                kind='info',
                verbosity=args.verbose,
            )
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
            file_list = [os.path.join(basepath, line.rstrip()) for line in file_list]
    else:
        raise FileNotFoundError('Invalid input file')

    # compute files from options (2) or (3)
    if len(file_list) > 0:
        # redirect stdout to logfile
        logfile = os.path.join(basepath, f'{TIMESTAMP}_{SCRIPT}.log')
        with open(logfile, 'w', newline='\n') as f:
            with redirect_stdout(f):
                xprint(f'Processing total of < {len(file_list)} > files', kind='info', verbosity=args.verbose)
                nprocessed = 0
                for file_path in tqdm(
                    file_list,
                    desc='Padding SEG-Y',
                    ncols=80,
                    total=len(file_list),
                    unit_scale=True,
                    unit=' files',
                ):
                    # pad traces of SEG-Y file
                    check_DelayRecordingTime = wrapper_delrt_padding_segy(file_path, args)
                    if check_DelayRecordingTime is False:
                        xprint(
                            'Continuous "DelayRecordingTime" for whole SEG-Y file --> skipped!',
                            kind='info',
                            verbosity=args.verbose,
                        )
                        continue
                    else:
                        nprocessed += 1
                xprint(
                    f'Padded a total of < {nprocessed} > out of < {len(file_list)} > files',
                    kind='info',
                    verbosity=args.verbose,
                )
        clean_log_file(logfile)
    else:
        sys.exit('No input files to process. Exit process.')


#%% MAIN
if __name__ == '__main__':

    main()
