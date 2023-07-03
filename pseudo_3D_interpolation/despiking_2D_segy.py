"""
Despike SEG-Y file(s) using 2D moving window function.

Remove noise bursts (only single trace!) from seismic data based on the amplitude
of the data within a time window. This amplitude is compared to the
background amplitude which is computed from a user defined number of
adjacent traces. If the amplitude in this window exceeds
the threshold x the background amplitude then the samples within
the selected window may be scaled, replaced by threshold x background amplitude
or set to zero. Tapering is applied if scaling.

"""
import os
import sys
import glob
import yaml
import argparse
import datetime
from shutil import copy2
from contextlib import redirect_stdout

import numpy as np
import segyio
from tqdm import tqdm

from pseudo_3D_interpolation.functions.signal import rms
from pseudo_3D_interpolation.functions.filter import moving_window_2D
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
        description='Despike SEG-Y file(s) using 2D moving window function.'
    )
    parser.add_argument('input_path', type=str, help='Input file or directory.')
    parser.add_argument('--output_dir', '-o', type=str,
                        help='Output directory for edited SEG-Y file(s)')
    parser.add_argument('--inplace', '-i', action='store_true',
                        help='Edit SEG-Y file(s) inplace')
    parser.add_argument('--suffix', '-s', type=str, default='sgy',
                        help='File suffix. Only used when "input_path" is a directory.')
    parser.add_argument('--filename_suffix', '-fns', type=str,
                        help='Filename suffix for guided selection (e.g. "env" or "despk"). Only used when "input_path" is a directory.')
    parser.add_argument('--use_delay', action='store_true',
                        help='Use delay recording time to split input data before despiking (e.g. for TOPAS, Parasound)')
    parser.add_argument('--byte_delay', type=int, default=109,
                        help='Byte position of input delay times in SEG-Y file(s). Default: 109')
    parser.add_argument('--txt_suffix', type=str, default='despk',
                        help='Additional text to append to output filename.')
    parser.add_argument('--mode', '-m', type=str, default='mean', choices=['mean', 'median', 'rms'],
                        help='Mode used to compute background amplitude and detect spikes in data')
    parser.add_argument('--window_time', '-wti', type=int, required=True,
                        help='Moving window shape in time domain (TWT [ms])')
    parser.add_argument('--window_traces', '-wtr', type=int, required=True,
                        help='Moving window shape in offset domain (traces [#])')
    parser.add_argument('--window_overlap', '-wo', type=int, default=10, metavar='PERC',
                        help='Time overlap of moving windows (in percent)')
    parser.add_argument('--threshold_factor', '-t', type=float,
                        help='Threshold x background amplitude will be used for spike detection')
    parser.add_argument('--out_amplitude', '-oa', type=str, default='threshold',
                        choices=['scaled', 'mode', 'threshold', 'zeros', 'median'],
                        help='Spike amplitudes are replaced using selected method')
    parser.add_argument('--verbose', '-V', type=int, nargs='?', default=0, choices=[0, 1, 2],
                        help='Level of output verbosity (default: 0).')
    return parser
# fmt: on


def despike_2D(
    array, window, dt, overlap=10, ntraces=5, mode='mean', threshold=2, out='scaled', verbosity=0
):
    """
    Remove single-trace noise bursts from seismic data.
    The algorithm is based on the amplitude of the data within a time window.
    This amplitude is compared to the background amplitude which is computed from a user-defined number of adjacent traces.
    If the amplitude in this window exceeds the threshold x the background amplitude then the samples within
    the selected window may be scaled, replaced by threshold x background amplitude or set to zero.
    Tapering is applied to any replacement operation.

    Parameters
    ----------
    array : np.ndarray
        Seismic data (samples x traces).
    window : int
        Time window (in ms).
    dt : float
        Sampling interval in milliseconds [ms].
    overlap : int, optional
        Window overlap in percentage (%).
    ntraces : int, optional
        Number of adjacent traces (default: `5`).
    mode : str, optional
        Algorithm to compute amplitude in window [mean, rms, median] (default: `mean`).
    threshold : float, optional
        Amplitude threshold for spike detection (default: `2`).
    out : str, optional
        Amplitude values replacing spike values (default: `scaled`).
        
        - `scaled`: Scale signal down to background amplitude (based on mode). Tapering applied.
        - `mode`: Replace with background amplitude values
        - `threshold`: Replace with threshold * background amplitude values.
        - `zeros`: Replace with zero values.
        - `median`: Replace with median values (calculated from neighboring traces).

    Returns
    -------
    despiked : np.ndarray
        Despiked input data.

    """
    functions = {'mean': np.mean, 'median': np.median, 'rms': rms}

    # checks
    if overlap < 0 or overlap > 100:
        raise ValueError('Overlap must be integer between 0 and 100 [%].')
    if threshold < 0:
        raise ValueError('Theshold must be positive.')
    if ntraces % 2 == 0:
        raise ValueError('Number of traces must be odd integer.')
    if mode not in ['mean', 'rms', 'median']:
        raise ValueError("Amplitude mode must be one of ['mean', 'rms', 'median'].")
    else:
        func = functions.get(mode)
    replace_amp_mode = ['scaled', 'mode', 'threshold', 'zeros', 'median']
    if out not in replace_amp_mode:
        raise ValueError(f"Output amplitude option must be one of {replace_amp_mode}.")

    # shape of moving window
    win = (int(window / dt), ntraces)
    xprint(f'win: {win}', kind='debug', verbosity=verbosity)

    # compute overlap in samples (from time [ms])
    overlap = np.around(overlap / 100 * win[0], 0)
    overlap = overlap if overlap >= 1 else 1
    # trace step size
    dx = 1
    # time step size
    dy = int(win[0] - overlap)
    xprint(f'dy (twt): {dy},  dx (traces): {dx}', kind='debug', verbosity=verbosity)

    # ---------------------------------------------------------------------
    #                       initial selection
    # ---------------------------------------------------------------------
    # get view of moving windows
    v = moving_window_2D(array, win, dx, dy)
    xprint('Creating moving window views', kind='debug', verbosity=verbosity)
    xprint('v.shape:  ', v.shape, kind='debug', verbosity=verbosity)
    xprint('v.strides:', v.strides, kind='debug', verbosity=verbosity)

    # one mode value per row (time sample) of search window (adjacent traces)
    v_mode = func(np.abs(v), axis=(-1))  # (-1): one value per row, (-2,-1): single mode
    xprint('v_mode.shape:  ', v_mode.shape, kind='debug', verbosity=verbosity)

    # 4D indices of samples representing potential spike
    v_idx = np.where(
        np.abs(v) > threshold * v_mode.reshape(v_mode.shape + (1,))
    )  # (1,): one value per row, (1,1): single mode

    # convert 4D view indices to 2D data array indices & filter for unique ones
    idx_uniq = np.unique(np.vstack((v_idx[0] * dy + v_idx[2], v_idx[1] * dx + v_idx[3])).T, axis=0)

    # filter indices based on total number of spike samples per trace
    # -> discard traces where total sample number <= 10% of input time window
    # (assumes reasonable user input for spike length)
    tr_idx_uniq, tr_idx_cnt = np.unique(
        idx_uniq[:, 1], return_counts=True
    )  # get trace indices & counts
    tr_idx_uniq_to_keep = np.where(tr_idx_cnt > win[0] * 0.1)  # create boolean filter
    tr_idx_to_keep = tr_idx_uniq[tr_idx_uniq_to_keep]  # trace indices to keep

    # select only valid data (number of samples > 10% of window length)
    if len(tr_idx_to_keep) > 0:
        idx_uniq = idx_uniq[np.isin(idx_uniq[:, 1], tr_idx_to_keep)]
    # create dummy array
    else:
        idx_uniq = np.empty((0, 2), dtype='int')
    xprint('idx_uniq.shape:  ', idx_uniq.shape, kind='debug', verbosity=verbosity)

    # ---------------------------------------------------------------------
    #                       additional selection
    # ---------------------------------------------------------------------
    # create additional stride views (if required)
    N = array.shape[0]  # number of rows in array (time domain)
    M = win[0]  # number of rows in moving window (time domain)
    # check if view left out some part of input array
    # i: start row index of moving window (based on dy)
    # (i - dy + M != N): check if previous view already covered rows until end of array, if not --> additional view needed!
    # (N - i < dy): check number of rows in last view are less than dy --> no view then!
    missing_views = [i for i in range(0, N, dy) if (N - i < dy)]  # (i - dy + M != N) and
    if any(missing_views):
        # compute start sample index for view ranging until last sample
        start = missing_views[0] - (M - (N - missing_views[0]))
        # get view of moving windows (previously missing data range)
        v_add = moving_window_2D(array[start:], win, dx, dy)
        xprint('v_add.shape:  ', v_add.shape, kind='debug', verbosity=verbosity)
        xprint('v_add.strides:', v_add.strides, kind='debug', verbosity=verbosity)

        # mode of additional views
        v_add_mode = func(np.abs(v_add), axis=(-1))  # (-1): one value per row, (-2,-1): single mode
        xprint('v_add_mode.strides:', v_add_mode.strides, kind='debug', verbosity=verbosity)

        # 4D indices of samples representing potential spike
        v_add_idx = np.where(
            np.abs(v_add) > threshold * v_add_mode.reshape(v_add_mode.shape + (1,))
        )  # (1,): one value per row, (1,1): single mode

        # convert 4D view indices to 2D data array indices (using `start` variable) & filter for unique ones
        idx_add_uniq = np.unique(
            np.vstack(
                (start + v_add_idx[0] * dy + v_add_idx[2], v_add_idx[1] * dx + v_add_idx[3])
            ).T,
            axis=0,
        )

        # filter indices based on total number of spike samples per trace
        # -> discard traces where total sample number <= 10% of input time window
        # (assumes reasonable user input for spike length)
        tr_idx_add_uniq, tr_idx_add_cnt = np.unique(idx_add_uniq[:, 1], return_counts=True)
        tr_idx_add_uniq_to_keep = np.where(tr_idx_add_cnt > win[0] * 0.1)
        tr_idx_add_to_keep = tr_idx_add_uniq[tr_idx_add_uniq_to_keep]

        # select only valid data (number of samples > 10% of window length)
        if len(tr_idx_add_to_keep) > 0:
            idx_add_uniq = idx_add_uniq[np.isin(idx_add_uniq[:, 1], tr_idx_add_to_keep)]
        # create dummy array
        else:
            idx_add_uniq = np.empty((0, 2), dtype='int')
    else:
        idx_add_uniq = np.empty((0, 2), dtype='int')
    xprint('idx_add_uniq.shape:', idx_add_uniq.shape, kind='debug', verbosity=verbosity)

    # ---------------------------------------------------------------------
    #                   combine spike sample indices
    # ---------------------------------------------------------------------
    # merge indices from both selection processes & select unique ones
    idx_merge = np.unique(np.concatenate((idx_uniq, idx_add_uniq), axis=0), axis=0)
    xprint('idx_merge.shape:', idx_merge.shape, kind='debug', verbosity=verbosity)

    if idx_merge.size == 0:
        xprint(
            f'No spikes detected ({array.shape[1]} traces). Consider adjusting the input parameters.',
            kind='info',
            verbosity=verbosity,
        )
        return array

    # sort indices based on (1) trace index and (2) sample index
    sorter = np.lexsort((idx_merge[:, 0], idx_merge[:, 1]))
    idx_merge_sorted = idx_merge[sorter]
    xprint('idx_merge_sorted.shape:', idx_merge_sorted.shape, kind='debug', verbosity=verbosity)

    # ---------------------------------------------------------------------
    #                   filter & discard false detections
    # ---------------------------------------------------------------------
    # split index array by trace index
    spike_index_arrays = []
    spikes_per_trace = np.split(idx_merge_sorted, np.where(np.diff(idx_merge_sorted[:, 1]))[0] + 1)
    # loop over every spike array
    for spike in spikes_per_trace:
        # filter indices if difference to following index > 5% of window length (in samples)
        sample_indices_change_idx = np.where(np.diff(spike[:, 0]) > win[0] * 0.05)[0] + 1
        # split spike indices array by indices of large changes & keep spike only if spike length > 5% of window length (in samples)
        rewrapper_despiking_2D_segying_spikes = [
            a
            for a in np.split(spike, sample_indices_change_idx, axis=0)
            if a.shape[0] > win[0] * 0.05
        ]
        # append spike index array to list of all spikes in data array
        spike_index_arrays.extend(rewrapper_despiking_2D_segying_spikes)

    # stack rewrapper_despiking_2D_segying spike index arrays after filtering
    try:
        idx_merge_sorted = np.concatenate(spike_index_arrays, axis=0)
        xprint(
            'idx_merge_sorted.shape (filtered):',
            idx_merge_sorted.shape,
            kind='debug',
            verbosity=verbosity,
        )

    except ValueError:
        xprint(
            f'No spikes detected ({array.shape[1]} traces). Consider adjusting the input parameters.',
            kind='info',
            verbosity=verbosity,
        )
        return array
    # ---------------------------------------------------------------------
    #                   replace spike amplitudes
    # ---------------------------------------------------------------------
    # make copy of input data
    array_out = array  # .copy()

    # init list for indices plotting
    spike_time_ranges_indices = []

    # loop over each spike index array
    for idx_arr in spike_index_arrays:
        # get index of trace
        idx_trace = idx_arr[0, 1]
        xprint('idx_trace.shape:', int(idx_trace), kind='debug', verbosity=verbosity)

        # extract shape of spike index array
        N_spike, M_spike = idx_arr.shape
        xprint('idx_arr.shape:', idx_arr.shape, kind='debug', verbosity=verbosity)

        # get data subset
        ## pad minimum and maximum sample index with 10% of spike sample range (for tapering)
        ### minimum
        idx_sample_min = idx_arr[0, 0] - int(N_spike * 0.1)
        idx_sample_min = idx_sample_min if idx_sample_min >= 0 else 0
        spike_time_ranges_indices.append((idx_sample_min, idx_trace))
        ## maximum
        idx_sample_max = idx_arr[-1, 0] + int(N_spike * 0.1) + 1
        idx_sample_max = idx_sample_max if idx_sample_max <= N else N
        spike_time_ranges_indices.append((idx_sample_max, idx_trace))
        ## account for traces at data limits
        ### get trace window half-width
        tr_win_idx = win[1] // 2
        xprint(f'tr_win_idx: {tr_win_idx}', kind='debug', verbosity=verbosity)

        idx_trace_min = idx_trace - tr_win_idx
        idx_trace_max = idx_trace + tr_win_idx + 1
        xprint(
            f'idx_trace_min: {int(idx_trace_min)}, idx_trace_max: {int(idx_trace_max)}',
            kind='debug',
            verbosity=verbosity,
        )

        # account for boundaries of seismic section
        if idx_trace_min < 0:
            idx_trace_min = 0  # no negative indexing possible
        xprint(f'tr_win_idx (new): {tr_win_idx}', kind='debug', verbosity=verbosity)
        ## select subset of input data based on calculated ranges
        # spike_win = array_out[idx_sample_min:idx_sample_max, idx_trace_min:idx_trace_max]
        spike_win = array_out[
            int(idx_sample_min) : int(idx_sample_max), int(idx_trace_min) : int(idx_trace_max)
        ]
        xprint(
            'idx_sample_min, idx_sample_max, tr_win_idx:',
            idx_sample_min,
            idx_sample_max,
            tr_win_idx,
            kind='debug',
            verbosity=verbosity,
        )
        xprint('spike_win:', spike_win.shape, kind='debug', verbosity=verbosity)

        # calc output amplitudes (using user-specified mode)
        spike_amps = spike_win[:, tr_win_idx]  # amplitudes of fishy trace

        if out == 'scaled':
            neighbor_amps = func(
                np.abs(spike_win), axis=1
            )  # * threshold # adjusted mode amplitude (x threshold)
            spike_win_amps_out = spike_amps / (spike_amps.max() / neighbor_amps)

            # creating taper window
            w = np.blackman(len(spike_win_amps_out))
            # tapering resulting amplitudes
            spike_win_amps_out = spike_win_amps_out * w

        elif out == 'mode':
            neighbor_amps = func(spike_win, axis=1)
            spike_win_amps_out = neighbor_amps

        elif out == 'threshold':
            neighbor_amps = func(spike_win, axis=1)
            spike_win_amps_out = neighbor_amps * threshold

        elif out == 'zeros':
            spike_win_amps_out = np.zeros_like(spike_amps)

        elif out == 'median':
            neighbor_amps = np.median(spike_win, axis=1)
            spike_win_amps_out = neighbor_amps

        # replace spike amplitudes in window with scaled ones
        spike_win[:, tr_win_idx] = spike_win_amps_out.astype(spike_win.dtype)

    return array_out


def wrapper_despiking_2D_segy(in_path, args):
    """Despike input SEG-Y file(s)."""
    basepath, filename = os.path.split(in_path)
    basename, suffix = os.path.splitext(filename)
    xprint(f'Processing file < {filename} >', kind='info', verbosity=args.verbose)

    default_txt_suffix = 'despk'
    
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

    with segyio.open(path, 'r+', strict=False, ignore_geometry=True) as src:
        # read common metadata
        dt = segyio.tools.dt(src) / 1000  # sample rate [ms]
        twt = src.samples  # two way travel time (TWTT) [ms]

        # field record number
        fldr = src.attributes(segyio.TraceField.FieldRecord)[:]

        # get DelayRecordingTimes from trace headers
        delrt = src.attributes(args.byte_delay)[:]  # segyio.TraceField.DelayRecordingTime

        # find indices where DelayRecordingTime changes
        delrt_idx = np.where(np.roll(delrt, 1) != delrt)[0]
        if delrt_idx.size != 0 and delrt_idx[0] == 0:
            delrt_idx = delrt_idx[1:]

        global data_src, data
        # get seismic data [amplitude]; transpose to fit numpy data structure
        data_src = src.trace.raw[:].T  # eager version (completely read into memory)
        if data_src.shape[-1] < args.window_traces:
            # input SEG-Y file is too small for despiking
            return path, None, None, dt, twt, fldr, delrt
        else:
            data = data_src.copy()

        # split seismic section according to DelayRecordingTime
        if args.use_delay and len(delrt_idx) >= 1:
            xprint('Splitting seismic section using `delrt`', kind='info', verbosity=args.verbose)

            # split seismic section based on delrt
            data_splits = np.array_split(data, delrt_idx, axis=1)

            data_despiked_list = []
            # despike individual despke splits
            for d in data_splits:
                d_despiked = despike_2D(
                    d,
                    window=args.window_time,
                    dt=dt,
                    overlap=args.window_overlap,
                    ntraces=args.window_traces,
                    mode=args.mode,
                    threshold=args.threshold_factor,
                    out=args.out_amplitude,
                    verbosity=args.verbose,
                )
                data_despiked_list.append(d_despiked)
            # merge despiked splits into single array
            data_despiked = np.concatenate(data_despiked_list, axis=1, dtype=data.dtype)

        else:
            # remove spikes in seismic section
            data_despiked = despike_2D(
                data,
                window=args.window_time,
                dt=dt,
                overlap=args.window_overlap,
                ntraces=args.window_traces,
                mode=args.mode,
                threshold=args.threshold_factor,
                out=args.out_amplitude,
                verbosity=args.verbose,
            )

        # sanity check
        msg = 'Input and output data must be of same shape'
        assert data.shape == data_despiked.shape, msg

        # set output amplitudes (transpose to fit SEG-Y format)
        src.trace = np.ascontiguousarray(data_despiked.T, dtype=data.dtype)

    # update textual header
    text = get_textual_header(path)
    text_updated = add_processing_info_header(text, 'DESPIKE', prefix='_TODAY_', newline=True)
    write_textual_header(path, text_updated)

    return path, data_src, data_despiked


def main(argv=sys.argv):  # noqa
    TIMESTAMP = datetime.datetime.now().isoformat(timespec='seconds').replace(':', '')
    SCRIPT = os.path.splitext(os.path.basename(__file__))[0]

    parser = define_input_args()
    args = parser.parse_args(argv[1:])  # exclude filename parameter at position 0
    xprint(args, kind='debug', verbosity=args.verbose)

    # sanity checks
    #     sys.exit('[ERROR]    Either "output_dir" OR "args.inplace" must be specified.')
    if args.window_overlap < 0 or args.window_overlap > 99:
        sys.exit('[ERROR]    Please set window overlap to a reasonable value [0-99].')
    if args.threshold_factor <= 0:
        sys.exit('[ERROR]    Threshold factor must be larger than zero.')

    # check input file(s)
    in_path = args.input_path
    basepath, filename = os.path.split(in_path)
    basename, suffix = os.path.splitext(filename)
    if suffix == '':
        basepath = in_path
        basename, suffix = None, None  # noqa

    # save input command line parameters to file
    if args.verbose >= 1:
        xprint('Saving argparse parameter to file', kind='info', verbosity=args.verbose)
        dir_args = args.output_dir if args.output_dir is not None else basepath
        path_args = os.path.join(dir_args, f'{TIMESTAMP}_{SCRIPT}_argparse_parameter.yml')
        with open(path_args, mode='w', newline='\n') as fargs:
            yaml.safe_dump(vars(args), fargs, encoding='utf-8')

    # (1) single input file
    if os.path.isfile(in_path) and suffix != '.txt':
        # despike input file
        path, data, data_despiked = wrapper_despiking_2D_segy(in_path, args)
        if data is None and data_despiked is None:
            xprint(
                'Input SEG-Y contains too less traces for despiking ---> skipped file!',
                kind='warning',
                verbosity=args.verbose,
            )
            os.remove(path)
        elif np.allclose(data, data_despiked):
            xprint(
                '*** No spikes removed! Consider adjusting the input parameters. ***',
                kind='warning',
                verbosity=args.verbose,
            )
            os.remove(path)

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
                xprint(
                    f'Processing total of < {len(file_list)} > files',
                    kind='info',
                    verbosity=args.verbose,
                )
                for file_path in tqdm(
                    file_list,
                    desc='Despike',
                    ncols=80,
                    total=len(file_list),
                    unit_scale=True,
                    unit=' files',
                ):
                    # despike input file
                    path, data, data_despiked = wrapper_despiking_2D_segy(file_path, args)

                    if data is None and data_despiked is None:
                        xprint(
                            'Input SEG-Y contains too less traces for despiking ---> skipped file!',
                            kind='warning',
                            verbosity=args.verbose,
                        )
                        os.remove(path)
                    elif np.allclose(data, data_despiked):
                        xprint(
                            '*** No spikes removed! Consider adjusting the input parameters. ***',
                            kind='warning',
                            verbosity=args.verbose,
                        )
                        os.remove(path)
        clean_log_file(logfile)
    else:
        sys.exit('No input files to process. Exit process.')


#%% MAIN

if __name__ == '__main__':

    main()
