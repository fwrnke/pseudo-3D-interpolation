"""
Utility script to fix incorrect _DelayRecordingTime_ in SEG-Y file(s).
It compares maximum amplitudes of neighboring traces in user-defined windows (e.g., 3 ms x 5 traces).

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
from tqdm import tqdm

from pseudo_3D_interpolation.functions.plot import plot_seismic_wiggle
from pseudo_3D_interpolation.functions.header import (
    get_textual_header,
    add_processing_info_header,
    write_textual_header,
)
from pseudo_3D_interpolation.functions.utils import xprint, clean_log_file

np.set_printoptions(precision=5, suppress=True)


#%% FUNCTIONS
# fmt: off
def define_input_args():  # noqa
    parser = argparse.ArgumentParser(
        description='Fix incorrect "DelayRecordingTime" in SEG-Y file(s).')
    parser.add_argument('input_path', type=str, help='Input file or directory.')
    parser.add_argument('--output_dir', '-o', type=str,
                        help='Output directory for edited SEG-Y file(s).')
    parser.add_argument('--inplace', '-i', action='store_true',
                        help='Edit SEG-Y file(s) inplace')
    parser.add_argument('--suffix', '-s', type=str,
                        help='File suffix. Only used when "input_path" is a directory.')
    parser.add_argument('--filename_suffix', '-fns', type=str,
                        help='Filename suffix for guided selection (e.g. "env" or "despk"). Only used when "input_path" is a directory.')
    parser.add_argument('--txt_suffix', type=str,
                        help='Additional text to append to output filename.')
    parser.add_argument('--byte_delay', type=int, default=109,
                        help='Byte position of input delay times in SEG-Y file(s) (default: 109, "DelayRecordingTime")')
    parser.add_argument('--win_ntraces', type=int, default=5,
                        help='Number of traces in comparison window.')
    parser.add_argument('--win_nsamples', type=int, default=120,
                        help='Number of samples in comparison window.')
    parser.add_argument('--verbose', '-V', type=int, nargs='?', default=0, choices=[0, 1, 2],
                        help='Level of output verbosity (default: 0).')
    return parser
# fmt: on


def check_varying_DelayRecordingTimes(path, byte_delay=109):
    """
    Check SEG-Y file for varying DelayRecordingTimes and return boolean type.

    Parameters
    ----------
    path : str
        File path of SEG-Y file to check.
    byte_delay : int, optional
        Byte position of DelayRecordingTime (default: 109).

    """
    with segyio.open(path, 'r', strict=False, ignore_geometry=True) as file:
        # get "DelayRecordingTime" for each trace
        delrt = file.attributes(byte_delay)[:]

    # find unique DelayRecordingTimes and corresponding trace index
    delrt_uniq, delrt_idx = np.unique(delrt, return_index=True)

    if len(delrt_uniq) > 1:
        return True
    else:
        return False


def correct_single_trace_DelayRecordingTime(
    idx, data, delrt, fldr, n_traces=5, n_samples=120, verbosity=0
):
    """
    Correct false trace DelayRecordingTime (delrt).
    Comparing data maximum amplitude to the max amplitude
    within a `n_samples` sample window of `n_traces` neighboring traces to each side.

    Parameters
    ----------
    idx : int
        Index of change in DelayRecordingTimes.
    data : numpy.ndarray
        2D data array subset (samples x traces).
    delrt : numpy.ndarray
        1D array subset of DelayRecordingTimes.
    fldr : numpy.ndarray
        1D array subset of FieldRecord numbers.
    n_traces : int, optional
        Number of neighboring traces of reference trace (default: 5).
    n_samples : int, optional
        Number of samples per trace for comparision of max. amplitude (default: 120).

    Returns
    -------
    ref_tr_delrt_corrected : int
        Correct DelayRecordingTime of reference trace.
    idx_n_traces : int
        Index of (updated) reference trace (only for visualization plots).

    """
    # select only reference trace
    ref_tr = data[:, n_traces]

    # find max amplitude value and correspondig array index
    ref_tr_peak_val = ref_tr.max()
    ref_tr_peak_idx = ref_tr.argmax()
    xprint(
        f'ref_tr_peak_idx:  {ref_tr_peak_idx} --> {ref_tr_peak_val}',
        kind='debug',
        verbosity=verbosity,
    )

    # get max amplitudes from data subset (per trace)
    ref_tr_peak_idx_min = (
        ref_tr_peak_idx - n_samples // 2 if ref_tr_peak_idx - n_samples // 2 >= 0 else 0
    )  # account for negative indices
    ref_tr_peak_idx_max = ref_tr_peak_idx + n_samples // 2 + 1
    tr_amp_maxima = data[ref_tr_peak_idx_min:ref_tr_peak_idx_max].max(axis=0)
    tr_amp_max_idx_trace = tr_amp_maxima[n_traces]
    tr_amp_maxima[
        tr_amp_maxima > tr_amp_max_idx_trace
    ] = tr_amp_max_idx_trace  # set all values > max amplitude of reference trace
    xprint(
        f'tr_amp_maxima:          {np.around(tr_amp_maxima,4)}', kind='debug', verbosity=verbosity
    )

    # difference amplitude to reference trace (absolut)
    tr_amp_max_diff_abs = np.abs(tr_amp_maxima - ref_tr_peak_val)
    xprint(
        f'tr_amp_max_diff_abs:    {np.around(tr_amp_max_diff_abs,4)}',
        kind='debug',
        verbosity=verbosity,
    )

    # difference amplitude to reference trace (relativ)
    tr_amp_max_diff_rel = tr_amp_max_diff_abs / ref_tr_peak_val
    xprint(
        f'tr_amp_max_diff_rel:    {np.around(tr_amp_max_diff_rel,4)}',
        kind='debug',
        verbosity=verbosity,
    )

    # boolean indices indicating similar trace amplitudes (based on max amplitude in n_samples)
    # tr_amp_similarity = np.around(tr_amp_max_diff_rel, 0).astype('int')           # 1st approach
    tr_amp_similarity = np.where(tr_amp_max_diff_rel > 0.8, 1, 0).astype(
        'int'
    )  # 2nd approach: quite sufficient (+ replace amp > ref with ref)
    # tr_amp_similarity = rescale_linear(tr_amp_maxima)                               # 3rd approach: linear scaling
    # tr_amp_similarity = np.around(tr_amp_similarity, 0).astype('int')
    xprint(f'tr_amp_similarity:    {tr_amp_similarity}', kind='debug', verbosity=verbosity)

    # boolean indices indicating similar delrt for traces in data subset
    delrt_similarity = np.where(delrt == delrt.max(), 1, 0)
    delrt_similarity_inv = np.abs(delrt_similarity - 1)
    xprint(f'delrt_similarity:        {delrt_similarity}', kind='debug', verbosity=verbosity)
    xprint(f'delrt_similarity_inv:    {delrt_similarity_inv}', kind='debug', verbosity=verbosity)

    # sanity check: max amplitudes and DelayRecordingTimes are matching
    # e.g. [1, 1, 1, 0, 0, 0, 0] or [0, 0, 0, 1, 1, 1, 1]
    r = tr_amp_similarity[n_traces]
    if (
        np.all(tr_amp_similarity[:n_traces] == r) and np.all(tr_amp_similarity[n_traces + 1 :] != r)
    ) or (
        np.all(tr_amp_similarity[:n_traces] != r) and np.all(tr_amp_similarity[n_traces + 1 :] == r)
    ):

        # [1]: no correction needed
        if np.array_equal(tr_amp_similarity, delrt_similarity) or np.array_equal(
            tr_amp_similarity, delrt_similarity_inv
        ):
            xprint('<<< No correction needed >>>', kind='debug', verbosity=verbosity)
            return None, None

        # [2]: incorrect DelayRecordingTime
        else:
            xprint('*** Incorrect DelayRecordingTime! ***', kind='warning', verbosity=verbosity)
            # get unique DelayRecordingTime and corresponding indices
            delrt_uniq, delrt_idx = np.unique(delrt, return_index=True)
            # get other DelayRecordingTime(s)
            ref_tr_delrt_corrected = delrt_uniq[delrt_uniq != delrt[n_traces]]

            idx_n_traces = n_traces

    # check if incorrect DelayRecordingTime trace is offset
    elif [np.sum(tr_amp_similarity[:n_traces]), np.sum(tr_amp_similarity[n_traces + 1 :])] in [
        [n_traces, 1],
        [1, n_traces],
    ]:
        # [np.sum(tr_amp_similarity[:n_traces]), np.sum(tr_amp_similarity[n_traces+1:])] in [[n_traces-1,0], [0,n_traces-1]] #3rd approach
        xprint('Eligible for adjusting offset trace', kind='debug', verbosity=verbosity)
        # convert for comprehension
        tr_amp_similarity = list(tr_amp_similarity)

        # check if first two and last two traces of subset are equal (boundary condition)
        if all(x in [tr_amp_similarity[:2], tr_amp_similarity[-2:]] for x in [[1, 1], [0, 0]]):
            xprint(
                '*** [OFFSET TRACE] Incorrect DelayRecordingTime! ***',
                kind='warning',
                verbosity=verbosity,
            )

            idx_peak_amp_changes = np.where(np.roll(tr_amp_similarity, 1) != tr_amp_similarity)[0]
            # print('idx_peak_amp_changes:', idx_peak_amp_changes)

            # fishy trace after before of DelayRecordingTime
            if len(idx_peak_amp_changes[idx_peak_amp_changes > n_traces]) < len(
                idx_peak_amp_changes[idx_peak_amp_changes < n_traces]
            ):
                idx_ = idx_peak_amp_changes[1]
            # fishy trace after change of DelayRecordingTime
            elif len(idx_peak_amp_changes[idx_peak_amp_changes > n_traces]) > len(
                idx_peak_amp_changes[idx_peak_amp_changes < n_traces]
            ):
                idx_ = idx_peak_amp_changes[-2]
                # two isolated traces with false DelayRecordingTime!
            else:
                sys.exit('[ERROR]    Something is really messed up here. Check your data!')

            # get unique DelayRecordingTime and corresponding indices
            delrt_uniq, delrt_idx = np.unique(delrt, return_index=True)
            # get other DelayRecordingTime(s)
            ref_tr_delrt_corrected = delrt_uniq[delrt_uniq != delrt[idx_]]
            # print('ref_tr_delrt_corrected:', ref_tr_delrt_corrected)
            # set correct index for print message
            idx_n_traces = idx_
        else:
            return None, None
    else:
        return None, None

    if len(ref_tr_delrt_corrected) > 1:
        xprint(
            'ref_tr_delrt_corrected: ', ref_tr_delrt_corrected, kind='debug', verbosity=verbosity
        )
        xprint('idx_n_traces:           ', idx_n_traces, kind='debug', verbosity=verbosity)
        xprint(
            'Found more than one DelayRecordingTime to choose from. No changes applied.',
            kind='error',
            verbosity=verbosity,
        )
        return None, None
    else:
        return ref_tr_delrt_corrected[0], idx_n_traces


def check_DelayRecordingTime_changes(
    file,
    tracecount,
    byte_delay=109,
    n_traces=5,
    n_samples=120,
    update_segy=False,
    plot_org=False,
    plot_corr=False,
    filename='',
    verbosity=0,
):
    """
    Check SEG-Y file (e.g. PARASOUND or TOPAS data) for potential incorrect DelayRecordingTimes.
    In case of a incorrectly assigned DelayRecordingTime the corresponding values will be updates inplace!

    Parameters
    ----------
    file : segyio.SegyFile
        An open segyio file handle.
    tracecount : int
        Number of traces in SEG-Y file.
    byte_delay : int
        Byte position of DelayRecordingTime in SEG-Y trace header.
    n_traces : int, optional
        Number of neighboring traces of reference trace (default: 5).
    n_samples : int, optional
        Number of samples per trace for comparision of max. amplitude (default: 7).
    update_segy : bool, optional
        Enable inplace updating of DelayRecordingTime values in SEG-Y trace header (default: False).
    plot_org : bool, optional
        Create plot of original traces with auxiliary information (default: False).
    plot_corr : bool, optional
        Create plot of corrected traces with auxiliary information (default: False).

    """
    # two way travel time (TWTT) [ms]
    twt = file.samples

    # trace sequence number within reel (starting at 1)
    tracr = file.attributes(segyio.TraceField.TRACE_SEQUENCE_FILE)[:]
    # get "FieldRecordNumber" for each trace
    fldr = file.attributes(segyio.TraceField.FieldRecord)[:]

    # get "DelayRecordingTime" for each trace
    delrt = file.attributes(byte_delay)[:]

    # find indices where DelayRecordingTime changes
    delrt_idx = np.where(np.roll(delrt, 1) != delrt)[0]
    # sanity check: in case first and last trace of file have same DelayRecordingTime!
    if delrt_idx[0] != 0:
        delrt_idx = np.insert(delrt_idx, 0, 0, axis=0)

    # check for multiple DelayRecordingTimes in trace headers
    if len(delrt_idx) > 1:
        xprint(
            f'Found < {len(delrt_idx)-1} > different DelayRecordingTimes: {dict(zip(delrt_idx, delrt[delrt_idx]))}',
            kind='info',
            verbosity=verbosity,
        )

        # loop over all DelayRecordingTime change indices
        for i, idx in enumerate(delrt_idx[1:]):  # skip index: 0
            xprint(
                f'[{i+1}] === FRN: {fldr[idx]} - trace idx: {idx} - Delay: {delrt[idx]} ms ===',
                kind='debug',
                verbosity=verbosity,
            )

            # generate indices for subsetting of input data
            idx_subset_min = idx - n_traces
            idx_subset_max = idx + n_traces + 1
            # sanity check: make sure there are at least `n_traces` neighboring traces in data range
            if (idx_subset_min < 0) or (idx_subset_max > tracecount + 1):
                xprint(
                    f'Not enough neighboring traces for idx: {idx} [{idx_subset_min}:{idx_subset_max}] with >{tracecount}< total traces. Skipped data subset.',
                    kind='warning',
                    verbosity=verbosity,
                )
                continue

            # get subset of DelayRecordingTimes
            delrt_subset = delrt[idx_subset_min:idx_subset_max]
            if len(np.unique(delrt_subset)) > 2:
                xprint(
                    f'Too many different `delrt` for idx: {idx} [{idx_subset_min}:{idx_subset_max}]. Skipped data subset.',
                    kind='warning',
                    verbosity=verbosity,
                )
                continue
            # get subset of FieldRecord numbers
            fldr_subset = fldr[idx_subset_min:idx_subset_max]
            # get subset of trace sequence numbers
            tracr_subset = tracr[idx_subset_min:idx_subset_max]
            # get seismic data [amplitude]; transpose to fit numpy data structure
            data_subset = file.trace.raw[
                idx_subset_min:idx_subset_max
            ].T  # eager version (completely read into memory)

            # plot trace data (with delrt)
            if plot_org and verbosity == 2:
                plot_seismic_wiggle(
                    data_subset,
                    twt=twt,
                    traces=np.arange(idx_subset_min, idx_subset_max),
                    title=f'{filename} - original -',
                    add_kind=delrt_subset,
                    gain=1.0,
                    plot_kwargs=None,
                )

            # get correct DelayRecordingTime for reference trace
            delrt_corrected, idx_trace_subset_win = correct_single_trace_DelayRecordingTime(
                idx,
                data_subset,
                delrt_subset,
                fldr_subset,
                n_traces,
                n_samples,
                verbosity=verbosity,
            )

            # perform update only if correct DelayRecordingTime was calculated
            if delrt_corrected is not None:
                # update DelayRecordingTime for input trace
                msg = f'Changing DelayRecordingTime for FRN #{fldr_subset[idx_trace_subset_win]} '
                msg += (
                    f'(idx:{tracr_subset[idx_trace_subset_win]-1}) [i:{idx_trace_subset_win}] from '
                )
                msg += f'> {delrt_subset[idx_trace_subset_win]} < to > {delrt_corrected} <'
                xprint(msg, kind='info', verbosity=verbosity)

                # plot trace data (with corrected delrt)
                if plot_corr and verbosity == 2:
                    # copy delrt
                    delrt_subset_corr = delrt_subset
                    # replace incorrect delrt
                    delrt_subset_corr[idx_trace_subset_win] = delrt_corrected

                    plot_seismic_wiggle(
                        data_subset,
                        twt=twt,
                        traces=np.arange(idx_subset_min, idx_subset_max),
                        title=f'{filename} - corrected -',
                        add_info=delrt_subset_corr,
                        gain=1.0,
                        plot_kwargs=None,
                    )

                # update trace header DelayRecordingTime inplace
                if update_segy:
                    xprint(
                        'Updating trace header value of DelayRecordingTime inplace!',
                        kind='warning',
                        verbosity=verbosity,
                    )
                    header = file.header[idx]
                    xprint(
                        f'file.header[{idx}] (old): {file.header[idx]}',
                        kind='debug',
                        verbosity=verbosity,
                    )
                    header[byte_delay] = delrt_corrected
                    xprint(
                        f'file.header[{idx}] (new): {file.header[idx]}',
                        kind='debug',
                        verbosity=verbosity,
                    )

            # break
        return True
    else:
        return False


def wrapper_delrt_correction_segy(in_path, args):
    """Correct DelayRecordingTime (`delrt`) for single SEG-Y file."""
    basepath, filename = os.path.split(in_path)
    basename, suffix = os.path.splitext(filename)
    xprint(f'Processing file < {filename} >', kind='info', verbosity=args.verbose)

    to_process = check_varying_DelayRecordingTimes(in_path, args.byte_delay)
    if not to_process:
        return False
    
    default_txt_suffix = 'delrt'
    
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
    with segyio.open(path, 'r+', strict=False, ignore_geometry=True) as src:
        n_traces = src.tracecount  # total number of traces

        # check each change in DelayRecordingTimes
        c = check_DelayRecordingTime_changes(
            src,
            n_traces,
            args.byte_delay,
            args.win_ntraces,
            args.win_nsamples,
            update_segy=args.inplace,
            plot_org=False,
            plot_corr=True,
            filename=filename,
            verbosity=args.verbose,
        )

        # update textual header
        text = get_textual_header(path)
        info = f'DELRT FIX (BYTE:{args.byte_delay})'
        text_updated = add_processing_info_header(text, info, prefix='_TODAY_')
        write_textual_header(path, text_updated)

    # return boolean for sanity check
    return c


def main(argv=sys.argv):  # noqa
    TIMESTAMP = datetime.datetime.now().isoformat(timespec='seconds').replace(':', '')
    SCRIPT = os.path.basename(__file__).split(".")[0]

    # get command line input
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
        # process input file
        r = wrapper_delrt_correction_segy(in_path, args)
        if r is False:
            xprint('Skipped: Identical "DelayRecordingTime" for whole SEG-Y file', kind='info', verbosity=args.verbose)
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

    if len(file_list) > 0:
        # redirect stdout to logfile
        logfile = os.path.join(basepath, f'{TIMESTAMP}_{SCRIPT}.log')
        with open(logfile, 'w', newline='\n') as f:
            with redirect_stdout(f):
                xprint(f'Processing total of < {len(file_list)} > files', kind='info', verbosity=args.verbose)
                nprocessed = 0
                for file_path in tqdm(
                    file_list,
                    desc='Correct delrt',
                    ncols=80,
                    total=len(file_list),
                    unit_scale=True,
                    unit=' files',
                ):
                    # pad traces of SEG-Y file
                    r = wrapper_delrt_correction_segy(file_path, args)

                    if r is False:
                        xprint(
                            'Skipped: Identical "DelayRecordingTime" for whole SEG-Y file',
                            kind='info',
                            verbosity=args.verbose,
                        )
                        continue
                    else:
                        nprocessed += 1
                xprint(
                    f'Fixed a total of < {nprocessed} > out of < {len(file_list)} > files',
                    kind='info',
                    verbosity=args.verbose,
                )
        clean_log_file(logfile)
    else:
        sys.exit('[INFO]    No input files to process. Exit process.')


#%% MAIN

if __name__ == '__main__':

    main()
