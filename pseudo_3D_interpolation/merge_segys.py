"""Utility script to merge short SEG-Y file(s) with neighboring ones."""

import os
import sys
import glob
import argparse
import datetime
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import segyio

from pseudo_3D_interpolation.functions.utils import xprint
from pseudo_3D_interpolation.functions.header import (
    get_textual_header,
    add_processing_info_header,
    write_textual_header,
)

#%% FUNCTIONS
# fmt: off
def define_input_args():  # noqa
    parser = argparse.ArgumentParser(
        description='Utility script to merge short SEG-Y file(s) with longer ones.'
        )
    parser.add_argument('input_path', type=str, help='Input file directory or path to datalist.')
    parser.add_argument('--filename_suffix', '-fns', type=str, default='',
                        help='Filename suffix for guided selection (e.g. "env" or "despk"). Only used when "input_path" is a directory.')
    parser.add_argument('--suffix', '-s', type=str, default='sgy',
                        help='File suffix. Only used when "input_path" is a directory.')
    parser.add_argument('--txt_suffix', type=str, default='merge',
                        help='Additional text to append to output filename.')
    parser.add_argument('--filesize_kB', type=float, default=2000,
                        help='Files smaller than filesize (in kB) will be merged with closest SEG-Y file.')
    parser.add_argument('--verbose', '-V', type=int, nargs='?', default=0, choices=[0, 1, 2],
                        help='Level of output verbosity (default: 0).')
    return parser
# fmt: on


def parse_trace_headers(segyfile):
    r"""
    Parse the SEG-Y file trace headers into a pandas DataFrame.

    Parameters
    ----------
    segyfile : segyio.SegyFile
        Input SEG-Y file.

    Returns
    -------
    df : pandas.DataFrame
        Dataframe of parsed trace headers.

    """
    # get number of traces
    n_traces = segyfile.tracecount

    # get all header keys
    headers = segyio.tracefield.keys

    # initialize dataframe with trace id as index and headers as columns
    df = pd.DataFrame(index=range(1, n_traces + 1), columns=headers.keys())
    
    # Fill dataframe with all header values
    for k, v in headers.items():
        df[k] = segyfile.attributes(v)[:]
        
    return df


def get_files_to_merge(list_segys, fsize_kB=2000, suffix='sgy', verbosity=1):
    """
    Create list holding tuples of files to merge.

    Parameters
    ----------
    list_segys : list
        Input list of SEG-Y files.
    fsize_kB : float, int, optional
        Threshold filesize below which files will be merged together (default: 2000 kB).
    verbosity : int, optional
        Print verbosity (default: 1).

    Returns
    -------
    list
        List of tuples of files to merge.

    """
    list_small_segys = []
    for idx, file in enumerate(list_segys):
        fsize = os.path.getsize(file) / 1024  # KB
        if fsize < fsize_kB:  # KB -> ~50 traces
            list_small_segys.append(
                (
                    idx,  # index of file in `list_segys`
                    idx,  # filename
                    fsize,  # file size (KB)
                    idx - 1,  # filename (previously recorded)
                    idx + 1,  # filename (recorded afterwards)
                )
            )
    # create dataframe
    df_small_segys = pd.DataFrame(
        list_small_segys, columns=['list_idx', 'file', 'size_kb', 'file_pre', 'file_post']
    )
    xprint(
        f'Found < {len(df_small_segys)} > files smaller than {fsize_kB} KB',
        kind='info',
        verbosity=verbosity,
    )

    # find consecutively recorded files
    df_small_segys['diff_idx'] = np.append(0, np.diff(df_small_segys['list_idx']))
    
    # aggregate consecutive files
    s = df_small_segys['diff_idx'].ne(1).cumsum()
    grouped = df_small_segys.groupby(s).agg(
        {
            'list_idx': 'first',
            'file': lambda x: tuple(x),  # ','.join(x),
            'size_kb': 'first',
            'file_pre': lambda x: tuple(x),
            'file_post': lambda x: tuple(x),
        }
    )
    xprint(f'Remaining < {len(grouped)} > files after groupby', kind='info', verbosity=verbosity)

    def _to_merge(row):
        if isinstance(row[1][0], str):
            diff_pre = int(row[1][0].split('_')[0]) - int(row[3][0].split('_')[0])
            diff_post = int(row[4][0].split('_')[0]) - int(row[1][0].split('_')[0])
        elif isinstance(row[1][0], int):
            diff_pre = row[1][0] - row[3][0]
            diff_post = row[4][0] - row[1][0]

        if diff_pre < diff_post:
            return tuple(sorted(set(row[3] + row[1])))
        else:
            return tuple(sorted(set(row[1] + row[4])))

    # create tuple of files to merge
    grouped['to_merge'] = grouped.apply(_to_merge, axis=1)
    indices2merge = grouped['to_merge'].to_list()

    # get file paths from general list by calculated indices
    files2merge = [list_segys[tup[0] : tup[-1] + 1] for tup in indices2merge]
    xprint(f'Prepared < {len(files2merge)} > merged files', kind='info', verbosity=verbosity)

    return files2merge


def wrapper_merge_segys(file_list, txt_suffix: str = 'merge', verbosity=1):
    """
    Merge SEG-Y files provided as list of input paths.
    Potential gaps will be interpolated (linear) and duplicates removed while
    keeping the last (aka newest) occurrence.

    Parameters
    ----------
    file_list : tuple,list
        Iterable of files to be merged.
    txt_suffix : str, optional
        Filename suffix for merged file (default: 'merge')
    verbosity : int, optional
        Verbosity constant for stdout printing (default: 1).

    """
    first_file = file_list[0]
    basepath, filename = os.path.split(first_file)
    basename, suffix = os.path.splitext(filename)
    out_name = f'{basename}_{txt_suffix}{suffix}'
    out_file = os.path.join(basepath, out_name)

    trace_headers_list = []

    specs = []
    header_bin = []
    swdep_list = []
    data_list = []

    trace_header_to_check = {
        'tracl': segyio.TraceField.TRACE_SEQUENCE_LINE,
        'tracr': segyio.TraceField.TRACE_SEQUENCE_FILE,
        'fldr': segyio.TraceField.FieldRecord,
        'ns': segyio.TraceField.TRACE_SAMPLE_COUNT,
        'nt': segyio.TraceField.TRACE_SAMPLE_INTERVAL,
    }
    trace_header_dict = dict(
        zip(
            list(trace_header_to_check.keys()),
            [np.array([], dtype='int')] * len(trace_header_to_check),
        )
    )

    trace_header_datetime = {
        'year': segyio.TraceField.YearDataRecorded,
        'day': segyio.TraceField.DayOfYear,
        'hour': segyio.TraceField.HourOfDay,
        'minute': segyio.TraceField.MinuteOfHour,
        'sec': segyio.TraceField.SecondOfMinute,
    }
    trace_header_datetime_dict = dict(
        zip(list(trace_header_datetime.keys()), [np.array([])] * len(trace_header_datetime))
    )

    trace_header_coords = {
        'sx': segyio.TraceField.SourceX,
        'sy': segyio.TraceField.SourceY,
        'gx': segyio.TraceField.GroupX,
        'gy': segyio.TraceField.GroupY,
        'cdpx': segyio.TraceField.CDP_X,
        'cdpy': segyio.TraceField.CDP_Y,
    }
    trace_header_coords_dict = dict(
        zip(list(trace_header_coords.keys()), [np.array([])] * len(trace_header_coords))
    )

    list_ntraces = []

    for f in file_list:
        xprint(f'Processing file < {os.path.split(f)[-1]} >', kind='info', verbosity=verbosity)
        with segyio.open(f, 'r', strict=False, ignore_geometry=True) as file:
            list_ntraces.append(file.tracecount)

            specs.append(segyio.tools.metadata(file))
            trace_headers_list.append(parse_trace_headers(file))

            # load binary header
            header_bin.append(file.bin)

            # check binary header elements
            binary_equal = all([header_bin[0] == b for b in header_bin])
            if not binary_equal:
                raise IOError(
                    'Specified SEG-Y files have different binary headers. No easy merging possible, please check your data!'
                )

            # load trace header values used for gap checking
            xprint('Check trace headers', kind='debug', verbosity=verbosity)
            for key, field in trace_header_to_check.items():
                # print(key, field)
                # get already stored values (initially empty array)
                values = trace_header_dict.get(key)
                # combine previous and new values
                tr_header_attr = np.sort(
                    np.concatenate((values, file.attributes(field)[:]), axis=None)
                )
                # update dictionary with combined values
                trace_header_dict[key] = tr_header_attr

            # load datetimes from trace headers
            xprint('Load timestamps from trace headers', kind='debug', verbosity=verbosity)
            for key, field in trace_header_datetime.items():
                # get already stored values (initially empty array)
                datetime_values = trace_header_datetime_dict.get(key)
                # combine previous and new values
                tr_header_datetime = np.concatenate(
                    (datetime_values, file.attributes(field)[:]), axis=None
                )
                # update dictionary with combined values
                trace_header_datetime_dict[key] = tr_header_datetime

            # load coordinates from trace headers
            xprint('Load coordinates from headers', kind='debug', verbosity=verbosity)
            for key, field in trace_header_coords.items():
                # get already stored values (initially empty array)
                coords = trace_header_coords_dict.get(key)
                # combine previous and new values
                tr_header_coords = np.concatenate((coords, file.attributes(field)[:]), axis=None)
                # update dictionary with combined values
                trace_header_coords_dict[key] = tr_header_coords

            # SourceWaterDepth
            xprint('Load SourceWaterDepth from headers', kind='debug', verbosity=verbosity)
            swdep_list.extend(file.attributes(segyio.TraceField.SourceWaterDepth)[:])

            # trace data (as 2D np.ndarray)
            xprint('Load seismic section', kind='debug', verbosity=verbosity)
            data_list.append(file.trace.raw[:])

    xprint('Merge trace headers', kind='debug', verbosity=verbosity)
    # concat trace headers
    trace_headers = pd.concat(trace_headers_list, ignore_index=True)

    # create index from TRACE_SEQUENCE_LINE and get not existing traces (aka gaps)
    trace_headers.set_index(trace_headers['TRACE_SEQUENCE_LINE'], inplace=True)

    # drop duplicate traces from different files (only *exact* matches!)
    # trace_headers.drop_duplicates(keep='last', inplace=True) # keep "newer" record
    mask_overlapping_duplicates = trace_headers.duplicated(keep='last')

    # drop duplicate traces from same file (!)
    col_subset = list(trace_headers.columns)
    col_subset.remove('TRACE_SEQUENCE_FILE')
    # trace_headers.drop_duplicates(subset=col_subset, keep='first', inplace=True)
    mask_internal_duplicates = trace_headers.duplicated(subset=col_subset, keep='first')
    mask = (mask_overlapping_duplicates + mask_internal_duplicates).astype('bool')
    # select non-duplicates
    trace_headers = trace_headers[~mask]
    nduplicates = np.count_nonzero(mask)
    if nduplicates > 0:
        xprint(f'Removed < {nduplicates} > duplicates', kind='debug', verbosity=verbosity)

    # create gap records
    trace_headers = trace_headers.reindex(
        pd.RangeIndex(
            trace_headers.iloc[0]['TRACE_SEQUENCE_LINE'],
            trace_headers.iloc[-1]['TRACE_SEQUENCE_LINE'] + 1,
        )
    )

    xprint('Merge seismic data', kind='debug', verbosity=verbosity)
    data = np.concatenate(data_list, axis=0)
    # remove duplicates
    data = data[~mask]

    # get gap indices
    idx_gaps = pd.isnull(trace_headers).any(1).to_numpy().nonzero()[0]

    # if gaps are present
    if len(idx_gaps) > 0:
        # interpolate gaps in merged trace header
        xprint('Interpolate gaps in merged trace header', kind='debug', verbosity=verbosity)
        trace_headers_interp = trace_headers.interpolate(method='linear').astype('int32')
        trace_headers_interp['TRACE_SEQUENCE_FILE'] = np.arange(
            1, trace_headers_interp.shape[0] + 1, 1
        )
        trace_headers_interp.rename(columns=segyio.tracefield.keys, inplace=True)

        xprint('Fill gaps with zero traces', kind='debug', verbosity=verbosity)
        idx_gaps_first = np.nonzero(np.diff(idx_gaps) > 1)[0] + 1
        if idx_gaps_first.size == 0:
            idx = [idx_gaps[0]]
            ntr = [idx_gaps.size]
        else:
            idx = idx_gaps[np.insert(idx_gaps_first, 0, 0)]
            ntr = [a.size for a in np.split(idx_gaps, idx_gaps_first)]

        for i, n in zip(idx, ntr):
            dummy_traces = np.zeros((n, data.shape[1]), dtype=data.dtype)
            data = np.insert(data, i, dummy_traces, axis=0)
    else:
        trace_headers_interp = trace_headers
        trace_headers_interp['TRACE_SEQUENCE_FILE'] = np.arange(
            1, trace_headers_interp.shape[0] + 1, 1
        )
        trace_headers_interp = trace_headers_interp.rename(columns=segyio.tracefield.keys)

    # init output SEG-Y
    spec = specs[0]
    spec.samples = specs[0].samples  # get sample TWT from first file!
    spec.tracecount = data.shape[0]  # get number of trace from combined array

    # save merged SEG-Y
    xprint('Write merged SEG-Y to disk', kind='debug', verbosity=verbosity)
    with segyio.open(first_file, 'r', ignore_geometry=True) as src:
        with segyio.create(out_file, spec) as dst:
            dst.text[0] = src.text[0]
            dst.bin = src.bin
            for i, trheader in zip(range(0, spec.tracecount + 1), dst.header):
                trheader.update(trace_headers_interp.iloc[i].to_dict())
            dst.trace = np.ascontiguousarray(data, dtype=data.dtype)

    # update textual header
    text = get_textual_header(out_file)
    ## add info about new CRS
    info = ','.join([os.path.split(f)[-1].split('.')[0] for f in file_list])
    text_updated = add_processing_info_header(text, info, prefix='MERGED')
    write_textual_header(out_file, text_updated)

    # save auxiliary file
    out_file_aux = out_file.split('.')[0] + '.parts'
    with open(out_file_aux, 'w', newline='\n') as fout:
        fout.write(f'The merged SEG-Y file < {out_name} > contains the following files:\n')
        for f, ntr in zip(file_list, list_ntraces):
            fout.write(f'    - {os.path.split(f)[-1]}    {ntr:>6d} trace(s)\n')
        string = f'Trace duplicates (different files):    {np.count_nonzero(mask_overlapping_duplicates):>3d}\n'
        string += f'Trace duplicates (within single file): {np.count_nonzero(mask_internal_duplicates):>3d}\n'
        fout.write(string)


def main(argv=sys.argv):  # noqa
    """Merge small SEG-Y files with others."""
    TIMESTAMP = datetime.datetime.now().isoformat(timespec='seconds').replace(':', '')
    SCRIPT = os.path.basename(__file__).split(".")[0]

    parser = define_input_args()
    args = parser.parse_args(argv[1:])  # exclude filename parameter at position 0

    verbosity = args.verbose

    # check input file(s)
    in_path = args.input_dir
    basepath, filename = os.path.split(in_path)
    basename, suffix = os.path.splitext(filename)
    if suffix == '':
        basepath = in_path
        basename, suffix = None, None  # noqa

    # (1) input directory (multiple files)
    if os.path.isdir(in_path):
        pattern = '*' + f'{args.filename_suffix}' + f'.{args.suffix}'
        file_list = glob.glob(os.path.join(in_path, pattern))

    # (2) file input is datalist (multiple files)
    elif os.path.isfile(in_path) and (suffix == '.txt'):
        with open(in_path, 'r') as datalist:
            file_list = datalist.readlines()
            file_list = [
                os.path.join(basepath, line.rstrip())
                if os.path.split(line.rstrip()) not in ['', '.']
                else line.rstrip()
                for line in file_list
            ]

    if len(file_list) > 0:
        # redirect stdout to logfile
        with open(
            os.path.join(basepath, f'{TIMESTAMP}_{SCRIPT}.log'),
            'w',
            newline='\n',
        ) as f:
            with redirect_stdout(f):
                # get list of tuples of files to merge
                files2merge = get_files_to_merge(file_list, verbosity=verbosity)

                xprint(
                    f'Processing total of < {len(files2merge)} > files',
                    kind='info',
                    verbosity=verbosity,
                )
                # merge SEG-Y(s)
                for i, file_list in enumerate(files2merge):
                    xprint(f'Merging {i+1}th set of files', kind='info', verbosity=verbosity)
                    wrapper_merge_segys(file_list, verbosity=verbosity)
    else:
        sys.exit('No input files to process.')


# %% MAIN
if __name__ == '__main__':
    
    main()
