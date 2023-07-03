"""
Compensate static on seismic profile(s).
Using either "SourceWaterDepth" (mode: `swdep`) or first positive amplitude peak of seafloor reflection (mode: `amp`).

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
from scipy.signal import savgol_filter

from pseudo_3D_interpolation.functions.utils import (
    depth2samples,
    samples2twt,
    twt2samples,
    xprint,
    slice_valid_data,
    clean_log_file,
)
from pseudo_3D_interpolation.functions.header import (
    get_textual_header,
    add_processing_info_header,
    write_textual_header,
)
from pseudo_3D_interpolation.functions.filter import (
    filter_interp_1d,
    detect_seafloor_reflection,
    mad_filter,
    polynominal_filter,
)

#%% FUNCTIONS
# fmt: off
def define_input_args():  # noqa
    parser = argparse.ArgumentParser(
        description='Compensate static on seismic profile(s) using either "SourceWaterDepth" (swdep) '
        + 'or first positive amplitude peak of seafloor reflection (amp).')
    parser.add_argument('input_path', type=str, help='Input file or directory.')
    parser.add_argument('--output_dir', '-o', type=str,
                        help='Output directory for edited SEG-Y file(s)')
    parser.add_argument('--suffix', '-s', type=str,
                        help='File suffix. Only used when "input_path" is a directory.')
    parser.add_argument('--inplace', '-i', action='store_true',
                        help='Edit SEG-Y file(s) inplace')
    parser.add_argument('--filename_suffix', '-fns', type=str,
                        help='Filename suffix for guided selection (e.g. "env" or "despk"). Only used when "input_path" is a directory.')
    parser.add_argument('--txt_suffix', type=str, default='static',
                        help='Additional text to append to output filename')
    
    parser.add_argument('--use_delay', action='store_true',
                        help='Use delay recording time to split input data before despiking (e.g. for TOPAS, Parasound)')
    parser.add_argument('--byte_delay', type=int, default=109,
                        help='Byte position of input delay times in SEG-Y file(s). Default: 109')
    
    parser.add_argument('--mode', '-m', type=str, default='amp', choices=['amp', 'swdep'],
                        help='Use either peak seafloor amplitude [amp] or stored SourceWaterDepth [swdep] (if available).')
    parser.add_argument('--win_samples', type=int, default=30,
                        help='Length of vertical padding (in samples) for seafloor detection.')
    parser.add_argument('--nsta', type=int, help='Length of short time average window (in samples).')
    parser.add_argument('--nlta', type=int, help='Length of long time average window (in samples).')
    parser.add_argument('--win_median', type=int, default=11, help='Length of median filter window (in traces).')
    parser.add_argument('--n_amp_samples', type=int, default=5,
                        help='Selecting `n_amp_samples` amplitude samples within seafloor detection window.')
    
    parser.add_argument('--win_mad', type=int,
                        help='Moving window length for MAD filter (traces [#])')
    parser.add_argument('--win_sg', type=int, default=7,
                        help='Moving window length for Savitzky-Golay filter (traces [#])')
    parser.add_argument('--limit_shift', nargs='?', type=int, default=12, const=12,
                        help='Limit maximum vertical shift of individual traes (in samples)')
    parser.add_argument('--limit_depressions', nargs='+', type=int, default=[10, 10, 5],
                        help='Limit maximum vertical shift in area of seafloor depressions '
                        + 'using a transition zone [pad, max_edges, max_center] (as integer)')
    parser.add_argument('--write_seafloor2trace', action='store_true',
                        help='If mode is "amp": write TWT of peak seafloor amplitude to SEG-Y trace header')
    parser.add_argument('--write_aux', action='store_true',
                        help='Write trace information and computed static to auxiliary file (*.sta)')
    parser.add_argument('--verbose', '-V', type=int, nargs='?', default=0, choices=[0, 1, 2],
                        help='Level of output verbosity (default: 0)')
    return parser
# fmt: on

# -----------------------------------------------------------------------------
#                               STATIC
# -----------------------------------------------------------------------------
def get_static(
    data,
    kind='diff',
    interp_kind='cubic',
    win_mad=None,
    win_sg=7,
    limit_perc=99,
    limit_samples=10,
    limit_by_MAD=False,
    limit_depressions=False,
):
    """
    Compute static (as deviation from global reference level) for each trace.
    Input data array is first filtered and interpolated to remove detecion outliers.
    Depending on selected option, the static will be calculated as:
        
    - `diff`: difference between filtered input data (removed outlier) and
              Savitzky-Golay lowpass filter result
    - `deriv`: output of Savitzky-Golay hihgpass filter (2nd derivative)

    Parameters
    ----------
    data : np.ndarray
        Input data array (1D).
    kind : str, optional
        Static calculation option (default: 'diff').
    interp_kind : str, optional
        Interpolation type used for scipy.interpolate.interp1d (default: 'cubic').
    win_mad : int, optional
        Windwow length used for initial data filtering (in traces).
        If None, it will about 5% of the input data length (at least 7 traces!).
    win_sg : int, optional
        Window length for Savitzky-Golay filter (in traces). It should be slightly larger
        than the period of the observed static amplitudes (default: 7).
    limit_perc: float, optional
        Clip calculated static shift (in samples) using np.percentile with given value (default: 99).
    limit_samples: float, optional
        Clip calculated static shift (in samples) using user-specified number of samples (default: 10).
    limit_by_MAD : float, optional,
        Clip calculated static shift (in samples) using median absolute deviation (MAD)
        multiplied by `limit_by_MAD`. If true, `limit_by_MAD` defaults to 3 (-> 3-sigma rule of thumb).
    limit_depressions : tuple(float, float, float), optional
        Account for seafloor depressions that represent significant topographic changes
        over short distances by limiting the maximum vertical shift using a linear function
        from depression center to flanks, e.g. limits: 10, 8, 6, 4, 6, 8, 10 (max samples shift).
        
        ```python
        tuple(
            lenght of transition zone to each side [default: 10],
            max shift at outer edge of transition zone [default: 10],
            max shift at depression center [default: 5],
            )
        ```

    Returns
    -------
    static : np.ndarray
        Computed static as deviation from reference level.

    """
    if data.ndim != 1:
        raise ValueError(f'Input array must have only one dimension not {data.ndim}.')
    if kind not in ['diff', 'deriv']:
        raise ValueError(f'Kind < {kind} > is not supported')

    data = np.asarray(data)

    if win_mad is None:
        win_mad = int(data.size * 0.05)
    win_mad = win_mad + 1 if win_mad % 2 == 0 else win_mad  # must be odd
    win_mad = 7 if win_mad < 7 else win_mad  # at least 7 traces

    # (1) outlier detection & removal
    data_mad_r = filter_interp_1d(
        data, method='r_doubleMAD', kind=interp_kind, threshold=3, win=win_mad
    )

    if kind == 'diff':
        # (2) apply Savitzky-Golay filter (lowpass)
        data_lowpass = savgol_filter(data_mad_r, window_length=win_sg, polyorder=1, deriv=0)
        static = (
            data_lowpass - data_mad_r
        )  # fit sign convention (<0: add at top of trace, >0: add at bottom of trace)

    elif kind == 'deriv':
        # (2) apply Savitzky-Golay filter (highpass)
        order = win_sg - 2
        static = savgol_filter(data_mad_r, window_length=win_sg, polyorder=order, deriv=2)

    if kind == 'diff' and limit_depressions:
        # apply polynominal filter to lowpass-filtered data
        detrend = polynominal_filter(data_lowpass, order=11) * -1
        # detect "outlier" (depressions)
        idx_detrend = mad_filter(detrend, threshold=3, mad_mode='double')
        if idx_detrend.size == 0:
            return static
        # get indices < 0 (only depressions)
        pockmark_idx = np.nonzero(detrend[idx_detrend] < 0)
        idx_detrend_filt = idx_detrend[pockmark_idx]
        if idx_detrend_filt.size == 0:
            return static
        # split indices array into individual arrays (per pockmark)
        idx_detrend_filt_splits = np.split(
            idx_detrend_filt, np.where(np.diff(idx_detrend_filt) > 1)[0] + 1
        )
        # remove detection where ntraces < 3
        idx_detrend_filt_splits = [a for a in idx_detrend_filt_splits if a.size >= 3]
        if len(idx_detrend_filt_splits) == 0:
            return static

        # define indices and limits to clip static shift sample array
        npad, limit_outer, limit_center = limit_depressions
        # npad = 10         # number of traces for transition zone
        # limit_outer = 10  # max. shift at boundary of transition zone
        # limit_center = 5  # max. shift for pockmark traces

        # get padded indices arrays per depression
        pockmark_limits_idx = np.concatenate(
            [np.arange(p[0] - npad, p[-1] + npad + 1, dtype='int') for p in idx_detrend_filt_splits]
        )
        # get custom limits per depression
        pockmark_limits = np.concatenate(
            [
                np.concatenate(
                    (
                        np.linspace(limit_outer, limit_center + 1, npad),
                        np.full_like(a, limit_center),
                        np.linspace(limit_center + 1, limit_outer, npad),
                    )
                ).astype('int')
                for a in idx_detrend_filt_splits
            ]
        )
        assert pockmark_limits_idx.shape == pockmark_limits.shape
        # account for boundary conditions
        mask_valid = np.nonzero(
            (pockmark_limits_idx < detrend.size) & (pockmark_limits_idx >= 0)
        )[0]
        pockmark_limits_idx = pockmark_limits_idx[mask_valid]
        pockmark_limits = pockmark_limits[mask_valid]

        # clip static shift using custom limits for depressions
        static[pockmark_limits_idx] = np.where(
            np.abs(static[pockmark_limits_idx]) > pockmark_limits,
            pockmark_limits * np.sign(static[pockmark_limits_idx]),
            static[pockmark_limits_idx],
        )

    # if set: clip static values using given percentile
    if limit_perc is not None and limit_perc is not False:
        clip = np.percentile(np.abs(static), limit_perc)
        static = np.where(np.abs(static) > clip, clip * np.sign(static), static)

    # if set: clip static values using user-specified number of samples
    if isinstance(limit_samples, (float, int)):
        static = np.where(np.abs(static) > limit_samples, limit_samples * np.sign(static), static)

    # if set: clip static values using median absolute deviation (multiplied by factor)
    if limit_by_MAD is True or isinstance(limit_by_MAD, (int, float)):
        limit_by_MAD = limit_by_MAD if isinstance(limit_by_MAD, (int, float)) else 3
        threshold = int(np.ceil(np.median(np.abs(static)) * limit_by_MAD))
        static = np.where(np.abs(static) > threshold, threshold * np.sign(static), static)

    return static


def compensate_static(data, static, dt=None, units='ms', cnv_d2s=False, v=1500, verbosity=1):
    """
    Apply computed static offsets to seismic traces.

    Parameters
    ----------
    data : np.ndarray
        2D array of input seismic traces (samples x traces).
    static : np.ndarray
        1D array of static offsets (per trace).
    dt : float, optional
        Sampling interval in specified units (default: milliseconds).
    units : str, optional
        Time unit (for `dt`) (default: `s`).
    cnv_d2s : bool, optional
        Convert static offset provided as depth (in m) to samples (default: `False`).
    v : int, optional
        Sound velocity (m/s) for depth/time/samples conversion. Only used if `cnv_d2s` is True.

    Returns
    -------
    np.ndarray(s)
        Compensated seismic section (and # of samples if converted).

    """
    if dt is not None:
        if units == 's':
            pass
        elif units == 'ms':
            dt = dt / 1000
        elif units == 'ns':
            dt = dt / 1e-6

    if cnv_d2s:
        if dt is None:
            print('[ERROR]   `dt` is required when converting depth to samples')
            return None
        static_samples = depth2samples(static, dt=dt, v=v, units='s')
        static_samples = np.around(static_samples, 0).astype(np.int32)
    else:
        static_samples = np.around(static, 0).astype(np.int32)

    # create copy of original data
    data_static = data.T.copy()

    for i, col in enumerate(data_static):
        offset = static_samples[i]
        if offset < 0:
            col[:] = np.hstack((col[abs(offset) :], np.zeros(abs(offset))))
            xprint(
                f'trace #{i}:{offset:>5}   ->   up: {col.shape}', kind='debug', verbosity=verbosity
            )
        elif offset > 0:
            col[:] = np.hstack((np.zeros(abs(offset)), col[: -abs(offset)]))
            xprint(
                f'trace #{i}:{offset:>5}   ->   down: {col.shape}',
                kind='debug',
                verbosity=verbosity,
            )
        else:
            pass  # no static correction

    return data_static.T, static_samples


def wrapper_static_correction_segy(in_path, args):  # noqa
    """Apply static correction to single SEG-Y file."""
    basepath, filename = os.path.split(in_path)
    basename, suffix = os.path.splitext(filename)
    xprint(f'Processing file < {filename} >', kind='info', verbosity=args.verbose)
    
    default_txt_suffix = 'static'
    
    if args.txt_suffix is not None:
        out_name = f'{basename}_{args.txt_suffix}'
    else:
        out_name = f'{basename}_{default_txt_suffix}'
    
    if args.inplace is True:  # `inplace` parameter supersedes any `output_dir`
        xprint('Updating SEG-Y inplace', kind='warning', verbosity=args.verbose)
        path = in_path
        out_dir = basepath
    else:
        if args.output_dir is None:  # default behavior
            xprint('Creating copy of file in INPUT directory:\n', basepath, kind='info', verbosity=args.verbose)
            out_dir = basepath
        elif args.output_dir is not None and os.path.isdir(args.output_dir):
            xprint('Creating copy of file in OUTPUT directory:\n', args.output_dir, kind='info', verbosity=args.verbose)
            out_dir = args.output_dir
        else:
            raise FileNotFoundError(f'The output directory > {args.output_dir} < does not exist')
            
        out_path = os.path.join(out_dir, f'{out_name}{suffix}')
    
        # sanity check
        if os.path.isfile(out_path):
            xprint('Output file already exists and will be removed!', kind='warning', verbosity=args.verbose)
            os.remove(out_path)

        copy2(in_path, out_path)
        path = out_path

    with segyio.open(path, 'r+', strict=False, ignore_geometry=True) as src:
        n_traces = src.tracecount  # total number of traces
        dt = segyio.tools.dt(src) / 1000  # sample rate [ms]
        twt = src.samples  # two way travel time (TWTT) [ms]

        tracl = src.attributes(segyio.TraceField.TRACE_SEQUENCE_LINE)[:]
        tracr = src.attributes(segyio.TraceField.TRACE_SEQUENCE_FILE)[:]
        fldr = src.attributes(segyio.TraceField.FieldRecord)[:]

        # get DelayRecordingTimes from trace headers
        delrt = src.attributes(args.byte_delay)[:]  # segyio.TraceField.DelayRecordingTime

        swdep = src.attributes(segyio.TraceField.SourceWaterDepth)[:]
        scalel = src.attributes(segyio.TraceField.ElevationScalar)[:]
        if all(s > 0 for s in scalel):
            swdep = swdep * np.abs(scalel)
        elif all(s < 0 for s in scalel):
            swdep = swdep / np.abs(scalel)

        # eager version (completely read into memory)
        data_src = src.trace.raw[:].T

        # extract infos from binary header
        hns = src.bin[segyio.BinField.Samples]  # samples per trace
        nso = src.bin[segyio.BinField.SamplesOriginal]  # samples per trace (original)

        # (A) using SourceWaterDepth
        if args.mode == 'swdep' and (np.count_nonzero(swdep) == n_traces):
            # compute static offset using recorded SourceWaterDepth
            static_depth = get_static(
                swdep,
                kind='diff',
                interp_kind='cubic',
                win_mad=args.win_mad,
                win_sg=args.win_sg,
                limit_perc=False,
                limit_samples=args.limit_shift,
                limit_by_MAD=3,
                limit_depressions=args.limit_depressions,
            )

            # compute static corrected data (and return sample index array)
            data_corr, static_samples = compensate_static(
                data_src, static_depth, dt=dt, units='ms', cnv_d2s=True, verbosity=args.verbose
            )

        # (B) using peak seafloor amplitude
        elif args.mode == 'amp':
            # find indices where DelayRecordingTime changes
            delrt_idx = np.where(np.roll(delrt, 1) != delrt)[0]
            if delrt_idx.size != 0 and delrt_idx[0] == 0:
                delrt_idx = delrt_idx[1:]

            ## (B.1) padded SEG-Y --> extrace only non-zero part of traces
            if 'pad' in path or (hns != nso):
                data_src_sliced, idx_start_slice = slice_valid_data(data_src, nso if nso != 0 else hns)
                idx_amp = detect_seafloor_reflection(
                    data_src_sliced,
                    nsta=args.nsta,
                    nlta=args.nlta,
                    win=args.win_samples,
                    win_median=args.win_median,
                    n=args.n_amp_samples,
                )
                idx_amp += idx_start_slice
                twt_seafloor = twt[idx_amp]

            ## (B.2) unpadded SEG-Y
            else:
                ### (B.2.1) single DelayRecordingTimes
                idx_amp = detect_seafloor_reflection(
                    data_src,
                    nsta=args.nsta,
                    nlta=args.nlta,
                    win=args.win_samples,
                    win_median=args.win_median,
                    n=args.n_amp_samples,
                )
                _idx_amp = idx_amp.copy()

                ### (B.2.2) variable DelayRecordingTimes
                if args.use_delay and (len(delrt_idx) >= 1):
                    xprint(
                        'Account for variable DelayRecordingTimes (`delrt`)',
                        kind='info',
                        verbosity=args.verbose,
                    )
                    # get DelayRecordingTime offset as samples
                    delrt_min = delrt.min()
                    delrt_offset = (delrt - delrt_min).astype('int')
                    delrt_offset_samples = twt2samples(delrt_offset, dt=dt).astype('int')
                    # add offset to indices of peak seafloor amplitude
                    idx_amp += delrt_offset_samples

                if args.write_seafloor2trace:
                    _delrt_offset = (delrt - delrt[0]).astype('int')
                    _delrt_offset_samples = twt2samples(_delrt_offset, dt=dt).astype('int')
                    _idx_amp4twt = _idx_amp + _delrt_offset_samples
                    twt_seafloor = twt[_idx_amp4twt]

            # compute static offset using peak seafloor amplitude sample indices
            static_samples = get_static(
                idx_amp,
                kind='diff',
                interp_kind='cubic',
                win_mad=args.win_mad,
                win_sg=args.win_sg,
                limit_perc=False,
                limit_samples=args.limit_shift,
                limit_by_MAD=3,
                limit_depressions=args.limit_depressions,
            )

            # compute static corrected data
            data_corr, static_samples = compensate_static(
                data_src, static_samples, dt=dt, units='ms'
            )

        # === OUTPUT ===
        static_ms = samples2twt(static_samples, dt=dt)
        if args.write_aux:
            header_names = ['tracl', 'tracr', 'fldr', 'static_samples', 'static_ms']
            if args.mode == 'swdep':
                header_names += ['swdep_m']
            elif args.mode == 'amp':
                header_names += ['seafloor_ms']

            with open(
                os.path.join(out_dir, f'{out_name}.sta'), mode='w', newline='\n'
            ) as sta:
                sta.write(','.join(header_names) + '\n')
                for i in range(len(tracr)):
                    line = (
                        f'{tracl[i]},{tracr[i]},{fldr[i]},'
                        + f'{static_samples[i]:d},{static_ms[i]:.3f},'
                    )
                    if args.mode == 'swdep':
                        line += f'{swdep[i]:.2f}\n'
                    elif args.mode == 'amp':
                        line += f'{twt_seafloor[i]:.2f}\n'
                    sta.write(line)

        # update textual header
        field_static = segyio.TraceField.TotalStaticApplied
        field_scalar = segyio.TraceField.UnassignedInt1
        field_amp = segyio.TraceField.UnassignedInt2

        # global text, text_updated, text_out
        text = get_textual_header(path)
        info = (
            f'STATIC CORRECTION:{args.mode} (byte:{field_static}) with SCALAR (byte:{field_scalar})'
        )
        text_updated = add_processing_info_header(text, info, prefix='_TODAY_')
        if args.mode == 'amp' and args.write_seafloor2trace:
            info = f'-> SEAFLOOR (byte:{field_amp}) with SCALAR (byte:{field_scalar})'
            text_updated = add_processing_info_header(text_updated, info, prefix='_TODAY_')
        write_textual_header(path, text_updated)

        # update trace header info
        static_scalar = 1000
        static_ms = (static_ms * static_scalar).astype('int32')  # ms to ns
        if args.mode == 'amp' and args.write_seafloor2trace:
            twt_seafloor = (twt_seafloor * static_scalar).astype('int32')  # ms to ns

        for i, h in enumerate(src.header[:]):
            h.update(
                {
                    field_static: static_ms[i],  # byte 103: TotalStaticApplied (ms)
                    field_scalar: -static_scalar,
                }
            )  # byte 233: custom scalar for TotalStaticApplied
            if args.mode == 'amp' and args.write_seafloor2trace:
                h.update(
                    {field_amp: twt_seafloor[i]}
                )  # byte 237: save detected seafloor amplitude TWT (ms)

        # write corrected traces to file
        src.trace = np.ascontiguousarray(data_corr.T, dtype=data_src.dtype)

    try:
        src.close()
    except IOError:
        xprint('SEG-Y file was already closed.', kind='warning', verbosity=args.verbose)

    return path, data_src, data_corr  # , static_samples, dt, tracl, tracr, delrt


def main(argv=sys.argv):  # noqa
    TIMESTAMP = datetime.datetime.now().isoformat(timespec='seconds').replace(':', '')
    SCRIPT = os.path.splitext(os.path.basename(__file__))[0]

    parser = define_input_args()
    args = parser.parse_args(argv[1:])  # exclude filename parameter at position 0
    xprint(args, kind='debug', verbosity=args.verbose)
    
    # sanity checks
    if (args.inplace is False and args.output_dir is None) or (
        args.inplace is True and args.output_dir is not None
    ):
        sys.exit('[ERROR]    Either `output_dir` OR `args.inplace` must be specified.')

    # check input file(s)
    in_path = args.input_path
    basepath, filename = os.path.split(in_path)
    basename, suffix = os.path.splitext(filename)
    if suffix == '':
        basepath = in_path
        basename, suffix = None, None  # noqa

    # (1) single input file
    if os.path.isfile(in_path) and suffix != '.txt':
        # comensate static of input file
        path, data, data_corr = wrapper_static_correction_segy(in_path, args)
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
                k = 0  # FIXME
                for file_path in tqdm(
                    file_list,
                    desc='Correct static',
                    ncols=80,
                    total=len(file_list),
                    unit_scale=True,
                    unit=' files',
                ):
                    # compensate static of input files
                    try:  # FIXME
                        path, data, data_corr = wrapper_static_correction_segy(file_path, args)
                    except Exception as e:  # FIXME
                        xprint(f'Failed: {e}', kind='error', verbosity=args.verbose)  # FIXME
                        k += 1
        clean_log_file(logfile)
        xprint(f'>{k}< out of >{len(file_list)}< files failed!', kind='info', verbosity=args.verbose)
    else:
        sys.exit('No input files to process. Exit process.')


#%% MAIN

if __name__ == '__main__':

    main()
