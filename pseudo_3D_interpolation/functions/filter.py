"""Utility functions to filter (multidimensional) data."""

from functools import partial

import numpy as np
import scipy.signal as signal
import scipy.interpolate as interp

from .utils import pad_array

#%%
def moving_window(a, window_length: int, step_size: int = 1):
    """
    Create moving windows of given window length over input array (as view).

    Parameters
    ----------
    a : np.ndarray
        1D input array.
    window_length : int
        Length of moving window.
    step_size : int
        Step size of moving window (default: 1).

    Returns
    -------
    view : np.ndarray
        View of array according to `window_length` and `step_size`.

    References
    ----------
    [^1]: [https://stackoverflow.com/a/6811241](https://stackoverflow.com/a/6811241)
    [^2]: [https://rigtorp.se/2011/01/01/rolling-statistics-numpy.html](https://rigtorp.se/2011/01/01/rolling-statistics-numpy.html)
    [^3]: [https://gist.github.com/codehacken/708f19ae746784cef6e68b037af65788](https://gist.github.com/codehacken/708f19ae746784cef6e68b037af65788)

    """
    shape = a.shape[:-1] + (a.shape[-1] - window_length + 1 - step_size + 1, window_length)
    strides = a.strides + (a.strides[-1] * step_size,)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def moving_average(a, win: int = 3):
    """
    Apply simple non-weighted moving average.

    Parameters
    ----------
    a : np.ndarray
        1D input data.
    win : int, optional
        Number of data points within moving window (default: `3`).

    Returns
    -------
    np.ndarray
        Moving average of input data.
    
    Reference
    ---------
    [^1]: [https://stackoverflow.com/a/42867926](https://stackoverflow.com/a/42867926)

    """
    ret = np.cumsum(a, dtype=float)
    ret[win:] = ret[win:] - ret[:-win]
    return ret[win - 1 :] / win


def moving_average_convolve(x, win: int = 3):
    """
    Compute moving average of array with given window length.

    Parameters
    ----------
    x : np.array
        1D input data.
    win : int, optional
        Number of data points within moving window (default: `3`).

    Returns
    -------
    np.array
        Moving average of input data.

    """
    return np.convolve(x, np.ones(win), mode='valid') / win


def moving_median(a, win: int = 3, padded=False):
    """
    Apply moving median of given window size.
    Optional padding of input array using half the window size to avoid edge effects.

    Parameters
    ----------
    a : np.ndarray
        Input data (1D).
    win : int, optional
        Number of data points within moving window (default: `3`).
    padded : bool, optional
        Pad start and end of array (default: `False`).

    Returns
    -------
    np.ndarray
        Moving median of input data.

    """
    if padded:
        half_win = (win - 1) // 2
        a = pad_array(a, half_win)

    windows = moving_window(a, window_length=win)

    return np.median(windows, axis=-1)


def moving_window_2D(a, w, dx=1, dy=1, writeable=False):
    """
    Create an array of moving windows (as view) into the input array using given step sizes in both dimensions.

    Parameters
    ----------
    a : np.ndarray
        2D input array.
    w : tuple
        Moving window shape.
    dx : int, optional
        Horizontal step size (columns, e.g. traces) (default: 1).
    dy : int, optional
        vertical step size (rows, e.g. time samples) (default: 1).
    writeable : bool, optional
        Set if view should be writeable (default: False). **Use with care!**

    Returns
    -------
    view : numpy.ndarray
        4D array representing view of input array.
    
    References
    ----------
    [^1]: [https://colab.research.google.com/drive/1Zru_-zzbtylgitbwxbi0eDBNhwr8qYl6#scrollTo=tXDRG-5-2jBV](https://colab.research.google.com/drive/1Zru_-zzbtylgitbwxbi0eDBNhwr8qYl6#scrollTo=tXDRG-5-2jBV)

    """
    shape = (
        a.shape[:-2] + ((a.shape[-2] - w[-2]) // dy + 1,) + ((a.shape[-1] - w[-1]) // dx + 1,) + w
    )
    strides = a.strides[:-2] + (a.strides[-2] * dy,) + (a.strides[-1] * dx,) + a.strides[-2:]

    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides, writeable=writeable)


# *****************************************************************************
#                       MEDIAN ABSOLUTE DEVIATION (MAD)
# *****************************************************************************
def median_abs_deviation(x, axis=-1):
    """
    Return the median absolute deviation (MAD) from given input array.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    mad : np.ndarray
        Median absolute deviation (MAD) of input array.

    """
    if x.ndim == 1:
        mad = np.median(np.abs(x - np.median(x, axis=axis)))
    elif x.ndim == 2:
        mad = np.median(np.abs(x.T - np.median(x, axis=axis)).T, axis=axis)
    else:
        raise ValueError(f'Input arrays with < {x.ndim} > dimensions are not supported!')
    return mad


def median_abs_deviation_double(x, axis=-1):
    """
    Return the median absolute deviation (MAD) for unsymmetric distributions.
    Computes the deviation from median for both sides (left & right).

    Parameters
    ----------
    x : np.ndarray
        Input array.
    axis : TYPE, optional
        Axis to compute median on (default: -1).

    Returns
    -------
    mad : np.ndarray
        Median absolute deviation (MAD) of input array.
    
    References
    ----------
    [^1]: [https://eurekastatistics.com/using-the-median-absolute-deviation-to-find-outliers/](https://eurekastatistics.com/using-the-median-absolute-deviation-to-find-outliers/)

    """
    if x.ndim == 1:
        med = np.median(x, axis=axis)
        diff = np.abs(x - med)
        mad_left = np.median(diff[x <= med])
        mad_right = np.median(diff[x >= med])
        if mad_left == 0 or mad_right == 0:
            raise ValueError('one side of median absolute deviation is zero')
        mad = np.repeat(mad_left, len(x))
        mad[x > med] = mad_right
    elif x.ndim == 2:
        # compute median for each window
        med = np.median(x, axis=axis)
        # difference from median (per window)
        diff = np.abs(x - med[:, None])

        # define column of reference value (in window)
        idx_col = x.shape[-1] // 2
        # left side MAD
        mad_left = np.median(diff[(x <= med[:, None])[:, idx_col]], axis=axis)
        mad_left[mad_left == 0] = 1
        # right side MAD
        mad_right = np.median(diff[(x >= med[:, None])[:, idx_col]], axis=axis)
        mad_right[mad_right == 0] = 1

        # create and fill output array
        mad = np.ones((x.shape[0],), dtype=x.dtype)
        mad[(x <= med[:, None])[:, idx_col]] = mad_left
        mad[(x >= med[:, None])[:, idx_col]] = mad_right
    else:
        raise ValueError(f'Input arrays with < {x.ndim} > dimensions are not supported!')

    return mad.astype(x.dtype)


# *****************************************************************************
#                                   FILTER
# *****************************************************************************
def smooth(data, window_len=11, window='hanning'):
    """
    Smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    Parameters
    ----------
    data : np.ndarray
        1D input data array.
    window_len : int, optional
        Input window length, should be odd integer (default: 11).
    window : str, optional
        Tpye of smoothing window function (default: 'hanning').

    Returns
    -------
    out :
        smoothed input data
    
    References
    ----------
    [^1]: [https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html](https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html)

    """
    if data.ndim != 1:
        raise ValueError('smooth only accepts 1 dimension arrays.')

    if data.size < window_len:
        raise ValueError(
            f'Input data should be longer ({data.size}) than the window length ({window_len}).'
        )

    if window_len < 3:
        return data
    window_len += 1 if window_len % 2 == 0 else 0

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
        
    left, right = window_len // 2, window_len // 2
    # print(f'left: {left}, right {right} ')

    # linear extrapolation of least-squares solution
    m_start, c_start = np.linalg.lstsq(np.vstack([np.arange(left), np.ones(left)]).T, data[:left], rcond=None)[0]
    m_end, c_end = np.linalg.lstsq(np.vstack([np.arange(right), np.ones(right)]).T, data[-right:], rcond=None)[0]
    s = np.r_[
        np.arange(-left, 0, 1) * m_start + c_start,
        data,
        np.arange(right, right * 2) * m_end + c_end
    ]
    # print('padded signal: ', len(s))

    if window == 'flat':
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    out = np.convolve(s, w / w.sum(), mode='valid')

    return out


def zscore_filter(data, axis=-1):
    """Z-score filter for outlier detection. Return array of outlier indices."""
    z_score = (data - np.mean(data, axis=axis)) / np.std(data, axis=axis)
    return np.nonzero(np.logical_or(z_score < -1, z_score > 1))[0]


def moving_zscore_filter(data, win, axis=-1):  # noqa
    """
    Return array of outlier indices using moving z-score filter for outlier detection of length `win`.
    
    """
    mean = smooth(data, window_len=win, window='hanning')
    z_score = (data - mean) / np.std(data, axis=axis)
    return np.nonzero(np.logical_or(z_score < -1, z_score > 1))[0]


def iqr_filter(a, axis=-1):
    """Inter-quartile range (IQR) filter for outlier detection. Return array of outlier indices."""
    quantiles = np.quantile(a, [0.25, 0.75], axis=axis, keepdims=True)
    q1 = quantiles[0]
    q3 = quantiles[1]
    iqr = q3 - q1
    iqr_upper = q3 + 1.5 * iqr
    iqr_lower = q1 - 1.5 * iqr

    return np.nonzero(np.logical_or(a < iqr_lower, a > iqr_upper))[0]


def mad_filter(a, threshold=3, axis=-1, mad_mode='single'):
    """Median Absolute Deviation (MAD) filter. Return array of outlier indices."""
    med = np.median(a, axis=axis)
    if mad_mode == 'single':
        mad = median_abs_deviation(a)
    elif mad_mode == 'double':
        mad = median_abs_deviation_double(a)
    return np.nonzero((np.abs(a - med) / mad) > threshold)[0]


def moving_mad_filter(a, win, threshold=3, axis=-1, mad_mode='single'):  # noqa
    """Moving Median Absolute Deviation (MAD) filter of length `win`. Return array of outlier indices."""
    if (type(win) != int) or (win % 2 != 1):
        raise ValueError('window length must be odd integer')

    win_half = (win - 1) // 2

    # pad start and end of input array
    a_pad = pad_array(a, win_half)

    # create moving windows (as views)
    windows = moving_window(a_pad, window_length=win)

    # compute moving median
    moving_med = np.median(windows, axis=-1)

    # compute moving MAD
    if mad_mode == 'single':
        moving_mad = median_abs_deviation(windows)
    elif mad_mode == 'double':
        moving_mad = median_abs_deviation_double(windows)

    # account for case MAD == 0 (prone to false outlier detection)
    moving_mad[moving_mad == 0] = 1

    return np.nonzero((np.abs(a - moving_med) / moving_mad) > threshold)[0]


def polynominal_filter(data, order=3, kind='high'):
    """
    Apply polynominal filter to input data.

    Parameters
    ----------
    data : np.ndarray
        Input data.
    order : int, optional
        Filter order (default: `3`).
    kind : str, optional
        Filter kind (default: `high`).

    Returns
    -------
    data : np.ndarray
        Filtered input data.

    """
    data = data.copy().astype('float')
    x = np.arange(len(data))
    fit = np.polyval(np.polyfit(x, data, deg=order), x)

    if kind == 'high':
        data -= fit
    elif kind == 'low':
        data = data - (data - fit)
    else:
        raise ValueError('filter kind `{kind}` is not available')

    return data


def filter_interp_1d(
    data, method='IQR', kind='cubic', win=11, threshold=3.0, filter_boundaries=True
):  # noqa
    """
    Remove outliers using the IQR (inter-quartile range) method and
    interpolate using user-specified `kind` (default: 'cubic').
    Return outlier-removed and interpolated input array.

    Parameters
    ----------
    data : np.ndarray
        Input data (1D).
    method : str, optional
        Filter method to use (default: `IQR`).
    kind : str, optional
        Interpolation method for scipy.interpolate.interp1d (default: `cubic`).
    win : int, optional
        Size of moving window if required by chosen method (default: `11`).
    threshold : float, optional
        Threshold used for median absolute deviation (MAD) (default: `3.0`).
    filter_boundaries : bool, optional
        Filter flagged outlier indices at start and end of input array to avoid
        edge effects (if present despite padding) (default: `True`).

    Returns
    -------
    data_interp : np.ndarray
        Filtered and interpolated data.

    """
    METHODS = ['IQR', 'z-score', 'r_z-score', 'MAD', 'doubleMAD', 'r_doubleMAD', 'r_singleMAD']
    KIND_LIST = [
        'linear',
        'nearest',
        'nearest-up',
        'zero',
        'slinear',
        'quadratic',
        'cubic',
        'previous',
        'next',
    ]

    if data.ndim != 1:
        raise ValueError('data must be 1D array!')
    if kind not in KIND_LIST:
        raise ValueError(f'Parameter `kind` must be one of {KIND_LIST}')

    # get outlier indices
    if method == 'IQR':
        idx = iqr_filter(data)
    elif method == 'z-score':
        idx = zscore_filter(data)
    elif method == 'r_z-score':
        idx = moving_zscore_filter(data, win=win)
    elif method == 'MAD':
        idx = mad_filter(data, threshold=threshold, mad_mode='single')
    elif method == 'doubleMAD':
        idx = mad_filter(data, threshold=threshold, mad_mode='double')
    elif method == 'r_doubleMAD':
        idx = moving_mad_filter(data, win=win, threshold=threshold, mad_mode='double')
    elif method == 'r_singleMAD':
        idx = moving_mad_filter(data, win=win, threshold=threshold, mad_mode='single')
    else:
        raise ValueError(f'Given method ist not valid. Choose from {METHODS}')

    # filter flagged outlier indices at start and end of input array
    if filter_boundaries:
        # find consecutive flagged values
        ## get differences
        diff_idx = np.diff(idx)
        ## split into arrays holding consecutive flagged values
        diff_idx_split = np.split(diff_idx, np.nonzero(diff_idx > 1)[0])

        # check if first index is in input
        if np.isin(0, idx):
            # number of consecutive indices at start (add one due to split location)
            n_exclude_start = diff_idx_split[0].size + 1
            # exclude indices from flagged ones
            idx = idx[n_exclude_start:]

        # check last index is in input
        if np.isin(data.size - 1, idx):
            # number of consecutive indices at end
            n_exclude_end = diff_idx_split[-1].size
            # exclude indices from flagged ones
            idx = idx[:-n_exclude_end]

    # compute sampling indices
    x = np.arange(data.size)  # updated/altered input data

    # mask outliers for interpolation
    mask = np.ones(data.size, dtype='bool')
    mask[idx] = 0
    _data = data[mask]
    _x = x[mask]

    # create interpolation function
    _interp = interp.interp1d(_x, _data, kind=kind)
    # interpolate masked values
    data_interp = _interp(x)

    return data_interp


# *****************************************************************************
#                       SEAFLOOR AMPLITUDE DETECTION
# *****************************************************************************
def sta_lta_filter(a, nsta: int, nlta: int, axis=-1):  # noqa
    """
    Compute the STA/LTA ratio (short-time-average / longe-time-average)
    by continuously calculating the average values of the absolute amplitude
    of a seismic trace in two consecutive moving-time windows.

    Parameters
    ----------
    a : np.ndarray
        Seismic trace (1D) or section (2D).
    nsta : int
        Length of short time average window (samples).
    nlta : int
        Length of long time average window (samples).
    axis : int, optional
        Axis for which to compute STA/LTA ratio (default: -1).

    Returns
    -------
    np.ndarray
        Either 1D or 2D array of STA/LTA ratio (per trace).
    
    References
    ----------
    [^1]: Withers et al. (1998) A comparison of select trigger algorithms for automated global seismic phase and event detection,
          [http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.116.245&rep=rep1&type=pdf](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.116.245&rep=rep1&type=pdf)
    [^2]: Trnkoczy, A. (2012) Understanding and parameter setting of STA/LTA trigger algorithm,
          [https://gfzpublic.gfz-potsdam.de/rest/items/item_4097_3/component/file_4098/content](https://gfzpublic.gfz-potsdam.de/rest/items/item_4097_3/component/file_4098/content)
    [^3]: ObsPy, [https://docs.obspy.org/_modules/obspy/signal/trigger.html#classic_sta_lta_py](https://docs.obspy.org/_modules/obspy/signal/trigger.html#classic_sta_lta_py)

    """
    if any(s == 1 for s in a.shape):
        a = np.squeeze(a, axis=-1).copy()

    # calculate moving average
    sta = np.cumsum(a**2, axis=axis).astype('float')

    # copy for LTA
    lta = sta.copy()

    # compute the STA and the LTA
    if a.ndim == 1:
        sta[nsta:] = sta[nsta:] - sta[:-nsta]
        sta /= nsta
        lta[nlta:] = lta[nlta:] - lta[:-nlta]
        lta /= nlta

        # pad zeros
        sta[: nlta - 1] = 0
    elif a.ndim == 2:
        sta[nsta:, :] = sta[nsta:, :] - sta[:-nsta, :]
        sta /= nsta
        lta[nlta:, :] = lta[nlta:, :] - lta[:-nlta, :]
        lta /= nlta

        # pad zeros
        sta[: nlta - 1, :] = 0

    # avoid division by zero!
    return np.divide(sta, lta, out=np.zeros_like(sta, dtype=sta.dtype), where=(lta != 0))


# TODO: consider index of padded files
def detect_seafloor_reflection(
    data,
    idx_slice_start=None,
    nsta: int = None,
    nlta: int = None,
    win: int = 30,
    threshold: float = None,
    win_mad: int = None,
    win_mad_post: int = None,
    win_median: int = 11,
    n: int = 5,
    post_detection_filter: bool = True,
):
    """
    Detect seafloor reflection using the STA/LTA algorithm.
    Its commonly applied in seismology that evaluates the ratio of short- and long-term energy density.
    The initially sample indices found by the STA/LTA algorithm are used to
    create individual search windows per trace (idx - win <= x <= idx + win).
    Return indices of maximum amplitude(s) within individual search windows (shape: (ntraces,)).

    Parameters
    ----------
    data : np.ndarray
        Input seismic section (samples x traces).
    idx_slice_start : np.ndarray, optional
        Index of first non-padded sample in original data.
    nsta : int, optional
        Length of short time average window (in samples). If `None`: 0.1% of total samples.
    nlta : int, optional
        Length of long time average window (in samples). If `None`: 5% of total samples.
    win : int, optional
        Number of samples to pad search window with (default: `30`).
        Set search window to `win` samples deeper and `win` x 2 samples shallower than baseline.
    threshold : float, optional
        Threshold for seafloor amplitude detection after STA/LTA computation.
        If None, using background STA/LTA amplitudes from water column (default).
    win_mad : int, optional
        Number of traces used for Median Absolute Deviation (MAD) filtering.
        If None (default), this window is set to 5% of total traces.
    win_mad_post : int, optional
        Number of traces used for Median Absolute Deviation (MAD) filtering (after detection).
        If None (default), this window is set to 1% of total traces.
    win_median : int, optional
        Number of traces for rolling median filter, should be odd integer (default: `11`).
    n : int, optional
        Number of _n_ hightest amplitudes for each trace (default: `5`).
    post_detection_filter : bool, optional
        Apply optional Median Absolute Deviation (MAD) filtering
        after actual seafloor detection (default: `True`).

    Returns
    -------
    np.ndarray
        Indices of samples at maximum amplitude (per trace).

    """
    nsamples, ntraces = data.shape

    # check for zero traces (e.g. from merging)
    cnt_zero_traces = np.count_nonzero(data, axis=0)
    n_zero_traces = ntraces - np.count_nonzero(cnt_zero_traces, axis=0)

    # mask zero traces if found
    if n_zero_traces > 0:
        mask_nonzero_traces = cnt_zero_traces.astype('bool')
        data = data[:, mask_nonzero_traces]

    if nsta is None:
        nsta = int(np.around(nsamples * 0.001))
    if nlta is None:
        nlta = int(np.around(nsamples * 0.05))

    if nsta < 3:
        nsta = 3
        nlta = 50
        print(f'[WARNING]    Changed nsta={nsta} and nlta={nlta}!')

    # (1) calc standard STA/LTA from data array
    sta_lta = sta_lta_filter(data, nsta, nlta, axis=0)

    # (2) detect first significant amplitude peak (sample indices)
    # CAUTION: could be outlier (e.g. noise bursts in water column)
    #          but that misdetection will be filtered in the subsequent step!
    threshold = sta_lta[nlta:nlta * 2, :].max() if threshold is None else threshold
    idx_sta_lta = np.argmax(sta_lta > threshold, axis=0)

    if idx_slice_start is not None:
        idx_sta_lta += idx_slice_start
        
    if idx_slice_start is not None:
        # (3) replace seafloor detections outside sample range with median value
        idx_sta_lta = np.where(
            np.logical_or(idx_sta_lta > nsamples - idx_slice_start, idx_sta_lta < idx_slice_start),
            np.median(idx_sta_lta),
            idx_sta_lta,
        )

    # # (3) outlier detection & removal  # TODO: unnecessary?
    if win_mad is None:
        win_mad = int(idx_sta_lta.size * 0.02)
        win_mad = win_mad + 1 if win_mad % 2 == 0 else win_mad  # must be odd
        win_mad = 7 if win_mad < 7 else win_mad                 # at least 7 traces

    idx_sta_lta = filter_interp_1d(
        idx_sta_lta, method='r_doubleMAD', kind='cubic', threshold=3, win=win_mad
    ).astype('int')

    # (4) apply moving median filter to remove large outliers
    win_median = int(0.3 * ntraces) if win_median > ntraces else win_median
    idx_sta_lta = moving_median(idx_sta_lta, win_median, padded=True).astype('int')
    
    # (5) detect `actual` first break amplitude
    if win > 0:
        # init index array
        idx_arr = np.arange(nsamples)[:, None]
        # create mask from slices (upper index <= slice <= lower index)
        idx_upper, idx_lower = (idx_sta_lta - win), (idx_sta_lta + win)  # *2
        mask = (idx_arr >= idx_upper) & (idx_arr <= idx_lower)
        # get indices from mask
        indices = np.apply_along_axis(np.nonzero, 0, mask).squeeze()
    
        # subset input array using indices of search window
        sta_lta_win = np.take_along_axis(data, indices, axis=0)
    
        # get `n` largest values for each trace subset
        # n = 5
        idx_nlargest = np.argpartition(-sta_lta_win, n, axis=0)[:n]
        # sort the indices for each trace (ascending order)
        idx_nlargest = np.take_along_axis(
            idx_nlargest, axis=0, indices=np.argsort(idx_nlargest, axis=0)
        )
    
        # get indices to split `idx_nlargest` into groups of different peak amplitudes
        idx_nlargest_sel = [
            np.nonzero(tr > 1)[0][0] if np.nonzero(tr > 1)[0].size > 0 else n
            for tr in np.diff(idx_nlargest, 1, axis=0).T
        ]
    
        # split the index array of `n` largest values and select first significant (positive) amplitude
        idx_nlargest_sel = [
            np.split(tr, [i])[0] if i != 0 else np.array([tr[i]])
            for tr, i in zip(idx_nlargest.T, idx_nlargest_sel)
        ]
    
        # get index of max. amplitude within selected maxima of first significant (positive) amplitude (NOTE: subset index!)
        idx_peak_amp = np.asarray(
            [
                nlarge[np.argmax(tr[i])]
                for nlarge, tr, i in zip(idx_nlargest.T, sta_lta_win.T, idx_nlargest_sel)
            ]
        )
    
        # convert subset indices to indices of seismic section
        idx_peak_amp += idx_upper
    else:
        idx_peak_amp = idx_sta_lta
    
    if n_zero_traces > 0:
        x = np.arange(0, ntraces)  # create trace idx WITH zero traces
        x_masked = x[mask_nonzero_traces]  # masked zero traces
        # create interpolation function
        _interp = interp.interp1d(x_masked, idx_peak_amp, kind='linear')
        # interpolate masked indices of zero traces
        idx_peak_amp = _interp(x).astype('int')

    # (6) additional outlier detection & removal
    if post_detection_filter:
        if win_mad_post is None:
            win_mad_post = int(idx_sta_lta.size * 0.01)
            win_mad_post = (
                win_mad_post + 1 if win_mad_post % 2 == 0 else win_mad_post
            )  # must be odd
            win_mad_post = 7 if win_mad_post < 7 else win_mad_post  # at least 7 traces
        idx_peak_amp = filter_interp_1d(
            idx_peak_amp, method='r_doubleMAD', kind='cubic', threshold=3, win=win_mad_post
        ).astype('int')

    return idx_peak_amp.astype('int')


# *****************************************************************************
#                            FREQUENCY FILTER
# *****************************************************************************
def _butterworth_filter_coefficients(btype: str, order: int, cutoff: float, fs: float):
    # Nyquist frequency
    nyq = fs / 2
    # normalized cutoff frequency
    cutoff = np.asarray(cutoff)
    cutoff_norm = cutoff / nyq
    # filter coefficients
    sos = signal.butter(order, cutoff_norm, analog=False, btype=btype, output='sos')
    return sos


def butterworth_filter(
    data: np.ndarray, btype: str, cutoff: float, fs: float, order: int = 9, axis: int = -1
) -> np.ndarray:
    """
    Apply butterworth filter to input signal. Can be `lowpass`, `highpass`, or `bandpass`.

    Parameters
    ----------
    data : np.ndarray
        Input data.
    cutoff : float | tuple
        Cutoff frequency (in Hz).
    fs : float
        Sampling frequency (in Hz).
    order : int, optional
        Butterworth filter order (default: 9).
    axis : int, optional
        The axis of x to which the filter is applied (default: -1).

    Returns
    -------
    y : np.ndarray
        Filtered input signal.

    References
    ----------
    [^1]: [https://stackoverflow.com/a/48677312](https://stackoverflow.com/a/48677312)

    """
    if btype not in ['lowpass', 'highpass', 'bandpass']:
        raise ValueError('``btype`` has to be ``lowpass``, ``highpass``, or ``bandpass``!')
    sos = _butterworth_filter_coefficients(btype, order, cutoff, fs)
    y = signal.sosfiltfilt(sos, data, axis=axis)
    return y


bandpass_butterworth = partial(butterworth_filter, btype='bandpass')
lowpass_butterworth = partial(butterworth_filter, btype='lowpass')
highpass_butterworth = partial(butterworth_filter, btype='highpass')


def filter_frequency(
    data: np.ndarray,
    freqs: list,
    fs: float,
    filter_type: str,
    gpass: int = 1,
    gstop: int = 10,
    axis: int = -1,
) -> np.ndarray:
    """
    Apply freqeuncy filter by specifing passband and stopband frequencies.
    Possible filter types:
    
      - `bandpass`:     freqs = [*f1*, *f2*, *f3*, *f4*]
      - `lowpass`:      freqs = [*f_stopband*, *f_cutoff*]
      - `highpass`:     freqs = [*f_cutoff*, *f_stopband*]

    Parameters
    ----------
    data : np.ndarray
        Input data.
    freqs : list
        List of frequencies defining filter (same unit as `fs`!).
    fs : float
        Sampling frequency (in Hz).
    filter_type : str
        Filter type to apply (`bandpass`, `lowpass`, `highpass`)
    gpass : int, optional
        The maximum loss in the passband (dB).
    gstop : int, optional
        The minimum attenuation in the stopband (dB).
    axis : int, optional
        The axis of x to which the filter is applied (default: `-1`).

    Returns
    -------
    np.ndarray
        Filtered input data.

    """
    if filter_type == 'bandpass':
        wp = [freqs[0], freqs[-1]]
        ws = [freqs[1], freqs[2]]
        if not freqs == sorted(freqs):
            raise ValueError('Invalid filter frequencies!')
    elif filter_type == 'lowpass':
        wp, ws = freqs
        if wp > ws:
            raise ValueError('Invalid filter frequencies!')
    elif filter_type == 'highpass':
        wp, ws = freqs
        if wp < ws:
            raise ValueError('Invalid filter frequencies!')

    # get Butterworth filter order and natural frequency
    N, Wn = signal.buttord(wp, ws, gpass, gstop, analog=False, fs=fs)
    # filter coefficients
    sos = signal.butter(N, Wn, analog=False, btype=filter_type, output='sos', fs=fs)

    return signal.sosfiltfilt(sos, data, axis=axis)


bandpass_filter = partial(filter_frequency, filter_type='bandpass')
lowpass_filter = partial(filter_frequency, filter_type='lowpass')
highpass_filter = partial(filter_frequency, filter_type='highpass')
