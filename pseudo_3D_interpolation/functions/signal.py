"""Utility functions for acoustic signals."""

import numpy as np
from scipy.signal import hilbert, convolve2d

from .filter import moving_window_2D
from .utils import rescale, pad_along_axis, get_array_module


def SNR(x, noise):
    """
    Signal-to-noise ratio (SNR).
    
    Parameters
    ----------
    x : np.ndarray
        Original signal.
    noise : np.ndarray
        Noisy signal.

    Returns
    -------
    float
        Signal-to-noise ratio between arrays (in dB).

    References
    ----------
    [^1]: Yang et al. (2012) Curvelet-based POCS interpolation of nonuniformly sampled seismic records

    """
    if np.linalg.norm(x - noise) == 0:
        return np.Inf
    else:
        return 10 * np.log10(np.sum(np.power(x, 2)) / np.sum(np.power(x - noise, 2)))


def PSNR(x, noise, max_pixel=1.0):
    """
    Peak signal-to-noise ratio (SNR).

    Parameters
    ----------
    x : np.ndarray
        Original signal.
    noise : np.ndarray
        Noisy signal.
    max_pixel : float, optional
        Maximum fluctuation in input image type. For `float`: 1, `uint8`: 255.
        If None, compute and use max(x).

    Returns
    -------
    float
        Peak signal-to-noise ratio between arrays (in dB).

    """
    MSE = np.mean((x - noise) ** 2)
    if MSE == 0:
        return np.inf
    if max_pixel is None:
        max_pixel = np.max(x)
    return 10 * np.log10(max_pixel / np.sqrt(MSE))


def estimate_noise_level(img):
    """
    Estimate image noise level based on Immerkær (1996) "Fast Noise Variance Estimation".

    Parameters
    ----------
    img : np.ndarray
        Input image.

    Returns
    -------
    sigma : float
        Noise level factor.
    
    References
    ----------
    [^1]: https://stackoverflow.com/a/25436112

    """
    nrows, ncols = img.shape

    M = [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]

    sigma = np.sum(np.sum(np.absolute(convolve2d(rescale(img, 0, 255), M))))
    sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (ncols - 2) * (nrows - 2))
    return sigma


# *****************************************************************************
#                           SIGNAL PROCESSING
# *****************************************************************************
def gain(
    data,
    twt,
    tpow=0.0,  # multiply data by t^tpow
    epow=0.0,  # multiply data by exp(epow*t)
    etpow=1.0,  # multiply data by exp(epow*t^etpow)
    ebase=None,  # use as base for exp function (default: e)
    gpow=0.0,  # take signed gpowth power of scaled data
    agc: bool = False,
    agc_win=0.05,
    agc_kind: str = 'rms',
    agc_sqrt: bool = False,
    clip=None,  # clip any value whose magnitude exceeds clipval
    pclip=None,  # clip any value greater than clipval
    nclip=None,  # clip any value less than clipval
    qclip=None,  # clip by quantile on absolute values on trace
    linear=None,
    pgc=None,
    bias=None,  # bias data by adding an overall bias value
    scale=1.0,  # multiply data by overall scale factor
    norm: bool = False,  # divide data by overall scale factor
    norm_rms: bool = False,  # normalize using RMS amplitude
    copy: bool = True,
    axis=-1,
):
    """
    Apply various different types of gain for either single trace (1D array) or seismic section (2D array).
    
    !!! warning "Copyright"
        
        This function is a Python implementation of the Seismic Unix `sugain` module.
        Please refer to the license file `LICENSE_SeismicUnix`!
    
    
    Parameters
    ----------
    data : np.ndarray
        Seismic trace (nsamples,) or section (nsamples, ntraces).
    twt : np.ndarray
        Array of samples (in seconds TWT) appropriate spacing of sample rate (`dt`).
    tpow : float, optional
        Multiply data by t^tpow (default: `0.0`).
    epow : float, optional
        Multiply data by exp(epow*t) (default: `0.0`).
    etpow : float, optional
        Multiply data by exp(epow*t^etpow) (default: `1.0`).
    ebase : float, optional
        Base of exponential function (default: `e`).
    gpow : float, optional
        Take signed gpowth power of scaled data (default: `0.0`).
    agc : bool, optional
        Whether to apply Automatic Gain Control (AGC) (default: `False`).
    agc_win : float, optional
        AGC window length (in seconds) (default: `0.05`).
    agc_kind : str, optional
        Kind of AGC: `rms` (default), 'mean', or 'median'.
    agc_sqrt : bool, optional
        Whether to square AGC values (default: `False`). Use with caution: Reduces noise but also weak amplitudes!
    clip : float, optional
        Clip any value whose magnitude exceeds clipval (default: `None`).
    pclip : float, optional
        Clip any value greater than clipval (default: `None`).
    nclip : float, optional
        Clip any value less than clipval (default: `None`).
    qclip : float, optional
        Clip by quantile on absolute values on trace (default: `None`).
    linear : tuple, optional
        Apply linear gain function by multiplying trace(s) with linearly interpolated array between (start, stop).
    pgc : dict, optional
        Apply Programmed Gain Control (PGC) function using defined dict(TWT:GAIN) pairs.
    bias : float, optional
        Bias data by adding an overall bias value (default: `None`).
    scale : float, optional
        Multiply data by overall scale factor (default: `1.0`).
    norm : bool, optional
        Divide data by overall scale factor (default: `False`).
    norm_rms : bool, optional
        Normalize using RMS amplitude (default: `False`).
    copy : bool, optional
        Copy input data (no change of input data) (default: `True`).
    axis : int, optional
        Axis along which to gain (default: `-1`).

    Returns
    -------
    data : np.ndarray
        Input data with applied gain function(s) along `axis`.
    
    Notes
    -----
    By default, the input array will be copied (`copy=True`) to avoid updating of the input data in place.
    
    References
    ----------
    [^1]: `sugain` module help, [http://sepwww.stanford.edu/oldsep/cliner/files/suhelp/sugain.txt](http://sepwww.stanford.edu/oldsep/cliner/files/suhelp/sugain.txt)

    """
    if copy:
        data = data.copy()

    if data.ndim == 1:
        nsamples, ntraces, ndim = data.size, None, 1
    elif data.ndim == 2:
        if axis == 0:
            nsamples, ntraces = data.shape
        else:
            ntraces, nsamples = data.shape
        ndim = 2
    else:
        if axis == 0:
            nsamples, nil, nxl = data.shape
        elif axis == 2 or axis == -1:
            nil, nxl, nsamples = data.shape
        else:
            raise ValueError('For 3D datasets the time axis must be either first or last.')
        ndim = 2
        
    axis_dims = 0 if axis == -1 else 1

    for param, name in zip(
        [tpow, epow, etpow, gpow, clip, pclip, nclip, qclip, bias, scale],
        ['tpow', 'epow', 'etpow', 'gpow', 'clip', 'pclip', 'nclip', 'qclip', 'bias', 'scale'],
    ):
        if (param is not None) and not isinstance(param, (int, float)):
            raise ValueError(f'`{name}` must be either int or float')
            
    dt = round(np.diff(twt.data).mean() * 1e9, 0) / 1e9

    # bias
    if (bias is not None) and (bias != 0.0):
        data += bias

    # tpow
    if (tpow is not None) and (tpow != 0.0):
        tpow_fact = np.power(twt, tpow)
        tpow_fact[0] = 0.0 if twt[0] == 0.0 else np.power(twt[0], tpow)
        if ndim == 1:
            data *= tpow_fact
        else:
            # data *= tpow_fact[:, None]
            data *= np.expand_dims(tpow_fact, axis=axis_dims)

    # epow & etpow (& ebase)
    if epow is not None and epow != 0.0:
        # etpow
        etpow_fact = np.power(twt, etpow)
        # epow
        if ebase is not None:
            epow_fact = np.power(ebase, epow * etpow_fact)
        else:
            epow_fact = np.exp(epow * etpow_fact)
        if ndim == 1:
            data *= epow_fact
        else:
            # data *= epow_fact[:, None]
            data *= np.expand_dims(epow_fact, axis=axis_dims)

    # gpow (take signed gpowth power of scaled data)
    if (gpow is not None) and (gpow != 0.0):
        # workaround to prevent numpy from complaining about negative numbers
        data = np.sign(data) * np.abs(data) ** gpow
        
    # AGC
    if agc:
        agc_win = get_AGC_samples(agc_win, dt)
        data = AGC(data, agc_win, kind=agc_kind, squared=agc_sqrt, axis=axis)

    # clip
    if clip is not None:
        data = np.where(np.abs(data) > clip, clip * np.sign(data), data)
        # data = np.where(data < -clip, -clip, data)

    # pclip
    if pclip is not None:
        data = np.where(data > pclip, pclip, data)

    # nclip
    if nclip is not None:
        data = np.where(data < nclip, nclip, data)

    # qclip
    if qclip is not None:
        qclip_per_trace = np.quantile(np.abs(data), q=qclip, axis=axis)
        data = np.where(np.abs(data) > qclip_per_trace, qclip_per_trace * np.sign(data), data)
        
    # linear
    if linear is not None:
        g = np.linspace(min(linear), max(linear), twt.size, endpoint=True)
        data *= np.expand_dims(g, axis=axis_dims) if ndim > 1 else g
        
    # Programmed Gain Control
    if isinstance(pgc, dict):
        g = programmed_gain_control(twt, pgc)
        data *= np.expand_dims(g, axis=axis_dims) if ndim > 1 else g

    # norm_rms
    if norm_rms:
        data = rms_normalization(data, axis=axis)

    # scale
    if (scale is not None) and (scale != 1.0):
        data = data * scale if not norm else data * 1 / scale

    return data


def get_AGC_samples(win: float, dt: float):
    """
    Convert AGC window length from TWT (seconds) to number of samples.
    In case of even number of samples, the window length is increased by one (1) sample.

    Parameters
    ----------
    win : int | float, np.ndarray
        AGC window length (in seconds).
    dt : float
        Sampling interval (in seconds).

    Returns
    -------
    samples : int | float, np.ndarray
        Number of samples in AGC window.

    """
    samples = int(win / dt)
    samples = samples + 1 if samples % 2 == 0 else samples
    return samples


def AGC(
        x,
        win: int,
        kind: str = 'rms',
        pad: bool = True,
        pad_mode: str = 'constant',
        squared: bool = False,
        return_gain_func: bool = False,
        axis: int = -1
):
    """
    Apply Automatic Gain Control (AGC) function to input data (1D trace, 2D profile, or 3D cube).

    Parameters
    ----------
    x : np.ndarray
        Input trace (1D), profile/iline/xline (2D) or cube (3D).
    win : int
        Samples (i.e. values) in AGC window.
    kind : str, optional
        AGC kind (default: `'rms'`). Available are `'rms'`, `'mean'`, or `'median'`.
    pad : bool, optional
        Pad time axis at top and bottom (default: `True`). Time axis will be truncated if not set!
    pad_mode : str, optional
        Pad mode (default: `'constant'`). Refer to `np.pad` documentation for more details.
    squared : bool, optional
        Compute squared AGC to enhance major amplitudes and redue noise (default: `False`). Use with caution!
    return_gain_func : bool, optional
        Whether to return gain function for later AGC removal (default: `False`).
    axis : int, optional
        Index of time axis (default: `-1`).

    Returns
    -------
    np.ndarray
        Input data with applied AGC function.

    """
    if not isinstance(win, int):
        raise TypeError(f'`win` must be integer not {type(win)}')
    win = win + 1 if win % 2 == 0 else win
    
    # check for dask or numpy array
    m = get_array_module(x)
    # print('module:', m)
    
    # print('x.shape:', x.shape)
    if pad:
        npad = win // 2
        pad_shape = [(0, 0) for i in range(x.ndim)]
        pad_shape[axis] = (npad, npad)
        x_pad = m.pad(x, pad_shape, mode=pad_mode)
    else:
        x_pad = x
    # print('x_pad:', type(x_pad))
    # print('x_pad.shape:', x_pad.shape)
    
    # create sliding windows (as memory views)
    # print('win:  ', win)
    # print('axis: ', axis)
    x_win = m.lib.stride_tricks.sliding_window_view(x_pad, win, axis=axis)
    # print('x_win:', type(x_win))
    # print('x_win.shape:', x_win.shape)
    
    # NOTE: always aggregate last axis of `sliding_window_view`!
    if kind == 'rms':
        g = m.sqrt(m.mean(x_win ** 2, axis=-1))
    elif kind == 'mean':
        g = m.mean(x_win, axis=-1)
    elif kind == 'median':
        g = m.median(x_win, axis=-1)
    else:
        raise ValueError(f'Unknown AGC kind "{kind}"')
    
    # print('g.shape:', g.shape)
    g[g == 0] = 1.0  # avoid division errors
    
    x *= 1 / g  # apply gain function
    
    if squared:
        x = m.sign(x) * x**2
    
    if return_gain_func:
        return x, g
    return x


def _find_nan_indices_1d(x):
    """Find indices of NaN values in `x`."""
    return np.isnan(x), lambda z: z.nonzero()[0]


def _find_nearest_twt(x, values, return_indices=False):
    """Find nearest TWT value(s)."""
    values = np.atleast_1d(values)
    indices = np.abs(np.subtract.outer(x, values)).argmin(0)
    out = x[indices]
    out = out if len(out) > 1 else out[0]
    if return_indices:
        return out, indices
    return out


def programmed_gain_control(
        twt: np.ndarray,
        twt_gain: dict,
):
    """
    Compute Programmed Gain Control (PGC) function using linear interpolation between `twt_gain` pairs.

    Parameters
    ----------
    twt : np.ndarray
        TWT values (in seconds).
    twt_gain : dict
        Dictionary of TWT (keys) and corresponding gain values (vals).

    Returns
    -------
    g : TYPE
        DESCRIPTION.

    """
    g = np.full_like(twt, np.nan, dtype='float32')  # initialize output array
    
    _twt = np.asarray(list(twt_gain.keys()))
    # print('_twt:', _twt)
    _idx = np.argsort(_twt)  # get TWT sorter
    # print('_idx:', _idx)
    _twt = _twt[_idx]  # sort TWT values
    _gain = np.asarray(list(twt_gain.values()))[_idx]  # sort gain values
    # print('_twt:', _twt)
    # print('_gain:', _gain)
    # select nearest TWT value(s) (and corresponding indices) from `twt` array
    _twt, idx = _find_nearest_twt(twt, _twt, return_indices=True)
    # print('_twt:', _twt)
    # print('idx (new):', idx)
    g[idx] = _gain  # set known gain values
    if np.isnan(g[0]):
        g[0] = _gain[0]
    if np.isnan(g[-1]):
        g[-1] = _gain[-1]
    
    # get indices of NaN values
    mask_nan, func = _find_nan_indices_1d(g)
    
    # interpolate missing values (linear)
    g[mask_nan] = np.interp(func(mask_nan), func(~mask_nan), g[~mask_nan])
    
    return g


def rms(array, axis=None):
    r"""
    Calculate the RMS amplitude(s) of a given array.

    Parameters
    ----------
    array : np.ndarray
        Amplitude array.
    axis : int, tuple, list (optional)
        Axis for RMS amplitude calculation (default: `None`, i.e. single value for whole array).

    Returns
    -------
    rms : np.ndarray
        Root mean square (RMS) amplitude(s).

    $$
    rms = \sqrt{\frac{\sum{a^2}}{N}}
    $$

    """
    if axis is None:
        N = array.size
    elif isinstance(axis, int):
        N = array.shape[axis]
    elif isinstance(axis, (tuple, list)):
        N = np.prod([array.shape[ax] for ax in axis])

    return np.sqrt(np.sum(array**2, axis=axis) / N)


def rms_normalization(signal, axis=None):
    """
    Normalize signal using RMS amplitude of input array.

    Parameters
    ----------
    signal : np.ndarray
        Input trace(s).
    axis : int, optional
        Axis used for RMS amplitude calculation (default: `None`, i.e. whole array).

    Returns
    -------
    np.ndarray
        Normalized signal using RMS amplitude.

    References
    ----------
    [^1]: [https://superkogito.github.io/blog/2020/04/30/rms_normalization.html](https://superkogito.github.io/blog/2020/04/30/rms_normalization.html)

    """
    signal = np.asarray(signal)
    _rms = rms(signal, axis=axis)
    if signal.ndim == 1:
        _rms = 1.0 if _rms == 0.0 else _rms
    else:
        _rms[_rms == 0.0] = 1.0

    return signal / _rms


def balance_traces(
    traces,
    scale: str = 'rms',
    n_traces: int = None,
    axis_samples: int = None,
) -> np.ndarray:
    """
    Balance (i.e. scale) adjacent seismic traces.
    This function uses one of the following reference amplitude(s) per trace(s):
    
      - `rms` (**default**)
      - `peak` (absolute value)
      - `mean` (absolute value)
      - `median` (absolute value)
    
    The reference amplitude is computed
      
      - for the whole dataset (`axis_samples = None`),
      - for each individual trace (`axis_samples >= 0`), or
      - in moving windows of `n_traces` length.

    Parameters
    ----------
    traces : np.ndarray
        Input traces, e.g. with shape: (nsamples x ntraces).
    scale : str, optional
        Amplitude scaling (balancing) mode (default: `rms`).
    n_traces : int, optional
        Number of traces used for windowed balacning (default: `None`).
    axis_samples : int, optional
        Axis for balancing computation (default: `0`, for nsamples x ntraces).

    Returns
    -------
    traces_eq : np.ndarray
        Equalized (i.e. balanced) traces.

    """
    if axis_samples is None:
        axis_samples = 0  # shape: (nsamples x ntraces)

    if scale.lower() not in ['rms', 'max', 'peak', 'mean', 'median']:
        raise ValueError(
            'Unknown equalizing method. Choose either "rms", "peak", "mean", or "median".'
        )
    else:
        scale = scale.lower()

    traces = np.asanyarray(traces)

    # (1) trace-by-trace balancing
    if n_traces is None or n_traces == 1:
        if scale == 'rms':
            amp_ref = rms(traces, axis=axis_samples)
        elif scale in ['peak', 'max']:
            amp_ref = np.max(np.abs(traces), axis=axis_samples)
        elif scale == 'mean':
            amp_ref = np.mean(np.abs(traces), axis=axis_samples)
        elif scale == 'median':
            amp_ref = np.median(np.abs(traces), axis=axis_samples)

    # (2) windowed trace balancing
    elif n_traces > 1:
        if traces.ndim != 2:
            raise ValueError('Input array must be 2D array!')

        n_traces = n_traces + 1 if n_traces % 2 == 0 else n_traces

        if axis_samples == 0:  # (nsamples x ntraces)
            win = (traces.shape[axis_samples], n_traces)
            axis = 1
        elif axis_samples == 1:  # (ntraces, nsamples)
            win = (n_traces, traces.shape[axis_samples])
            axis = 0
        else:
            raise ValueError('False value for ``axis_samples`` (either 0 or 1)!')

        npad = (n_traces - 1) // 2
        traces_pad = pad_along_axis(traces, n=npad, axis=axis)
        traces_win = moving_window_2D(traces_pad, w=win, dx=1, dy=1).squeeze()

        axis = (-2, -1)
        if scale == 'rms':
            amp_ref = rms(traces_win, axis=axis)
        elif scale in ['peak', 'max']:
            amp_ref = np.max(np.abs(traces_win), axis=axis)
        elif scale == 'mean':
            amp_ref = np.mean(np.abs(traces_win), axis=axis)
        elif scale == 'median':
            amp_ref = np.median(np.abs(traces_win), axis=axis)

    if traces.ndim == 1:
        amp_ref = 1.0 if amp_ref == 0 else amp_ref
    else:
        amp_ref[amp_ref == 0.0] = 1.0
    traces_balanced = traces / amp_ref
    # traces_balanced = rescale(traces, vmin=-amp_ref, vmax=amp_ref)

    assert traces.shape == traces_balanced.shape, 'Something went wrong here...'

    return traces_balanced


def calc_reference_amplitude(traces, axis: int = None, scale: str = 'rms'):
    """
    Calculate reference amplitude per trace using user-defined scaling (`rms` or `max`).

    Parameters
    ----------
    traces : np.ndarray
        Input traces.
    axis : int, optional
        Axis along that reference amplitudes will be calculated (default: `None`).
    scale : str, optional
        Scale using either `rms` (default) or `max` amplitudes.

    Returns
    -------
    amp_ref : np.ndarray
        Reference amplitude array.

    """
    if scale == 'rms':
        amp_ref = rms(traces, axis=axis)
    elif scale in ['peak', 'max']:
        amp_ref = np.max(np.abs(traces), axis=axis)

    # amp_ref[amp_ref == 0.0] = 1.0
    amp_ref = np.where(amp_ref == 0.0, 1.0, amp_ref)

    return amp_ref


def envelope(signal, axis=-1):
    """
    Compute envelope of a seismic trace (1D), section (2D) or cube (3D) using the Hilbert transform.

    Parameters
    ----------
    signal : np.ndarray
        Seismic trace (1D) or section (2D).
    axis : int, optional
        Axis along which to do the transformation (default: `-1`).

    Returns
    -------
    np.ndarray
        Amplitude envelope of input array along `axis`.

    """
    signal_analytic = hilbert(signal, axis=axis)
    return np.abs(signal_analytic).astype(signal.dtype)


def get_resampled_twt(twt, n_resamples, n_samples):
    """
    Return resampled TWT array.

    Parameters
    ----------
    twt : np.ndarray
        Orignial TWT array.
    n_resamples : int
        Number of resampled trace samples.
    n_samples : int
        Number of original trace samples.

    Returns
    -------
    np.ndarray
        Resampled twt.

    """
    return np.arange(0, n_resamples) * (twt[1] - twt[0]) * n_samples / float(n_resamples) + twt[0]


def freq_spectrum(signal, Fs, n=None, taper=True, return_minmax=False):
    """
    Compute frequency spectrum of input signal given a sampling rate (`Fs`).

    Parameters
    ----------
    signal : np.ndarray
        1D signal array.
    Fs : int
        Sampling rate/frequency (Hz).
    n : int, optional
        Length of FFT, i.e. number of points (default: len(signal)).
    taper : TYPE, optional
        Window function applied to time signal to improve frequency domain properties (default: `True`)

    Returns
    -------
    f : np.ndarray
        Array of signal frequencies.
    a_norm : np.ndarray
        Magnitude of amplitudes per frequency.
    f_min : float
        Minimum frequency with actual signal content.
    f_max : float
        Maximum frequency with actual signal content.

    """
    # signal length (samples)
    N = len(signal)
    # select window function
    if taper:
        win = np.blackman(N)
    else:
        win = np.ones((N))
    # apply tapering
    s = signal * win

    # number of points to use for FFT
    if n is None:
        n = N

    # calc real part of FFT
    a = np.abs(np.fft.rfft(s, n))
    # calc frequency array
    f = np.fft.rfftfreq(n, 1 / Fs)

    # scale magnitude of FFT by used window and factor of 2 (only half-spectrum)
    a_norm = a * 2 / np.sum(win)

    if return_minmax:
        # get frequency limits using calculated amplitude threshold
        slope = np.abs(np.diff(a_norm) / np.diff(f))  # calculate slope
        threshold = (slope.max() - slope.min()) * 0.001  # threshold amplitude
        f_limits = np.where(a_norm > threshold)[0]  # get frequency limits
        f_min, f_max = f[f_limits[0]], f[f_limits[-1]]  # select min/max frequencies
        f_min, f_max = np.min(f_limits), np.max(f_limits)  # select min/max frequencies
        return f, a_norm, f_min, f_max
    else:
        return f, a_norm
