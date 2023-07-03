"""
Thresholding operators for POCS algorithm.
These functions are only used when `pywavelets` package is not installed (fallback option).

"""
import numpy as np


def _soft_threshold(data, value, substitute=0):
    """
    Soft thresholding (from `pywavelet`).

    Parameters
    ----------
    data : np.ndarray
        Input coefficients.
    value : float
        Threshold value.
    substitute : float, optional
        Value to insert for values below threshold (default: `0`).

    Returns
    -------
    np.ndarray
        Coefficients with applied threshold.

    References
    ----------
    [^1]: PyWavelets, [https://github.com/PyWavelets/pywt/blob/master/pywt/_thresholding.py](https://github.com/PyWavelets/pywt/blob/master/pywt/_thresholding.py)

    """
    data = np.asarray(data)
    magnitude = np.absolute(data)

    with np.errstate(divide='ignore'):
        # divide by zero okay as np.inf values get clipped, so ignore warning.
        thresholded = 1 - value / magnitude
        thresholded.clip(min=0, max=None, out=thresholded)
        thresholded = data * thresholded

    if substitute == 0:
        return thresholded
    else:
        cond = np.less(magnitude, value)
        return np.where(cond, substitute, thresholded)


def _nn_garrote(data, value, substitute=0):
    """
    Non-negative Garrote thresholding (from `pywavelet`).

    Parameters
    ----------
    data : np.ndarray
        Input coefficients.
    value : float
        Threshold value.
    substitute : float, optional
        Value to insert for values below threshold (default: `0`).

    Returns
    -------
    np.ndarray
        Coefficients with applied threshold.

    References
    ----------
    [^1]: PyWavelets, [https://github.com/PyWavelets/pywt/blob/master/pywt/_thresholding.py](https://github.com/PyWavelets/pywt/blob/master/pywt/_thresholding.py)

    """
    data = np.asarray(data)
    magnitude = np.absolute(data)

    with np.errstate(divide='ignore'):
        # divide by zero okay as np.inf values get clipped, so ignore warning.
        thresholded = 1 - value**2 / magnitude**2
        thresholded.clip(min=0, max=None, out=thresholded)
        thresholded = data * thresholded

    if substitute == 0:
        return thresholded
    else:
        cond = np.less(magnitude, value)
        return np.where(cond, substitute, thresholded)


def _hard_threshold(data, value, substitute=0):
    """
    Hard thresholding (from `pywavelet`).

    Parameters
    ----------
    data : np.ndarray
        Input coefficients.
    value : float
        Threshold value.
    substitute : float, optional
        Value to insert for values below threshold (default: `0`).

    Returns
    -------
    np.ndarray
        Coefficients with applied threshold.

    References
    ----------
    [^1]: PyWavelets, [https://github.com/PyWavelets/pywt/blob/master/pywt/_thresholding.py](https://github.com/PyWavelets/pywt/blob/master/pywt/_thresholding.py)

    """
    data = np.asarray(data)
    cond = np.less(np.absolute(data), value)
    return np.where(cond, substitute, data)
