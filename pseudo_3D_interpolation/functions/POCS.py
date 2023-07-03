"""Functions used for POCS interpolation script."""

import time
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .backends import pywt_enabled

if pywt_enabled:
    from pywt._thresholding import soft as _soft_threshold
    from pywt._thresholding import hard as _hard_threshold
    from pywt._thresholding import nn_garrote as _nn_garrote
else:
    from .threshold_operator import _soft_threshold, _hard_threshold, _nn_garrote


# FUNCTIONS
def get_number_scales(x):
    """
    Compute number of shearlet scales based on input array shape.

    References
    ----------
    [^1]: [https://github.com/grlee77/PyShearlets/blob/master/FFST/_scalesShearsAndSpectra.py](https://github.com/grlee77/PyShearlets/blob/master/FFST/_scalesShearsAndSpectra.py)
    
    """
    scales = int(np.floor(0.5 * np.log2(np.max(x.shape))))
    return scales if scales >= 1 else 1


def add_cbar(im, ax, extend='neither'):  # noqa
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, extend=extend)


# =================================================================================================
#                                           CORE FUNCTIONS
# =================================================================================================
def _hard_threshold_perc(x, perc, sub):  # noqa
    """Wrapper for hard thresholding using percentile of abs(x)."""
    thresh = np.percentile(np.abs(x), perc)
    return _hard_threshold(x, thresh, sub)


def _soft_threshold_perc(x, perc, sub):  # noqa
    """Wrapper for soft thresholding using percentile of abs(x)."""
    thresh = np.percentile(np.abs(x), perc)
    return _soft_threshold(x, thresh, sub)


def _nn_garrote_perc(x, perc, sub):  # noqa
    """Wrapper for garrote thresholding using percentile of abs(x)."""
    thresh = np.percentile(np.abs(x), perc)
    return _nn_garrote(x, thresh, sub)


def threshold(data, thresh, sub=0, kind='soft'):
    """
    Apply user-defined threshold to input data (2D).

    Parameters
    ----------
    data : np.ndarray
        Input data.
    thresh : float, complex
        Threshold cut-off value.
    sub : int, float, optional
        Substitution value (default: `0`).
    kind : str, optional
        Threshold method:
            
          - `soft` (**default**)
          - `garrote`
          - `hard`
          - `soft-percentile`
          - `garrote-percentile`
          - `hard-percentile`

    Returns
    -------
    np.ndarray
        Updated input array using specified thresholding function.

    """
    data = np.asarray(data)

    if kind == 'soft':
        return _soft_threshold(data, thresh, sub)
    elif kind == 'hard':
        return _hard_threshold(data, thresh, sub)
    elif kind == 'soft-percentile':
        return _soft_threshold_perc(data, thresh, sub)
    elif kind == 'hard-percentile':
        return _hard_threshold_perc(data, thresh, sub)
    elif kind in ['garotte', 'garrote']:
        return _nn_garrote(data, thresh, sub)
    elif kind in ['garotte-percentile', 'garrote-percentile']:
        return _nn_garrote_perc(data, thresh, sub)


def threshold_wavelet(data, thresh, sub=0, kind='soft'):
    """
    Apply user-defined threshold to input data (2D).
    Compatible with output from `pywavelet.wavedec2` (multilevel Discrete Wavelet Transform).

    Parameters
    ----------
    data : np.ndarray
        Input data.
    thresh : float, complex
        Threshold cut-off value.
    sub : int, float, optional
        Substitution value (default: `0`).
    kind : str, optional
        Threshold method:
            
          - `soft` (**default**)
          - `garrote`
          - `hard`
          - `soft-percentile`
          - `garrote-percentile`
          - `hard-percentile`

    Returns
    -------
    np.ndarray
        Updated input array using specified thresholding function.

    """
    thresh = [list(d) for d in list(thresh)]
    dlen = len(data[-1])

    if kind == 'soft':
        return [
            [_soft_threshold(data[lvl][d], thresh[lvl][d], sub) for d in range(dlen)]
            for lvl in range(len(data))
        ]
    elif kind == 'hard':
        return [
            [_hard_threshold(data[lvl][d], thresh[lvl][d], sub) for d in range(dlen)]
            for lvl in range(len(data))
        ]
    elif kind == 'soft-percentile':
        return [
            [_soft_threshold_perc(data[lvl][d], thresh[lvl][d], sub) for d in range(dlen)]
            for lvl in range(len(data))
        ]
    elif kind == 'hard-percentile':
        return [
            [_hard_threshold_perc(data[lvl][d], thresh[lvl][d], sub) for d in range(dlen)]
            for lvl in range(len(data))
        ]
    elif kind in ['garotte', 'garrote']:
        return [
            [_nn_garrote(data[lvl][d], thresh[lvl][d], sub) for d in range(dlen)]
            for lvl in range(len(data))
        ]
    elif kind in ['garotte-percentile', 'garrote-percentile']:
        return [
            [_nn_garrote_perc(data[lvl][d], thresh[lvl][d], sub) for d in range(dlen)]
            for lvl in range(len(data))
        ]


def get_threshold_decay(
    thresh_model,
    niter: int,
    transform_kind: str = None,
    p_max: float = 0.99,
    p_min: float = 1e-3,
    x_fwd=None,
    kind: str = 'values',
):
    """
    Calculate iteration-based decay for thresholding function.
    Can be one of the following:
      
      - `values` (based on max value in data)
      - `factors` (for usage as multiplier).

    Parameters
    ----------
    thresh_model : str
        Thresholding decay function.
            
            - `linear`                  Gao et al. (2010)
            - `exponential`             Yang et al. (2012), Zhang et al. (2015), Zhao et al. (2021)
            - `data-driven`             Gao et al. (2013)
            - `inverse_proportional`    Ge et al. (2015)
    niter : int
        Maximum number of iterations.
    transform_kind : str
        Name of the specified transform (e.g. FFT, WAVELET, SHEARLET, CURVELET).
    p_max : float, optional
        Maximum regularization percentage (float).
    p_min : float, str, optional
        Minimum regularization percentage (float) or
        'adaptive': adaptive calculation of minimum threshold according to sparse coefficient.
    x_fwd : np.ndarray, optional
        Forward transformed input data (required for thresh_model=`data-driven` and kind=`values`).
    kind : str, optional
        Return either data `values` or multiplication `factors`.

    Returns
    -------
    tau : np.ndarray
        Array of decay values or factors (based on "kind" paramter).
    
    References
    ----------
    [^1]: Gao, J.-J., Chen, X.-H., Li, J.-Y., Liu, G.-C., & Ma, J. (2010).
        Irregular seismic data reconstruction based on exponential threshold model of POCS method.
        Applied Geophysics, 7(3), 229–238. [https://doi.org/10.1007/s11770-010-0246-5](https://doi.org/10.1007/s11770-010-0246-5)
    [^2]: Yang, P., Gao, J., & Chen, W. (2012).
        Curvelet-based POCS interpolation of nonuniformly sampled seismic records.
        Journal of Applied Geophysics, 79, 90–99. [https://doi.org/10.1016/j.jappgeo.2011.12.004](https://doi.org/10.1016/j.jappgeo.2011.12.004)
    [^3]: Zhang, H., Chen, X., & Li, H. (2015).
        3D seismic data reconstruction based on complex-valued curvelet transform in frequency domain.
        Journal of Applied Geophysics, 113, 64–73. [https://doi.org/10.1016/j.jappgeo.2014.12.004](https://doi.org/10.1016/j.jappgeo.2014.12.004)
    [^4]: Zhao, H., Yang, T., Ni, Y.-D., Liu, X.-G., Xu, Y.-P., Zhang, Y.-L., & Zhang, G.-R. (2021).
        Reconstruction method of irregular seismic data with adaptive thresholds based on different sparse transform bases.
        Applied Geophysics, 18(3), 345–360. [https://doi.org/10.1007/s11770-021-0903-5](https://doi.org/10.1007/s11770-021-0903-5)
    [^5]: Gao, J., Stanton, A., Naghizadeh, M., Sacchi, M. D., & Chen, X. (2013).
        Convergence improvement and noise attenuation considerations for beyond alias projection onto convex sets reconstruction.
        Geophysical Prospecting, 61, 138–151. [https://doi.org/10.1111/j.1365-2478.2012.01103.x](https://doi.org/10.1111/j.1365-2478.2012.01103.x)
    [^6]: Ge, Z.-J., Li, J.-Y., Pan, S.-L., & Chen, X.-H. (2015).
        A fast-convergence POCS seismic denoising and reconstruction method.
        Applied Geophysics, 12(2), 169–178. [https://doi.org/10.1007/s11770-015-0485-1](https://doi.org/10.1007/s11770-015-0485-1)

    """
    TRANSFORMS = ('FFT', 'WAVELET', 'SHEARLET', 'CURVELET', 'DCT')
    if transform_kind is None:
        pass
    elif transform_kind.upper() not in TRANSFORMS and (
        kind == 'values' or thresh_model == 'data-driven'
    ):
        raise ValueError(f'Unsupported transform. Please select one of: {TRANSFORMS}')
    elif transform_kind is not None:
        transform_kind = transform_kind.upper()

    if x_fwd is None and (kind == 'values' or thresh_model == 'data-driven'):
        raise ValueError(
            '`x_fwd` must be specified for thresh_model="data-driven" or kind="values"!'
        )

    # (A) inversely proportional threshold model (Ge et al., 2015)
    if all([s in thresh_model for s in ['inverse', 'proportional']]):
        if transform_kind == 'WAVELET':
            x_fwd_max = np.asarray([[np.abs(d).max() for d in level] for level in x_fwd])
            x_fwd_min = np.asarray([[np.abs(d).min() for d in level] for level in x_fwd])
            _iiter = np.arange(1, niter + 1)[:, None, None]
        elif transform_kind == 'SHEARLET':
            x_fwd_max = np.max(np.abs(x_fwd), axis=(0, 1))
            x_fwd_min = np.min(np.abs(x_fwd), axis=(0, 1))
            _iiter = np.arange(1, niter + 1)[:, None]
        elif transform_kind in ['FFT', 'CURVELET', 'DCT']:
            x_fwd_max = np.abs(x_fwd).max()
            x_fwd_min = np.abs(x_fwd).min()
            _iiter = np.arange(1, niter + 1)

        # arbitrary variable to adjust descent rate (most cases: 1 <= q <=3)
        q = thresh_model.split('-')[-1] if '-' in thresh_model else 1.0
        try:
            q = float(q)
        except:  # noqa
            q = 1.0

        a = (niter**q * (x_fwd_max - x_fwd_min)) / (niter**q - 1)
        b = (niter**q * x_fwd_min - x_fwd_max) / (niter**q - 1)
        return a / (_iiter**q) + b

    # (B) "classic" thresholding models
    if kind == 'values':
        # max (absolute) value in forward transformed data
        if transform_kind == 'WAVELET':
            # x_fwd_max = np.asarray([[np.abs(d).max() for d in level] for level in x_fwd])
            x_fwd_max = np.asarray([[d.max() for d in level] for level in x_fwd])
        elif transform_kind == 'SHEARLET':
            axis = (0, 1)
            # x_fwd_max = np.max(np.abs(x_fwd), axis=axis)
            x_fwd_max = np.max(x_fwd, axis=axis)
        elif transform_kind in ['FFT', 'CURVELET', 'DCT']:
            axis = None
            x_fwd_max = x_fwd.max()
        else:
            raise ValueError(
                '`transform_kind` must be specified for thresh_model="data-driven" or kind="values"!'
            )

        # min/max regularization factors
        #   adaptive calculation of minimum threshold (Zhao et al., 2021)
        if isinstance(p_min, str) and p_min == 'adaptive':
            # single-scale transform
            if transform_kind in ['FFT', 'DCT']:
                tau_min = 0.01 * np.sqrt(np.linalg.norm(x_fwd, axis=axis) ** 2 / x_fwd.size)

            # mulit-scale transform
            elif transform_kind in ['SHEARLET']:
                # calculate regularization factor `tau_min` for each scale
                nscales = get_number_scales(x_fwd)
                j = np.hstack(
                    (
                        np.array([0]),  # low-pass solution
                        np.repeat(
                            np.arange(1, nscales + 1), [2 ** (j + 2) for j in range(nscales)]
                        ),
                    )
                )
                tau_min = (
                    1
                    / 3
                    * np.median(  # noqa
                        np.log10(j + 1)
                        * np.sqrt(np.linalg.norm(x_fwd, axis=axis) ** 2 / x_fwd.size)
                    )
                )
            else:
                raise NotImplementedError(
                    f'p_min=`adaptive` is not implemented for {transform_kind} transform'
                )
        else:
            tau_min = p_min * x_fwd_max
        tau_max = p_max * x_fwd_max

    elif kind == 'factors':
        tau_max = p_max
        tau_min = p_min
    else:
        raise ValueError('Parameter `kind` only supports arguments "values" or "factors"')

    # --- iteration-based threshold factor ---
    _iiter = np.arange(1, niter + 1)

    if transform_kind == 'WAVELET':
        imultiplier = ((_iiter - 1) / (niter - 1))[:, None, None]
    elif transform_kind == 'SHEARLET':
        imultiplier = ((_iiter - 1) / (niter - 1))[:, None]
    elif transform_kind in ['FFT', 'CURVELET', 'DCT']:
        imultiplier = (_iiter - 1) / (niter - 1)
    elif transform_kind is None:
        imultiplier = (_iiter - 1) / (niter - 1)

    # --- thresholding operator ---
    if thresh_model == 'linear':
        tau = tau_max - (tau_max - tau_min) * imultiplier

    elif 'exponential' in thresh_model:
        q = float(thresh_model.split('-')[-1]) if '-' in thresh_model else 1.0  # Zhao et al. (2021)
        c = np.log(tau_min / tau_max)
        tau = tau_max * np.exp(c * imultiplier**q)

    elif thresh_model == 'data-driven' and transform_kind in ['FFT', 'DCT', 'CURVELET']:
        tau = np.zeros((_iiter.size,), dtype=x_fwd.dtype)
        idx = (x_fwd > tau_min) & (x_fwd < tau_max)
        v = np.sort(x_fwd[idx])[::-1]
        Nv = v.size
        tau[0] = v[0]
        tau[1:] = v[np.ceil((_iiter[1:] - 1) * (Nv - 1) / (niter - 1)).astype('int')]
    else:
        raise NotImplementedError(
            f'{thresh_model} is not implemented for {transform_kind} transform!'
        )

    return tau


def POCS_algorithm(
    x,
    mask,
    auxiliary_data=None,
    transform=None,
    itransform=None,
    transform_kind: str = None,
    niter: int = 50,
    thresh_op: str = 'hard',
    thresh_model: str = 'exponential',
    eps: float = 1e-9,
    alpha: int = 1.0,
    p_max: float = 0.99,
    p_min: float = 1e-5,
    sqrt_decay: str = False,
    decay_kind: str = 'values',
    verbose: bool = False,
    version: str = 'regular',
    results_dict: dict = None,
    path_results: str = None,
):
    """
    Interpolate sparse input grid using Point Onto Convex Sets (POCS) algorithm.
    Applying a user-specified **transform** method:
        
      - `FFT`
      - `Wavelet`
      - `Shearlet`
      - `Curvelet`

    Parameters
    ----------
    x : np.ndarray
        Sparse input data (2D).
    mask : np.ndarray
        Boolean mask of input data (`1`: data cell, `0`: nodata cell).
    auxiliary_data: np.ndarray
        Auxiliary data only required by `shearlet` transform.
    transform : callable
        Forward transform to apply.
    itransform : callable
        Inverse transform to apply.
    transform_kind : str
        Name of the specified transform.
    niter : int, optional
        Maximum number of iterations (default: `50`).
    thresh_op : str, optional
        Threshold operator (default: `soft`).
    thresh_model : str, optional
        Thresholding decay function.
            
            - `linear`                   Gao et al. (2010)
            - `exponential`              Yang et al. (2012), Zhang et al. (2015), Zhao et al. (2021)
            - `data-driven`              Gao et al. (2013)
            - `inverse_proportional`     Ge et al. (2015)
    eps : float, optional
        Covergence threshold (default: `1e-9`).
    alpha : float, optional
        Weighting factor to scale re-insertion of input data (default: `1.0`).
    sqrt_decay : bool, optional
        Use squared decay values for thresholding (default: `False`).
    decay_kind : str, optional
        Return either data "values" or multiplication "factors".
    verbose : bool, optional
        Print information about iteration steps (default: `False`).
    version : str, optional
        Version of POCS algorithm. One of the following:
            
            - `regular`     Abma and Kabir (2006), Yang et al. (2012)
            - `fast`        Yang et al. (2013), Gan et al (2015)
            - `adaptive`    Wang et al. (2015, 2016)
    results_dict : dict, optional
        If provided: return dict with total iterations, runtime (in seconds) and cost function.

    Returns
    -------
    x_inv : np.ndarray
        Reconstructed (i.e. interpolated) input data.

    References
    ----------
    [^1]: Gao, J.-J., Chen, X.-H., Li, J.-Y., Liu, G.-C., & Ma, J. (2010).
        Irregular seismic data reconstruction based on exponential threshold model of POCS method.
        Applied Geophysics, 7(3), 229–238. [https://doi.org/10.1007/s11770-010-0246-5](https://doi.org/10.1007/s11770-010-0246-5)
    [^2]: Yang, P., Gao, J., & Chen, W. (2012).
        Curvelet-based POCS interpolation of nonuniformly sampled seismic records.
        Journal of Applied Geophysics, 79, 90–99. [https://doi.org/10.1016/j.jappgeo.2011.12.004](https://doi.org/10.1016/j.jappgeo.2011.12.004)
    [^3]: Zhang, H., Chen, X., & Li, H. (2015).
        3D seismic data reconstruction based on complex-valued curvelet transform in frequency domain.
        Journal of Applied Geophysics, 113, 64–73. [https://doi.org/10.1016/j.jappgeo.2014.12.004](https://doi.org/10.1016/j.jappgeo.2014.12.004)
    [^4]: Zhao, H., Yang, T., Ni, Y.-D., Liu, X.-G., Xu, Y.-P., Zhang, Y.-L., & Zhang, G.-R. (2021).
        Reconstruction method of irregular seismic data with adaptive thresholds based on different sparse transform bases.
        Applied Geophysics, 18(3), 345–360. [https://doi.org/10.1007/s11770-021-0903-5](https://doi.org/10.1007/s11770-021-0903-5)
    [^5]: Gao, J., Stanton, A., Naghizadeh, M., Sacchi, M. D., & Chen, X. (2013).
        Convergence improvement and noise attenuation considerations for beyond alias projection onto convex sets reconstruction.
        Geophysical Prospecting, 61, 138–151. [https://doi.org/10.1111/j.1365-2478.2012.01103.x](https://doi.org/10.1111/j.1365-2478.2012.01103.x)
    [^6]: Ge, Z.-J., Li, J.-Y., Pan, S.-L., & Chen, X.-H. (2015).
        A fast-convergence POCS seismic denoising and reconstruction method.
        Applied Geophysics, 12(2), 169–178. [https://doi.org/10.1007/s11770-015-0485-1](https://doi.org/10.1007/s11770-015-0485-1)
    [^7]: Abma, R., & Kabir, N. (2006). 3D interpolation of irregular data with a POCS algorithm.
        Geophysics, 71(6), E91–E97. [https://doi.org/10.1190/1.2356088](https://doi.org/10.1190/1.2356088)
    [^8]: Yang, P., Gao, J., & Chen, W. (2013)
        On analysis-based two-step interpolation methods for randomly sampled seismic data.
        Computers & Geosciences, 51, 449–461. [https://doi.org/10.1016/j.cageo.2012.07.023](https://doi.org/10.1016/j.cageo.2012.07.023)
    [^9]: Gan, S., Wang, S., Chen, Y., Zhang, Y., & Jin, Z. (2015).
        Dealiased Seismic Data Interpolation Using Seislet Transform With Low-Frequency Constraint.
        IEEE Geoscience and Remote Sensing Letters, 12(10), 2150–2154. [https://doi.org/10.1109/LGRS.2015.2453119](https://doi.org/10.1109/LGRS.2015.2453119)
    [^10]:  Wang, B., Wu, R.-S., Chen, X., & Li, J. (2015).
        Simultaneous seismic data interpolation and denoising with a new adaptive method based on dreamlet transform.
        Geophysical Journal International, 201(2), 1182–1194. [https://doi.org/10.1093/gji/ggv072](https://doi.org/10.1093/gji/ggv072)
    [^11]: Wang, B., Chen, X., Li, J., & Cao, J. (2016).
        An Improved Weighted Projection Onto Convex Sets Method for Seismic Data Interpolation and Denoising.
        IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 9(1), 228–235.
        [https://doi.org/10.1109/jstars.2015.2496374](https://doi.org/10.1109/jstars.2015.2496374)

    """
    # sanity checks
    if np.max(mask) > 1:
        raise ValueError(f'mask should be quasi-boolean (0 or 1) but has maximum of {np.max(mask)}')

    if any(v is None for v in [transform, itransform]):
        raise ValueError('Forward and inverse transform function have to be supplied')

    TRANSFORMS = ('FFT', 'WAVELET', 'SHEARLET', 'CURVELET', 'DCT')
    if transform_kind.upper() not in TRANSFORMS:
        raise ValueError(f'Unsupported transform. Please select one of: {TRANSFORMS}')
    else:
        transform_kind = transform_kind.upper()

    if transform_kind == 'SHEARLET' and auxiliary_data is None:
        raise ValueError(
            f'{transform_kind} requires pre-computed shearlets in Fourier domain (Psi)'
        )
    
    niter = int(niter)
    eps = float(eps)
    p_max = float(p_max)
    alpha = float(alpha)

    # get input paramter
    is_complex_input = np.iscomplexobj(x)
    shape = x.shape
    original_shape = tuple(slice(s) for s in shape)

    if np.count_nonzero(x) == 0:
        niterations = 0
        runtime = 0
        cost = 0
        costs = [0]

        x_inv = x
    else:
        # initial forward transform
        if transform_kind == 'WAVELET':  # and isinstance(x_fwd, list):
            x_fwd = transform(x)[1:]  # exclude low-pass filter
        elif transform_kind == 'SHEARLET':  # and isinstance(x_fwd, tuple):
            x_fwd = transform(x, Psi=auxiliary_data)  # [0]   # output is like (ST, Psi)
        elif (
            transform_kind == 'CURVELET'
            and hasattr(transform, '__name__')
            and transform.__name__ == 'matvec'
        ):
            x_fwd = transform(x.ravel())
        else:
            x_fwd = transform(x)

        # get threshold decay array
        decay = get_threshold_decay(
            thresh_model=thresh_model,
            niter=niter,
            transform_kind=transform_kind,
            p_max=p_max,
            p_min=p_min,
            x_fwd=x_fwd,
            kind=decay_kind,
        )

        # init data variables
        x_old = x
        x_inv = x

        # init variable for improved convergence (Yang et al., 2013)
        if version == 'fast':
            v = 1

        t0 = time.perf_counter()
        if path_results is not None:
            costs = []

        for iiter in range(niter):
            if verbose:
                print(f'[Iteration: <{iiter+1:3d}>]')

            if version == 'regular':
                x_input = x_old
            elif version == 'fast':  # Yang et al. (2013)
                # improved convergence
                v1 = (1 + np.sqrt(1 + 4 * v**2)) / 2
                frac = (v - 1) / (v1 + 1)  # Gan et al. (2015)
                v = v1
                x_input = x_inv + frac * (x_inv - x_old)  # prediction
            elif version == 'adaptive':  # Wang et al. (2015, 2016)
                # init adaptive input data
                x_tmp = alpha * x + (1 - alpha * mask) * x_old
                x_input = x_tmp + (1 - alpha) * (x - mask * x_old)
                # x_input = x_inv + (1 - alpha) * (x - mask * x_old)

            # (1) forward transform
            if (
                transform_kind == 'CURVELET'
                and hasattr(transform, '__name__')
                and transform.__name__ == 'matvec'
            ):
                X = transform(x_input.ravel())
            elif transform_kind == 'WAVELET':
                X = transform(x_input)
                lowpass = X[0].copy()
                X = X[1:]
            elif transform_kind == 'SHEARLET':
                X = transform(x_input, Psi=auxiliary_data)
            else:
                X = transform(x_input)

            # (2) thresholding
            _decay = np.sqrt(decay[iiter]) if sqrt_decay else decay[iiter]
            if transform_kind == 'WAVELET' and isinstance(X, list):
                X_thresh = threshold_wavelet(X, _decay, kind=thresh_op)
            else:
                X_thresh = threshold(X, _decay, kind=thresh_op)

            # (3) inverse transform
            if (
                transform_kind == 'CURVELET'
                and hasattr(itransform, '__name__')
                and itransform.__name__ == 'rmatvec'
            ):
                x_inv = itransform(X_thresh).reshape(shape)
            elif transform_kind == 'WAVELET':
                x_inv = itransform([lowpass] + X_thresh)[original_shape]
            elif transform_kind == 'SHEARLET':
                x_inv = itransform(X_thresh, Psi=auxiliary_data)
            else:
                x_inv = itransform(X_thresh)

            # (4) apply mask (scaled by weighting factor)
            x_inv *= 1 - alpha * mask

            # (5) add original data (scaled by weighting factor)
            x_inv += x * alpha

            # cost function from Gao et al. (2013)
            cost = np.sum(np.abs(x_inv) - np.abs(x_old)) ** 2 / np.sum(np.abs(x_inv)) ** 2
            if path_results is not None:
                costs.append(cost)
            if verbose:
                print('[INFO]   cost:', cost)

            # set result from previous iteration as new input
            x_old = x_inv

            if iiter > 2 and cost < eps:
                break

        niterations = iiter + 1
        runtime = time.perf_counter() - t0

    if verbose:
        print('\n' + '-' * 20)
        print(f'# iterations:  {niterations:4d}')
        print(f'cost function: {cost}')
        print(f'runtime:       {runtime:.3f} s')
        print('-' * 20)

    if isinstance(results_dict, dict):
        results_dict['niterations'] = niterations
        results_dict['runtime'] = round(runtime, 3)
        results_dict['cost'] = cost

    if path_results is not None:
        with open(path_results, mode="a", newline='\n') as f:
            f.write(';'.join([str(i) for i in [niterations, runtime] + costs]) + '\n')

    if is_complex_input:
        return x_inv

    return np.real(x_inv)


POCS = partial(POCS_algorithm, version='regular')
FPOCS = partial(POCS_algorithm, version='fast')
APOCS = partial(POCS_algorithm, version='adaptive')

# =================================================================================================
#                                           PLOTTING
# =================================================================================================
def _plot_inversion_xr(
    x,  # xr.DataArray
    x_inv,  # xr.DataArray
    x_inv_filt=None,  # xr.DataArray
    cmap=None,
    title=None,
    metadata=None,
    plot_abs=False,
    subplot_kwargs=None,
    cbar_kwargs=None,
):
    IS_COMPLEX = any(np.iscomplexobj(a) for a in [x.data, x_inv.data])

    ncols = 2 if x_inv_filt is None else 3
    nrows = 2 if IS_COMPLEX else 1

    if subplot_kwargs is None:
        subplot_kwargs = dict(figsize=(6 * ncols, 8), sharex=True, sharey=True)
    if cbar_kwargs is None:
        cbar_kwargs = dict(fraction=0.05, aspect=50, pad=0.02)

    fig, ax = plt.subplots(nrows, ncols, **subplot_kwargs)

    # sparse input data
    if IS_COMPLEX:
        vmax = np.percentile(np.abs(x), 99)
        vmin = -vmax
        cmap = 'RdBu'

        x.real.T.plot(ax=ax[0, 0], cmap=cmap, vmin=vmin, vmax=vmax, cbar_kwargs=cbar_kwargs)
        x.imag.T.plot(ax=ax[1, 0], cmap=cmap, vmin=vmin, vmax=vmax, cbar_kwargs=cbar_kwargs)
        coord = [c for c in list(x.coords) if c not in ['iline', 'x', 'xline', 'y']][0]
        if x[coord].attrs is not None and x[coord].attrs != {}:
            title_info = (
                f'| {x[coord].attrs["long_name"]}: {float(x[coord]):.3f} {x[coord].attrs["units"]}'
            )
        else:
            title_info = f'| {coord}: {float(x[coord]):.3f}'
        ax[0, 0].set_title(f'sparse input data (real) {title_info}')
        ax[1, 0].set_title(f'sparse input data (imag) {title_info}')

        # reconstructed data
        x_inv.real.T.plot(ax=ax[0, 1], cmap=cmap, vmin=vmin, vmax=vmax, cbar_kwargs=cbar_kwargs)
        x_inv.imag.T.plot(ax=ax[1, 1], cmap=cmap, vmin=vmin, vmax=vmax, cbar_kwargs=cbar_kwargs)
        ax[0, 1].set_title('reconstructed data (real)')
        ax[1, 1].set_title('reconstructed data (imag)')

        if x_inv_filt is not None:
            x_inv_filt.real.T.plot(
                ax=ax[0, 2], cmap=cmap, vmin=vmin, vmax=vmax, cbar_kwargs=cbar_kwargs
            )
            x_inv_filt.imag.T.plot(
                ax=ax[1, 2], cmap=cmap, vmin=vmin, vmax=vmax, cbar_kwargs=cbar_kwargs
            )
            ax[0, 2].set_title('reconstructed data (real, filtered)')
            ax[1, 2].set_title('reconstructed data (imag, filtered)')
    else:
        ax = ax.ravel()
        vmax = np.percentile(np.abs(x), 99.9)
        vmin = 0 if np.min(x) >= 0 else -vmax
        if cmap is None:
            cmap = 'cividis' if vmin == 0 else 'RdBu'

        x.T.plot(ax=ax[0], cmap=cmap, vmin=vmin, vmax=vmax, cbar_kwargs=cbar_kwargs)
        coord = [c for c in list(x.coords) if c not in ['iline', 'x', 'xline', 'y']][0]
        if x[coord].attrs is not None and x[coord].attrs != {}:
            title_info = (
                f'| {x[coord].attrs["long_name"]}: {float(x[coord]):.3f} {x[coord].attrs["units"]}'
            )
        else:
            title_info = f'| {coord}: {float(x[coord]):.3f}'
        ax[0].set_title(f'sparse input data {title_info}')

        # reconstructed data
        x_inv.T.plot(ax=ax[1], cmap=cmap, vmin=vmin, vmax=vmax, cbar_kwargs=cbar_kwargs)
        ax[1].set_title('reconstructed data')

        if x_inv_filt is not None:
            x_inv_filt.T.plot(ax=ax[2], cmap=cmap, vmin=vmin, vmax=vmax, cbar_kwargs=cbar_kwargs)
            ax[2].set_title('reconstructed data (filtered)')

    if title is not None:
        fig.suptitle(title)
    elif metadata is not None and metadata is not {}:
        title = (
            f'{metadata["transform_kind"]} transform | {metadata["version"]}'
            + f' (iterations: {metadata["niterations"]}/{metadata["niter"]}) '
            + f'| {metadata["runtime"]:.1f} s\n'
            + f'eps={metadata["eps"]}, kind="{metadata["thresh_op"]}", '
            + f'thresh="{metadata["thresh_model"]}", alpha={metadata["alpha"]}, '
            + f'p={metadata["p_max"]}/{metadata["p_min"]}, sqrt={metadata["sqrt_decay"]}'
            + '\n'
            + f'noise level: {metadata["noise_lvl"]:.5f} dB (Immerkær) | '
            + f'{metadata["noise_lvl_IEEE"]:.5f} dB (Chen et al.)'
        )

    fig.tight_layout(pad=1)

    return fig, ax
