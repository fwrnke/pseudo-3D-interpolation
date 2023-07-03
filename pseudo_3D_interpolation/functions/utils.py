"""Miscellaneous utility functions."""

import re
import time
import pstats
import cProfile
from functools import wraps, partial
from contextlib import contextmanager
from math import ceil, floor

import numpy as np
import dask

#%%
# *****************************************************************************
#                               CONSTANTS
# *****************************************************************************
TRACE_HEADER_COORDS = {
    'source': (73, 77),  # SourceX, SourceY
    'CDP': (181, 185),   # CDP_X, CDP_Y
    'group': (81, 85),   # GroupX, GroupY
}

#%%
# *****************************************************************************
#                           SMALL HELPERS
# *****************************************************************************
ffloat = partial(np.format_float_positional, trim='-')

def get_array_module(a):
    """Return appropriate array module."""
    if hasattr(a, 'chunks'):
        return dask.array
    else:
        return np

def round_to_multiple(x, multiple=10, method='nearest'):
    """Round value `x` to next `multiple` using `method`."""
    dtype = type(x)
    if method == 'nearest':
        return dtype(round(x / multiple) * multiple)
    if method == 'up':
        return dtype(ceil(x / multiple) * multiple)
    if method == 'down':
        return dtype(floor(x / multiple) * multiple)

round_up = partial(round_to_multiple, method='up')
round_down = partial(round_to_multiple, method='down')


def log_message(msg_lvl: int, verbosity: int, *msg_args) -> None:
    """Print log messages depending on level of verbosity."""
    if msg_lvl <= verbosity:
        print(*msg_args)


def xprint(*args, kind: str = 'info', verbosity: int = 0, **kwargs) -> None:
    """Thin wrapper function for build-in print() to add informative prefix and color-coded print statements."""
    verbosity = 1 if verbosity is True else verbosity
    prefixes = {
        'info': ('\033[39m', '[INFO]  ', 1),
        'warning': ('\033[33m\033[1m', '[WARN]  ', 0),
        'error': ('\033[31m\033[1m', '[ERROR]  ', 0),
        'success': ('\033[32m', '[SUCCESS]  ', 1),
        'debug': ('\033[36m', '[DEBUG]  ', 2),
    }
    prefix = prefixes.get(kind, None)
    if prefix is None:
        args = args
        verbosity_lvl = 1
    else:
        color, prefix_, verbosity_lvl = prefix
        args = [f'{color}{prefix_}'] + ['{arg}'.format(arg=i) for i in args] + ['\033[0m']

    if verbosity_lvl <= verbosity:
        print(*args, **kwargs)
        

def clean_log_file(path_log: str, newline: str = '\n') -> None:
    """Remove colored terminal output characters from ASCII log."""
    with open(path_log, 'r') as f:
        log = f.readlines()
        log_cleanded = [re.sub(r'\x1b\[[0-9;]*m', '', line) for line in log]  # noqa

    with open(path_log, 'w', newline=newline) as f:
        f.writelines(log_cleanded)


def timeit(func):
    """Decorate function to measure its runtime."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        results = func(*args, **kwargs)
        end_time = time.perf_counter() - start_time
        print(f'[RUNTIME]    {func.__name__}(): {int(end_time//60):d} min {(end_time%60):.4f} sec')
        return results

    return wrapper


def profile(output_file=None, sort_by='cumulative', lines_to_print=None, strip_dirs=False):
    """
    Profile function.
    
    Inspired by and modified the profile decorator of Giampaolo Rodola.
    Copied code by Ehsan Khodabandeh from `towardsdatascience.com`.
        
    
    Parameters
    ----------
    output_file : str or None
        Path of the output file. If only name of the file is given, it's saved in the current directory.
        If it's None, the name of the decorated function is used (default: `None`).
    sort_by : str or SortKey enum or tuple/list of str/SortKey enum
        Sorting criteria for the Stats object.
        For a list of valid string and SortKey refer to:
        https://docs.python.org/3/library/profile.html#pstats.Stats.sort_stats
    lines_to_print : int or None
        Number of lines to print. Default (None) is for all the lines.
        This is useful in reducing the size of the printout, especially
        that sorting by 'cumulative', the time consuming operations
        are printed toward the top of the file.
    strip_dirs : bool
        Whether to remove the leading path info from file names.
        This is also useful in reducing the size of the printout
    
    Returns
    -------
    func
        Profile of the decorated function
    
    References
    ----------
    [^1]: Giampaolo Rodola, [http://code.activestate.com/recipes/577817-profile-decorator/](http://code.activestate.com/recipes/577817-profile-decorator/)
    [^2]: Ehsan Khodabandeh, [https://towardsdatascience.com/how-to-profile-your-code-in-python-e70c834fad89](https://towardsdatascience.com/how-to-profile-your-code-in-python-e70c834fad89)
        
    """
    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _output_file = output_file or func.__name__ + '.prof'
            pr = cProfile.Profile()
            pr.enable()
            retval = func(*args, **kwargs)
            pr.disable()
            pr.dump_stats(_output_file)

            with open(_output_file, 'w') as f:
                ps = pstats.Stats(pr, stream=f)
                if strip_dirs:
                    ps.strip_dirs()
                if isinstance(sort_by, (tuple, list)):
                    ps.sort_stats(*sort_by)
                else:
                    ps.sort_stats(sort_by)
                ps.print_stats(lines_to_print)
            return retval

        return wrapper

    return inner


def debug(func):
    """Decorate function to print debugging information."""
    @wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]  # 1
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]  # 2
        signature = ", ".join(args_repr + kwargs_repr)  # 3
        print(f"Call {func.__name__}({signature})")
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} return {value!r}")  # 4
        return value

    return wrapper_debug


@contextmanager
def show_progressbar(progressbar, verbose=False):
    """Wrap `dask` progress bar."""
    if verbose:
        with progressbar:
            yield
    else:
        yield
    
    
# *****************************************************************************
#                           MISC: ARRAY
# *****************************************************************************
def pad_array(a, n: int, zeros=False):
    """
    Pad 1D input array with `n` elements at start and end (mirror of array).

    Parameters
    ----------
    a : np.ndarray
        1D input array.
    n : int
        Number of elements to add at start and end.
    zeros : bool, optional
        Add zeros instead of mirrored values (default: `False`).

    Returns
    -------
    np.ndarray
        Padded input array.

    """
    if zeros:
        return np.concatenate((np.zeros(n), a, np.zeros(n)))
    else:
        # # mirror array
        # pad_start = a[0] + np.abs(a[1:n+1][::-1] - a[0])
        # pad_end = a[-1] + np.abs(a[-n-1:-1][::-1] - a[-1])

        # mirror array AND flip upside down
        pad_start = a[0] - np.abs(a[1 : n + 1][::-1] - a[0])
        pad_end = a[-1] - np.abs(a[-n - 1 : -1][::-1] - a[-1])
        return np.concatenate((pad_start, a, pad_end))


def pad_along_axis(array, n: int, mode: str = 'constant', kwargs: dict = None, axis: int = -1):
    """
    Pad 2D array along given axis (default: `-1`).

    Parameters
    ----------
    array : np.ndarray
        Input data.
    n : int
        Number of values padded to the edges of specified axis.
    mode : str, optional
        How to pad array edges (see `np.pad`, default: `constant`).
    kwargs : dict, optional
        OPtional keyword arguments for `np.pad` (default: `None`).
    axis : int, optional
        Axis to pad (default: `-1`).

    Returns
    -------
    np.ndarray
        Padded input array.

    """
    array = np.asarray(array)

    if n <= 0:
        return array

    if isinstance(n, (int, float)):
        n_before = n_after = int(n)
    elif isinstance(n, (tuple, list)):
        n_before, n_after = n

    if n_before == 0 and n_after == 0:
        return array

    npad = [(0, 0)] * array.ndim
    npad[axis] = (n_before, n_after)
    print(npad)

    if kwargs is None:
        kwargs = dict(constant_values=0)

    return np.pad(array, pad_width=npad, mode=mode, **kwargs)


def slice_valid_data(data, nso):
    """
    Account for zero padded input data and return only valid data samples (!= 0).

    Parameters
    ----------
    data : np.ndarray
        Seismic section (samples x traces).
    nso : int
        Original number samples per trace (from binary header).

    Returns
    -------
    np.ndarray
        "Unpadded" seismic section with only valid (non-zero) samples (2D).
    idx_start_slice : np.ndarray
        Array of starting indices for valid data slice per trace.

    """
    # get indices of first valid sample
    # = 0: trace was padded at bottom
    # > 0: trace was padded at top
    idx_start_slice = (data != 0).argmax(axis=0)
    # create index array
    indexer = np.transpose(np.arange(nso) + idx_start_slice[:, None])
    # return sliced traces and indices
    return np.take_along_axis(data, indexer, axis=0), idx_start_slice


# *****************************************************************************
#                           CONVERSIONS
# *****************************************************************************
def depth2twt(depth, v=1500):
    """Convert depth (m) to two-way travel time (TWT in sec)."""
    return depth / (v / 2)


def twt2depth(twt, v=1500, units='s'):
    """Convert two-way travel time (TWT in sec) to depth (m)."""
    if units == 's':
        return (v / 2) * twt
    elif units == 'ms':
        return (v / 2) * (twt / 1000)
    elif units == 'ns':
        return (v / 2) * (twt / 1e-06)


def twt2samples(twt, dt: float, units='s'):
    """Convert TWT (sec) to samples (#) based on sampling interval `dt` (sec)."""
    if units == 's':
        pass
    elif units == 'ms':
        dt = dt / 1000
    elif units == 'ns':
        dt = dt / 1e-6

    return twt / dt


def samples2twt(samples, dt: float):
    """Convert samples (#) to TWT (in dt units!) based on sampling interval `dt`."""
    return samples * dt


def depth2samples(depth, dt: float, v=1500, units='s'):
    """Convert depth (m) to samples (#) given a sampling interval `dt` and acoustic velocity."""
    _twt = depth2twt(depth, v=v)

    if units == 's':
        pass
    elif units == 'ms':
        dt = dt / 1000
    elif units == 'ns':
        dt = dt / 1e-6

    return twt2samples(_twt, dt=dt)


def samples2depth(samples, dt: float, v=1500, units='s'):
    """Convert samples (#) to depth (m) given a sampling interval `dt` and acoustic velocity."""
    if units == 's':
        pass
    elif units == 'ms':
        dt = dt / 1000
    elif units == 'ns':
        dt = dt / 1e-6
    else:
        raise ValueError(f'Unit "{units}" is not supported for `dt`.')

    _twt = samples2twt(samples, dt=dt)

    return twt2depth(_twt, v=v)


def convert_twt(twt, unit_in: str, unit_out: str):
    """
    Convert TWT unit.

    Parameters
    ----------
    twt : int, float, np.ndarray
        Input TWT value(s).
    unit_in : str
        Input time unit, one of ['s', 'ms', 'us', 'ns'].
    unit_out : str
        Output time unit, one of ['s', 'ms', 'us', 'ns'].

    Returns
    -------
    int, float, np.ndarray
        Converted TWT value(s).

    """
    UNITS = {
        's': 1,
        'ms': 1e-3,
        'us': 1e-6,
        'ns': 1e-9,
    }
    if unit_in not in UNITS:
        raise ValueError(f'Input unit `{unit_in}` is not supported. Choose one of {UNITS.keys()}')
    if unit_out not in UNITS:
        raise ValueError(f'Output unit `{unit_out}` is not supported. Choose one of {UNITS.keys()}')
    
    fact_in = UNITS.get(unit_in)
    fact_out = UNITS.get(unit_out)
    factor = fact_in / fact_out if fact_in > fact_out else fact_in * fact_out

    return twt * factor


def euclidean_distance(coords):
    """Calculate euclidean distance between consecutive points in array (row-wise)."""
    diff = np.diff(coords, axis=0)
    dist = np.sqrt((diff**2).sum(axis=1))
    return dist


# *****************************************************************************
#                           miscellaneous
# *****************************************************************************
def rescale(a, vmin=0, vmax=1):
    """
    Rescale array to given range (default: [0, 1]).

    Parameters
    ----------
    a : np.ndarray
        Input array to rescale/normalize.
    vmin : float, optional
        New minimum value (default: `0`).
    vmax : float, optional
        New maximum value (default: `1`).

    Returns
    -------
    np.ndarray
        Rescaled input array.

    """
    a = np.asarray(a)
    amin = np.nanmin(a)
    amax = np.nanmax(a)

    vmin = amin if vmin is None else vmin
    vmax = amax if vmax is None else vmax

    if amin == amax:
        return a
    return vmin + (a - amin) * ((vmax - vmin) / (amax - amin))


def rescale_dask(a, vmin=0, vmax=1, amin=None, amax=None):
    """
    Rescale array to given range (default: [0, 1]).

    Parameters
    ----------
    a : np.ndarray
        Input array to rescale/normalize.
    vmin : float, optional
        New minimum value (default: `0`).
    vmax : float, optional
        New maximum value (default: `1`).
    amin : float, optional
        Minimum value (default: `None`).
    amax : float, optional
        New maximum value (default: `None`).

    Returns
    -------
    np.ndarray
        Rescaled input array.

    """
    a = np.asarray(a)
    amin = np.nanmin(a) if amin is None else amin
    amax = np.nanmax(a) if amax is None else amax

    if amin == amax:
        return a
    return vmin + (a - amin) * ((vmax - vmin) / (amax - amin))
