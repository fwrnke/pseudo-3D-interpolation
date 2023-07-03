"""
Compensate mistie for SEG-Y file(s) via cross-correlation of nearest traces of intersecting lines.

"""
import os
import sys
import glob
import argparse
import datetime
from shutil import copy2
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import segyio
from tqdm import tqdm

from scipy.signal import correlate as scipy_correlate
from scipy.stats import pearsonr as scipy_pearsonr

from pseudo_3D_interpolation.functions.utils import samples2twt, rescale, timeit, xprint, clean_log_file
from pseudo_3D_interpolation.functions.utils_io import (
    read_auxiliary_files,
    extract_navigation_from_segy,
)
from pseudo_3D_interpolation.functions.signal import envelope
from pseudo_3D_interpolation.functions.header import (
    get_textual_header,
    add_processing_info_header,
    write_textual_header,
)
from pseudo_3D_interpolation.functions.backends import numba_enabled, geopandas_enabled
if numba_enabled:
    from numba import jit
else:
    def jit(*args, **kwargs):  # noqa
        """Dummy numba.jit function."""
        return lambda f: f

#%% DEFAULTS
GEOMETRY = dict([(i.name, i.value) for i in shapely.GeometryType])

#%% FUNCTIONS
# fmt: off
def define_input_args():  # noqa
    parser = argparse.ArgumentParser(
        description='Compensate mistie for SEG-Y file(s) via cross-correlation of nearest traces of intersecting lines.')
    parser.add_argument('input_path', type=str, help='Input datalist or directory.')
    parser.add_argument('--output_dir', '-o', type=str,
                        help='Output directory for corrected SEG-Y file(s).')
    parser.add_argument('--inplace', '-i', action='store_true',
                        help='Edit SEG-Y file(s) inplace.')
    parser.add_argument('--filename_suffix', '-fns', type=str,
                        help='Filename suffix for guided selection (e.g. "env" or "despk"). Only used when "input_path" is a directory.')
    parser.add_argument('--suffix', '-s', type=str, default='sgy',
                        help='File suffix. Only used when "input_path" is a directory.')
    parser.add_argument('--txt_suffix', type=str, help='Additional text to append to output filename.')
    #
    parser.add_argument('--coords_origin', choices=['header', 'aux'], default='header',
                        help='Origin of (shotpoint) coordinates (i.e. navigation).')
    parser.add_argument('--coords_path', type=str, required=True,
                        help='Path to SEG-Y directory (coords_origin=header) or navigation file with coordinates (coords_origin=aux).')
    parser.add_argument('--coords_fsuffix', type=str,
                        help='File suffix of auxiliary or SEG-Y files (depending on chosen parameter for `coords_origin`.')
    parser.add_argument('--coords_text_suffix', type=str,
                        help='Filename text suffix to filter auxiliary or SEG-Y files.')
    #
    parser.add_argument('--win_cc', nargs='*',
                        help='Upper/lower trace window limits used for cross-correlation (in ms).')
    parser.add_argument('--quality_threshold', type=float, default=0.5,
                        help='Cut-off threshold for cross-correlation [0-1].')
    parser.add_argument('--write_aux', action='store_true',
                        help='Write mistie offsets to auxiliary file (*.mst).')
    parser.add_argument('--write_QC', action='store_true',
                        help='Write line intersections and nearest traces to GeoPackage (*.gpkg).')
    parser.add_argument('--verbose', '-V', type=int, nargs='?',
                        default=0, const=1, choices=[0, 1, 2],
                        help='Level of output verbosity.')
    return parser
# fmt: on


@timeit
def create_geometries(
    df,
    coord_cols: list = ['x', 'y'],
    idx_col: str = 'line_id',
):
    """
    Create `shapely` geometries from input dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe holding merged navigation of all SEG-Y files to process.
    coord_cols : list, optional
        Column names of X and Y coordinates (default: [`x`, `y`]).
    idx_col : str, optional
        Column name holding corresponding line indices (default: `line_id`).

    Returns
    -------
    linestrings : np.ndarray (n,)
        Array of LineString geometries (`shapely.geometry.LineString`) of `n`
        input lines (i.e. SEG-Y files).
    points_split : np.ndarray (n,)
        Array of coordinate arrays ((k,2)) with X and Y columns where `n` is the
        number of lines (i.e. SEG-Y files) and `k` the number of vertices
        (i.e. shotpoints) per line.
    points_line_idx : np.ndarray (n*k,)
        Array of line indices for each point (i.e. shotpoint).

    """
    # create LineString geometries (using line indices)
    linestrings = shapely.linestrings(df[coord_cols].to_numpy(), indices=df[idx_col].to_numpy())

    # prepare geometries (faster predicate operations)
    shapely.prepare(linestrings)

    # create Point array (*no* geometry needed for custom distance function!)
    points = df[coord_cols].to_numpy()
    points_line_idx = df[idx_col].to_numpy()

    # split point array into array corresponding to individual lines
    points_split = np.asarray(
        np.split(points, np.nonzero(np.diff(points_line_idx) > 0)[0] + 1), dtype='object'
    )

    return linestrings, points_split, points_line_idx


@timeit
def find_intersections(linestrings):
    """
    Find all intersections of input array of LineStrings.
    Returns array of Points (`shapely.geometry.Point`) with array of their indices and
    corresponding line intersection indices array (referring to input).

    Parameters
    ----------
    linestrings : np.ndarray
        Array of `shapely.geometry.LineString` geometries for each SEG-Y file from:
            
            1. navigation file or
            2. read from SEG-Y trace headers.

    Returns
    -------
    intersections_pts_exploded : np.ndarray
        Array of exploded intersection points (`shapely.geometry.Point`).
    intersections_pts_exploded_idx : np.ndarray
        Index array of exploded intersection points (referring to rows of `line_intersections_idx`).
    line_intersections_idx : np.ndarray
        Array of line intersection indices referring to input linestring array.

    Notes
    -----
    The point array returns the exploded version of detected geometries
    (i.e. separated segments of multipart features).
    For multiple intersections of a line pair, this results in multiple
    geometries with identical reference indices and, thus, could result
    in a larger number of output features.

    """
    # build STRTree
    tree = shapely.STRtree(linestrings)

    # query STRTree (bulk)
    line_inter_idx = tree.query(linestrings, predicate='intersects').T

    # sort intersection indice pairs (row-wise)
    line_inter_idx_sorted = np.sort(line_inter_idx, axis=1)

    # get unique combinations of intersecting lines
    uniq, uniq_idx, uniq_inv, uniq_cnt = np.unique(
        line_inter_idx_sorted, axis=0, return_index=True, return_inverse=True, return_counts=True
    )
    # mask = (uniq_cnt != 1) -> mask self-intersections
    line_inter_idx_uniq = uniq[(uniq_cnt != 1)]

    # (0) get intersection points (vectorized)
    # -> unpack colums (LINESTRING_1, LINESTRING_2)
    intersections_pts = shapely.intersection(*linestrings[line_inter_idx_uniq].T)

    # (1) explode multipart geometries (MULTIPOINT, GEOMETRYCOLLECTION)
    intersections_pts_single, intersections_pts_single_idx = shapely.get_parts(
        intersections_pts, return_index=True
    )

    # (2) extract points from LINESTRING
    tmp = [
        np.asarray([shapely.get_point(g, j) for j in range(shapely.get_num_points(g))])
        if shapely.get_type_id(g) == GEOMETRY.get('LINESTRING')
        else np.asarray([g])
        for g in intersections_pts_single
    ]

    # get number of points in arrays (POINT: 1, LINSTRING: > 1)
    tmp_len = [a.size for a in tmp]

    # create index array for intersection loop
    intersections_pts_exploded_idx = np.repeat(intersections_pts_single_idx, repeats=tmp_len)

    # merge all points into 1D array
    intersections_pts_exploded = np.concatenate(tmp)

    # get (n,2) array of indices of intersection lines (matching exploded intersections!)
    line_intersections_idx = line_inter_idx_uniq[intersections_pts_exploded_idx]

    return intersections_pts_exploded, intersections_pts_exploded_idx, line_intersections_idx


@timeit
def nearest_intersection_vertices(
    points_split, intersections_pts_exploded, line_intersections_idx
):  # noqa
    """
    Return nearest intersection vertex indices and corresponding distances to
    intersection point for both intersecting lines referenced in input index
    array `line_intersections_idx`.

    Parameters
    ----------
    points_split : np.ndarray (n,)
        Array of coordinate arrays ((k,2)) with X and Y columns where `n` is the
        number of lines (i.e. SEG-Y files) and `k` the number of vertices
        (i.e. shotpoints) per line.
    intersections_pts_exploded : np.ndarray (j,)
        Array of intersection points (exploded).
    line_intersections_idx : np.ndarray
        Array of line intersection indices referring to input point array.

    Returns
    -------
    nearest_0 : np.ndarray
        Array of nearest line vertex indices and distance for each intersection
        based on first line (i.e. column 0 in `line_intersections_idx`).
    nearest_1 : np.ndarray
        Array of nearest line vertex indices and distance for each intersection
        based on second line (i.e. column 1 in `line_intersections_idx`).
    """
    # prepare intersection points
    pts_x = shapely.get_x(intersections_pts_exploded)
    pts_y = shapely.get_y(intersections_pts_exploded)
    pts_intersections = np.concatenate((pts_x[:, None], pts_y[:, None]), axis=1)

    # select intersection line points
    points_sel = points_split[line_intersections_idx]

    # calculate distances between LineString verties and intersection point
    # -> return (nearest index, distance)
    nearest_0 = np.empty((pts_intersections.shape[0], 2), dtype=np.float32)
    nearest_1 = np.empty((pts_intersections.shape[0], 2), dtype=np.float32)
    for k, (pts, pi) in enumerate(zip(points_sel, pts_intersections)):
        nearest_0[k] = euclidian_distance(pts[0], pi)
        nearest_1[k] = euclidian_distance(pts[1], pi)

    return nearest_0, nearest_1


def load_trace(path, idx_tr: int, check_bad_traces: bool = False, ntraces2mix: int = 3):
    """
    Load and return single trace from SEG-Y file.

    Parameters
    ----------
    path : str
        File path to SEG-Y on disk.
    idx_tr : int
        Index of trace to load (starting at 0).
    check_bad_traces : bool, optional
        Simple check if loaded trace is too bad/noisy by testing if amplitude
        mean or rescaled trace is greater than 0.4 (default: `False`).
    ntraces2mix : int, optional
        Number of traces to load and average if bad trace was detected.
        The "bad" trace is excluded from averaging (default: `3`).

    Returns
    -------
    trace : np.ndarray (nsamples,)
        Array of trace amplitudes.
    dt : float
        Sampling rate (in ms).
    twt : np.ndarray
        Array of samples with appropriate intervals (determined by `dt`).

    """
    with segyio.open(path, 'r', strict=False, ignore_geometry=True) as file:
        ntraces = file.tracecount  # total number of traces
        dt = segyio.tools.dt(file) / 1000  # sample rate [ms]
        twt = file.samples  # two way travel time (TWTT) [ms]

        # load trace
        trace = file.trace[idx_tr]

        try:
            m = np.mean(rescale(trace))
            if (m > 0.4) and check_bad_traces:
                # prepare number of neighboring traces to mix
                ntraces2mix = ntraces2mix if ntraces2mix % 2 != 0 else ntraces2mix + 1
                nmix_left = ntraces2mix // 2
                nmix_right = ntraces2mix - nmix_left

                # load ``ntraces2mix`` from file
                idx_left = idx_tr - nmix_left if idx_tr - nmix_left >= 0 else 0
                idx_right = idx_tr + nmix_right if idx_tr + nmix_right <= ntraces else ntraces
                trace = file.trace.raw[slice(idx_left, idx_right)]
                # exclude bad/noisy trace and average neighboring traces
                trace = np.delete(trace, nmix_left, axis=0).mean(axis=0)
        except IndexError as err:
            print(path)
            print(idx_tr)
            print('nmix_left, nmix_right:', nmix_left, nmix_right)
            print(idx_tr)
            raise IndexError(err)

    if 'env' not in path:
        trace = envelope(trace)  # calculate envelope (via Hilbert transformation)

    return trace, dt, twt


@timeit
def compute_misties(
    segy_dir,
    line_intersections_names,  # array (nintersections, 2)
    line_intersections_idx,  # index array (nintersections, 2)
    nearest_0,
    nearest_1,  # arrays (vertex indices, distance)
    win: tuple = (False, False),  # upper/lower limit for CC
    quality: float = 0,  # correlation quality threshold
    lookup_df=None,  # lookup dataframe (e.g. datalist)
    lookup_col=None,  # lookup column in df
    check_bad_traces: bool = False,  # check for bad/noisy traces
    ntraces2mix: int = 3,  # odd number of traces to mix
    return_ms: bool = False,  # return mistie shift in ms
    return_coeff: bool = False,  # return correlation coefficients
    verbosity: int = 1,
):
    """
    Compute mistie at line intersections (in number of samples).
    Returns tuple of single shift per input line and shift per line vertex.
    Option to additionally return shift in milliseconds (ms) and/or
    cross-correlation coefficients (i.e. quality indicator).

    Parameters
    ----------
    segy_dir : str
        Directory path of SEG-Y files.
    line_intersections_names : np.ndarray
        Array of string objects of SEG-Y names.
    line_intersections_idx : np.ndarray
        Index array of SEG-Y names.
    nearest_0 : np.ndarray
        Array of nearest line vertex indices and distance for each intersection
        based on first line (i.e. column 0 in `line_intersections_idx`).
    nearest_1 : np.ndarray
        Array of nearest line vertex indices and distance for each intersection.
        based on second line (i.e. column 1 in `line_intersections_idx`).
    win : tuple, optional
        Upper and lower limit of two-way traveltime (TWT) window used for
        cross-correlation. If (False, False), full overlapping trace
        range is used.
    quality : float, optional
        Threshold quality of cross-correlations to include to solve for
        global offsets (default: `0`, i.e. only positive correlations).
    lookup_df : pd.DataFrame, optional
        Dataframe with index set corresponding to names in
        `line_intersections_names` and column of acutal file names to use.
        Default: None (i.e. using file name from provided array).
    lookup_col : str, optional
        Column name with reference SEG-Y file names (default: `None`).
    check_bad_traces : bool, optional
        Simple check if loaded trace is too bad/noisy by testing if amplitude
        mean or rescaled trace is greater than 0.4 (default: `False`).
    ntraces2mix : int, optional
        Number of traces to load and average if bad trace was detected.
        The "bad" trace is excluded from averaging (default: `3`).
    return_ms : bool, optional
        Return shift in milliseconds (ms) (default: `False`).
    return_coeff : bool, optional
        Return cross-correlation coefficients (i.e. quality indicator).
        Default: False.
    verbosity : int
        Control level of output messages (default: `1`, i.e. `info`).

    Returns
    -------
    offsets : np.ndarray
        Array of sample shift per line with shape (nlines,).
    offsets_ms : np.ndarray
        Array of shift (in milliseconds) per line with shape (nlines,).
    coeff : np.ndarray
        Array of quality indications (Pearson’s correlation coefficient).

    """
    # unpack window limits
    win_upper, win_lower = win

    # allocate output arrays
    intersections_misties = np.empty((len(line_intersections_names),), dtype=np.int16)
    intersections_coeffs = np.empty((len(line_intersections_names),), dtype=np.float32)

    # loop over all intersection
    for idx_intersection, lines in enumerate(
        tqdm(
            line_intersections_names,
            desc='Computing misties',
            ncols=80,
            total=len(line_intersections_names),
            unit_scale=True,
            unit=' intersections',
            disable=True if verbosity <= 1 else False,
        )
    ):
        # set upper and lower window extent from user input
        win_up, win_lo = win_upper, win_lower

        # input files
        line_0 = lines[0]
        line_1 = lines[1]

        if (lookup_df is not None) and (lookup_col != None):  # noqa
            path_0 = os.path.join(segy_dir, lookup_df.loc[line_0, 'line'])  # use lookup df
            path_1 = os.path.join(segy_dir, lookup_df.loc[line_1, 'line'])  # use lookup df
        else:
            path_0 = os.path.join(segy_dir, line_0 + '.sgy')
            path_1 = os.path.join(segy_dir, line_1 + '.sgy')

        # (1) read trace(s) and auxiliary information
        ## load trace(s) of first intersecting line
        idx_tr_0 = int(nearest_0[idx_intersection, 0])
        trace_0, dt_0, twt_0 = load_trace(
            path_0, idx_tr_0, check_bad_traces=True, ntraces2mix=ntraces2mix
        )

        ## load trace(s) of second intersecting line
        idx_tr_1 = int(nearest_1[idx_intersection, 0])
        trace_1, dt_1, twt_1 = load_trace(
            path_1, idx_tr_1, check_bad_traces=True, ntraces2mix=ntraces2mix
        )

        assert dt_0 == dt_1, f'Identical sample interval required: {dt_0} != {dt_1} ({line_0} != {line_1})'

        # (2) define trace extents to correlate
        ## get extent of overlapping trace range (in ms) if not explicitely specified
        if not all([win_up, win_lo]):
            win_up = max(twt_0.min(), twt_1.min())
            win_lo = min(twt_0.max(), twt_1.max())

        if (max(twt_0.min(), twt_1.min()) > win_up) or (min(twt_0.max(), twt_1.max()) < win_lo):
            xprint(
                *(
                    f'Adjust window range ({win_up}:{win_lo} ms) to valid data range ',
                    f'({max(twt_0.min(), twt_1.min())}:{min(twt_0.max(), twt_1.max())} ms)',
                ),
                kind='warning',
                verbosity=verbosity,
            )
            if max(twt_0.min(), twt_1.min()) > win_up:
                win_up = max(twt_0.min(), twt_1.min())
            if min(twt_0.max(), twt_1.max()) < win_lo:
                win_lo = min(twt_0.max(), twt_1.max())

        ## subset trace windows for cross-correlation (in ms)
        # boolean masking of subset samples
        mask_0 = (twt_0 >= win_up) & (twt_0 <= win_lo)
        tr_0 = trace_0[mask_0]

        # boolean masking of subset samples
        mask_1 = (twt_1 >= win_up) & (twt_1 <= win_lo)
        tr_1 = trace_1[mask_1]

        # exclude all zero samples of either trace (from padding)
        mask_zero_samples = (tr_0 == 0) | (tr_1 == 0)
        tr_0 = tr_0[~mask_zero_samples]
        tr_1 = tr_1[~mask_zero_samples]

        # (3) compute cross-correlation
        cc = scipy_correlate(tr_0, tr_1, mode='same', method='fft')

        ## get mistie (in samples)
        mistie = cross_correlation_shift(cc)

        ## calculate Pearson’s correlation coefficient
        coeff = scipy_pearsonr(tr_0, tr_1)[0]

        # (4) populate output arrays
        intersections_misties[idx_intersection] = mistie
        intersections_coeffs[idx_intersection] = coeff

    # (5) filter misties & coefficients by quality
    mask_quality = np.abs(intersections_coeffs) >= quality
    xprint(
        f'Filtered < {np.count_nonzero(~mask_quality)} > values below quality threshold ({quality})',
        kind='info',
        verbosity=verbosity,
    )

    ## mistie (cross-correlation shift) < 0  --->  line_0 deeper than line_1
    ## mistie (cross-correlation shift) > 0  --->  line_0 shallower than line_1
    misties = intersections_misties[mask_quality]
    coeffs = intersections_coeffs[mask_quality]

    # (6) filter line names/indices by quality
    line_intersections_names = line_intersections_names[mask_quality]
    line_intersections_idx = line_intersections_idx[mask_quality]

    nintersections = len(misties)  # number of intersections (after filtering)
    nlines = len(lookup_df)  # number of SEG-Y lines

    # (7) compute global mistie for each line for all intersections (using least-squares solution)
    ## init adjacency matrix (non-square)
    A = np.zeros((nintersections, nlines), dtype=np.int32)

    ## populate adjacency matrix (c.f. Bishop and Nunns, 1994)
    i = np.arange(nintersections)
    A[i, line_intersections_idx[:, 0]] = 1  # a_ij =  1, if k(i) = j
    A[i, line_intersections_idx[:, 1]] = -1  # a_ij = -1, if l(i) = j

    ## least-squares solution of linear matrix equation
    offsets, residuals, rank, s = np.linalg.lstsq(A, misties, rcond=None)
    offsets = np.around(offsets, 0).astype('int16')  # single shift per line
    # offsets_per_pnt = offsets[points_line_idx]        # shift per line point

    if return_ms and return_coeff:
        offsets_ms = samples2twt(offsets, dt=dt_0)
        # offsets_per_pnt_ms = offsets_ms[points_line_idx]
        # return (offsets, offsets_per_pnt), (offsets_ms, offsets_per_pnt_ms), coeff
        return (offsets, residuals), offsets_ms, coeffs

    elif return_ms and not return_coeff:
        offsets_ms = samples2twt(offsets, dt=dt_0)
        # offsets_per_pnt_ms = offsets_ms[points_line_idx]
        # return (offsets, offsets_per_pnt), (offsets_ms, offsets_per_pnt_ms)
        return (offsets, residuals), offsets_ms

    elif not return_ms and return_coeff:
        return (offsets, residuals), coeffs

    return (offsets, residuals)


@jit(nopython=True, fastmath=True)
def euclidian_distance(a, b):
    """Calculate euclidian distance between array of points (n,2) and single point (2,)."""
    n = a.shape[0]
    dist = np.empty((n,), dtype=np.float64)
    for i in range(n):
        dist[i] = np.linalg.norm(a[i] - b)

    idx = np.argmin(dist)

    return (idx, dist[idx])


def cross_correlation_shift(cc):
    """
    Return cross-correlation shift (in samples) between correlated signals.

    Parameters
    ----------
    cc : np.ndarray
        Cross-correlation between two signals (1D).

    Returns
    -------
    shift : int
        Lag shift between both signals.
        
        - `shift` < 0  --->  signal A later than signal B
        - `shift` > 0  --->  signal A earlier than signal B

    """
    zero_idx = int(np.floor(len(cc) / 2))
    idx = np.argmax(cc) if np.abs(np.max(cc)) >= np.abs(np.min(cc)) else np.argmin(cc)
    return zero_idx - idx


def compensate_mistie(data, mistie, verbosity=1):
    """
    Apply computed static offsets to seismic traces (2D array).

    Parameters
    ----------
    data : np.ndarray
        2D array of input seismic traces (`samples` x `traces`).
    mistie : int, float
        1D array of mistie offsets (for all traces).

    Returns
    -------
    data_mistie : np.ndarray(s)
        Compensated seismic section.

    """
    mistie_samples = int(np.around(mistie, 0))

    # create copy of original data
    data_mistie = data.copy()
    n_samples, n_traces = data_mistie.shape

    if mistie_samples < 0:
        data_mistie = np.concatenate(
            (data_mistie[abs(mistie_samples) :, :], np.zeros((abs(mistie_samples), n_traces)))
        )
        xprint(
            f'#samples:{mistie_samples:>5}   ->   up: {data_mistie.shape}',
            kind='debug',
            verbosity=verbosity,
        )
    elif mistie_samples > 0:
        data_mistie = np.concatenate(
            (np.zeros((abs(mistie_samples), n_traces)), data_mistie[: -abs(mistie_samples), :])
        )
        xprint(
            f'#samples:{mistie_samples:>5}   ->   down: {data_mistie.shape}',
            kind='debug',
            verbosity=verbosity,
        )
    else:
        pass

    return data_mistie


def write_intersections_QC(
    args,
    pts_split,
    line_intersections_idx,
    line_intersections_names,
    nearest_0,
    nearest_1,
    intersections_pts,
    crs='EPSG:32760',
) -> None:
    """Write GeoPackage of line intersections and nearest traces."""
    TODAY = datetime.date.today().isoformat()

    work_dir = args.output_dir if args.output_dir is not None else os.path.dirname(args.input_path)
    gpkg_file = os.path.join(work_dir, f'{TODAY}_QC_{os.path.basename(work_dir)}_intersections.gpkg')

    # (1) NEAREST VERTICES
    # list of number of points per line
    lines_npts = [a.shape[0] for a in pts_split]
    # compute index offsets for first line indices
    lines_npts_cumsum = np.concatenate((np.array([0]), np.cumsum(lines_npts)[:-1].T))
    # get offsets for intersection line pairs
    lines_offset_indices = lines_npts_cumsum[line_intersections_idx]

    # flatten array of coordinate arrays
    points = np.concatenate(pts_split)

    # get X and Y coordinates of nearest intersection points
    points_0 = points[nearest_0[:, 0].astype('int') + lines_offset_indices[:, 0]]
    points_1 = points[nearest_1[:, 0].astype('int') + lines_offset_indices[:, 1]]

    # create GeoDataFrame (line 0)
    gpd_nearest_pts_0 = gpd.GeoDataFrame(
        data=dict(
            dist=nearest_0[:, 1], geom=gpd.points_from_xy(x=points_0[:, 0], y=points_0[:, 1])
        ),
        geometry='geom',
        crs=crs,
    )
    # create GeoDataFrame (line 1)
    gpd_nearest_pts_1 = gpd.GeoDataFrame(
        data=dict(
            dist=nearest_1[:, 1], geom=gpd.points_from_xy(x=points_1[:, 0], y=points_1[:, 1])
        ),
        geometry='geom',
        crs=crs,
    )

    # export nearest vertices to GeoPackage
    gpd_nearest_pts_0.to_file(gpkg_file, driver='GPKG', layer='nearest_vertices_line_0')
    gpd_nearest_pts_1.to_file(gpkg_file, driver='GPKG', layer='nearest_vertices_line_1')

    # (2) INTERSECTION
    # create DataFrame from intersection points
    df_intersection_pts = pd.DataFrame(
        data=dict(
            x=shapely.get_x(intersections_pts),
            y=shapely.get_y(intersections_pts),
            line_0=line_intersections_names[:, 0],
            dist_0=nearest_0[:, 1],
            x_0=points_0[:, 0],
            y_0=points_0[:, 1],
            line_1=line_intersections_names[:, 1],
            dist_1=nearest_1[:, 1],
            x_1=points_1[:, 0],
            y_1=points_1[:, 1],
        )
    )

    # cconvert into GeoDataFrame
    gpd_intersection_pts = gpd.GeoDataFrame(
        df_intersection_pts, geometry=intersections_pts, crs=crs
    )
    # export intersection points to GeoPackage
    gpd_intersection_pts.to_file(gpkg_file, driver='GPKG', layer='intersections')


def main_misties(args):
    """
    Compute line intersection misties.

    Parameters
    ----------
    args : argparse.Namespace
        Input parameter.

    Returns
    -------
    list_segy : list
        List of SEG-Y files to correct.
    offsets : np.ndarray
        Vertical misties per intersection (in samples).
    offsets_ms : np.ndarray
        Vertical mistie per intersection (in milliseconds TWT).

    """
    # coords_path = args.coords_path if args.coords_path is not None else args.input_path

    # (1) Load SEG-Y coordinates (navigation)
    ## OPTION A: from auxiliary files (*.nav)
    if args.coords_origin == 'aux':
        # set file suffix
        fsuffix = args.coords_fsuffix if args.coords_fsuffix is not None else 'nav'

        # read navigation data
        # df = pd.read_csv(args.coords_path, index_col='tracl') # index_col=0
        xprint(
            f'Load navigation from auxiliary files (*.{fsuffix})',
            kind='info',
            verbosity=args.verbose,
        )
        df_nav = read_auxiliary_files(
            args.coords_path, fsuffix=fsuffix, suffix=args.coords_text_suffix, index_cols=None
        )

    ## OPTION B: from SEG-Y headers
    elif args.coords_origin == 'header':
        # set file suffix
        fsuffix = args.suffix if args.suffix is not None else 'sgy'

        # read navigation data
        xprint(
            'Extract and load navigation from headers of SEG-Y files',
            kind='info',
            verbosity=args.verbose,
        )
        df_nav = extract_navigation_from_segy(
            args.coords_path, fsuffix=fsuffix, fnsuffix=args.coords_text_suffix, write_aux=False
        )

    # create int representation of line names
    df_nav['line_id'] = df_nav['line'].factorize()[0]

    # create array of unique SEG-Y files (from navigation file)
    intersections_lines = df_nav[['line_id', 'line']].groupby('line').first().index.to_numpy()

    # (2) Create list of SEG-Y files to process
    ## OPTION A: from datalist
    if os.path.isfile(args.input_path):
        df_segy = pd.read_csv(args.input_path, header=None, names=['line'])

        # extract first part of line names (to match names from navigation file)
        df_segy['line_core'] = df_segy['line'].str.extract(r'(.*)_UTM')  # TODO: make dynamic!
        df_segy.set_index('line_core', inplace=True)

        # create list of SEG-Y file for later import
        list_segy = df_segy['line'].to_list()

        dir_segy, file_datalist = os.path.split(args.input_path)

    ## OPTION B: from directory of SEG-Y files
    elif os.path.isdir(args.input_path):
        dir_segy = args.input_path

        pattern = '*'
        pattern += f'{args.filename_suffix}' if args.filename_suffix is not None else pattern
        pattern += f'.{args.suffix}' if args.suffix is not None else '.sgy'

        # create list of SEG-Y file for later import
        list_segy = glob.glob(os.path.join(args.input_path, pattern))
        list_segy = [os.path.basename(f) for f in list_segy]

        # create lookup dataframe
        df_segy = pd.DataFrame(list_segy, columns=['line'])
        df_segy['line_core'] = df_segy['line'].str.extract(r'(.*)_UTM')
        df_segy.set_index('line_core', inplace=True)
    else:
        raise OSError(f'Cannot find >{args.input_path}<')
    
    unused_nav = [x not in df_segy.index for x in intersections_lines]
    if any(True for x in unused_nav):
        # remove navigation without corresponding SEG-Y
        df_nav.drop(df_nav[df_nav['line'].isin(intersections_lines[unused_nav])].index, inplace=True)
        
        # create int representation of line names
        df_nav['line_id'] = df_nav['line'].factorize()[0]

        # create array of unique SEG-Y files (from navigation file)
        intersections_lines = df_nav[['line_id', 'line']].groupby('line').first().index.to_numpy()

    if (args.quality_threshold < 0) or (args.quality_threshold > 1):
        raise ValueError('`quality_threshold` must be float in range [0-1]')
    else:
        quality = args.quality_threshold

    if args.win_cc is None:
        win = (False, False)
    else:
        upper, lower = args.win_cc[0], args.win_cc[1]
        if upper > lower:
            upper, lower = args.win_cc[1], args.win_cc[0]
        win = (float(upper), float(lower))

    # ===== PROCESSING =====
    # (1) create shapely geometries
    linestrings, pts_split, pts_line_idx = create_geometries(
        df_nav, coord_cols=['x', 'y'], idx_col='line_id'
    )

    # (2) find intersections (Points, MultiPoints, LineStrings, GeometryCollections)
    intersections_pts, intersections_pts_idx, line_intersections_idx = find_intersections(
        linestrings
    )

    # get line names for all intersections
    line_intersections_names = intersections_lines[line_intersections_idx]
    assert line_intersections_names.shape == line_intersections_idx.shape

    # (3) get array of nearest vertices (with distance)
    nearest_0, nearest_1 = nearest_intersection_vertices(
        pts_split, intersections_pts, line_intersections_idx
    )

    # (4) get mistie in both samples and milliseconds
    (offsets, residuals), offsets_ms, coeffs = compute_misties(
        dir_segy,
        line_intersections_names,
        line_intersections_idx,
        nearest_0,
        nearest_1,
        win=win,
        quality=quality,
        lookup_df=df_segy,
        lookup_col='line',
        check_bad_traces=True,
        ntraces2mix=3,
        return_ms=True,
        return_coeff=True,
        verbosity=args.verbose,
    )

    if args.write_QC:
        if geopandas_enabled:
            xprint(
                'Save line intersections and nearest traces as GeoPackage',
                kind='info',
                verbosity=args.verbose,
            )
            write_intersections_QC(
                args,
                pts_split,
                line_intersections_idx,
                line_intersections_names,
                nearest_0,
                nearest_1,
                intersections_pts,
            )
        else:
            xprint('Missing GeoPandas package. No GeoPackage export possible.', kind='warn', verbosity=args.verbose)

    return list_segy, offsets, offsets_ms, residuals


def wrapper_mistie_correction_segy(in_path, offset, offset_ms, args):
    """Apply computed misties to each line (SEG-Y file)."""
    basepath, filename = os.path.split(in_path)
    basename, suffix = os.path.splitext(filename)
    xprint(f'Processing file < {filename} >', kind='info', verbosity=args.verbose)
    
    default_txt_suffix = 'mistie'
    
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
        dt = segyio.tools.dt(src) / 1000  # sample rate [ms]
        n_samples = src.samples.size  # total number of samples
        # twt = src.samples                   # two way travel time (TWTT) [ms]

        xprint(f'n_traces:  {n_traces}', kind='debug', verbosity=args.verbose)
        xprint(f'n_samples: {n_samples}', kind='debug', verbosity=args.verbose)
        xprint(f'dt:        {dt}', kind='debug', verbosity=args.verbose)

        tracl = src.attributes(segyio.TraceField.TRACE_SEQUENCE_LINE)[:]
        tracr = src.attributes(segyio.TraceField.TRACE_SEQUENCE_FILE)[:]
        fldr = src.attributes(segyio.TraceField.FieldRecord)[:]

        # get seismic data [amplitude]; transpose to fit numpy data structure
        data_src = src.trace.raw[:].T  # eager version (completely read into memory)

        # apply tidal compensation to seismic traces
        xprint('Apply mistie compensation', kind='info', verbosity=args.verbose)
        data_comp = compensate_mistie(data_src, offset, verbosity=args.verbose)

        # set output amplitudes (transpose to fit SEG-Y format)
        xprint('Writing compensated data to disk', kind='info', verbosity=args.verbose)
        src.trace = np.ascontiguousarray(data_comp.T, dtype=data_src.dtype)

    # update textual header
    text = get_textual_header(path)
    text_updated = add_processing_info_header(text, 'MISTIE', prefix='_TODAY_', newline=True)
    write_textual_header(path, text_updated)

    if args.write_aux:
        xprint(f'Creating auxiliary file < {out_name}.mst >', kind='info', verbosity=args.verbose)
        aux_path = os.path.join(out_dir, f'{out_name}.mst')

        with open(aux_path, 'w', newline='\n') as fout:
            header = 'tracl,tracr,fldr,mistie_samples,mistie_ms\n'
            fout.write(header)
            for i in range(len(tracr)):
                line = f'{tracl[i]},{tracr[i]},{fldr[i]},' + f'{offset},{offset_ms:.2f}\n'
                fout.write(line)


def main(argv=sys.argv):  # noqa
    TIMESTAMP = datetime.datetime.now().isoformat(timespec='seconds').replace(':', '')
    SCRIPT = os.path.basename(__file__).split(".")[0]

    # get command line input
    parser = define_input_args()
    args = parser.parse_args(argv[1:])  # exclude filename parameter at position 0
    xprint(args, kind='debug', verbosity=args.verbose)
    
    # compute misties
    list_segy, offsets, offsets_ms, residuals = main_misties(args)

    # check input file(s)
    in_path = args.input_path
    basepath, filename = os.path.split(in_path)
    basename, suffix = os.path.splitext(filename)
    if suffix == '':
        basepath = in_path
        basename, suffix = None, None  # noqa

    # (2) input directory (multiple files)
    if os.path.isdir(in_path):
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
                for idx, file_path in enumerate(file_list):
                    # correct misties
                    wrapper_mistie_correction_segy(file_path, offsets[idx], offsets_ms[idx], args)
        clean_log_file(logfile)
    else:
        sys.exit('No input files to process. Exit process.')


#%% MAIN
if __name__ == '__main__':

    main()
