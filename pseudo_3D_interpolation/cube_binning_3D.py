"""
Create (sparse) inline/xline cube from intersecting 2D profiles.
Applying a user-defined binning method ('average', 'median', 'nearest', or 'IDW').

"""
import os
import sys
import glob
import re
import yaml
import datetime
import argparse
import warnings
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr
import pyproj

from tqdm import tqdm
import dask
from dask.diagnostics import ProgressBar

import segyio
from segysak import open_seisnc
from segysak.segy import segy_header_scrape, segy_bin_scrape

from pseudo_3D_interpolation.functions.transform import Affine
from pseudo_3D_interpolation.functions.utils import xprint, ffloat, round_up, show_progressbar
from pseudo_3D_interpolation.functions.utils_io import (
    read_auxiliary_files,
    extract_navigation_from_segy,
)

xr.set_options(keep_attrs=True)  # preserve metadata during computations

# %% DEFAULTS
STACK_METHODS = ["average", "median", "nearest", "IDW"]

# %% FUNCTIONS
def check_sampling_interval(df):
    """Check and return sampling interval (dt) from datafram of all scraped SEG-Y files."""
    # filter dataframe by unique lines
    df_filt = df.groupby("line_id")["TRACE_SAMPLE_INTERVAL"].min()

    dt_minmax = (df_filt.min(), df_filt.max())
    if np.mean(dt_minmax) != dt_minmax[0]:
        raise ValueError(f"SEG-Y files with different sampling intervals (dt: {dt_minmax})")

    return float(dt_minmax[0])


def distance(p1, p2):
    """Calculate euclidian distance between to points p1 and p2 (row-wise)."""
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    assert p1.shape == p2.shape, "Input points must have identical shape!"

    if p1.ndim == 1:
        return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    elif p1.ndim == 2:
        return np.sqrt((p2[:, 0] - p1[:, 0]) ** 2 + (p2[:, 1] - p1[:, 1]) ** 2)
    else:
        raise NotImplementedError("Only 1D or 2D input arrays are supported!")
        

def find_nearest_ilxl(ilxl_reference: np.ndarray, ilxl_data: np.ndarray, return_index: bool = False):
    """Find nearest ilxl of each input trace."""
    ilxl_reference_middles = ilxl_reference[1:] - np.diff(ilxl_reference.astype('f')) / 2
    idx = np.searchsorted(ilxl_reference_middles, ilxl_data)
    if return_index:
        return ilxl_reference[idx], idx
    return ilxl_reference[idx]


def points_from_extent(extent: tuple):
    """
    Return numpy array from extent provided as `(w, e, s, n)` or `(xmin, xmax, ymin, ymax)`, respectively.

    Parameters
    ----------
    extent : tuple
        Data extent provided as `(w, e, s, n)` or `(xmin, xmax, ymin, ymax)`.

    Returns
    -------
    np.ndarray
        Corner coordinates of given extent as `(lower_left, upper_left, upper_right, lower_right)`.

    """
    if not isinstance(extent, (tuple, list, np.ndarray)):
        raise ValueError("extent must be either tuple, list or np.array")

    return np.array(
        [
            [extent[0], extent[2]],  # lower left
            [extent[0], extent[3]],  # upper left
            [extent[1], extent[3]],  # upper right
            [extent[1], extent[2]],  # lower right
        ]
    )


def extent_from_points(points):
    """Return bounding box extent from points (N, 2)."""
    points = np.asarray(points)
    if points.shape[1] != 2:
        raise ValueError("input points must be array-like of shape (N,2)")
    return (points[:, 0].min(), points[:, 0].max(), points[:, 1].min(), points[:, 1].max())


def get_polygon_area(pts: np.ndarray) -> float:
    """
    Calculate polygon area (in cartesian coordinates).

    Parameters
    ----------
    pts : np.ndarray
        2D input array of X and Y coordinates with shape (n_points, 2).

    Returns
    -------
    area : float
        Polygon area (in coordinate units).

    References
    ----------
    [^1]: Stackoverflow post [https://stackoverflow.com/a/66801704](https://stackoverflow.com/a/66801704)

    """
    pts = np.vstack((pts, pts[0, :]))
    xs = pts[:, 0]
    ys = pts[:, 1]
    return 0.5 * np.abs(np.dot(xs, np.roll(ys, 1)) - np.dot(np.roll(xs, 1), ys))


def get_polygon_centroid(xy: np.ndarray) -> np.ndarray:
    """
    Calculate polygon centroid (in cartesian coordinates).

    Parameters
    ----------
    xy : np.ndarray
        2D input array of X and Y coordinates with shape (n_points, 2).

    Returns
    -------
    centroid : np.ndarray
        Centroid coordinate array (x, y).

    References
    ----------
    [^1]: Stackoverflow post [https://stackoverflow.com/a/66801704](https://stackoverflow.com/a/66801704)

    """
    xs = xy[:, 0]
    ys = xy[:, 1]
    return np.dot(xy.T + np.roll(xy.T, 1, axis=1), xs * np.roll(ys, 1) - np.roll(xs, 1) * ys) / (
        6 * get_polygon_area(xy)
    )


def affine_transform_coords_to_ilxl(
    corner_points: np.ndarray = None,
    extent: tuple = None,
    spacing: float = None,
    base_transform: Affine = None,
    inverted: bool = False,
    original_coords: bool = False,
    verbose: bool = False,
):
    """
    Create Affine transformation object from (a) `corner_points` or (b) `extent` and `spacing`.
    If `base_transform` is provided, it will be used to setup final transform.

    Parameters
    ----------
    corner_points : np.ndarray
        2D array of corner points with shape (4,2):
        Either this variable or `extent` are required!
    extent : tuple
        Tuple of extent (xmin, xmax, ymin, ymax).
        Either this variable or `corner_points` are required!
    spacing : float | tuple(float, float)
        Grid bin size (in CRS units). Can be single value or tuple(ysize, xsize), i.e. (iline, xline).
    base_transform : Affine, optional
        Base transformation used to set up the returned transformation.
    inverted : bool, optional
        `base_transform` is inverted (default: `False`).
    original_coords : TYPE, optional
        Priginal, non-transformed coordinates provided (default: `False`).
    verbose : bool, optional
        Print verbose output (default: `False`).

    Returns
    -------
    Affine
        Affine transformation.

    """
    if (corner_points is None) and (extent is None):
        raise ValueError("Either `corner_points` or `extent` must be specified")
    if spacing is None:
        raise ValueError("`spacing` must be specified")

    # get corner coordinates from corner_points
    if corner_points is not None:
        if (original_coords) and (base_transform is not None) and (not inverted):
            warnings.warn("Untested option - unexpected results possible!")
            corner_points = base_transform.transform(corner_points)
        elif (original_coords) and (base_transform is not None) and (inverted):
            warnings.warn("Untested option - unexpected results possible!")
            corner_points = base_transform.inverse().transform(corner_points)
        else:
            corner_points = corner_points
    # get corner coordinates from extent
    elif extent is not None:
        if (original_coords) and (base_transform is not None) and (not inverted):
            warnings.warn("Untested option - unexpected results possible!")
            corner_points = base_transform.transform(points_from_extent(extent))
        elif (original_coords) and (base_transform is not None) and (inverted):
            warnings.warn("Untested option - unexpected results possible!")
            corner_points = base_transform.inverse().transform(points_from_extent(extent))
        else:
            corner_points = points_from_extent(extent)

    if isinstance(spacing, (tuple, list)):
        yspacing, xspacing = spacing
    elif isinstance(spacing, (int, float)):
        xspacing = yspacing = spacing

    # calc bin center points from corner points and bin spacing(s)
    xprint(f"corner_points:\n{corner_points!r}", kind="debug", verbosity=verbose)
    center_points = corner_points + np.array(
        [
            [ xspacing / 2,  yspacing / 2],  # noqa
            [ xspacing / 2, -yspacing / 2],  # noqa
            [-xspacing / 2, -yspacing / 2],  # noqa
            [-xspacing / 2,  yspacing / 2],  # noqa
        ]
    )
    xprint(f"center_points:\n{center_points!r}", kind="debug", verbosity=verbose)

    # calc length of extent in x/y directions (in CRS units)
    dist_x = distance(center_points[0], center_points[-1])
    dist_y = distance(center_points[0], center_points[1])
    xprint(f"distance (x-axis):  {dist_x:.2f}", kind="debug", verbosity=verbose)
    xprint(f"distance (y-axis):  {dist_y:.2f}", kind="debug", verbosity=verbose)

    # create 1D iline/xline arrays from distances
    n_ilines = int(np.around(dist_x / xspacing, 0))  # dist: ll -> lr
    n_xlines = int(np.around(dist_y / yspacing, 0))  # dist: ll -> ul
    xprint(f"# ilines: {n_ilines}", kind="debug", verbosity=verbose)
    xprint(f"# xlines: {n_xlines}", kind="debug", verbosity=verbose)

    # create affine transform from CRS coordinates to iline/xline
    affine_coords2ilxl = (
        Affine()
        .translation(-center_points[0])
        .scaling(scale=(1.0 / (np.around(dist_x)), 1.0 / (np.around(dist_y))))
        .scaling((n_ilines, n_xlines))
        .translation((1, 1))  # iline/xline always start with 1
    )

    if (base_transform is not None) and (inverted is False):
        return affine_coords2ilxl @ base_transform
    elif (base_transform is not None) and (inverted is True):
        return affine_coords2ilxl @ base_transform.inverse()

    return affine_coords2ilxl


def round_ilxl_extent(points):
    """
    Round array of cube il/xl indices to appropriate integer indice values.

        - lower/left indices  --> rounded up
        - upper/right indices --> rounded down

    Parameters
    ----------
    points : np.ndarray
        Array of il/xl for corner points of cube.

    Returns
    -------
    np.ndarray
        Rounded input array.

    """
    offset = 1e-9
    offset_array = np.array(
        [[offset, offset], [offset, -offset], [-offset, -offset], [-offset, offset]]
    )
    return np.around(points + offset_array, 0).astype("int")


def pad_trace(da, delrt: int, twt, dt: float):
    """
    Pad seismic trace at top and bottom to fit global `twt` array.

    Parameters
    ----------
    da : dask.array
        Input seismic trace.
    delrt : int
        Delay recording time of seismic trace (ms).
    twt : np.ndarray
        Reference array of output two-way traveltimes (TWT).
    dt : float
        Sampling rate (ms).

    Returns
    -------
    out : dask.array
        Padded seismic trace.

    """
    # calculate TWT of last sample in input trace
    twt_tr_end = delrt + da.size * dt - dt

    # top
    if delrt < twt[0]:  # clip input trace
        ntop = int(round((twt[0] - delrt) / dt))
        da = da[ntop:]
        ntop = 0
    else:  # pad input trace
        ntop = int((delrt - twt[0]) / dt)

    # bottom
    if twt_tr_end > twt[-1]:  # clip input trace
        nbottom = int(round((twt_tr_end - twt[-1]) / dt)) * -1
        da = da[:nbottom]
        nbottom = 0
    else:  # pad input trace
        nbottom = int(twt.size - da.size)

    out = dask.array.pad(da, ((ntop, nbottom),), mode="constant", constant_values=0.0)
    assert out.size == twt.size, (ntop, nbottom, out)

    return out


def adjust_extent(extent, spacing):
    """
    Adjust extent to fit given spacing by adding to min/max of X and Y coordinates.

    Parameters
    ----------
    extent : tuple
        Extent of point data as coordinate tuple `(xmin, xmax, ymin, ymax)`.
    spacing : float | tuple(float, float)
        Grid bin size (in CRS units). Can be single value or tuple featuring (xsize, ysize).

    Returns
    -------
    extent_adj : tuple
        Adjusted extent of original data extent.

    """
    if isinstance(spacing, (int, float)):
        spacing = tuple(spacing, spacing)
    elif isinstance(spacing, Iterable):
        if not all([isinstance(s, (int, float)) for s in spacing]):
            raise TypeError('extent must be int, float or tuple(int, float)')
    else:
        raise TypeError('extent must be int, float or tuple(int, float)')
    
    diff_x = extent[1] - extent[0]
    pad_x = round_up(diff_x, spacing[0]) - diff_x
    
    diff_y = extent[3] - extent[2]
    pad_y = round_up(diff_y, spacing[1]) - diff_y
    
    extent_adj = (
        extent[0] - pad_x / 2,
        extent[1] + pad_x / 2,
        extent[2] - pad_y / 2,
        extent[3] + pad_y / 2,
    )
    
    return extent_adj


def transform_and_adjust_extent(extent_pts: tuple, spacing: tuple, transform: Affine) -> tuple:
    """
    Transform using Affine `transform` and adjust cube extent according to bin spacing.

    Parameters
    ----------
    extent_pts : tuple
        Data extent as tuple of `(w, e, s, n)` or `(xmin, xmax, ymin, ymax)`.
    spacing : tuple
        Bin spacing (il, xl) or (y, x).
    transform : Affine
        Affine transformation to create NS-aligned geometry for input corner points.

    Returns
    -------
    tuple
        Transformed (e.g. rotated) and adjusted extent.

    """
    extent_pts_t = transform.transform(extent_pts)
    if all(isinstance(s, int) for s in spacing):  # account for transformation precision errors
        extent_pts_t = (extent_pts_t + 0.5).astype('int').astype('float')
    extent_rect = extent_from_points(extent_pts_t)
    extent_rect = adjust_extent(extent_rect, spacing=spacing)
    return extent_rect


def get_cube_parameter(
    transform_forward,
    transform_reverse,
    df_nav: pd.DataFrame,
    bin_size: int,
    cube_corner_pts: np.ndarray,
    bin_size_region: int = None,
    region_corner_pts: np.ndarray = None,
    return_geometry: bool = False,
    verbose: bool = False,
):
    """
    Create cube inline/xline bin indices and associated bin center coordinates.
    Based on provided forward/inverse coordinate transform objects, bin size (in coordinate units)
    and given extent of output cube.
    Inline/xline indices start with 1 if `region_corner_pts` argument is not provided.

    Parameters
    ----------
    transform_forward : Affine.transform
        Forward Affine transform used as base transform for coordinate transformation.
    transform_reverse : Affine.transform
        Inverse Affine transform.
    df_nav : pd.DataFrame
        Dataframe with X and Y coordinates for each seismic trace from all SEGY files.
    bin_size : float | tuple(float, float)
        Size of iline/xline bins (in CRS units, e.g. meter).
    bin_size_region : float | tuple(float, float)
        Size of iline/xline bins for region (in CRS units, e.g. meter).
    cube_corner_pts : np.ndarray
        Corner point coordinates of cube (lower_left, upper_left, upper_right, lower_right)
        with shape (4, 2).
    region_corner_pts : np.ndarray, optional
        Optional corner point coordinates of region (lower_left, upper_left, upper_right, lower_right)
        with shape (4, 2).
        Useful if `cube` is part of a larger region and bin indices should not start with 1.
    return_geometry : bool, optional
        Return adjusted geometry for cube_corner_pts (and region_corner_pts if specified)
    verbose : bool, optional
        Print optional information to console (default: `False`).

    Returns
    -------
    df_bins : pd.DataFrame
        Dataframe defining output cube with inline 'il' and xline 'xl' indices
        and bin center coordinates ('x', 'y').
    df_ilxl : pd.DataFrame
        Dataframe with inline and xline indices for each trace position provided in `df_nav`.

    """
    USE_REGION = region_corner_pts is not None
    
    # input corner points --> transform & adjust
    extent_cube_t = transform_and_adjust_extent(
        cube_corner_pts, spacing=bin_size if not USE_REGION else bin_size_region, transform=transform_forward
    )
    extent_region_t = (
        transform_and_adjust_extent(region_corner_pts, spacing=bin_size_region, transform=transform_forward)
        if USE_REGION else None
    )
    # cube corner points (transformed)
    cube_corner_pts_t = points_from_extent(extent_cube_t)
    region_corner_pts_t = points_from_extent(extent_region_t) if USE_REGION else None

    # affine transformation X/Y to il/xl
    transform_xy_to_ilxl = affine_transform_coords_to_ilxl(
        extent=extent_region_t if USE_REGION else extent_cube_t,
        spacing=bin_size_region if USE_REGION else bin_size,
        base_transform=transform_forward,
        verbose=verbose
    )
    # get max iline/xline numbers from extent
    cube_corner_pts_cnv = transform_reverse.transform(cube_corner_pts_t)
    ilxl_extent = transform_xy_to_ilxl.transform(cube_corner_pts_cnv)
    xprint(f"ilxl_extent:\n{ilxl_extent.astype('float')!r}", kind="debug", verbosity=verbose)
    ilxl_extent = round_ilxl_extent(ilxl_extent)
    xprint(f"ilxl_extent (rounded):\n{ilxl_extent!r}", kind="debug", verbosity=verbose)

    il_range = (ilxl_extent[0, 0], ilxl_extent[-1, 0])
    xl_range = (ilxl_extent[0, 1], ilxl_extent[1, 1])

    il_step = 1 if bin_size[1] == bin_size_region[1] else bin_size[1] // bin_size_region[1]  # using XLINE bin size
    il_indices = np.arange(il_range[0], il_range[-1] + 1, il_step)
    xl_step = 1 if bin_size[0] == bin_size_region[0] else bin_size[0] // bin_size_region[0]  # using ILINE bin size
    xl_indices = np.arange(xl_range[0], xl_range[-1] + 1, xl_step)
    xprint(f"# ilines: {il_indices.size:3d}   {il_range}", f", step: {il_step}", kind="info", verbosity=verbose)
    xprint(f"# xlines: {xl_indices.size:3d}   {xl_range}", f", step: {xl_step}", kind="info", verbosity=verbose)
    
    # compute center coordinates of iline/xline bins
    bins_ilxl = np.meshgrid(il_indices, xl_indices)
    bins_ilxl = np.asarray(bins_ilxl).T.reshape(-1, 2)
    bins_xy = transform_xy_to_ilxl.inverse().transform(bins_ilxl)

    # assign iline/xline numbers to each shotpoint
    ilxl = transform_xy_to_ilxl.transform(df_nav[["x", "y"]].to_numpy())
    if USE_REGION:
        step_min = min(il_step, xl_step)
        step_max = max(il_step, xl_step)
        cutoff = step_max / step_min
        if il_step > 1:  # map assigned data `iline` to output `iline` (with respect to `bin_size`)
            il_mapped = find_nearest_ilxl(bins_ilxl[:, 0], ilxl[:, 0])
            ilxl[:, 0] = np.where(
                np.logical_and.reduce((
                    np.abs(ilxl[:, 0] - il_mapped) < cutoff + 1,
                    ilxl[:, 0] >= (il_indices[0] - cutoff / 2),
                    ilxl[:, 0] <= (il_indices[-1] + cutoff / 2)
                )), il_mapped, ilxl[:, 0]
            )
        if xl_step > 1:  # map assigned data `xline` to output `xline` (with respect to `bin_size`)
            xl_mapped = find_nearest_ilxl(bins_ilxl[:, 1], ilxl[:, 1])
            ilxl[:, 1] = np.where(
                np.logical_and.reduce((
                    np.abs(ilxl[:, 1] - xl_mapped) < cutoff + 1,
                    ilxl[:, 1] >= (xl_indices[0] - cutoff / 2),
                    ilxl[:, 1] <= (xl_indices[-1] + cutoff / 2)
                )), xl_mapped, ilxl[:, 1]
            )
    df_ilxl = pd.DataFrame(data=np.around(ilxl, 0).astype("int32"), columns=["il", "xl"])
    warnings.warn(
        "\nCoordinates at the boundary between two ilines/xlines "
        + "are assigned to next SMALLER index (x.5 --> x)!"
    )

    # create temporary bin center dataframe
    cols_bin = ["il", "xl", "x", "y"]
    dtypes_bin = ["int32", "int32", "float64", "float64"]
    df_bins = pd.DataFrame(np.hstack((bins_ilxl, bins_xy)), columns=cols_bin).astype(
        dict(zip(cols_bin, dtypes_bin))
    )

    if return_geometry:
        extent_cube = transform_reverse.transform(cube_corner_pts_t)
        if USE_REGION:
            extent_region = transform_reverse.transform(region_corner_pts_t)
            region_corner_pts_t = region_corner_pts_t + np.array([
                    [ bin_size_region[1] / 2,  bin_size_region[0] / 2],  # noqa
                    [ bin_size_region[1] / 2, -bin_size_region[0] / 2],  # noqa
                    [-bin_size_region[1] / 2, -bin_size_region[0] / 2],  # noqa
                    [-bin_size_region[1] / 2,  bin_size_region[0] / 2],  # noqa
            ])
            region_center_pts = transform_reverse.transform(region_corner_pts_t)
        else:
            extent_region = region_center_pts = None
        return df_bins, df_ilxl, (extent_cube, extent_cube_t), (extent_region, extent_region_t), region_center_pts

    return df_bins, df_ilxl


def get_segy_header_dataframe(
    path: str,
    dir_seismic: str,
    df_ilxl: pd.DataFrame,
    df_bins: pd.DataFrame,
    byte_filter: list = [1, 5, 9, 71, 73, 77, 109, 115, 117],
    suffix: str = "sgy",
    parallel: bool = False,
    include_files: list = None,
    return_headers: bool = False,
    verbose: int = False,
):
    """
    Scrape SEG-Y headers and return pandas.DataFrame of selected header entries.

    Parameters
    ----------
    path : str
        Path of textfile with filenames of SEGY files to scrape.
    dir_seismic : str
        Directory of SEGY files.
    df_ilxl : pd.DataFrame
        Dataframe of iline/xline indices for each trace position (from auxiliary navigation file).
    df_bins : pd.DataFrame
        Dataframe of cube bins featuring iline/xline indices and bin center coordinates.
    byte_filter : list, optional
        List of byte indices to scrape.
    suffix : str, optional
        SEG-Y file suffix (default: `sgy`).
    parallel : bool, optional
        Perform operation in parallel using `dask.delayed` (default: `False`).
    include_files : list, optional
        List of SEG-Y files to scrape headers from (default: `None`).
        Useful if more SEG-Y files in specified folder/datalist.
    return_headers : bool, optional
        Whether to return DataFrame of all available traces (default: False).
    verbose : int, optional
        Print optional information to console (default: `False`).

    Returns
    -------
    pandas.DataFrame
        Dataframe of scraped header information with inline/xline numbers for each trace.

    """
    # load list of SEG-Y files to scrape headers
    if path is None:
        files_segy = glob.glob(os.path.join(dir_seismic, f"*.{suffix}"))
    else:
        with open(path, "r") as f:
            files_segy = f.read().splitlines()
            files_segy = sorted(files_segy)

    if include_files is not None:
        files_segy = [
            file for file in files_segy if any(pattern in file for pattern in include_files)
        ]

    # scrape essential header information from segys
    xprint(
        f"Scrape trace headers of > {len(files_segy)} < SEG-Y files", kind="info", verbosity=verbose
    )
    if parallel:
        segy_header_scrape_parallel = dask.delayed(segy_header_scrape)

        list_headers = [
            segy_header_scrape_parallel(
                os.path.join(dir_seismic, segyf), bytes_filter=byte_filter, silent=True
            )
            for segyf in files_segy
        ]

        with show_progressbar(ProgressBar(), verbose=verbose):
            list_headers = dask.compute(*list_headers)  # single-machine scheduler
    else:
        list_headers = [
            segy_header_scrape(
                os.path.join(dir_seismic, segyf), bytes_filter=byte_filter, silent=False
            )
            for segyf in files_segy
        ]

    # merge scraped headers into single Dataframe
    df_headers = (
        pd.concat(
            list_headers,
            keys=list(np.arange(len(list_headers))),
        )
        .reset_index()
        .drop(columns=["level_1"])
        .rename(columns={"level_0": "line_id"})
    )
    df_headers["TRACE_SAMPLE_INTERVAL"] = df_headers["TRACE_SAMPLE_INTERVAL"] / 1000  # to ms
    df_headers["twt_max"] = (
        df_headers["DelayRecordingTime"]
        + df_headers["TRACE_SAMPLE_COUNT"] * df_headers["TRACE_SAMPLE_INTERVAL"]
    )
    df_headers["x"] = (
        df_headers["SourceX"] / df_headers["SourceGroupScalar"].abs()
        if df_headers.loc[0, "SourceGroupScalar"] < 0
        else df_headers["SourceX"] * df_headers["SourceGroupScalar"]
    )
    df_headers["y"] = (
        df_headers["SourceY"] / df_headers["SourceGroupScalar"].abs()
        if df_headers.loc[0, "SourceGroupScalar"] < 0
        else df_headers["SourceY"] * df_headers["SourceGroupScalar"]
    )
    df_headers.drop(columns=["SourceGroupScalar", "SourceX", "SourceY"], inplace=True)

    # join df with line names
    df_headers = pd.merge(
        df_headers,
        pd.Series(data=files_segy, name="line"),
        left_on="line_id",
        right_index=True,
        sort=False,
    )
    df_headers["line"] = df_headers["line"].astype("category")

    # add iline/xline columns
    # WARNING: use `include_files` when using subset of available SEG-Y files!
    df_headers = pd.concat((df_headers, df_ilxl), axis="columns", copy=False)

    # merge SEGY headers with bin center coords
    #   how='inner' --> drop all traces outside defined cube extent
    df_extent = pd.merge(
        df_headers,
        df_bins,
        how="inner",
        on=["il", "xl"],
        suffixes=[None, "_bin"],
        sort=False,
    )

    xprint(
        f"Traces: >{len(df_extent)}< valid out of >{len(df_headers)}< traces within extent ({len(df_extent)/len(df_headers)*100:.2f}%)",
        kind='info',
        verbosity=verbose,
    )
    # nbins = np.prod((df_bins.il.max() - df_bins.il.min(), df_bins.xl.max() - df_bins.xl.min()))
    # nbins = np.prod((np.unique(df_bins.il).size, np.unique(df_bins.xl).size))
    # xprint(
    #     f"Bin fold (approx.):  {len(df_extent) / nbins * 100:.2f}%  ({len(df_extent)} out of {nbins} bins)",
    #     kind="info",
    #     verbosity=verbose,
    # )
    
    if return_headers:
        return df_extent, df_headers

    return df_extent


def get_segy_binary_header(
    path: str = None,
    segy_dir: str = None,
    suffix: str = "sgy",
    byte_filter: list = None,
    check_all: bool = False,
    return_byte_keys: bool = False,
    parallel: bool = False,
    verbose: bool = False,
    **segyio_kwargs,
):
    """
    Scrape binary header fields of SEG-Y file(s).

    Specify either single datalist (`path`) or folder with SEG-Y files (`segy_dir`).

    Parameters
    ----------
    path : str, optional
        File path of datalist with SEG-Y file(s). Either `path` or `segy_dir` must be specified.
    segy_dir : str, optional
        Directory of SEG-Y file(s). Either `path` or `segy_dir` must be specified.
    suffix : str, optional
        SEG-Y file suffix (default: `sgy`).
    byte_filter : list, optional
        Exclude specified byte locations from check if `check_all` is True.
    check_all : bool, optional
        Check binary header of all SEG-Y files (default: `False`).
        Raise warning if different values are detected.
    return_byte_keys : bool, optional
        Return byte locations as keys. The default is a descriptive label.
    parallel : bool, optional
        Scrape SEG-Y files in parallel using `dask.delayed` (default: `False`).
    verbose : bool, optional
         Print optional information to console (default: `False`).
    **segyio_kwargs : dict
        Optional kwargs for `segyio.open()`.

    Raises
    ------
    ValueError
        If neither `path` nor `segy_dir` are specified.

    Warns
    -----
    UserWarning
        If different binary headers are found in searched SEG-Y files.

    Returns
    -------
    list_headers : dict
        Binary header dictionary.

    """
    # load list of SEG-Y files to scrape headers
    if path:
        with open(path, "r") as f:
            segy_dir = os.path.dirname(path)
            files_segy = f.read().splitlines()
            files_segy = sorted([os.path.join(segy_dir, f) for f in files_segy])
    elif segy_dir:
        files_segy = glob.glob(os.path.join(segy_dir, f"*.{suffix}"))
    else:
        raise ValueError("Either `path` or `segy_dir` must be specified!")

    segyio_kwargs["strict"] = False
    if check_all:
        if parallel:
            segy_bin_scrape_parallel = dask.delayed(segy_bin_scrape)

            list_headers = [
                segy_bin_scrape_parallel(segyf, **segyio_kwargs) for segyf in files_segy
            ]

            with show_progressbar(ProgressBar(), verbose=verbose):
                list_headers = dask.compute(*list_headers)  # single-machine scheduler
        else:
            list_headers = [segy_bin_scrape(segyf, **segyio_kwargs) for segyf in files_segy]
    else:
        list_headers = [segy_bin_scrape(files_segy[0], **segyio_kwargs)]

    if return_byte_keys:
        list_headers = [
            {b: v for b, (k, v) in zip(segyio.binfield.keys.values(), list_headers[i].items())}
            for i in range(len(list_headers))
        ]

    if check_all:
        from functools import reduce

        segyio_binfield_values = dict([(value, key) for key, value in segyio.binfield.keys.items()])
        if byte_filter is None:
            byte_filter = [
                segyio.BinField.Samples
                if return_byte_keys
                else segyio_binfield_values[segyio.BinField.Samples],
            ]
        else:
            byte_filter = [
                byte if return_byte_keys else segyio_binfield_values[byte] for byte in byte_filter
            ]
        # filter using `byte_filter`
        list_headers = [
            {k: v for k, v in bheader.items() if k not in byte_filter} for bheader in list_headers
        ]
        # searching for differences in binary headers
        diff = [r for r in reduce(set.intersection, (set(h.items()) for h in list_headers))]
        diff = [k for k in list_headers[0].keys() if k not in list(dict(diff).keys())]
        if len(diff) > 0:
            warnings.warn(
                (
                    "Found different values in binary headers of SEG-Y files:"
                    f"\n> {', '.join(diff)} <"
                ),
                UserWarning,
            )

    return list_headers[0]


def open_seismic_netcdfs(
    dir_seismic: str,
    suffix: str = "seisnc",
    datalist_path: str = None,
    include_files: list = None,
    kwargs_seisnc: dict = None,
    parallel: bool = True,
    verbose: bool = False,
):
    """
    Open multiple seismic netCDF files and return list of SEGY datasets.

    Parameters
    ----------
    dir_seismic : str
        Directory of seismic files.
    suffix : str, optional
        File suffix (default: `seisnc`).
    datalist_path : str, optional
        Filter available seismic files based on list of SEGY files to use.
    include_files : list, optional
        List of SEG-Y files to scrape headers from (default: `None`).
        Useful if more SEG-Y files in specified folder/datalist.
    kwargs_seisnc : dict, optional
        Keyword arguments for segysak.open_seisnc (default: `None`).
    parallel : bool, optional
        Open netCDF files in parallel (default: `True`).
    verbose : bool, optional
        Print optional information to console (default: `False`).

    Returns
    -------
    datasets : list
        List of SEGY datasets.

    """
    if kwargs_seisnc is None or kwargs_seisnc == {}:
        kwargs_seisnc = dict(chunks={"cdp": 1})

    # list of all seismic files
    files_seisnc = sorted(glob.glob(dir_seismic + f"/*.{suffix}"))
    if len(files_seisnc) == 0:
        raise IOError(f'No input files (*.{suffix}) found in "{dir_seismic}"')

    if datalist_path is not None and os.path.isfile(datalist_path):
        # filter using datalist suffix
        datalist_suffix = os.path.splitext(os.path.basename(datalist_path))[0].split("_")[-1]
        # files_seisnc = [f for f in files_seisnc if datalist_suffix in f]
        files_seisnc = [
            f
            for f in files_seisnc
            if datalist_suffix == os.path.splitext(os.path.basename(f))[0].split("_")[-1]
        ]

        # filter by lines within cube extent
        with open(datalist_path, "r") as f:
            files_filter = f.read().splitlines()
            files_filter = sorted(files_filter)

        files_seisnc = [
            f
            for f in files_seisnc
            if any(os.path.splitext(os.path.basename(f))[0] in fu for fu in files_filter)
        ]

    xprint(
        f"Found > {len(files_seisnc)} < input files (*.{suffix})", kind="info", verbosity=verbose
    )

    if include_files is not None:
        files_seisnc = [f for f in files_seisnc if any(x in f for x in include_files)]
        xprint(
            f"Inclusion filter returned > {len(files_seisnc)} < input files",
            kind="info",
            verbosity=verbose,
        )

    if parallel:
        open_seisnc_parallel = dask.delayed(open_seisnc)
        datasets = [open_seisnc_parallel(f, **kwargs_seisnc) for f in files_seisnc]
        with show_progressbar(ProgressBar(), verbose=verbose):
            datasets = dask.compute(*datasets)  # single-machine scheduler
    else:
        datasets = [open_seisnc(f, **kwargs_seisnc) for f in files_seisnc]

    return datasets


def inlines_from_seismic(
    df: pd.DataFrame,
    df_bins: pd.DataFrame,
    datasets: list,
    out_dir: str,
    bin_size: int,
    bin_size_str: str,
    stacking_method: str = "average",
    factor_dist: float = 1.0,
    twt_minmax: tuple = None,
    encode: bool = False,
    is_envelope: bool = None,
    dtype_data: str = None,
    ilxl_labels: list = ["il", "xl"],
    kwargs_nc: dict = None,
    verbose: int = False,
) -> None:
    """
    Create inline netCDF files from seismic profiles (*.seisnc).
    Using auxiliary information from SEGY header scrape and one of the following stacking methods:

      - `average`
      - `median`
      - `nearest`
      - `IDW` (_Inverse Distance Weighting_)

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of scraped SEGY header information and assigned iline/xline bin indices.
    df_bins : pd.DataFrame
        Dataframe of cube iline/xline indices and bin center coordinates.
    datasets : iterable[tuple, list]
        Iterable of xr.Datasets of opened *.seisnc files on disk.
    out_dir : str
        Output directory.
    bin_size : float | tuple(float, float)
        Bin size in inline/xline direction (in CRS units, e.g. meter).
    bin_size_str : str
        String representation of `bin_size` (suitable for filenames).
    stacking_method : str, optional
        Method to stack traces within single bin (default: `average`):
            ['average', 'median', 'nearest', 'IDW']
    factor_dist : float, optional
        User-specified factor controlling the severity of weighting function: 1/(dist**factor).
        Larger values (>1) increase influence of closer while reducing impact of more distant traces
        and vis versa for smaller values (<1).
    twt_minmax : tuple
        Minimum and maximum TWT (in ms): (min, max)
    encode : bool, optional
        Use encoding to compress output file size (default: `False`).
    is_envelope : bool, optional
        Boolean variable defining if input traces are provided as envelope (of amplitude).
    dtype_data : str, optional
        Output data array dtype. Use input datasets dtype as default.
    ilxl_labels : list
        List of iline/xline column names in `df`.
    kwargs_nc : dict
        Dictionary of netCDF attributes and parameters (from YAML file).
    verbose : int, optional
        Print optional information to console (default: `False`).

    """

    def _norm_weights_per_bin(df_subset, factor_dist, col="dist_bin_center"):
        """Compute normalized weights from distance to bin center (per bin)."""
        weights = 1 / df_subset[col] ** factor_dist
        df_subset["weights_norm"] = weights / weights.sum()
        return df_subset

    # init stacking methods
    fold_cnts = None
    if stacking_method in ["nearest", "IDW"]:
        # calculate distance of traces to assigned bin center
        col_dist = "dist_bin_center"
        df[col_dist] = distance(df[["x", "y"]].to_numpy(), df[["x_bin", "y_bin"]].to_numpy())

    if stacking_method == "IDW":
        df = df.groupby(by=ilxl_labels, group_keys=False).apply(
            _norm_weights_per_bin, factor_dist=factor_dist, col=col_dist
        )
    elif stacking_method == "nearest":
        fold_cnts = df.groupby(by=ilxl_labels)[col_dist].count()
        df = df.loc[df.groupby(by=ilxl_labels)[col_dist].idxmin()]
    elif stacking_method == "average":
        func = dask.array.mean
    elif stacking_method == "median":
        func = dask.array.median

    # create list of ilines to combine
    df = df.sort_values(ilxl_labels[0])
    idx_split = np.nonzero(np.diff(df[ilxl_labels[0]]))[0] + 1
    inlines = {il["il"].iloc[0]: il.sort_values(ilxl_labels[1]) for il in np.split(df, idx_split)}

    # sample rate
    dt = df.loc[df.index.min(), "TRACE_SAMPLE_INTERVAL"]

    il_indices = df_bins["il"].unique()  # iline indices from dataframe
    xl_indices = df_bins["xl"].unique()  # xline indices from dataframe

    # define global TWT range (i.e. number of samples/trace)
    if twt_minmax is not None:
        twt_start, twt_end = twt_minmax
    else:
        twt_start = df["DelayRecordingTime"].max()
        twt_end = df["twt_max"].min()

    nsamples = int((twt_end - twt_start) / dt)
    shape = (xl_indices.size, nsamples)  # (nxlines, nsamples)
    xprint(
        f"Using data between > {twt_start} < and > {twt_end} < ms TWT ({nsamples} samples @ {dt} ms sampling rate)",
        kind="info",
        verbosity=verbose,
    )

    # create global TWT array (defines cube time axis)
    twt = np.around(np.arange(twt_start, twt_end, dt, dtype=np.float64), 5)

    # check for amplitude envelopes
    if is_envelope is None:
        perc_ = datasets[0].attrs.get("percentiles", None)
        is_envelope = perc_.min() >= 0.0 if perc_ is not None else False

    if is_envelope:
        var_name = "env"
    else:
        var_name = "amp"

    # get attributes
    if kwargs_nc is not None:
        ATTRIBUTES = kwargs_nc["attrs_time"]
        attrs_data = ATTRIBUTES.get(var_name, {})
        encoding = kwargs_nc["encodings"].get(var_name, {})
    else:
        attrs_data = {}
        encoding = {}

    # create output directory if not existent
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # output data array dtype
    if dtype_data is None:
        dtype_data = datasets[0].data.dtype
    else:
        dtype_data = np.dtype(dtype_data)

    # get textual header from first SEG-Y dataset
    textual_header = datasets[0].attrs.get("text", None)
    if textual_header is not None:
        textual_header = "\n".join(
            [
                line[4:]
                for line in textual_header.split("\n")
                if re.search(r"^\d{4}-\d{2}-\d{2}:", line[4:])  # noqa
            ]  # search for time string
        )

    xprint(f"Creating > {len(il_indices)} < inline netCDF files", kind="info", verbosity=verbose)

    # loop to create individual inlines of 3D cube
    for id_iline in tqdm(
        il_indices,
        desc="Creating inlines",
        ncols=80,
        total=len(il_indices),
        unit_scale=True,
        unit=" files",
    ):
        # get inline ID
        # id_iline = int(inline[ilxl_labels[0]].iloc[0])

        # get inline df fro
        inline = inlines.get(id_iline, None)

        # initialize fold array
        fold = xr.DataArray(
            name="fold",
            data=dask.array.zeros((shape[0],), dtype=np.uint8),
            coords={"xline": ("xline", xl_indices)},
            attrs=ATTRIBUTES.get("fold", {}) if kwargs_nc is not None else {},
        )

        if inline is not None:
            # ----- TRACES -----
            # list of all traces in inline (lazy dask.arrays)
            traces = [
                datasets[li]["data"].sel(cdp=cdp).data
                for li, cdp in inline[["line_id", "TRACE_SEQUENCE_FILE"]].to_numpy()
            ]

            # ----- PADDING -----
            # pad traces to fit number of samples (global)
            traces_pad = [
                pad_trace(t, delrt, twt, dt)
                for t, delrt in zip(traces, inline.DelayRecordingTime.to_numpy())
            ]

            # ----- STACKING -----
            # get xline indices as list
            xl_list = inline[ilxl_labels[1]].to_list()

            # get unique xline indices
            xl_uniq, xl_idx, xl_cnt = np.unique(xl_list, return_index=True, return_counts=True)

            # `stack` (sum) traces with same xline index
            if stacking_method in ["average", "median"]:
                traces_stk = [
                    traces_pad[j]
                    if xl_cnt[i] == 1
                    else func(dask.array.stack(traces_pad[j : j + xl_cnt[i]]), axis=0)
                    for i, j in enumerate(xl_idx)
                ]
            elif stacking_method == "nearest":
                traces_stk = [traces_pad[j] for i, j in enumerate(xl_idx)]
            elif stacking_method == "IDW":
                traces_stk = [
                    traces_pad[j]
                    if xl_cnt[i] == 1
                    else (
                        dask.array.stack(traces_pad[j : j + xl_cnt[i]])
                        * inline["weights_norm"][j : j + xl_cnt[i]].values[..., np.newaxis]
                    ).sum(axis=0)
                    for i, j in enumerate(xl_idx)
                ]

            # assign `fold` numbers
            fold.loc[xl_uniq] = xl_cnt if fold_cnts is None else fold_cnts.loc[id_iline].values

            # ----- CREATE INFILL TRACES -----
            # get indices of infill traces
            mask = np.isin(xl_indices, xl_uniq, assume_unique=True)
            xl_indices_gaps = xl_indices[~mask]

            # init infill traces
            traces_gaps = [
                dask.array.zeros((twt.size,), chunks=(twt.size,), dtype=dtype_data)
                for n in range(xl_indices_gaps.size)
            ]

            # combine and sort traces in correct order
            sorter = np.argsort(np.concatenate([xl_uniq, xl_indices_gaps]))
            traces_comb = traces_stk + traces_gaps
            traces_comb = [traces_comb[i] for i in sorter]
            iline_dask_stk = dask.array.stack(traces_comb).astype(dtype_data)

        else:
            iline_dask_stk = dask.array.zeros(shape, dtype=dtype_data)

        # create inline dataset
        file_name = f"iline_{id_iline:04d}_{bin_size_str}"
        _stacking_method = (
            f"{stacking_method} (distance factor={factor_dist})"
            if stacking_method == "IDW"
            else stacking_method
        )

        # mask of iline xy coordinates
        # mask_il = np.nonzero(bins_ilxl[:,0] == id_iline)[0]
        inline_bins = df_bins[df_bins["il"] == id_iline]

        crs = pyproj.CRS(kwargs_nc.get("spatial_ref"))
        attrs = {
            "long_name": f"Inline #{id_iline:d}",
            "bin_units": "m",
            "measurement_system": 'm' if crs.is_projected else 'deg',
            "epsg": crs.to_epsg(),
            "stacking_method": _stacking_method,
            "spatial_ref": kwargs_nc["spatial_ref"] if kwargs_nc is not None else "None",
            "text": textual_header if textual_header is not None else "",
        }
        if isinstance(bin_size, (int, float)):
            attrs["bin_size"] = bin_size
            bin_il = bin_xl = bin_size
        else:
            # bin size along `iline` dimenstion == distance between crosslines
            attrs["bin_size_iline"] = bin_il = bin_size[0]
            # bin size along `xline` dimenstion == distance between inlines
            attrs["bin_size_xline"] = bin_xl = bin_size[1]

        ds = xr.Dataset(
            data_vars={var_name: (("xline", "twt"), iline_dask_stk, attrs_data), "fold": fold},
            coords={
                "xline": ("xline", xl_indices),
                "twt": ("twt", twt),
                "iline": id_iline,
                "x": ("xline", inline_bins["x"].values),  # bins_xy[mask_il, 0]),
                "y": ("xline", inline_bins["y"].values),  # bins_xy[mask_il, 1]),
            },
            attrs=attrs,
        )

        # add attributes to coordinates
        if kwargs_nc is not None:
            # update attributes
            ATTRIBUTES.get("twt", {}).update({"dt": dt})
            ATTRIBUTES.get("iline", {}).update(
                {
                    "bin_il": bin_il,
                    "comment": "`bin_il` is the bin distance ALONG dim `iline` and NOT the inline spacing",
                }
            )
            ATTRIBUTES.get("xline", {}).update(
                {
                    "bin_xl": bin_xl,
                    "comment": "`bin_xl` is the distance ALONG dim `xline` and NOT the crossline spacing",
                }
            )

            for coord in ["iline", "xline", "twt", "x", "y"]:
                ds[coord].attrs.update(ATTRIBUTES.get(coord, {}))

        with dask.config.set(scheduler="threads"):
            # ~2x speedup when "pre-compute" array before export
            ds.load().to_netcdf(
                os.path.join(out_dir, f"{file_name}.nc"),
                engine="h5netcdf",
                encoding={var_name: encoding} if encode else None,
            )


def merge_inlines_to_cube(
    inlines_dir,
    cube_path,
    kwargs_mfdataset: dict = None,
    kwargs_nc: dict = None,
    verbose: int = False,
) -> None:
    """
    Merge individual inline netCDF files into single 3D cube netCDF.

    Parameters
    ----------
    inlines_dir : str
        Directory of inline netCDF files.
    cube_path : str
        Path of cube file on disk.
    kwargs_mfdataset : dict, optional
        Keyword arguments for xarry.open_kwargs_mfdataset (default: `None`).
    kwargs_nc : dict
        Dictionary of netCDF attributes and parameters (from JSON file).
    verbose : int, optional
        Print optional information to console (default: `False`).

    Returns
    -------
    cube : xr.Dataset
        Merged dataset from individual inline files.

    """
    if kwargs_mfdataset is None or kwargs_mfdataset == {}:
        kwargs_mfdataset = dict(
            chunks={"iline": 1, "xline": -1, "twt": 1000},
            combine="nested",
            concat_dim="iline",
            engine="h5netcdf",
            parallel=True,
            coords="all",
        )

    # open all iline_*.nc netCDFs as one combined dataset
    cube = xr.open_mfdataset(inlines_dir, **kwargs_mfdataset)

    # compute fold mask
    cube["fold"] = cube["fold"].load()
    if verbose:
        _fold = np.count_nonzero(cube["fold"].data)
        _bins = np.prod(cube["fold"].shape)
    xprint(
        f"Bin fold:  {_fold/_bins*100:.2f}%  ({_fold} out of {_bins} bins)", kind="info", verbosity=verbose,
    )

    # add global attributes
    if kwargs_nc is not None:
        cube.attrs.update(kwargs_nc["attrs_time"].get("cube", {}))
    cube["fold"].attrs.update(
        {"coverage_perc": round(np.count_nonzero(cube["fold"].values) / cube["fold"].size * 100, 2)}
    )

    # export combined lines to single netCDF
    xprint(
        f"Combine > {cube.iline.size} < merged inlines to single netCDF file",
        kind="info",
        verbosity=verbose,
    )
    with show_progressbar(ProgressBar(), verbose=verbose):
        cube.to_netcdf(cube_path, engine=kwargs_mfdataset["engine"])

    return cube


def transpose_slice_major(
    cube_path, cube_twt_path, kwargs_dataset: dict = None, verbose: int = False
) -> None:
    """
    Transpose cube file (inline, xline, twt) to time-major layout (twt, inline, xline).
    Practical for faster time/frequency slice access.

    Parameters
    ----------
    cube_path : str
        Path of cube file on disk.
    cube_twt_path : str
        Output file path of transposed cube.
    kwargs_dataset : dict, optional
        Keyword arguments for xarry.open_dataset (default: `None`).
    verbose : int, optional
        Print optional information to console (default: `False`).

    Returns
    -------
    cube_file : xr.Dataset
        Dataset of sparse 3D cube on disk.

    """
    if kwargs_dataset is None or kwargs_dataset == {}:
        kwargs_dataset = dict(chunks={"iline": -1, "xline": -1, "twt": 500}, engine="h5netcdf")
    # open cube
    cube_file = xr.open_dataset(cube_path, **kwargs_dataset)

    # save transposed cube
    xprint(
        "[INFO]    Transpose sparse 3D cube to time-major layout (twt, inline, xline)",
        kind="info",
        verbosity=verbose,
    )
    with show_progressbar(ProgressBar(), verbose=verbose):
        cube_file.transpose("twt", ...).to_netcdf(cube_twt_path, engine=kwargs_dataset["engine"])

    return cube_file


# fmt: off
def define_input_args():  # noqa
    parser = argparse.ArgumentParser(description="Create sparse 3D cube from several 2D profiles.")
    parser.add_argument("path_input", type=str, help="Input directory or path to datalist.")
    parser.add_argument(
        "--params_netcdf", type=str, required=True,
        help="Path of netCDF parameter file (YAML format).",
    )
    parser.add_argument(
        "--params_spatial_ref", type=str, required=True,
        help="Path of spatial reference parameter file with CRS as WKT string (YAML format).",
    )
    parser.add_argument(
        "--params_cube_setup", type=str, required=True,
        help="Path of config file for cube geometry setup (YAML format).",
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, help="Output directory for edited SEG-Y file(s)."
    )
    parser.add_argument(
        "--suffix", "-s", type=str, help='File suffix. Only used when "path_input" is a directory.'
    )
    parser.add_argument(
        "--filename_suffix", "-fns", type=str,
        help='Filename suffix for guided selection (e.g. "env" or "despk"). Only used when "input_path" is a directory.',
    )
    parser.add_argument(
        "--attribute", "-a", type=str, choices=["amp", "env"], help="Seismic attribute to compute."
    )
    # coordinates
    parser.add_argument(
        "--coords_origin", choices=["header", "aux"], default="header",
        help="Origin of (shotpoint) coordinates (i.e. navigation).",
    )
    parser.add_argument(
        "--path_coords", type=str, required=True,
        help="Path to SEG-Y directory (coords_origin == header) or auxiliary "
        + "navigation file with coordinates (coords_origin == aux).",
    )
    parser.add_argument(
        "--coords_fsuffix", type=str,
        help="File suffix of auxiliary or SEG-Y files (depending on chosen parameter for `coords_origin`.",
    )
    # binning parameters
    parser.add_argument(
        "--bin_size", type=float, nargs="+",
        help="Bin size(s) in inline and crossline direction(s) given in CRS units (e.g., meter). \
              Single value or space-separated `inline` and `crossline` values.",
    )
    parser.add_argument(
        "--twt_limits", type=float, nargs="+",
        help="Vertical two-way travel time range of output 3D cube (in ms).",
    )
    parser.add_argument(
        "--parallel", action="store_true",
        help="Process files in parallel (requires dask as backend!).",
    )
    parser.add_argument(
        "--encode", action="store_true",
        help="Use encoding to compress output file size based on JSON parameter file (``params_netcdf``).",
    )
    parser.add_argument(
        "--stacking_method", type=str, choices=STACK_METHODS,
        help="Stacking method for multiple traces within one bin.",
    )
    parser.add_argument(
        "--factor_dist", type=float, default=1.0,
        help='Distance factor controlling the impact of weighting function: 1/(distance**factor). \
                        Only used if stacking_method="IDW".',
    )
    #
    parser.add_argument(
        "--dtype_data", type=str, default="float32", help="Output dtype of created 3D cube."
    )
    parser.add_argument(
        "--name", type=str, default="", help="Optional identifier string to add to exported files."
    )
    parser.add_argument(
        "--write_aux", action="store_true",
        help="Write auxiliary files featuring key cube parameters.",
    )
    #
    parser.add_argument(
        "--verbose", "-V", type=int, nargs="?", default=0, const=1, choices=[0, 1, 2],
        help="Level of output verbosity (default: 0)",
    )
    return parser
# fmt: on


def main(argv=sys.argv, return_dataset=False):  # noqa
    TODAY = datetime.date.today().strftime("%Y-%m-%d")
    SCRIPT = os.path.splitext(os.path.basename(__file__))[0]

    parser = define_input_args()
    args = parser.parse_args(argv[1:])  # exclude filename parameter at position 0

    path_input = args.path_input
    dir_work, filename = os.path.split(path_input)
    basename, suffix = os.path.splitext(filename)
    if suffix == "":
        dir_work = args.path_input
        basename, suffix = None, None  # noqa

    path_params_nc = args.params_netcdf
    path_spatial_ref = args.params_spatial_ref
    path_params_cube = args.params_cube_setup

    dir_out = args.output_dir if args.output_dir is not None else dir_work

    if args.coords_origin == "header":
        fsuffix = args.coords_fsuffix if args.coords_fsuffix is not None else "sgy"
    elif args.coords_origin == "aux":
        fsuffix = args.coords_fsuffix if args.coords_fsuffix is not None else "nav"

    parallel = args.parallel
    verbose = args.verbose
    WRITE_AUX = args.write_aux

    # === Open and read config files ===

    # load netCDF metadata
    with open(path_params_nc, "r") as f_attrs, open(path_spatial_ref, "r") as f_crs:
        kwargs_nc = yaml.safe_load(f_attrs)
        kwargs_nc["spatial_ref"] = yaml.safe_load(f_crs)

    # load cube geometry config file
    xprint("Load cube geometry parameter from config file", kind="info", verbosity=verbose)
    with open(path_params_cube, mode="r") as f:
        cfg = yaml.safe_load(f)

    # BIN SIZES
    if args.bin_size is not None:
        bin_size = (
            (args.bin_size[0], args.bin_size[0])
            if len(args.bin_size) == 1
            else tuple(args.bin_size)
        )
    elif cfg.get("bin_size") is not None:
        bin_size = (
            (cfg["bin_size"], cfg["bin_size"])
            if isinstance(cfg["bin_size"], (int, float))
            else tuple(cfg["bin_size"])
        )
    else:
        raise ValueError(
            "`bin_size` is required! Either (1) as command line parameter or "
            + "(2) from cube geometry config file."
        )

    if isinstance(bin_size, (int, float)):
        bin_size_str = f"{bin_size:.0f}m" if bin_size % 1 == 0 else f"{bin_size}m"
    else:
        bin_size_iline = f"{bin_size[0]:.0f}" if bin_size[0] % 1 == 0 else f"{bin_size[0]}"
        bin_size_xline = f"{bin_size[1]:.0f}" if bin_size[1] % 1 == 0 else f"{bin_size[1]}"
        bin_size_str = f'{bin_size_iline.replace(".","+")}x{bin_size_xline.replace(".","+")}m'
        
    if "bin_size_region" in cfg.keys():
        bin_size_region = (
            (cfg["bin_size_region"], cfg["bin_size_region"])
            if isinstance(cfg["bin_size_region"], (int, float))
            else tuple(cfg["bin_size_region"])
        )
        bin_size_iline = f"{bin_size_region[0]:.0f}" if bin_size_region[0] % 1 == 0 else f"{bin_size_region[0]}"
        bin_size_xline = f"{bin_size_region[1]:.0f}" if bin_size_region[1] % 1 == 0 else f"{bin_size_region[1]}"
        bin_size_region_str = f'{bin_size_iline.replace(".","+")}x{bin_size_xline.replace(".","+")}m'
    else:
        bin_size_region = bin_size  # needed for comparison

    # TWT LIMITS
    if args.twt_limits is not None:
        twt_limits = tuple(args.twt_limits)
    elif cfg.get("twt_limits") is not None and len(cfg.get("twt_limits")) == 2:
        twt_limits = tuple(cfg["twt_limits"])
    else:
        raise ValueError(
            "`twt_limits` are required! Either (1) as command line parameter or "
            + "(2) from cube geometry config file."
        )

    # STACKING METHOD
    if args.stacking_method is not None:
        stacking_method = args.stacking_method
        factor_dist = args.factor_dist
    elif cfg.get("stacking_method") is not None and cfg.get("stacking_method") in STACK_METHODS:
        stacking_method = cfg["stacking_method"]
        factor_dist = cfg["factor_dist"]
    else:
        stacking_method = "average"
        factor_dist = None
        xprint(
            f"No `stacking_method` provided, using default method: < {stacking_method} >",
            kind="warning",
            verbosity=verbose,
        )

    # NAME: optional identifier for filenames (e.g. "cube_center")
    name = (
        f"{args.name}"
        if args.name != ""
        else cfg.get("name", f"{os.path.split(os.path.split(path_params_cube)[0])[-1]}")
    )
    name += f"_{stacking_method}"

    # ATTRIBUTE
    attr = args.attribute if args.attribute is not None else cfg.get("attribute", "")
    attr = f"_{attr}" if attr != "" else attr

    # update kwargs_nc defaults
    kwargs_nc["attrs_time"]["cube"].update({"long_name": cfg["long_name"]})
    kwargs_nc["attrs_time"]["cube"].update(
        {
            "history": kwargs_nc["attrs_time"]["cube"].get("history", "")
            + f"{SCRIPT}: create sparse 3D volume;",
            "text": kwargs_nc["attrs_time"]["cube"].get("text", "")
            + '\n=== 3D PROCESSING ==='
            + f"\n{TODAY}: 3D BINNING {stacking_method} ILINE:{bin_size_iline} "
            + f"XLINE:{bin_size_xline} UNIT:METER",
        }
    )

    # Setup cube geometry parameter
    # get CUBE extent (array of corner points)
    extent_pts_cube = np.asarray(list(cfg["extent_cube"].values()))

    rotation_angle = float(cfg["rotation_angle"])
    rotation_center = tuple(cfg.get("rotation_center", get_polygon_centroid(extent_pts_cube)))

    if "extent_region" in cfg.keys():
        USE_REGION = True
        # get STUDY AREA extent (array of corner points)
        extent_pts_region = np.asarray(list(cfg["extent_region"].values()))
    else:
        USE_REGION = False

    # [CRS] check for mismatches provided CRS and init extents
    crs_cube = pyproj.crs.CRS(kwargs_nc["spatial_ref"])
    crs_extent = pyproj.crs.CRS(cfg["spatial_ref"])
    if not crs_cube == crs_extent:
        warnings.warn(
            "Coordinate reference system mismatch found. "
            + "Reprojecting extent coordinates to cube CRS!"
        )

        transformer = pyproj.Transformer.from_crs(crs_extent, "EPSG:4326")

        # (1) transform CUBE coords
        extent_pts_cube = transformer.transform(extent_pts_cube[:, 0], extent_pts_cube[:, 1])
        extent_pts_cube = np.column_stack(extent_pts_cube)

        # (2) transform STUDY AREA coords
        extent_pts_region = transformer.transform(
            extent_pts_region[:, 0], extent_pts_region[:, 1]
        )
        extent_pts_region = np.column_stack(extent_pts_region)

    # Create coordinate transformation from center coordinate and rotation angle
    xprint("Create forward and inverse transformation matrices", kind="info", verbosity=verbose)
    transform_forward = Affine().rotate_around(angle=-rotation_angle, origin=rotation_center)
    transform_reverse = transform_forward.inverse()

    # [OPTIONAL] include only specific lines  # FIXME
    # FILTER_FILES = False
    #
    # if FILTER_FILES:
    #     include = 'xlines'
    #     path_include = f'C:/PhD/processing/TOPAS/TAN2006/TOPAS_only_{include}.csv'
    #     include_segys = pd.read_csv(path_include)
    #     include_segys_list = include_segys['line'].to_list()
    #     include_segys_list = ['_'.join(x.split('_')[:-1]) for x in include_segys_list]
    #     xprint(f'Using inclusion list of > {len(include_segys_list)} < files', kind='info', verbosity=verbose)

    # Load navigation of all TOPAS profiles
    if args.coords_origin == "aux":
        xprint(
            f"Load navigation from auxiliary files (*.{fsuffix})", kind="info", verbosity=verbose
        )
        df_nav = read_auxiliary_files(dir_work, fsuffix=fsuffix, prefix="2020", index_cols=None)
    elif args.coords_origin == "header":
        xprint("Extract and load navigation using input datalist", kind="info", verbosity=verbose)
        df_nav = extract_navigation_from_segy(path_input, write_aux=False)

    line_id, line_id_uniq = df_nav["line"].factorize()
    df_nav["line_id"] = line_id.astype(np.float32)

    # if FILTER_FILES:  # FIXME
    #     df_nav = df_nav[df_nav["line"].isin(include_segys_list)]

    # Affine transformation
    xprint("Create bins and ilxl dataframes", kind="info", verbosity=verbose)
    df_bins, df_ilxl, geom_cube, geom_region, bin_center_corners = get_cube_parameter(
        transform_forward,
        transform_reverse,
        df_nav,
        bin_size=bin_size,
        cube_corner_pts=extent_pts_cube,
        bin_size_region=bin_size_region,
        region_corner_pts=extent_pts_region if USE_REGION else None,
        return_geometry=True,
        verbose=verbose,
    )
    
    if WRITE_AUX:
        # [AUX] save dataframe of data bins with center coordiantes (il, xl, x, y)
        df_bins.to_csv(
            os.path.join(dir_out, f"aux_{name}_{bin_size_str}_bins.txt"), sep=",", index=False
        )

        # [AUX] save CUBE (and STUDY AREA) corner points as txt
        np.savetxt(
            os.path.join(dir_out, f"aux_{name}_{bin_size_str}_extent_corner_points.txt"),
            geom_cube[0],
            fmt="%.10f",
            delimiter=";",
            newline="\n",
            header="x;y",
            comments="",
        )
        if USE_REGION:
            np.savetxt(
                os.path.join(dir_out, f"aux_region_{bin_size_str}_extent_corner_points.txt"),
                geom_region[0],
                fmt="%.10f",
                delimiter=";",
                newline="\n",
                header="x;y",
                comments="",
            )
            np.savetxt(
                os.path.join(dir_out, f"aux_region_{bin_size_region_str}_outer_bin_center_points.txt"),
                bin_center_corners,
                fmt="%.10f",
                delimiter=";",
                newline="\n",
                header="x;y",
                comments="",
            )
    
    # header scrape
    df_extent, df_headers = get_segy_header_dataframe(
        path_input if os.path.isfile(path_input) else None,
        dir_work,
        df_ilxl,
        df_bins,
        parallel=parallel,
        # include_files=include_segys_list if FILTER_FILES else None,
        return_headers=True,
        verbose=verbose,
    )
    if WRITE_AUX:
        df_extent.to_csv(
            os.path.join(dir_out, f"aux_{name}_{bin_size_str}_selected_traces.txt"), sep=",", index=False
        )
        df_headers.to_csv(
            os.path.join(dir_out, f"aux_{name}_{bin_size_str}_all_traces.txt"), sep=",", index=False
        )
    
    # set file paths
    dt = check_sampling_interval(df_extent)

    fname = f'{name}{attr}_{bin_size_str}_{ffloat(dt).replace(".","+")}ms'
    # if FILTER_FILES:  # FIXME
    #     fname += f"_{include.upper()}"
    cube_path = os.path.join(dir_out, f"{fname}.nc")
    cube_twt_path = os.path.join(dir_out, f"{fname}_twt-il-xl.nc")
    path_inlines = os.path.join(dir_out, f"inlines_{fname}")

    # Open all netCDF files
    xprint("Opening seismic netCDF files for processing", kind="info", verbosity=verbose)
    datasets = open_seismic_netcdfs(
        dir_work,
        datalist_path=path_input if os.path.isfile(path_input) else None,
        # include_files=include_segys_list if FILTER_FILES else None,  # FIXME
        parallel=parallel,
        verbose=verbose,
    )

    # Create iline netCDF files from TOPAS profiles (*.seisnc)
    xprint("Computing inline netCDF files from seismic profiles", kind="info", verbosity=verbose)
    inlines_from_seismic(
        df_extent,
        df_bins,
        datasets=datasets,
        out_dir=path_inlines,
        bin_size=bin_size,
        bin_size_str=bin_size_str,
        stacking_method=stacking_method,
        factor_dist=factor_dist,
        twt_minmax=twt_limits,
        dtype_data=args.dtype_data,
        kwargs_nc=kwargs_nc,  # netCDF attributes and parameter
        verbose=verbose,
    )

    # [CUBE] Create single cube netCDF file from individual inlines
    cube_merged = merge_inlines_to_cube(
        os.path.join(path_inlines, "iline_*.nc"), cube_path, kwargs_nc=kwargs_nc, verbose=verbose
    )

    # [CUBE] Transpose to time-major layout: (il, xl, twt) --> (twt, il, xl)
    cube = transpose_slice_major(cube_path, cube_twt_path, verbose=verbose)
    
    if return_dataset:
        return cube_merged, cube


# %% MAIN

if __name__ == "__main__":

    main()
