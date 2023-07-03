"""Miscellaneous utility functions for file I/O."""

import os
import glob

import numpy as np
import pandas as pd
import segyio
from tqdm import tqdm

from .header import scale_coordinates

#%%
def read_and_merge(files, splitter='UTM', **kwargs_csv):
    """
    Read `files` list into combined `pandas.DataFrame`.

    Parameters
    ----------
    files : list
        List of file paths.
    splitter : str, optional
        String used to split filename to derive "original" filename (default: `UTM60S`).
    **kwargs_csv : dict
        Optional parameter for `pd.read_csv` function.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame of all read files.

    """
    list_loaded = []

    for file in tqdm(
        files,
        desc='Load auxiliary files',
        ncols=80,
        total=len(files),
        unit_scale=True,
        unit=' files',
    ):
        basepath, filename = os.path.split(file)
        basename, suffix = os.path.splitext(filename)

        df = pd.read_csv(file, **kwargs_csv)

        basename_splits = basename.split('_')
        # idx = basename_splits.index(splitter)
        idx = list(map(lambda x: splitter in x, basename_splits)).index(True)
        df['line'] = '_'.join(basename_splits[:idx])

        list_loaded.append(df)

    return pd.concat(list_loaded, ignore_index=True)


def read_auxiliary_files(
    path, fsuffix: str, prefix: str = None, suffix: str = None, index_cols=['line', 'tracl'], splitter: str = 'UTM',
):
    """
    Read auxiliary files (e.g. `nav`, `tide`, or `mistie`) into Pandas DataFrame.

    Parameters
    ----------
    path : str
        Path of (a) directory or (b) datalist with SEG-Y file(s).
    fsuffix : str
        File suffix.
    prefix : str, optional
        Filename prefix for filtering (default: `None`).
    suffix : str, optional
        Filename suffix for filtering (default: `None`)
    index_cols : list, optional
        Index columns of returned DataFrame.
    splitter : str
        Split filename using given string (default: `UTM`).

    Returns
    -------
    df : pd.DataFrame
        Combined DataFrame of all auxiliary files with (filtered using `fsuffix` and `prefix`).

    """
    basepath, filename = os.path.split(path)

    if fsuffix is not None and fsuffix.find('.') == -1:
        fsuffix = '.' + fsuffix

    # (A) path -> directory
    if os.path.isdir(path):

        files = sorted(glob.glob(os.path.join(path, f'*{fsuffix}')))
        if len(files) == 0:
            # raise FileNotFoundError('No auxiliary files found. Please check your directory!')
            return None
        if prefix is not None:
            files = [
                f for f in files if os.path.split(f)[-1].startswith(prefix)
            ]  # filter by prefix
        if suffix is not None:
            files = [
                f for f in files if os.path.splitext(os.path.split(f)[-1])[0].endswith(suffix)
            ]  # filter by suffix

    # (B) path -> datalist
    elif os.path.isfile(path) and path.endswith('.txt'):
        with open(path, 'r') as datalist:
            files = datalist.readlines()
            files = [
                os.path.join(basepath, os.path.splitext(line.rstrip())[0] + fsuffix)
                if os.path.split(line.rstrip()) not in ['', '.']
                else line.rstrip()
                for line in files
            ]
    else:
        raise IOError('Invalid input for `path` parameter. Should be either directory or datalist!')

    df = None
    if len(files) > 0:
        kwargs_csv = dict(sep=',')
        df = read_and_merge(files, splitter='UTM', **kwargs_csv)  # FIXME: UTM60S
        if index_cols is not None:
            df = df.set_index(index_cols, drop=True)

    return df


def export_coords(
    out_path,
    xcoords: np.ndarray,
    ycoords: np.ndarray,
    coordinate_units: int,
    index_label: str = None,
    aux_info: list = None,
    aux_cols: list = None,
) -> None:
    """
    Export function for (reprojected) coordinates.

    Parameters
    ----------
    out_path : str
        Output file path.
    xcoords : np.ndarray
        Array of X coordinates.
    ycoords : np.ndarray
        Array of Y coordinates.
    coordinate_units : int
        SEG-Y specific coordinate unit specifier integer.
    index_label : str, optional
        Label to use for pandas index column (default: None).
    aux_info : list, optional
        List of auxiliary SEG-Y header information to write (default: None).
    aux_cols : list, optional
        List of auxiliary SEG-Y header names (default: None).

    """
    kwargs = dict(sep=',', header=True, index=False, index_label=None, lineterminator='\n')
    if index_label is not None and index_label is not False:
        kwargs.update({'index': True, 'index_label': index_label})
        
    if aux_cols is None:
        aux_cols = ['tracl', 'tracr', 'fldr', 'n_traces', 'n_samples']

    # create pandas DataFrame
    if coordinate_units == 1:
        if aux_info is not None:
            cols = aux_cols + ['x', 'y']
            vals = aux_info + [xcoords, ycoords]
        else:
            cols = ['x', 'y']
            vals = [xcoords, ycoords]
        coords = pd.DataFrame(dict(zip(cols, vals)))
        kwargs['float_format'] = '%.2f'
    else:
        if aux_info is not None:
            cols = aux_cols + ['lat', 'lon']
            vals = aux_info + [ycoords, xcoords]
        else:
            cols = ['lat', 'lon']
            vals = [xcoords, ycoords]
        coords = pd.DataFrame(dict(zip(cols, vals)))
        kwargs['float_format'] = '%.6f'

    # save coordinate columns to text
    coords.to_csv(out_path, **kwargs)


def extract_navigation_from_segy(
    path,
    fsuffix: str = 'sgy',
    fnprefix: str = None,
    fnsuffix: str = None,
    splitter: str = 'UTM60S',
    src_coords_bytes: tuple = (73, 77),
    kwargs_segyio: dict = None,
    write_aux: bool = False,
) -> None:
    """
    Extract coordinates from SEG-Y file(s) in given directory or specified in datalist.

    Parameters
    ----------
    path : str
        Path of (a) directory or (b) datalist with SEG-Y file(s).
    fsuffix : str, optional
        File suffix (default: 'sgy'). Only used if `path` is a directory.
    fnprefix : str, optional
        Filename prefix for filtering (default: None). Only used if `path` is a directory.
    fnsuffix : str, optional
        Filename suffix for filtering (default: None). Only used if `path` is a directory.
    splitter : str, optional
        String used to split filename to derive "original" filename (default: 'UTM60S').
    src_coords_bytes : tuple, optional
        Byte position of coordinates in SEG-Y trace headers (default: (73, 77)).
    kwargs_segyio : dict, optional
        Parameter for `segyio.open()` (default: None).
    write_aux : bool, optional
        Write extracted navigation to individual auxiliary files (*.nav).

    """
    basepath, filename = os.path.split(path)

    # (A) path -> directory
    if os.path.isdir(path):
        if fsuffix.find('.') != -1:
            fsuffix = '*' + fsuffix
        else:
            fsuffix = '*.' + fsuffix

        files = sorted(glob.glob(os.path.join(path, fsuffix)))
        if fnprefix is not None:
            files = [
                f for f in files if os.path.split(f)[-1].startswith(fnprefix)
            ]  # filter by prefix
        if fnsuffix is not None:
            files = [
                f for f in files if os.path.splitext(os.path.split(f)[-1])[0].endswith(fnsuffix)
            ]  # filter by suffix

    # (B) path -> datalist
    elif os.path.isfile(path) and path.endswith('.txt'):
        with open(path, 'r') as datalist:
            files = datalist.readlines()
            files = [
                os.path.join(basepath, line.rstrip())
                if os.path.split(line.rstrip()) not in ['', '.']
                else line.rstrip()
                for line in files
            ]
    else:
        raise IOError('Invalid input for `path` parameter. Should be either directory or datalist!')

    if kwargs_segyio is None or kwargs_segyio == {}:
        kwargs_segyio = dict(strict=False, ignore_geometry=True)

    if len(files) > 0:
        list_df = []
        for file in tqdm(
            files, desc='Extract coords', ncols=80, total=len(files), unit_scale=True, unit=' files'
        ):
            basepath, filename = os.path.split(file)
            basename, suffix = os.path.splitext(filename)

            path_out = os.path.join(basepath, f'{basename}.nav')

            with segyio.open(file, 'r', **kwargs_segyio) as f:
                # get scaled coordinates
                xcoords, ycoords, coordinate_units = scale_coordinates(f, src_coords_bytes)

            # create pandas DataFrame
            df = pd.DataFrame(np.column_stack((xcoords, ycoords)), columns=['x', 'y'])
            basename_splits = basename.split('_')
            idx = basename_splits.index(splitter)
            df['line'] = '_'.join(basename_splits[:idx])

            if write_aux:
                # export coordinates to file
                export_coords(
                    path_out,
                    xcoords,
                    ycoords,
                    coordinate_units=coordinate_units,
                    index_label=None,
                    aux_info=None,
                )

            list_df.append(df)

        return pd.concat(list_df, ignore_index=True)
    else:
        return
