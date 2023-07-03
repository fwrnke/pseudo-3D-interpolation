"""Utility functions for SEG-Y header manipulations."""

import datetime
from warnings import warn

import numpy as np
import segyio

#%%
# *****************************************************************************
#                           TRACE HEADER
# *****************************************************************************
def scale_coordinates(segyfile, src_coords_bytes: tuple = (73, 77)):
    """
    Scale coordinates with `SourceGroupScalar` from seismic header.
    Returns arrays of scaled X and Y coordinates.

    Parameters
    ----------
    segyfile : segyio.SegyFile
        Input SEG-Y file object.
    src_coords_bytes : tuple, optional
        Byte position of coordinates in trace header (default: (73, 77)).

    Returns
    -------
    x : np.ndarray
        Array of X coordinates.
    y : np.ndarray
        Array of Y coordinates.
    CoordinateUnits : int
        Identifier for coordinate units.

    """
    xcoords_byte, ycoords_byte = src_coords_bytes  # unpack source coordinates byte IDs
    CoordinateUnits = segyfile.header[0].get(segyio.TraceField.CoordinateUnits)  # 89
    # print('CoordinateUnits:   ', CoordinateUnits)

    x, y = [], []
    for header in segyfile.header:
        x.append(header.get(xcoords_byte))  # get X coordinate
        y.append(header.get(ycoords_byte))  # get Y coordinate

    x, y = np.array(x), np.array(y)  # convert lists to arrays

    if CoordinateUnits == 1:  # length (meter or feet)
        SourceGroupScalar = segyfile.header[0].get(segyio.TraceField.SourceGroupScalar)  # 71
        # print('SourceGroupScalar: ', SourceGroupScalar)
        if SourceGroupScalar < 0:
            x = x / np.abs(SourceGroupScalar)
            y = y / np.abs(SourceGroupScalar)
        elif SourceGroupScalar > 0:
            x = x * np.abs(SourceGroupScalar)
            y = y * np.abs(SourceGroupScalar)
        elif SourceGroupScalar == 0:
            pass  # no scaling applied
    elif CoordinateUnits == 2:  # seconds of arc
        x = x / 3600000
        y = y / 3600000
    elif CoordinateUnits == 3:  # decimal degrees
        raise NotImplementedError('Functionality to convert DD data is not implemented.')
    elif CoordinateUnits == 4:  # degrees, minutes, seconds (DMS)
        raise NotImplementedError('Functionality to convert DMS data is not implemented.')

    return x, y, CoordinateUnits


def unscale_coordinates(
        X, Y,
        coords_bytes: tuple = (73, 77),
        coords_units: int = 1,
        scale_factor: int = -100
):
    """
    Convert X/Y coordinates to int32 using scale factor for SEG-Y writing.
    Returns arrays of 32-bit integer X and Y coordinates.

    Parameters
    ----------
    X : np.ndarray
        Input X coordinates.
    Y : np.ndarray
        Input Y coordinates.
    coords_bytes : tuple, optional
        Byte position of coordinates (default: `(73, 77)`).
    coords_units : int, optional
        SEG-Y specific coordinate unit identifier (default: `1`).
    scale_factor : int, optional
        Coordinate scaler for conversion from int to actual format (default: `-100`).

    Returns
    -------
    np.ndarray(s)
        Unscaled X and Y coordinate arrays.

    """
    xcoords_byte, ycoords_byte = coords_bytes  # unpack source coordinates byte IDs

    x, y = np.asarray(X), np.asarray(Y)  # convert lists to arrays

    if coords_units == 1:  # length (meter or feet)
        if scale_factor < 0:
            x = x * np.abs(scale_factor)
            y = y * np.abs(scale_factor)
        elif scale_factor > 0:
            x = x / np.abs(scale_factor)
            y = y / np.abs(scale_factor)
        elif scale_factor == 0:
            pass  # no scaling applied
    elif coords_units == 2:  # seconds of arc
        x = x * 3600000
        y = y * 3600000
    elif coords_units == 3:  # decimal degrees
        raise NotImplementedError('Functionality to convert DD data is not implemented.')
    elif coords_units == 4:  # degrees, minutes, seconds (DMS)
        raise NotImplementedError('Functionality to convert DMS data is not implemented.')

    return np.around(x, 0).astype('int'), np.around(y, 0).astype('int')


def set_coordinates(segyfile, X, Y, crs_dst, dst_coords_bytes=(73, 77), coordinate_units=1, scaler=-100):
    """
    Set X and Y coordinates using given input arrays.

    Parameters
    ----------
    segyfile : segyio.SegyFile
        Loaded SEG-Y file object.
    X : numpy.array
        Transformed X coordinates.
    Y : numpy.array
        Transformed Y coordinates.
    crs_dst : pyproj.crs.CRS
        Output CRS (of given coordinate arrays).
    dst_coords_bytes : tuple, optional
        Tuple of starting byte of X and Y coordinates in seismic trace header (default: `(73,77)`).
    coordinate_units: int,
        Integer code for coordinate units (default: `1` i.e. meter).
    scaler : int, optional
        Coordinate scaler for conversion from int to actual format (default: `-100`).

    """
    if crs_dst.is_geographic:
        raise NotImplementedError(
            'Functionality to convert to geographic output CRS is not yet implemented.'
        )
    elif crs_dst.is_projected:
        # unpack source coordinates byte IDs
        xcoords_byte, ycoords_byte = dst_coords_bytes

        # unscale coordinates
        X_unscale, Y_unscale = unscale_coordinates(
            X, Y, dst_coords_bytes, coordinate_units, scaler
        )

        # set (transformed) coordinates
        for i, h in enumerate(segyfile.header[:]):
            h.update(
                {
                    xcoords_byte: X_unscale[i],  # 73
                    ycoords_byte: Y_unscale[i],  # 77
                    segyio.TraceField.CoordinateUnits: coordinate_units,  # 89
                    segyio.TraceField.SourceGroupScalar: scaler,  # 71
                }
            )
    else:
        raise AttributeError('Issues with output CRS. Please check!')


def check_coordinate_scalar(
    coord_scalar: int,
    xcoords: np.ndarray = None,
    ycoords: np.ndarray = None,
):
    """
    Check input coordinate scalar.
    Return coordinate scalar (`coord_scalar`) and its mulitplier (`coord_scalar_mult`).

    Parameters
    ----------
    coord_scalar : int
        Input coordinate scalar (-1000, -100, -10, 0, 10, 100, 1000, or 'auto').

    Returns
    -------
    coord_scalar : int
        Coordinate scalar to apply.
    coord_scalar_mult : float
        Factor to scale coordinates.

    """
    # coord_scalar = args.scalar_coords
    if coord_scalar is None:
        coord_scalar = 0
    if coord_scalar == 'auto':
        max_digits = 10  # for 4 byte field --> 2,147,483,647 (int32 )
        n_digits_x = str(xcoords.flat[0]).find('.')
        n_digits_y = str(ycoords.flat[0]).find('.')
        n_digits = max(n_digits_x, n_digits_y)

        coord_scalar_mult = 10 ** (max_digits - n_digits - 1)
        coord_scalar = -coord_scalar_mult if coord_scalar_mult > 1 else int(1 / coord_scalar_mult)
    elif coord_scalar > 0:
        coord_scalar_mult = 1 / abs(coord_scalar)
    elif coord_scalar < 0:
        coord_scalar_mult = abs(coord_scalar)
    else:
        coord_scalar_mult = 1

    return coord_scalar, coord_scalar_mult


# *****************************************************************************
#                           TEXTUAL HEADER
# *****************************************************************************
def wrap_text(txt, width=80):
    """Format textual header for pretty printing."""
    return '\n'.join([txt[i : i + width] for i in range(0, len(txt), width)])


def whitespac_indices(s):
    """Return list of whitespace indices in given string."""
    return [i for i, c in enumerate(s) if c == ' ']


def find_empty_line(txt, splitter='\n'):
    """Return individual lines of textual header and index of first empty line."""
    if not isinstance(txt, list):
        lines = txt.split(splitter)
    else:
        lines = txt

    cnts = [r[3:].count(' ') for r in lines]
    try:
        line_idx = cnts.index(77)
    except ValueError:
        line_idx = None
    return lines, line_idx


def find_line_by_str(lines, search_str='PROCESSING'):
    """Return indices for lines starting with search string."""
    if not isinstance(lines, list):
        lines = lines.split('\n')
    if search_str is None:
        return []
    return [i for i in range(len(lines)) if lines[i][4:].startswith(search_str)]


def add_processing_info_header(
    txt, info_str, prefix=None, header=True, header_line=25, overwrite=True, newline=False
):
    """
    Add processing information annotation to textual header string.
    The info will be added to a line starting with `prefix` if provided
    and line is not filled.

    Parameters
    ----------
    txt : str
        Textual header string read from SEG-Y file.
    info_str : str
        String to add to header.
    prefix : str, optional
        Line prefix where to insert `info` (default: `None`).
    header : bool, str, optional
        If `True`: update header string with default header line
        If `str`:  use provided string for header line
    header_line: int, optional
        Line in textual header where to insert header line (default: `25`).
    overwrite: bool, optional
        Overwrite existing text in specified header line and following lines (default: `True`).
    newline: bool, optional
        Force text to be added to newline instead of appending to existing one (default: `False`).

    Returns
    -------
    str
        Updated textual header string.

    """
    TOTAL_LENGTH = 3200
    LINE_LENGTH = 80
    LINE_PREFIX_LEN = 3
    idx_header = None

    # set prefix to today's date if keyoword provided
    if isinstance(prefix, str) and prefix.upper() in ['_TODAY_', '_DATE_']:
        prefix = datetime.date.today().strftime('%Y-%m-%d')

    # set header line to DEFAULT_HEADER
    if header is True:
        txt, line_header = set_header_line(txt, header=header, line=header_line, overwrite=True)
        idx_header = line_header - 1
    # set custom header line
    elif isinstance(header, str):
        # check if header alrady exists
        idx_header = find_header_line(txt, header)
        if idx_header is None:
            # create new header line
            txt, line_header = set_header_line(
                txt, header=header, line=header_line, overwrite=overwrite
            )
            idx_header = line_header - 1
    else:
        raise ValueError(f'Parameter < {header} > is not permitted as input for `header`')

    # split textual header into list of lines & get indices of empty lines
    lines, idx_line = find_empty_line(txt)
    if idx_line is None:
        raise IndexError(
            'SEG-Y textual header is already full. Adding more information is not possible.'
        )

    # check for already existing lines with prefix
    idx_prefix = find_line_by_str(lines, search_str=prefix)

    # filter lines starting with prefix to only occur AFTER header line (if header set)
    if idx_header is not None:
        idx_prefix = [i for i in idx_prefix if i > idx_header]

    _inserted = False
    if prefix is not None and (newline is False):
        # get number of characters in string to paste
        len_info = len(info_str)

        for idx in idx_prefix:
            # print('-------------')
            # print(idx, _inserted)
            line = lines[idx]
            idx_last_char = len(line.rstrip())

            # enough space to add info to already existing line?
            if len_info < (LINE_LENGTH - idx_last_char):
                # print('Enough space? ', len_info < (LINE_LENGTH - idx_last_char), LINE_LENGTH - idx_last_char)
                lines[idx] = (
                    line[: idx_last_char + 2] + f'{info_str}' + line[idx_last_char + len_info + 2 :]
                )
                _inserted = True
                break

    # if (a) no line with prefix, (b) info not inserted, or (c) no prefix provided
    if any([(len(idx_prefix) == 0), (not _inserted), (prefix is None)]):
        # print('[INFO] No prefix provided or found or already full')
        to_add = f' {prefix}: {info_str}' if prefix is not None else f'{info_str}'

        # if header line exists BUT index of empty line is before header -> find empty line after header
        if idx_header is not None and (idx_line < idx_header):
            _, idx_line = find_empty_line(lines[idx_header:])
            idx_line += idx_header

        # construct and set line with info
        lines[idx_line] = (
            lines[idx_line][:LINE_PREFIX_LEN]
            + to_add
            + ' ' * (LINE_LENGTH - LINE_PREFIX_LEN - len(to_add))
        )  # lines[idx_line][LINE_PREFIX_LEN + len(to_add):]

    test = len(''.join(lines))
    assert (
        test == TOTAL_LENGTH
    ), f'Length of updated textual header ({test}) is not correct ({TOTAL_LENGTH} characters)'

    return '\n'.join(lines)


def find_header_line(lines, header):
    """Return index of header in list of lines."""
    if isinstance(lines, str):
        lines = lines.split('\n')
    check = [True if line.find(header) > -1 else False for line in lines]
    if any(check):
        return check.index(True)
    else:
        return None


def set_header_line(lines, header, line=25, overwrite=True):
    """
    Set custom header line to textual header line 'C25'.

    Parameters
    ----------
    lines : str
        String of textual header lines.
    header : str or bool
        Header string to set to line.
    line : int, optional
        Header line selcetion (default: `25`).
    overwrite : bool, optional
        Overwrite existing text in specified header line and following lines (default: `True`).

    Returns
    -------
    str
        Updated textual header.
    int
        Line number of header.

    """
    DEFAULT_HEADER = '***** PROCESSING WORKFLOW *****'
    if header is True:
        header = DEFAULT_HEADER

    if isinstance(lines, str):
        lines = lines.split('\n')
    elif isinstance(lines, list):
        pass
    else:
        raise ValueError(f'Not supported textual header type: {type(lines)}')

    # check for already existing header in lines
    idx_header = find_header_line(lines, header)
    if idx_header is not None:
        # warn('Specified header line already exists.', UserWarning)
        return '\n'.join(lines), idx_header + 1

    empty_header_line = lines[line - 1].count(' ') >= 77
    if not empty_header_line and not overwrite:
        raise Exception(
            f'Selected header line ({line}) is already in use and overwrite is set False.'
        )
    elif not empty_header_line and overwrite:
        warn('Selected header line is already in use and will be overwritten!', UserWarning)

    len_header = len(header)
    pad = 77 - len_header
    pad_left = pad // 2
    pad_right = pad_left if pad % 2 == 0 else pad_left + 1

    # set header to specified line
    lines[line - 1] = lines[line - 1][:3] + ' ' * pad_left + header + ' ' * pad_right
    for i in range(39 - line):  # exclude last line
        lines[line + i] = lines[line + i][:3] + ' ' * 77

    return '\n'.join(lines), line


def _isascii(txt):
    try:
        txt.decode("ascii")
    except UnicodeDecodeError:
        return False
    else:
        return True


def get_textual_header(path):
    """Read SEG-Y file and return textual header as string."""
    with open(path, 'rb') as file:
        file.seek(0, 0)  # find byte position zero relative to start of file (0)
        text_byte = file.read(3200)

    if _isascii(text_byte):
        text = text_byte.decode("ascii")
        text = wrap_text(text)
    else:
        text = text_byte.decode("cp500")  # EBCDIC encoding
        text = wrap_text(text)
    return text


def write_textual_header(path, txt: str, **kwargs_segy):
    """Write textual header string to SEG-Y file."""
    if isinstance(txt, str):
        txt = ''.join([t[:80] for t in txt.split('\n')])
    elif isinstance(txt, list):
        txt = ''.join([t[:80] for t in txt])  # silent truncating of each line if too long!
    else:
        raise ValueError(f'Not supported textual header type: {type(txt)}')

    header = bytes(txt, 'utf-8')
    assert len(header) == 3200, 'Binary string is too long, something went wrong...'

    kwargs_segy['ignore_geometry'] = True
    with segyio.open(path, 'r+', **kwargs_segy) as f:
        f.text[0] = header
