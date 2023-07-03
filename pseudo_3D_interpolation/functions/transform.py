"""Custom Affine transformation class."""

import numpy as np


class Affine:
    """Affine transformation."""

    def __init__(self, scaling=1, translation=0, rotation=0, shear=0, matrix=None):
        """
        Initialize Affine transform.
        Using either user-specified parameters or `numpy.ndarray` of shape (3, 3)
        holding affine transform (`matrix` parameter).

        References
        ----------
        [^1]: [https://stackoverflow.com/a/53691628](https://stackoverflow.com/a/53691628)

        """
        if np.isscalar(scaling):
            sx = sy = scaling
        else:
            sx, sy = scaling

        if np.isscalar(translation):
            tx = ty = translation
        else:
            tx, ty = translation

        rotation = np.deg2rad(rotation)

        if np.isscalar(shear):
            cx = cy = shear
        else:
            cx, cy = shear
        cx, cy = np.deg2rad(cx), np.deg2rad(cy)

        if matrix is None:
            self.matrix = np.array(
                [
                    [sx * np.cos(rotation), -np.sin(rotation) + cx, tx],
                    [np.sin(rotation) + cy, sy * np.cos(rotation), ty],
                    [0, 0, 1],
                ]
            )
        else:
            if isinstance(matrix, np.ndarray) and matrix.shape == (3, 3):
                self.matrix = matrix
            else:
                raise AttributeError('Matrix must be numpy array of shape (3,3)!')

    def __repr__(self):  # noqa
        return f'Affine({self.matrix})'
        
    @staticmethod
    def get_scalar(param):
        """Return (unpacked) scalars from input tuple or duplicated input scalar."""
        if np.isscalar(param):
            px = py = param
        else:
            px, py = param
        return px, py

    @staticmethod
    def fix_floating_point_error(value):
        """Fix floating point error."""

        def _func(x):
            return float(f'{x:g}')

        func = np.vectorize(_func)
        return func(value)

    def _transform(self, points):
        """Return input points (2D array) and corresponding transformation matrix."""
        p = np.atleast_2d(points)
        nrows, ncols = p.shape
        if ncols == 2:
            p = np.hstack((p, np.ones((nrows, 1), dtype=p.dtype)))
        return p, self.matrix

    @classmethod
    def identity(cls):
        """Return identity matrix (transform to original image)."""
        return cls()

    def scaling(self, scale, overwrite=False):
        """Return `self` with updated matrix using given scaling."""
        ax = (0, 1)
        sx, sy = self.get_scalar(scale)
        if overwrite:
            self.matrix[ax, ax] = sx, sy
        else:
            self.matrix = Affine(scaling=(sx, sy)).matrix @ self.matrix
        return self

    def translation(self, offset, overwrite=False):
        """Return `self` with updated matrix using given translation."""
        tx, ty = self.get_scalar(offset)
        if overwrite:
            self.matrix[2, :2] = tx, ty
        else:
            # self.matrix = self.matrix @ Affine(translation=(tx, ty)).matrix
            self.matrix = Affine(translation=(tx, ty)).matrix @ self.matrix
        return self

    def rotation(self, angle: float, overwrite=False):
        """Return `self` with updated matrix using given rotation."""
        angle = np.deg2rad(angle)
        _matrix = np.array(
            [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        )
        if overwrite:
            self.matrix[:2, :2] = _matrix[:2, :2]
        else:
            # correct order when using chained `.` operators
            self.matrix = _matrix @ self.matrix
        return self

    def rotate_around(self, angle: float, origin: tuple = (0, 0)):
        """Return `self` rotated around provided point (default: `(0, 0)`)."""
        self.translation(tuple(-np.asarray(origin)))
        if angle is not None:
            self.rotation(angle)
        self.translation(origin)
        return self

    def rotate(self, points, angle: float = None, origin: tuple = (0, 0), fix_float: bool = False):
        """
        Return rotated input points (N, 2) or (N, 3).
        Positive angles indicate a counter-clockwise roation,
        negative angles a clockwiseroation.

        Parameters
        ----------
        points : np.ndarray
            Input point coordinates with shape (N, 2) for **2D** or (N, 3) for **3D**.
        angle : float, optional
            Rotation angle (will overwrite previously set rotation) (default: `None`).
        origin : tuple, optional
            Rotation around given point (default: `(0, 0)`).
        fix_float : bool, optional
            Fix floating point precision error (default: `False`).

        Returns
        -------
        np.ndarray
            Rotated input points.

        """
        points = np.asarray(points)
        npts, ndim = points.shape

        o = np.atleast_2d(origin)
        if o.shape[1] == 2:
            o = np.hstack((o, np.array([[1]])))

        self.translation(tuple(-np.asarray(origin)))
        if angle is not None:
            self.rotation(angle)
        self.translation(origin)

        p, A = self._transform(points)
        t = (p @ A.T)[:, :ndim]  # ((p-o) @ A.T + o)[:,:ndim]

        if fix_float:
            return self.fix_floating_point_error(t)
        return t

    def skew(self, shear, overwrite=False):
        """Return `self` with updated matrix using given shear angle (deg)."""
        cx, cy = self.get_scalar(shear)
        if overwrite:
            self.matrix[(0, 1), (1, 0)] = np.tan(cx), np.tan(cy)
        else:
            self.matrix = Affine(shear=(cx, cy)).matrix @ self.matrix
        return self

    def transform(self, points, fix_float: bool = False):
        """
        Return transformed input points (based on parameters set on initiation).

        Parameters
        ----------
        points : np.ndarray
            2D array of coordinates with shape `(npts, 2)`.
        fix_float : bool, optional
            Fix floating point precision error (default: `False`).

        Returns
        -------
        np.ndarray
            Transformed input points.

        """
        points = np.atleast_2d(np.asarray(points))
        npts, ndim = points.shape
        p, A = self._transform(points)
        t = (p @ A.T)[:, :ndim]

        if fix_float:
            return self.fix_floating_point_error(t)
        return t

    def __matmul__(self, other):
        """Matrix multiplication using @ operator (**order-dependent**!)."""
        if isinstance(other, Affine):
            _matrix = self.matrix @ other.matrix
        elif isinstance(other, np.ndarray):
            _matrix = self.matrix @ other
        else:
            raise NotImplementedError(
                'Other must be either Affine() or numpy.ndarry of shape (3,3)'
            )
        return Affine(matrix=_matrix)

    def __mul__(self, other):
        """Matrix multiplication (**order-dependent**!)."""
        if isinstance(other, Affine):
            _matrix = self.matrix @ other.matrix
        elif isinstance(other, np.ndarray):
            _matrix = self.matrix @ other
        else:
            raise NotImplementedError(
                'Other must be either Affine() or numpy.ndarry of shape (3,3)'
            )
        return Affine(matrix=_matrix)

    def __add__(self, other):  # noqa
        """
        Combine Affine transformations so that `C = A + B` equals
        `C.transform(x) = B.transform(A.transform(x))`.

        """
        if isinstance(other, Affine):
            _matrix = self.matrix @ other.matrix
        elif isinstance(other, np.ndarray):
            _matrix = self.matrix @ other
        else:
            raise NotImplementedError(
                'Other must be either Affine() or numpy.ndarry of shape (3,3)'
            )
        return Affine(matrix=_matrix)

    def inverse(self, inplace: bool = False):
        """
        Apply inverse transform.

        Parameters
        ----------
        inplace : bool, optional
            Assigns to `self.matrix` if True (default: `False`).

        Returns
        -------
        Affine
            Inverse Affine matrix.

        """
        m = self.matrix

        _inv = np.linalg.inv(m[:2, :2])  # only upper left (2,2) sub-matrix
        _t = np.array(
            [
                [-m[0, 2] * _inv[0, 0] - m[1, 2] * _inv[0, 1]],
                [-m[0, 2] * _inv[1, 0] - m[1, 2] * _inv[1, 1]],
            ]
        )
        # _t = np.atleast_2d(1 / (np.flip(m[:2,2]) * -1)).T
        _matrix = np.vstack((np.hstack((_inv, _t)), np.array([0, 0, 1])))
        if inplace:
            self.matrix = _matrix
            return self
        else:
            return Affine(matrix=_matrix)

    def copy(self):
        """Return copy of `self`."""
        return Affine(matrix=self.matrix)
