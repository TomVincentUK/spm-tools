"""Tools for levelling SPM images."""
import numpy as np

from ._helpers import _uniform_XYZ, _uniform_XY


def poly_surface(X, Y, coeffs):
    """Return points on a polynomial surface plane.

    Parameters
    ----------
    X : array_like
        The x coordinates at which to evaluate the surface. Must be the same
        shape as `Y`.
    Y : array_like
        The y coordinates at which to evaluate the surface. Must be the same
        shape as `X`.
    coeffs : array_like
        The polynomial coefficient matrix. If `coeffs` is 1D, this corresponds
        to the coefficients of `X`. If `coeffs` is 2D, this corresponds to the
        coefficients of `X` and `Y`, such that `coeffs[i, j]`` scales the term
        `X**i * Y**j`.

    Returns
    -------
    Z : numpy.ndarray
        The polynomial surface evaluated at `X`-`Y`. Returned array is the same
        shape as `X` and `Y`.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    coeffs = np.asarray(coeffs)

    if X.shape != Y.shape:
        raise TypeError("`X` and `Y` must have the same shape")
    if coeffs.ndim < 2:
        coeffs = coeffs.reshape(coeffs.shape + (2 - coeffs.ndim) * (1,))
    elif coeffs.ndim > 2:
        raise TypeError("`coeffs` must be 1D or 2D")

    # Generate a matrix of X**i * Y**j
    i, j = np.indices(coeffs.shape)
    X_powers = X[..., np.newaxis] ** i.flatten()[np.newaxis]
    Y_powers = Y[..., np.newaxis] ** j.flatten()[np.newaxis]
    XY_powers = X_powers * Y_powers

    # Actual polynomial performed as a matrix multiplication
    Z = XY_powers @ coeffs.flatten()
    return Z


def fit_poly_surface(X, Y, Z, order=1, XY_order=None, allowed_coeffs=None, rcond=None):
    """Fit a polynomial surface in 2D.

    Parameters
    ----------
    X : array_like
        The x coordinates at which to evaluate the surface. Must be the same
        shape as `Y` and `Z`.
    Y : array_like
        The y coordinates at which to evaluate the surface. Must be the same
        shape as `X` and `Z`.
    Z : array_like
        The polynomial surface corresponding to inputs `X` and `Y`. Must be the
        same shape as `X` and `Y`.
    order : int, optional
        The maximum combined x and y degree of the 2D polynomial to include in
        the fit. For coefficients `i` and `j` scaling terms `X**i * Y**j`, only
        terms where `i + j < order` will be included. The default value is 1,
        which fits a flat plane. This argument is ignored if either 'XY_order'
        or `allowed_coeffs` are not `None`.
    XY_order : 2-tuple of int, optional
        Axis-wise maximum degrees to include in the fit, such that
        `XY_order[0]` is the maximum x degree and `XY_order[1]` is the maximum
        y degree.
        This argument supersedes `order`, but is ignored if `allowed_coeffs` is
        not `None`.
    allowed_coeffs : array_like of bool, optional
        A 2D boolean array whose terms `allowed_coeffs[i, j]` explicitly
        specify which coefficients, scaling terms `X**i * Y**j`, to include in
        the fit.
        This argument supersedes both `order` and `XY_order`.
    rcond : float, optional
        Cut-off ratio used by `numpy.linalg.lstsq()`.

    Returns
    -------
    coeffs : numpy.ndarray
        The polynomial coefficient matrix found by the fit. This corresponds to
        the coefficients of `X` and `Y`, such that `coeffs[i, j]`` scales the
        term `X**i * Y**j`.
    """
    if not (X.shape == Y.shape == Z.shape):
        raise TypeError("`X`, `Y` and `Z` must have the same shape")

    # Parse inputs to get a boolean matrix of allowed_coeffs
    if allowed_coeffs is not None:
        allowed_coeffs = np.asarray(allowed_coeffs)
        if not np.array_equal(allowed_coeffs, allowed_coeffs.astype(bool)):
            raise TypeError("`allowed_coeffs` must be interpretable as a boolean array")
        if allowed_coeffs.ndim != 2:
            raise TypeError("`allowed_coeffs` must be 2D")
    elif XY_order is not None:
        if len(XY_order) != 2:
            raise TypeError("`XY_order` must have length 2")
        allowed_coeffs = np.ones((XY_order[0] + 1, XY_order[1] + 1)).astype(bool)
    else:
        ij = np.arange(order + 1)
        allowed_coeffs = ij + ij[..., np.newaxis] <= order

    # Generate a matrix of X**i * Y**j for allowed i and j
    i, j = np.indices(allowed_coeffs.shape)
    i = i[allowed_coeffs]
    j = j[allowed_coeffs]
    X_powers = X.flatten()[..., np.newaxis] ** i[np.newaxis]
    Y_powers = Y.flatten()[..., np.newaxis] ** j[np.newaxis]
    XY_powers = X_powers * Y_powers

    # Inverse of the vector operation performed in `poly_surface()`
    coeff_vector, _, _, _ = np.linalg.lstsq(XY_powers, Z.flatten(), rcond)

    # Pad out the 1D vector to a matrix including the disallowed i and j values
    coeffs = np.zeros_like(allowed_coeffs).astype(float)
    coeffs[allowed_coeffs] = coeff_vector

    return coeffs


class LevelBase:
    """Base class for levelling objects."""

    def __init__(self):
        self.surface = None

    def fit(self, *args):
        raise NotImplementedError

    def evaluate(self, X, Y):
        raise NotImplementedError

    def subtract(self, *args, extent=None, origin=None):
        if self.surface is None:
            raise TypeError(
                "".join(
                    "Background not yet evaluated. Try running ",
                    f"`{type(self).__name__}.fit()`",
                    " first.",
                )
            )
        X, Y, Z = _uniform_XYZ(*args, extent=extent, origin=origin)
        return Z - self.surface

    def func_subtract(self, *args):
        X, Y, Z = _uniform_XYZ(*args)
        return Z - self.evaluate(X, Y)


class PolyLevel(LevelBase):
    def __init__(self, coeffs=None):
        if coeffs is None:
            self.coeffs = np.array([[0]])
        else:
            coeffs = np.asarray(coeffs)
            if coeffs.ndim < 2:
                coeffs = coeffs.reshape(coeffs.shape + (2 - coeffs.ndim) * (1,))
            elif coeffs.ndim > 2:
                raise TypeError("`coeffs` must be 1D or 2D")
            self.coeffs = coeffs
        super().__init__()

    def fit(
        self,
        *args,
        extent=None,
        origin=None,
        order=1,
        XY_order=None,
        allowed_coeffs=None,
        rcond=None,
    ):
        X, Y, Z = _uniform_XYZ(*args, extent=extent, origin=origin)
        self.coeffs = fit_poly_surface(
            X,
            Y,
            Z,
            order=order,
            XY_order=XY_order,
            allowed_coeffs=allowed_coeffs,
            rcond=rcond,
        )
        self.surface = self.evaluate(X, Y)
        return self

    def evaluate(self, X, Y):
        X, Y = _uniform_XY(X, Y)
        return poly_surface(X, Y, self.coeffs)
