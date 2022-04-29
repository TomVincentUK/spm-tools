"""Tools for levelling SPM images."""
import warnings

import numpy as np

from .helpers import uniform_XYZ_decorator


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


def geometric_median(points, dr=None, max_iter=1000):
    """Calculate a geometric median using Weiszfeld's algorithm.

    Based on the modified algorithm described by Vardi and Zhang in:
        https://doi.org/10.1073/pnas.97.4.1423

    Parameters
    ----------
    points : array_like
        The array of points from which to calculate the geometric median.
        The size of the first axis is used as the number of dimensions for the
        space, `ndim`.
    dr : float
        The convergence threshold for the algorithm. The function will return
        when the difference between successive estimates are closer than `dr`.
        If not specified, the default value is 1e-6 times the minimum
        dimension-wise standard deviation.
    max_iter : float
        The maximum number of iterations.

    Returns
    -------
    geo_med : numpy.ndarray
        An `ndim`-length array containing estimated geometric median vector.
    """
    points = np.asarray(points)
    ndim = points.shape[0]
    points = points.reshape(ndim, -1).T

    # If all vectors are the same, just return that vector
    if (points - points[0] == 0).all():
        geo_med = points[0]
        return geo_med

    if dr is None:
        dr = 1e-6 * points.std(axis=0).min()

    geo_med_est = np.median(points, axis=0)
    dr_i = 10 * dr
    i = 0
    while (dr_i > dr) and (i < max_iter):
        dists = np.linalg.norm(points - geo_med_est, axis=-1)[..., np.newaxis]
        zero_vals = (dists == 0).flatten()

        weights = 1 / dists[~zero_vals]
        weight_sum = weights.sum()
        geo_med = (weights * points[~zero_vals, :]).sum(axis=0) / weight_sum
        dr_i = np.linalg.norm(geo_med - geo_med_est)

        # Account for cases where the estimate is one of the points
        n_zeros = zero_vals.sum()
        if (dr_i != 0) and (n_zeros != 0):
            mix_factor = n_zeros / (dr_i * weight_sum)
            geo_med = (
                max(0, 1 - mix_factor) * geo_med + min(1, mix_factor) * geo_med_est
            )
            dr_i = np.linalg.norm(geo_med - geo_med_est)

        i += 1
        geo_med_est[:] = geo_med

    if i >= max_iter:
        warnings.warn("`geometric_median` did not converge.", category=UserWarning)

    return geo_med


@uniform_XYZ_decorator
def poly_level(X, Y, Z, **kwargs):
    coeffs = fit_poly_surface(X, Y, Z, **kwargs)
    surface = poly_surface(X, Y, coeffs)
    leveled = Z - surface
    return leveled, surface, coeffs


@uniform_XYZ_decorator
def facet_level(X, Y, Z, **kwargs):
    dZ_by_dY_full, dZ_by_dX_full = np.gradient(Z, Y[:, 0], X[0])
    dZ_by_dX, dZ_by_dY = geometric_median([dZ_by_dX_full, dZ_by_dY_full])
    coeffs = np.array([[0, dZ_by_dY], [dZ_by_dX, 0]])
    surface = poly_surface(X, Y, coeffs)
    leveled = Z - surface
    return leveled, surface, coeffs
