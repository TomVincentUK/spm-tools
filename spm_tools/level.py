import numpy as np

from ._helpers import _uniform_XYZ, _uniform_XY


def poly_surface(X, Y, coeffs):
    X = np.asarray(X)
    Y = np.asarray(Y)
    coeffs = np.asarray(coeffs)

    if coeffs.ndim < 2:
        coeffs = coeffs.reshape(coeffs.shape + (2 - coeffs.ndim) * (1,))
    elif coeffs.ndim > 2:
        raise TypeError("coeffs must be 1D or 2D")

    # Generate a matrix of X**i * Y**j
    i, j = np.indices(coeffs.shape)
    X_powers = X[..., np.newaxis] ** i.flatten()[np.newaxis]
    Y_powers = Y[..., np.newaxis] ** j.flatten()[np.newaxis]
    XY_powers = X_powers * Y_powers

    # Actual polynomial performed as a matrix multiplication
    return XY_powers @ coeffs.flatten()


def fit_poly_surface(X, Y, Z, order=1, XY_order=None, allowed_coeffs=None, rcond=None):
    # Parse inputs to get a boolean matrix of allowed_coeffs
    if allowed_coeffs is not None:
        allowed_coeffs = np.asarray(allowed_coeffs)
        if not np.array_equal(allowed_coeffs, allowed_coeffs.astype(bool)):
            raise TypeError("allowed_coeffs must be interpretable as a boolean array")
        if allowed_coeffs.ndim != 2:
            raise TypeError("allowed_coeffs must be 2D")
    elif XY_order is not None:
        if len(XY_order) != 2:
            raise TypeError("XY_order must have length 2")
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
    def __init__(self):
        self.surface = None

    def fit(self, *args):
        raise NotImplementedError

    def evaluate(self, X, Y):
        raise NotImplementedError

    def subtract(self, *args, extent=None, origin=None):
        if self.surface is None:
            raise TypeError(
                f"Background not yet evaluated. Try running `{type(self).__name__}.fit()` first."
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
                raise TypeError("coeffs must be 1D or 2D")
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
