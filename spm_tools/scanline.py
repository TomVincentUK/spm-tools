"""Tools for correcting scan line offsets."""
import numpy as np
from scipy.optimize import minimize

from ._uniform_coords import uniform_XYZ_decorator


def _mean_correction(Z, axis):
    return np.mean(Z.T, axis=axis)


def _median_correction(Z, axis):
    return np.median(Z.T, axis=axis)


def _median_diff_correction(Z, axis):
    """
    I'm pretty sure this is incorrect, and gives the same value as median.
    """
    if axis == 1:
        Z = Z.T

    n_lines = Z.shape[0]

    relative = np.zeros(n_lines - 1)
    for i in range(n_lines - 1):
        res = minimize(lambda dZ: np.median(Z[i + 1] - Z[i] - dZ) ** 2, 0)
        relative[i] = res.x
    correction = np.ones(n_lines) * np.median(Z[0])
    correction[1:] += np.cumsum(relative)
    return correction


_method_dict = {
    "mean": _mean_correction,
    "median": _median_correction,
    "median_diff": _median_diff_correction,
}


@uniform_XYZ_decorator
def line_correction(X, Y, Z, axis=0, method="median", **kwargs):
    correction = _method_dict[method](Z, axis, **kwargs)
    if axis == 0:
        correction = correction[..., np.newaxis]
    return Z - correction, correction
