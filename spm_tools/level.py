"""Tools for levelling SPM images."""
import numpy as np

from .utilities import poly_surface, fit_poly_surface, geometric_median
from ._uniform_coords import uniform_XYZ_decorator


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
