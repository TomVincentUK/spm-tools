import numpy as np
from pytest import warns

import spm_tools as spm


xy = np.linspace(-1, 1, 128)
X, Y = np.meshgrid(xy, xy)
coeffs = np.arange(9).reshape(3, 3)
surface = spm.utilities.poly_surface(X, Y, coeffs)


def test_fit_poly_surface_reproducible():
    fit_coeffs = spm.utilities.fit_poly_surface(X, Y, surface, XY_order=(2, 2))
    np.testing.assert_almost_equal(coeffs, fit_coeffs)


def test_fit_poly_surface_overfit():
    fit_coeffs = spm.utilities.fit_poly_surface(X, Y, surface, XY_order=(4, 4))
    overfit_coeffs = fit_coeffs[np.indices(fit_coeffs.shape).max(axis=0) >= 3]
    np.testing.assert_almost_equal(overfit_coeffs, np.zeros_like(overfit_coeffs))


def test_fit_poly_surface_max_combined():
    order = coeffs.shape[0]
    fit_coeffs = spm.utilities.fit_poly_surface(X, Y, surface, order=order)
    ij = np.arange(order + 1)
    forbidden_coeffs = fit_coeffs[ij + ij[..., np.newaxis] > order]
    np.testing.assert_almost_equal(forbidden_coeffs, np.zeros_like(forbidden_coeffs))


def test_geometric_median_circle():
    theta = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    circle = np.array([np.cos(theta), np.sin(theta)])
    np.testing.assert_almost_equal(spm.utilities.geometric_median(circle), np.zeros(2))


def test_geometric_median_result_in_points():
    points = np.array([[0, 0, 1], [0, 0, 1]])
    np.testing.assert_equal(spm.utilities.geometric_median(points), np.zeros(2))


def test_geometric_median_same_vector():
    v = np.arange(9)
    v_stack = v[:, np.newaxis] * np.ones(10)
    np.testing.assert_equal(spm.utilities.geometric_median(v_stack), v)


def test_geometric_median_transform_invariant():
    points = np.array([[1, 2, 3], [1, 2, 1]])

    def linear_transform(points):
        return points * 10 + 10

    np.testing.assert_almost_equal(
        linear_transform(spm.utilities.geometric_median(points)),
        spm.utilities.geometric_median(linear_transform(points)),
    )


def test_geometric_median_zero_covariance():
    rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(0)))
    x = rs.poisson(10, 128)
    y = x * 10 + 10
    z = x / 10 - 10
    points = np.array([x, y, z])

    np.testing.assert_almost_equal(
        spm.utilities.geometric_median(points), np.median(points, axis=-1)
    )


def test_geometric_median_convergance_warning():
    theta = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    circle = np.array([np.cos(theta), np.sin(theta)])
    with warns(UserWarning, match="`geometric_median` did not converge."):
        spm.utilities.geometric_median(circle, dr=1e-16, max_iter=2)
