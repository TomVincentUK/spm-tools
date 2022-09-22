import numpy as np

import spm_tools as spm


def test_geometric_median_circle():
    theta = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    circle = np.array([np.cos(theta), np.sin(theta)])
    np.testing.assert_almost_equal(spm.utilities.geometric_median(circle), np.zeros(2))


def test_geometric_median_result_in_points():
    points = np.array([[0, 0, 1], [0, 0, 1]])
    np.testing.assert_equal(spm.utilities.geometric_median(points), np.zeros(2))


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
