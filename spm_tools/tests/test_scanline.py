import numpy as np

import spm_tools as spm

test_image = spm.example_data.synthetic.pillar
X = test_image["X"].copy()[:400]
Y = test_image["Y"].copy()[:400]
Z = test_image["Z"].copy()[:400]

rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(0)))
Z += rs.randn(*Z.shape)


def test_line_correction_mean():
    corrected_0, correction_0 = spm.scanline.line_correction(
        X, Y, Z, axis=0, method="mean"
    )
    np.testing.assert_almost_equal(
        corrected_0.mean(axis=-1), np.zeros(corrected_0.shape[0])
    )
    np.testing.assert_almost_equal(corrected_0 + correction_0, Z)

    corrected_1, correction_1 = spm.scanline.line_correction(
        X, Y, Z, axis=1, method="mean"
    )
    np.testing.assert_almost_equal(
        corrected_1.mean(axis=0), np.zeros(corrected_1.shape[1])
    )
    np.testing.assert_almost_equal(corrected_1 + correction_1[np.newaxis], Z)


def test_line_correction_median():
    corrected_0, correction_0 = spm.scanline.line_correction(
        X, Y, Z, axis=0, method="median"
    )
    np.testing.assert_almost_equal(
        np.median(corrected_0, axis=-1), np.zeros(corrected_0.shape[0])
    )
    np.testing.assert_almost_equal(corrected_0 + correction_0, Z)

    corrected_1, correction_1 = spm.scanline.line_correction(
        X, Y, Z, axis=1, method="median"
    )
    np.testing.assert_almost_equal(
        np.median(corrected_1, axis=0), np.zeros(corrected_1.shape[1])
    )
    np.testing.assert_almost_equal(corrected_1 + correction_1[np.newaxis], Z)


def test_line_correction_median_diff():
    corrected_0, correction_0 = spm.scanline.line_correction(
        X, Y, Z, axis=0, method="median_diff"
    )
    np.testing.assert_almost_equal(
        np.median(np.diff(corrected_0, axis=0), axis=-1),
        np.zeros(corrected_0.shape[0] - 1),
    )
    np.testing.assert_almost_equal(corrected_0 + correction_0, Z)
