"""
Temporary testing script while I'm working on line correction code
"""
import numpy as np
import matplotlib.pyplot as plt

from spm_tools.example_data import pillar
from spm_tools.scanline import line_correction

test_image = pillar
X = test_image["X"]
Y = test_image["Y"]
Z = test_image["Z"]
extent = test_image["extent"]

line_noise = 1 * np.random.randn(Z.shape[0])
noisy = Z + line_noise[:, np.newaxis]

mean_flat, mean_correction = line_correction(noisy, method="mean")
median_flat, median_correction = line_correction(noisy, method="median")
median_diff_flat, median_diff_correction = line_correction(noisy, method="median_diff")

fig, axes = plt.subplots(ncols=5)
imshow_params = dict(extent=extent, origin="lower")
axes[0].imshow(Z, **imshow_params)
axes[1].imshow(noisy, **imshow_params)
axes[2].imshow(mean_flat, **imshow_params)
axes[3].imshow(median_flat, **imshow_params)
axes[4].imshow(median_diff_flat, **imshow_params)
fig.tight_layout()
plt.show()
