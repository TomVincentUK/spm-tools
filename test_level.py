"""
Temporary testing script while I'm working on image leveling code
"""
import numpy as np
import matplotlib.pyplot as plt

from spm_tools.example_data import anisotropic
from spm_tools.level import PolyLevel, FacetLevel

test_image = anisotropic
X = test_image["X"]
Y = test_image["Y"]
Z = test_image["Z"]
extent = test_image["extent"]

Z = Z + 0.1 * np.random.randn(*Z.shape)

p_level_0 = PolyLevel().fit(Z, extent=extent, origin="lower")
p_plane_0 = p_level_0.surface
p_subtracted_0 = p_level_0.subtract(Z)

fig, axes = plt.subplots(nrows=2, ncols=5)
for ax, data in zip(axes[0], (Z, p_plane_0, p_subtracted_0)):
    im = ax.imshow(data)
    fig.colorbar(im, ax=ax, orientation="horizontal")

f_level_0 = FacetLevel().fit(Z, extent=extent, origin="lower")
f_plane_0 = f_level_0.surface
f_subtracted_0 = f_level_0.subtract(Z)

for ax, data in zip(axes[1], (Z, f_plane_0, f_subtracted_0)):
    im = ax.imshow(data)
    fig.colorbar(im, ax=ax, orientation="horizontal")

f_level_1 = FacetLevel().fit(p_subtracted_0, extent=extent, origin="lower")
f_plane_1 = f_level_1.surface
f_subtracted_1 = f_level_1.subtract(p_subtracted_0)

for ax, data in zip(axes[0, 3:], (f_plane_1, f_subtracted_1)):
    im = ax.imshow(data)
    fig.colorbar(im, ax=ax, orientation="horizontal")

p_level_1 = PolyLevel().fit(f_subtracted_0, extent=extent, origin="lower")
p_plane_1 = p_level_1.surface
p_subtracted_1 = p_level_1.subtract(f_subtracted_0)

for ax, data in zip(axes[1, 3:], (p_plane_1, p_subtracted_1)):
    im = ax.imshow(data)
    fig.colorbar(im, ax=ax, orientation="horizontal")


fig.tight_layout()
plt.show()
