# import numpy as np
import matplotlib.pyplot as plt
from spm_tools.example_data import anisotropic
from spm_tools.level import PolyLevel

test_image = anisotropic
X = test_image["X"]
Y = test_image["Y"]
Z = test_image["Z"]
extent = test_image["extent"]

poly_level = PolyLevel().fit(Z, extent=extent, origin='lower')
plane = poly_level.surface
subtracted = poly_level.subtract(Z)

fig, axes = plt.subplots(ncols=3)
for ax, data in zip(axes, (Z, plane, subtracted)):
    im = ax.imshow(data)
    fig.colorbar(im, ax=ax)
fig.tight_layout()
plt.show()
