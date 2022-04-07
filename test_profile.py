import numpy as np
import matplotlib.pyplot as plt
from spm_tools.example_data import anisotropic
from spm_tools.profile import LinearProfile

test_image = anisotropic
X = test_image["X"]
Y = test_image["Y"]
Z = test_image["Z"]
extent = test_image["extent"]

profile = LinearProfile(start=(255, 255), end=(255, 0), width=10)
fig, axes = plt.subplots(ncols=2, nrows=2)
for row, origin in zip(axes, ("upper", "lower")):
    row[0].imshow(Z, origin=origin)
    row[0].plot(*np.array([profile.start, profile.end]).T, c="w")
    row[0].scatter(*profile.start, c="r")
    row[0].scatter(*profile.corners.T, c="w", s=1)
    row[1].plot(*profile.extract(Z, origin=origin))
fig.tight_layout()
plt.show()

profile = LinearProfile(start=(1/8, 1/8), end=(1/8, 1), width=0.2)
fig, axes = plt.subplots(ncols=2, nrows=2)
for row, origin in zip(axes, ("upper", "lower")):
    row[0].imshow(Z, extent=extent, origin=origin)
    row[0].plot(*np.array([profile.start, profile.end]).T, c="w")
    row[0].scatter(*profile.start, c="r")
    row[0].scatter(*profile.corners.T, c="w", s=1)
    row[1].plot(*profile.extract(Z, extent=extent, origin=origin))
fig.tight_layout()
plt.show()

profile = LinearProfile(start=(1/8, 1/8), end=(1/8, 1), width=0.2)
fig, axes = plt.subplots(ncols=2, nrows=1)
axes[0].pcolormesh(X, Y, Z)
axes[0].plot(*np.array([profile.start, profile.end]).T, c="w")
axes[0].scatter(*profile.start, c="r")
axes[0].scatter(*profile.corners.T, c="w", s=1)
axes[1].plot(*profile.extract(X, Y, Z))
fig.tight_layout()
plt.show()
