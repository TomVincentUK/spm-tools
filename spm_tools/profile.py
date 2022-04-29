"""Tools for extracting line profiles from SPM images."""
import numpy as np
from scipy.ndimage import map_coordinates

from .helpers import uniform_XYZ_decorator

MAP_DEFAULTS = {"order": 1, "mode": "constant", "cval": np.nan}


@uniform_XYZ_decorator
def linear_profile(
    X, Y, Z, start, end, width=0, n_points=None, n_averages=None, map_kw=None
):
    start = np.asarray(start)
    end = np.asarray(end)
    v_length = end - start
    length = np.linalg.norm(v_length)
    v_width = width * ([1, -1] * v_length / length)[::-1]
    corners = np.array(
        [
            start + v_width / 2,
            start - v_width / 2,
            end + v_width / 2,
            end - v_width / 2,
        ]
    )

    # Use first line of each axis to calibrate pixels to real coordinates
    x = X[0]
    y = Y[:, 0]

    # Pixel spacings (falling back to median in case of nonuniform spacing)
    d_x = np.median(np.diff(x))
    d_y = np.median(np.diff(y))

    # Base n_points and n_averages on pixel size if not given
    if n_points is None:
        n_points = int(np.ceil(np.linalg.norm(v_length / (d_x, d_y))))
    if n_averages is None:
        if width == 0:
            n_averages = 1
        else:
            n_averages = int(np.ceil(np.linalg.norm(v_width / (d_x, d_y))))

    # Grid of coordinates to sample at (like several parallel profiles)
    real_coords = np.linspace(
        np.linspace(*corners[:2], n_averages),
        np.linspace(*corners[2:], n_averages),
        n_points,
    )

    # Use pixel spacings to map between real and pixel coordinates
    pixel_coords = (real_coords - (x[0], y[0])) / (d_x, d_y)

    # Change to expected shape for map_coordinates
    pixel_coords = pixel_coords.T[::-1]
    resampled_area = map_coordinates(
        input=Z, coordinates=pixel_coords, **dict(MAP_DEFAULTS, **(map_kw or {}))
    )
    z = resampled_area.mean(axis=0, where=~np.isnan(resampled_area))

    # Scalar distance from start of profile
    xy = np.linspace(0, length, z.size)

    return xy, z, corners
