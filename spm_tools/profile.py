import numpy as np
from scipy.ndimage import map_coordinates

from ._helpers import uniform_XYZ


class ProfileBase:
    def __init__(self, n_points=None, n_averages=None, map_kw=None):
        self.n_points = n_points
        self.n_averages = n_averages
        map_defaults = {"order": 1, "mode": "constant", "cval": np.nan}
        self.map_kw = dict(map_defaults, **(map_kw or {}))

    def extract(
        self,
        *args,
        extent=None,
        origin=None,
        n_points=None,
        n_averages=None,
        map_kw=None
    ):
        X, Y, Z = uniform_XYZ(*args, extent=extent, origin=origin)
        # Output values
        pixel_coords = self._pixel_coords(
            X, Y, Z, n_points or self.n_points, n_averages or self.n_averages
        )
        resampled_area = map_coordinates(
            input=Z, coordinates=pixel_coords, **dict(self.map_kw, **(map_kw or {}))
        )
        z = resampled_area.mean(axis=0, where=~np.isnan(resampled_area))

        # Scalar distance from start of profile
        xy = np.linspace(0, self.length, z.size)

        return xy, z

    def _pixel_coords(self, X, Y, Z, n_points, n_averages, map_kw):
        raise NotImplementedError


class LinearProfile(ProfileBase):
    def __init__(self, start, end, width=0, **kwargs):
        self.start = np.asarray(start)
        self.end = np.asarray(end)
        self.width = width
        self.length = np.linalg.norm(self.end - self.start)

        # Lengthwise and perpendicular vectors
        self.v_length = self.end - self.start
        self.v_width = self.width * ([1, -1] * self.v_length / self.length)[::-1]
        self.corners = np.array(
            [
                self.start + self.v_width / 2,
                self.start - self.v_width / 2,
                self.end + self.v_width / 2,
                self.end - self.v_width / 2,
            ]
        )
        super().__init__(**kwargs)

    def _pixel_coords(self, X, Y, Z, n_points, n_averages):
        # Reduce to 1D (only first line of each axis)
        x = X[0]
        y = Y[:, 0]

        # Pixel spacings (falling back to median in case of nonuniform spacing)
        d_x = np.median(np.diff(x))
        d_y = np.median(np.diff(y))

        # Base n_points and n_averages on pixel size if not given
        if n_points is None:
            n_points = int(np.ceil(np.linalg.norm(self.v_length / (d_x, d_y))))
        if n_averages is None:
            if self.width == 0:
                n_averages = 1
            else:
                n_averages = int(np.ceil(np.linalg.norm(self.v_width / (d_x, d_y))))

        # Grid of coordinates to sample at (like several parallel profiles)
        real_coords = np.linspace(
            np.linspace(*self.corners[:2], n_averages),
            np.linspace(*self.corners[2:], n_averages),
            n_points,
        )

        # Use pixel spacings to map between real and pixel coordinates
        pixel_coords = (real_coords - (x[0], y[0])) / (d_x, d_y)

        # Change to expected shape for map_coordinates
        pixel_coords = pixel_coords.T[::-1]
        return pixel_coords
