import numpy as np


def uniform_XY(X, Y):
    """Return uniform x and y arguments for 1D and 2D inputs.

    Parameters
    ----------
    X, Y : array_like,
        If `X` and `Y` are 1D, then function returns `numpy.meshgrid(X, Y)`.
        If `X` and `Y` are 2D, then function checks they are the same shape and
        returns them as is.

    Returns
    -------
    X, Y : (M, N) numpy.ndarray
        Two 2D arrays of the same shape containing x and y coordinates.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    if X.ndim == Y.ndim == 1:
        X, Y = np.meshgrid(X, Y)
    elif X.ndim == Y.ndim == 2:
        if X.shape != Y.shape:
            raise TypeError("2D `X` and `Y` must have the same shape")
    else:
        raise TypeError("`X` and `Y` must be 1D or 2D")

    return X, Y


def uniform_XYZ(*args, extent=None, origin=None):
    """Return uniform x, y and z arguments for several different input forms.

    Call signature::

        uniform_XYZ([X, Y,] Z, extent=None, origin=None)

    This function allows image size and shape to be specified in several
    different ways, based on the image plotting functions in `matplotlib`:
    `matplotlib.pyplot.imshow()`, `matplotlib.pyplot.pcolormesh()`,
    `matplotlib.pyplot.contour()`, and `matplotlib.pyplot.countourf()`.

    This function is based on the behaviour of `matplotlib.pyplot.contour()`.

    Parameters
    ----------
    X, Y : array_like, optional
        The coordinates of the values in `Z`.

        `X` and `Y` must both be 2D with the same shape as `Z` (e.g. created
        via `numpy.meshgrid`), or they must both be 1D such that `len(X) == N`
        is the number of columns in `Z` and `len(Y) == M` is the number of rows
        in `Z`.

        `X` and `Y` must both be ordered monotonically.

        If not given, they are assumed to be integer indices, i.e.
        X = range(N)`, `Y = range(M)`.
    Z : (M, N) array_like
        The image corresponding to `X` and `Y`.
    extent : (x0, x1, y0, y1), optional
        If `origin` is not `None`, then `extent` is interpreted as in
        `matplotlib.pyplot.imshow`: it gives the outer pixel boundaries. In
        this case, the position of Z[0, 0] is the center of the pixel, not a
        corner. If `origin` is `None`, then (`x0`, `y0`) is the position of
        `Z[0, 0]`, and (`x1`, `y1`) is the position of `Z[-1, -1]`.

        This argument is ignored if `X` and `Y` are not `None`.
    origin : {`None`, 'lower', 'upper'}, optional
        Determines the orientation and exact position of `Z` by specifying the
        position of `Z[0, 0]`.  This is only relevant, if `X`, `Y` are not
        given.

        - `None`: `Z[0, 0]` is at x=0, y=0 in the lower left corner.
        - `lower`: `Z[0, 0]` is at x=0.5, y=0.5 in the lower left corner.
        - `upper`: `Z[0, 0]` is at x=N+0.5, y=0.5 in the upper left corner.

    Returns
    -------
    X, Y, Z : (M, N) numpy.ndarray
        Three 2D arrays of the same shape containing x, y, and z coordinates.
    """
    n_args = len(args)

    # matplotlib imshow style behaviour
    if n_args == 1:
        Z = np.asarray(args[0])
        if Z.ndim != 2:
            raise TypeError(f"`Z` must be 2D, not {Z.ndim}D")
        n_y, n_x = Z.shape
        if extent is None:
            x = np.arange(n_x)
            y = np.arange(n_y)
        else:
            x_0, x_1, y_0, y_1 = extent
            if origin is None:
                x = np.linspace(x_0, x_1, n_x)
                y = np.linspace(y_0, y_1, n_y)
            else:
                d_x = (x_1 - x_0) / n_x
                d_y = (y_1 - y_0) / n_y
                x = np.arange(x_0, x_1, d_x) + d_x / 2
                y = np.arange(y_0, y_1, d_y) + d_y / 2
            if origin == "upper":
                y = y[::-1]
        X, Y = np.meshgrid(x, y)

    # matplotlib contour style behaviour
    elif n_args == 3:
        X, Y, Z = np.asarray(args)

        if Z.ndim != 2:
            raise TypeError(f"`Z` must be 2D, not {Z.ndim}D")

        if X.ndim == Y.ndim == 1:
            if (Y.size, X.size) != Z.shape:
                raise TypeError(
                    "1D `X` and `Y` must have sizes `Z.shape[1]` and `Z.shape[0]`"
                )
            X, Y = np.meshgrid(X, Y)
        elif not (X.shape == Y.shape == Z.shape):
            raise TypeError("`X` and `Y` must be 1D, or have the same shape as `Z`")
    else:
        raise TypeError(
            f"`uniform_XYZ` expected 1 or 3 non-keyword arguments, not {n_args}."
        )
    return X, Y, Z
