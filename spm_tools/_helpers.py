import numpy as np


def _uniform_XY(X, Y):
    X = np.asarray(X)
    Y = np.asarray(Y)

    if X.ndim == Y.ndim == 1:
        X, Y = np.meshgrid(X, Y)
    elif X.ndim == Y.ndim == 2:
        if X.shape != Y.shape:
            raise TypeError("2D X and Y inputs must have the same shape")
    else:
        raise TypeError("X and Y inputs must be 1D or 2D")

    return X, Y


def _uniform_XYZ(*args, extent=None, origin=None):
    n_args = len(args)

    # matplotlib imshow style behaviour
    if n_args == 1:
        Z = np.asarray(args[0])
        if Z.ndim != 2:
            raise TypeError(f"Input Z must be 2D, not {Z.ndim}D")
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
            raise TypeError(f"Input Z must be 2D, not {Z.ndim}D")

        if X.ndim == Y.ndim == 1:
            if (Y.size, X.size) != Z.shape:
                raise TypeError(
                    "1D X and Y inputs must have sizes Z.shape[1] and Z.shape[0]"
                )
            X, Y = np.meshgrid(X, Y)
        elif not (X.shape == Y.shape == Z.shape):
            raise TypeError("Input X and Y must be 1D, or have the same shape as Z")

    return X, Y, Z
