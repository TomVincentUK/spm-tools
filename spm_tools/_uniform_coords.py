import functools
import warnings

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


def uniform_XYZ_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        n_args = len(args)
        if n_args == 0:
            if all([kw in kwargs for kw in ("X", "Y", "Z")]):
                X = kwargs.pop("X")
                Y = kwargs.pop("Y")
                Z = kwargs.pop("Z")
            elif "Z" in kwargs and not any([kw in kwargs for kw in ("X", "Y")]):
                X = Y = None
                Z = kwargs.pop("Z")
            else:
                raise TypeError("`X`, `Y` and `Z` arguments could not be understood")
        elif n_args == 1:
            X = Y = None
            (Z,) = args
        elif n_args == 3:
            X, Y, Z = args
        else:
            raise TypeError(
                f"`{func.__name__}` expected 0, 1, or 3 positional arguments"
            )

        if "extent" in kwargs:
            extent = kwargs.pop("extent")
        else:
            extent = None

        if "origin" in kwargs:
            origin = kwargs.pop("origin")
        else:
            origin = None

        if Z.ndim != 2:
            raise TypeError(f"`Z` must be 2D, not {Z.ndim}D")

        # matplotlib imshow style behaviour
        if (X is None) or (Y is None):
            if not ((X is None) and (Y is None)):
                raise TypeError("If either `X` or `Y` is None, then both must be None")

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

        # matplotlib pcolormesh/contour style behaviour
        else:
            if extent is not None:
                warnings.warn(
                    "`extent` keyword argument ignored when `X` and `Y` are specified",
                    category=UserWarning,
                )

            if X.ndim == Y.ndim == 1:
                if (Y.size, X.size) != Z.shape:
                    raise TypeError(
                        "1D `X` and `Y` must have sizes `Z.shape[1]` and `Z.shape[0]`"
                    )
                X, Y = np.meshgrid(X, Y)
            elif not (X.shape == Y.shape == Z.shape):
                raise TypeError("`X` and `Y` must be 1D, or have the same shape as `Z`")

        return func(X, Y, Z, **kwargs)

    return wrapper


@uniform_XYZ_decorator
def uniform_XYZ(X, Y, Z, **kwargs):
    if kwargs:
        inner_str = "`, `".join([k for k in kwargs.keys()])
        warnings.warn(f"unused kwargs: `{inner_str}`")
    return X, Y, Z
