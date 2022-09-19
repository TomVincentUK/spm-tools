# Plans for spm-tools
This should be a library to supplement the already existing image analysis tools in numpy, scipy, scikit-image with common routines needed for SPM analysis. It shouldn't try to reinvent the wheel.

## Code style
* Consistent calling signature:
  * Almost all public functions should expect data to be given like `func(X, Y, Z, *args, **kwargs)`, where `X` and `Y` should be 2D arrays, as if created by `numpy.meshgrid`. `Z` should have the same shape as `X` and `Y`.
  * The functions can then be decorated by `spm_tools._uniform_coords.uniform_XYZ_decorator`, so that the user can call them with the same signature as `matplotlib.pyplot.imshow`, `.pcolormesh` or `.contour`.
* All code should be formatted by black
* numpy style docstrings

## Things to add
- [ ] Unit tests
- [ ] Optional mask argument, to exclude some values from the analysis in the levelling and line correction code
- [ ] Laplace solver for correcting defects
- [ ] Grain marking (for SPM-specific defects not already covered by scikit image)
- [ ] n-point levelling
- [ ] Radial line profiles
- [ ] Upload to conda-forge

## Things to fix
- [ ] Functions need docstrings
- [ ] median of differences line correction gives wrong values

## Things to maybe add (in future)
- [ ] Reading various SPM datafiles
