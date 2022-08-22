"""
Temporary testing script while I'm working on gsf import
"""
import pathlib

import numpy as np
import matplotlib.pyplot as plt

from spm_tools.io import import_data

fname = pathlib.Path(r"spm_tools").joinpath(r"example_data").joinpath(r"example_gsf.gsf")

X, Y, Z, metadata = import_data(fname)