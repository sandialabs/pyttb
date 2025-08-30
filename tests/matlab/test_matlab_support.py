# Copyright 2024 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.
from __future__ import annotations

import numpy as np
import pytest

from pyttb import matlab_support, tensor


def test_matlab_printing_negative():
    with pytest.raises(ValueError):
        matlab_support.matlab_print("foo")


def test_np_printing():
    """These are just smoke tests since formatting needs manual style verification."""
    # Check different dimensionality support
    one_d_array = np.ones((1,))
    matlab_support.matlab_print(one_d_array)
    two_d_array = np.ones((1, 1))
    matlab_support.matlab_print(two_d_array)
    three_d_array = np.ones((1, 1, 1))
    matlab_support.matlab_print(three_d_array)

    # Check name and format
    matlab_support.matlab_print(one_d_array, format="5.1f", name="X")
    matlab_support.matlab_print(two_d_array, format="5.1f", name="X")
    matlab_support.matlab_print(three_d_array, format="5.1f", name="X")


def test_dense_printing():
    """These are just smoke tests since formatting needs manual style verification."""
    # Check different dimensionality support
    example = tensor(np.arange(16), shape=(2, 2, 2, 2))
    # 4D
    matlab_support.matlab_print(example)
    # 2D
    matlab_support.matlab_print(example[:, :, 0, 0])
    # 1D
    matlab_support.matlab_print(example[:, 0, 0, 0])

    # Check name and format
    matlab_support.matlab_print(example, format="5.1f", name="X")
