# Copyright 2024 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.
from __future__ import annotations

import numpy as np
import pytest

import pyttb as ttb


def test_khatrirao():
    A = np.array([[1, 2, 3], [4, 5, 6]])
    # This result was from MATLAB tensortoolbox, didn't verify by hand
    answer = np.array(
        [
            [1, 8, 27],
            [4, 20, 54],
            [4, 20, 54],
            [16, 50, 108],
            [4, 20, 54],
            [16, 50, 108],
            [16, 50, 108],
            [64, 125, 216],
        ]
    )
    assert np.array_equal(ttb.khatrirao(*[A, A, A]), answer)
    assert np.array_equal(ttb.khatrirao(*[A, A, A], reverse=True), answer)
    assert np.array_equal(ttb.khatrirao(A, A, A), answer)

    # Test case where inputs are column vectors
    a_1 = np.array([[1], [1], [1], [1]])
    a_2 = np.array([[0], [1], [2], [3]])
    a_3 = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
    result = np.vstack(
        (
            a_2[0, 0] * np.ones((16, 1)),
            a_2[1, 0] * np.ones((16, 1)),
            a_2[2, 0] * np.ones((16, 1)),
            a_2[3, 0] * np.ones((16, 1)),
        )
    )
    assert np.array_equal(ttb.khatrirao(*[a_2, a_1, a_1]), result)
    khatrirao_result = ttb.khatrirao(a_2, a_1, a_1)
    assert np.array_equal(khatrirao_result, result)
    assert khatrirao_result.flags["F_CONTIGUOUS"]

    with pytest.raises(AssertionError) as excinfo:
        ttb.khatrirao(a_2, a_1, np.ones((2, 2, 2)))
    assert "Each argument must be a matrix" in str(excinfo)

    with pytest.raises(AssertionError) as excinfo:
        ttb.khatrirao(a_2, a_1, a_3)
    assert "All matrices must have the same number of columns." in str(excinfo)

    # Check old interface error
    with pytest.raises(ValueError):
        ttb.khatrirao([a_1, a_1, a_1])

    with pytest.raises(ValueError):
        ttb.khatrirao(a_1, a_1, reverse="cat")
