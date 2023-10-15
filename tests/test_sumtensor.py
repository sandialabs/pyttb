# Copyright 2022 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.
from copy import deepcopy

import pytest

import pyttb as ttb


def test_sumtensor_initialization_init():
    # Basic smoke test
    T1 = ttb.tenones((3, 4, 5))
    T2 = ttb.sptensor(shape=(3, 4, 5))
    ttb.sumtensor([T1, T2])

    # Verify copy
    T1 = ttb.tenones((3, 4, 5))
    T2 = ttb.tenones((3, 4, 5))
    S = ttb.sumtensor([T1, T2], copy=True)
    S.parts[0] *= 2
    assert not S.parts[0].isequal(T1)

    # Negative Tests
    ## Mismatched shapes
    T1 = ttb.tenones((3, 4, 5))
    T2 = ttb.tenones((3, 4, 6))
    with pytest.raises(AssertionError):
        ttb.sumtensor([T1, T2])


def test_sumtensor_initialization_shape():
    # Basic smoke test
    shape = (3, 4, 5)
    T1 = ttb.tenones(shape)
    T2 = ttb.sptensor(shape=shape)
    S = ttb.sumtensor([T1, T2])
    assert S.shape == shape


def test_sumtensor_initialization_str():
    shape = (3, 4, 5)
    T1 = ttb.tenones(shape)
    T2 = ttb.tenones(shape)
    S = ttb.sumtensor([T1, T2])
    assert str(S) == S.__repr__()
    assert f"sumtensor of shape {shape}" in str(S)


def test_sumtensor_deepcopy():
    T1 = ttb.tenones((3, 4, 5))
    T2 = ttb.tenones((3, 4, 5))
    S1 = ttb.sumtensor([T1, T2])
    S2 = S1.copy()
    S2.parts[0] *= 2
    assert not S1.parts[0].isequal(S2.parts[0])

    T1 = ttb.tenones((3, 4, 5))
    T2 = ttb.tenones((3, 4, 5))
    S1 = ttb.sumtensor([T1, T2])
    S2 = deepcopy(S1)
    S2.parts[0] *= 2
    assert not S1.parts[0].isequal(S2.parts[0])
