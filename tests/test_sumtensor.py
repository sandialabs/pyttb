# Copyright 2024 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

import pyttb as ttb
from tests.test_utils import assert_consistent_order


@pytest.fixture()
def example_ttensor():
    """Simple TTENSOR to verify by hand"""
    core_values = np.ones((2, 2))
    core = ttb.tensor(core_values)
    factors = [np.ones((2, 2))] * len(core_values.shape)
    return ttb.ttensor(core, factors)


@pytest.fixture()
def example_kensor():
    """Simple KTENSOR to verify by hand"""
    weights = np.array([1.0, 2.0])
    fm0 = np.array([[1.0, 2.0], [3.0, 4.0]])
    fm1 = np.array([[5.0, 6.0], [7.0, 8.0]])
    return ttb.ktensor([fm0, fm1], weights)


def test_sumtensor_initialization_init():
    # Create empty sumtensor
    S = ttb.sumtensor()
    assert len(S.parts) == 0

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

    # Empty case
    assert ttb.sumtensor().shape == ()


def test_sumtensor_initialization_str():
    shape = (3, 4, 5)
    T1 = ttb.tenones(shape)
    T2 = ttb.tenones(shape)
    S = ttb.sumtensor([T1, T2])
    assert str(S) == S.__repr__()
    assert f"sumtensor of shape {shape}" in str(S)

    assert "Empty sumtensor" in str(ttb.sumtensor())


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


def test_sumtensor_pos():
    T1 = ttb.tenones((3, 4, 5))
    T2 = ttb.tenones((3, 4, 5))
    S1 = ttb.sumtensor([T1, T2])
    S2 = +S1
    assert S1 is not S2
    assert S1.parts == S2.parts


def test_sumtensor_neg():
    T1 = ttb.tenones((3, 4, 5))
    T2 = ttb.tenones((3, 4, 5))
    S1 = ttb.sumtensor([T1, T2])
    S2 = -S1
    assert S1 is not S2
    assert S1.parts == [-part for part in S2.parts]


def test_sumtensor_add_tensors():
    T1 = ttb.tenones((3, 4, 5))
    T2 = ttb.tenones((3, 4, 5))
    S1 = ttb.sumtensor([T1])
    assert len(S1.parts) == 1
    assert len((S1 + T2).parts) == 2
    assert len((T2 + S1).parts) == 2
    assert len((S1 + [T1, T2]).parts) == 3


def test_sumtensor_add_sptensors():
    T1 = ttb.sptensor(shape=(3, 4, 5))
    S1 = ttb.sumtensor([T1])
    assert len(S1.parts) == 1
    assert len((S1 + T1).parts) == 2
    assert len((T1 + S1).parts) == 2
    assert len((S1 + [T1, T1]).parts) == 3


def test_sumtensor_add_ktensors(example_kensor):
    K = example_kensor
    S1 = ttb.sumtensor([K])
    assert len(S1.parts) == 1
    assert len((S1 + K).parts) == 2
    assert len((K + S1).parts) == 2
    assert len((S1 + [K, K]).parts) == 3


def test_sumtensor_add_ttensors(example_ttensor):
    K = example_ttensor
    S1 = ttb.sumtensor([K])
    assert len(S1.parts) == 1
    assert len((S1 + K).parts) == 2
    assert len((K + S1).parts) == 2
    assert len((S1 + [K, K]).parts) == 3


def test_sumtensor_add_incorrect_type():
    T1 = ttb.tenones((3, 4, 5))
    S1 = ttb.sumtensor([T1])
    non_tensor_object = 5
    with pytest.raises(TypeError):
        S1 + non_tensor_object


def test_sumtensor_full_double(example_ttensor, example_kensor):
    T1 = ttb.tenones((2, 2))
    T2 = ttb.sptensor(shape=(2, 2))
    K = example_kensor
    TT = example_ttensor
    S = ttb.sumtensor([T1, T2, K, TT])
    # Smoke test that all type combine
    assert isinstance(S.full(), ttb.tensor)
    double_array = S.double()
    assert isinstance(double_array, np.ndarray)
    assert_consistent_order(S, double_array)

    # Verify immutability
    double_array = S.double(True)
    with pytest.raises(ValueError):
        double_array[0] = 1


def test_sumtensor_innerprod(example_ttensor, example_kensor):
    T1 = ttb.tenones((2, 2))
    T2 = T1.to_sptensor()
    K = example_kensor
    TT = example_ttensor
    S = ttb.sumtensor([T1, T2, K, TT])
    result = S.innerprod(T1)
    expected = sum(part.innerprod(T1) for part in S.parts)
    assert result == expected


def test_sumtensor_ttv(example_ttensor, example_kensor):
    T1 = ttb.tenones((2, 2))
    T2 = ttb.sptensor(shape=(2, 2))
    K = example_kensor
    TT = example_ttensor
    S = ttb.sumtensor([T1, T2, K, TT])
    S.ttv(np.ones(2), 0)


def test_sumtensor_initialization_norm():
    with pytest.warns(Warning):
        norm = ttb.sumtensor().norm()
    assert norm == 0.0
