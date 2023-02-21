# Copyright 2022 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

import numpy as np
import pyttb as ttb
import pytest

@pytest.fixture()
def sample_ttensor():
    core = ttb.tensor.from_data(np.ones((2, 2, 2)))
    factors = [np.ones((1, 2))] * len(core.shape)
    ttensorInstance = ttb.ttensor().from_data(core, factors)
    return ttensorInstance

@pytest.mark.indevelopment
def test_ttensor_initialization_empty():
    empty_tensor = ttb.tensor()

    # No args
    ttensorInstance = ttb.ttensor()
    assert ttensorInstance.core == empty_tensor
    assert ttensorInstance.u == []

@pytest.mark.indevelopment
def test_ttensor_initialization_from_data(sample_ttensor):
    ttensorInstance = sample_ttensor
    assert isinstance(ttensorInstance.core, ttb.tensor)
    assert all([isinstance(a_factor, np.ndarray) for a_factor in ttensorInstance.u])

@pytest.mark.indevelopment
def test_ttensor_initialization_from_tensor_type(sample_ttensor):

    # Copy constructor
    ttensorInstance = sample_ttensor
    ttensorCopy = ttb.ttensor.from_tensor_type(ttensorInstance)
    assert ttensorCopy.core == ttensorInstance.core
    assert ttensorCopy.u == ttensorInstance.u
    assert ttensorCopy.shape == ttensorInstance.shape

@pytest.mark.indevelopment
def test_ttensor_full(sample_ttensor):
    ttensorInstance = sample_ttensor
    tensor = ttensorInstance.full()
    # This sanity check only works for all 1's
    assert tensor.double() == np.prod(ttensorInstance.core.shape)

@pytest.mark.indevelopment
def test_ttensor_double(sample_ttensor):
    ttensorInstance = sample_ttensor
    # This sanity check only works for all 1's
    assert ttensorInstance.double() == np.prod(ttensorInstance.core.shape)

@pytest.mark.indevelopment
def test_ttensor_ndims(sample_ttensor):
    ttensorInstance = sample_ttensor

    assert ttensorInstance.ndims == 3

def test_ttensor__pos__(sample_ttensor):
    ttensorInstance = sample_ttensor
    ttensorInstance2 = +ttensorInstance

    assert ttensorInstance.isequal(ttensorInstance2)

def test_sptensor__neg__(sample_ttensor):
    ttensorInstance = sample_ttensor
    ttensorInstance2 = -ttensorInstance
    ttensorInstance3 = -ttensorInstance2

    assert not ttensorInstance.isequal(ttensorInstance2)
    assert ttensorInstance.isequal(ttensorInstance3)

@pytest.mark.indevelopment
def test_ttensor_innerproduct(sample_ttensor):
    ttensorInstance = sample_ttensor

    # TODO these are an overly simplistic edge case for ttensors that are a single float

    # ttensor innerprod ttensor
    assert ttensorInstance.innerprod(ttensorInstance) == ttensorInstance.double()**2

    # ttensor innerprod tensor
    assert ttensorInstance.innerprod(ttensorInstance.full()) == ttensorInstance.double() ** 2

def test_ttensor__mul__(sample_ttensor):
    ttensorInstance = sample_ttensor
    mul_factor = 2

    # This sanity check only works for all 1's
    assert (ttensorInstance * mul_factor).double() == np.prod(ttensorInstance.core.shape) * mul_factor
    assert (ttensorInstance * float(2)).double() == np.prod(ttensorInstance.core.shape) * float(mul_factor)

def test_ttensor__rmul__(sample_ttensor):
    ttensorInstance = sample_ttensor
    mul_factor = 2

    # This sanity check only works for all 1's
    assert (mul_factor * ttensorInstance).double() == np.prod(ttensorInstance.core.shape) * mul_factor
    assert (float(2) * ttensorInstance).double() == np.prod(ttensorInstance.core.shape) * float(mul_factor)
