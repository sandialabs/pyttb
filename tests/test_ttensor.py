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
