"""Pyttb pytest configuration."""

# Copyright 2025 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.
from __future__ import annotations

import numpy
import numpy as np

# content of conftest.py
import pytest

import pyttb
import pyttb as ttb


@pytest.fixture(autouse=True)
def add_packages(doctest_namespace):  # noqa: D103
    doctest_namespace["np"] = numpy
    doctest_namespace["ttb"] = pyttb


@pytest.fixture(params=[{"order": "F"}, {"order": "C"}])
def memory_layout(request):
    """Test C and F memory layouts."""
    return request.param


def pytest_addoption(parser):  # noqa: D103
    parser.addoption(
        "--packaging",
        action="store_true",
        dest="packaging",
        default=False,
        help="enable slow packaging tests",
    )


def pytest_configure(config):  # noqa: D103
    if not config.option.packaging:
        config.option.markexpr = "not packaging"


@pytest.fixture()
def sample_tensor_2way():  # noqa: D103
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    shape = (2, 3)
    params = {"data": data, "shape": shape}
    tensorInstance = ttb.tensor(data, shape)
    return params, tensorInstance


@pytest.fixture()
def sample_tensor_3way():  # noqa: D103
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    shape = (2, 3, 2)
    params = {"data": np.reshape(data, np.array(shape), order="F"), "shape": shape}
    tensorInstance = ttb.tensor(data, shape)
    return params, tensorInstance


@pytest.fixture()
def sample_ndarray_1way():  # noqa: D103
    shape = (16,)
    ndarrayInstance = np.reshape(np.arange(1, 17), shape, order="F")
    params = {"data": ndarrayInstance, "shape": shape}
    return params, ndarrayInstance


@pytest.fixture()
def sample_ndarray_2way():  # noqa: D103
    shape = (4, 4)
    ndarrayInstance = np.reshape(np.arange(1, 17), shape, order="F")
    params = {"data": ndarrayInstance, "shape": shape}
    return params, ndarrayInstance


@pytest.fixture()
def sample_ndarray_4way():  # noqa: D103
    shape = (2, 2, 2, 2)
    ndarrayInstance = np.reshape(np.arange(1, 17), shape, order="F")
    params = {"data": ndarrayInstance, "shape": shape}
    return params, ndarrayInstance


@pytest.fixture()
def sample_tenmat_4way():  # noqa: D103
    shape = (4, 4)
    data = np.reshape(np.arange(1, 17), shape, order="F")
    tshape = (2, 2, 2, 2)
    rdims = np.array([0, 1])
    cdims = np.array([2, 3])
    tenmatInstance = ttb.tenmat()
    tenmatInstance.tshape = tshape
    tenmatInstance.rindices = rdims.copy("K")
    tenmatInstance.cindices = cdims.copy("K")
    tenmatInstance.data = data.copy("K")
    params = {
        "data": data,
        "rdims": rdims,
        "cdims": cdims,
        "tshape": tshape,
        "shape": shape,
    }
    return params, tenmatInstance


@pytest.fixture()
def sample_tensor_4way():  # noqa: D103
    data = np.arange(1, 17)
    shape = (2, 2, 2, 2)
    params = {"data": np.reshape(data, np.array(shape), order="F"), "shape": shape}
    tensorInstance = ttb.tensor(data, shape)
    return params, tensorInstance


@pytest.fixture()
def sample_ktensor_2way():  # noqa: D103
    weights = np.array([1.0, 2.0])
    fm0 = np.array([[1.0, 2.0], [3.0, 4.0]])
    fm1 = np.array([[5.0, 6.0], [7.0, 8.0]])
    factor_matrices = [fm0, fm1]
    data = {"weights": weights, "factor_matrices": factor_matrices}
    ktensorInstance = ttb.ktensor(factor_matrices, weights)
    return data, ktensorInstance


@pytest.fixture()
def sample_ktensor_3way():  # noqa: D103
    rank = 2
    shape = (2, 3, 4)
    vector = np.arange(1, rank * sum(shape) + 1).astype(float)
    weights = 2 * np.ones(rank).astype(float)
    vector_with_weights = np.concatenate((weights, vector), axis=0)
    # vector_with_weights = vector_with_weights.reshape((len(vector_with_weights), 1))
    # ground truth
    fm0 = np.array([[1.0, 3.0], [2.0, 4.0]])
    fm1 = np.array([[5.0, 8.0], [6.0, 9.0], [7.0, 10.0]])
    fm2 = np.array([[11.0, 15.0], [12.0, 16.0], [13.0, 17.0], [14.0, 18.0]])
    factor_matrices = [fm0, fm1, fm2]
    data = {
        "weights": weights,
        "factor_matrices": factor_matrices,
        "vector": vector,
        "vector_with_weights": vector_with_weights,
        "shape": shape,
    }
    ktensorInstance = ttb.ktensor(factor_matrices, weights)
    return data, ktensorInstance


@pytest.fixture()
def sample_ktensor_symmetric():  # noqa: D103
    weights = np.array([1.0, 1.0])
    fm0 = np.array(
        [[2.340431417384394, 4.951967353890655], [4.596069112758807, 8.012451489774961]]
    )
    fm1 = np.array(
        [[2.340431417384394, 4.951967353890655], [4.596069112758807, 8.012451489774961]]
    )
    factor_matrices = [fm0, fm1]
    data = {"weights": weights, "factor_matrices": factor_matrices}
    ktensorInstance = ttb.ktensor(factor_matrices, weights)
    return data, ktensorInstance
