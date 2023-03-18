# Copyright 2022 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

import logging

import numpy as np
import pytest
import scipy.sparse as sparse

import pyttb as ttb
from pyttb.sptensor import tt_from_sparse_matrix, tt_to_sparse_matrix


@pytest.fixture()
def sample_sptensor():
    subs = np.array([[1, 1, 1], [1, 1, 3], [2, 2, 2], [3, 3, 3]])
    vals = np.array([[0.5], [1.5], [2.5], [3.5]])
    shape = (4, 4, 4)
    data = {"subs": subs, "vals": vals, "shape": shape}
    sptensorInstance = ttb.sptensor().from_data(subs, vals, shape)
    return data, sptensorInstance


@pytest.mark.indevelopment
def test_sptensor_initialization_empty():
    empty = np.array([], ndmin=2, dtype=int)

    # No args
    sptensorInstance = ttb.sptensor()
    assert (sptensorInstance.subs == empty).all()
    assert (sptensorInstance.vals == empty).all()
    assert (sptensorInstance.shape == empty).all()


@pytest.mark.indevelopment
def test_sptensor_initialization_from_data(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor
    assert (sptensorInstance.subs == data["subs"]).all()
    assert (sptensorInstance.vals == data["vals"]).all()
    assert sptensorInstance.shape == data["shape"]


@pytest.mark.indevelopment
def test_sptensor_initialization_from_tensor_type(sample_sptensor):
    # Copy constructor
    (data, sptensorInstance) = sample_sptensor
    sptensorCopy = ttb.sptensor.from_tensor_type(sptensorInstance)
    assert (sptensorCopy.subs == data["subs"]).all()
    assert (sptensorCopy.vals == data["vals"]).all()
    assert sptensorCopy.shape == data["shape"]

    # Convert Tensor
    inputData = np.array([[1, 2, 3], [4, 5, 6]])
    tensorInstance = ttb.tensor.from_data(inputData)
    sptensorFromTensor = ttb.sptensor.from_tensor_type(tensorInstance)
    logging.debug(f"inputData = {inputData}")
    logging.debug(f"tensorInstance = {tensorInstance}")
    logging.debug(f"sptensorFromTensor = {sptensorFromTensor}")
    assert (
        sptensorFromTensor.subs
        == ttb.tt_ind2sub(inputData.shape, np.arange(0, inputData.size))
    ).all()
    assert (
        sptensorFromTensor.vals == inputData.reshape((inputData.size, 1), order="F")
    ).all()
    assert sptensorFromTensor.shape == inputData.shape

    # From coo sparse matrix
    inputData = sparse.random(11, 4, 0.2)
    sptensorFromCOOMatrix = ttb.sptensor.from_tensor_type(sparse.coo_matrix(inputData))
    assert (sptensorFromCOOMatrix.spmatrix() != sparse.coo_matrix(inputData)).nnz == 0

    # Negative Tests
    with pytest.raises(AssertionError):
        invalid_tensor_type = []
        ttb.sptensor.from_tensor_type(invalid_tensor_type)


@pytest.mark.indevelopment
def test_sptensor_initialization_from_function():
    # Random Tensor Success
    def function_handle(*args):
        return np.array([[0.5], [1.5], [2.5], [3.5], [4.5], [5.5]])

    np.random.seed(123)
    shape = (4, 4, 4)
    nz = 6
    sptensorInstance = ttb.sptensor.from_function(function_handle, shape, nz)
    assert (sptensorInstance.vals == function_handle()).all()
    assert sptensorInstance.shape == shape
    assert len(sptensorInstance.subs) == nz

    # NZ as a propotion in [0,1)
    nz = 0.09375
    sptensorInstance = ttb.sptensor.from_function(function_handle, shape, nz)
    assert (sptensorInstance.vals == function_handle()).all()
    assert sptensorInstance.shape == shape
    assert len(sptensorInstance.subs) == int(nz * np.prod(shape))

    # Random Tensor exception for negative non-zeros
    nz = -1
    with pytest.raises(AssertionError) as excinfo:
        a = ttb.sptensor.from_function(function_handle, shape, nz)
    assert (
        "Requested number of non-zeros must be positive and less than the total size"
        in str(excinfo)
    )

    # Random Tensor exception for negative non-zeros
    nz = np.prod(shape) + 1
    with pytest.raises(AssertionError) as excinfo:
        a = ttb.sptensor.from_function(function_handle, shape, nz)
    assert (
        "Requested number of non-zeros must be positive and less than the total size"
        in str(excinfo)
    )


@pytest.mark.indevelopment
def test_sptensor_initialization_from_aggregator(sample_sptensor):
    subs = np.array([[1, 1, 1], [1, 1, 3], [2, 2, 2], [3, 3, 3], [1, 1, 1], [1, 1, 1]])
    vals = np.array([[0.5], [1.5], [2.5], [3.5], [4.5], [5.5]])
    shape = (4, 4, 4)
    a = ttb.sptensor.from_aggregator(subs, vals, shape)
    assert (a.subs == np.array([[1, 1, 1], [1, 1, 3], [2, 2, 2], [3, 3, 3]])).all()
    assert (a.vals == np.array([[10.5], [1.5], [2.5], [3.5]])).all()
    assert a.shape == shape

    a = ttb.sptensor.from_aggregator(subs, vals)
    assert (a.subs == np.array([[1, 1, 1], [1, 1, 3], [2, 2, 2], [3, 3, 3]])).all()
    assert (a.vals == np.array([[10.5], [1.5], [2.5], [3.5]])).all()
    assert a.shape == shape

    a = ttb.sptensor.from_aggregator(np.array([]), vals, shape)
    assert a.isequal(ttb.sptensor.from_data(np.array([]), np.array([]), shape))

    with pytest.raises(AssertionError) as excinfo:
        ttb.sptensor.from_aggregator(subs, np.concatenate((vals, np.array([[1.0]]))))
    assert "Number of subscripts and values must be equal" in str(excinfo)

    with pytest.raises(AssertionError) as excinfo:
        ttb.sptensor.from_aggregator(
            np.concatenate((subs, np.ones((6, 1))), axis=1), vals, shape
        )
    assert "More subscripts than specified by shape" in str(excinfo)

    badSubs = subs.copy()
    badSubs[0, 0] = 11
    with pytest.raises(AssertionError) as excinfo:
        ttb.sptensor.from_aggregator(badSubs, vals, shape)
    assert "Subscript exceeds sptensor shape" in str(excinfo)


@pytest.mark.indevelopment
def test_sptensor_and_scalar(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor

    b = sptensorInstance.logical_and(0)
    assert (b.subs == np.array([])).all()
    assert (b.vals == np.array([])).all()
    assert b.shape == data["shape"]

    b = sptensorInstance.logical_and(0.5)
    assert (b.subs == data["subs"]).all()
    assert (b.vals == np.array([[True], [False], [False], [False]])).all()
    assert b.shape == data["shape"]


@pytest.mark.indevelopment
def test_sptensor_and_sptensor(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor
    b = sptensorInstance.logical_and(sptensorInstance)

    assert (b.subs == data["subs"]).all()
    assert (b.vals == np.array([[True], [True], [True], [True]])).all()
    assert b.shape == data["shape"]

    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance.logical_and(
            ttb.sptensor.from_data(data["subs"], data["vals"], (5, 5, 5))
        )
    assert "Must be tensors of the same shape" in str(excinfo)

    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance.logical_and(np.ones(data["shape"]))
    assert "The arguments must be two sptensors or an sptensor and a scalar." in str(
        excinfo
    )


@pytest.mark.indevelopment
def test_sptensor_and_tensor(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor
    b = sptensorInstance.logical_and(ttb.tensor.from_tensor_type(sptensorInstance))
    assert (b.subs == data["subs"]).all()
    assert (b.vals == np.ones(data["vals"].shape)).all()


@pytest.mark.indevelopment
def test_sptensor_full(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor
    densetensor = sptensorInstance.full()
    denseData = np.zeros(sptensorInstance.shape)
    actualIdx = tuple(data["subs"].transpose())
    denseData[actualIdx] = data["vals"].transpose()[0]

    assert (densetensor.data == denseData).all()
    assert densetensor.shape == data["shape"]

    # Empty, no shape tensor conversion
    emptySptensor = ttb.sptensor()
    emptyTensor = ttb.tensor()
    assert emptyTensor.isequal(emptySptensor.full())

    # Empty, no non-zeros tensor conversion
    emptySptensor = ttb.sptensor.from_data(np.array([]), np.array([]), data["shape"])
    assert (emptySptensor.full().data == np.zeros(data["shape"])).all()


@pytest.mark.indevelopment
def test_sptensor_subdims(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor

    assert (sptensorInstance.subdims([[1], [1], [1, 3]]) == np.array([0, 1])).all()
    assert (
        sptensorInstance.subdims((1, 1, slice(None, None, None))) == np.array([0, 1])
    ).all()

    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance.subdims([[1], [1, 3]])
    assert "Number of subdimensions must equal number of dimensions" in str(excinfo)

    with pytest.raises(ValueError):
        sptensorInstance.subdims(("bad", "region", "types"))


@pytest.mark.indevelopment
def test_sptensor_ndims(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor

    assert sptensorInstance.ndims == 3


@pytest.mark.indevelopment
def test_sptensor_extract(sample_sptensor, capsys):
    (data, sptensorInstance) = sample_sptensor

    # Out of range subs case
    # Too large
    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance.extract(np.array([[4, 4, 4], [1, 1, 1]]))
    assert "Invalid subscripts" in str(excinfo)
    capsys.readouterr()
    # Negative #TODO, would we like to support reverse indexing which is pythonic
    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance.extract(np.array([[-1, -1, -1], [1, 1, 1]]))
    assert "Invalid subscripts" in str(excinfo)
    capsys.readouterr()

    # List of subs case
    assert (
        sptensorInstance.extract(np.array([[1, 1, 1], [1, 1, 3]])) == [[0.5], [1.5]]
    ).all()

    # Single sub case
    # TODO if you pass a single sub should you get a list of vals with one entry or just a single val
    assert (sptensorInstance.extract(np.array([1, 1, 1])) == [[0.5]]).all()


@pytest.mark.indevelopment
def test_sptensor__getitem__(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor
    ## Case 1
    # Empty value slice
    assert sptensorInstance[0, 0, 0] == 0
    # Full value slice
    assert sptensorInstance[1, 1, 1] == 0.5
    # Empty subtensor
    emptyResult = sptensorInstance[0:1, 0:1, 0:1]
    assert isinstance(emptyResult, ttb.sptensor)
    assert emptyResult.isequal(
        ttb.sptensor.from_data(np.array([]), np.array([]), (1, 1, 1))
    )
    # Full subtensor
    assert isinstance(sptensorInstance[:, :, :], ttb.sptensor)
    assert isinstance(
        sptensorInstance[[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]], ttb.sptensor
    )
    assert sptensorInstance[:, :, :].isequal(
        sptensorInstance[[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
    )

    # TODO need to understand what this intends to do
    ## Case 2 subscript indexing
    assert sptensorInstance[np.array([[1, 2, 1]])] == np.array([[0]])
    assert (
        sptensorInstance[np.array([[1, 2, 1], [1, 3, 1]])] == np.array([[0], [0]])
    ).all()

    ## Case 2 Linear Indexing
    ind = ttb.tt_sub2ind(data["shape"], np.array([[1, 1, 1], [1, 1, 3], [2, 2, 2]]))
    assert (sptensorInstance[ind] == np.array([[0.5], [1.5], [2.5]])).all()
    list_ind = list(ind)
    assert (sptensorInstance[list_ind] == np.array([[0.5], [1.5], [2.5]])).all()
    ind2 = ttb.tt_sub2ind(data["shape"], np.array([[1, 1, 1], [1, 1, 3]]))
    assert (sptensorInstance[ind2] == np.array([[0.5], [1.5]])).all()
    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance[ind2[:, None]]
    assert "Expecting a row index" in str(excinfo)
    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance["string"]
    assert "Invalid indexing" in str(excinfo)


@pytest.mark.indevelopment
def test_sptensor_setitem_Case1(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor

    # Empty sptensor assigned with nothing
    emptyTensor = ttb.sptensor()
    emptyTensor[:, :, :] = []
    assert emptyTensor.vals.size == 0
    emptyTensor[:, :, :] = np.array([])
    assert emptyTensor.vals.size == 0

    # Case I(a): Set empty tensor with sptensor
    emptyTensor = ttb.sptensor()
    emptyTensor[:, :, :] = sptensorInstance
    assert (emptyTensor.subs == data["subs"]).all()
    assert (emptyTensor.vals == data["vals"]).all()
    assert emptyTensor.shape == data["shape"]

    # Case I(a): Set empty tensor with sptensor, none none end slice
    emptyTensor = ttb.sptensor()
    emptyTensor[0:4, 0:4, 0:4] = sptensorInstance
    assert (emptyTensor.subs == data["subs"]).all()
    assert (emptyTensor.vals == data["vals"]).all()
    assert emptyTensor.shape == data["shape"]

    # Case I(a): Set sptensor with empty tensor
    emptyTensor = ttb.sptensor()
    emptyTensor.shape = (4, 4, 4)
    sptensorCopy = ttb.sptensor.from_tensor_type(sptensorInstance)
    sptensorCopy[:, :, :] = emptyTensor
    assert (sptensorCopy.subs == emptyTensor.subs).all()
    assert (sptensorCopy.vals == emptyTensor.vals).all()
    assert sptensorCopy.shape == data["shape"]

    # Case I(a): Set sptensor with smaller tensor
    emptyTensor = ttb.sptensor()
    emptyTensor.shape = (4, 4, 4)
    sptensorCopy = ttb.sptensor.from_tensor_type(sptensorInstance)
    sptensorInstanceCopy = ttb.sptensor.from_tensor_type(sptensorInstance)
    sptensorInstanceCopy[1, 1, 1] = 0
    sptensorCopy[4, 4, 4] = 1
    sptensorCopy[:4, :4, :4] = sptensorInstanceCopy
    assert (sptensorCopy.subs[1:, :] == sptensorInstanceCopy.subs).all()
    assert (sptensorCopy.vals[1:] == sptensorInstanceCopy.vals).all()
    assert sptensorCopy.shape == (5, 5, 5)

    # Case I(a): Set sptensor with smaller tensor
    emptyTensor = ttb.sptensor()
    emptyTensor.shape = (4, 4, 4)
    sptensorCopy = ttb.sptensor.from_tensor_type(sptensorInstance)
    sptensorCopy[4, 4, 4] = 1
    sptensorCopy[:4, :4, :4] = emptyTensor
    assert sptensorCopy.subs[1:, :].size == 0
    assert sptensorCopy.vals[1:].size == 0
    assert sptensorCopy.shape == (5, 5, 5)

    # Case I(a): Set sptensor with larger empty tensor
    emptyTensor = ttb.sptensor.from_data(np.array([]), np.array([]), (4, 4, 4, 4))
    sptensorCopy = ttb.sptensor.from_tensor_type(sptensorInstance)
    sptensorCopy[:4, :4, :4, :4] = emptyTensor
    assert (sptensorCopy.subs == emptyTensor.subs).all()
    assert (sptensorCopy.vals == emptyTensor.vals).all()
    assert sptensorCopy.shape == emptyTensor.shape

    # Case I(a): Set sptensor with sptensor
    subs = np.array([[2, 1, 1], [2, 1, 3]])
    vals = np.array([[2.5], [3.5]])
    shape = (4, 4, 4)
    sptensorRHS = ttb.sptensor.from_data(subs, vals, shape)

    sptensorCopy = ttb.sptensor.from_tensor_type(sptensorInstance)
    # TODO slicing in this way isn't supported in tensor toolbox
    # sptensorCopy[2, 1, :] = sptensorRHS
    # assert (sptensorCopy.subs == np.vstack((sptensorCopy.subs, sptensorRHS.subs))).all()
    # assert (sptensorCopy.vals == np.vstack((sptensorCopy.vals, sptensorRHS.vals))).all()
    # assert (sptensorCopy.shape == data['shape'])

    # Case I(a): Set empty with same size sptensor
    emptyTensor = ttb.sptensor.from_data(np.array([]), np.array([]), (1, 1, 1))
    sptensorCopy = ttb.sptensor.from_tensor_type(emptyTensor)
    sptensorCopy[0, 0, 0] = 1
    emptyTensor[0, 0, 0] = sptensorCopy
    # TODO: This ne should be eq once irenumber is resolved
    assert sptensorCopy.subs.shape != emptyTensor.subs.shape
    assert sptensorCopy.vals == emptyTensor.vals
    assert sptensorCopy.shape == emptyTensor.shape

    # Case I(a): Set empty with same size sptensor
    emptyTensor = ttb.sptensor.from_data(np.array([]), np.array([]), (1, 1, 1))
    sptensorCopy = ttb.sptensor.from_data(np.array([]), np.array([]), (1, 1, 1, 1))
    sptensorCopy[0, 0, 0, 0] = 1
    emptyTensor[0, 0, 0, 0] = sptensorCopy
    # TODO: This ne should be eq once irenumber is resolved
    assert sptensorCopy.subs.shape != emptyTensor.subs.shape
    assert sptensorCopy.vals == emptyTensor.vals
    # Since we do a single index set item the size is only set large enough for that element
    assert sptensorCopy.shape == emptyTensor.shape

    # Case I(a): Set empty with same size sptensor
    emptyTensor = ttb.sptensor.from_data(np.array([]), np.array([]), (2, 2, 2))
    sptensorCopy = ttb.sptensor.from_data(np.array([]), np.array([]), (2, 2, 2, 2))
    sptensorCopy[0, 0, 0, 0] = 1
    sptensorCopy[1, 1, 1, 1] = 1
    emptyTensor[[0, 1], [0, 1], [0, 1], [0, 1]] = sptensorCopy
    # TODO: This ne should be eq once irenumber is resolved
    assert (sptensorCopy.subs == emptyTensor.subs).all()
    assert (sptensorCopy.vals == emptyTensor.vals).all()
    assert sptensorCopy.shape == emptyTensor.shape

    sptensorCopy = ttb.sptensor.from_data(np.array([]), np.array([]), (1, 1, 1, 1))
    with pytest.raises(AssertionError) as excinfo:
        emptyTensor[[0, 1], [0, 1], [0, 1], [0, 1]] = sptensorCopy
    assert "RHS does not match range size" in str(excinfo)

    with pytest.raises(AssertionError) as excinfo:
        emptyTensor[
            np.array([0, 1]), np.array([0, 1]), np.array([0, 1]), np.array([0, 1])
        ] = sptensorCopy
    assert "RHS does not match range size" in str(excinfo)

    # Case I(b)i: Set with zero, sub already exists
    old_value = data["vals"][1, 0]
    sptensorInstance[1, 1, 3] = 0
    subSelection = [0, 2, 3]
    assert (sptensorInstance.subs == data["subs"][subSelection]).all()
    assert (sptensorInstance.vals == data["vals"][subSelection]).all()
    assert sptensorInstance.shape == data["shape"]

    # Case I(b)i: Set with zero, sub doesn't exist
    sptensorInstance[1, 1, 3] = old_value
    reorder = [0, 2, 3, 1]
    assert (sptensorInstance.subs == data["subs"][reorder]).all()
    assert (sptensorInstance.vals == data["vals"][reorder]).all()
    assert sptensorInstance.shape == data["shape"]
    # Reset tensor data
    data["subs"] = data["subs"][reorder]
    data["vals"] = data["vals"][reorder]

    # Case I(b)i: Set slice with zero, sub already exists
    old_value = data["vals"][3, 0]
    sptensorInstance[1:2, 1:2, 3:4] = 0
    subSelection = [0, 1, 2]
    assert (sptensorInstance.subs == data["subs"][subSelection]).all()
    assert (sptensorInstance.vals == data["vals"][subSelection]).all()
    assert sptensorInstance.shape == data["shape"]
    # Reset tensor data
    sptensorInstance[1, 1, 3] = old_value

    # Case I(b)i: Set slice with zero, sub already exists
    old_value = data["vals"][2, 0]
    sptensorInstance[3:, 3:, 3:] = 0
    subSelection = [0, 1, 3]
    assert (sptensorInstance.subs == data["subs"][subSelection]).all()
    assert (sptensorInstance.vals == data["vals"][subSelection]).all()
    assert sptensorInstance.shape == data["shape"]
    # Reset tensor data
    sptensorInstance[3, 3, 3] = old_value
    reorder = [0, 1, 3, 2]
    # Reset tensor data
    data["subs"] = data["subs"][reorder]
    data["vals"] = data["vals"][reorder]

    # Case I(b)i: Expand Shape of sptensor with set item
    sptensorInstanceLarger = ttb.sptensor.from_tensor_type(sptensorInstance)
    sptensorInstanceLarger[1, 1, 1, 1] = 0
    assert sptensorInstanceLarger.shape == (4, 4, 4, 2)

    # Case I(b)i: Expand Shape of sptensor with set item
    sptensorInstanceLarger = ttb.sptensor.from_tensor_type(sptensorInstance)
    sptensorInstanceLarger[1, 1, 1, 1:2] = 0
    assert sptensorInstanceLarger.shape == (4, 4, 4, 2)

    # Case I(b)i: Expand Shape of sptensor with set item
    sptensorInstanceLarger = ttb.sptensor.from_tensor_type(sptensorInstance)
    sptensorInstanceLarger[1, 1, 1, np.array([1])] = 0
    assert sptensorInstanceLarger.shape == (4, 4, 4, 2)

    # Case I(b)i: Expand Shape of sptensor with set item
    sptensorInstanceLarger = ttb.sptensor.from_tensor_type(sptensorInstance)
    with pytest.raises(AssertionError) as excinfo:
        sptensorInstanceLarger[1, 1, 1, 1:] = 0
    assert (
        "Must have well defined slice when expanding sptensor shape with setitem"
        in str(excinfo)
    )

    # Case I(b)ii: Set with scalar, sub already exists
    old_value = data["vals"][2, 0]
    sptensorInstance[1, 1, 3] = 7
    modifiedVals = data["vals"].copy()
    modifiedVals[2] = 7
    assert (sptensorInstance.subs == data["subs"]).all()
    assert (sptensorInstance.vals == modifiedVals).all()
    assert sptensorInstance.shape == data["shape"]
    sptensorInstance[1, 1, 3] = old_value  # Reset tensor

    # Case I(b)ii: Set with scalar, sub already exists
    old_value = data["vals"][2, 0]
    sptensorInstance[1:2, 1:2, 3:4] = 7
    modifiedVals = data["vals"].copy()
    modifiedVals[2] = 7
    assert (sptensorInstance.subs == data["subs"]).all()
    assert (sptensorInstance.vals == modifiedVals).all()
    assert sptensorInstance.shape == data["shape"]
    sptensorInstance[1, 1, 3] = old_value  # Reset tensor

    # Case I(b)ii: Set with scalar, sub doesn't exist yet
    sptensorInstance[1, 1, 2] = 7
    assert (
        sptensorInstance.subs == np.vstack((data["subs"], np.array([[1, 1, 2]])))
    ).all()
    assert (sptensorInstance.vals == np.vstack((data["vals"], np.array([[7]])))).all()
    assert sptensorInstance.shape == data["shape"]

    # Case I: Assign with non-scalar or sptensor
    sptensorInstanceLarger = ttb.sptensor.from_tensor_type(sptensorInstance)
    with pytest.raises(AssertionError) as excinfo:
        sptensorInstanceLarger[1, 1, 1] = "String"
    assert "Invalid assignment value" in str(excinfo)


@pytest.mark.indevelopment
def test_sptensor_setitem_Case2(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor

    # Case II: Too few modes in setitem key
    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance[np.array([1, 1]).astype(int)] = 999.0
    assert "Invalid subscripts" in str(excinfo)

    # Case II: Too few keys in setitem for number of assignement values
    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance[np.array([1, 1, 1]).astype(int)] = np.array([[999.0], [888.0]])
    assert "Number of subscripts and number of values do not match!" in str(excinfo)

    # Case II: Warning For duplicates
    with pytest.warns(Warning) as record:
        sptensorInstance[np.array([[1, 1, 1], [1, 1, 1]]).astype(int)] = np.array(
            [[999.0], [999.0]]
        )
    assert "Duplicate assignments discarded" in str(record[0].message)

    # Case II: Single entry, for single sub that exists
    sptensorInstance[np.array([1, 1, 1]).astype(int)] = 999.0
    assert (sptensorInstance[np.array([[1, 1, 1]])] == np.array([[999]])).all()
    assert (sptensorInstance.subs == data["subs"]).all()

    # Case II: Single entry, for multiple subs that exist
    (data, sptensorInstance) = sample_sptensor
    sptensorInstance[np.array([[1, 1, 1], [1, 1, 3]]).astype(int)] = 999.0
    assert (
        sptensorInstance[np.array([[1, 1, 1], [1, 1, 3]])] == np.array([[999], [999]])
    ).all()
    assert (sptensorInstance.subs == data["subs"]).all()

    # Case II: Multiple entries, for multiple subs that exist
    (data, sptensorInstance) = sample_sptensor
    sptensorInstance[np.array([[1, 1, 1], [1, 1, 3]]).astype(int)] = np.array(
        [[888], [999]]
    )
    assert (
        sptensorInstance[np.array([[1, 1, 3], [1, 1, 1]])] == np.array([[999], [888]])
    ).all()
    assert (sptensorInstance.subs == data["subs"]).all()

    # Case II: Single entry, for single sub that doesn't exist
    (data, sptensorInstance) = sample_sptensor
    copy = ttb.sptensor.from_tensor_type(sptensorInstance)
    copy[np.array([[1, 1, 2]]).astype(int)] = 999.0
    assert (copy[np.array([[1, 1, 2]])] == np.array([999])).all()
    assert (copy.subs == np.concatenate((data["subs"], np.array([[1, 1, 2]])))).all()

    # Case II: Single entry, for single sub that doesn't exist, expand dimensions
    (data, sptensorInstance) = sample_sptensor
    copy = ttb.sptensor.from_tensor_type(sptensorInstance)
    copy[np.array([[1, 1, 2, 1]]).astype(int)] = 999.0
    assert (copy[np.array([[1, 1, 2, 1]])] == np.array([999])).all()
    # assert (copy.subs == np.concatenate((data['subs'], np.array([[1, 1, 2]])))).all()

    # Case II: Single entry, for multiple subs one that exists and the other doesn't
    (data, sptensorInstance) = sample_sptensor
    copy = ttb.sptensor.from_tensor_type(sptensorInstance)
    copy[np.array([[1, 1, 1], [2, 1, 3]]).astype(int)] = 999.0
    assert (copy[np.array([[2, 1, 3]])] == np.array([999])).all()
    assert (copy.subs == np.concatenate((data["subs"], np.array([[2, 1, 3]])))).all()

    # Case II: Multiple entries, for multiple subs that don't exist
    (data, sptensorInstance) = sample_sptensor
    copy = ttb.sptensor.from_tensor_type(sptensorInstance)
    copy[np.array([[1, 1, 2], [2, 1, 3]]).astype(int)] = np.array([[888], [999]])
    assert (copy[np.array([[1, 1, 2], [2, 1, 3]])] == np.array([[888], [999]])).all()
    assert (
        copy.subs == np.concatenate((data["subs"], np.array([[1, 1, 2], [2, 1, 3]])))
    ).all()

    # Case II: Multiple entries, for multiple subs that exist and need to be removed
    (data, sptensorInstance) = sample_sptensor
    copy = ttb.sptensor.from_tensor_type(sptensorInstance)
    copy[np.array([[1, 1, 1], [1, 1, 3]]).astype(int)] = np.array([[0], [0]])
    assert (copy[np.array([[1, 1, 2], [2, 1, 3]])] == np.array([[0], [0]])).all()
    assert (copy.subs == np.array([[2, 2, 2], [3, 3, 3]])).all()


@pytest.mark.indevelopment
def test_sptensor_norm(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor
    assert sptensorInstance.norm() == np.linalg.norm(data["vals"])


@pytest.mark.indevelopment
def test_sptensor_allsubs(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor
    result = []
    for i in range(0, data["shape"][0]):
        for j in range(0, data["shape"][1]):
            for k in range(0, data["shape"][2]):
                result.append([i, j, k])
    assert (sptensorInstance.allsubs() == np.array(result)).all()


@pytest.mark.indevelopment
def test_sptensor_logical_not(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor
    result = []
    data_subs = data["subs"].tolist()
    for i in range(0, data["shape"][0]):
        for j in range(0, data["shape"][1]):
            for k in range(0, data["shape"][2]):
                if [i, j, k] not in data_subs:
                    result.append([i, j, k])
    notSptensorInstance = sptensorInstance.logical_not()
    assert (notSptensorInstance.vals == 1).all()
    assert (notSptensorInstance.subs == np.array(result)).all()
    assert notSptensorInstance.shape == data["shape"]


@pytest.mark.indevelopment
def test_sptensor_logical_or(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor

    # Sptensor logical or with another sptensor
    sptensorOr = sptensorInstance.logical_or(sptensorInstance)
    assert sptensorOr.shape == data["shape"]
    assert (sptensorOr.subs == data["subs"]).all()
    assert (sptensorOr.vals == np.ones((data["vals"].shape[0], 1))).all()

    # Sptensor logical or with tensor
    sptensorOr = sptensorInstance.logical_or(
        ttb.tensor.from_tensor_type(sptensorInstance)
    )
    nonZeroMatrix = np.zeros(data["shape"])
    nonZeroMatrix[tuple(data["subs"].transpose())] = 1
    assert (sptensorOr.data == nonZeroMatrix).all()

    # Sptensor logical or with scalar, 0
    sptensorOr = sptensorInstance.logical_or(0)
    assert (sptensorOr.data == nonZeroMatrix).all()

    # Sptensor logical or with scalar, not 0
    sptensorOr = sptensorInstance.logical_or(1)
    assert (sptensorOr.data == np.ones(data["shape"])).all()

    # Sptensor logical or with wrong shape sptensor
    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance.logical_or(
            ttb.sptensor.from_data(data["subs"], data["vals"], (5, 5, 5))
        )
    assert "Logical Or requires tensors of the same size" in str(excinfo)

    # Sptensor logical or with not scalar or tensor
    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance.logical_or(np.ones(data["shape"]))
    assert "Sptensor Logical Or argument must be scalar or sptensor" in str(excinfo)


@pytest.mark.indevelopment
def test_sptensor__eq__(sample_sptensor):
    # TODO fix == against empty sptensor
    (data, sptensorInstance) = sample_sptensor

    eqSptensor = sptensorInstance == 0.0
    assert (eqSptensor.subs == sptensorInstance.logical_not().subs).all()

    eqSptensor = sptensorInstance == 0.5
    assert (eqSptensor.subs == data["subs"][0]).all()

    eqSptensor = sptensorInstance == sptensorInstance
    assert (
        eqSptensor.subs
        == np.vstack((sptensorInstance.logical_not().subs, data["subs"]))
    ).all()

    denseTensor = ttb.tensor.from_tensor_type(sptensorInstance)
    eqSptensor = sptensorInstance == denseTensor
    logging.debug(f"\ndenseTensor = {denseTensor}")
    logging.debug(f"\nsptensorInstance = {sptensorInstance}")
    logging.debug(f"\ntype(eqSptensor.subs) = \n{type(eqSptensor.subs)}")
    for i in range(eqSptensor.subs.shape[0]):
        logging.debug(f"{i}\t{eqSptensor.subs[i,:]}")
    logging.debug(f"\neqSptensor.subs = \n{eqSptensor.subs}")
    logging.debug(f"\neqSptensor.subs.shape[0] = {eqSptensor.subs.shape[0]}")
    logging.debug(f"\nsptensorInstance.shape = {sptensorInstance.shape}")
    logging.debug(
        f"\nnp.prod(sptensorInstance.shape) = {np.prod(sptensorInstance.shape)}"
    )
    assert eqSptensor.subs.shape[0] == np.prod(sptensorInstance.shape)

    denseTensor = ttb.tensor.from_data(np.ones((5, 5, 5)))
    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance == denseTensor
    assert "Size mismatch in sptensor equality" in str(excinfo)

    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance == np.ones((4, 4, 4))
    assert "Sptensor == argument must be scalar or sptensor" in str(excinfo)


@pytest.mark.indevelopment
def test_sptensor__ne__(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor

    X = ttb.sptensor.from_data(np.array([[0, 0], [1, 1]]), np.array([[2], [2]]), (2, 2))
    Y = ttb.sptensor.from_data(np.array([[0, 0], [0, 1]]), np.array([[3], [3]]), (2, 2))
    assert (X != Y).isequal(
        ttb.sptensor.from_data(
            np.array([[1, 1], [0, 1], [0, 0]]),
            np.array([True, True, True])[:, None],
            (2, 2),
        )
    )

    eqSptensor = sptensorInstance != 0.0
    assert (eqSptensor.vals == 0 * sptensorInstance.vals + 1).all()

    eqSptensor = sptensorInstance != 0.5
    assert (
        eqSptensor.subs
        == np.vstack((data["subs"][1:], sptensorInstance.logical_not().subs))
    ).all()

    eqSptensor = sptensorInstance != sptensorInstance
    assert eqSptensor.vals.size == 0

    denseTensor = ttb.tensor.from_tensor_type(sptensorInstance)
    eqSptensor = sptensorInstance != denseTensor
    assert eqSptensor.vals.size == 0

    denseTensor = ttb.tensor.from_tensor_type(sptensorInstance)
    denseTensor[1, 1, 2] = 1
    eqSptensor = sptensorInstance != denseTensor
    assert (eqSptensor.subs == np.array([1, 1, 2])).all()

    denseTensor = ttb.tensor.from_data(np.ones((5, 5, 5)))
    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance != denseTensor
    assert "Size mismatch" in str(excinfo)

    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance != np.ones((4, 4, 4))
    assert "The arguments must be two sptensors or an sptensor and a scalar." in str(
        excinfo
    )


def test_sptensor__end(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor
    assert sptensorInstance.end() == np.prod(data["shape"]) - 1
    assert sptensorInstance.end(k=0) == data["shape"][0] - 1


def test_sptensor__find(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor
    subs, vals = sptensorInstance.find()
    assert (subs == data["subs"]).all()
    assert (vals == data["vals"]).all()


def test_sptensor__sub__(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor

    # Sptensor - sptensor
    subSptensor = sptensorInstance - sptensorInstance
    assert subSptensor.vals.size == 0

    # Sptensor - sptensor of wrong size
    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance - ttb.sptensor.from_data(np.array([]), np.array([]), (6, 6, 6))
    assert "Must be two sparse tensors of the same shape" in str(excinfo)

    # Sptensor - tensor
    subSptensor = sptensorInstance - ttb.tensor.from_tensor_type(sptensorInstance)
    assert (subSptensor.data == np.zeros(data["shape"])).all()

    # Sptensor - scalar
    subSptensor = sptensorInstance - 0
    assert (
        subSptensor.data == ttb.tensor.from_tensor_type(sptensorInstance).data
    ).all()


def test_sptensor__add__(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor

    # Sptensor + sptensor
    subSptensor = sptensorInstance + sptensorInstance
    assert (subSptensor.vals == 2 * data["vals"]).all()

    # Sptensor + sptensor of wrong size
    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance + ttb.sptensor.from_data(np.array([]), np.array([]), (6, 6, 6))
    assert "Must be two sparse tensors of the same shape" in str(excinfo)

    # Sptensor + tensor
    subSptensor = sptensorInstance + ttb.tensor.from_tensor_type(sptensorInstance)
    results = ttb.tensor.from_tensor_type(sptensorInstance).data * 2
    assert (subSptensor.data == results).all()

    # Sptensor + scalar
    subSptensor = sptensorInstance + 0
    assert (
        subSptensor.data == ttb.tensor.from_tensor_type(sptensorInstance).data
    ).all()


def test_sptensor_isequal(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor

    # Wrong shape sptensor
    assert not sptensorInstance.isequal(
        ttb.sptensor.from_data(np.array([]), np.array([]), (6, 6, 6))
    )

    # Sptensor is equal to itself
    assert sptensorInstance.isequal(sptensorInstance)

    # Sptensor equality with tensor
    assert sptensorInstance.isequal(ttb.tensor.from_tensor_type(sptensorInstance))

    # Sptensor equality with not sptensor or tensor
    assert not sptensorInstance.isequal(np.ones(data["shape"]))


def test_sptensor__pos__(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor
    sptensorInstance2 = +sptensorInstance

    assert sptensorInstance.isequal(sptensorInstance2)


def test_sptensor__neg__(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor
    sptensorInstance2 = -sptensorInstance
    sptensorInstance3 = -sptensorInstance2

    assert not sptensorInstance.isequal(sptensorInstance2)
    assert sptensorInstance.isequal(sptensorInstance3)


def test_sptensor__mul__(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor

    # Test mul with int
    assert ((sptensorInstance * 2).vals == 2 * data["vals"]).all()
    # Test mul with float
    assert ((sptensorInstance * 2.0).vals == 2 * data["vals"]).all()
    # Test mul with sptensor
    assert (
        (sptensorInstance * sptensorInstance).vals == data["vals"] * data["vals"]
    ).all()
    # Test mul with tensor
    assert (
        (sptensorInstance * ttb.tensor.from_tensor_type(sptensorInstance)).vals
        == data["vals"] * data["vals"]
    ).all()
    # Test mul with ktensor
    weights = np.array([1.0, 2.0])
    fm0 = np.array([[1.0, 2.0], [3.0, 4.0]])
    fm1 = np.array([[5.0, 6.0], [7.0, 8.0]])
    factor_matrices = [fm0, fm1]
    K = ttb.ktensor.from_data(weights, factor_matrices)
    subs = np.array([[0, 0], [0, 1], [1, 1]])
    vals = np.array([[0.5], [1.0], [1.5]])
    shape = (2, 2)
    S = ttb.sptensor().from_data(subs, vals, shape)
    assert (S * K).full().isequal(K.full() * S)

    # Test mul with wrong shape
    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance * ttb.sptensor.from_data(np.array([]), np.array([]), (5, 5, 5))
    assert "Sptensor Multiply requires two tensors of the same shape." in str(excinfo)

    # Test mul with wrong type
    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance * "string"
    assert "Sptensor cannot be multiplied by that type of object" in str(excinfo)


def test_sptensor__rmul__(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor

    # Test mul with int
    assert ((2 * sptensorInstance).vals == 2 * data["vals"]).all()
    # Test mul with float
    assert ((2.0 * sptensorInstance).vals == 2 * data["vals"]).all()
    # Test mul with ktensor
    weights = np.array([1.0, 2.0])
    fm0 = np.array([[1.0, 2.0], [3.0, 4.0]])
    fm1 = np.array([[5.0, 6.0], [7.0, 8.0]])
    factor_matrices = [fm0, fm1]
    K = ttb.ktensor.from_data(weights, factor_matrices)
    subs = np.array([[0, 0], [0, 1], [1, 1]])
    vals = np.array([[0.5], [1.0], [1.5]])
    shape = (2, 2)
    S = ttb.sptensor().from_data(subs, vals, shape)
    assert (S * K).full().isequal(S * K.full())

    # Test mul with wrong type
    with pytest.raises(AssertionError) as excinfo:
        "string" * sptensorInstance
    assert "This object cannot be multiplied by sptensor" in str(excinfo)


def test_sptensor_ones(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor

    assert (sptensorInstance.ones().vals == (0.0 * data["vals"] + 1)).all()


@pytest.mark.indevelopment
def test_sptensor_double(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor
    denseData = np.zeros(sptensorInstance.shape)
    actualIdx = tuple(data["subs"].transpose())
    denseData[actualIdx] = data["vals"].transpose()[0]

    assert (sptensorInstance.double() == denseData).all()
    assert sptensorInstance.double().shape == data["shape"]


@pytest.mark.indevelopment
def test_sptensor__le__(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor

    # Test comparison to negative scalar
    assert ((-sptensorInstance <= -0.1).vals == 0 * data["vals"] + 1).all()
    # Test comparison to positive scalar
    assert ((sptensorInstance <= 0.1).vals == sptensorInstance.logical_not().vals).all()

    # Test comparison to tensor
    assert (
        (sptensorInstance <= sptensorInstance.full()).vals
        == np.ones((np.prod(data["shape"]), 1))
    ).all()

    # Test comparison to sptensor
    assert (
        (sptensorInstance <= sptensorInstance).vals
        == np.ones((np.prod(data["shape"]), 1))
    ).all()

    # Test comparison of empty tensor with sptensor, both ways
    emptySptensor = ttb.sptensor.from_data(np.array([]), np.array([]), data["shape"])
    assert (
        (emptySptensor <= sptensorInstance).vals == np.ones((np.prod(data["shape"]), 1))
    ).all()
    assert (
        (sptensorInstance <= emptySptensor).vals == sptensorInstance.logical_not().vals
    ).all()

    # Test comparison with different size
    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance <= ttb.sptensor.from_data(
            np.array([]), np.array([]), (5, 5, 5)
        )
    assert "Size mismatch" in str(excinfo)

    # Test comparison with incorrect type
    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance <= "string"
    assert "Cannot compare sptensor with that type" in str(excinfo)


@pytest.mark.indevelopment
def test_sptensor__ge__(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor

    # Test comparison to positive scalar
    assert ((sptensorInstance >= 0.1).vals == 0 * data["vals"] + 1).all()
    # Test comparison to negative scalar
    assert (
        (sptensorInstance >= -0.1).vals == np.ones((np.prod(data["shape"]), 1))
    ).all()

    # Test comparison to tensor
    assert (
        (sptensorInstance >= sptensorInstance.full()).vals
        == np.ones((np.prod(data["shape"]), 1))
    ).all()

    # Test comparison to sptensor
    assert (
        (sptensorInstance >= sptensorInstance).vals
        == np.ones((np.prod(data["shape"]), 1))
    ).all()

    # Test comparison with different size
    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance >= ttb.sptensor.from_data(
            np.array([]), np.array([]), (5, 5, 5)
        )
    assert "Size mismatch" in str(excinfo)

    # Test comparison with incorrect type
    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance >= "string"
    assert "Cannot compare sptensor with that type" in str(excinfo)


@pytest.mark.indevelopment
def test_sptensor__gt__(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor

    # Test comparison to positive scalar
    assert ((sptensorInstance > 0.1).vals == 0 * data["vals"] + 1).all()
    # Test comparison to negative scalar
    assert (
        (sptensorInstance > -0.1).vals == np.ones((np.prod(data["shape"]), 1))
    ).all()

    # Test comparison to tensor
    assert (sptensorInstance > sptensorInstance.full()).vals.size == 0

    # Test comparison to tensor of different sparsity patter
    denseTensor = sptensorInstance.full()
    denseTensor[1, 1, 2] = -1
    assert ((sptensorInstance > denseTensor).subs == np.array([1, 1, 2])).all()

    # Test comparison to sptensor
    assert (sptensorInstance > sptensorInstance).vals.size == 0

    # Test comparison with different size
    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance > ttb.sptensor.from_data(np.array([]), np.array([]), (5, 5, 5))
    assert "Size mismatch" in str(excinfo)

    # Test comparison with incorrect type
    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance > "string"
    assert "Cannot compare sptensor with that type" in str(excinfo)


@pytest.mark.indevelopment
def test_sptensor__lt__(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor

    # Test comparison to negative scalar
    assert ((-sptensorInstance < -0.1).vals == 0 * data["vals"] + 1).all()
    # Test comparison to positive scalar
    assert ((sptensorInstance < 0.1).vals == sptensorInstance.logical_not().vals).all()

    # Test comparison to tensor
    assert (sptensorInstance < sptensorInstance.full()).vals.size == 0

    # Test comparison to sptensor
    assert (sptensorInstance < sptensorInstance).vals.size == 0

    # Test comparison of empty tensor with sptensor, both ways
    emptySptensor = ttb.sptensor.from_data(np.array([]), np.array([]), data["shape"])
    assert ((emptySptensor < sptensorInstance).subs == data["subs"]).all()
    assert (sptensorInstance < emptySptensor).vals.size == 0

    # Test comparison with different size
    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance < ttb.sptensor.from_data(np.array([]), np.array([]), (5, 5, 5))
    assert "Size mismatch" in str(excinfo)

    # Test comparison with incorrect type
    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance < "string"
    assert "Cannot compare sptensor with that type" in str(excinfo)


@pytest.mark.indevelopment
def test_sptensor_innerprod(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor

    # Empty sptensor innerproduct
    emptySptensor = ttb.sptensor.from_data(np.array([]), np.array([]), data["shape"])
    assert sptensorInstance.innerprod(emptySptensor) == 0
    assert emptySptensor.innerprod(sptensorInstance) == 0

    # Sptensor innerproduct
    assert sptensorInstance.innerprod(sptensorInstance) == data["vals"].transpose().dot(
        data["vals"]
    )

    # Sptensor innerproduct, other has more elements
    sptensorCopy = ttb.sptensor.from_tensor_type(sptensorInstance)
    sptensorCopy[0, 0, 0] = 1
    assert sptensorInstance.innerprod(sptensorCopy) == data["vals"].transpose().dot(
        data["vals"]
    )

    # Wrong shape sptensor
    emptySptensor = ttb.sptensor.from_data(np.array([]), np.array([]), (1, 1))
    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance.innerprod(emptySptensor)
    assert "Sptensors must be same shape for innerproduct" in str(excinfo)

    # Tensor innerproduct
    assert sptensorInstance.innerprod(
        ttb.tensor.from_tensor_type(sptensorInstance)
    ) == data["vals"].transpose().dot(data["vals"])

    # Wrong shape tensor
    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance.innerprod(ttb.tensor.from_data(np.array([1])))
    assert "Sptensor and tensor must be same shape for innerproduct" in str(excinfo)

    # Wrong type for innerprod
    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance.innerprod(5)
    assert f"Inner product between sptensor and {type(5)} not supported" in str(excinfo)


@pytest.mark.indevelopment
def test_sptensor_logical_xor(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor
    nonZeroMatrix = np.zeros(data["shape"])
    nonZeroMatrix[tuple(data["subs"].transpose())] = 1

    # Sptensor logical xor with scalar, 0
    sptensorXor = sptensorInstance.logical_xor(0)
    assert (sptensorXor.data == nonZeroMatrix).all()

    # Sptensor logical xor with scalar, not 0
    sptensorXor = sptensorInstance.logical_xor(1)
    assert (sptensorXor.data == sptensorInstance.logical_not().full().data).all()

    # Sptensor logical xor with another sptensor
    sptensorXor = sptensorInstance.logical_xor(sptensorInstance)
    assert sptensorXor.shape == data["shape"]
    assert sptensorXor.vals.size == 0

    # Sptensor logical xor with tensor
    sptensorXor = sptensorInstance.logical_xor(
        ttb.tensor.from_tensor_type(sptensorInstance)
    )
    assert (sptensorXor.data == np.zeros(data["shape"], dtype=bool)).all()

    # Sptensor logical xor with wrong shape sptensor
    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance.logical_xor(
            ttb.sptensor.from_data(data["subs"], data["vals"], (5, 5, 5))
        )
    assert "Logical XOR requires tensors of the same size" in str(excinfo)

    # Sptensor logical xor with not scalar or tensor
    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance.logical_xor(np.ones(data["shape"]))
    assert "The argument must be an sptensor, tensor or scalar" in str(excinfo)


@pytest.mark.indevelopment
def test_sptensor_squeeze(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor

    # No singleton dimensions
    assert (sptensorInstance.squeeze().vals == data["vals"]).all()
    assert (sptensorInstance.squeeze().subs == data["subs"]).all()

    # All singleton dimensions
    assert (
        ttb.sptensor.from_data(
            np.array([[0, 0, 0]]), np.array([4]), (1, 1, 1)
        ).squeeze()
        == 4
    )

    # A singleton dimension
    assert np.array_equal(
        ttb.sptensor.from_data(np.array([[0, 0, 0]]), np.array([4]), (2, 2, 1))
        .squeeze()
        .subs,
        np.array([[0, 0]]),
    )
    assert (
        ttb.sptensor.from_data(np.array([[0, 0, 0]]), np.array([4]), (2, 2, 1))
        .squeeze()
        .vals
        == np.array([4])
    ).all()

    # Singleton dimension with empty sptensor
    assert ttb.sptensor.from_data(
        np.array([]), np.array([]), (2, 2, 1)
    ).squeeze().shape == (2, 2)


@pytest.mark.indevelopment
def test_sptensor_scale(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor

    # Scale with np array
    assert (
        sptensorInstance.scale(np.array([4, 4, 4, 4]), 1).vals == 4 * data["vals"]
    ).all()

    # Scale with sptensor
    assert (
        sptensorInstance.scale(sptensorInstance, np.arange(0, 3)).vals
        == data["vals"] ** 2
    ).all()

    # Scale with tensor
    assert (
        sptensorInstance.scale(
            ttb.tensor.from_tensor_type(sptensorInstance), np.arange(0, 3)
        ).vals
        == data["vals"] ** 2
    ).all()

    # Incorrect shape np array, sptensor and tensor
    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance.scale(np.array([4, 4, 4, 4, 4]), 1)
    assert "Size mismatch in scale" in str(excinfo)
    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance.scale(
            ttb.sptensor.from_data(np.array([]), np.array([]), (1, 1, 1, 1, 1)),
            np.arange(0, 3),
        )
    assert "Size mismatch in scale" in str(excinfo)
    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance.scale(
            ttb.tensor.from_data(np.ones((1, 1, 1, 1, 1))), np.arange(0, 3)
        )
    assert "Size mismatch in scale" in str(excinfo)

    # Scale with non nparray, sptensor or tensor
    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance.scale(1, 1)
    assert "Invalid scaling factor" in str(excinfo)


@pytest.mark.indevelopment
def test_sptensor_reshape(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor

    # Standard reshape
    assert sptensorInstance.reshape((16, 4, 1)).shape == (16, 4, 1)

    # Reshape first and last modes, leave middle alone
    assert sptensorInstance.reshape((16, 1), np.array([0, 2])).shape == (4, 16, 1)

    # Reshape empty sptensor
    assert ttb.sptensor.from_data(np.array([]), np.array([]), (4, 4, 4)).reshape(
        (16, 4, 1)
    ).shape == (16, 4, 1)

    # Improper reshape
    with pytest.raises(AssertionError) as excinfo:
        assert sptensorInstance.reshape((16, 1), np.array([0])).shape == (4, 16, 1)
    assert "Reshape must maintain tensor size" in str(excinfo)


@pytest.mark.indevelopment
def test_sptensor_mask(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor

    # Mask captures all non-zero entries
    assert (sptensorInstance.mask(sptensorInstance) == data["vals"]).all()

    # Mask too large
    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance.mask(
            ttb.sptensor.from_data(np.array([]), np.array([]), (3, 3, 5))
        )
    assert "Mask cannot be bigger than the data tensor" in str(excinfo)


@pytest.mark.indevelopment
def test_sptensor_permute(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor

    emptySptensor = ttb.sptensor.from_data(np.array([]), np.array([]), (4, 4, 4))
    assert emptySptensor.permute(np.arange(0, 3)).isequal(emptySptensor)

    assert sptensorInstance.permute(np.arange(0, 3)).isequal(sptensorInstance)

    # Permute with too many dimensions
    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance.permute(np.arange(0, 4))
    assert "Invalid permutation order" in str(excinfo)

    # Permute that doesn't consider each dimension
    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance.permute(np.array([0, 0, 0]))
    assert "Invalid permutation order" in str(excinfo)


@pytest.mark.indevelopment
def test_sptensor__rtruediv__(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor

    # Scalar / Spensor yields tensor, only resolves when left object doesn't have appropriate __truediv__
    # We ignore the divide by zero errors because np.inf/np.nan is an appropriate representation
    with np.errstate(divide="ignore", invalid="ignore"):
        assert ((2 / sptensorInstance).data == (2 / sptensorInstance.full().data)).all()

    # Tensor / Spensor yields tensor should be calling tensor.__truediv__
    # We ignore the divide by zero errors because np.inf/np.nan is an appropriate representation
    with np.errstate(divide="ignore", invalid="ignore"):
        np.testing.assert_array_equal(
            (sptensorInstance.full() / sptensorInstance).data,
            (sptensorInstance.full().data / sptensorInstance.full().data),
        )

    # Non-Scalar / Spensor yields tensor, only resolves when left object doesn't have appropriate __truediv__
    with pytest.raises(AssertionError) as excinfo:
        (("string" / sptensorInstance).data == (2 / sptensorInstance.full().data))
    assert "Dividing that object by an sptensor is not supported" in str(excinfo)


@pytest.mark.indevelopment
def test_sptensor__truediv__(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor

    emptySptensor = ttb.sptensor.from_data(np.array([]), np.array([]), (4, 4, 4))

    # Sptensor/ non-zero scalar
    assert ((sptensorInstance / 5).vals == data["vals"] / 5).all()

    # Sptensor/zero scalar
    np.testing.assert_array_equal(
        (sptensorInstance / 0).vals,
        np.vstack(
            (
                np.inf * np.ones((data["subs"].shape[0], 1)),
                np.nan * np.ones((np.prod(data["shape"]) - data["subs"].shape[0], 1)),
            )
        ),
    )

    # Sptensor/sptensor
    np.testing.assert_array_equal(
        (sptensorInstance / sptensorInstance).vals,
        np.vstack(
            (
                data["vals"] / data["vals"],
                np.nan * np.ones((np.prod(data["shape"]) - data["subs"].shape[0], 1)),
            )
        ),
    )

    # Sptensor/ empty tensor
    np.testing.assert_array_equal(
        (sptensorInstance / emptySptensor).vals,
        np.nan * np.ones((np.prod(data["shape"]), 1)),
    )

    # empty tensor/Sptensor
    np.testing.assert_array_equal(
        (emptySptensor / sptensorInstance).vals,
        np.vstack(
            (
                np.zeros((data["subs"].shape[0], 1)),
                np.nan * np.ones((np.prod(data["shape"]) - data["subs"].shape[0], 1)),
            )
        ),
    )

    # Sptensor/tensor
    assert (
        (sptensorInstance / sptensorInstance.full()).vals == data["vals"] / data["vals"]
    ).all()

    # Sptensor/ktensor
    weights = np.array([1.0, 2.0])
    fm0 = np.array([[1.0, 2.0], [3.0, 4.0]])
    fm1 = np.array([[5.0, 6.0], [7.0, 8.0]])
    factor_matrices = [fm0, fm1]
    K = ttb.ktensor.from_data(weights, factor_matrices)
    subs = np.array([[0, 0], [0, 1], [1, 1]])
    vals = np.array([[0.5], [1.0], [1.5]])
    shape = (2, 2)
    S = ttb.sptensor().from_data(subs, vals, shape)
    assert (S / K).full().isequal(S.full() / K.full())

    # Sptensor/ invalid
    with pytest.raises(AssertionError) as excinfo:
        (sptensorInstance / "string")
    assert "Invalid arguments for sptensor division" in str(excinfo)

    with pytest.raises(AssertionError) as excinfo:
        emptySptensor.shape = (5, 5, 5)
        (sptensorInstance / emptySptensor)
    assert "Sptensor division requires tensors of the same shape" in str(excinfo)


@pytest.mark.indevelopment
def test_sptensor_collapse(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor
    emptySptensor = ttb.sptensor.from_data(np.array([]), np.array([]), (4, 4, 4))

    # Test with no arguments
    assert sptensorInstance.collapse() == np.sum(data["vals"])

    # Test with custom function
    assert sptensorInstance.collapse(fun=sum) == np.sum(data["vals"])

    # Test partial collapse, output vector
    assert (
        sptensorInstance.collapse(dims=np.array([0, 1])) == np.array([0, 0.5, 2.5, 5])
    ).all()
    assert (
        emptySptensor.collapse(dims=np.array([0, 1])) == np.array([0, 0, 0, 0])
    ).all()

    # Test partial collapse, output sptensor
    collapseSptensor = sptensorInstance.collapse(dims=np.array([0]))
    assert (collapseSptensor.vals == data["vals"]).all()
    assert collapseSptensor.shape == (4, 4)
    assert (collapseSptensor.subs == data["subs"][:, 1:3]).all()
    emptySptensorSmaller = ttb.sptensor.from_tensor_type(emptySptensor)
    emptySptensorSmaller.shape = (4, 4)
    assert emptySptensor.collapse(dims=np.array([0])).isequal(emptySptensorSmaller)


@pytest.mark.indevelopment
def test_sptensor_contract(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor
    emptySptensor = ttb.sptensor.from_data(np.array([]), np.array([]), (5, 4, 4))

    with pytest.raises(AssertionError) as excinfo:
        emptySptensor.contract(0, 1)
    assert "Must contract along equally sized dimensions" in str(excinfo)

    with pytest.raises(AssertionError) as excinfo:
        emptySptensor.contract(0, 0)
    assert "Must contract along two different dimensions" in str(excinfo)

    contractableSptensor = ttb.sptensor.from_tensor_type(sptensorInstance)
    contractableSptensor = contractableSptensor.collapse(np.array([0]))
    assert contractableSptensor.contract(0, 1) == 6.5

    contractableSptensor = ttb.sptensor.from_tensor_type(sptensorInstance)
    assert (
        contractableSptensor.contract(0, 1).data == np.array([0, 0.5, 2.5, 5])
    ).all()

    contractableSptensor = ttb.sptensor.from_tensor_type(sptensorInstance)
    contractableSptensor[3, 3, 3, 3] = 1
    assert contractableSptensor.contract(0, 1).shape == (4, 4)


@pytest.mark.indevelopment
def test_sptensor_elemfun(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor

    def plus1(y):
        return y + 1

    assert (sptensorInstance.elemfun(plus1).vals == 1 + data["vals"]).all()
    assert (sptensorInstance.elemfun(plus1).subs == data["subs"]).all()

    emptySptensor = ttb.sptensor.from_data(np.array([]), np.array([]), (4, 4, 4))
    assert emptySptensor.elemfun(plus1).vals.size == 0


@pytest.mark.indevelopment
def test_sptensor_spmatrix(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor

    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance.spmatrix()
    assert "Sparse tensor must be two dimensional" in str(excinfo)

    # Test empty sptensor to empty sparse matrix
    emptySptensor = ttb.sptensor.from_data(np.array([]), np.array([]), (4, 4))
    a = emptySptensor.spmatrix()

    assert a.data.size == 0
    assert a.shape == emptySptensor.shape

    NonEmptySptensor = ttb.sptensor.from_data(
        np.array([[0, 0]]), np.array([[1]]), (4, 4)
    )
    fullData = np.zeros(NonEmptySptensor.shape)
    fullData[0, 0] = 1
    b = NonEmptySptensor.spmatrix()
    assert (b.toarray() == fullData).all()

    NonEmptySptensor = ttb.sptensor.from_data(
        np.array([[0, 1], [1, 0]]), np.array([[1], [2]]), (4, 4)
    )
    fullData = np.zeros(NonEmptySptensor.shape)
    fullData[0, 1] = 1
    fullData[1, 0] = 2
    b = NonEmptySptensor.spmatrix()
    assert (b.toarray() == fullData).all()

    NonEmptySptensor = ttb.sptensor.from_data(
        np.array([[0, 1], [2, 3]]), np.array([[1], [2]]), (4, 4)
    )
    fullData = np.zeros(NonEmptySptensor.shape)
    fullData[0, 1] = 1
    fullData[2, 3] = 2
    b = NonEmptySptensor.spmatrix()
    assert (b.toarray() == fullData).all()


@pytest.mark.indevelopment
def test_sptensor_ttv(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor

    # Returns single value
    onesSptensor = ttb.sptensor.from_tensor_type(ttb.tensor.from_data(np.ones((4, 4))))
    vector = np.array([1, 1, 1, 1])
    assert onesSptensor.ttv(np.array([vector, vector])) == 16

    # Wrong shape vector
    with pytest.raises(AssertionError) as excinfo:
        onesSptensor.ttv([vector, np.array([1, 2])])
    assert "Multiplicand is wrong size" in str(excinfo)

    # Returns vector shaped object
    emptySptensor = ttb.sptensor.from_data(np.array([]), np.array([]), (4, 4))
    onesSptensor = ttb.sptensor.from_tensor_type(ttb.tensor.from_data(np.ones((4, 4))))

    assert emptySptensor.ttv(vector, 0).isequal(
        ttb.sptensor.from_data(np.array([]), np.array([]), (4,))
    )
    assert onesSptensor.ttv(vector, 0).isequal(
        ttb.tensor.from_data(np.array([4, 4, 4, 4]))
    )
    emptySptensor[0, 0] = 1
    assert (emptySptensor.ttv(vector, 0).full().data == np.array([1, 0, 0, 0])).all()

    # Returns tensor shaped object
    emptySptensor = ttb.sptensor.from_data(np.array([]), np.array([]), (4, 4, 4))
    onesSptensor = ttb.sptensor.from_tensor_type(
        ttb.tensor.from_data(np.ones((4, 4, 4)))
    )
    assert emptySptensor.ttv(vector, 0).isequal(
        ttb.sptensor.from_data(np.array([]), np.array([]), (4, 4))
    )
    assert onesSptensor.ttv(vector, 0).isequal(
        ttb.tensor.from_data(4 * np.ones((4, 4)))
    )


@pytest.mark.indevelopment
def test_sptensor_mttkrp(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor

    # MTTKRP with array of matrices
    # Note this is more of a regression test against the output of MATLAB TTB
    matrix = np.ones((4, 4))
    assert (
        sptensorInstance.mttkrp(np.array([matrix, matrix, matrix]), 0)
        == np.array(
            [[0, 0, 0, 0], [2, 2, 2, 2], [2.5, 2.5, 2.5, 2.5], [3.5, 3.5, 3.5, 3.5]]
        )
    ).all()
    assert (
        sptensorInstance.mttkrp(np.array([matrix, matrix, matrix]), 1)
        == sptensorInstance.mttkrp(np.array([matrix, matrix, matrix]), 0)
    ).all()
    assert (
        sptensorInstance.mttkrp(np.array([matrix, matrix, matrix]), 2)
        == np.array(
            [[0, 0, 0, 0], [0.5, 0.5, 0.5, 0.5], [2.5, 2.5, 2.5, 2.5], [5, 5, 5, 5]]
        )
    ).all()

    # MTTKRP with factor matrices from ktensor
    K = ttb.ktensor.from_factor_matrices([matrix, matrix, matrix])
    assert (
        sptensorInstance.mttkrp(np.array([matrix, matrix, matrix]), 0)
        == sptensorInstance.mttkrp(K, 0)
    ).all()
    assert (
        sptensorInstance.mttkrp(np.array([matrix, matrix, matrix]), 1)
        == sptensorInstance.mttkrp(K, 1)
    ).all()
    assert (
        sptensorInstance.mttkrp(np.array([matrix, matrix, matrix]), 2)
        == sptensorInstance.mttkrp(K, 2)
    ).all()

    # Wrong length input
    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance.mttkrp(np.array([matrix, matrix, matrix, matrix]), 0)
    assert "List is the wrong length" in str(excinfo)

    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance.mttkrp("string", 0)
    assert "Second argument must be ktensor or array" in str(excinfo)


@pytest.mark.indevelopment
def test_sptensor_nvecs(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor

    # Test for one eigenvector
    assert np.allclose((sptensorInstance.nvecs(1, 1)), np.array([0, 0, 0, 1])[:, None])
    assert np.allclose(
        (sptensorInstance.nvecs(1, 2)),
        np.array([[0, 0, 0, 1], [0, 0, 1, 0]]).transpose(),
    )

    # Test for r >= N-1, requires cast to dense
    with pytest.warns(Warning) as record:
        ans = np.zeros((4, 3))
        ans[3, 0] = 1
        ans[2, 1] = 1
        ans[1, 2] = 1
        assert np.allclose((sptensorInstance.nvecs(1, 3)), ans)
    assert (
        "Greater than or equal to sptensor.shape[n] - 1 eigenvectors requires cast to dense to solve"
        in str(record[0].message)
    )

    # Negative test, check for only singleton dims
    with pytest.raises(ValueError):
        single_val_sptensor = ttb.sptensor.from_data(
            np.array([[0, 0]]), np.array([1]), shape=(1, 1)
        )
        single_val_sptensor.nvecs(0, 0)


@pytest.mark.indevelopment
def test_sptensor_ttm(sample_sptensor):
    (data, sptensorInstance) = sample_sptensor
    result = np.zeros((4, 4, 4))
    result[:, 1, 1] = 0.5
    result[:, 1, 3] = 1.5
    result[:, 2, 2] = 2.5
    result[:, 3, 3] = 3.5
    result = ttb.tensor.from_data(result)
    result = ttb.sptensor.from_tensor_type(result)
    assert sptensorInstance.ttm(sparse.coo_matrix(np.ones((4, 4))), dims=0).isequal(
        result
    )
    assert sptensorInstance.ttm(
        sparse.coo_matrix(np.ones((4, 4))), dims=0, transpose=True
    ).isequal(result)

    # This is a multiway multiplication yielding a sparse tensor, yielding a dense tensor relies on tensor.ttm
    matrix = sparse.coo_matrix(np.eye(4))
    list_of_matrices = [matrix, matrix, matrix]
    assert sptensorInstance.ttm(list_of_matrices, dims=[0, 1, 2]).isequal(
        sptensorInstance
    )

    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance.ttm(sparse.coo_matrix(np.ones((5, 5))), dims=0)
    assert "Matrix shape doesn't match tensor shape" in str(excinfo)

    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance.ttm(np.array([1, 2, 3, 4]), dims=0)
    assert "Sptensor.ttm: second argument must be a matrix" in str(excinfo)

    with pytest.raises(AssertionError) as excinfo:
        sptensorInstance.ttm(sparse.coo_matrix(np.ones((4, 4))), dims=4)
    assert "dims must contain values in [0,self.dims)" in str(excinfo)

    sptensorInstance[0, :, :] = 1
    sptensorInstance[3, :, :] = 1
    result = np.zeros((4, 4, 4))
    result[0, :, :] = 4.0
    result[3, :, :] = 4.0
    result[1, :, 1] = 0.5
    result[1, :, 3] = 1.5
    result[2, :, 2] = 2.5

    # TODO: Ensure mode mappings are consistent between matlab and numpy
    # MATLAB is opposite orientation so the mapping from matlab to numpy is
    # {3:0, 2:2, 1:1}
    assert sptensorInstance.ttm(sparse.coo_matrix(np.ones((4, 4))), dims=1).isequal(
        ttb.tensor.from_data(result)
    )

    result = 2 * np.ones((4, 4, 4))
    result[:, 1, 1] = 2.5
    result[:, 1, 3] = 3.5
    result[:, 2, 2] = 4.5
    assert sptensorInstance.ttm(sparse.coo_matrix(np.ones((4, 4))), dims=0).isequal(
        ttb.tensor.from_data(result)
    )

    result = np.zeros((4, 4, 4))
    result[0, :, :] = 4.0
    result[3, :, :] = 4.0
    result[1, 1, :] = 2
    result[2, 2, :] = 2.5
    assert sptensorInstance.ttm(sparse.coo_matrix(np.ones((4, 4))), dims=2).isequal(
        ttb.tensor.from_data(result)
    )

    # Confirm reshape for non-square matrix
    assert sptensorInstance.ttm(sparse.coo_matrix(np.ones((1, 4))), dims=2).shape == (
        4,
        4,
        1,
    )


@pytest.mark.indevelopment
def test_sptensor_to_sparse_matrix():
    subs = np.array([[1, 1, 1], [1, 1, 3], [2, 2, 2], [3, 3, 3]])
    vals = np.array([[0.5], [1.5], [2.5], [3.5]])
    shape = (4, 4, 4)
    mode0 = sparse.coo_matrix(([0.5, 1.5, 2.5, 3.5], ([5, 13, 10, 15], [1, 1, 2, 3])))
    mode1 = sparse.coo_matrix(([0.5, 1.5, 2.5, 3.5], ([5, 13, 10, 15], [1, 1, 2, 3])))
    mode2 = sparse.coo_matrix(([0.5, 1.5, 2.5, 3.5], ([5, 5, 10, 15], [1, 3, 2, 3])))
    Ynt = [mode0, mode1, mode2]
    sptensorInstance = ttb.sptensor().from_data(subs, vals, shape)

    for mode in range(sptensorInstance.ndims):
        Xnt = tt_to_sparse_matrix(sptensorInstance, mode, True)
        assert (Xnt != Ynt[mode]).nnz == 0
        assert Xnt.shape == Ynt[mode].shape


@pytest.mark.indevelopment
def test_sptensor_from_sparse_matrix():
    subs = np.array([[1, 1, 1], [1, 1, 3], [2, 2, 2], [3, 3, 3]])
    vals = np.array([[0.5], [1.5], [2.5], [3.5]])
    shape = (4, 4, 4)
    sptensorInstance = ttb.sptensor().from_data(subs, vals, shape)
    for mode in range(sptensorInstance.ndims):
        sptensorCopy = ttb.sptensor.from_tensor_type(sptensorInstance)
        Xnt = tt_to_sparse_matrix(sptensorCopy, mode, True)
        Ynt = tt_from_sparse_matrix(Xnt, sptensorCopy.shape, mode, 0)
        assert sptensorCopy.isequal(Ynt)

    for mode in range(sptensorInstance.ndims):
        sptensorCopy = ttb.sptensor.from_tensor_type(sptensorInstance)
        Xnt = tt_to_sparse_matrix(sptensorCopy, mode, False)
        Ynt = tt_from_sparse_matrix(Xnt, sptensorCopy.shape, mode, 1)
        assert sptensorCopy.isequal(Ynt)
