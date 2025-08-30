# Copyright 2024 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.
from __future__ import annotations

import logging
from copy import deepcopy

import numpy as np
import pytest

import pyttb as ttb
from tests.test_utils import assert_consistent_order

DEBUG_tests = False


def test_tenmat_initialization_empty():
    empty = np.array([])

    # No args
    tenmatInstance = ttb.tenmat()
    assert tenmatInstance.shape == ()
    assert tenmatInstance.tshape == ()
    assert (tenmatInstance.rindices == empty).all()
    assert (tenmatInstance.cindices == empty).all()
    assert (tenmatInstance.data == empty).all()
    assert_consistent_order(tenmatInstance, tenmatInstance.data)


def test_tenmat_initialization_from_data(
    sample_ndarray_1way, sample_ndarray_2way, sample_ndarray_4way, sample_tenmat_4way
):
    (params, tenmatInstance) = sample_tenmat_4way
    tshape = params["tshape"]
    rdims = params["rdims"]
    cdims = params["cdims"]
    params["data"]
    (_, ndarrayInstance1) = sample_ndarray_1way
    (_, ndarrayInstance2) = sample_ndarray_2way
    (_, ndarrayInstance4) = sample_ndarray_4way

    # Constructor from empty array, rdims, cdims, and tshape
    tenmatNdarraye = ttb.tenmat(np.array([[]]), np.array([]), np.array([]), ())
    assert (tenmatNdarraye.data == np.array([])).all()
    assert (tenmatNdarraye.rindices == np.array([])).all()
    assert (tenmatNdarraye.cindices == np.array([])).all()
    assert tenmatNdarraye.shape == ()
    assert tenmatNdarraye.tshape == ()
    assert_consistent_order(tenmatNdarraye, tenmatNdarraye.data)

    # Constructor from 1d array
    tenmatNdarray1 = ttb.tenmat(ndarrayInstance1, rdims, cdims, tshape)
    assert (
        tenmatNdarray1.data
        == np.reshape(
            tenmatInstance.data, (1, np.prod(tenmatNdarray1.shape)), order="F"
        )
    ).all()
    assert (tenmatNdarray1.rindices == tenmatInstance.rindices).all()
    assert (tenmatNdarray1.cindices == tenmatInstance.cindices).all()
    assert np.prod(tenmatNdarray1.shape) == np.prod(tenmatInstance.shape)
    assert tenmatNdarray1.tshape == tenmatInstance.tshape
    assert_consistent_order(tenmatNdarray1, tenmatNdarray1.data)

    # Constructor from 1d array converted to 2d row vector
    tenmatNdarray1r = ttb.tenmat(
        np.reshape(ndarrayInstance1, (1, ndarrayInstance1.shape[0]), order="F"),
        rdims,
        cdims,
        tshape,
    )
    assert tenmatNdarray1r.isequal(tenmatNdarray1)
    assert_consistent_order(tenmatNdarray1r, tenmatNdarray1r.data)

    # Constructor from 2d array
    tenmatNdarray2 = ttb.tenmat(ndarrayInstance2, rdims, cdims, tshape)
    assert tenmatNdarray2.isequal(tenmatInstance)
    assert_consistent_order(tenmatNdarray2, tenmatNdarray2.data)

    # Reference instead of copy
    tenmatNdarray2 = ttb.tenmat(ndarrayInstance2, rdims, cdims, tshape, copy=False)
    assert tenmatNdarray2.isequal(tenmatInstance)
    assert np.may_share_memory(ndarrayInstance2, tenmatNdarray2.data)

    # Exceptions

    ## data is not numpy.ndarray
    exc = "First argument must be a numeric numpy.ndarray."
    with pytest.raises(AssertionError) as excinfo:
        ttb.tenmat([7], rdims, cdims, tshape)
    assert exc in str(excinfo)

    ## data is numpy.ndarray but not numeric
    exc = "First argument must be a numeric numpy.ndarray."
    with pytest.raises(AssertionError) as excinfo:
        ttb.tenmat(ndarrayInstance2 > 0, rdims, cdims, tshape)
    assert exc in str(excinfo)

    # data is empty numpy.ndarray, but other params are not
    exc = "When data is empty, rdims, cdims, and tshape must also be empty."
    with pytest.raises(AssertionError) as excinfo:
        ttb.tenmat(np.array([]), rdims, np.array([]), ())
    assert exc in str(excinfo)
    with pytest.raises(AssertionError) as excinfo:
        ttb.tenmat(np.array([]), np.array([]), cdims, ())
    assert exc in str(excinfo)
    with pytest.raises(AssertionError) as excinfo:
        ttb.tenmat(np.array([]), np.array([]), np.array([]), tshape)
    assert exc in str(excinfo)

    ## data is 1D numpy.ndarray
    exc = "tshape must be specified when data is 1d array."
    with pytest.raises(AssertionError) as excinfo:
        ttb.tenmat(ndarrayInstance1, rdims, cdims)
    assert exc in str(excinfo)
    with pytest.raises(AssertionError) as excinfo:
        ttb.tenmat(ndarrayInstance1, rdims, cdims, None)
    assert exc in str(excinfo)

    # products of tshape and data.shape do not match
    exc = (
        "Incorrect dimensions specified: products of data.shape and tuple do not match"
    )
    with pytest.raises(AssertionError) as excinfo:
        ttb.tenmat(ndarrayInstance2, rdims, cdims, tuple(np.array(tshape) + 1))
    assert exc in str(excinfo)

    # products of tshape and data.shape do not match
    exc = "data.shape does not match shape specified by rdims, cdims, and tshape."
    D = []
    # do not span all dimensions
    D.append([np.array([0]), np.array([1])])
    ## dimension not in range
    # D.append([np.array([0]), np.array([1,2,4])])
    ## too many dimensions specified
    # D.append([np.array([0]), np.array([1,2,3,4])])
    ## duplicate dimensions
    # D.append([np.array([0,1,1]), np.array([2,3])])
    # D.append([np.array([0,1,1]), np.array([3])])
    for d in D:
        with pytest.raises(AssertionError) as excinfo:
            ttb.tenmat(ndarrayInstance2, d[0], d[1], tshape)
        assert exc in str(excinfo)

    # Passing tensor shaped data
    with pytest.raises(ValueError) as excinfo:
        ttb.tenmat(np.ones((4, 4, 4)), np.arange(3))


def test_tenmat_initialization_from_tensor_type(
    sample_tenmat_4way, sample_tensor_3way, sample_tensor_4way
):
    (_, tensorInstance) = sample_tensor_4way
    (_, tensorInstance3) = sample_tensor_3way
    (params, tenmatInstance) = sample_tenmat_4way
    tshape = params["tshape"]
    rdims = params["rdims"]
    cdims = params["cdims"]
    data = params["data"]

    # Copy Constructor
    tenmatCopy = tenmatInstance.copy()
    assert (tenmatCopy.data == data).all()
    assert (tenmatCopy.rindices == rdims).all()
    assert (tenmatCopy.cindices == cdims).all()
    assert tenmatCopy.shape == data.shape
    assert tenmatCopy.tshape == tshape

    tenmatCopy = deepcopy(tenmatInstance)
    assert (tenmatCopy.data == data).all()
    assert (tenmatCopy.rindices == rdims).all()
    assert (tenmatCopy.cindices == cdims).all()
    assert tenmatCopy.shape == data.shape
    assert tenmatCopy.tshape == tshape

    # Constructor from tensor using rdims only
    tenmatTensorRdims = tensorInstance.to_tenmat(rdims=rdims)
    assert tenmatInstance.isequal(tenmatTensorRdims)

    # Constructor from tensor using empty rdims
    tenmatTensorRdims = tensorInstance3.to_tenmat(rdims=np.array([]))
    data = np.reshape(np.arange(1, 13), (1, 12))
    assert (tenmatTensorRdims.data == data).all()

    # Constructor from tensor using cdims only
    tenmatTensorCdims = tensorInstance.to_tenmat(cdims=cdims)
    assert tenmatInstance.isequal(tenmatTensorCdims)

    # Constructor from tensor using empty cdims
    tenmatTensorCdims = tensorInstance3.to_tenmat(cdims=np.array([]))
    data = np.reshape(np.arange(1, 13), (12, 1))
    assert (tenmatTensorCdims.data == data).all()

    # Constructor from tensor using rdims and cdims
    tenmatTensorRdimsCdims = tensorInstance.to_tenmat(rdims=rdims, cdims=cdims)
    assert tenmatInstance.isequal(tenmatTensorRdimsCdims)

    # Constructor from tensor using 1D rdims and cdims_cyclic='fc' (forward cyclic)
    rdimsFC = np.array([1])
    cdimsFC = np.array([2, 3, 0])
    tshapeFC = (2, 2, 2, 2)
    shapeFC = (2, 8)
    dataFC = np.array([[1, 5, 9, 13, 2, 6, 10, 14], [3, 7, 11, 15, 4, 8, 12, 16]])
    tenmatTensorFC = tensorInstance.to_tenmat(rdims=rdimsFC, cdims_cyclic="fc")
    assert (tenmatTensorFC.rindices == rdimsFC).all()
    assert (tenmatTensorFC.cindices == cdimsFC).all()
    assert (tenmatTensorFC.data == dataFC).all()
    assert tenmatTensorFC.shape == shapeFC
    assert tenmatTensorFC.tshape == tshapeFC

    # Constructor from tensor using 1D rdims and cdims_cyclic='bc' (backward cyclic)
    rdimsBC = np.array([1])
    cdimsBC = np.array([0, 3, 2])
    tshapeBC = (2, 2, 2, 2)
    shapeBC = (2, 8)
    dataBC = np.array([[1, 2, 9, 10, 5, 6, 13, 14], [3, 4, 11, 12, 7, 8, 15, 16]])
    tenmatTensorBC = tensorInstance.to_tenmat(rdims=rdimsBC, cdims_cyclic="bc")
    assert (tenmatTensorBC.rindices == rdimsBC).all()
    assert (tenmatTensorBC.cindices == cdimsBC).all()
    assert (tenmatTensorBC.data == dataBC).all()
    assert tenmatTensorBC.shape == shapeBC
    assert tenmatTensorBC.tshape == tshapeBC

    # Reference only
    simple_tensor = ttb.tenones((2, 2))
    rdims = np.array([0])
    tenmat_reference = simple_tensor.to_tenmat(rdims=rdims, copy=False)
    assert np.may_share_memory(tenmat_reference.data, simple_tensor.data)

    # Exceptions

    # cdims_cyclic has incorrect value
    exc = 'Unrecognized value for cdims_cyclic pattern, must be "fc" or "bc".'
    with pytest.raises(AssertionError) as excinfo:
        tensorInstance.to_tenmat(rdims=rdimsBC, cdims_cyclic="c")
    assert exc in str(excinfo)

    # rdims and cdims cannot both be None
    exc = "Either rdims or cdims or both must be specified."
    with pytest.raises(AssertionError) as excinfo:
        tensorInstance.to_tenmat(rdims=None, cdims=None)
    assert exc in str(excinfo)

    # rdims must be valid dimensions
    exc = "Values in rdims must be in [0, source.ndims]."
    with pytest.raises(AssertionError) as excinfo:
        tensorInstance.to_tenmat(rdims=np.array([0, 1, 4]), cdims=cdims)
    assert exc in str(excinfo)

    # cdims must be valid dimensions
    exc = "Values in cdims must be in [0, source.ndims]."
    with pytest.raises(AssertionError) as excinfo:
        tensorInstance.to_tenmat(rdims=rdims, cdims=np.array([2, 3, 4]))
    assert exc in str(excinfo)

    # incorrect dimensions
    exc = "Incorrect specification of dimensions, the sorted concatenation of rdims and cdims must be range(source.ndims)."
    D = []
    # do not span all dimensions
    D.append([np.array([0]), np.array([1])])
    # duplicate dimensions
    D.append([np.array([0, 1, 1]), np.array([2, 3])])
    D.append([np.array([0, 1, 1]), np.array([3])])
    for d in D:
        with pytest.raises(AssertionError) as excinfo:
            tensorInstance.to_tenmat(d[0], d[1], tshape)
        assert exc in str(excinfo)


def test_tenmat_to_tensor(caplog):
    tensorInstance = ttb.tenrand((4, 3))
    tensorInstance4 = ttb.tenrand((4, 3, 2, 2))
    # tenmat
    tenmatInstance = tensorInstance.to_tenmat(np.array([0]))
    tensorTenmatInstance = tenmatInstance.to_tensor()
    assert tensorInstance.isequal(tensorTenmatInstance)

    # 1D 1-element tenmat
    tensorInstance1 = ttb.tensor(np.array([3]))
    tenmatInstance1 = tensorInstance1.to_tenmat(np.array([0]))
    tensorTenmatInstance1 = tenmatInstance1.to_tensor()
    assert tensorInstance1.isequal(tensorTenmatInstance1)

    # 4D tenmat
    tenmatInstance4 = tensorInstance4.to_tenmat(np.array([3, 0]))
    tensorTenmatInstance4 = tenmatInstance4.to_tensor()
    assert tensorInstance4.isequal(tensorTenmatInstance4)
    assert not np.may_share_memory(tensorTenmatInstance4.data, tenmatInstance4.data)

    # Reference instead of copy
    with caplog.at_level(logging.WARNING):
        tensorTenmatInstance4_ref = tenmatInstance4.to_tensor(copy=False)
        assert not np.may_share_memory(
            tensorTenmatInstance4_ref.data, tenmatInstance4.data
        )


def test_tenmat_ctranspose(sample_tenmat_4way):
    (params, tenmatInstance) = sample_tenmat_4way

    print("\ntenmatInstance")
    print(tenmatInstance)
    print("\ntenmatInstance.data.conj().T:")
    print(tenmatInstance.data.conj().T)
    tenmatInstanceCtranspose = tenmatInstance.ctranspose()
    print("\ntenmatInstanceCtanspose")
    print(tenmatInstanceCtranspose)
    assert (tenmatInstanceCtranspose.data == tenmatInstance.data.conj().T).all()


def test_tenmat_double(sample_tenmat_4way):
    (params, tenmatInstance) = sample_tenmat_4way

    double_array = tenmatInstance.double()
    assert (double_array == tenmatInstance.data.astype(np.float64)).all()
    assert_consistent_order(tenmatInstance, double_array)

    # Verify immutability
    double_array = tenmatInstance.double(True)
    with pytest.raises(ValueError):
        double_array[0] = 1


def test_tenmat_ndims(sample_tenmat_4way):
    (params, tenmatInstance) = sample_tenmat_4way

    # tenmat of 4-way tensor -> 2 dims
    assert tenmatInstance.ndims == 2

    # empty tenmat -> 0 dims
    assert ttb.tenmat().ndims == 0


def test_tenmat_norm(sample_ndarray_1way, sample_tenmat_4way):
    (params, tenmatInstance) = sample_tenmat_4way
    params["tshape"]
    params["rdims"]
    params["cdims"]
    params["data"]
    (_, ndarrayInstance1) = sample_ndarray_1way

    # tenmat of 4-way tensor
    assert tenmatInstance.norm() == np.linalg.norm(params["data"].ravel())

    # 1D tenmat
    tensor1 = ttb.tensor(ndarrayInstance1, shape=(16,))
    tenmat1 = tensor1.to_tenmat(cdims=np.array([0]))
    assert tenmat1.norm() == np.linalg.norm(ndarrayInstance1.ravel())

    # empty tenmat
    assert ttb.tenmat().norm() == 0


def test_tenmat__setitem__():
    ndarrayInstance = np.reshape(np.arange(1, 17), (2, 2, 2, 2), order="F")
    tensorInstance = ttb.tensor(ndarrayInstance, shape=(2, 2, 2, 2))
    tenmatInstance = tensorInstance.to_tenmat(rdims=np.array([0, 1]))

    # single element -> scalar
    tenmatInstance2 = tenmatInstance.copy()
    for i in range(4):
        for j in range(4):
            tenmatInstance2[i, j] = i * 4 + j + 10
    for i in range(4):
        for j in range(4):
            assert tenmatInstance2[i, j] == i * 4 + j + 10

    # Exceptions

    # checking that index out of bounds throws exception
    exc = "index 5 is out of bounds for axis 1 with size 4"
    with pytest.raises(IndexError) as excinfo:
        tenmatInstance2[0, 5] = 100
    assert exc in str(excinfo)


def test_tenmat__getitem__():
    ndarrayInstance = np.reshape(np.arange(1, 17), (4, 4), order="F")
    tensorInstance = ttb.tensor(ndarrayInstance, shape=(4, 4))
    tenmatInstance = tensorInstance.to_tenmat(rdims=np.array([0]))

    # single element -> scalar
    for i in range(4):
        for j in range(4):
            assert ndarrayInstance[i, j] == tenmatInstance[i, j]

    # slicing -> numpy.ndarray
    assert (ndarrayInstance[0, :] == tenmatInstance[0, :]).all()
    assert (ndarrayInstance[:, 1] == tenmatInstance[:, 1]).all()

    # submatrix -> numpy.ndarray
    assert (ndarrayInstance[[0, 2], [1, 3]] == tenmatInstance[[0, 2], [1, 3]]).all()

    # end -> scalar
    assert (ndarrayInstance[-1, -1] == tenmatInstance[-1, -1]).all()


def test_tenmat__mul__(sample_ndarray_1way, sample_ndarray_4way, sample_tenmat_4way):
    (params, tenmatInstance) = sample_tenmat_4way
    tshape = params["tshape"]
    rdims = params["rdims"]
    cdims = params["cdims"]
    data = params["data"]
    (_, ndarrayInstance1) = sample_ndarray_1way
    (_, ndarrayInstance4) = sample_ndarray_4way

    # scalar * Tenmat -> Tenmat
    assert ((tenmatInstance * 5).data == (params["data"] * 5)).all()
    assert ((5 * tenmatInstance).data == (5 * params["data"])).all()
    assert ((tenmatInstance * 2.1).data == (params["data"] * 2.1)).all()
    assert ((2.2 * tenmatInstance).data == (2.2 * params["data"])).all()
    assert ((tenmatInstance * np.int64(3)).data == (params["data"] * np.int64(3))).all()
    assert ((np.int64(3) * tenmatInstance).data == (np.int64(3) * params["data"])).all()

    # Tenmat * Tenmat -> 2x2 result
    tensor0 = ttb.tensor(ndarrayInstance4)
    tenmat1 = tensor0.to_tenmat(rdims=np.array([0]), cdims=np.array([1, 2, 3]))
    tenmat2 = tensor0.to_tenmat(rdims=np.array([0, 1, 2]), cdims=np.array([3]))
    tenmatProd = tenmat1 * tenmat2
    data = np.array([[372, 884], [408, 984]])
    assert (tenmatProd.data == data).all()
    assert (tenmatProd.rindices == np.array([0])).all()
    assert (tenmatProd.cindices == np.array([1])).all()
    assert tenmatProd.tshape == (2, 2)
    assert tenmatProd.shape == (2, 2)

    # 1D column Tenmat * 1D row Tenmat -> scalar result
    tensor1 = ttb.tensor(ndarrayInstance1, shape=(16,))
    tenmat1 = tensor1.to_tenmat(cdims=np.array([0]))
    tenmat2 = tensor1.to_tenmat(rdims=np.array([0]))
    tenmatProd = tenmat1 * tenmat2
    assert np.isscalar(tenmatProd)
    assert tenmatProd == 1496

    # Exceptions

    # shape mismatch
    exc = "tenmat shape mismatch: number or columns of left operand must match number of rows of right operand."
    tenmat1 = tensor0.to_tenmat(rdims=np.array([0, 1]), cdims=np.array([2, 3]))
    tenmat2 = tensor0.to_tenmat(rdims=np.array([0, 1, 2]), cdims=np.array([3]))
    with pytest.raises(AssertionError) as excinfo:
        tenmat1 * tenmat2
    assert exc in str(excinfo)

    # type mismatch
    exc = "tenmat multiplication only valid with scalar or tenmat objects."
    tenmatNdarray4 = tensor0.to_tenmat(rdims, cdims, tshape)
    with pytest.raises(AssertionError) as excinfo:
        tenmatInstance * tenmatNdarray4.data
    assert exc in str(excinfo)


def test_tenmat__add__(sample_ndarray_2way, sample_tenmat_4way):
    (params, tenmatInstance) = sample_tenmat_4way
    tshape = params["tshape"]
    rdims = params["rdims"]
    cdims = params["cdims"]
    params["data"]
    (_, ndarrayInstance2) = sample_ndarray_2way

    # Tenmat + scalar
    assert ((tenmatInstance + 1).data == (params["data"] + 1)).all()
    assert ((1 + tenmatInstance).data == (1 + params["data"])).all()
    assert ((tenmatInstance + 2.1).data == (params["data"] + 2.1)).all()
    assert ((2.2 + tenmatInstance).data == (2.2 + params["data"])).all()
    assert ((tenmatInstance + np.int64(3)).data == (params["data"] + np.int64(3))).all()
    assert ((np.int64(3) + tenmatInstance).data == (np.int64(3) + params["data"])).all()

    # Tenmat + Tenmat
    assert (
        (tenmatInstance + tenmatInstance).data == (params["data"] + params["data"])
    ).all()

    # Exceptions

    # shape mismatch
    exc = "tenmat shape mismatch."
    tenmatNdarray2 = ttb.tenmat(np.ones((5, 5)), rdims, cdims, (1, 1, 1, 25))
    with pytest.raises(AssertionError) as excinfo:
        tenmatInstance + tenmatNdarray2
    assert exc in str(excinfo)
    with pytest.raises(AssertionError) as excinfo:
        tenmatNdarray2 + tenmatInstance
    assert exc in str(excinfo)

    # type mismatch
    exc = "tenmat addition only valid with scalar or tenmat objects."
    tenmatNdarray2 = ttb.tenmat(ndarrayInstance2, rdims, cdims, tshape)
    with pytest.raises(AssertionError) as excinfo:
        tenmatInstance + tenmatNdarray2.data
    assert exc in str(excinfo)


def test_tenmat__sub__(sample_ndarray_2way, sample_tenmat_4way):
    (params, tenmatInstance) = sample_tenmat_4way
    tshape = params["tshape"]
    rdims = params["rdims"]
    cdims = params["cdims"]
    params["data"]
    (_, ndarrayInstance2) = sample_ndarray_2way

    # Tenmat + scalar
    assert ((tenmatInstance - 1).data == (params["data"] - 1)).all()
    assert ((1 - tenmatInstance).data == (1 - params["data"])).all()
    assert ((tenmatInstance - 2.1).data == (params["data"] - 2.1)).all()
    assert ((2.2 - tenmatInstance).data == (2.2 - params["data"])).all()
    assert ((tenmatInstance - np.int64(3)).data == (params["data"] - np.int64(3))).all()
    assert ((np.int64(3) - tenmatInstance).data == (np.int64(3) - params["data"])).all()

    # Tenmat + Tenmat
    assert (
        (tenmatInstance - tenmatInstance).data == (params["data"] - params["data"])
    ).all()

    # Exceptions

    # shape mismatch
    exc = "tenmat shape mismatch."
    tenmatNdarray2 = ttb.tenmat(np.ones((5, 5)), rdims, cdims, (1, 1, 1, 25))
    with pytest.raises(AssertionError) as excinfo:
        tenmatInstance - tenmatNdarray2
    assert exc in str(excinfo)

    # type mismatch
    exc = "tenmat subtraction only valid with scalar or tenmat objects."
    tenmatNdarray2 = ttb.tenmat(ndarrayInstance2, rdims, cdims, tshape)
    with pytest.raises(AssertionError) as excinfo:
        tenmatInstance - tenmatNdarray2.data
    assert exc in str(excinfo)


def test_tenmat__rsub__(sample_ndarray_2way, sample_tenmat_4way):
    (params, tenmatInstance) = sample_tenmat_4way
    tshape = params["tshape"]
    rdims = params["rdims"]
    cdims = params["cdims"]
    params["data"]
    (_, ndarrayInstance2) = sample_ndarray_2way

    # Tenmat + scalar
    assert ((1 - tenmatInstance).data == (1 - params["data"])).all()
    assert ((2.2 - tenmatInstance).data == (2.2 - params["data"])).all()
    assert ((np.int64(3) - tenmatInstance).data == (np.int64(3) - params["data"])).all()

    # Tenmat + Tenmat
    assert (
        (tenmatInstance.__rsub__(tenmatInstance)).data
        == (params["data"] - params["data"])
    ).all()

    # Exceptions

    # shape mismatch
    exc = "tenmat shape mismatch."
    tenmatNdarray2 = ttb.tenmat(np.ones((5, 5)), rdims, cdims, (1, 1, 1, 25))
    with pytest.raises(AssertionError) as excinfo:
        tenmatInstance.__rsub__(tenmatNdarray2)
    assert exc in str(excinfo)

    # type mismatch
    exc = "tenmat subtraction only valid with scalar or tenmat objects."
    tenmatNdarray2 = ttb.tenmat(ndarrayInstance2, rdims, cdims, tshape)
    with pytest.raises(AssertionError) as excinfo:
        tenmatInstance.__rsub__(tenmatNdarray2.data)
    assert exc in str(excinfo)


def test_tenmat__pos__(sample_tenmat_4way):
    (params, tenmatInstance) = sample_tenmat_4way
    params["data"]

    # +Tenmat yields no change
    assert ((+tenmatInstance).data == params["data"]).all()


def test_tenmat__neg__(sample_tenmat_4way):
    (params, tenmatInstance) = sample_tenmat_4way
    params["data"]

    # +Tenmat yields no change
    assert ((-tenmatInstance).data == -params["data"]).all()


def test_tenmat__str__(
    sample_ndarray_1way, sample_ndarray_2way, sample_ndarray_4way, sample_tenmat_4way
):
    (params, tenmatInstance) = sample_tenmat_4way
    tshape = params["tshape"]
    rdims = params["rdims"]
    cdims = params["cdims"]
    params["data"]
    (_, ndarrayInstance1) = sample_ndarray_1way
    (_, ndarrayInstance2) = sample_ndarray_2way
    (_, ndarrayInstance4) = sample_ndarray_4way

    # Empty
    tenmatInstance = ttb.tenmat()
    s = ""
    s += "matrix corresponding to a tensor of shape () with order F\n"
    s += "rindices = [  ] (modes of tensor corresponding to rows)\n"
    s += "cindices = [  ] (modes of tensor corresponding to columns)\n"
    s += "data = []\n"
    assert s == tenmatInstance.__str__()

    # Test 1D
    tensor1 = ttb.tensor(ndarrayInstance1, shape=tshape)
    tenmatInstance = tensor1.to_tenmat(rdims, cdims, tshape)
    s = ""
    s += "matrix corresponding to a tensor of shape "
    s += str(tenmatInstance.tshape)
    s += " with order F\n"
    s += "rindices = "
    s += "[ " + (", ").join([str(int(d)) for d in tenmatInstance.rindices]) + " ] "
    s += "(modes of tensor corresponding to rows)\n"
    s += "cindices = "
    s += "[ " + (", ").join([str(int(d)) for d in tenmatInstance.cindices]) + " ] "
    s += "(modes of tensor corresponding to columns)\n"
    s += "data[:, :] = \n"
    s += str(tenmatInstance.data)
    s += "\n"
    assert s == tenmatInstance.__str__()

    ## Test 2D
    tenmatInstance = ttb.tenmat(ndarrayInstance2, rdims, cdims, tshape)
    s = ""
    s += "matrix corresponding to a tensor of shape "
    s += str(tenmatInstance.tshape)
    s += " with order F\n"
    s += "rindices = "
    s += "[ " + (", ").join([str(int(d)) for d in tenmatInstance.rindices]) + " ] "
    s += "(modes of tensor corresponding to rows)\n"
    s += "cindices = "
    s += "[ " + (", ").join([str(int(d)) for d in tenmatInstance.cindices]) + " ] "
    s += "(modes of tensor corresponding to columns)\n"
    s += "data[:, :] = \n"
    s += str(tenmatInstance.data)
    s += "\n"
    assert s == tenmatInstance.__str__()

    # Test 4D
    tensor0 = ttb.tensor(ndarrayInstance4, shape=tshape)
    tenmatInstance = tensor0.to_tenmat(rdims, cdims, tshape)
    s = ""
    s += "matrix corresponding to a tensor of shape "
    s += str(tenmatInstance.tshape)
    s += " with order F\n"
    s += "rindices = "
    s += "[ " + (", ").join([str(int(d)) for d in tenmatInstance.rindices]) + " ] "
    s += "(modes of tensor corresponding to rows)\n"
    s += "cindices = "
    s += "[ " + (", ").join([str(int(d)) for d in tenmatInstance.cindices]) + " ] "
    s += "(modes of tensor corresponding to columns)\n"
    s += "data[:, :] = \n"
    s += str(tenmatInstance.data)
    s += "\n"
    assert s == tenmatInstance.__str__()


def test_tenmat_isequal():
    # Negative test
    with pytest.raises(ValueError):
        ttb.tenmat().isequal("Not a tenmat")
