# Copyright 2024 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.
from __future__ import annotations

import copy

import numpy as np
import pytest

import pyttb as ttb
from pyttb.tensor import min_split, mttv_left, mttv_mid
from tests.test_utils import assert_consistent_order


@pytest.fixture()
def sample_tensor_4way():
    # FIXME: Make the tests that depend on this
    #  configured to use the common version under conftest.py
    data = np.arange(1, 82)
    shape = (3, 3, 3, 3)
    params = {"data": np.reshape(data, np.array(shape), order="F"), "shape": shape}
    tensorInstance = ttb.tensor(data, shape)
    return params, tensorInstance


def test_tensor_initialization_empty():
    empty = np.array([])

    # No args
    tensorInstance = ttb.tensor()
    assert np.array_equal(tensorInstance.data, empty)
    assert tensorInstance.shape == ()
    assert_consistent_order(tensorInstance, tensorInstance.data)


def test_tensor_initialization_from_data(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way
    assert np.array_equal(tensorInstance.data, params["data"])
    assert tensorInstance.shape == params["shape"]
    assert_consistent_order(tensorInstance, tensorInstance.data)

    with pytest.raises(AssertionError) as excinfo:
        ttb.tensor(params["data"], ())
    assert "Empty tensor cannot contain any elements" in str(excinfo)

    with pytest.raises(AssertionError) as excinfo:
        ttb.tensor(params["data"], (2, 4))
    assert "TTB:WrongSize, Size of data does not match specified size of tensor" in str(
        excinfo
    )

    # TODO how else to break this logical statement?
    data = np.array([["a", 2, 3], [4, 5, 6]])
    with pytest.raises(AssertionError) as excinfo:
        ttb.tensor(data, (2, 3))
    assert "First argument must be a multidimensional array." in str(excinfo)

    # 1D tensors
    # no shape specified
    tensorInstance1 = ttb.tensor(np.array([1, 2, 3]))
    data = np.array([1, 2, 3])
    assert tensorInstance1.data.shape == data.shape, (
        f"tensorInstance1:\n{tensorInstance1}"
    )
    assert np.array_equal(tensorInstance1.data, data), (
        f"tensorInstance1:\n{tensorInstance1}"
    )
    assert_consistent_order(tensorInstance, tensorInstance.data)

    # shape is 1 x 3
    tensorInstance1 = ttb.tensor(np.array([1, 2, 3]), (1, 3))
    data = np.array([[1, 2, 3]])
    assert tensorInstance1.data.shape == data.shape, (
        f"tensorInstance1:\n{tensorInstance1}"
    )
    assert np.array_equal(tensorInstance1.data, data), (
        f"tensorInstance1:\n{tensorInstance1}"
    )
    assert_consistent_order(tensorInstance, tensorInstance.data)

    # shape is 3 x 1
    tensorInstance1 = ttb.tensor(np.array([1, 2, 3]), (3, 1))
    data = np.array([[1], [2], [3]])
    assert tensorInstance1.data.shape == data.shape, (
        f"tensorInstance1:\n{tensorInstance1}"
    )
    assert np.array_equal(tensorInstance1.data, data), (
        f"tensorInstance1:\n{tensorInstance1}"
    )
    assert_consistent_order(tensorInstance, tensorInstance.data)


def test_tensor_initialization_from_function(memory_layout):
    order = memory_layout["order"]

    # Dummy function handle
    def function_handle(x):  # noqa: ARG001
        return np.array([[1, 2, 3], [4, 5, 6]], order=order)

    shape = (2, 3)
    data = np.array([[1, 2, 3], [4, 5, 6]], order=order)

    a = ttb.tensor.from_function(function_handle, shape)
    assert np.array_equal(a.data, data)
    assert a.shape == shape
    assert_consistent_order(a, a.data)


def test_tensor_copy(sample_tensor_2way):
    (params, tensor_instance) = sample_tensor_2way
    copy_tensor = tensor_instance.copy()
    assert copy_tensor.isequal(tensor_instance)

    # make sure it is a deep copy
    copy_tensor[0] = 0
    assert copy_tensor[0] != tensor_instance[0]


def test_tensor__deepcopy__(sample_tensor_2way):
    (params, tensor_instance) = sample_tensor_2way
    copy_tensor = copy.deepcopy(tensor_instance)
    assert copy_tensor.isequal(tensor_instance)

    # make sure it is a deep copy
    copy_tensor[0] = 0
    assert copy_tensor[0] != tensor_instance[0]


def test_tensor_find(sample_tensor_2way, sample_tensor_3way, sample_tensor_4way):
    (params, tensorInstance) = sample_tensor_2way
    subs, vals = tensorInstance.find()
    a = ttb.sptensor(subs, vals, tensorInstance.shape).to_tensor()
    assert np.array_equal(a.data, tensorInstance.data), f"subs: {subs}\nvals: {vals}"
    assert a.shape == tensorInstance.shape, f"subs: {subs}\nvals: {vals}"
    assert_consistent_order(tensorInstance, a.data)

    (params, tensorInstance) = sample_tensor_3way
    subs, vals = tensorInstance.find()
    a = ttb.sptensor(subs, vals, tensorInstance.shape).to_tensor()
    assert np.array_equal(a.data, tensorInstance.data), f"subs: {subs}\nvals: {vals}"
    assert a.shape == tensorInstance.shape, f"subs: {subs}\nvals: {vals}"

    (params, tensorInstance) = sample_tensor_4way
    subs, vals = tensorInstance.find()
    a = ttb.sptensor(subs, vals, tensorInstance.shape).to_tensor()
    assert np.array_equal(a.data, tensorInstance.data), f"subs: {subs}\nvals: {vals}"
    assert a.shape == tensorInstance.shape, f"subs: {subs}\nvals: {vals}"


def test_tensor_ndims(sample_tensor_2way, sample_tensor_3way, sample_tensor_4way):
    (params, tensorInstance) = sample_tensor_2way
    assert tensorInstance.ndims == len(params["shape"])

    (params, tensorInstance) = sample_tensor_3way
    assert tensorInstance.ndims == len(params["shape"])

    (params, tensorInstance) = sample_tensor_4way
    assert tensorInstance.ndims == len(params["shape"])

    # Empty tensor has zero dimensions
    assert ttb.tensor(np.array([])) == 0


class TestSetItem:
    def test_linear_single_value(self, sample_tensor_2way):
        (params, tensorInstance) = sample_tensor_2way
        dataCopy = params["data"].copy()
        first_element_idx = np.unravel_index([0], dataCopy.shape, "F")
        last_element_idx = np.unravel_index(
            [np.prod(dataCopy.shape) - 1], dataCopy.shape, "F"
        )
        arbitrary_value = 13.0

        # NP Array Key
        tensorInstance[np.array([0])] = arbitrary_value
        dataCopy[first_element_idx] = arbitrary_value
        assert np.array_equal(tensorInstance.data, dataCopy)

        # Int Key
        tensorInstance[0] = arbitrary_value + 1
        dataCopy[first_element_idx] = arbitrary_value + 1
        assert np.array_equal(tensorInstance.data, dataCopy)
        tensorInstance[-1] = arbitrary_value + 2
        dataCopy[last_element_idx] = arbitrary_value + 2
        assert np.array_equal(tensorInstance.data, dataCopy)

        # Slice Key
        tensorInstance[0:1] = arbitrary_value + 3
        dataCopy[first_element_idx] = arbitrary_value + 3
        assert np.array_equal(tensorInstance.data, dataCopy)
        tensorInstance[-1:] = arbitrary_value + 4
        dataCopy[last_element_idx] = arbitrary_value + 4
        assert np.array_equal(tensorInstance.data, dataCopy)

    def test_linear_multiple_values(self, sample_tensor_2way):
        (params, tensorInstance) = sample_tensor_2way
        dataCopy = params["data"].copy()
        tensor_idx = [0, 1, 2]
        multi_element_idx = np.unravel_index(tensor_idx, dataCopy.shape, "F")
        arbitray_value = 13.0

        # NP Array Key
        tensorInstance[np.array(tensor_idx)] = arbitray_value
        dataCopy[multi_element_idx] = arbitray_value
        assert np.array_equal(tensorInstance.data, dataCopy)

        # List Key
        tensorInstance[tensor_idx] = arbitray_value + 1
        dataCopy[multi_element_idx] = arbitray_value + 1
        assert np.array_equal(tensorInstance.data, dataCopy)

        # Slice Key
        tensorInstance[0 : tensor_idx[-1] + 1] = arbitray_value + 2
        dataCopy[multi_element_idx] = arbitray_value + 2
        assert np.array_equal(tensorInstance.data, dataCopy)

        # Assign with array
        tensor_vector = ttb.tensor(np.array([0, 0, 0, 0]))
        tensor_vector[np.array([0, 1, 2])] = np.array([3, 4, 5])
        assert np.array_equal(tensor_vector.data, np.array([3, 4, 5, 0]))

        # Linear Index with constant, index out of bounds
        with pytest.raises(AssertionError) as excinfo:
            tensorInstance[np.array([0, 3, 99])] = 13.0
        assert (
            "TTB:BadIndex In assignment X[I] = Y, a tensor X cannot be resized"
            in str(excinfo)
        )

    def test_invalid_options(self, sample_tensor_2way):
        (params, tensorInstance) = sample_tensor_2way
        with pytest.raises(ValueError) as excinfo:
            tensorInstance[0, "a", 5] = 13.0
        assert "must be numeric" in str(excinfo)

        with pytest.raises(AssertionError) as excinfo:

            class BadKey:
                pass

            tensorInstance[BadKey] = 13.0
        assert "Invalid use of tensor setitem" in str(excinfo)

    def test_subtensor_assign(self, sample_tensor_2way):
        (params, tensorInstance) = sample_tensor_2way
        dataCopy = params["data"].copy()
        arbitray_value = 13.0

        # Assign with constant
        dataCopy[1, 1] = arbitray_value + 1
        tensorInstance[1, 1] = arbitray_value + 1
        assert np.array_equal(tensorInstance.data, dataCopy)

        # Assign with np array
        dataCopy[0:2, 0:3] = arbitray_value + 2
        tensorInstance[0:2, 0:3] = dataCopy[0:2, 0:3]
        assert np.array_equal(tensorInstance.data, dataCopy)

        # Assign with tensor
        tensorInstance[:, :] = tensorInstance
        assert np.array_equal(tensorInstance.data, dataCopy)

        # Keys containing lists
        tensorInstance[[0, 1], [0, 1]] = arbitray_value + 3
        dataCopy[[0, 1], [0, 1]] = arbitray_value + 3
        assert np.array_equal(tensorInstance.data, dataCopy)

        # Assign with negative indices
        empty_tensor = ttb.tensor()
        empty_tensor[0, 0] = 1
        empty_tensor[-1, -1] = 2
        assert empty_tensor[0, 0] == 2

        # Assign with negative slice
        empty_tensor = ttb.tensor()
        empty_tensor[1, 1] = 0
        empty_tensor[:-1, :-1] = 3
        assert empty_tensor[0, 0] == 3

    def test_subtensor_growth(
        self,
    ):
        arbitray_value = 0

        # Empty Tensor growth
        empty_tensor = ttb.tensor()
        empty_tensor[0, 0, 0] = arbitray_value
        assert np.array_equal(empty_tensor.data, np.array([[[arbitray_value]]]))
        assert empty_tensor.shape == (1, 1, 1)

        # Add dimension
        empty_tensor[0, 0, 0, 0] = arbitray_value + 1
        assert np.array_equal(empty_tensor.data, np.array([[[[arbitray_value + 1]]]]))
        assert empty_tensor.shape == (1, 1, 1, 1)

    def test_subscripts_assign(self, sample_tensor_2way):
        (params, tensorInstance) = sample_tensor_2way
        dataCopy = params["data"].copy()
        arbitrary_value = 13

        # Subscripts with constant
        tensorInstance[np.array([[1, 1]])] = arbitrary_value
        dataCopy[1, 1] = arbitrary_value
        assert np.array_equal(tensorInstance.data, dataCopy)

        # Multiple Subscripts with constant
        subs = np.array([[1, 1], [1, 2]])
        tensorInstance[subs] = arbitrary_value + 1
        dataCopy[([1, 1], [1, 2])] = arbitrary_value + 1
        assert np.array_equal(tensorInstance.data, dataCopy)
        # Make sure they are self consistent with subtensor
        one_by_one_values = []
        for one_sub in subs:
            one_by_one_values.append(tensorInstance[tuple(one_sub)])
        assert np.array_equal(tensorInstance[subs].data, one_by_one_values)

    def test_subscript_growth(
        self,
    ):
        # Subscripts add element to empty tensor
        empty_tensor = ttb.tensor()
        first_arbitrary_index = np.array([[0, 1], [2, 2]])
        second_arbitrary_index = np.array([[1, 2], [3, 3]])
        value = 4
        # Subscripts grow existing tensor
        empty_tensor[first_arbitrary_index] = value
        empty_tensor[second_arbitrary_index] = value

        # Test Empty Tensor Set Item, subscripts
        emptyTensor = ttb.tensor(np.array([]))
        emptyTensor[np.array([[0, 0, 0]])] = 0
        assert np.array_equal(emptyTensor.data, np.array([[[0]]]))
        assert emptyTensor.shape == (1, 1, 1)


class TestGetItem:
    def test_linear(self, sample_tensor_2way):
        (params, tensorInstance) = sample_tensor_2way
        assert tensorInstance[np.array([0])] == params["data"][0, 0]
        assert tensorInstance[0] == params["data"][0, 0]
        assert tensorInstance[0:1] == params["data"][0, 0]
        negative_first_idx = -np.prod(tensorInstance.shape)
        assert tensorInstance[negative_first_idx] == params["data"][0, 0]
        with pytest.raises(AssertionError) as excinfo:
            tensorInstance[np.array([0]), np.array([0]), np.array([0])]
        assert "Linear indexing requires single input array" in str(excinfo)
        assert np.array_equal(
            tensorInstance[[0, 1, 2]], tensorInstance[np.array([0, 1, 2])]
        )

    def test_subtensor(self, sample_tensor_2way):
        (params, tensorInstance) = sample_tensor_2way
        # Single element (using positive and negative indices)
        assert tensorInstance[0, 0] == params["data"][0, 0]
        negative_first_idx = tuple(-1 * dim for dim in tensorInstance.shape)
        assert tensorInstance[negative_first_idx] == params["data"][0, 0]
        # Slice
        assert tensorInstance[:, :].isequal(tensorInstance)
        three_way_data = np.random.random((2, 3, 4))
        two_slices = (slice(None, None, None), 0, slice(None, None, None))
        assert np.array_equal(
            ttb.tensor(three_way_data)[two_slices].double(),
            three_way_data[two_slices],
        )
        # Combining slice with (multi-)integer indices
        assert np.array_equal(
            tensorInstance[np.array([0, 1]), :].data, tensorInstance.data[[0, 1], :]
        )
        assert np.array_equal(tensorInstance[0, :].data, tensorInstance.data[0, :])
        assert np.array_equal(tensorInstance[:, 0].data, tensorInstance.data[:, 0])

    def test_tensor_subscripts(self, sample_tensor_2way):
        (params, tensorInstance) = sample_tensor_2way
        assert tensorInstance[np.array([[0, 0]])] == params["data"][0, 0]
        subs = np.array([[0, 2], [1, 1], [1, 2]])
        subscript_values = tensorInstance[subs]
        # Make sure they match
        assert np.array_equal(
            subscript_values,
            params["data"][([0, 1, 1], [2, 1, 2])],
        )
        # Make sure they are self consistent with subtensor
        one_by_one_values = []
        for one_sub in subs:
            one_by_one_values.append(tensorInstance[tuple(one_sub)])
        assert np.array_equal(subscript_values, one_by_one_values)
        with pytest.raises(AssertionError) as excinfo:
            # Must use numpy arrays for subscripts
            tensorInstance[[[0, 2], [1, 1], [1, 2]]]
        assert "Invalid use of tensor getitem" in str(excinfo)


def test_tensor_logical_and(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way

    # Tensor And
    tensor_and = tensorInstance.logical_and(tensorInstance).data
    assert np.array_equal(tensor_and, np.ones(params["shape"]))
    assert tensor_and.dtype == tensorInstance.data.dtype
    assert_consistent_order(tensorInstance, tensor_and)

    # Non-zero And
    non_zero_and = tensorInstance.logical_and(1).data
    assert np.array_equal(non_zero_and, np.ones(params["shape"]))
    assert non_zero_and.dtype == tensorInstance.data.dtype

    # Zero And
    zero_and = tensorInstance.logical_and(0).data
    assert np.array_equal(zero_and, np.zeros(params["shape"]))
    assert zero_and.dtype == tensorInstance.data.dtype


def test_tensor__eq__(sample_tensor_2way, sample_tensor_3way, sample_tensor_4way):
    (params, tensorInstance) = sample_tensor_2way

    # Tensor tensor equality
    eq_tensor = tensorInstance == tensorInstance
    assert np.all(eq_tensor.data)
    assert_consistent_order(tensorInstance, eq_tensor.data)

    # Tensor scalar equality, not equal
    assert not (tensorInstance == 7).data.any()

    # Tensor scalar equality, is equal
    data = np.zeros(params["data"].shape)
    data[0, 0] = 1
    assert np.array_equal((tensorInstance == 1).data, data)

    (params3, tensorInstance3) = sample_tensor_3way

    # Tensor tensor equality
    assert np.all((tensorInstance3 == tensorInstance3).data)

    (params4, tensorInstance4) = sample_tensor_4way

    # Tensor tensor equality
    assert np.all((tensorInstance4 == tensorInstance4).data)


def test_tensor__ne__(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way

    # Tensor tensor equality
    ne_tensor = tensorInstance != tensorInstance
    assert not (ne_tensor.data).any()
    assert_consistent_order(tensorInstance, ne_tensor.data)

    # Tensor scalar equality, not equal
    assert np.all((tensorInstance != 7).data)

    # Tensor scalar equality, is equal
    data = np.zeros(params["data"].shape)
    data[0, 0] = 1
    assert not ((tensorInstance != 1).data == data).any()


def test_tensor_full(sample_tensor_2way, sample_tensor_3way, sample_tensor_4way):
    (params2, tensorInstance2) = sample_tensor_2way
    debug_str = (
        f"param2['data']: {params2['data']}\n"
        f"tensorInstace2.data: {tensorInstance2.data}\n"
        f"tensorInstace2.full(): {tensorInstance2.full()}"
    )
    full_tensor = tensorInstance2.full()
    assert np.array_equal(full_tensor.data, params2["data"]), debug_str
    assert_consistent_order(tensorInstance2, full_tensor.data)

    (params3, tensorInstance3) = sample_tensor_3way
    debug_str = (
        f"param3['data']: {params3['data']}\n"
        f"tensorInstace3.data: {tensorInstance3.data}\n"
        f"tensorInstace3.full(): {tensorInstance3.full()}"
    )
    assert np.array_equal(tensorInstance3.full().data, params3["data"]), debug_str

    (params4, tensorInstance4) = sample_tensor_4way
    debug_str = (
        f"param4['data']: {params4['data']}\n"
        f"tensorInstace4.data: {tensorInstance4.data}\n"
        f"tensorInstace4.full(): {tensorInstance4.full()}"
    )
    assert np.array_equal(tensorInstance4.full().data, params4["data"]), debug_str


def test_tensor__ge__(sample_tensor_2way, sample_tensor_3way, sample_tensor_4way):
    (params, tensorInstance) = sample_tensor_2way

    tensorLarger = ttb.tensor(params["data"] + 1)
    tensorSmaller = ttb.tensor(params["data"] - 1)

    assert np.all((tensorInstance >= tensorInstance).data)
    assert np.all((tensorInstance >= tensorSmaller).data)
    ge_tensor = tensorInstance >= tensorLarger
    assert not (ge_tensor.data).any()
    assert_consistent_order(tensorInstance, ge_tensor.data)

    (params, tensorInstance) = sample_tensor_3way

    tensorLarger = ttb.tensor(params["data"] + 1)
    tensorSmaller = ttb.tensor(params["data"] - 1)

    assert np.all((tensorInstance >= tensorInstance).data)
    assert np.all((tensorInstance >= tensorSmaller).data)
    assert not ((tensorInstance >= tensorLarger).data).any()

    (params, tensorInstance) = sample_tensor_4way

    tensorLarger = ttb.tensor(params["data"] + 1)
    tensorSmaller = ttb.tensor(params["data"] - 1)

    assert np.all((tensorInstance >= tensorInstance).data)
    assert np.all((tensorInstance >= tensorSmaller).data)
    assert not ((tensorInstance >= tensorLarger).data).any()


def test_tensor__gt__(sample_tensor_2way, sample_tensor_3way, sample_tensor_4way):
    (params, tensorInstance) = sample_tensor_2way

    tensorLarger = ttb.tensor(params["data"] + 1)
    tensorSmaller = ttb.tensor(params["data"] - 1)

    assert not ((tensorInstance > tensorInstance).data).any()
    assert np.all((tensorInstance > tensorSmaller).data)
    gt_tensor = tensorInstance > tensorLarger
    assert not (gt_tensor.data).any()
    assert_consistent_order(tensorInstance, gt_tensor.data)

    (params, tensorInstance) = sample_tensor_3way

    tensorLarger = ttb.tensor(params["data"] + 1)
    tensorSmaller = ttb.tensor(params["data"] - 1)

    assert not ((tensorInstance > tensorInstance).data).any()
    assert np.all((tensorInstance > tensorSmaller).data)
    assert not ((tensorInstance > tensorLarger).data).any()

    (params, tensorInstance) = sample_tensor_4way

    tensorLarger = ttb.tensor(params["data"] + 1)
    tensorSmaller = ttb.tensor(params["data"] - 1)

    assert not ((tensorInstance > tensorInstance).data).any()
    assert np.all((tensorInstance > tensorSmaller).data)
    assert not ((tensorInstance > tensorLarger).data).any()


def test_tensor__le__(sample_tensor_2way, sample_tensor_3way, sample_tensor_4way):
    (params, tensorInstance) = sample_tensor_2way

    tensorLarger = ttb.tensor(params["data"] + 1)
    tensorSmaller = ttb.tensor(params["data"] - 1)

    assert np.all((tensorInstance <= tensorInstance).data)
    assert not ((tensorInstance <= tensorSmaller).data).any()
    le_tensor = tensorInstance <= tensorLarger
    assert np.all(le_tensor.data)
    assert_consistent_order(tensorInstance, le_tensor.data)

    (params, tensorInstance) = sample_tensor_3way

    tensorLarger = ttb.tensor(params["data"] + 1)
    tensorSmaller = ttb.tensor(params["data"] - 1)

    assert np.all((tensorInstance <= tensorInstance).data)
    assert not ((tensorInstance <= tensorSmaller).data).any()
    assert np.all((tensorInstance <= tensorLarger).data)

    (params, tensorInstance) = sample_tensor_4way

    tensorLarger = ttb.tensor(params["data"] + 1)
    tensorSmaller = ttb.tensor(params["data"] - 1)

    assert np.all((tensorInstance <= tensorInstance).data)
    assert not ((tensorInstance <= tensorSmaller).data).any()
    assert np.all((tensorInstance <= tensorLarger).data)


def test_tensor__lt__(sample_tensor_2way, sample_tensor_3way, sample_tensor_4way):
    (params, tensorInstance) = sample_tensor_2way

    tensorLarger = ttb.tensor(params["data"] + 1)
    tensorSmaller = ttb.tensor(params["data"] - 1)

    assert not ((tensorInstance < tensorInstance).data).any()
    assert not ((tensorInstance < tensorSmaller).data).any()
    lt_tensor = tensorInstance < tensorLarger
    assert np.all(lt_tensor.data)
    assert_consistent_order(tensorInstance, lt_tensor.data)

    (params, tensorInstance) = sample_tensor_3way

    tensorLarger = ttb.tensor(params["data"] + 1)
    tensorSmaller = ttb.tensor(params["data"] - 1)

    assert not ((tensorInstance < tensorInstance).data).any()
    assert not ((tensorInstance < tensorSmaller).data).any()
    assert np.all((tensorInstance < tensorLarger).data)

    (params, tensorInstance) = sample_tensor_4way

    tensorLarger = ttb.tensor(params["data"] + 1)
    tensorSmaller = ttb.tensor(params["data"] - 1)

    assert not ((tensorInstance < tensorInstance).data).any()
    assert not ((tensorInstance < tensorSmaller).data).any()
    assert np.all((tensorInstance < tensorLarger).data)


def test_tensor_norm(sample_tensor_2way, sample_tensor_3way, sample_tensor_4way):
    # 2-way tensor
    (params2, tensorInstance2) = sample_tensor_2way
    assert tensorInstance2.norm() == np.linalg.norm(params2["data"].ravel()), (
        f"tensorInstace2.norm(): {tensorInstance2.norm()}"
    )

    # 3-way tensor
    (params3, tensorInstance3) = sample_tensor_3way
    assert tensorInstance3.norm() == np.linalg.norm(params3["data"].ravel()), (
        f"tensorInstace3.norm(): {tensorInstance3.norm()}"
    )

    # 4-way tensor
    (params4, tensorInstance4) = sample_tensor_4way
    assert tensorInstance4.norm() == np.linalg.norm(params4["data"].ravel()), (
        f"tensorInstace4.norm(): {tensorInstance4.norm()}"
    )


def test_tensor_logical_not(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way

    not_tensor = tensorInstance.logical_not().data
    assert np.array_equal(not_tensor, np.logical_not(params["data"]))
    assert not_tensor.dtype == tensorInstance.data.dtype
    assert_consistent_order(tensorInstance, not_tensor)


def test_tensor_logical_or(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way

    # Tensor Or
    or_tensor = tensorInstance.logical_or(tensorInstance).data
    assert np.array_equal(or_tensor, np.ones(params["shape"]))
    assert or_tensor.dtype == tensorInstance.data.dtype
    assert_consistent_order(tensorInstance, or_tensor)

    # Non-zero Or
    non_zero_or = tensorInstance.logical_or(1).data
    assert np.array_equal(non_zero_or, np.ones(params["shape"]))
    assert non_zero_or.dtype == tensorInstance.data.dtype

    # Zero Or
    zero_or = tensorInstance.logical_or(0).data
    assert np.array_equal(zero_or, np.ones(params["shape"]))
    assert zero_or.dtype == tensorInstance.data.dtype


def test_tensor_logical_xor(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way

    # Tensor xor
    xor_tensor = tensorInstance.logical_xor(tensorInstance).data
    assert np.array_equal(xor_tensor, np.zeros(params["shape"]))
    assert xor_tensor.dtype == tensorInstance.data.dtype
    assert_consistent_order(tensorInstance, xor_tensor)

    # Non-zero xor
    non_zero_xor = tensorInstance.logical_xor(1).data
    assert np.array_equal(non_zero_xor, np.zeros(params["shape"]))
    assert non_zero_xor.dtype == tensorInstance.data.dtype

    # Zero xor
    zero_xor = tensorInstance.logical_xor(0).data
    assert np.array_equal(zero_xor, np.ones(params["shape"]))
    assert zero_xor.dtype == tensorInstance.data.dtype


def test_tensor__add__(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way

    # Tensor + Tensor
    tensor_plus_tensor = tensorInstance + tensorInstance
    assert np.array_equal(tensor_plus_tensor.data, 2 * (params["data"]))
    assert_consistent_order(tensorInstance, tensor_plus_tensor.data)

    # Tensor + scalar
    tensor_plus_scalar = tensorInstance + 1
    assert np.array_equal(tensor_plus_scalar.data, 1 + (params["data"]))
    assert_consistent_order(tensorInstance, tensor_plus_scalar.data)

    # scalar + Tensor
    scalar_plus_tensor = 1 + tensorInstance
    assert np.array_equal(scalar_plus_tensor.data, 1 + (params["data"]))
    assert_consistent_order(tensorInstance, scalar_plus_tensor.data)


def test_tensor__sub__(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way

    # Tensor - Tensor
    tensor_minus_tensor = tensorInstance - tensorInstance
    assert np.array_equal(tensor_minus_tensor.data, 0 * (params["data"]))
    assert_consistent_order(tensorInstance, tensor_minus_tensor.data)

    # Tensor - scalar
    tensor_minus_scalar = tensorInstance - 1
    assert np.array_equal((tensorInstance - 1).data, (params["data"] - 1))
    assert_consistent_order(tensorInstance, tensor_minus_scalar.data)


def test_tensor__pow__(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way

    # Tensor** Tensor
    tensor_pow_tensor = tensorInstance**tensorInstance
    assert np.array_equal(tensor_pow_tensor.data, (params["data"] ** params["data"]))
    assert_consistent_order(tensorInstance, tensor_pow_tensor.data)

    # Tensor**Scalar
    tensor_pow_scalar = tensorInstance**2
    assert np.array_equal(tensor_pow_scalar.data, (params["data"] ** 2))
    assert_consistent_order(tensorInstance, tensor_pow_scalar.data)


def test_tensor__mul__(sample_tensor_2way, sample_ktensor_2way):
    (params, tensorInstance) = sample_tensor_2way
    (_, ktensorInstance) = sample_ktensor_2way

    # Tensor* Tensor
    tensor_mul_tensor = tensorInstance * tensorInstance
    assert np.array_equal(tensor_mul_tensor.data, (params["data"] * params["data"]))
    assert_consistent_order(tensorInstance, tensor_mul_tensor.data)

    # Tensor*Scalar
    tensor_mul_scalar = tensorInstance * 2
    assert np.array_equal(tensor_mul_scalar.data, (params["data"] * 2))
    assert_consistent_order(tensorInstance, tensor_mul_scalar.data)

    # Tensor * Sptensor
    tensor_mul_sptensor = tensorInstance * tensorInstance.to_sptensor()
    assert np.array_equal(
        tensor_mul_sptensor.data,
        (params["data"] * params["data"]),
    )
    assert_consistent_order(tensorInstance, tensor_mul_sptensor.data)

    # Make 2x2 into 2x3
    ktensorInstance.factor_matrices[1] = np.array([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
    tensor_mul_ktensor = tensorInstance * ktensorInstance
    assert np.array_equal(
        tensor_mul_ktensor.data,
        tensorInstance.data * ktensorInstance.full().data,
    )
    assert_consistent_order(tensorInstance, tensor_mul_ktensor.data)


def test_tensor__rmul__(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way

    # Scalar * Tensor, only resolves when left object doesn't have appropriate __mul__
    rmul_tensor = 2 * tensorInstance
    assert np.array_equal(rmul_tensor.data, (params["data"] * 2))
    assert_consistent_order(tensorInstance, rmul_tensor.data)


def test_tensor__pos__(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way

    # +Tensor yields no change
    pos_tensor = +tensorInstance
    assert np.array_equal(pos_tensor.data, params["data"])
    assert_consistent_order(tensorInstance, pos_tensor.data)


def test_tensor__neg__(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way

    # -Tensor yields negated copy of tensor
    neg_tensor = -tensorInstance
    assert np.array_equal(neg_tensor.data, -1 * params["data"])
    # Original tensor should remain unchanged
    assert np.array_equal((tensorInstance).data, params["data"])
    assert_consistent_order(tensorInstance, neg_tensor.data)


def test_tensor_double(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way

    double_array = tensorInstance.double()
    assert np.array_equal(double_array, params["data"])
    assert_consistent_order(tensorInstance, double_array)

    # Verify immutability
    double_array = tensorInstance.double(True)
    with pytest.raises(ValueError):
        double_array[0] = 1


def test_tensor_isequal(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way
    subs = []
    vals = []
    for j in range(3):
        for i in range(2):
            subs.append([i, j])
            vals.append([params["data"][i, j]])
    sptensorInstance = ttb.sptensor(np.array(subs), np.array(vals), params["shape"])

    assert tensorInstance.isequal(tensorInstance)
    assert tensorInstance.isequal(sptensorInstance)

    # Tensor is not equal to scalar
    assert not tensorInstance.isequal(1)


def test_tensor__truediv__(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way

    # Tensor / Tensor
    tensor_div_tensor = tensorInstance / tensorInstance
    assert np.array_equal(tensor_div_tensor.data, (params["data"] / params["data"]))
    assert_consistent_order(tensorInstance, tensor_div_tensor.data)

    # Tensor / Sptensor
    tensor_div_sptensor = tensorInstance / tensorInstance.to_sptensor()
    assert np.array_equal(
        tensor_div_sptensor.data,
        (params["data"] / params["data"]),
    )
    assert_consistent_order(tensorInstance, tensor_div_sptensor.data)

    # Tensor / Scalar
    tensor_div_scalar = tensorInstance / 2
    assert np.array_equal(tensor_div_scalar.data, (params["data"] / 2))
    assert_consistent_order(tensorInstance, tensor_div_scalar.data)


def test_tensor__rtruediv__(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way

    # Scalar / Tensor, only resolves when left object doesn't have appropriate __mul__
    scalar_div_tensor = 2 / tensorInstance
    assert np.array_equal(scalar_div_tensor.data, (2 / params["data"]))
    assert_consistent_order(tensorInstance, scalar_div_tensor.data)


def test_tensor_nnz(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way

    # NNZ for full tensor
    assert tensorInstance.nnz == 6

    # NNZ for tensor with at least one zero
    tensorInstance[0, 0] = 0
    assert tensorInstance.nnz == 5


def test_tensor_reshape(sample_tensor_2way, sample_tensor_3way, sample_tensor_4way):
    (params2, tensorInstance2) = sample_tensor_2way

    # Reshape with tuple
    tensorInstance2 = tensorInstance2.reshape((3, 2))
    assert tensorInstance2.shape == (
        3,
        2,
    ), f"tensorInstance2.reshape((3, 2)): {tensorInstance2}"
    data = np.array([[1.0, 5.0], [4.0, 3.0], [2.0, 6.0]])
    assert np.array_equal(tensorInstance2.data, data)
    assert_consistent_order(tensorInstance2, tensorInstance2.data)

    with pytest.raises(AssertionError) as excinfo:
        tensorInstance2.reshape((3, 3))
    assert "Reshaping a tensor cannot change number of elements" in str(excinfo)

    (params3, tensorInstance3) = sample_tensor_3way
    tensorInstance3 = tensorInstance3.reshape((3, 2, 2))
    assert tensorInstance3.shape == (
        3,
        2,
        2,
    ), f"tensorInstance3.reshape((3, 2, 2)): {tensorInstance3}"
    data3 = np.array(
        [
            [[1.0, 7.0], [4.0, 10.0]],
            [[2.0, 8.0], [5.0, 11.0]],
            [[3.0, 9.0], [6.0, 12.0]],
        ]
    )
    assert np.array_equal(tensorInstance3.data, data3)

    (params4, tensorInstance4) = sample_tensor_4way
    tensorInstance4 = tensorInstance4.reshape((1, 3, 3, 9))
    assert tensorInstance4.shape == (
        1,
        3,
        3,
        9,
    ), f"tensorInstance4.reshape((1, 3, 3, 9)): {tensorInstance4}"
    data4 = np.array(
        [
            [
                [
                    [1, 10, 19, 28, 37, 46, 55, 64, 73],
                    [4, 13, 22, 31, 40, 49, 58, 67, 76],
                    [7, 16, 25, 34, 43, 52, 61, 70, 79],
                ],
                [
                    [2, 11, 20, 29, 38, 47, 56, 65, 74],
                    [5, 14, 23, 32, 41, 50, 59, 68, 77],
                    [8, 17, 26, 35, 44, 53, 62, 71, 80],
                ],
                [
                    [3, 12, 21, 30, 39, 48, 57, 66, 75],
                    [6, 15, 24, 33, 42, 51, 60, 69, 78],
                    [9, 18, 27, 36, 45, 54, 63, 72, 81],
                ],
            ]
        ]
    )
    assert np.array_equal(tensorInstance4.data, data4)


def test_tensor_permute(sample_tensor_2way, sample_tensor_3way, sample_tensor_4way):
    (params, tensorInstance) = sample_tensor_2way

    # Permute rows and columns
    permuted_tensor = tensorInstance.permute(np.array([1, 0]))
    assert np.array_equal(permuted_tensor.data, np.transpose(params["data"]))
    assert_consistent_order(tensorInstance, permuted_tensor.data)

    # len(order) != ndims
    with pytest.raises(AssertionError) as excinfo:
        tensorInstance.permute(np.array([1, 0, 2]))
    assert "Invalid permutation order" in str(excinfo)

    # Try to permute order-1 tensor
    assert np.array_equal(
        ttb.tensor(np.array([1, 2, 3, 4])).permute(np.array([1])).data,
        np.array([1, 2, 3, 4]),
    )

    # Empty order
    assert np.array_equal(
        ttb.tensor(np.array([])).permute(np.array([])).data, np.array([])
    )

    # 2-way
    (params2, tensorInstance2) = sample_tensor_2way
    tensorInstance2 = tensorInstance2.permute(np.array([1, 0]))
    assert tensorInstance2.shape == (
        3,
        2,
    ), f"tensorInstance2.permute(np.array([1, 0])): {tensorInstance2}"
    data2 = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    assert np.array_equal(tensorInstance2.data, data2)

    # 3-way
    (params3, tensorInstance3) = sample_tensor_3way
    tensorInstance3 = tensorInstance3.permute(np.array([2, 1, 0]))
    assert tensorInstance3.shape == (
        2,
        3,
        2,
    ), f"tensorInstance3.permute(np.array([2, 1, 0])): {tensorInstance3}"
    data3 = np.array(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]]
    )
    assert np.array_equal(tensorInstance3.data, data3)

    # 4-way
    (params4, tensorInstance4) = sample_tensor_4way
    tensorInstance4 = tensorInstance4.permute(np.array([3, 1, 2, 0]))
    assert tensorInstance4.shape == (
        3,
        3,
        3,
        3,
    ), f"tensorInstance4.permute(np.array([3, 1, 2, 0])): {tensorInstance4}"
    data4 = np.array(
        [
            [
                [[1, 2, 3], [10, 11, 12], [19, 20, 21]],
                [[4, 5, 6], [13, 14, 15], [22, 23, 24]],
                [[7, 8, 9], [16, 17, 18], [25, 26, 27]],
            ],
            [
                [[28, 29, 30], [37, 38, 39], [46, 47, 48]],
                [[31, 32, 33], [40, 41, 42], [49, 50, 51]],
                [[34, 35, 36], [43, 44, 45], [52, 53, 54]],
            ],
            [
                [[55, 56, 57], [64, 65, 66], [73, 74, 75]],
                [[58, 59, 60], [67, 68, 69], [76, 77, 78]],
                [[61, 62, 63], [70, 71, 72], [79, 80, 81]],
            ],
        ]
    )
    assert np.array_equal(tensorInstance4.data, data4)


def test_tensor_collapse(sample_tensor_2way, sample_tensor_3way, sample_tensor_4way):
    (params2, tensorInstance2) = sample_tensor_2way

    assert tensorInstance2.collapse() == 21
    assert tensorInstance2.collapse(fun=np.max) == 6

    (params3, tensorInstance3) = sample_tensor_3way

    assert tensorInstance3.collapse() == 78
    assert tensorInstance3.collapse(fun=np.max) == 12

    (params4, tensorInstance4) = sample_tensor_4way

    assert tensorInstance4.collapse() == 3321
    assert tensorInstance4.collapse(fun=np.max) == 81

    # single dimension collapse
    data = np.array([5, 7, 9])
    tensorCollapse = tensorInstance2.collapse(np.array([0]))
    assert np.array_equal(tensorCollapse.data, data)
    assert_consistent_order(tensorInstance2, tensorCollapse.data)

    # single dimension collapse using max function
    datamax = np.array([4, 5, 6])
    tensorCollapseMax = tensorInstance2.collapse(np.array([0]), fun=np.max)
    assert np.array_equal(tensorCollapseMax.data, datamax)

    # multiple dimensions collapse
    data4 = np.array([[99, 342, 585], [126, 369, 612], [153, 396, 639]])
    tensorCollapse4 = tensorInstance4.collapse(np.array([0, 2]))
    assert np.array_equal(tensorCollapse4.data, data4)

    # multiple dimensions collapse
    data4max = np.array([[21, 48, 75], [24, 51, 78], [27, 54, 81]])
    tensorCollapse4Max = tensorInstance4.collapse(np.array([0, 2]), fun=np.max)
    assert np.array_equal(tensorCollapse4Max.data, data4max)

    # Empty tensor collapse
    empty_data = np.array([])
    empty_tensor = ttb.tensor(empty_data)
    assert np.all(empty_tensor.collapse() == empty_data)

    # Empty dims
    assert tensorInstance2.collapse(empty_data).isequal(tensorInstance2)


def test_tensor_contract(sample_tensor_2way, sample_tensor_3way, sample_tensor_4way):
    (params2, tensorInstance2) = sample_tensor_2way

    with pytest.raises(AssertionError) as excinfo:
        tensorInstance2.contract(0, 1)
    assert "Must contract along equally sized dimensions" in str(excinfo)

    with pytest.raises(AssertionError) as excinfo:
        tensorInstance2.contract(0, 0)
    assert "Must contract along two different dimensions" in str(excinfo)

    contractableTensor = ttb.tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]), (3, 3))
    assert contractableTensor.contract(0, 1) == 15

    (params3, tensorInstance3) = sample_tensor_3way
    print("\ntensorInstance3.contract(0,2) = ")
    print(tensorInstance3.contract(0, 2))
    data3 = np.array([9, 13, 17])
    contracted_tensor = tensorInstance3.contract(0, 2)
    assert np.array_equal(contracted_tensor.data, data3)
    assert_consistent_order(tensorInstance3, contracted_tensor.data)

    (params4, tensorInstance4) = sample_tensor_4way
    print("\ntensorInstance4.contract(0,1) = ")
    print(tensorInstance4.contract(0, 1))
    data4 = np.array([[15, 96, 177], [42, 123, 204], [69, 150, 231]])
    assert np.array_equal(tensorInstance4.contract(0, 1).data, data4)

    print("\ntensorInstance4.contract(1,3) = ")
    print(tensorInstance4.contract(1, 3))
    data4a = np.array([[93, 120, 147], [96, 123, 150], [99, 126, 153]])
    assert np.array_equal(tensorInstance4.contract(1, 3).data, data4a)


def test_tensor__repr__(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way
    # Test that we can capture each of these
    str(tensorInstance)

    str(ttb.tensor(np.array([1, 2, 3])))

    str(ttb.tensor(np.arange(0, 81).reshape(3, 3, 3, 3)))

    str(ttb.tensor())


def test_tensor_exp(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way
    exp_tensor = tensorInstance.exp()
    assert np.array_equal(tensorInstance.exp().data, np.exp(params["data"]))
    assert_consistent_order(tensorInstance, exp_tensor.data)


def test_tensor_innerprod(sample_tensor_2way, sample_tensor_3way, sample_tensor_4way):
    (params, tensorInstance) = sample_tensor_2way
    # Tensor innerproduct
    tensor_innerprod = tensorInstance.innerprod(tensorInstance)
    assert tensor_innerprod == np.arange(1, 7).dot(np.arange(1, 7))

    # Sptensor innerproduct
    sptensor_innerprod = tensorInstance.innerprod(tensorInstance.to_sptensor())
    assert sptensor_innerprod == np.arange(1, 7).dot(np.arange(1, 7))

    # Wrong size innerproduct
    with pytest.raises(AssertionError) as excinfo:
        tensorInstance.innerprod(ttb.tensor(np.ones((4, 4))))
    assert "Inner product must be between tensors of the same size" in str(excinfo)

    # Wrong class innerproduct
    with pytest.raises(AssertionError) as excinfo:
        tensorInstance.innerprod(5)
    assert "Inner product between tensor and that class is not supported" in str(
        excinfo
    )

    # 2-way
    (params2, tensorInstance2) = sample_tensor_2way
    assert tensorInstance2.innerprod(tensorInstance2) == 91, (
        f"tensorInstance2.innerprod(tensorInstance2): {tensorInstance2.innerprod(tensorInstance2)}"
    )

    # 3-way
    (params3, tensorInstance3) = sample_tensor_3way
    assert tensorInstance3.innerprod(tensorInstance3) == 650, (
        f"tensorInstance3.innerprod(tensorInstance3): {tensorInstance3.innerprod(tensorInstance3)}"
    )

    # 4-way
    (params4, tensorInstance4) = sample_tensor_4way
    assert tensorInstance4.innerprod(tensorInstance4) == 180441, (
        f"tensorInstance4.innerprod(tensorInstance4): {tensorInstance4.innerprod(tensorInstance4)}"
    )


def test_tensor_mask(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way
    W = ttb.tensor(np.array([[0, 1, 0], [1, 0, 0]]))
    mask_result = tensorInstance.mask(W)
    assert np.array_equal(mask_result, np.array([4, 2]))
    assert_consistent_order(tensorInstance, mask_result)

    # Wrong shape mask
    with pytest.raises(AssertionError) as excinfo:
        tensorInstance.mask(ttb.tensor(np.ones((11, 3))))
    assert "Mask cannot be bigger than the data tensor" in str(excinfo)


def test_tensor_scale():
    T = ttb.tenones((3, 4, 5))
    S = np.arange(5, dtype=float)
    Y = T.scale(S, 2)
    assert np.array_equal(Y.data[0, 0, :], S)
    assert_consistent_order(T, Y.data)

    S = ttb.tensor(np.arange(5, dtype=float))
    Y = T.scale(S, 2)
    assert np.array_equal(Y.data[0, 0, :], S.data)

    S = ttb.tensor(np.arange(12, dtype=float), shape=(3, 4))
    Y = T.scale(S, [0, 1])
    assert np.array_equal(Y.data[:, :, 0], S.data)

    S = ttb.tensor(np.arange(60, dtype=float), shape=(3, 4, 5))
    Y = T.scale(S, [0, 1, 2])
    assert np.array_equal(Y.data, S.data)

    # Negative test
    with pytest.raises(ValueError):
        S = ttb.tensor(np.arange(60, dtype=float), shape=(3, 4, 5))
        Y = T.scale(S, 0)


def test_tensor_squeeze(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way

    # No singleton dimensions
    assert np.array_equal(tensorInstance.squeeze().data, params["data"])
    assert_consistent_order(tensorInstance, tensorInstance.squeeze().data)

    # All singleton dimensions
    squeeze_result = ttb.tensor(np.array([[[4]]])).squeeze()
    assert squeeze_result == 4
    assert np.isscalar(squeeze_result)

    # A singleton dimension
    assert np.array_equal(
        ttb.tensor(np.array([[1, 2, 3]])).squeeze().data, np.array([1, 2, 3])
    )


def test_tensor_ttm(sample_tensor_2way, sample_tensor_3way, sample_tensor_4way):
    (params2, tensorInstance2) = sample_tensor_2way
    (params3, tensorInstance3) = sample_tensor_3way
    (params4, tensorInstance4) = sample_tensor_4way

    M2 = np.reshape(np.arange(1, 2 * 2 + 1), [2, 2], order="F")
    M3 = np.reshape(np.arange(1, 3 * 3 + 1), [3, 3], order="F")

    # 3-way single matrix
    T3 = tensorInstance3.ttm(M2, 0)
    assert isinstance(T3, ttb.tensor)
    assert T3.shape == (2, 3, 2)
    data3 = np.array([[[7, 31], [15, 39], [23, 47]], [[10, 46], [22, 58], [34, 70]]])
    assert np.array_equal(T3.data, data3)
    assert_consistent_order(tensorInstance3, T3.data)

    # 3-way single matrix, transposed
    T3 = tensorInstance3.ttm(M2, 0, transpose=True)
    assert isinstance(T3, ttb.tensor)
    assert T3.shape == (2, 3, 2)
    data3 = np.array([[[5, 23], [11, 29], [17, 35]], [[11, 53], [25, 67], [39, 81]]])
    assert np.array_equal(T3.data, data3)

    # 3-way, two matrices, negative dimension
    T3 = tensorInstance3.ttm([M2, M2], exclude_dims=1)
    assert isinstance(T3, ttb.tensor)
    assert T3.shape == (2, 3, 2)
    data3 = np.array(
        [[[100, 138], [132, 186], [164, 234]], [[148, 204], [196, 276], [244, 348]]]
    )
    assert np.array_equal(T3.data, data3)

    # 3-way, two matrices, explicit dimensions
    T3 = tensorInstance3.ttm([M2, M3], [2, 1])
    assert isinstance(T3, ttb.tensor)
    assert T3.shape == (2, 3, 2)
    data3 = np.array(
        [[[408, 576], [498, 702], [588, 828]], [[456, 648], [558, 792], [660, 936]]]
    )
    assert np.array_equal(T3.data, data3)

    # 3-way, 3 matrices, no dimensions specified
    T3 = tensorInstance3.ttm([M2, M3, M2])
    assert isinstance(T3, ttb.tensor)
    assert T3.shape == (2, 3, 2)
    data3 = np.array(
        [
            [[1776, 2520], [2172, 3078], [2568, 3636]],
            [[2640, 3744], [3228, 4572], [3816, 5400]],
        ]
    )
    assert np.array_equal(T3.data, data3)

    # 3-way, matrix must be np.ndarray
    Tmat = ttb.tenmat(M2, rdims=np.array([0]))
    with pytest.raises(AssertionError) as excinfo:
        tensorInstance3.ttm(Tmat, 0)
    assert "matrix must be of type numpy.ndarray" in str(excinfo)

    # 3-way, dims must be in range [0,self.ndims]
    with pytest.raises(AssertionError) as excinfo:
        tensorInstance3.ttm(M2, tensorInstance3.ndims + 1)
    assert "dims must contain values in [0,self.dims)" in str(excinfo)


def test_tensor_ttt():
    M31 = ttb.tensor(np.reshape(np.arange(1, 2 * 3 * 4 + 1), [4, 3, 2], order="F"))
    M32 = ttb.tensor(np.reshape(np.arange(1, 2 * 3 * 4 + 1), [3, 4, 2], order="F"))

    # outer product of M31 and M32
    TTT1 = M31.ttt(M32)
    assert TTT1.shape == (4, 3, 2, 3, 4, 2)
    # choose two random 2-way slices
    data11 = np.array([1, 2, 3, 4])
    data12 = np.array([289, 306, 323, 340])
    data13 = np.array([504, 528, 552, 576])
    assert np.array_equal(TTT1[:, 0, 0, 0, 0, 0].data, data11)
    assert np.array_equal(TTT1[:, 1, 1, 1, 1, 1].data, data12)
    assert np.array_equal(TTT1[:, 2, 1, 2, 3, 1].data, data13)
    assert_consistent_order(M31, TTT1.data)

    TTT1_with_dims = M31.ttt(
        M31, selfdims=np.array([0, 1, 2]), otherdims=np.array([0, 1, 2])
    )
    assert np.allclose(TTT1_with_dims, M31.innerprod(M31))

    # Negative tests
    with pytest.raises(AssertionError):
        invalid_tensor_type = []
        M31.ttt(invalid_tensor_type)

    with pytest.raises(AssertionError):
        M31.ttt(M31, selfdims=np.array([0, 1, 2]), otherdims=np.array([0, 2, 1]))

    M2 = ttb.tensor(np.reshape(np.arange(0, 2), [1, 2], order="F"))
    result = M2.ttt(M2, 0, 0)
    row_vector = M2.data
    column_vector = M2.data.transpose()
    assert np.allclose(result.data, row_vector * column_vector)


def test_tensor_ttv(sample_tensor_2way, sample_tensor_3way, sample_tensor_4way):
    (params2, tensorInstance2) = sample_tensor_2way
    (params3, tensorInstance3) = sample_tensor_3way
    (params4, tensorInstance4) = sample_tensor_4way

    # Wrong shape vector
    with pytest.raises(AssertionError) as excinfo:
        tensorInstance2.ttv(np.array([np.array([1, 2]), np.array([1, 2])]))
    assert "Multiplicand is wrong size" in str(excinfo)

    # 2-way Multiply by single vector
    T2 = tensorInstance2.ttv(np.array([2, 2]), 0)
    assert isinstance(T2, ttb.tensor)
    assert T2.shape == (3,)
    assert np.array_equal(T2.data, np.array([10, 14, 18]))
    assert_consistent_order(tensorInstance2, T2.data)

    # 2-way Multiply by single vector (exclude dims)
    T2 = tensorInstance2.ttv(np.array([2, 2]), exclude_dims=1)
    assert isinstance(T2, ttb.tensor)
    assert T2.shape == (3,)
    assert np.array_equal(T2.data, np.array([10, 14, 18]))

    # Multiply by multiple vectors, infer dimensions
    assert tensorInstance2.ttv([np.array([2, 2]), np.array([1, 1, 1])]) == 42

    # Multiply by multiple vectors as list of numpy.ndarrays
    assert tensorInstance2.ttv([np.array([2, 2]), np.array([1, 1, 1])]) == 42

    # 3-way Multiply by single vector
    T3 = tensorInstance3.ttv(2 * np.ones((tensorInstance3.shape[0],)), 0)
    assert isinstance(T3, ttb.tensor)
    assert T3.shape == (tensorInstance3.shape[1], tensorInstance3.shape[2])
    assert np.array_equal(T3.data, np.array([[6, 30], [14, 38], [22, 46]]))

    # Multiply by multiple vectors, infer dimensions
    assert (
        tensorInstance3.ttv([np.array([2, 2]), np.array([1, 1, 1]), np.array([2, 2])])
        == 312
    )

    # 4-way Multiply by single vector
    T4 = tensorInstance4.ttv(2 * np.ones((tensorInstance4.shape[0],)), 0)
    assert isinstance(T4, ttb.tensor)
    assert T4.shape == (
        tensorInstance4.shape[1],
        tensorInstance4.shape[2],
        tensorInstance4.shape[3],
    )

    # 4-way Multiply by single vector (exclude dims)
    T4 = tensorInstance4.ttv(
        2 * np.ones((tensorInstance4.shape[0],)), exclude_dims=np.array([1, 2, 3])
    )
    assert isinstance(T4, ttb.tensor)
    assert T4.shape == (
        tensorInstance4.shape[1],
        tensorInstance4.shape[2],
        tensorInstance4.shape[3],
    )

    data4 = np.array(
        [
            [[12, 174, 336], [66, 228, 390], [120, 282, 444]],
            [[30, 192, 354], [84, 246, 408], [138, 300, 462]],
            [[48, 210, 372], [102, 264, 426], [156, 318, 480]],
        ]
    )
    assert np.array_equal(T4.data, data4)

    # Multiply by multiple vectors, infer dimensions
    assert (
        tensorInstance4.ttv(
            np.array(
                [
                    np.array([1, 1, 1]),
                    np.array([1, 1, 1]),
                    np.array([1, 1, 1]),
                    np.array([1, 1, 1]),
                ]
            )
        )
        == 3321
    )


def test_tensor_ttsv(sample_tensor_4way):
    # 3-way
    tensorInstance3 = ttb.tensor(np.ones((4, 4, 4)))
    vector3 = np.array([4, 3, 2, 1])
    assert tensorInstance3.ttsv(vector3, version=1) == 1000
    assert np.array_equal(
        tensorInstance3.ttsv(vector3, skip_dim=0, version=1), 100 * np.ones((4,))
    )
    ttsv_result = tensorInstance3.ttsv(vector3, skip_dim=1, version=1)
    assert np.array_equal(ttsv_result, 10 * np.ones((4, 4)))
    assert_consistent_order(tensorInstance3, ttsv_result)

    # Invalid dims
    with pytest.raises(ValueError) as excinfo:
        tensorInstance3.ttsv(vector3, skip_dim=-1)
    assert "Invalid modes in ttsv" in str(excinfo)

    # 4-way tensor
    (params4, tensorInstance4) = sample_tensor_4way
    T4ttsv = tensorInstance4.ttsv(np.array([1, 2, 3]), 2, version=1)
    data4_3 = np.array(
        [
            [[222, 276, 330], [240, 294, 348], [258, 312, 366]],
            [[228, 282, 336], [246, 300, 354], [264, 318, 372]],
            [[234, 288, 342], [252, 306, 360], [270, 324, 378]],
        ]
    )
    assert np.array_equal(T4ttsv.data, data4_3)

    # 5-way dense tensor
    shape = (3, 3, 3, 3, 3)
    T5 = ttb.tensor(np.arange(1, np.prod(shape) + 1), shape)
    T5ttsv = T5.ttsv(np.array([1, 2, 3]), 2, version=1)
    data5_3 = np.array(
        [
            [[5220, 5544, 5868], [5328, 5652, 5976], [5436, 5760, 6084]],
            [[5256, 5580, 5904], [5364, 5688, 6012], [5472, 5796, 6120]],
            [[5292, 5616, 5940], [5400, 5724, 6048], [5508, 5832, 6156]],
        ]
    )
    assert np.array_equal(T5ttsv.data, data5_3)

    # Test new algorithm, version=2

    # 3-way
    assert tensorInstance3.ttsv(vector3) == 1000
    assert np.array_equal(tensorInstance3.ttsv(vector3, 0), 100 * np.ones((4,)))
    assert np.array_equal(tensorInstance3.ttsv(vector3, 1), 10 * np.ones((4, 4)))

    # 4-way tensor
    T4ttsv2 = tensorInstance4.ttsv(np.array([1, 2, 3]), 2)
    assert np.array_equal(T4ttsv2.data, data4_3)

    # Incorrect version requested
    with pytest.raises(AssertionError) as excinfo:
        tensorInstance4.ttsv(np.array([1, 2, 3]), 2, version=3)
    assert "Invalid value for version; should be None, 1, or 2" in str(excinfo)


def test_tensor_issymmetric(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way

    assert tensorInstance.issymmetric() is False
    assert tensorInstance.issymmetric(version=1) is False

    symmetricData = np.array(
        [
            [[0.5, 0, 0.5, 0], [0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 2.5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0.5, 0, 0, 0], [0, 0, 0, 0], [0, 0, 3.5, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ]
    )
    symmetricTensor = ttb.tensor(symmetricData)
    assert symmetricTensor.issymmetric() is True
    assert symmetricTensor.issymmetric(version=1) is True
    answer, diffs, perms = symmetricTensor.issymmetric(version=1, return_details=True)
    assert answer is True
    assert np.all(diffs == 0)
    # Ensure we return details even if old version not requested
    answer, diffs, perms = symmetricTensor.issymmetric(return_details=True)
    assert answer is True
    assert np.all(diffs == 0)

    symmetricData[3, 1, 0] = 3
    symmetricTensor = ttb.tensor(symmetricData)
    assert symmetricTensor.issymmetric() is False
    assert symmetricTensor.issymmetric(version=1) is False


def test_tensor_symmetrize(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way

    # Test new default version

    # 2-way
    symmetricData = np.array(
        [
            [[0.5, 0, 0.5, 0], [0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 2.5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0.5, 0, 0, 0], [0, 0, 0, 0], [0, 0, 3.5, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ]
    )
    symmetricTensor = ttb.tensor(symmetricData)
    symmetrized_tensor = symmetricTensor.symmetrize()
    assert symmetrized_tensor.isequal(symmetricTensor)
    assert_consistent_order(symmetricTensor, symmetrized_tensor.data)

    # 3-way
    symmetricData = np.zeros((4, 4, 4))
    symmetricData[1, 2, 1] = 1
    symmetricTensor = ttb.tensor(symmetricData)
    print(f"\nsymmetricTensor:\n{symmetricTensor}")
    assert symmetricTensor.issymmetric() is False
    print(f"\nsymmetricTensor.symmetrize():\n{symmetricTensor.symmetrize()}")
    assert (symmetricTensor.symmetrize()).issymmetric()

    # 3-way
    shape = (2, 2, 2)
    T3 = ttb.tensor(np.arange(1, np.prod(shape) + 1), shape)
    T3sym = T3.symmetrize()
    print("\nT3sym:")
    print(T3sym)
    data3 = np.array(
        [
            [[1, 3 + 1 / 3], [3 + 1 / 3, 5 + 2 / 3]],
            [[3 + 1 / 3, 5 + 2 / 3], [5 + 2 / 3, 8]],
        ]
    )
    assert np.array_equal(T3sym.data, data3)

    # T3syms_2_1_3 = T3.symmetrize(grps=[[1], [0,2]])
    # print(f'\nT3syms_2_1_3:')
    # print(T3syms_2_1_3)

    with pytest.raises(AssertionError) as excinfo:
        symmetricTensor.symmetrize(grps=np.array([[0, 1], [1, 2]]))
    assert "Cannot have overlapping symmetries" in str(excinfo)

    # Improper shape tensor for symmetry
    asymmetricData = np.zeros((5, 4, 6))
    asymmetricTensor = ttb.tensor(asymmetricData)
    with pytest.raises(AssertionError) as excinfo:
        asymmetricTensor.symmetrize()
    assert "Dimension mismatch for symmetrization" in str(excinfo)

    # Test older keyword version
    symmetricData = np.array(
        [
            [[0.5, 0, 0.5, 0], [0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 2.5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0.5, 0, 0, 0], [0, 0, 0, 0], [0, 0, 3.5, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ]
    )

    symmetricTensor = ttb.tensor(symmetricData)
    assert symmetricTensor.symmetrize(version=1).isequal(symmetricTensor)

    symmetricData = np.zeros((4, 4, 4))
    symmetricData[1, 2, 1] = 1
    symmetricTensor = ttb.tensor(symmetricData)
    assert symmetricTensor.issymmetric() is False
    assert (symmetricTensor.symmetrize(version=1)).issymmetric()

    with pytest.raises(AssertionError) as excinfo:
        symmetricTensor.symmetrize(grps=np.array([[0, 1], [1, 2]]), version=1)
    assert "Cannot have overlapping symmetries" in str(excinfo)

    # Improper shape tensor for symmetry
    asymmetricData = np.zeros((5, 4, 6))
    asymmetricTensor = ttb.tensor(asymmetricData)
    with pytest.raises(AssertionError) as excinfo:
        asymmetricTensor.symmetrize(version=1)
    assert "Dimension mismatch for symmetrization" in str(excinfo)


def test_tensor__str__():
    # Test 1D
    data = np.random.normal(size=(4,))
    tensorInstance = ttb.tensor(data)
    s = ""
    s += f"tensor of shape {tensorInstance.shape} with order F"
    s += "\ndata"
    s += "[:] =\n"
    s += data.__str__()
    assert s == tensorInstance.__str__()

    # Test 2D
    data = np.random.normal(size=(4, 3))
    tensorInstance = ttb.tensor(data)
    s = ""
    s += f"tensor of shape {tensorInstance.shape} with order F"
    s += "\ndata"
    s += "[:, :] =\n"
    s += data.__str__()
    assert s == tensorInstance.__str__()

    # Test 3D,shape in decreasing and increasing order
    data = np.random.normal(size=(4, 3, 2))
    tensorInstance = ttb.tensor(data)
    s = ""
    s += f"tensor of shape {tensorInstance.shape} with order F"
    for i in range(data.shape[-1]):
        s += "\ndata"
        s += f"[:, :, {i}] =\n"
        s += data[:, :, i].__str__()
    assert s == tensorInstance.__str__()

    data = np.random.normal(size=(2, 3, 4))
    tensorInstance = ttb.tensor(data)
    s = ""
    s += f"tensor of shape {tensorInstance.shape} with order F"
    for i in range(data.shape[-1]):
        s += "\ndata"
        s += f"[:, :, {i}] =\n"
        s += data[:, :, i].__str__()
    assert s == tensorInstance.__str__()

    # Test 4D
    data = np.random.normal(size=(4, 4, 3, 2))
    tensorInstance = ttb.tensor(data)
    s = ""
    s += f"tensor of shape {tensorInstance.shape} with order F"
    for i in range(data.shape[-1]):
        for j in range(data.shape[-2]):
            s += "\ndata"
            s += f"[:, :, {j}, {i}] =\n"
            s += data[:, :, j, i].__str__()
    assert s == tensorInstance.__str__()

    # Test 5D
    data = np.random.normal(size=(2, 2, 2, 2, 2))
    tensorInstance = ttb.tensor(data)
    s = ""
    s += f"tensor of shape {tensorInstance.shape} with order F"
    for i in range(data.shape[-1]):
        for j in range(data.shape[-2]):
            for k in range(data.shape[-3]):
                s += "\ndata"
                s += f"[:, :, {k}, {j}, {i}] =\n"
                s += data[:, :, k, j, i].__str__()
    assert s == tensorInstance.__str__()


def test_tensor_mttkrp(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way
    tensorInstance = ttb.tensor.from_function(np.ones, (2, 3, 4))

    # 2-way sparse tensor
    weights = np.array([2.0, 2.0])
    fm0 = np.array([[1.0, 3.0], [2.0, 4.0]])
    fm1 = np.array([[5.0, 8.0], [6.0, 9.0], [7.0, 10.0]])
    fm2 = np.array([[11.0, 15.0], [12.0, 16.0], [13.0, 17.0], [14.0, 18.0]])
    factor_matrices = [fm0, fm1, fm2]
    ktensorInstance = ttb.ktensor(factor_matrices, weights)

    m0 = np.array([[1800.0, 3564.0], [1800.0, 3564.0]])
    m1 = np.array([[300.0, 924.0], [300.0, 924.0], [300.0, 924.0]])
    m2 = np.array([[108.0, 378.0], [108.0, 378.0], [108.0, 378.0], [108.0, 378.0]])
    mttkrp_0 = tensorInstance.mttkrp(ktensorInstance, 0)
    assert np.allclose(mttkrp_0, m0)
    assert_consistent_order(tensorInstance, mttkrp_0)

    mttkrp_1 = tensorInstance.mttkrp(ktensorInstance, 1)
    assert np.allclose(mttkrp_1, m1)
    assert_consistent_order(tensorInstance, mttkrp_1)

    mttkrp_2 = tensorInstance.mttkrp(ktensorInstance, 2)
    assert np.allclose(mttkrp_2, m2)
    assert_consistent_order(tensorInstance, mttkrp_2)

    # 5-way dense tensor
    shape = (2, 3, 4, 5, 6)
    T = ttb.tensor(np.arange(1, np.prod(shape) + 1), shape)
    U = []
    for s in shape:
        U.append(np.ones((s, 2)))

    data0 = np.array([[129600, 129600], [129960, 129960]])
    assert np.array_equal(T.mttkrp(U, 0), data0)

    data1 = np.array([[86040, 86040], [86520, 86520], [87000, 87000]])
    assert np.array_equal(T.mttkrp(U, 1), data1)

    data2 = np.array([[63270, 63270], [64350, 64350], [65430, 65430], [66510, 66510]])
    assert np.array_equal(T.mttkrp(U, 2), data2)

    data3 = np.array(
        [[45000, 45000], [48456, 48456], [51912, 51912], [55368, 55368], [58824, 58824]]
    )
    assert np.array_equal(T.mttkrp(U, 3), data3)

    data4 = np.array(
        [
            [7260, 7260],
            [21660, 21660],
            [36060, 36060],
            [50460, 50460],
            [64860, 64860],
            [79260, 79260],
        ]
    )
    assert np.array_equal(T.mttkrp(U, 4), data4)

    # tensor too small
    with pytest.raises(AssertionError) as excinfo:
        tensorInstance2 = ttb.tensor(np.array([1]))
        tensorInstance2.mttkrp([], 0)
    assert "MTTKRP is invalid for tensors with fewer than 2 dimensions" in str(excinfo)

    # second argument not a ktensor or list
    with pytest.raises(AssertionError) as excinfo:
        tensorInstance.mttkrp(5, 0)
    assert "Second argument must be a sequence of numpy.ndarray's or a ktensor" in str(
        excinfo
    )

    # second argument list is not the correct length
    with pytest.raises(AssertionError) as excinfo:
        m0 = np.ones((2, 2))
        tensorInstance.mttkrp([m0, m0, m0, m0], 0)
    assert "List of factor matrices is the wrong length" in str(excinfo)

    # arrays not the correct shape
    with pytest.raises(AssertionError) as excinfo:
        m0 = np.ones((2, 2))
        m1 = np.ones((3, 2))
        m2 = np.ones((5, 2))
        tensorInstance.mttkrp([m0, m1, m2], 0)
    assert "Entry 2 of list of arrays is wrong size" in str(excinfo)


def test_tensor_nvecs(sample_tensor_2way):
    (data, tensorInstance) = sample_tensor_2way

    nv1 = np.array([[0.4286671335486261, 0.5663069188480352, 0.7039467041474443]]).T
    nv2 = np.array(
        [
            [0.4286671335486261, 0.5663069188480352, 0.7039467041474443],
            [0.8059639085892916, 0.1123824140966059, -0.5811990803961161],
        ]
    ).T

    # Test for one eigenvector
    nvecs_tensor = tensorInstance.nvecs(1, 1)
    assert np.allclose(nvecs_tensor, nv1)
    assert_consistent_order(tensorInstance, nvecs_tensor)

    # Test for r >= N-1, requires cast to dense
    nvecs_tensor = tensorInstance.nvecs(1, 2)
    assert np.allclose(nvecs_tensor, nv2)
    assert_consistent_order(tensorInstance, nvecs_tensor)


def test_tenones():
    arbitrary_shape = (3, 3, 3)
    ones_tensor = ttb.tenones(arbitrary_shape)
    data_tensor = ttb.tensor(np.ones(arbitrary_shape))
    assert np.equal(ones_tensor, data_tensor), "Tenones should match all ones tensor"
    assert_consistent_order(data_tensor, ones_tensor.data)


def test_tenzeros():
    arbitrary_shape = (3, 3, 3)
    zeros_tensor = ttb.tenzeros(arbitrary_shape)
    data_tensor = ttb.tensor(np.zeros(arbitrary_shape))
    assert np.equal(zeros_tensor, data_tensor), "Tenzeros should match all zeros tensor"
    assert_consistent_order(data_tensor, zeros_tensor.data)


def test_tenrand():
    arbitrary_shape = (3, 3, 3)
    rand_tensor = ttb.tenrand(arbitrary_shape)
    in_unit_interval = np.all((rand_tensor >= 0).data) and np.all(
        (rand_tensor <= 1).data
    )
    assert in_unit_interval and rand_tensor.shape == arbitrary_shape
    assert_consistent_order(rand_tensor, rand_tensor.data)


def test_tendiag():
    N = 4
    elements = np.arange(0, N)
    exact_shape = [N] * N

    # Inferred shape
    X = ttb.tendiag(elements)
    for i in range(N):
        diag_index = (i,) * N
        assert X[diag_index] == i

    # Exact shape
    X = ttb.tendiag(elements, tuple(exact_shape))
    for i in range(N):
        diag_index = (i,) * N
        assert X[diag_index] == i

    # Larger shape
    larger_shape = exact_shape.copy()
    larger_shape[0] += 1
    X = ttb.tendiag(elements, tuple(larger_shape))
    for i in range(N):
        diag_index = (i,) * N
        assert X[diag_index] == i

    # Smaller Shape
    smaller_shape = exact_shape.copy()
    smaller_shape[0] -= 1
    X = ttb.tendiag(elements, tuple(smaller_shape))
    for i in range(N):
        diag_index = (i,) * N
        assert X[diag_index] == i
    assert_consistent_order(X, X.data)


def test_teneye():
    with pytest.raises(ValueError):
        ttb.teneye(1, 0)
    size = 5
    order = 4
    T = ttb.teneye(order, size)
    x = np.random.random((size,))
    x = x / np.linalg.norm(x)
    np.testing.assert_almost_equal(T.ttsv(x, 0), x)
    assert_consistent_order(T, T.data)


def test_mttv_left():
    m1 = 2
    mi = [range(1, 4)]
    C = 5
    U = np.ones((m1, C))
    W = np.ones((m1 * np.prod(mi), C))
    W_out = mttv_left(W, U)
    assert W_out.shape == (np.prod(mi), C)


def test_mttv_mid():
    m1 = 2
    mi = list(range(1, 4))
    C = 5
    U = [np.ones((m, C)) for m in mi]
    W = np.ones((m1 * np.prod(mi), C))
    W_out = mttv_mid(W, U)
    assert W_out.shape == (m1, C)

    W_out = mttv_mid(W, [])
    assert W_out is W


def test_min_split():
    shape = (3, 3, 3, 3)
    idx = min_split(shape)
    assert idx == 1


def test_mttkrps():
    model = ttb.ktensor([k * np.random.random((k, 2)) for k in (2, 3, 4)])
    data = ttb.tenrand(model.shape)
    direct = list(data.mttkrp(model.factor_matrices, k) for k in range(data.ndims))

    optimized = data.mttkrps(model.factor_matrices)
    assert all(
        np.allclose(a_direct, an_optimized)
        for a_direct, an_optimized in zip(direct, optimized)
    )

    # Using ktensor directly
    optimized = data.mttkrps(model)
    assert all(
        np.allclose(a_direct, an_optimized)
        for a_direct, an_optimized in zip(direct, optimized)
    )

    # More dims to hit various combinations and iterations
    model = ttb.ktensor([np.ones((k, 2)) for k in (2, 2, 2, 2, 2, 2)])
    data = ttb.tenones(model.shape)
    direct = list(data.mttkrp(model.factor_matrices, k) for k in range(data.ndims))
    optimized = data.mttkrps(model)
    assert all(
        np.allclose(a_direct, an_optimized)
        for a_direct, an_optimized in zip(direct, optimized)
    )


def test_tenfun(memory_layout):
    order = memory_layout["order"]
    data = np.array([[1, 2, 3], [4, 5, 6]])
    t1 = ttb.tensor(data)
    t2 = ttb.tensor(data)

    # Binary case
    def add(x, y):
        return x + y

    tenfun_result = t1.tenfun(add, t2)
    assert np.array_equal(tenfun_result.data, 2 * data)
    assert_consistent_order(t1, tenfun_result.data)

    # Single argument case
    def add1(x):
        return x + 1

    assert np.array_equal(t1.tenfun(add1).data, (data + 1))

    # Multi argument case
    def tensor_max(x):
        return np.max(x, axis=0).astype(np.float64, order=order)

    assert np.array_equal(t1.tenfun(tensor_max, t1, t1).data, data)

    # No np array case
    assert np.array_equal(t1.tenfun(tensor_max, data, data).data, data)

    # No list case
    with pytest.raises(AssertionError) as excinfo:
        t1.tenfun(tensor_max, [1, 2, 3])
    assert "Invalid input to ten fun" in str(excinfo)

    # Scalar argument not in first two positions
    with pytest.raises(AssertionError) as excinfo:
        t1.tenfun(tensor_max, t1, 1)
    assert "Invalid input to ten fun" in str(excinfo)

    # Tensors of different sizes
    with pytest.raises(AssertionError) as excinfo:
        t1.tenfun(
            tensor_max,
            t1,
            ttb.tensor(np.concatenate((data, np.array([[7, 8, 9]])))),
        )
    assert "Tensor 1 is not the same size as the first tensor input" in str(excinfo)

    with pytest.raises(ValueError) as excinfo:

        def three_arg_function(x, y, z):
            pass

        t1.tenfun(three_arg_function)
    assert "only supports binary and unary function handles" in str(excinfo)

    with pytest.raises(AssertionError) as excinfo:
        _ = t1.tenfun_unary(add1, 1)
    assert "scalar but expected a tensor" in str(excinfo)
