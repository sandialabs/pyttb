# Copyright 2022 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

import pyttb as ttb
import numpy as np
import pytest

DEBUG_tests = False

@pytest.fixture()
def sample_tensor_2way():
    data = np.array([[1., 2., 3.], [4., 5., 6.]])
    shape = (2, 3)
    params = {'data':data, 'shape': shape}
    tensorInstance = ttb.tensor().from_data(data, shape)
    return params, tensorInstance

@pytest.fixture()
def sample_tensor_3way():
    data = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.])
    shape = (2, 3, 2)
    params = {'data':np.reshape(data, np.array(shape), order='F'), 'shape': shape}
    tensorInstance = ttb.tensor().from_data(data, shape)
    return params, tensorInstance

@pytest.fixture()
def sample_tensor_4way():
    data = np.arange(1, 82)
    shape = (3, 3, 3, 3)
    params = {'data':np.reshape(data, np.array(shape), order='F'), 'shape': shape}
    tensorInstance = ttb.tensor().from_data(data, shape)
    return params, tensorInstance

@pytest.mark.indevelopment
def test_tensor_initialization_empty():
    empty = np.array([])

    # No args
    tensorInstance = ttb.tensor()
    assert (tensorInstance.data == empty).all()
    assert (tensorInstance.shape == ())

@pytest.mark.indevelopment
def test_tensor_initialization_from_data(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way
    assert (tensorInstance.data == params['data']).all()
    assert (tensorInstance.shape == params['shape'])

    with pytest.raises(AssertionError) as excinfo:
        a = ttb.tensor.from_data(params['data'], ())
    assert "Empty tensor cannot contain any elements" in str(excinfo)

    with pytest.raises(AssertionError) as excinfo:
        a = ttb.tensor.from_data(params['data'], (2, 4))
    assert "TTB:WrongSize, Size of data does not match specified size of tensor" in str(excinfo)

    with pytest.raises(AssertionError) as excinfo:
        a = ttb.tensor.from_data(params['data'], np.array([2, 3]))
    assert "Second argument must be a tuple." in str(excinfo)

    # TODO how else to break this logical statement?
    data = np.array([['a', 2, 3], [4, 5, 6]])
    with pytest.raises(AssertionError) as excinfo:
        a = ttb.tensor.from_data(data, (2, 3))
    assert "First argument must be a multidimensional array." in str(excinfo)

    # 1D tensors
    # no shape spaecified
    tensorInstance1 = ttb.tensor.from_data(np.array([1, 2, 3]))
    if DEBUG_tests:
        print('\ntensorInstance1:')
        print(tensorInstance1)
    data = np.array([1, 2, 3])
    assert tensorInstance1.data.shape == data.shape
    assert (tensorInstance1.data == data).all()

    # shape is 1 x 3
    tensorInstance1 = ttb.tensor.from_data(np.array([1, 2, 3]), (1,3))
    if DEBUG_tests:
        print('\ntensorInstance1:')
        print(tensorInstance1)
    data = np.array([[1, 2, 3]])
    assert tensorInstance1.data.shape == data.shape
    assert (tensorInstance1.data == data).all()

    # shape is 3 x 1
    tensorInstance1 = ttb.tensor.from_data(np.array([1, 2, 3]), (3,1))
    if DEBUG_tests:
        print('\ntensorInstance1:')
        print(tensorInstance1)
    data = np.array([[1], 
                     [2], 
                     [3]])
    assert tensorInstance1.data.shape == data.shape
    assert (tensorInstance1.data == data).all()


@pytest.mark.indevelopment
def test_tensor_initialization_from_tensor_type(sample_tensor_2way, sample_tensor_4way):
    (params, tensorInstance) = sample_tensor_2way
    (_, tensorInstance4) = sample_tensor_4way

    # Copy Constructor
    tensorCopy = ttb.tensor.from_tensor_type(tensorInstance)
    assert (tensorCopy.data == params['data']).all()
    assert (tensorCopy.shape == params['shape'])

    subs = np.array([[0, 0], [0, 1], [0, 2], [1, 0]])
    vals = np.array([[1], [2], [3], [4]])
    shape = (2, 3)
    data = np.array([[1, 2, 3], [4, 0, 0]])
    a = ttb.sptensor.from_data(subs, vals, shape)

    # Sptensor
    b = ttb.tensor.from_tensor_type(a)
    assert (b.data == data).all()
    assert (b.shape == shape)

    # tenmat
    tenmatInstance = ttb.tenmat.from_tensor_type(tensorInstance, np.array([0]))
    tensorTenmatInstance = ttb.tensor.from_tensor_type(tenmatInstance)
    assert tensorInstance.isequal(tensorTenmatInstance)

    # 1D 1-element tenmat
    tensorInstance1 = ttb.tensor.from_data(np.array([3]))
    tenmatInstance1 = ttb.tenmat.from_tensor_type(tensorInstance1, np.array([0]))
    tensorTenmatInstance1 = ttb.tensor.from_tensor_type(tenmatInstance1)
    assert tensorInstance1.isequal(tensorTenmatInstance1)

    # 4D tenmat
    tenmatInstance4 = ttb.tenmat.from_tensor_type(tensorInstance4, np.array([3,0]))
    tensorTenmatInstance4 = ttb.tensor.from_tensor_type(tenmatInstance4)
    assert tensorInstance4.isequal(tensorTenmatInstance4)

@pytest.mark.indevelopment
def test_tensor_initialization_from_function():
    def function_handle(x):
        return np.array([[1, 2, 3], [4, 5, 6]])
    shape = (2, 3)
    data = np.array([[1, 2, 3], [4, 5, 6]])

    a = ttb.tensor.from_function(function_handle, shape)
    assert (a.data == data).all()
    assert (a.shape == shape)

    with pytest.raises(AssertionError) as excinfo:
        ttb.tensor.from_function(function_handle, [2, 3])
    assert 'TTB:BadInput, Shape must be a tuple' in str(excinfo)

@pytest.mark.indevelopment
def test_tensor_find(sample_tensor_2way, sample_tensor_3way, sample_tensor_4way):
    (params, tensorInstance) = sample_tensor_2way
    subs, vals = tensorInstance.find()
    if DEBUG_tests: 
        print('\nsubs:')
        print(subs)
        print('\nvals:')
        print(vals)
    a = ttb.tensor.from_tensor_type(ttb.sptensor.from_data(subs, vals, tensorInstance.shape))
    assert (a.data == tensorInstance.data).all()
    assert (a.shape == tensorInstance.shape)

    (params, tensorInstance) = sample_tensor_3way
    subs, vals = tensorInstance.find()
    if DEBUG_tests: 
        print('\nsubs:')
        print(subs)
        print('\nvals:')
        print(vals)
    a = ttb.tensor.from_tensor_type(ttb.sptensor.from_data(subs, vals, tensorInstance.shape))
    assert (a.data == tensorInstance.data).all()
    assert (a.shape == tensorInstance.shape)

    (params, tensorInstance) = sample_tensor_4way
    subs, vals = tensorInstance.find()
    if DEBUG_tests: 
        print('\nsubs:')
        print(subs)
        print('\nvals:')
        print(vals)
    a = ttb.tensor.from_tensor_type(ttb.sptensor.from_data(subs, vals, tensorInstance.shape))
    assert (a.data == tensorInstance.data).all()
    assert (a.shape == tensorInstance.shape)


@pytest.mark.indevelopment
def test_tensor_ndims(sample_tensor_2way, sample_tensor_3way, sample_tensor_4way):
    (params, tensorInstance) = sample_tensor_2way
    assert tensorInstance.ndims == len(params['shape'])

    (params, tensorInstance) = sample_tensor_3way
    assert tensorInstance.ndims == len(params['shape'])

    (params, tensorInstance) = sample_tensor_4way
    assert tensorInstance.ndims == len(params['shape'])

    # Empty tensor has zero dimensions
    assert ttb.tensor.from_data(np.array([])) == 0

@pytest.mark.indevelopment
def test_tensor_setitem(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way

    # Subtensor assign with constant
    dataCopy = params['data'].copy()
    dataCopy[1, 1] = 0.0
    tensorInstance[1, 1] = 0.0
    assert (tensorInstance.data == dataCopy).all()
    dataGrowth = np.zeros((5, 5))
    dataGrowth[0:2, 0:3] = dataCopy
    tensorInstance[4, 4] = 0.0
    assert (tensorInstance.data == dataGrowth).all()
    tensorInstance[0:2, 0:2] = 99.0
    dataGrowth[0:2, 0:2] = 99.0
    assert (tensorInstance.data == dataGrowth).all()

    #Subtensor assign with np array
    tensorInstance[0:2, 0:3] = dataCopy
    dataGrowth[0:2, 0:3] = dataCopy
    assert (tensorInstance.data == dataGrowth).all()

    # Subtensor assign with tensor
    tensorInstance[:, :] = tensorInstance
    assert (tensorInstance.data == dataGrowth).all()


    # Subscripts with constant
    tensorInstance[np.array([[1, 1]])] = 13.0
    dataGrowth[1, 1] = 13.0
    assert (tensorInstance.data == dataGrowth).all()

    # Subscripts with array
    tensorVector = ttb.tensor.from_data(np.array([0, 0, 0, 0]))
    tensorVector[np.array([0, 1, 2])] = np.array([3, 4, 5])
    assert (tensorVector.data == np.array([3, 4, 5, 0])).all()


    # Subscripts with constant
    tensorInstance[np.array([[1, 1], [1, 2]])] = 13.0
    dataGrowth[([1, 1], [1, 2])] = 13.0
    assert (tensorInstance.data == dataGrowth).all()

    # Linear Index with constant
    tensorInstance[np.array([0])] = 13.0
    dataGrowth[0, 0] = 13.0
    assert (tensorInstance.data == dataGrowth).all()

    # Linear Index with constant
    tensorInstance[np.array([0, 3, 4])] = 13.0
    dataGrowth[0, 0] = 13.0
    dataGrowth[0, 3] = 13.0
    dataGrowth[0, 4] = 13.0
    assert (tensorInstance.data == dataGrowth).all()

    # Test Empty Tensor Set Item, subtensor
    emptyTensor = ttb.tensor.from_data(np.array([]))
    emptyTensor[0, 0, 0] = 0
    assert (emptyTensor.data == np.array([[[0]]])).all()
    assert emptyTensor.shape == (1, 1, 1)

    # Test Empty Tensor Set Item, subscripts
    emptyTensor = ttb.tensor.from_data(np.array([]))
    emptyTensor[np.array([0, 0, 0])] = 0
    assert (emptyTensor.data == np.array([[[0]]])).all()
    assert emptyTensor.shape == (1, 1, 1)

    # Linear Index with constant, index out of bounds
    with pytest.raises(AssertionError) as excinfo:
        tensorInstance[np.array([0, 3, 99])] = 13.0
    assert 'TTB:BadIndex In assignment X[I] = Y, a tensor X cannot be resized' in str(excinfo)

    # Attempting to set some other way
    with pytest.raises(AssertionError) as excinfo:
        tensorInstance[0, 'a', 5] = 13.0
    assert 'Invalid use of tensor setitem' in str(excinfo)

@pytest.mark.indevelopment
def test_tensor_getitem(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way
    # Case 1 single element
    assert tensorInstance[0, 0] == params['data'][0, 0]
    # Case 1 Subtensor
    assert (tensorInstance[:, :] == tensorInstance).data.all()
    # Case 1 Subtensor
    assert (tensorInstance[np.array([0, 1]), :].data == tensorInstance.data[[0, 1], :]).all()
    # Case 1 Subtensor
    assert (tensorInstance[0, :].data == tensorInstance.data[0, :]).all()
    assert (tensorInstance[:, 0].data == tensorInstance.data[:, 0]).all()

    # Case 2a:
    assert tensorInstance[np.array([0, 0]), 'extract'] == params['data'][0, 0]
    assert (tensorInstance[np.array([[0, 0], [1, 1]]), 'extract'] == params['data'][([0, 1], [0, 1])]).all()

    # Case 2b: Linear Indexing
    assert tensorInstance[np.array([0])] == params['data'][0, 0]
    with pytest.raises(AssertionError) as excinfo:
        tensorInstance[np.array([0]), np.array([0]), np.array([0])]
    assert "Linear indexing requires single input array" in str(excinfo)

@pytest.mark.indevelopment
def test_tensor_logical_and(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way

    # Tensor And
    assert (tensorInstance.logical_and(tensorInstance).data == np.ones((params['shape']))).all()

    # Non-zero And
    assert (tensorInstance.logical_and(1).data == np.ones((params['shape']))).all()

    # Zero And
    assert (tensorInstance.logical_and(0).data == np.zeros((params['shape']))).all()

@pytest.mark.indevelopment
def test_tensor__eq__(sample_tensor_2way, sample_tensor_3way, sample_tensor_4way):
    (params, tensorInstance) = sample_tensor_2way

    # Tensor tensor equality
    assert ((tensorInstance == tensorInstance).data).all()

    #Tensor scalar equality, not equal
    assert not ((tensorInstance == 7).data).any()

    #Tensor scalar equality, is equal
    data = np.zeros(params['data'].shape)
    data[0, 0] = 1
    assert ((tensorInstance == 1).data == data).all()

    (params3, tensorInstance3) = sample_tensor_3way

    # Tensor tensor equality
    assert ((tensorInstance3 == tensorInstance3).data).all()

    (params4, tensorInstance4) = sample_tensor_4way

    # Tensor tensor equality
    assert ((tensorInstance4 == tensorInstance4).data).all()



@pytest.mark.indevelopment
def test_tensor__ne__(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way

    # Tensor tensor equality
    assert not ((tensorInstance != tensorInstance).data).any()

    #Tensor scalar equality, not equal
    assert ((tensorInstance != 7).data).all()

    #Tensor scalar equality, is equal
    data = np.zeros(params['data'].shape)
    data[0, 0] = 1
    assert not ((tensorInstance != 1).data == data).any()

@pytest.mark.indevelopment
def test_tensor_full(sample_tensor_2way, sample_tensor_3way, sample_tensor_4way):
    (params2, tensorInstance2) = sample_tensor_2way
    if DEBUG_tests: 
        print("\nparam2['data']:")
        print(params2['data'])
        print('\ntensorInstace2.data:')
        print(tensorInstance2.data)
        print('\ntensorInstace2.full():')
        print(tensorInstance2.full())
    assert (tensorInstance2.full().data == params2['data']).all()

    (params3, tensorInstance3) = sample_tensor_3way
    if DEBUG_tests: 
        print("\nparam3['data']:")
        print(params3['data'])
        print('\ntensorInstace3.data:')
        print(tensorInstance3.data)
        print('\ntensorInstace3.full():')
        print(tensorInstance3.full())
    assert (tensorInstance3.full().data == params3['data']).all()

    (params4, tensorInstance4) = sample_tensor_4way
    if DEBUG_tests: 
        print("\nparam4['data']:")
        print(params4['data'])
        print('\ntensorInstace4.data:')
        print(tensorInstance4.data)
        print('\ntensorInstace4.full():')
        print(tensorInstance4.full())
    assert (tensorInstance4.full().data == params4['data']).all()

@pytest.mark.indevelopment
def test_tensor_ge(sample_tensor_2way, sample_tensor_3way, sample_tensor_4way):
    (params, tensorInstance) = sample_tensor_2way

    tensorLarger = ttb.tensor.from_data(params['data']+1)
    tensorSmaller = ttb.tensor.from_data(params['data'] - 1)

    assert ((tensorInstance >= tensorInstance).data).all()
    assert ((tensorInstance >= tensorSmaller).data).all()
    assert not ((tensorInstance >= tensorLarger).data).any()

    (params, tensorInstance) = sample_tensor_3way

    tensorLarger = ttb.tensor.from_data(params['data']+1)
    tensorSmaller = ttb.tensor.from_data(params['data'] - 1)

    assert ((tensorInstance >= tensorInstance).data).all()
    assert ((tensorInstance >= tensorSmaller).data).all()
    assert not ((tensorInstance >= tensorLarger).data).any()

    (params, tensorInstance) = sample_tensor_4way

    tensorLarger = ttb.tensor.from_data(params['data']+1)
    tensorSmaller = ttb.tensor.from_data(params['data'] - 1)

    assert ((tensorInstance >= tensorInstance).data).all()
    assert ((tensorInstance >= tensorSmaller).data).all()
    assert not ((tensorInstance >= tensorLarger).data).any()

@pytest.mark.indevelopment
def test_tensor_gt(sample_tensor_2way, sample_tensor_3way, sample_tensor_4way):
    (params, tensorInstance) = sample_tensor_2way

    tensorLarger = ttb.tensor.from_data(params['data']+1)
    tensorSmaller = ttb.tensor.from_data(params['data'] - 1)

    assert not ((tensorInstance > tensorInstance).data).any()
    assert ((tensorInstance > tensorSmaller).data).all()
    assert not ((tensorInstance > tensorLarger).data).any()

    (params, tensorInstance) = sample_tensor_3way

    tensorLarger = ttb.tensor.from_data(params['data']+1)
    tensorSmaller = ttb.tensor.from_data(params['data'] - 1)

    assert not ((tensorInstance > tensorInstance).data).any()
    assert ((tensorInstance > tensorSmaller).data).all()
    assert not ((tensorInstance > tensorLarger).data).any()

    (params, tensorInstance) = sample_tensor_4way

    tensorLarger = ttb.tensor.from_data(params['data']+1)
    tensorSmaller = ttb.tensor.from_data(params['data'] - 1)

    assert not ((tensorInstance > tensorInstance).data).any()
    assert ((tensorInstance > tensorSmaller).data).all()
    assert not ((tensorInstance > tensorLarger).data).any()

@pytest.mark.indevelopment
def test_tensor_le(sample_tensor_2way, sample_tensor_3way, sample_tensor_4way):
    (params, tensorInstance) = sample_tensor_2way

    tensorLarger = ttb.tensor.from_data(params['data'] + 1)
    tensorSmaller = ttb.tensor.from_data(params['data'] - 1)

    assert ((tensorInstance <= tensorInstance).data).all()
    assert not ((tensorInstance <= tensorSmaller).data).any()
    assert ((tensorInstance <= tensorLarger).data).all()

    (params, tensorInstance) = sample_tensor_3way

    tensorLarger = ttb.tensor.from_data(params['data'] + 1)
    tensorSmaller = ttb.tensor.from_data(params['data'] - 1)

    assert ((tensorInstance <= tensorInstance).data).all()
    assert not ((tensorInstance <= tensorSmaller).data).any()
    assert ((tensorInstance <= tensorLarger).data).all()

    (params, tensorInstance) = sample_tensor_4way

    tensorLarger = ttb.tensor.from_data(params['data'] + 1)
    tensorSmaller = ttb.tensor.from_data(params['data'] - 1)

    assert ((tensorInstance <= tensorInstance).data).all()
    assert not ((tensorInstance <= tensorSmaller).data).any()
    assert ((tensorInstance <= tensorLarger).data).all()

@pytest.mark.indevelopment
def test_tensor_lt(sample_tensor_2way, sample_tensor_3way, sample_tensor_4way):
    (params, tensorInstance) = sample_tensor_2way

    tensorLarger = ttb.tensor.from_data(params['data'] + 1)
    tensorSmaller = ttb.tensor.from_data(params['data'] - 1)

    assert not ((tensorInstance < tensorInstance).data).any()
    assert not ((tensorInstance < tensorSmaller).data).any()
    assert ((tensorInstance < tensorLarger).data).all()

    (params, tensorInstance) = sample_tensor_3way

    tensorLarger = ttb.tensor.from_data(params['data'] + 1)
    tensorSmaller = ttb.tensor.from_data(params['data'] - 1)

    assert not ((tensorInstance < tensorInstance).data).any()
    assert not ((tensorInstance < tensorSmaller).data).any()
    assert ((tensorInstance < tensorLarger).data).all()

    (params, tensorInstance) = sample_tensor_4way

    tensorLarger = ttb.tensor.from_data(params['data'] + 1)
    tensorSmaller = ttb.tensor.from_data(params['data'] - 1)

    assert not ((tensorInstance < tensorInstance).data).any()
    assert not ((tensorInstance < tensorSmaller).data).any()
    assert ((tensorInstance < tensorLarger).data).all()

@pytest.mark.indevelopment
def test_tensor_norm(sample_tensor_2way, sample_tensor_3way, sample_tensor_4way):
    # 2-way tensor
    (params2, tensorInstance2) = sample_tensor_2way
    if DEBUG_tests: 
        print('\ntensorInstace2.norm():')
        print(tensorInstance2.norm())
    assert tensorInstance2.norm() == np.linalg.norm(params2['data'].ravel())

    # 3-way tensor
    (params3, tensorInstance3) = sample_tensor_3way
    if DEBUG_tests: 
        print('\ntensorInstace3.norm():')
        print(tensorInstance3.norm())
    assert tensorInstance3.norm() == np.linalg.norm(params3['data'].ravel())

    # 4-way tensor
    (params4, tensorInstance4) = sample_tensor_4way
    if DEBUG_tests: 
        print('\ntensorInstace4.norm():')
        print(tensorInstance4.norm())
    assert tensorInstance4.norm() == np.linalg.norm(params4['data'].ravel())

@pytest.mark.indevelopment
def test_tensor_logical_not(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way

    assert (tensorInstance.logical_not().data == np.logical_not(params['data'])).all()

@pytest.mark.indevelopment
def test_tensor_logical_or(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way

    # Tensor Or
    assert (tensorInstance.logical_or(tensorInstance).data == np.ones((params['shape']))).all()

    # Non-zero Or
    assert (tensorInstance.logical_or(1).data == np.ones((params['shape']))).all()

    # Zero Or
    assert (tensorInstance.logical_or(0).data == np.ones((params['shape']))).all()

@pytest.mark.indevelopment
def test_tensor_logical_xor(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way

    # Tensor Or
    assert (tensorInstance.logical_xor(tensorInstance).data == np.zeros((params['shape']))).all()

    # Non-zero Or
    assert (tensorInstance.logical_xor(1).data == np.zeros((params['shape']))).all()

    # Zero Or
    assert (tensorInstance.logical_xor(0).data == np.ones((params['shape']))).all()

@pytest.mark.indevelopment
def test_tensor__add__(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way

    # Tensor + Tensor
    assert ((tensorInstance + tensorInstance).data == 2*(params['data'])).all()

    # Tensor + scalar
    assert ((tensorInstance + 1).data == 1 + (params['data'])).all()

@pytest.mark.indevelopment
def test_tensor__sub__(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way

    # Tensor - Tensor
    assert ((tensorInstance - tensorInstance).data == 0*(params['data'])).all()

    # Tensor - scalar
    assert ((tensorInstance - 1).data == (params['data'] - 1)).all()

@pytest.mark.indevelopment
def test_tensor__pow__(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way

    # Tensor** Tensor
    assert ((tensorInstance**tensorInstance).data == (params['data']**params['data'])).all()
    # Tensor**Scalar
    assert ((tensorInstance**2).data == (params['data']**2)).all()

@pytest.mark.indevelopment
def test_tensor__mul__(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way

    # Tensor* Tensor
    assert ((tensorInstance * tensorInstance).data == (params['data'] * params['data'])).all()
    # Tensor*Scalar
    assert ((tensorInstance * 2).data == (params['data'] * 2)).all()
    # Tensor * Sptensor
    assert ((tensorInstance * ttb.sptensor.from_tensor_type(tensorInstance)).data ==
            (params['data'] * params['data'])).all()

    # TODO tensor * ktensor

@pytest.mark.indevelopment
def test_tensor__rmul__(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way

    # Scalar * Tensor, only resolves when left object doesn't have appropriate __mul__
    assert ((2 * tensorInstance).data == (params['data'] * 2)).all()

@pytest.mark.indevelopment
def test_tensor__pos__(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way

    # +Tensor yields no change
    assert ((+tensorInstance).data == params['data']).all()

@pytest.mark.indevelopment
def test_tensor__neg__(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way

    # -Tensor yields negated copy of tensor
    assert ((-tensorInstance).data == -1*params['data']).all()
    # Original tensor should remain unchanged
    assert ((tensorInstance).data == params['data']).all()

@pytest.mark.indevelopment
def test_tensor_double(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way

    assert (tensorInstance.double() == params['data']).all()

@pytest.mark.indevelopment
def test_tensor_end(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way

    assert tensorInstance.end() == np.prod(params['shape']) - 1
    assert tensorInstance.end(k=0) == params['shape'][0] - 1

@pytest.mark.indevelopment
def test_tensor_isequal(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way
    subs = []
    vals = []
    for j in range(3):
        for i in range(2):
            subs.append([i, j])
            vals.append([params['data'][i, j]])
    sptensorInstance = ttb.sptensor.from_data(np.array(subs), np.array(vals),  params['shape'])

    assert tensorInstance.isequal(tensorInstance)
    assert tensorInstance.isequal(sptensorInstance)

    # Tensor is not equal to scalar
    assert not tensorInstance.isequal(1)

@pytest.mark.indevelopment
def test_tensor__truediv__(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way

    # Tensor / Tensor
    assert ((tensorInstance / tensorInstance).data == (params['data'] / params['data'])).all()

    # Tensor / Sptensor
    assert ((tensorInstance / ttb.sptensor.from_tensor_type(tensorInstance)).data == (params['data'] / params['data'])).all()

    # Tensor / Scalar
    assert ((tensorInstance / 2).data == (params['data'] / 2)).all()

@pytest.mark.indevelopment
def test_tensor__rtruediv__(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way

    # Scalar / Tensor, only resolves when left object doesn't have appropriate __mul__
    assert ((2 / tensorInstance).data == (2 / params['data'])).all()

@pytest.mark.indevelopment
def test_tensor_nnz(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way

    # NNZ for full tensor
    assert tensorInstance.nnz == 6

    # NNZ for tensor with at least one zero
    tensorInstance[0, 0] = 0
    assert tensorInstance.nnz == 5

@pytest.mark.indevelopment
def test_tensor_reshape(sample_tensor_2way, sample_tensor_3way, sample_tensor_4way):
    (params2, tensorInstance2) = sample_tensor_2way

    # Reshape with tuple
    tensorInstance2 = tensorInstance2.reshape((3, 2))
    if DEBUG_tests:
        print('\ntensorInstance2.reshape(3, 2):')
        print(tensorInstance2.reshape(3, 2))
    assert tensorInstance2.shape == (3, 2)
    data = np.array([[1., 5.], 
                     [4., 3.], 
                     [2., 6.]])
    assert (tensorInstance2.data == data).all()

    # Reshape with multiple arguments
    tensorInstance2a = tensorInstance2.reshape(2, 3)
    if DEBUG_tests:
        print('\ntensorInstance.reshape(2, 3):')
        print(tensorInstance2.reshape(2, 3))
    assert tensorInstance2a.shape == (2, 3)
    data2 = np.array([[1., 2., 3.], 
                      [4., 5., 6.]])
    assert (tensorInstance2a.data == data2).all()

    with pytest.raises(AssertionError) as excinfo:
        tensorInstance2.reshape(3, 3)
    assert "Reshaping a tensor cannot change number of elements" in str(excinfo)

    (params3, tensorInstance3) = sample_tensor_3way
    tensorInstance3 = tensorInstance3.reshape((3, 2, 2))
    if DEBUG_tests:
        print('\ntensorInstance3.reshape(3, 2, 2):')
        print(tensorInstance3)
    assert tensorInstance3.shape == (3, 2, 2)
    data3 = np.array([[[1., 7.], [4., 10.]], 
                      [[2., 8.], [5., 11.]], 
                      [[3., 9.], [6., 12.]]])
    assert (tensorInstance3.data == data3).all()

    (params4, tensorInstance4) = sample_tensor_4way
    tensorInstance4 = tensorInstance4.reshape((1, 3, 3, 9))
    if DEBUG_tests:
        print('\ntensorInstance4.reshape(1, 3, 3, 9):')
        print(tensorInstance4)
    assert tensorInstance4.shape == (1, 3, 3, 9)
    data4 = np.array([[[[ 1, 10, 19, 28, 37, 46, 55, 64, 73], 
                        [ 4, 13, 22, 31, 40, 49, 58, 67, 76], 
                        [ 7, 16, 25, 34, 43, 52, 61, 70, 79]],
                       [[ 2, 11, 20, 29, 38, 47, 56, 65, 74],
                        [ 5, 14, 23, 32, 41, 50, 59, 68, 77],
                        [ 8, 17, 26, 35, 44, 53, 62, 71, 80]],
                       [[ 3, 12, 21, 30, 39, 48, 57, 66, 75],
                        [ 6, 15, 24, 33, 42, 51, 60, 69, 78],
                        [ 9, 18, 27, 36, 45, 54, 63, 72, 81]]]])
    assert (tensorInstance4.data == data4).all()


@pytest.mark.indevelopment
def test_tensor_permute(sample_tensor_2way, sample_tensor_3way, sample_tensor_4way):
    (params, tensorInstance) = sample_tensor_2way

    # Permute rows and columns
    assert (tensorInstance.permute(np.array([1, 0])).data == np.transpose(params['data'])).all()

    # len(order) != ndims
    with pytest.raises(AssertionError) as excinfo:
        tensorInstance.permute(np.array([1, 0, 2]))
    assert "Invalid permutation order" in str(excinfo)

    # Try to permute order-1 tensor
    assert (ttb.tensor.from_data(np.array([1, 2, 3, 4])).permute(np.array([1])).data == np.array([1, 2, 3, 4])).all()

    # Empty order
    assert (ttb.tensor.from_data(np.array([])).permute(np.array([])).data == np.array([])).all()

    # 2-way
    (params2, tensorInstance2) = sample_tensor_2way
    tensorInstance2 = tensorInstance2.permute(np.array([1, 0]))
    if DEBUG_tests:
        print('\ntensorInstance2.permute(np.array([1, 0])):')
        print(tensorInstance2)
    assert tensorInstance2.shape == (3, 2)
    data2 = np.array([[1., 4.], 
                     [2., 5.], 
                     [3., 6.]])
    assert (tensorInstance2.data == data2).all()
    
    # 3-way
    (params3, tensorInstance3) = sample_tensor_3way
    tensorInstance3 = tensorInstance3.permute(np.array([2, 1, 0]))
    if DEBUG_tests:
        print('\ntensorInstance3.permute(np.array([2, 1, 0])):')
        print(tensorInstance3)
    assert tensorInstance3.shape == (2, 3, 2)
    data3 = np.array([[[ 1.,  2.], 
                       [ 3.,  4.], 
                       [ 5.,  6.]], 
                      [[ 7.,  8.], 
                       [ 9., 10.], 
                       [11., 12.]]])
    assert (tensorInstance3.data == data3).all()
    
    # 4-way
    (params4, tensorInstance4) = sample_tensor_4way
    tensorInstance4 = tensorInstance4.permute(np.array([3, 1, 2, 0]))
    if DEBUG_tests:
        print('\ntensorInstance4.permute(np.array([3, 1, 2, 0])):')
        print(tensorInstance4)
    assert tensorInstance4.shape == (3, 3, 3, 3)
    data4 = np.array([[[[ 1,  2,  3],
                       [10, 11, 12],
                       [19, 20, 21]],
                      [[ 4,  5,  6],
                       [13, 14, 15],
                       [22, 23, 24]],
                      [[ 7,  8,  9],
                       [16, 17, 18],
                       [25, 26, 27]]],
                     [[[28, 29, 30],
                       [37, 38, 39],
                       [46, 47, 48]],
                      [[31, 32, 33],
                       [40, 41, 42],
                       [49, 50, 51]],
                      [[34, 35, 36],
                       [43, 44, 45],
                       [52, 53, 54]]],
                     [[[55, 56, 57],
                       [64, 65, 66],
                       [73, 74, 75]],
                      [[58, 59, 60],
                       [67, 68, 69],
                       [76, 77, 78]],
                      [[61, 62, 63],
                       [70, 71, 72],
                       [79, 80, 81]]]])
    assert (tensorInstance4.data == data4).all()

@pytest.mark.indevelopment
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
    assert (tensorCollapse.data == data).all()

    # single dimension collapse using max function
    datamax = np.array([4, 5, 6])
    tensorCollapseMax = tensorInstance2.collapse(np.array([0]), fun=np.max)
    assert (tensorCollapseMax.data == datamax).all()

    # multiple dimensions collapse
    data4 = np.array([[ 99, 342, 585],
                      [126, 369, 612],
                      [153, 396, 639]])
    tensorCollapse4 = tensorInstance4.collapse(np.array([0, 2]))
    assert (tensorCollapse4.data == data4).all()

    # multiple dimensions collapse
    data4max = np.array([[21, 48, 75],
                         [24, 51, 78],
                         [27, 54, 81]])
    tensorCollapse4Max = tensorInstance4.collapse(np.array([0, 2]), fun=np.max)
    assert (tensorCollapse4Max.data == data4max).all()

@pytest.mark.indevelopment
def test_tensor_contract(sample_tensor_2way, sample_tensor_3way, sample_tensor_4way):
    (params2, tensorInstance2) = sample_tensor_2way

    with pytest.raises(AssertionError) as excinfo:
        tensorInstance2.contract(0, 1)
    assert "Must contract along equally sized dimensions" in str(excinfo)

    with pytest.raises(AssertionError) as excinfo:
        tensorInstance2.contract(0, 0)
    assert "Must contract along two different dimensions" in str(excinfo)

    contractableTensor = ttb.tensor.from_data(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),(3,3))
    assert contractableTensor.contract(0, 1) == 15

    (params3, tensorInstance3) = sample_tensor_3way
    print("\ntensorInstance3.contract(0,2) = ")
    print(tensorInstance3.contract(0,2))
    data3 = np.array([9, 13, 17])
    assert (tensorInstance3.contract(0,2).data == data3).all()

    (params4, tensorInstance4) = sample_tensor_4way
    print("\ntensorInstance4.contract(0,1) = ")
    print(tensorInstance4.contract(0,1))
    data4 = np.array([[15,  96, 177], 
                      [42, 123, 204], 
                      [69, 150, 231]])
    assert (tensorInstance4.contract(0, 1).data == data4).all()

    print("\ntensorInstance4.contract(1,3) = ")
    print(tensorInstance4.contract(1,3))
    data4a = np.array([[93, 120, 147], 
                      [96, 123, 150], 
                      [99, 126, 153]])
    assert (tensorInstance4.contract(1, 3).data == data4a).all()

@pytest.mark.indevelopment
def test_tensor__repr__(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way
    # Test that we can capture each of these
    str(tensorInstance)

    str(ttb.tensor.from_data(np.array([1, 2, 3])))

    str(ttb.tensor.from_data(np.arange(0, 81).reshape(3, 3, 3, 3)))

    str(ttb.tensor())

@pytest.mark.indevelopment
def test_tensor_exp(sample_tensor_2way, sample_tensor_3way, sample_tensor_4way):
    (params, tensorInstance) = sample_tensor_2way
    assert (tensorInstance.exp().data == np.exp(params['data'])).all()

@pytest.mark.indevelopment
def test_tensor_innerprod(sample_tensor_2way, sample_tensor_3way, sample_tensor_4way):
    (params, tensorInstance) = sample_tensor_2way
    # Tensor innerproduct
    assert tensorInstance.innerprod(tensorInstance) == np.arange(1, 7).dot(np.arange(1, 7))

    # Sptensor innerproduct
    assert tensorInstance.innerprod(ttb.sptensor.from_tensor_type(tensorInstance)) == \
           np.arange(1, 7).dot(np.arange(1, 7))

    # Wrong size innerproduct
    with pytest.raises(AssertionError) as excinfo:
        tensorInstance.innerprod(ttb.tensor.from_data(np.ones((4, 4))))
    assert 'Inner product must be between tensors of the same size' in str(excinfo)

    # Wrong class innerproduct
    with pytest.raises(AssertionError) as excinfo:
        tensorInstance.innerprod(5)
    assert "Inner product between tensor and that class is not supported" in str(excinfo)

    # 2-way
    (params2, tensorInstance2) = sample_tensor_2way
    if DEBUG_tests: 
        print(f'\ntensorInstance2.innerprod(tensorInstance2): {tensorInstance2.innerprod(tensorInstance2)}')
    assert tensorInstance2.innerprod(tensorInstance2) == 91

    # 3-way
    (params3, tensorInstance3) = sample_tensor_3way
    if DEBUG_tests: 
        print(f'\ntensorInstance3.innerprod(tensorInstance3): {tensorInstance3.innerprod(tensorInstance3)}')
    assert tensorInstance3.innerprod(tensorInstance3) == 650

    # 4-way
    (params4, tensorInstance4) = sample_tensor_4way
    if DEBUG_tests: 
        print(f'\ntensorInstance4.innerprod(tensorInstance4): {tensorInstance4.innerprod(tensorInstance4)}')
    assert tensorInstance4.innerprod(tensorInstance4) == 180441

@pytest.mark.indevelopment
def test_tensor_mask(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way
    assert (tensorInstance.mask(ttb.tensor.from_data(np.ones(params['shape']))) == params['data'].reshape((6,))).all()

    # Wrong shape mask
    with pytest.raises(AssertionError) as excinfo:
        tensorInstance.mask(ttb.tensor.from_data(np.ones((11, 3))))
    assert "Mask cannot be bigger than the data tensor" in str(excinfo)

@pytest.mark.indevelopment
def test_tensor_squeeze(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way

    # No singleton dimensions
    assert (tensorInstance.squeeze().data == params['data']).all()

    # All singleton dimensions
    assert (ttb.tensor.from_data(np.array([[[4]]])).squeeze() == 4)

    # A singleton dimension
    assert (ttb.tensor.from_data(np.array([[1, 2, 3]])).squeeze().data == np.array([1, 2, 3])).all()

@pytest.mark.indevelopment
def test_tensor_ttv(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way

    # Wrong shape vector
    with pytest.raises(AssertionError) as excinfo:
        tensorInstance.ttv(np.array([np.array([1, 2]), np.array([1, 2])]))
    assert "Multiplicand is wrong size" in str(excinfo)

    # Multiply by single vector
    assert (tensorInstance.ttv(np.array([2, 2]), 0).data == np.array([2, 2]).dot(params['data'])).all()

    # Multiply by multiple vectors, infer dimensions
    assert (tensorInstance.ttv(np.array([np.array([2, 2]), np.array([1, 1, 1])])) ==
            np.array([1, 1, 1]).dot(np.array([2, 2]).dot(params['data'])))

@pytest.mark.indevelopment
def test_tensor_ttsv(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way
    tensorInstance = ttb.tensor.from_data(np.ones((4, 4, 4)))
    vector = np.array([1, 1, 1, 1])

    # Invalid dims
    with pytest.raises(AssertionError) as excinfo:
        tensorInstance.ttsv(vector, dims = 1)
    assert "Invalid modes in ttsv" in str(excinfo)


    assert tensorInstance.ttsv(vector, version=1) == 64
    assert (tensorInstance.ttsv(vector, dims=-1, version=1) == np.array([16, 16, 16, 16])).all()

    tensorInstance = ttb.tensor.from_data(np.ones((4, 4, 4, 4)))
    assert tensorInstance.ttsv(vector, dims=-3, version=1).isequal(tensorInstance.ttv(vector, 0))

    # Test new algorithm

    # Only works for all modes of equal length
    with pytest.raises(AssertionError) as excinfo:
        tensorInstance.ttsv(vector, dims=-1)
    assert "New version only support vector times all modes" in str(excinfo)
    tensorInstance = ttb.tensor.from_data(np.ones((4, 4, 4)))
    assert tensorInstance.ttsv(vector) == 64

@pytest.mark.indevelopment
def test_tensor_issymmetric(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way

    assert tensorInstance.issymmetric() is False
    assert tensorInstance.issymmetric(version=1) is False

    symmetricData = np.array([[[0.5, 0, 0.5, 0], [0, 0, 0, 0],[0.5, 0, 0, 0],[0, 0, 0, 0]],
                              [[0, 0, 0, 0], [0, 2.5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                              [[0.5, 0, 0, 0], [0, 0, 0, 0], [0, 0, 3.5, 0], [0, 0, 0, 0]],
                              [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])
    symmetricTensor = ttb.tensor.from_data(symmetricData)
    assert symmetricTensor.issymmetric() is True
    assert symmetricTensor.issymmetric(version=1) is True
    answer, diffs, perms = symmetricTensor.issymmetric(version=1, return_details=True)
    assert answer is True
    assert (diffs == 0).all()

    symmetricData[3, 1, 0] = 3
    symmetricTensor = ttb.tensor.from_data(symmetricData)
    assert symmetricTensor.issymmetric() is False
    assert symmetricTensor.issymmetric(version=1) is False

@pytest.mark.indevelopment
def test_tensor_symmetrize(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way

    # Test new default version

    symmetricData = np.array([[[0.5, 0, 0.5, 0], [0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0, 0, 0]],
                              [[0, 0, 0, 0], [0, 2.5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                              [[0.5, 0, 0, 0], [0, 0, 0, 0], [0, 0, 3.5, 0], [0, 0, 0, 0]],
                              [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])
    symmetricTensor = ttb.tensor.from_data(symmetricData)
    assert symmetricTensor.symmetrize().isequal(symmetricTensor)

    symmetricData = np.zeros((4, 4, 4))
    symmetricData[1, 2, 1] = 1
    symmetricTensor = ttb.tensor.from_data(symmetricData)
    assert symmetricTensor.issymmetric() is False
    assert (symmetricTensor.symmetrize()).issymmetric()

    with pytest.raises(AssertionError) as excinfo:
        symmetricTensor.symmetrize(grps=np.array([[0, 1], [1, 2]]))
    assert "Cannot have overlapping symmetries" in str(excinfo)

    # Improper shape tensor for symmetry
    asymmetricData = np.zeros((5, 4, 6))
    asymmetricTensor = ttb.tensor.from_data(asymmetricData)
    with pytest.raises(AssertionError) as excinfo:
        asymmetricTensor.symmetrize()
    assert "Dimension mismatch for symmetrization" in str(excinfo)

    # Test older keyword version
    symmetricData = np.array([[[0.5, 0, 0.5, 0], [0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0, 0, 0]],
                              [[0, 0, 0, 0], [0, 2.5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                              [[0.5, 0, 0, 0], [0, 0, 0, 0], [0, 0, 3.5, 0], [0, 0, 0, 0]],
                              [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])

    symmetricTensor = ttb.tensor.from_data(symmetricData)
    assert symmetricTensor.symmetrize(version=1).isequal(symmetricTensor)

    symmetricData = np.zeros((4, 4, 4))
    symmetricData[1, 2, 1] = 1
    symmetricTensor = ttb.tensor.from_data(symmetricData)
    assert symmetricTensor.issymmetric() is False
    assert (symmetricTensor.symmetrize(version=1)).issymmetric()

    with pytest.raises(AssertionError) as excinfo:
        symmetricTensor.symmetrize(grps=np.array([[0, 1], [1, 2]]), version=1)
    assert "Cannot have overlapping symmetries" in str(excinfo)

    # Improper shape tensor for symmetry
    asymmetricData = np.zeros((5, 4, 6))
    asymmetricTensor = ttb.tensor.from_data(asymmetricData)
    with pytest.raises(AssertionError) as excinfo:
        asymmetricTensor.symmetrize(version=1)
    assert "Dimension mismatch for symmetrization" in str(excinfo)

@pytest.mark.indevelopment
def test_tensor__str__(sample_tensor_2way):
    # Test 1D
    data = np.random.normal(size=(4,))
    tensorInstance = ttb.tensor.from_data(data)
    s = ''
    s += 'tensor of shape '
    s += (' x ').join([str(int(d)) for d in tensorInstance.shape])
    s += '\n'
    s += 'data'
    s += '[:] = \n'
    s += data.__str__()
    s += '\n'
    assert s == tensorInstance.__str__()

    # Test 2D
    data = np.random.normal(size=(4, 3))
    tensorInstance = ttb.tensor.from_data(data)
    s = ''
    s += 'tensor of shape '
    s += (' x ').join([str(int(d)) for d in tensorInstance.shape])
    s += '\n'
    s += 'data'
    s += '[:, :] = \n'
    s += data.__str__()
    s += '\n'
    assert s == tensorInstance.__str__()

    # Test 3D,shape in decreasing and increasing order
    data = np.random.normal(size=(4, 3, 2))
    tensorInstance = ttb.tensor.from_data(data)
    s = ''
    s += 'tensor of shape '
    s += (' x ').join([str(int(d)) for d in tensorInstance.shape])
    s += '\n'
    for i in range(data.shape[0]):
        s += 'data'
        s += '[{}, :, :] = \n'.format(i)
        s += data[i, :, :].__str__()
        s += '\n'
    assert s == tensorInstance.__str__()

    data = np.random.normal(size=(2, 3, 4))
    tensorInstance = ttb.tensor.from_data(data)
    s = ''
    s += 'tensor of shape '
    s += (' x ').join([str(int(d)) for d in tensorInstance.shape])
    s += '\n'
    for i in range(data.shape[0]):
        s += 'data'
        s += '[{}, :, :] = \n'.format(i)
        s += data[i, :, :].__str__()
        s += '\n'
    assert s == tensorInstance.__str__()

    # Test > 3D
    data = np.random.normal(size=(4, 4, 3, 2))
    tensorInstance = ttb.tensor.from_data(data)
    s = ''
    s += 'tensor of shape '
    s += (' x ').join([str(int(d)) for d in tensorInstance.shape])
    s += '\n'
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            s += 'data'
            s += '[{}, {}, :, :] = \n'.format(i,j)
            s += data[i, j, :, :].__str__()
            s += '\n'
    assert s == tensorInstance.__str__()


@pytest.mark.indevelopment
def test_tensor_mttkrp(sample_tensor_2way):
    (params, tensorInstance) = sample_tensor_2way
    tensorInstance = ttb.tensor.from_function(np.ones, (2, 3, 4))

    # 2-way sparse tensor
    weights = np.array([2., 2.])
    fm0 = np.array([[1., 3.], [2., 4.]])
    fm1 = np.array([[5., 8.], [6., 9.], [7., 10.]])
    fm2 = np.array([[11., 15.], [12., 16.], [13., 17.], [14., 18.]])
    factor_matrices = [fm0, fm1, fm2]
    ktensorInstance = ttb.ktensor.from_data(weights, factor_matrices)

    m0 = np.array([[1800., 3564.],
                   [1800., 3564.]])
    m1 = np.array([[300., 924.],
                   [300., 924.],
                   [300., 924.]])
    m2 = np.array([[108., 378.],
                   [108., 378.],
                   [108., 378.],
                   [108., 378.]])
    assert np.allclose(tensorInstance.mttkrp(ktensorInstance, 0), m0)
    assert np.allclose(tensorInstance.mttkrp(ktensorInstance, 1), m1)
    assert np.allclose(tensorInstance.mttkrp(ktensorInstance, 2), m2)

    # 5-way dense tensor
    shape = (2,3,4,5,6)
    T = ttb.tensor.from_data(np.arange(1,np.prod(shape)+1), shape)
    U = [];
    for s in shape:
        U.append(np.ones((s,2)))

    data0 = np.array([[129600, 129600],
                      [129960, 129960]])
    assert (T.mttkrp(U,0) == data0).all()

    data1 = np.array([[86040, 86040],
                      [86520, 86520],
                      [87000, 87000]])
    assert (T.mttkrp(U,1) == data1).all()

    data2 = np.array([[63270, 63270],
                      [64350, 64350],
                      [65430, 65430],
                      [66510, 66510]])
    assert (T.mttkrp(U,2) == data2).all()

    data3 = np.array([[45000, 45000],
                      [48456, 48456],
                      [51912, 51912],
                      [55368, 55368],
                      [58824, 58824]])
    assert (T.mttkrp(U,3) == data3).all()

    data4 = np.array([[ 7260,  7260],
                      [21660, 21660],
                      [36060, 36060],
                      [50460, 50460],
                      [64860, 64860],
                      [79260, 79260]])
    assert (T.mttkrp(U,4) == data4).all()

    # tensor too small
    with pytest.raises(AssertionError) as excinfo:
        tensorInstance2 = ttb.tensor.from_data(np.array([1]))
        tensorInstance2.mttkrp([], 0)
    assert 'MTTKRP is invalid for tensors with fewer than 2 dimensions' in str(excinfo)

    # second argument not a ktensor or list
    with pytest.raises(AssertionError) as excinfo:
        tensorInstance.mttkrp(5, 0)
    assert 'Second argument should be a list of arrays or a ktensor' in str(excinfo)

    # second argument list is not the correct length
    with pytest.raises(AssertionError) as excinfo:
        m0 = np.ones((2,2))
        tensorInstance.mttkrp([m0, m0, m0, m0], 0)
    assert 'Second argument contains the wrong number of arrays' in str(excinfo)

    # arrays not the correct shape
    with pytest.raises(AssertionError) as excinfo:
        m0 = np.ones((2, 2))
        m1 = np.ones((3, 2))
        m2 = np.ones((5, 2))
        tensorInstance.mttkrp([m0, m1, m2], 0)
    assert 'Entry 2 of list of arrays is wrong size' in str(excinfo)

@pytest.mark.indevelopment
def test_tensor_nvecs(sample_tensor_2way):
    (data, tensorInstance) = sample_tensor_2way

    nv1 = np.array([[ 0.4286671335486261, 0.5663069188480352,  0.7039467041474443]]).T
    nv2 = np.array([[ 0.4286671335486261, 0.5663069188480352,  0.7039467041474443],
                    [ 0.8059639085892916, 0.1123824140966059, -0.5811990803961161]]).T

    # Test for one eigenvector
    assert np.allclose((tensorInstance.nvecs(1, 1)), nv1)

    # Test for r >= N-1, requires cast to dense
    with pytest.warns(Warning) as record:
        assert np.allclose((tensorInstance.nvecs(1, 2)), nv2)
    assert 'Greater than or equal to tensor.shape[n] - 1 eigenvectors requires cast to dense to solve' \
           in str(record[0].message)
