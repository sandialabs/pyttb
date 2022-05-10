# Copyright 2022 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

import numpy as np
import pytest
import TensorToolbox as ttb

@pytest.fixture()
def sample_tensor():
    data = np.array([[1., 2., 3.], [4., 5., 6.]])
    shape = (2, 3)
    params = {'data':data, 'shape': shape}
    tensorInstance = ttb.tensor().from_data(data, shape)
    return params, tensorInstance

@pytest.fixture()
def sample_tensor_3way():
    data = np.array([[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]])
    shape = (2, 3, 2)
    params = {'data':data, 'shape': shape}
    tensorInstance = ttb.tensor().from_data(data, shape)
    return params, tensorInstance

@pytest.mark.indevelopment
def test_tensor_initialization_Empty():
    empty = np.array([])

    # No args
    tensorInstance = ttb.tensor()
    assert (tensorInstance.data == empty).all()
    assert (tensorInstance.shape == ())

@pytest.mark.indevelopment
def test_tensor_initialization_explicit(sample_tensor):
    (params, tensorInstance) = sample_tensor
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

@pytest.mark.indevelopment
def test_tensor_initialization_tensors(sample_tensor):
    (params, tensorInstance) = sample_tensor

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

@pytest.mark.indevelopment
def test_tensor_initialization_fhAndSize():
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
def test_tensor_find(sample_tensor):
    (params, tensorInstance) = sample_tensor
    subs, vals = tensorInstance.find()
    assert (vals == params['data'].reshape(np.prod(params['shape']), 1)).all()
    assert (subs == ttb.tt_ind2sub(params['shape'], np.arange(0, np.prod(params['shape'])))).all()

@pytest.mark.indevelopment
def test_tensor_ndims(sample_tensor):
    (params, tensorInstance) = sample_tensor
    assert tensorInstance.ndims == len(params['shape'])

    # Empty tensor has zero dimensions
    assert ttb.tensor.from_data(np.array([])) == 0

@pytest.mark.indevelopment
def test_tensor_setitem(sample_tensor):
    (params, tensorInstance) = sample_tensor

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
def test_tensor_getitem(sample_tensor):
    (params, tensorInstance) = sample_tensor
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
def test_tensor_logical_and(sample_tensor):
    (params, tensorInstance) = sample_tensor

    # Tensor And
    assert (tensorInstance.logical_and(tensorInstance).data == np.ones((params['shape']))).all()

    # Non-zero And
    assert (tensorInstance.logical_and(1).data == np.ones((params['shape']))).all()

    # Zero And
    assert (tensorInstance.logical_and(0).data == np.zeros((params['shape']))).all()

@pytest.mark.indevelopment
def test_tensor__eq__(sample_tensor):
    (params, tensorInstance) = sample_tensor

    # Tensor tensor equality
    assert ((tensorInstance == tensorInstance).data).all()

    #Tensor scalar equality, not equal
    assert not ((tensorInstance == 7).data).any()

    #Tensor scalar equality, is equal
    data = np.zeros(params['data'].shape)
    data[0, 0] = 1
    assert ((tensorInstance == 1).data == data).all()

@pytest.mark.indevelopment
def test_tensor__ne__(sample_tensor):
    (params, tensorInstance) = sample_tensor

    # Tensor tensor equality
    assert not ((tensorInstance != tensorInstance).data).any()

    #Tensor scalar equality, not equal
    assert ((tensorInstance != 7).data).all()

    #Tensor scalar equality, is equal
    data = np.zeros(params['data'].shape)
    data[0, 0] = 1
    assert not ((tensorInstance != 1).data == data).any()

@pytest.mark.indevelopment
def test_tensor_full(sample_tensor):
    (params, tensorInstance) = sample_tensor

    assert (tensorInstance.full().data == params['data']).all()

@pytest.mark.indevelopment
def test_tensor_ge(sample_tensor):
    (params, tensorInstance) = sample_tensor

    tensorLarger = ttb.tensor.from_data(params['data']+1)
    tensorSmaller = ttb.tensor.from_data(params['data'] - 1)

    assert ((tensorInstance >= tensorInstance).data).all()
    assert ((tensorInstance >= tensorSmaller).data).all()
    assert not ((tensorInstance >= tensorLarger).data).any()

@pytest.mark.indevelopment
def test_tensor_gt(sample_tensor):
    (params, tensorInstance) = sample_tensor

    tensorLarger = ttb.tensor.from_data(params['data']+1)
    tensorSmaller = ttb.tensor.from_data(params['data'] - 1)

    assert not ((tensorInstance > tensorInstance).data).any()
    assert ((tensorInstance > tensorSmaller).data).all()
    assert not ((tensorInstance > tensorLarger).data).any()

@pytest.mark.indevelopment
def test_tensor_le(sample_tensor):
    (params, tensorInstance) = sample_tensor

    tensorLarger = ttb.tensor.from_data(params['data'] + 1)
    tensorSmaller = ttb.tensor.from_data(params['data'] - 1)

    assert ((tensorInstance <= tensorInstance).data).all()
    assert not ((tensorInstance <= tensorSmaller).data).any()
    assert ((tensorInstance <= tensorLarger).data).all()

@pytest.mark.indevelopment
def test_tensor_lt(sample_tensor):
    (params, tensorInstance) = sample_tensor

    tensorLarger = ttb.tensor.from_data(params['data'] + 1)
    tensorSmaller = ttb.tensor.from_data(params['data'] - 1)

    assert not ((tensorInstance < tensorInstance).data).any()
    assert not ((tensorInstance < tensorSmaller).data).any()
    assert ((tensorInstance < tensorLarger).data).all()

@pytest.mark.indevelopment
def test_tensor_norm(sample_tensor, sample_tensor_3way):
    # 2-way tensor
    (params2, tensorInstance2) = sample_tensor
    assert tensorInstance2.norm() == np.linalg.norm(params2['data'].ravel())

    # 3-way tensor
    (params3, tensorInstance3) = sample_tensor_3way
    assert tensorInstance3.norm() == np.linalg.norm(params3['data'].ravel())

@pytest.mark.indevelopment
def test_tensor_logical_not(sample_tensor):
    (params, tensorInstance) = sample_tensor

    assert (tensorInstance.logical_not().data == np.logical_not(params['data'])).all()

@pytest.mark.indevelopment
def test_tensor_logical_or(sample_tensor):
    (params, tensorInstance) = sample_tensor

    # Tensor Or
    assert (tensorInstance.logical_or(tensorInstance).data == np.ones((params['shape']))).all()

    # Non-zero Or
    assert (tensorInstance.logical_or(1).data == np.ones((params['shape']))).all()

    # Zero Or
    assert (tensorInstance.logical_or(0).data == np.ones((params['shape']))).all()

@pytest.mark.indevelopment
def test_tensor_logical_xor(sample_tensor):
    (params, tensorInstance) = sample_tensor

    # Tensor Or
    assert (tensorInstance.logical_xor(tensorInstance).data == np.zeros((params['shape']))).all()

    # Non-zero Or
    assert (tensorInstance.logical_xor(1).data == np.zeros((params['shape']))).all()

    # Zero Or
    assert (tensorInstance.logical_xor(0).data == np.ones((params['shape']))).all()

@pytest.mark.indevelopment
def test_tensor__add__(sample_tensor):
    (params, tensorInstance) = sample_tensor

    # Tensor + Tensor
    assert ((tensorInstance + tensorInstance).data == 2*(params['data'])).all()

    # Tensor + scalar
    assert ((tensorInstance + 1).data == 1 + (params['data'])).all()

@pytest.mark.indevelopment
def test_tensor__sub__(sample_tensor):
    (params, tensorInstance) = sample_tensor

    # Tensor - Tensor
    assert ((tensorInstance - tensorInstance).data == 0*(params['data'])).all()

    # Tensor - scalar
    assert ((tensorInstance - 1).data == (params['data'] - 1)).all()

@pytest.mark.indevelopment
def test_tensor__pow__(sample_tensor):
    (params, tensorInstance) = sample_tensor

    # Tensor** Tensor
    assert ((tensorInstance**tensorInstance).data == (params['data']**params['data'])).all()
    # Tensor**Scalar
    assert ((tensorInstance**2).data == (params['data']**2)).all()

@pytest.mark.indevelopment
def test_tensor__mul__(sample_tensor):
    (params, tensorInstance) = sample_tensor

    # Tensor* Tensor
    assert ((tensorInstance * tensorInstance).data == (params['data'] * params['data'])).all()
    # Tensor*Scalar
    assert ((tensorInstance * 2).data == (params['data'] * 2)).all()
    # Tensor * Sptensor
    assert ((tensorInstance * ttb.sptensor.from_tensor_type(tensorInstance)).data ==
            (params['data'] * params['data'])).all()

    # TODO tensor * ktensor

@pytest.mark.indevelopment
def test_tensor__rmul__(sample_tensor):
    (params, tensorInstance) = sample_tensor

    # Scalar * Tensor, only resolves when left object doesn't have appropriate __mul__
    assert ((2 * tensorInstance).data == (params['data'] * 2)).all()

@pytest.mark.indevelopment
def test_tensor__pos__(sample_tensor):
    (params, tensorInstance) = sample_tensor

    # +Tensor yields no change
    assert ((+tensorInstance).data == params['data']).all()

@pytest.mark.indevelopment
def test_tensor__neg__(sample_tensor):
    (params, tensorInstance) = sample_tensor

    # -Tensor yields negated copy of tensor
    assert ((-tensorInstance).data == -1*params['data']).all()
    # Original tensor should remain unchanged
    assert ((tensorInstance).data == params['data']).all()

@pytest.mark.indevelopment
def test_tensor_double(sample_tensor):
    (params, tensorInstance) = sample_tensor

    assert (tensorInstance.double() == params['data']).all()

@pytest.mark.indevelopment
def test_tensor_end(sample_tensor):
    (params, tensorInstance) = sample_tensor

    assert tensorInstance.end() == np.prod(params['shape']) - 1
    assert tensorInstance.end(k=0) == params['shape'][0] - 1

@pytest.mark.indevelopment
def test_tensor_isequal(sample_tensor):
    (params, tensorInstance) = sample_tensor
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
def test_tensor__truediv__(sample_tensor):
    (params, tensorInstance) = sample_tensor

    # Tensor / Tensor
    assert ((tensorInstance / tensorInstance).data == (params['data'] / params['data'])).all()

    # Tensor / Sptensor
    assert ((tensorInstance / ttb.sptensor.from_tensor_type(tensorInstance)).data == (params['data'] / params['data'])).all()

    # Tensor / Scalar
    assert ((tensorInstance / 2).data == (params['data'] / 2)).all()

@pytest.mark.indevelopment
def test_tensor__rtruediv__(sample_tensor):
    (params, tensorInstance) = sample_tensor

    # Scalar / Tensor, only resolves when left object doesn't have appropriate __mul__
    assert ((2 / tensorInstance).data == (2 / params['data'])).all()

@pytest.mark.indevelopment
def test_tensor_nnz(sample_tensor):
    (params, tensorInstance) = sample_tensor

    # NNZ for full tensor
    assert tensorInstance.nnz == 6

    # NNZ for tensor with at least one zero
    tensorInstance[0, 0] = 0
    assert tensorInstance.nnz == 5

@pytest.mark.indevelopment
def test_tensor_reshape(sample_tensor):
    (params, tensorInstance) = sample_tensor

    # Reshape with tuple
    tensorInstance = tensorInstance.reshape((3, 2))
    assert tensorInstance.shape == (3, 2)

    # Reshape with multiple arguments
    tensorInstance = tensorInstance.reshape(2, 3)
    assert tensorInstance.shape == (2, 3)

    with pytest.raises(AssertionError) as excinfo:
        tensorInstance.reshape(3, 3)
    assert "Reshaping a tensor cannot change number of elements" in str(excinfo)

@pytest.mark.indevelopment
def test_tensor_permute(sample_tensor):
    (params, tensorInstance) = sample_tensor

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

@pytest.mark.indevelopment
def test_tensor_contract(sample_tensor):
    (params, tensorInstance) = sample_tensor

    with pytest.raises(AssertionError) as excinfo:
        tensorInstance.contract(0, 1)
    assert "Must contract along equally sized dimensions" in str(excinfo)

    with pytest.raises(AssertionError) as excinfo:
        tensorInstance.contract(0, 0)
    assert "Must contract along two different dimensions" in str(excinfo)

    contractableTensor = ttb.tensor.from_data(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    assert contractableTensor.contract(0, 1) == 15

    contractableNonScalarTensor = ttb.tensor.from_data(np.arange(1, 82).reshape((3, 3, 3, 3)))
    assert (contractableNonScalarTensor.contract(0, 1).data ==
            np.array([[15, 42, 69], [96, 123, 150], [177, 204, 231]])).all()

@pytest.mark.indevelopment
def test_tensor__repr__(sample_tensor):
    (params, tensorInstance) = sample_tensor
    # Test that we can capture each of these
    str(tensorInstance)

    str(ttb.tensor.from_data(np.array([1, 2, 3])))

    str(ttb.tensor.from_data(np.arange(0, 81).reshape(3, 3, 3, 3)))

    str(ttb.tensor())

@pytest.mark.indevelopment
def test_tensor_exp(sample_tensor):
    (params, tensorInstance) = sample_tensor
    assert (tensorInstance.exp().data == np.exp(params['data'])).all()

@pytest.mark.indevelopment
def test_tensor_innerprod(sample_tensor):
    (params, tensorInstance) = sample_tensor
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

@pytest.mark.indevelopment
def test_tensor_mask(sample_tensor):
    (params, tensorInstance) = sample_tensor
    assert (tensorInstance.mask(ttb.tensor.from_data(np.ones(params['shape']))) == params['data'].reshape((6,))).all()

    # Wrong shape mask
    with pytest.raises(AssertionError) as excinfo:
        tensorInstance.mask(ttb.tensor.from_data(np.ones((11, 3))))
    assert "Mask cannot be bigger than the data tensor" in str(excinfo)

@pytest.mark.indevelopment
def test_tensor_squeeze(sample_tensor):
    (params, tensorInstance) = sample_tensor

    # No singleton dimensions
    assert (tensorInstance.squeeze().data == params['data']).all()

    # All singleton dimensions
    assert (ttb.tensor.from_data(np.array([[[4]]])).squeeze() == 4)

    # A singleton dimension
    assert (ttb.tensor.from_data(np.array([[1, 2, 3]])).squeeze().data == np.array([1, 2, 3])).all()

@pytest.mark.indevelopment
def test_tensor_ttv(sample_tensor):
    (params, tensorInstance) = sample_tensor

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
def test_tensor_ttsv(sample_tensor):
    (params, tensorInstance) = sample_tensor
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
def test_tensor_issymmetric(sample_tensor):
    (params, tensorInstance) = sample_tensor

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
def test_tensor_symmetrize(sample_tensor):
    (params, tensorInstance) = sample_tensor

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
def test_tensor__str__(sample_tensor):
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
    # s += '\n'
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
    # s += '\n'
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
        # s += '\n'
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
        # s += '\n'
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
            # s += '\n'
    assert s == tensorInstance.__str__()


@pytest.mark.indevelopment
def test_tensor_mttkrp(sample_tensor):
    (params, tensorInstance) = sample_tensor
    tensorInstance = ttb.tensor.from_function(np.ones, (2, 3, 4))

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
def test_tensor_nvecs(sample_tensor):
    (data, tensorInstance) = sample_tensor

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
