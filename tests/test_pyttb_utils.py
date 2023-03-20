# Copyright 2022 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

import logging

import numpy as np
import pytest
import scipy.sparse as sparse

import pyttb as ttb


@pytest.mark.indevelopment
def test_sptensor_to_dense_matrix():
    subs = np.array([[1, 1, 1], [1, 1, 3], [2, 2, 2], [3, 3, 3]])
    vals = np.array([[0.5], [1.5], [2.5], [3.5]])
    shape = (4, 4, 4)
    mode0 = sparse.coo_matrix(
        ([0.5, 1.5, 2.5, 3.5], ([5, 13, 10, 15], [1, 1, 2, 3]))
    ).toarray()
    mode1 = sparse.coo_matrix(
        ([0.5, 1.5, 2.5, 3.5], ([5, 13, 10, 15], [1, 1, 2, 3]))
    ).toarray()
    mode2 = sparse.coo_matrix(
        ([0.5, 1.5, 2.5, 3.5], ([5, 5, 10, 15], [1, 3, 2, 3]))
    ).toarray()
    Ynt = [mode0, mode1, mode2]

    sptensorInstance = ttb.sptensor().from_data(subs, vals, shape)
    tensorInstance = sptensorInstance.full()

    for mode in range(sptensorInstance.ndims):
        Xnt = ttb.tt_to_dense_matrix(tensorInstance, mode, True)
        assert np.array_equal(Xnt, Ynt[mode])


@pytest.mark.indevelopment
def test_sptensor_from_dense_matrix():
    tensorInstance = ttb.tensor.from_data(np.random.normal(size=(4, 4, 4)))
    for mode in range(tensorInstance.ndims):
        tensorCopy = ttb.tensor.from_tensor_type(tensorInstance)
        Xnt = ttb.tt_to_dense_matrix(tensorCopy, mode, True)
        Ynt = ttb.tt_from_dense_matrix(Xnt, tensorCopy.shape, mode, 0)
        assert tensorCopy.isequal(Ynt)

    for mode in range(tensorInstance.ndims):
        tensorCopy = ttb.tensor.from_tensor_type(tensorInstance)
        Xnt = ttb.tt_to_dense_matrix(tensorCopy, mode, False)
        Ynt = ttb.tt_from_dense_matrix(Xnt, tensorCopy.shape, mode, 1)
        assert tensorCopy.isequal(Ynt)


@pytest.mark.indevelopment
def test_tt_union_rows():
    a = np.array([[4, 6], [1, 9], [2, 6], [2, 6], [99, 0]])
    b = np.array([[1, 7], [1, 8], [2, 6]])
    assert (
        ttb.tt_union_rows(a, b)
        == np.array([[1, 7], [1, 8], [4, 6], [1, 9], [2, 6], [99, 0]])
    ).all()
    _, idx = np.unique(a, axis=0, return_index=True)
    assert (ttb.tt_union_rows(a, np.array([])) == a[np.sort(idx)]).all()
    assert (ttb.tt_union_rows(np.array([]), a) == a[np.sort(idx)]).all()


@pytest.mark.indevelopment
def test_tt_dimscheck():
    #  Empty
    rdims, ridx = ttb.tt_dimscheck(np.array([]), 6)
    assert (rdims == np.array([0, 1, 2, 3, 4, 5])).all()
    assert ridx is None

    # Minus
    rdims, ridx = ttb.tt_dimscheck(np.array([-1]), 6)
    assert (rdims == np.array([1, 2, 3, 4, 5])).all()
    assert ridx is None

    # Invalid minus
    with pytest.raises(AssertionError) as excinfo:
        ttb.tt_dimscheck(np.array([-7]), 6, 6)
    assert "Invalid magnitude for negative dims selection" in str(excinfo)

    # Positive
    rdims, ridx = ttb.tt_dimscheck(np.array([5]), 6)
    assert (rdims == np.array([5])).all()
    assert ridx is None

    # M==P
    rdims, ridx = ttb.tt_dimscheck(np.array([-1]), 6, 5)
    assert (rdims == np.array([1, 2, 3, 4, 5])).all()
    assert (ridx == np.arange(0, 5)).all()

    # M==N
    rdims, ridx = ttb.tt_dimscheck(np.array([-1]), 6, 6)
    assert (rdims == np.array([1, 2, 3, 4, 5])).all()
    assert (ridx == rdims).all()

    # M>N
    with pytest.raises(AssertionError) as excinfo:
        ttb.tt_dimscheck(np.array([-1]), 6, 7)
    assert "Cannot have more multiplicands than dimensions" in str(excinfo)

    # M!=P and M!=N
    with pytest.raises(AssertionError) as excinfo:
        ttb.tt_dimscheck(np.array([-1]), 6, 4)
    assert "Invalid number of multiplicands" in str(excinfo)


@pytest.mark.indevelopment
def test_tt_tenfun():
    data = np.array([[1, 2, 3], [4, 5, 6]])
    t1 = ttb.tensor.from_data(data)
    t2 = ttb.tensor.from_data(data)

    # Binary case
    def add(x, y):
        return x + y

    assert (ttb.tt_tenfun(add, t1, t2).data == 2 * data).all()

    # Single argument case
    def add1(x):
        return x + 1

    assert (ttb.tt_tenfun(add1, t1).data == (data + 1)).all()

    # Multi argument case
    def tensor_max(x):
        return np.max(x, axis=0)

    assert (ttb.tt_tenfun(tensor_max, t1, t1, t1).data == data).all()
    # TODO: sptensor arguments, depends on fixing the indexing ordering

    # No np array case
    assert (ttb.tt_tenfun(tensor_max, data, data, data).data == data).all()

    # No argument case
    with pytest.raises(AssertionError) as excinfo:
        ttb.tt_tenfun(tensor_max)
    assert "Must provide element(s) to perform operation on" in str(excinfo)

    # No list case
    with pytest.raises(AssertionError) as excinfo:
        ttb.tt_tenfun(tensor_max, [1, 2, 3])
    assert "Invalid input to ten fun" in str(excinfo)

    # Scalar argument not in first two positions
    with pytest.raises(AssertionError) as excinfo:
        ttb.tt_tenfun(tensor_max, t1, t1, 1)
    assert "Argument 2 is a scalar but expected a tensor" in str(excinfo)

    # Tensors of different sizes
    with pytest.raises(AssertionError) as excinfo:
        ttb.tt_tenfun(
            tensor_max,
            t1,
            t1,
            ttb.tensor.from_data(np.concatenate((data, np.array([[7, 8, 9]])))),
        )
    assert "Tensor 2 is not the same size as the first tensor input" in str(excinfo)


@pytest.mark.indevelopment
def test_tt_setdiff_rows():
    a = np.array([[4, 6], [1, 9], [2, 6], [2, 6], [99, 0]])
    b = np.array(
        [[1, 7], [1, 8], [2, 6], [2, 1], [2, 4], [4, 6], [4, 7], [5, 9], [5, 2], [5, 1]]
    )
    assert (ttb.tt_setdiff_rows(a, b) == np.array([1, 4])).all()

    a = np.array([[4, 6], [1, 9]])
    b = np.array([[1, 7], [1, 8]])
    assert (ttb.tt_setdiff_rows(a, b) == np.array([0, 1])).all()

    a = np.array([[4, 6], [1, 9]])
    b = np.array([])
    assert (ttb.tt_setdiff_rows(a, b) == np.arange(a.shape[0])).all()
    assert (ttb.tt_setdiff_rows(b, a) == b).all()


@pytest.mark.indevelopment
def test_tt_intersect_rows():
    a = np.array([[4, 6], [1, 9], [2, 6], [2, 6]])
    b = np.array(
        [[1, 7], [1, 8], [2, 6], [2, 1], [2, 4], [4, 6], [4, 7], [5, 9], [5, 2], [5, 1]]
    )
    assert (ttb.tt_intersect_rows(a, b) == np.array([2, 0])).all()

    a = np.array([[4, 6], [1, 9]])
    b = np.array([])
    assert (ttb.tt_intersect_rows(a, b) == b).all()
    assert (ttb.tt_intersect_rows(a, b) == ttb.tt_intersect_rows(b, a)).all()


@pytest.mark.indevelopment
def test_tt_ismember_rows():
    a = np.array([[4, 6], [1, 9], [2, 6]])
    b = np.array(
        [[1, 7], [1, 8], [2, 6], [2, 1], [2, 4], [4, 6], [4, 7], [5, 9], [5, 2], [5, 1]]
    )
    assert (ttb.tt_ismember_rows(a, b) == np.array([5, -1, 2])).all()
    assert (
        ttb.tt_ismember_rows(b, a) == np.array([-1, -1, 2, -1, -1, 0, -1, -1, -1, -1])
    ).all()


@pytest.mark.indevelopment
def test_tt_irenumber():
    # TODO: Note this is more of a regression test by exploring the behaviour in MATLAB still not totally clear on WHY it behaves this way
    # Constant shouldn't effect performance
    const = 1

    subs = np.array([[const, const, 0], [const, const, 1]])
    vals = np.array([[0.5], [1.5]])
    shape = (4, 4, 4)
    data = {"subs": subs, "vals": vals, "shape": shape}
    sptensorInstance = ttb.sptensor().from_data(subs, vals, shape)
    slice_tuple = (
        slice(None, None, None),
        slice(None, None, None),
        slice(None, None, None),
    )
    extended_result = np.array(
        [[const, const, const, const, const, 0], [const, const, const, const, const, 1]]
    )

    # Pad equal to number of modes
    assert (
        ttb.tt_irenumber(sptensorInstance, shape, (const, const, const))
        == extended_result
    ).all()

    # Full slice should equal original
    assert (ttb.tt_irenumber(sptensorInstance, shape, slice_tuple) == subs).all()

    # Verify that list and equivalent slice act the same
    assert (
        ttb.tt_irenumber(sptensorInstance, shape, (const, const, slice(0, 1, 1)))
        == np.array([[const, const, const, const, 0], [const, const, const, const, 1]])
    ).all()
    assert (
        ttb.tt_irenumber(sptensorInstance, shape, (const, const, [0, 1]))
        == np.array([[const, const, const, const, 0], [const, const, const, const, 1]])
    ).all()


@pytest.mark.indevelopment
def test_tt_assignment_type():
    # type(x)==type(rhs)
    x = 5
    rhs = 5
    subs = 5
    assert ttb.tt_assignment_type(x, subs, rhs) == "subtensor"

    # type(x)!=type(rhs), subs dimensionality >=2
    rhs = "cat"
    subs = (1, 1, 1)
    assert ttb.tt_assignment_type(x, subs, rhs) == "subtensor"

    subs = (np.array([1, 2, 3]),)
    assert ttb.tt_assignment_type(x, subs, rhs) == "subscripts"

    # type(x)!=type(rhs), subs dimensionality <2
    subs = np.array([1])
    assert ttb.tt_assignment_type(x, subs, rhs) == "subscripts"


@pytest.mark.indevelopment
def test_tt_renumberdims():
    # Singleton output
    shape = 5
    number_range = 1
    idx = 3
    assert ttb.tt_renumberdim(idx, shape, number_range) == (0, 0)

    # Array output
    shape = 5
    number_range = np.array([1, 3, 4])
    idx = [1, 3, 4]
    newidx, newshape = ttb.tt_renumberdim(idx, shape, number_range)
    assert (newidx == np.array([0, 1, 2])).all()
    assert newshape == 3

    # Slice input
    shape = 5
    number_range = slice(1, 3, None)
    idx = [1, 2]
    newidx, newshape = ttb.tt_renumberdim(idx, shape, number_range)
    assert (newidx == np.array([0, 1])).all()
    assert newshape == 2


@pytest.mark.indevelopment
def test_tt_renumber():
    # Singleton in each dimension
    shape = (5, 5, 5)
    number_range = (1, 1, 1)
    subs = np.array([[3, 3, 3]])
    newsubs, newshape = ttb.tt_renumber(subs, shape, number_range)
    assert (newsubs == np.array([[0, 0, 0]])).all()
    assert newshape == (0, 0, 0)

    # Array in each dimension
    shape = (5, 5, 5)
    number_range = ([1, 3, 4], [1, 3, 4], [1, 3, 4])
    subs = np.array([[1, 1, 1], [3, 3, 3]])
    newsubs, newshape = ttb.tt_renumber(subs, shape, number_range)
    assert (newsubs == np.array([[0, 0, 0], [1, 1, 1]])).all()
    assert newshape == (3, 3, 3)

    # Slice in each dimension
    shape = (5, 5, 5)
    number_range = (slice(1, 3, None), slice(1, 3, None), slice(1, 3, None))
    subs = np.array([[2, 2, 2]])
    newsubs, newshape = ttb.tt_renumber(subs, shape, number_range)
    assert (newsubs == np.array([[1, 1, 1]])).all()
    assert newshape == (2, 2, 2)

    # Slice in each dimension, empty subs
    shape = (5, 5, 5)
    number_range = (slice(1, 3, None), slice(1, 3, None), slice(1, 3, None))
    subs = np.array([])
    newsubs, newshape = ttb.tt_renumber(subs, shape, number_range)
    assert newsubs.size == 0
    assert newshape == (2, 2, 2)

    # Not slice in each dimension, empty subs
    shape = (5, 5, 5)
    number_range = ([1, 3, 4], [1, 3, 4], [1, 3, 4])
    subs = np.array([])
    newsubs, newshape = ttb.tt_renumber(subs, shape, number_range)
    assert newsubs.size == 0
    assert newshape == (3, 3, 3)


@pytest.mark.indevelopment
def test_tt_sub2ind_valid():
    subs = np.array([[0, 0, 0], [1, 1, 1], [3, 3, 3]])
    idx = np.array([0, 21, 63])
    siz = np.array([4, 4, 4])
    assert (ttb.tt_sub2ind(siz, subs) == idx).all()

    empty = np.array([])
    assert (ttb.tt_sub2ind(siz, empty) == empty).all()


@pytest.mark.indevelopment
def test_tt_ind2sub_valid():
    subs = np.array([[0, 0, 0], [1, 1, 1], [3, 3, 3]])
    idx = np.array([0, 21, 63])
    shape = (4, 4, 4)
    logging.debug(f"\nttb.tt_ind2sub(shape, idx): {ttb.tt_ind2sub(shape, idx)}")
    assert (ttb.tt_ind2sub(shape, idx) == subs).all()

    subs = np.array([[1, 0], [0, 1]])
    idx = np.array([1, 2])
    shape = (2, 2)
    logging.debug(f"\nttb.tt_ind2sub(shape, idx): {ttb.tt_ind2sub(shape, idx)}")
    assert (ttb.tt_ind2sub(shape, idx) == subs).all()

    empty = np.array([])
    assert (
        ttb.tt_ind2sub(shape, empty) == np.empty(shape=(0, len(shape)), dtype=int)
    ).all()


@pytest.mark.indevelopment
def test_tt_subsubsref_valid():
    subs = np.array([[0, 0, 0], [1, 1, 1], [3, 3, 3]])
    vals = np.array([[0], [21], [63]])
    shape = (4, 4, 4)
    a = ttb.sptensor.from_data(subs, vals, shape)
    assert isinstance(ttb.tt_subsubsref(a, [1]), ttb.sptensor)
    # TODO need to understand behavior better
    assert True


@pytest.mark.indevelopment
def test_tt_intvec2str_valid():
    """This function is slotted to be removed because it is probably unnecessary in python"""
    v = np.array([1, 2, 3])
    assert ttb.tt_intvec2str(v) == "[1 2 3]"


@pytest.mark.indevelopment
def test_tt_sizecheck_empty():
    assert ttb.tt_sizecheck(())


@pytest.mark.indevelopment
def test_tt_sizecheck_valid():
    assert ttb.tt_sizecheck((2, 2, 2))


@pytest.mark.indevelopment
def test_tt_sizecheck_invalid():
    # Float
    assert not ttb.tt_sizecheck((1.0, 2, 2))
    # Too many dimensions
    assert not ttb.tt_sizecheck(np.array([[2, 2], [2, 2]]))
    # Nan
    assert not ttb.tt_sizecheck((np.nan, 2, 2))
    # Inf
    assert not ttb.tt_sizecheck((np.inf, 2, 2))
    # Zero
    assert not ttb.tt_sizecheck((0, 2, 2))


@pytest.mark.indevelopment
def test_tt_sizecheck_errorMessage():
    # Raise when nargout == 0
    with pytest.raises(AssertionError) as excinfo:
        ttb.tt_sizecheck((1.0, 2, 2), nargout=False)
    assert "Size must be a row vector of real positive integers" in str(excinfo)


@pytest.mark.indevelopment
def test_tt_subscheck_empty():
    assert ttb.tt_subscheck(np.array([]))


@pytest.mark.indevelopment
def test_tt_subscheck_valid():
    assert ttb.tt_subscheck(np.array([[2, 2], [2, 2]]))


@pytest.mark.indevelopment
def test_tt_subscheck_invalid():
    # Too Few Dims
    assert not ttb.tt_subscheck(np.array([2]))
    # Too Many Dims
    assert not ttb.tt_subscheck(np.array([[[2]]]))
    # Less than 0
    assert not ttb.tt_subscheck(np.array([[-1, 2], [2, 2]]))
    # Nan
    assert not ttb.tt_subscheck(np.array([[np.nan, 2], [2, 2]]))
    # Inf
    assert not ttb.tt_subscheck(np.array([[np.inf, 2], [2, 2]]))
    # Non-int
    assert not ttb.tt_subscheck(np.array([[1.0, 2], [2, 2]]))


@pytest.mark.indevelopment
def test_tt_subscheck_errorMessage():
    # Raise when nargout == 0
    with pytest.raises(AssertionError) as excinfo:
        ttb.tt_subscheck(np.array([1.0, 2, 2]), nargout=False)
    assert "Subscripts must be a matrix of real positive integers" in str(excinfo)


@pytest.mark.indevelopment
def test_tt_valscheck_empty():
    assert ttb.tt_valscheck(np.array([]))


@pytest.mark.indevelopment
def test_tt_valscheck_valid():
    assert ttb.tt_valscheck(np.array([[0.5], [1.5], [2.5]]))


@pytest.mark.indevelopment
def test_tt_valscheck_invalid():
    # Row array
    assert not ttb.tt_valscheck(np.array([2, 2]))
    # Matrix, too many dimensions
    assert not ttb.tt_valscheck(np.array([[2, 2], [2, 2]]))


@pytest.mark.indevelopment
def test_tt_valscheck_errorMessage():
    # Raise when nargout == 0
    with pytest.raises(AssertionError) as excinfo:
        ttb.tt_valscheck(np.array([2, 2]), nargout=False)
    assert "Values must be in array" in str(excinfo)


@pytest.mark.indevelopment
def test_isrow_empty():
    assert not ttb.isrow(np.array([[]]))


@pytest.mark.indevelopment
def test_isrow_valid():
    assert ttb.isrow(np.array([[2, 2, 2]]))


@pytest.mark.indevelopment
def test_isrow_invalid():
    # 2 x 2 Matrix
    assert not ttb.isrow(np.array([[2, 2], [2, 2]]))
    # Column vector
    assert not ttb.isrow(np.array([[2, 2, 2]]).T)


@pytest.mark.indevelopment
def test_isvector_empty():
    assert ttb.isvector(np.array([[]]))


@pytest.mark.indevelopment
def test_isvector_valid():
    assert ttb.isvector(np.array([[2, 2, 2]]))
    assert ttb.isvector(np.array([[2, 2, 2]]).T)


@pytest.mark.indevelopment
def test_isvector_invalid():
    # 2 x 2 Matrix
    assert not ttb.isvector(np.array([[2, 2], [2, 2]]))


@pytest.mark.indevelopment
def test_islogical_empty():
    assert not ttb.islogical(np.array([[]]))


@pytest.mark.indevelopment
def test_islogical_valid():
    assert ttb.islogical(True)


@pytest.mark.indevelopment
def test_islogical_invalid():
    assert not ttb.islogical(np.array([[2, 2, 2]]))
    assert not ttb.islogical(1.1)
    assert not ttb.islogical(0)
