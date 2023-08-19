# Copyright 2022 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

import logging

import numpy as np
import pytest
import scipy.sparse as sparse

import pyttb as ttb
import pyttb.pyttb_utils as ttb_utils


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
        Xnt = ttb_utils.tt_to_dense_matrix(tensorInstance, mode, True)
        assert np.array_equal(Xnt, Ynt[mode])


@pytest.mark.indevelopment
def test_sptensor_from_dense_matrix():
    tensorInstance = ttb.tensor(np.random.normal(size=(4, 4, 4)))
    for mode in range(tensorInstance.ndims):
        tensorCopy = ttb.tensor.from_tensor_type(tensorInstance)
        Xnt = ttb_utils.tt_to_dense_matrix(tensorCopy, mode, True)
        Ynt = ttb_utils.tt_from_dense_matrix(Xnt, tensorCopy.shape, mode, 0)
        assert tensorCopy.isequal(Ynt)

    for mode in range(tensorInstance.ndims):
        tensorCopy = ttb.tensor.from_tensor_type(tensorInstance)
        Xnt = ttb_utils.tt_to_dense_matrix(tensorCopy, mode, False)
        Ynt = ttb_utils.tt_from_dense_matrix(Xnt, tensorCopy.shape, mode, 1)
        assert tensorCopy.isequal(Ynt)


@pytest.mark.indevelopment
def test_tt_union_rows():
    a = np.array([[4, 6], [1, 9], [2, 6], [2, 6], [99, 0]])
    b = np.array([[1, 7], [1, 8], [2, 6]])
    assert np.array_equal(
        ttb_utils.tt_union_rows(a, b),
        np.array([[1, 7], [1, 8], [4, 6], [1, 9], [2, 6], [99, 0]]),
    )
    _, idx = np.unique(a, axis=0, return_index=True)
    assert np.array_equal(ttb_utils.tt_union_rows(a, np.array([])), a[np.sort(idx)])
    assert np.array_equal(ttb_utils.tt_union_rows(np.array([]), a), a[np.sort(idx)])


@pytest.mark.indevelopment
def test_tt_dimscheck():
    #  Empty
    rdims, ridx = ttb_utils.tt_dimscheck(6, dims=np.array([]))
    assert np.array_equal(rdims, np.array([0, 1, 2, 3, 4, 5]))
    assert ridx is None

    # Exclude Dims
    rdims, ridx = ttb_utils.tt_dimscheck(6, exclude_dims=np.array([1]))
    assert np.array_equal(rdims, np.array([0, 2, 3, 4, 5]))
    assert ridx is None

    # Invalid minus
    with pytest.raises(ValueError) as excinfo:
        ttb_utils.tt_dimscheck(6, 6, exclude_dims=np.array([7]))
    assert "Exclude dims" in str(excinfo)

    # Positive
    rdims, ridx = ttb_utils.tt_dimscheck(6, dims=np.array([5]))
    assert np.array_equal(rdims, np.array([5]))
    assert ridx is None

    # M==P
    rdims, ridx = ttb_utils.tt_dimscheck(6, 5, exclude_dims=np.array([0]))
    assert np.array_equal(rdims, np.array([1, 2, 3, 4, 5]))
    assert np.array_equal(ridx, np.arange(0, 5))

    # M==N
    rdims, ridx = ttb_utils.tt_dimscheck(6, 6, exclude_dims=np.array([0]))
    assert np.array_equal(rdims, np.array([1, 2, 3, 4, 5]))
    assert np.array_equal(ridx, rdims)

    # M>N
    with pytest.raises(AssertionError) as excinfo:
        ttb_utils.tt_dimscheck(6, 7, exclude_dims=np.array([0]))
    assert "Cannot have more multiplicands than dimensions" in str(excinfo)

    # M!=P and M!=N
    with pytest.raises(AssertionError) as excinfo:
        ttb_utils.tt_dimscheck(6, 4, exclude_dims=np.array([0]))
    assert "Invalid number of multiplicands" in str(excinfo)

    # Both dims and exclude dims
    with pytest.raises(ValueError) as excinfo:
        ttb_utils.tt_dimscheck(6, dims=[], exclude_dims=[])
    assert "not both" in str(excinfo)

    # We no longer support negative dims. Make sure that is explicit
    with pytest.raises(ValueError) as excinfo:
        ttb_utils.tt_dimscheck(6, dims=np.array([-1]))
    assert "Negative dims" in str(excinfo), f"{str(excinfo)}"


@pytest.mark.indevelopment
def test_tt_tenfun():
    data = np.array([[1, 2, 3], [4, 5, 6]])
    t1 = ttb.tensor(data)
    t2 = ttb.tensor(data)

    # Binary case
    def add(x, y):
        return x + y

    assert np.array_equal(ttb_utils.tt_tenfun(add, t1, t2).data, 2 * data)

    # Single argument case
    def add1(x):
        return x + 1

    assert np.array_equal(ttb_utils.tt_tenfun(add1, t1).data, (data + 1))

    # Multi argument case
    def tensor_max(x):
        return np.max(x, axis=0)

    assert np.array_equal(ttb_utils.tt_tenfun(tensor_max, t1, t1, t1).data, data)
    # TODO: sptensor arguments, depends on fixing the indexing ordering

    # No np array case
    assert np.array_equal(ttb_utils.tt_tenfun(tensor_max, data, data, data).data, data)

    # No argument case
    with pytest.raises(AssertionError) as excinfo:
        ttb_utils.tt_tenfun(tensor_max)
    assert "Must provide element(s) to perform operation on" in str(excinfo)

    # No list case
    with pytest.raises(AssertionError) as excinfo:
        ttb_utils.tt_tenfun(tensor_max, [1, 2, 3])
    assert "Invalid input to ten fun" in str(excinfo)

    # Scalar argument not in first two positions
    with pytest.raises(AssertionError) as excinfo:
        ttb_utils.tt_tenfun(tensor_max, t1, t1, 1)
    assert "Argument 2 is a scalar but expected a tensor" in str(excinfo)

    # Tensors of different sizes
    with pytest.raises(AssertionError) as excinfo:
        ttb_utils.tt_tenfun(
            tensor_max,
            t1,
            t1,
            ttb.tensor(np.concatenate((data, np.array([[7, 8, 9]])))),
        )
    assert "Tensor 2 is not the same size as the first tensor input" in str(excinfo)


@pytest.mark.indevelopment
def test_tt_setdiff_rows():
    a = np.array([[4, 6], [1, 9], [2, 6], [2, 6], [99, 0]])
    b = np.array(
        [[1, 7], [1, 8], [2, 6], [2, 1], [2, 4], [4, 6], [4, 7], [5, 9], [5, 2], [5, 1]]
    )
    assert np.array_equal(ttb_utils.tt_setdiff_rows(a, b), np.array([1, 4]))

    a = np.array([[4, 6], [1, 9]])
    b = np.array([[1, 7], [1, 8]])
    assert np.array_equal(ttb_utils.tt_setdiff_rows(a, b), np.array([0, 1]))

    a = np.array([[4, 6], [1, 9]])
    b = np.array([])
    assert np.array_equal(ttb_utils.tt_setdiff_rows(a, b), np.arange(a.shape[0]))
    assert np.array_equal(ttb_utils.tt_setdiff_rows(b, a), b)


@pytest.mark.indevelopment
def test_tt_intersect_rows():
    a = np.array([[4, 6], [1, 9], [2, 6], [2, 6]])
    b = np.array(
        [[1, 7], [1, 8], [2, 6], [2, 1], [2, 4], [4, 6], [4, 7], [5, 9], [5, 2], [5, 1]]
    )
    assert np.array_equal(ttb_utils.tt_intersect_rows(a, b), np.array([2, 0]))

    a = np.array([[4, 6], [1, 9]])
    b = np.array([])
    assert np.array_equal(ttb_utils.tt_intersect_rows(a, b), b)
    assert np.array_equal(
        ttb_utils.tt_intersect_rows(a, b), ttb_utils.tt_intersect_rows(b, a)
    )


@pytest.mark.indevelopment
def test_tt_ismember_rows():
    a = np.array([[4, 6], [1, 9], [2, 6]])
    b = np.array(
        [[1, 7], [1, 8], [2, 6], [2, 1], [2, 4], [4, 6], [4, 7], [5, 9], [5, 2], [5, 1]]
    )
    assert np.array_equal(ttb_utils.tt_ismember_rows(a, b), np.array([5, -1, 2]))
    assert np.array_equal(
        ttb_utils.tt_ismember_rows(b, a),
        np.array([-1, -1, 2, -1, -1, 0, -1, -1, -1, -1]),
    )


@pytest.mark.indevelopment
def test_tt_irenumber():
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
    assert np.array_equal(
        ttb_utils.tt_irenumber(sptensorInstance, shape, (const, const, const)),
        extended_result,
    )

    # Full slice should equal original
    assert np.array_equal(
        ttb_utils.tt_irenumber(sptensorInstance, shape, slice_tuple), subs
    )

    # Verify that list and equivalent slice act the same
    assert np.array_equal(
        ttb_utils.tt_irenumber(sptensorInstance, shape, (const, const, slice(0, 1, 1))),
        np.array([[const, const, const, const, 0], [const, const, const, const, 1]]),
    )
    assert np.array_equal(
        ttb_utils.tt_irenumber(sptensorInstance, shape, (const, const, [0, 1])),
        np.array([[const, const, const, const, 0], [const, const, const, const, 1]]),
    )


@pytest.mark.indevelopment
def test_tt_renumberdims():
    # Singleton output
    shape = 5
    number_range = 1
    idx = 3
    assert ttb_utils.tt_renumberdim(idx, shape, number_range) == (0, 0)

    # Array output
    shape = 5
    number_range = np.array([1, 3, 4])
    idx = [1, 3, 4]
    newidx, newshape = ttb_utils.tt_renumberdim(idx, shape, number_range)
    assert np.array_equal(newidx, np.array([0, 1, 2]))
    assert newshape == 3

    # Slice input
    shape = 5
    number_range = slice(1, 3, None)
    idx = [1, 2]
    newidx, newshape = ttb_utils.tt_renumberdim(idx, shape, number_range)
    assert np.array_equal(newidx, np.array([0, 1]))
    assert newshape == 2


@pytest.mark.indevelopment
def test_tt_renumber():
    # Singleton in each dimension
    shape = (5, 5, 5)
    number_range = (1, 1, 1)
    subs = np.array([[3, 3, 3]])
    newsubs, newshape = ttb_utils.tt_renumber(subs, shape, number_range)
    assert np.array_equal(newsubs, np.array([[0, 0, 0]]))
    assert newshape == (0, 0, 0)

    # Array in each dimension
    shape = (5, 5, 5)
    number_range = ([1, 3, 4], [1, 3, 4], [1, 3, 4])
    subs = np.array([[1, 1, 1], [3, 3, 3]])
    newsubs, newshape = ttb_utils.tt_renumber(subs, shape, number_range)
    assert np.array_equal(newsubs, np.array([[0, 0, 0], [1, 1, 1]]))
    assert newshape == (3, 3, 3)

    # Slice in each dimension
    shape = (5, 5, 5)
    number_range = (slice(1, 3, None), slice(1, 3, None), slice(1, 3, None))
    subs = np.array([[2, 2, 2]])
    newsubs, newshape = ttb_utils.tt_renumber(subs, shape, number_range)
    assert np.array_equal(newsubs, np.array([[1, 1, 1]]))
    assert newshape == (2, 2, 2)

    # Slice in each dimension, empty subs
    shape = (5, 5, 5)
    number_range = (slice(1, 3, None), slice(1, 3, None), slice(1, 3, None))
    subs = np.array([])
    newsubs, newshape = ttb_utils.tt_renumber(subs, shape, number_range)
    assert newsubs.size == 0
    assert newshape == (2, 2, 2)

    # Not slice in each dimension, empty subs
    shape = (5, 5, 5)
    number_range = ([1, 3, 4], [1, 3, 4], [1, 3, 4])
    subs = np.array([])
    newsubs, newshape = ttb_utils.tt_renumber(subs, shape, number_range)
    assert newsubs.size == 0
    assert newshape == (3, 3, 3)


@pytest.mark.indevelopment
def test_tt_sub2ind_valid():
    subs = np.array([[0, 0, 0], [1, 1, 1], [3, 3, 3]])
    idx = np.array([0, 21, 63])
    siz = np.array([4, 4, 4])
    assert np.array_equal(ttb_utils.tt_sub2ind(siz, subs), idx)

    empty = np.array([])
    assert np.array_equal(ttb_utils.tt_sub2ind(siz, empty), empty)


@pytest.mark.indevelopment
def test_tt_ind2sub_valid():
    subs = np.array([[0, 0, 0], [1, 1, 1], [3, 3, 3]])
    idx = np.array([0, 21, 63])
    shape = (4, 4, 4)
    logging.debug(
        f"\nttb_utils.tt_ind2sub(shape, idx): {ttb_utils.tt_ind2sub(shape, idx)}"
    )
    assert np.array_equal(ttb_utils.tt_ind2sub(shape, idx), subs)

    subs = np.array([[1, 0], [0, 1]])
    idx = np.array([1, 2])
    shape = (2, 2)
    logging.debug(
        f"\nttb_utils.tt_ind2sub(shape, idx): {ttb_utils.tt_ind2sub(shape, idx)}"
    )
    assert np.array_equal(ttb_utils.tt_ind2sub(shape, idx), subs)

    empty = np.array([])
    assert np.array_equal(
        ttb_utils.tt_ind2sub(shape, empty), np.empty(shape=(0, len(shape)), dtype=int)
    )

    # Single negative index
    shape = (2, 2)
    neg_idx = np.array([-1])
    assert np.array_equal(ttb_utils.tt_ind2sub(shape, neg_idx), np.array([[1, 1]]))

    # Multiple negative indices
    shape = (2, 2)
    neg_idx = np.array([-1, -2])
    assert np.array_equal(
        ttb_utils.tt_ind2sub(shape, neg_idx), np.array([[1, 1], [0, 1]])
    )


@pytest.mark.indevelopment
def test_tt_subsubsref_valid():
    subs = np.array([[0, 0, 0], [1, 1, 1], [3, 3, 3]])
    vals = np.array([[0], [21], [63]])
    shape = (4, 4, 4)
    a = ttb.sptensor.from_data(subs, vals, shape)
    assert isinstance(ttb_utils.tt_subsubsref(a, [1]), ttb.sptensor)
    # TODO need to understand behavior better
    assert True


@pytest.mark.indevelopment
def test_tt_intvec2str_valid():
    """This function is slotted to be removed because it is probably unnecessary in python"""
    v = np.array([1, 2, 3])
    assert ttb_utils.tt_intvec2str(v) == "[1 2 3]"


@pytest.mark.indevelopment
def test_tt_sizecheck_empty():
    assert ttb_utils.tt_sizecheck(())


@pytest.mark.indevelopment
def test_tt_sizecheck_valid():
    assert ttb_utils.tt_sizecheck((2, 2, 2))


@pytest.mark.indevelopment
def test_tt_sizecheck_invalid():
    # Float
    assert not ttb_utils.tt_sizecheck((1.0, 2, 2))
    # Too many dimensions
    assert not ttb_utils.tt_sizecheck(np.array([[2, 2], [2, 2]]))
    # Nan
    assert not ttb_utils.tt_sizecheck((np.nan, 2, 2))
    # Inf
    assert not ttb_utils.tt_sizecheck((np.inf, 2, 2))
    # Zero
    assert not ttb_utils.tt_sizecheck((0, 2, 2))


@pytest.mark.indevelopment
def test_tt_sizecheck_errorMessage():
    # Raise when nargout == 0
    with pytest.raises(AssertionError) as excinfo:
        ttb_utils.tt_sizecheck((1.0, 2, 2), nargout=False)
    assert "Size must be a row vector of real positive integers" in str(excinfo)


@pytest.mark.indevelopment
def test_tt_subscheck_empty():
    assert ttb_utils.tt_subscheck(np.array([]))


@pytest.mark.indevelopment
def test_tt_subscheck_valid():
    assert ttb_utils.tt_subscheck(np.array([[2, 2], [2, 2]]))


@pytest.mark.indevelopment
def test_tt_subscheck_invalid():
    # Too Few Dims
    assert not ttb_utils.tt_subscheck(np.array([2]))
    # Too Many Dims
    assert not ttb_utils.tt_subscheck(np.array([[[2]]]))
    # Less than 0
    assert not ttb_utils.tt_subscheck(np.array([[-1, 2], [2, 2]]))
    # Nan
    assert not ttb_utils.tt_subscheck(np.array([[np.nan, 2], [2, 2]]))
    # Inf
    assert not ttb_utils.tt_subscheck(np.array([[np.inf, 2], [2, 2]]))
    # Non-int
    assert not ttb_utils.tt_subscheck(np.array([[1.0, 2], [2, 2]]))


@pytest.mark.indevelopment
def test_tt_subscheck_errorMessage():
    # Raise when nargout == 0
    with pytest.raises(AssertionError) as excinfo:
        ttb_utils.tt_subscheck(np.array([1.0, 2, 2]), nargout=False)
    assert "Subscripts must be a matrix of real positive integers" in str(excinfo)


@pytest.mark.indevelopment
def test_tt_valscheck_empty():
    assert ttb_utils.tt_valscheck(np.array([]))


@pytest.mark.indevelopment
def test_tt_valscheck_valid():
    assert ttb_utils.tt_valscheck(np.array([[0.5], [1.5], [2.5]]))


@pytest.mark.indevelopment
def test_tt_valscheck_invalid():
    # Row array
    assert not ttb_utils.tt_valscheck(np.array([2, 2]))
    # Matrix, too many dimensions
    assert not ttb_utils.tt_valscheck(np.array([[2, 2], [2, 2]]))


@pytest.mark.indevelopment
def test_tt_valscheck_errorMessage():
    # Raise when nargout == 0
    with pytest.raises(AssertionError) as excinfo:
        ttb_utils.tt_valscheck(np.array([2, 2]), nargout=False)
    assert "Values must be in array" in str(excinfo)


@pytest.mark.indevelopment
def test_isrow_empty():
    assert not ttb_utils.isrow(np.array([[]]))


@pytest.mark.indevelopment
def test_isrow_valid():
    assert ttb_utils.isrow(np.array([[2, 2, 2]]))


@pytest.mark.indevelopment
def test_isrow_invalid():
    # 2 x 2 Matrix
    assert not ttb_utils.isrow(np.array([[2, 2], [2, 2]]))
    # Column vector
    assert not ttb_utils.isrow(np.array([[2, 2, 2]]).T)


@pytest.mark.indevelopment
def test_isvector_empty():
    assert ttb_utils.isvector(np.array([[]]))


@pytest.mark.indevelopment
def test_isvector_valid():
    assert ttb_utils.isvector(np.array([[2, 2, 2]]))
    assert ttb_utils.isvector(np.array([[2, 2, 2]]).T)


@pytest.mark.indevelopment
def test_isvector_invalid():
    # 2 x 2 Matrix
    assert not ttb_utils.isvector(np.array([[2, 2], [2, 2]]))


@pytest.mark.indevelopment
def test_islogical_empty():
    assert not ttb_utils.islogical(np.array([[]]))


@pytest.mark.indevelopment
def test_islogical_valid():
    assert ttb_utils.islogical(True)


@pytest.mark.indevelopment
def test_islogical_invalid():
    assert not ttb_utils.islogical(np.array([[2, 2, 2]]))
    assert not ttb_utils.islogical(1.1)
    assert not ttb_utils.islogical(0)


def test_get_index_variant_linear():
    assert ttb_utils.get_index_variant(1) == ttb_utils.IndexVariant.LINEAR
    assert ttb_utils.get_index_variant(1.0) == ttb_utils.IndexVariant.LINEAR
    assert ttb_utils.get_index_variant(slice(1, 5)) == ttb_utils.IndexVariant.LINEAR
    assert ttb_utils.get_index_variant(np.int32(2)) == ttb_utils.IndexVariant.LINEAR
    assert (
        ttb_utils.get_index_variant(np.array([1, 2, 3]))
        == ttb_utils.IndexVariant.LINEAR
    )
    assert ttb_utils.get_index_variant([1, 2, 3]) == ttb_utils.IndexVariant.LINEAR


def test_get_index_variant_subscripts():
    assert (
        ttb_utils.get_index_variant(np.array([[1, 2, 3]]))
        == ttb_utils.IndexVariant.SUBSCRIPTS
    )


def test_get_index_variant_subtensor():
    assert ttb_utils.get_index_variant((1, 2, 3)) == ttb_utils.IndexVariant.SUBTENSOR


def test_get_index_variant_unknown():
    assert ttb_utils.get_index_variant("a") == ttb_utils.IndexVariant.UNKNOWN
