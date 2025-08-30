# Copyright 2024 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.
from __future__ import annotations

import logging

import numpy as np
import pytest

import pyttb as ttb
import pyttb.pyttb_utils as ttb_utils
from pyttb.pyttb_utils import parse_one_d, parse_shape


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


def test_tt_dimscheck():
    #  Empty
    rdims, ridx = ttb_utils.tt_dimscheck(6, dims=None)
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


def test_tt_ismember_rows():
    a = np.array([[4, 6], [1, 9], [2, 6]])
    b = np.array(
        [[1, 7], [1, 8], [2, 6], [2, 1], [2, 4], [4, 6], [4, 7], [5, 9], [5, 2], [5, 1]]
    )
    valid, result = ttb_utils.tt_ismember_rows(a, b)
    assert np.array_equal(result, np.array([5, -1, 2]))
    assert np.all(result[valid] >= 0)
    assert np.all(result[~valid] < 0)
    valid, result = ttb_utils.tt_ismember_rows(b, a)
    assert np.array_equal(
        result,
        np.array([-1, -1, 2, -1, -1, 0, -1, -1, -1, -1]),
    )
    assert np.all(result[valid] >= 0)
    assert np.all(result[~valid] < 0)


def test_tt_irenumber():
    # Constant shouldn't effect performance
    const = 1

    subs = np.array([[const, const, 0], [const, const, 1]])
    vals = np.array([[0.5], [1.5]])
    shape = (4, 4, 4)
    sptensorInstance = ttb.sptensor(subs, vals, shape)
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


def test_tt_sub2ind_valid():
    subs = np.array([[0, 0, 0], [1, 1, 1], [3, 3, 3]])
    idx = np.array([0, 21, 63])
    siz = np.array([4, 4, 4])
    assert np.array_equal(ttb_utils.tt_sub2ind(siz, subs), idx)

    empty = np.array([])
    assert np.array_equal(ttb_utils.tt_sub2ind(siz, empty), empty)


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


def test_tt_subsubsref_valid():
    subs = np.array([[0, 0, 0], [1, 1, 1], [3, 3, 3]])
    vals = np.array([[0], [21], [63]])
    shape = (4, 4, 4)
    a = ttb.sptensor(subs, vals, shape)
    assert isinstance(ttb_utils.tt_subsubsref(a, [1]), ttb.sptensor)
    # TODO need to understand behavior better
    assert True


def test_tt_sizecheck_empty():
    assert ttb_utils.tt_sizecheck(())


def test_tt_sizecheck_valid():
    assert ttb_utils.tt_sizecheck((2, 2, 2))


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


def test_tt_sizecheck_errorMessage():
    # Raise when nargout == 0
    with pytest.raises(AssertionError) as excinfo:
        ttb_utils.tt_sizecheck((1.0, 2, 2), nargout=False)
    assert "Size must be a row vector of real positive integers" in str(excinfo)


def test_tt_subscheck_empty():
    assert ttb_utils.tt_subscheck(np.array([]))


def test_tt_subscheck_valid():
    assert ttb_utils.tt_subscheck(np.array([[2, 2], [2, 2]]))


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


def test_tt_subscheck_errorMessage():
    # Raise when nargout == 0
    with pytest.raises(AssertionError) as excinfo:
        ttb_utils.tt_subscheck(np.array([1.0, 2, 2]), nargout=False)
    assert "Subscripts must be a matrix of real positive integers" in str(excinfo)


def test_tt_valscheck_empty():
    assert ttb_utils.tt_valscheck(np.array([]))


def test_tt_valscheck_valid():
    assert ttb_utils.tt_valscheck(np.array([[0.5], [1.5], [2.5]]))


def test_tt_valscheck_invalid():
    # Row array
    assert not ttb_utils.tt_valscheck(np.array([2, 2]))
    # Matrix, too many dimensions
    assert not ttb_utils.tt_valscheck(np.array([[2, 2], [2, 2]]))


def test_tt_valscheck_errorMessage():
    # Raise when nargout == 0
    with pytest.raises(AssertionError) as excinfo:
        ttb_utils.tt_valscheck(np.array([2, 2]), nargout=False)
    assert "Values must be in array" in str(excinfo)


def test_isrow_empty():
    assert not ttb_utils.isrow(np.array([[]]))


def test_isrow_valid():
    assert ttb_utils.isrow(np.array([[2, 2, 2]]))


def test_isrow_invalid():
    # 2 x 2 Matrix
    assert not ttb_utils.isrow(np.array([[2, 2], [2, 2]]))
    # Column vector
    assert not ttb_utils.isrow(np.array([[2, 2, 2]]).T)


def test_isvector_empty():
    assert ttb_utils.isvector(np.array([[]]))


def test_isvector_valid():
    assert ttb_utils.isvector(np.array([[2, 2, 2]]))
    assert ttb_utils.isvector(np.array([[2, 2, 2]]).T)


def test_isvector_invalid():
    # 2 x 2 Matrix
    assert not ttb_utils.isvector(np.array([[2, 2], [2, 2]]))


def test_islogical_empty():
    assert not ttb_utils.islogical(np.array([[]]))


def test_islogical_valid():
    assert ttb_utils.islogical(True)


def test_islogical_invalid():
    assert not ttb_utils.islogical(np.array([[2, 2, 2]]))
    assert not ttb_utils.islogical(1.1)
    assert not ttb_utils.islogical(0)


def test_get_index_variant_linear():
    assert ttb_utils.get_index_variant(1) == ttb_utils.IndexVariant.LINEAR
    assert ttb_utils.get_index_variant(1.0) == ttb_utils.IndexVariant.UNKNOWN
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


def test_parse_shape():
    with pytest.raises(ValueError):
        parse_shape(np.ones((4,), dtype=float))
    with pytest.raises(ValueError):
        parse_shape(np.ones((4, 2), dtype=int))


def test_parse_one_d():
    with pytest.raises(ValueError):
        parse_one_d(np.ones((4, 2)))
