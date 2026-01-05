"""PYTTB shared utilities across tensor types."""

# Copyright 2025 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from __future__ import annotations

from collections.abc import Iterable, Sequence
from enum import Enum
from math import prod
from typing import (
    Any,
    Literal,
    cast,
    get_args,
    overload,
)

import numpy as np
from scipy import sparse

import pyttb as ttb

Shape = int | Iterable[int]
"""Shape represents the object size or dimensions. It can be specified as
either a single integer or an iterable of integers, which will be normalized
to a tuple internally."""

OneDArray = int | float | Iterable[int] | Iterable[float] | np.ndarray
"""OneDArray represents any one-dimensional array, which can be a single
integer or float, and iterable of integerss or floats, or a NumPy array."""

MemoryLayout = Literal["F"] | Literal["C"]
"""MemoryLayout is the set of options for the layout of a tensor.
It can be "F", meaning Fortran ordered and analogous to column-major for matrices,
or "C", meaning C ordered and analogous to row-major for matrices.
Order "F" is how tensors are stored in MATLAB, and order "C" is the default
for NumPy arrays."""


def tt_union_rows(MatrixA: np.ndarray, MatrixB: np.ndarray) -> np.ndarray:
    """Reproduce functionality of MATLABS intersect(a,b,'rows').

    Parameters
    ----------
    MatrixA:
        First matrix.
    MatrixB:
        Second matrix.

    Returns
    -------
    location:
        List of intersection indices

    Examples
    --------
    >>> a = np.array([[1, 2], [3, 4]])
    >>> b = np.array([[0, 0], [1, 2], [3, 4], [0, 0]])
    >>> tt_union_rows(a, b)
    array([[0, 0],
           [1, 2],
           [3, 4]])
    """
    # TODO ismember and unique are very similar in function
    if MatrixA.size > 0:
        MatrixAUnique, idxA = np.unique(MatrixA, axis=0, return_index=True)
    else:
        MatrixA = MatrixAUnique = np.empty(shape=MatrixB.shape)
        idxA = np.array([], dtype=int)
    if MatrixB.size > 0:
        MatrixBUnique, idxB = np.unique(MatrixB, axis=0, return_index=True)
    else:
        MatrixB = MatrixBUnique = np.empty(shape=MatrixA.shape)
        idxB = np.array([], dtype=int)
    _, location = tt_ismember_rows(
        MatrixBUnique[np.argsort(idxB)], MatrixAUnique[np.argsort(idxA)]
    )
    union = np.vstack(
        (MatrixB[np.sort(idxB[np.where(location < 0)])], MatrixA[np.sort(idxA)])
    )
    return union


@overload
def tt_dimscheck(
    N: int,
    M: None = None,
    dims: OneDArray | None = None,
    exclude_dims: OneDArray | None = None,
) -> tuple[np.ndarray, None]: ...  # pragma: no cover see coveragepy/issues/970


@overload
def tt_dimscheck(
    N: int,
    M: int,
    dims: OneDArray | None = None,
    exclude_dims: OneDArray | None = None,
) -> tuple[np.ndarray, np.ndarray]: ...  # pragma: no cover see coveragepy/issues/970


def tt_dimscheck(  # noqa: PLR0912
    N: int,
    M: int | None = None,
    dims: OneDArray | None = None,
    exclude_dims: OneDArray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Preprocess dimensions for tensor operations.

    Parameters
    ----------
    N: Tensor order
    M: Num of multiplicands
    dims: Dimensions to check
    exclude_dims: Check all dimensions but these. (Mutually exclusive with dims)

    Returns
    -------
    sdims: New dimensions
    vidx: Index into the multiplicands (if M defined).

    Examples
    --------
    # Default captures all dims and no index

    >>> rdims, _ = tt_dimscheck(6)
    >>> np.array_equal(rdims, np.arange(6))
    True

    # Exclude single dim and still no index

    >>> rdims, _ = tt_dimscheck(6, exclude_dims=np.array([5]))
    >>> np.array_equal(rdims, np.arange(5))
    True

    # Exclude single dim and number of multiplicands equals resulting size

    >>> rdims, ridx = tt_dimscheck(6, 5, exclude_dims=np.array([0]))
    >>> np.array_equal(rdims, np.array([1, 2, 3, 4, 5]))
    True
    >>> np.array_equal(ridx, np.arange(0, 5))
    True
    """
    if dims is not None and exclude_dims is not None:
        raise ValueError("Either specify dims to include or exclude, but not both")
    if dims is not None:
        dims = parse_one_d(dims)
    if exclude_dims is not None:
        exclude_dims = parse_one_d(exclude_dims)

    dim_array: np.ndarray = np.empty((1,))

    # Explicit exclude to resolve ambiguous -0
    if exclude_dims is not None:
        # Check that all members in range
        valid_indices = np.isin(exclude_dims, np.arange(0, N))
        if not np.all(valid_indices):
            invalid_indices = np.logical_not(valid_indices)
            raise ValueError(
                f"Exclude dims provided: {exclude_dims} "
                f"but, {exclude_dims[invalid_indices]} were out of valid range"
                f"[0,{N}]"
            )
        dim_array = np.setdiff1d(np.arange(0, N), exclude_dims)

    # Fix empty case
    # if (dims is None or dims.size == 0) and exclude_dims is None:
    if dims is None and exclude_dims is None:
        dim_array = np.arange(0, N)
    elif isinstance(dims, np.ndarray):
        dim_array = dims

    # Catch minus case to avoid silent errors
    if np.any(dim_array < 0):
        raise ValueError(
            "Negative dims aren't allowed in pyttb, see exclude_dims argument instead"
        )

    # Save dimensions of dims
    P = len(dim_array)

    # Reorder dims from smallest to largest (this matters in particular for the vector
    # multiplicand case, where the order affects the result)
    sidx = np.argsort(dim_array)
    sdims = dim_array[sidx]
    vidx = None

    if M is not None:
        # Can't have more multiplicands than dimensions
        if M > N:
            assert False, "Cannot have more multiplicands than dimensions"

        # Check that the number of multiplicands must either be full dimensional or
        # equal to the specified dimensions (M==N) or M(==P) respectively
        if M not in (N, P):
            assert False, "Invalid number of multiplicands"

        # Check sizes to determine how to index multiplicands
        if P == M:
            # Case 1: Number of items in dims and number of multiplicands are equal;
            # therefore, index in order of sdims
            vidx = sidx
        else:
            # Case 2: Number of multiplicands is equal to the number of dimensions of
            # tensor; therefore, index multiplicands by dimensions in dims argument.
            vidx = sdims

    return sdims, vidx


def tt_setdiff_rows(MatrixA: np.ndarray, MatrixB: np.ndarray) -> np.ndarray:
    """Reproduce functionality of MATLABS setdiff(a,b,'rows').

    Parameters
    ----------
    MatrixA:
        First matrix.
    MatrixB:
        Second matrix.

    Returns
    -------
    List of set difference indices.
    """
    # TODO intersect and setdiff are very similar in function
    if MatrixA.size > 0:
        MatrixAUnique, idxA = np.unique(MatrixA, axis=0, return_index=True)
    else:
        MatrixAUnique = idxA = np.array([], dtype=int)
    if MatrixB.size > 0:
        MatrixBUnique, idxB = np.unique(MatrixB, axis=0, return_index=True)
    else:
        MatrixBUnique = idxB = np.array([], dtype=int)
    valid, location = tt_ismember_rows(
        MatrixBUnique[np.argsort(idxB)], MatrixAUnique[np.argsort(idxA)]
    )
    return np.setdiff1d(idxA, location[valid])


def tt_intersect_rows(MatrixA: np.ndarray, MatrixB: np.ndarray) -> np.ndarray:
    """Reproduce functionality of MATLABS intersect(a,b,'rows').

    Parameters
    ----------
    MatrixA:
        First matrix.
    MatrixB:
        Second matrix.

    Returns
    -------
    location:
        List of intersection indices.

    Examples
    --------
    >>> a = np.array([[1, 2], [3, 4]])
    >>> b = np.array([[0, 0], [1, 2], [3, 4], [0, 0]])
    >>> tt_intersect_rows(a, b)
    array([0, 1])
    >>> tt_intersect_rows(b, a)
    array([1, 2])
    """
    # TODO ismember and unique are very similar in function
    if MatrixA.size > 0:
        MatrixAUnique, idxA = np.unique(MatrixA, axis=0, return_index=True)
    else:
        MatrixAUnique = idxA = np.array([], dtype=int)
    if MatrixB.size > 0:
        MatrixBUnique, idxB = np.unique(MatrixB, axis=0, return_index=True)
    else:
        MatrixBUnique = idxB = np.array([], dtype=int)
    valid, location = tt_ismember_rows(
        MatrixBUnique[np.argsort(idxB)], MatrixAUnique[np.argsort(idxA)]
    )
    return location[valid]


def tt_irenumber(
    t: ttb.sptensor, shape: tuple[int, ...], number_range: Sequence[IndexType]
) -> np.ndarray:
    """Renumber indices for sptensor __setitem__.

    Parameters
    ----------
    t:
        Sptensor we are trying to assign from
    shape:
        Shape of destination tensor
    number_range:
        Key from __setitem__ for destination tensor

    Returns
    -------
    Subscripts for sptensor assignment
    """
    nz = t.nnz
    if nz == 0:
        newsubs = np.array([])
        return newsubs

    newsubs = t.subs.astype(int)
    for i, r in enumerate(number_range):
        if isinstance(r, slice):
            start = r.start or 0
            stop = r.stop or shape[i]
            newsubs[:, i] = np.arange(start, stop + 1)[newsubs[:, i]]
        elif isinstance(r, int):
            # This appears to be inserting new keys as rows to our subs here
            newsubs = np.insert(newsubs, obj=i, values=r, axis=1)
        else:
            if not isinstance(r, np.ndarray):
                r = np.array(r)  # noqa: PLW2901
            newsubs[:, i] = r[newsubs[:, i]]
    return newsubs


def tt_renumber(
    subs: np.ndarray, shape: tuple[int, ...], number_range: Sequence[IndexType]
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Renumber indices for sptensor __getitem__.

    [NEWSUBS,NEWSZ] = RENUMBER(SUBS,SZ,RANGE) takes a set of
    original subscripts SUBS with entries from a tensor of size
    SZ. All the entries in SUBS are assumed to be within the
    specified RANGE. These subscripts are then renumbered so that,
    in dimension i, the numbers range from 1:numel(RANGE(i)).

    Parameters
    ----------
    subs:
        Original subscripts for source tensor.
    shape:
        Shape of source tensor.
    number_range:
        Key from __getitem__ for tensor.

    Returns
    -------
    newsubs:
        Updated subscripts.
    newshape:
        Resulting shape.
    """
    newshape = np.array(shape)
    newsubs = subs
    for i in range(len(shape)):
        if not number_range[i] == slice(None, None, None):
            if subs.size == 0:
                if not isinstance(number_range[i], slice):
                    # This should be statically determinable but mypy unhappy
                    # without intermediate
                    number_range_i = number_range[i]
                    if isinstance(number_range_i, (int, float, np.integer)):
                        newshape[i] = number_range_i
                    else:
                        assert not isinstance(number_range_i, (int, slice, np.integer))
                        newshape[i] = len(number_range_i)
                else:
                    # TODO get this length without generating the range
                    #   This should be statically determinable but mypy unhappy
                    #   without assert
                    number_range_i = number_range[i]
                    assert isinstance(number_range_i, slice)
                    newshape[i] = len(range(shape[i])[number_range_i])
            else:
                newsubs[:, i], newshape[i] = tt_renumberdim(
                    subs[:, i], shape[i], number_range[i]
                )

    return newsubs, tuple(newshape)


def tt_renumberdim(
    idx: np.ndarray, shape: int, number_range: IndexType
) -> tuple[int, int]:
    """Renumber a single dimension.

    Helper function for RENUMBER.

    Parameters
    ----------
    idx:
    shape:
    number_range:

    Returns
    -------
    newidx:
    newshape:
    """
    # Determine the size of the new range
    if isinstance(number_range, (int, np.integer)):
        number_range = [int(number_range)]
        newshape = 0
    elif isinstance(number_range, slice):
        number_range = list(range(shape))[number_range]
        newshape = len(number_range)
    elif isinstance(number_range, (Sequence, np.ndarray)):
        newshape = len(number_range)
    else:
        raise ValueError(f"Bad number range: {number_range}")

    # Create map from old range to the new range
    idx_map = np.zeros(shape=shape)
    for i in range(newshape):
        idx_map[number_range[i]] = int(i)

    # Do the mapping
    newidx = idx_map[idx]
    return newidx, newshape


# TODO make more efficient
# https://stackoverflow.com/questions/22699756/python-version-of-ismember-with-rows-and-index
# For thoughts on how to speed this up
def tt_ismember_rows(
    search: np.ndarray, source: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Find location of search rows in source array.

    Parameters
    ----------
    search:
        Array to match to source array.
    source:
        Array to be matched against.

    Returns
    -------
    matched:
        len(results)==len(matched) Boolean for indexing matched results.
    results:
        search.size==results.size,
        if search[0,:] == source[3,:], then results[0] = 3
        if exists i such that search[i,:] != source[j,:] for any j, then results[i] = -1

    Examples
    --------
    >>> a = np.array([[4, 6], [1, 9], [2, 6]])
    >>> b = np.array([[2, 6], [2, 1], [2, 4], [4, 6], [4, 7], [5, 9], [5, 2], [5, 1]])
    >>> matched, results = tt_ismember_rows(a, b)
    >>> print(results)
    [ 3 -1  0]
    >>> print(matched)
    [ True False  True]

    """
    matched = np.zeros(shape=search.shape[0], dtype=bool)
    results = np.ones(shape=search.shape[0]) * -1
    if search.size == 0:
        return matched, results.astype(int)
    if source.size == 0:
        return matched, results.astype(int)
    (row_idx, col_idx) = np.nonzero(np.all(source == search[:, np.newaxis], axis=2))
    results[row_idx] = col_idx
    matched[row_idx] = True
    return matched, results.astype(int)


def tt_ind2sub(
    shape: tuple[int, ...],
    idx: np.ndarray,
    order: MemoryLayout = "F",
) -> np.ndarray:
    """
    Multiple subscripts from linear indices.

    Parameters
    ----------
    shape: Shape of tensor indexing into.
    idx: Array of linear indices into the tensor.

    Returns
    -------
    Multi-dimensional indices for the tensor.

    Example
    -------
    >>> shape = (2, 2, 2)
    >>> linear_indices = np.array([0, 1])
    >>> tt_ind2sub(shape, linear_indices)
    array([[0, 0, 0],
           [1, 0, 0]])
    """
    if idx.size == 0:
        return np.empty(shape=(0, len(shape)), dtype=int)
    idx[idx < 0] += prod(shape)  # Handle negative indexing as simply as possible
    return np.array(np.unravel_index(idx, shape, order=order)).transpose()


def tt_subsubsref(obj: np.ndarray, s: Any) -> float | np.ndarray:  # noqa: ARG001
    """Helper function for tensor toolbox subsref.

    Parameters
    ----------
    obj:
        Tensor Data Structure
    s:
        Reference into tensor

    Returns
    -------
    Still uncertain to this functionality
    """  # noqa: D401
    # TODO figure out when subsref yields key of length>1 for now ignore this logic and
    #  just return
    # if len(s) == 1:
    #    return obj
    # else:
    #   return obj[s[1:]]
    if isinstance(obj, np.ndarray) and obj.size == 1:
        # TODO: Globally figure out why typing thinks item is a string
        return cast("float", obj.item())
    return obj


def tt_sub2ind(
    shape: tuple[int, ...],
    subs: np.ndarray,
    order: MemoryLayout = "F",
) -> np.ndarray:
    """Convert multidimensional subscripts to linear indices.

    Parameters
    ----------
    shape:
        Shape of tensor
    subs:
        Subscripts for tensor
    order:
        Memory layout

    Examples
    --------
    >>> shape = (2, 2, 2)
    >>> full_indices = np.array([[0, 0, 0], [1, 0, 0]], dtype=int)
    >>> tt_sub2ind(shape, full_indices)
    array([0, 1])

    See Also
    --------
    :func:`tt_ind2sub`:
    """
    if subs.size == 0:
        return np.array([])
    idx = np.ravel_multi_index(tuple(subs.transpose()), shape, order=order)
    return idx


def tt_sizecheck(shape: tuple[int, ...], nargout: bool = True) -> bool:
    """
    TT_SIZECHECK Checks that the shape is valid.

    TT_SIZECHECK(S) throws an error if S is not a valid shape tuple,
    which means that it is a row vector with strictly positive,
    real-valued, finite integer values.

    Parameters
    ----------
    shape:
        Shape of tensor
    nargout:
        Controls if response returned or just acts as assert

    Returns
    -------
    bool

    Examples
    --------
    >>> tt_sizecheck((0, -1, 2))
    False
    >>> tt_sizecheck((1, 1, 1))
    True

    See Also
    --------
    :func:`tt_subscheck`:
    """
    siz = np.array(shape)
    if (
        len(siz.shape) == 1
        and all(np.isfinite(siz))
        and issubclass(siz.dtype.type, np.integer)
        and all(siz > 0)
    ):
        ok = True
    elif siz.size == 0:
        ok = True
    else:
        ok = False

    if not ok and not nargout:
        assert False, "Size must be a row vector of real positive integers"
    return ok


def tt_subscheck(subs: np.ndarray, nargout: bool = True) -> bool:
    """
    TT_SUBSCHECK Checks for valid subscripts.

    TT_SUBSCHECK(S) throws an error if S is not a valid subscript
    array, which means that S is a matrix of real-valued, finite,
    positive, integer subscripts.

    Parameters
    ----------
    subs:
        Subs of tensor
    nargout:
        Controls if response returned or just acts as assert

    Returns
    -------
    bool

    Examples
    --------
    >>> tt_subscheck(np.array([[2, 2], [3, 3]]))
    True
    >>> tt_subscheck(np.array([[2, 2], [3, -1]]))
    False

    See Also
    --------
    :func:`tt_sizecheck`:
    :func:`tt_valscheck`:
    """
    if subs.size == 0:
        ok = True
    elif (
        len(subs.shape) == 2
        and (np.isfinite(subs)).all()
        and issubclass(subs.dtype.type, np.integer)
        and (subs >= 0).all()
    ):
        ok = True
    else:
        ok = False

    if not ok and not nargout:
        assert False, "Subscripts must be a matrix of real positive integers"
    return ok


def tt_valscheck(vals: np.ndarray, nargout: bool = True) -> bool:
    """
    TT_VALSCHECK Checks for valid values.

    TT_VALSCHECK(S) throws an error if S is not a valid values
    array, which means that S is a column array.

    Parameters
    ----------
    vals:
        Values of tensor
    nargout:
        Controls if response returned or just acts as assert

    Returns
    -------
    bool

    Examples
    --------
    >>> tt_valscheck(np.array([[1], [2]]))
    True
    >>> tt_valscheck(np.array([[1, 2, 3], [2, 2, 2]]))
    False

    See Also
    --------
    :func:`tt_sizecheck`:
    :func:`tt_subscheck`:
    """
    if vals.size == 0:
        ok = True
    elif len(vals.shape) == 2 and vals.shape[1] == 1:
        ok = True
    else:
        ok = False
    if not ok and not nargout:
        assert False, f"Values must be in array but got {vals}"
    return ok


def isrow(v: np.ndarray) -> bool:
    """
    ISROW Checks if vector is a row vector.

    ISROW(V) returns True if V is a row vector; otherwise returns False.

    Parameters
    ----------
    v:
        Vector input

    Examples
    --------
    >>> isrow(np.array([[1, 2]]))
    True
    >>> isrow(np.array([[1, 2], [3, 4]]))
    False
    """
    return v.ndim == 2 and v.shape[0] == 1 and v.shape[1] >= 1


def isvector(a: np.ndarray) -> bool:
    """
    ISVECTOR Checks if vector is a row vector.

    ISVECTOR(A) returns True if A is a vector; otherwise returns False.

    Parameters
    ----------
    a:

    Returns
    -------
    bool
    """
    return a.ndim == 1 or (a.ndim == 2 and (a.shape[0] == 1 or a.shape[1] == 1))


# TODO: this is a challenge, since it may need to apply to either Python built in types
#  or numpy types
def islogical(a: np.ndarray) -> bool:
    """
    ISLOGICAL Checks if vector is a logical vector.

    ISLOGICAL(A) returns True if A is a logical array; otherwise returns False.

    Parameters
    ----------
    a:

    Returns
    -------
    bool
    """
    return isinstance(a, bool)


# Adding all sorts of index support here, might consider splitting out to
# more specific file later


class IndexVariant(Enum):
    """Methods for indexing entries of tensors."""

    UNKNOWN = 0
    LINEAR = 1
    SUBTENSOR = 2
    SUBSCRIPTS = 3


# We probably want to create a specific file for utility types
LinearIndexType = int | np.integer | slice
IndexType = LinearIndexType | Sequence[int] | np.ndarray


def get_index_variant(indices: IndexType) -> IndexVariant:
    """Decide on intended indexing variant. No correctness checks.

    See getitem or setitem in :class:`pyttb.tensor` for elaboration of the
    various indexing options.
    """
    variant = IndexVariant.UNKNOWN
    if isinstance(indices, get_args(LinearIndexType)):
        variant = IndexVariant.LINEAR
    elif isinstance(indices, np.ndarray):
        # TODO this is technically slightly stricter than what
        #  we currently have but probably clearer
        if len(indices.shape) == 1:
            variant = IndexVariant.LINEAR
        else:
            variant = IndexVariant.SUBSCRIPTS
    elif isinstance(indices, tuple):
        variant = IndexVariant.SUBTENSOR
    elif isinstance(indices, Sequence) and isinstance(indices[0], int):
        # TODO this is slightly redundant/inefficient
        key = np.array(indices)
        if len(key.shape) == 1 or key.shape[1] == 1:
            variant = IndexVariant.LINEAR
    return variant


def get_mttkrp_factors(
    U: ttb.ktensor | Sequence[np.ndarray], n: int | np.integer, ndims: int
) -> Sequence[np.ndarray]:
    """Apply standard checks and type conversions for mttkrp factors."""
    if isinstance(U, ttb.ktensor):
        U = U.copy()
        # Absorb lambda into one of the factors but not the one that is skipped
        if n == 0:
            U.redistribute(1)
        else:
            U.redistribute(0)

        # Extract the factor matrices
        U = U.factor_matrices

    assert isinstance(U, (Sequence, np.ndarray)), (
        "Second argument must be a sequence of numpy.ndarray's or a ktensor"
    )

    assert len(U) == ndims, "List of factor matrices is the wrong length"

    return U


def gather_wrap_dims(
    ndims: int,
    rdims: np.ndarray | None = None,
    cdims: np.ndarray | None = None,
    cdims_cyclic: Literal["fc"] | Literal["bc"] | Literal["t"] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract tensor modes mapped to rows and columns for matricized tensors.

    Parameters
    ----------
    ndims:
        Number of dimensions.
    rdims:
        Mapping of row indices.
    cdims:
        Mapping of column indices.
    cdims_cyclic:
        When only rdims is specified maps a single rdim to the rows and
            the remaining dimensions span the columns. _fc_ (forward cyclic[1]_)
            in the order range(rdims,self.ndims()) followed by range(0, rdims).
            _bc_ (backward cyclic[2]_) range(rdims-1, -1, -1) then
            range(self.ndims(), rdims, -1).

    Notes
    -----
    Forward cyclic is defined by Kiers [1]_ and backward cyclic is defined by
        De Lathauwer, De Moor, and Vandewalle [2]_.

    References
    ----------
    .. [1] KIERS, H. A. L. 2000. Towards a standardized notation and terminology
           in multiway analysis. J. Chemometrics 14, 105-122.
    .. [2] DE LATHAUWER, L., DE MOOR, B., AND VANDEWALLE, J. 2000b. On the best
           rank-1 and rank-(R1, R2, ... , RN ) approximation of higher-order
           tensors. SIAM J. Matrix Anal. Appl. 21, 4, 1324-1342.
    """
    alldims = np.array([range(ndims)])

    if rdims is not None and cdims is None:
        # Single row mapping
        if len(rdims) == 1 and cdims_cyclic is not None:
            # TODO we should be able to remove this since we can just specify
            #   cdims alone
            if cdims_cyclic == "t":
                cdims = rdims
                rdims = np.setdiff1d(alldims, rdims)
            elif cdims_cyclic == "fc":
                cdims = np.array(
                    [i for i in range(rdims[0] + 1, ndims)]
                    + [i for i in range(rdims[0])]
                )
            elif cdims_cyclic == "bc":
                cdims = np.array(
                    [i for i in range(rdims[0] - 1, -1, -1)]
                    + [i for i in range(ndims - 1, rdims[0], -1)]
                )
            else:
                assert False, (
                    'Unrecognized value for cdims_cyclic pattern, must be "fc" or "bc".'
                )
        else:
            # Multiple row mapping
            cdims = np.setdiff1d(alldims, rdims)

    elif rdims is None and cdims is not None:
        rdims = np.setdiff1d(alldims, cdims)

    assert rdims is not None and cdims is not None
    return rdims.astype(int), cdims.astype(int)


def np_to_python(
    iterable: Iterable,
) -> Iterable:
    """Convert a structure containing numpy scalars to pure python types.

    Mostly useful for prettier printing post numpy 2.0.

    Parameters
    ----------
    iterable:
        Structure potentially containing numpy scalars.
    """
    output_type = type(iterable)
    return output_type(  # type: ignore [call-arg]
        element.item() if isinstance(element, np.generic) else element
        for element in iterable
    )


def parse_shape(shape: Shape) -> tuple[int, ...]:
    """Parse flexible type into shape tuple.

    Examples
    --------
    >>> integer_shape = 4
    >>> parse_shape(integer_shape)
    (4,)
    >>> flat_numpy_shape = np.ones((4,), dtype=int)
    >>> parse_shape(flat_numpy_shape)
    (1, 1, 1, 1)
    >>> stacked_numpy_shape = np.ones((4, 1, 1), dtype=int)
    >>> parse_shape(stacked_numpy_shape)
    (1, 1, 1, 1)
    >>> list_shape = [1, 1, 1, 1]
    >>> parse_shape(list_shape)
    (1, 1, 1, 1)
    """
    # FIXME do we care to map numpy ints to python ints?
    if isinstance(shape, (int, np.integer)):
        return (shape,)
    if isinstance(shape, np.ndarray):
        if not np.issubdtype(shape.dtype, np.integer):
            raise ValueError("Numpy arrays used as shapes must be integer valued")
        squeezed_shape = shape.squeeze()
        if squeezed_shape.ndim == 0:
            # If it's an array containing a single scalar
            return (int(squeezed_shape),)
        if squeezed_shape.ndim > 1:
            raise ValueError(
                "Numpy arrays used as shapes can only have one non-trivial dimension"
            )
        return tuple(map(int, squeezed_shape))

    shape = tuple(shape)
    if not all(isinstance(ele, (int, np.integer)) for ele in shape):
        raise ValueError("Shapes entries must be integers")
    return shape


def parse_one_d(maybe_vector: OneDArray) -> np.ndarray:
    """Parse flexible type into numpy array.

    Examples
    --------
    >>> int_scalar = 1
    >>> parse_one_d(int_scalar)
    array([1])
    >>> np_int_scalar = np.int8(1)
    >>> parse_one_d(np_int_scalar)
    array([1], dtype=int8)
    >>> float_scalar = 1.0
    >>> parse_one_d(float_scalar)
    array([1.])
    >>> np_float_scalar = 1.0
    >>> parse_one_d(np_float_scalar)
    array([1.])
    >>> example_list = [1.0, 1.0]
    >>> parse_one_d(example_list)
    array([1., 1.])
    >>> extra_dims = np.array([[1, 1]])
    >>> parse_one_d(extra_dims)
    array([1, 1])
    """
    if isinstance(maybe_vector, (int, float, np.integer, np.floating)):
        return np.array([maybe_vector])
    if isinstance(maybe_vector, np.ndarray):
        squeezed_vector = maybe_vector.squeeze()
        if squeezed_vector.ndim == 1:
            return squeezed_vector
        elif squeezed_vector.ndim == 0:
            # Squeezed to scalar so force vector
            return squeezed_vector[None]
        else:
            raise ValueError(
                "Vector can have at most one non-trivial dimension but "
                f"had shape {maybe_vector.shape}"
            )
    return np.array(maybe_vector)


@overload
def to_memory_order(
    array: np.ndarray, order: MemoryLayout, copy: bool = False
) -> np.ndarray:
    pass


@overload
def to_memory_order(
    array: sparse.coo_matrix, order: MemoryLayout, copy: bool = False
) -> sparse.coo_matrix:
    pass


def to_memory_order(
    array: np.ndarray | sparse.coo_matrix, order: MemoryLayout, copy: bool = False
) -> np.ndarray | sparse.coo_matrix:
    """Convert an array to the specified memory layout.

    Parameters
    ----------
    array: Data to ensure matches memory order.
    order: Desired memory order.
    copy: Whether to force a copy even if data already in supported memory order.

    Examples
    --------
    >>> c_order = np.arange(16).reshape((2, 2, 2, 2))
    >>> c_order.flags["C_CONTIGUOUS"]
    True
    >>> to_memory_order(c_order, "F").flags["F_CONTIGUOUS"]
    True
    """
    if copy:
        # This could be slightly optimized
        # in worst case two copies occur
        if isinstance(array, np.ndarray):
            array = array.copy("K")
        else:
            array = array.copy()
    if isinstance(array, sparse.coo_matrix):
        return array
    if order == "F":
        return np.asfortranarray(array)
    elif order == "C":
        return np.ascontiguousarray(array)
    raise ValueError(f"Unsupported order {order}")


if __name__ == "__main__":
    import doctest  # pragma: no cover

    doctest.testmod()  # pragma: no cover
