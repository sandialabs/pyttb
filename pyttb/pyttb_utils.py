"""PYTTB shared utilities across tensor types"""

# Copyright 2022 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.
from __future__ import annotations

from enum import Enum
from inspect import signature
from typing import (
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    get_args,
    overload,
)

import numpy as np

import pyttb as ttb


def tt_union_rows(MatrixA: np.ndarray, MatrixB: np.ndarray) -> np.ndarray:
    """
    Helper function to reproduce functionality of MATLABS intersect(a,b,'rows')

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
    >>> a = np.array([[1,2],[3,4]])
    >>> b = np.array([[0,0],[1,2],[3,4],[0,0]])
    >>> tt_union_rows(a,b)
    array([[0, 0],
           [1, 2],
           [3, 4]])
    """
    # TODO ismember and uniqe are very similar in function
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
    dims: Optional[np.ndarray] = None,
    exclude_dims: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, None]: ...  # pragma: no cover see coveragepy/issues/970


@overload
def tt_dimscheck(
    N: int,
    M: int,
    dims: Optional[np.ndarray] = None,
    exclude_dims: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]: ...  # pragma: no cover see coveragepy/issues/970


def tt_dimscheck(
    N: int,
    M: Optional[int] = None,
    dims: Optional[np.ndarray] = None,
    exclude_dims: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Used to preprocess dimensions for tensor dimensions

    Parameters
    ----------

    Returns
    -------

    """
    if dims is not None and exclude_dims is not None:
        raise ValueError("Either specify dims to include or exclude, but not both")

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
    if (dims is None or dims.size == 0) and exclude_dims is None:
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
            # therfore, index in order of sdims
            vidx = sidx
        else:
            # Case 2: Number of multiplicands is equal to the number of dimensions of
            # tensor; therefore, index multiplicands by dimensions in dims argument.
            vidx = sdims

    return sdims, vidx


def tt_tenfun(function_handle, *inputs):  # noqa: PLR0912
    """
    Apply a function to each element in a tensor

    Parameters
    ----------
    function_handle:
        callable
    inputs:
        tensor type, or np.array

    Returns
    -------
    :class:`pyttb.tensor`
    """
    # Allow inputs to be mutable in case of type conversion
    inputs = list(inputs)

    if len(inputs) == 0:
        assert False, "Must provide element(s) to perform operation on"

    assert callable(function_handle), "function_handle must be callable"

    # Convert inputs to tensors if they aren't already
    for i, an_input in enumerate(inputs):
        if isinstance(an_input, (ttb.tensor, float, int)):
            continue
        if isinstance(an_input, np.ndarray):
            inputs[i] = ttb.tensor(an_input)
        elif isinstance(
            an_input,
            (
                ttb.ktensor,
                ttb.ttensor,
                ttb.sptensor,
                ttb.sumtensor,
                ttb.symtensor,
                ttb.symktensor,
            ),
        ):
            inputs[i] = an_input.to_tensor()
        else:
            assert False, "Invalid input to ten fun"

    # It's ok if there are two input and one is a scalar; otherwise all inputs have to
    # be the same size
    if (
        (len(inputs) == 2)
        and isinstance(inputs[0], (float, int))
        and isinstance(inputs[1], ttb.tensor)
    ):
        sz = inputs[1].shape
    elif (
        (len(inputs) == 2)
        and isinstance(inputs[1], (float, int))
        and isinstance(inputs[0], ttb.tensor)
    ):
        sz = inputs[0].shape
    else:
        for i, an_input in enumerate(inputs):
            if isinstance(an_input, (float, int)):
                assert False, f"Argument {i} is a scalar but expected a tensor"
            elif i == 0:
                sz = an_input.shape
            elif sz != an_input.shape:
                assert (
                    False
                ), f"Tensor {i} is not the same size as the first tensor input"

    # Number of inputs for function handle
    nfunin = len(signature(function_handle).parameters)

    # Case I: Binary function
    if len(inputs) == 2 and nfunin == 2:
        X = inputs[0]
        Y = inputs[1]
        if not isinstance(X, (float, int)):
            X = X.data
        if not isinstance(Y, (float, int)):
            Y = Y.data

        data = function_handle(X, Y)
        Z = ttb.tensor(data, copy=False)
        return Z

    # Case II: Expects input to be matrix and applies operation on each columns
    if len(inputs) == 1:
        X = inputs[0].data
        X = np.reshape(X, (1, -1))
    else:
        X = np.zeros((len(inputs), np.prod(sz)))
        for i, an_input in enumerate(inputs):
            X[i, :] = np.reshape(an_input.data, (np.prod(sz)))
    data = function_handle(X)
    data = np.reshape(data, sz)
    Z = ttb.tensor(data, copy=False)
    return Z


def tt_setdiff_rows(MatrixA: np.ndarray, MatrixB: np.ndarray) -> np.ndarray:
    """
    Helper function to reproduce functionality of MATLABS setdiff(a,b,'rows')

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
    """
    Helper function to reproduce functionality of MATLABS intersect(a,b,'rows')

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
    >>> a = np.array([[1,2],[3,4]])
    >>> b = np.array([[0,0],[1,2],[3,4],[0,0]])
    >>> tt_intersect_rows(a,b)
    array([0, 1])
    >>> tt_intersect_rows(b,a)
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


def tt_irenumber(t: ttb.sptensor, shape: Tuple[int, ...], number_range) -> np.ndarray:
    """
    RENUMBER indices for sptensor subsasgn

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
            if isinstance(r, list):
                r = np.array(r)  # noqa: PLW2901
            newsubs[:, i] = r[newsubs[:, i]]
    return newsubs


def tt_renumber(
    subs: np.ndarray, shape: Tuple[int, ...], number_range
) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """
    RENUMBER indices for sptensor subsref

    [NEWSUBS,NEWSZ] = RENUMBER(SUBS,SZ,RANGE) takes a set of
    original subscripts SUBS with entries from a tensor of size
    SZ. All the entries in SUBS are assumed to be within the
    specified RANGE. These subscripts are then renumbered so that,
    in dimension i, the numbers range from 1:numel(RANGE(i)).

    Parameters
    ----------
    subs:
    shape:
        Shape of source tensor.
    range:

    Returns
    -------
    newsubs:
        Updated subscripts.
    newshape:
        Resulting shape.
    """
    newshape = np.array(shape)
    newsubs = subs
    for i in range(0, len(shape)):
        if not number_range[i] == slice(None, None, None):
            if subs.size == 0:
                if not isinstance(number_range[i], slice):
                    if isinstance(number_range[i], (int, float, np.integer)):
                        newshape[i] = number_range[i]
                    else:
                        newshape[i] = len(number_range[i])
                else:
                    # TODO get this length without generating the range
                    newshape[i] = len(range(0, shape[i])[number_range[i]])
            else:
                newsubs[:, i], newshape[i] = tt_renumberdim(
                    subs[:, i], shape[i], number_range[i]
                )

    return newsubs, tuple(newshape)


def tt_renumberdim(idx: np.ndarray, shape: int, number_range) -> Tuple[int, int]:
    """
    RENUMBERDIM helper function for RENUMBER

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
        newshape = 0
    elif isinstance(number_range, slice):
        number_range = range(0, shape)[number_range]
        newshape = len(number_range)
    else:
        newshape = len(number_range)

    # Create map from old range to the new range
    idx_map = np.zeros(shape=shape)
    for i in range(0, newshape):
        idx_map[number_range[i]] = int(i)

    # Do the mapping
    newidx = idx_map[idx]
    return newidx, newshape


# TODO make more efficient
# https://stackoverflow.com/questions/22699756/python-version-of-ismember-with-rows-and-index
# For thoughts on how to speed this up
def tt_ismember_rows(
    search: np.ndarray, source: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find location of search rows in source array

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
    >>> b = np.array([[2, 6],[2, 1],[2, 4],[4, 6],[4, 7],[5, 9],[5, 2],[5, 1]])
    >>> matched, results = tt_ismember_rows(a,b)
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


def tt_ind2sub(shape: Tuple[int, ...], idx: np.ndarray) -> np.ndarray:
    """
    Multiple subscripts from linear indices.

    Parameters
    ----------
    shape:
    idx:

    Returns
    -------
    :class:`numpy.ndarray`
    """
    if idx.size == 0:
        return np.empty(shape=(0, len(shape)), dtype=int)
    idx[idx < 0] += np.prod(shape)  # Handle negative indexing as simply as possible
    return np.array(np.unravel_index(idx, shape, order="F")).transpose()


def tt_subsubsref(obj, s):
    """
    Helper function for tensor toolbox subsref.

    Parameters
    ----------
    obj:
        Tensor Data Structure
    s:
        Reference into tensor

    Returns
    -------
    Still uncertain to this functionality
    """
    # TODO figure out when subsref yields key of length>1 for now ignore this logic and
    #  just return
    # if len(s) == 1:
    #    return obj
    # else:
    #   return obj[s[1:]]
    if isinstance(obj, np.ndarray) and obj.size == 1:
        return obj.item()
    return obj


def tt_intvec2str(v: np.ndarray) -> str:
    """
    Print integer vector to a string with brackets. Numpy should already handle this so
    it is a placeholder stub

    Parameters
    ----------
    v:
        Integer vector

    Returns
    -------
    Formatted string to print
    """
    return np.array2string(v)


def tt_sub2ind(shape: Tuple[int, ...], subs: np.ndarray) -> np.ndarray:
    """
    Converts multidimensional subscripts to linear indices.

    Parameters
    ----------
    shape:
        Shape of tensor
    subs:
        Subscripts for tensor

    Returns
    -------
    :class:`numpy.ndarray`

    See Also
    --------

    :func:`tt_ind2sub`:
    """
    if subs.size == 0:
        return np.array([])
    idx = np.ravel_multi_index(tuple(subs.transpose()), shape, order="F")
    return idx


def tt_sizecheck(shape: Tuple[int, ...], nargout: bool = True) -> bool:
    """
    TT_SIZECHECK Checks that the shape is valid.

    TT_SIZECHECK(S) throws an error if S is not a valid shape tuple,
    which means that it is a row vector with strictly postitive,
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
    """
    if vals.size == 0:
        ok = True
    elif len(vals.shape) == 2 and vals.shape[1] == 1:
        ok = True
    else:
        ok = False
    if not ok and not nargout:
        assert False, "Values must be in array"
    return ok


def isrow(v: np.ndarray) -> bool:
    """
    ISROW Checks if vector is a row vector.

    ISROW(V) returns True if V is a row vector; otherwise returns False.

    Parameters
    ----------
    v:
        Vector input

    Returns
    -------
    bool
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
    """Methods for indexing entries of tensors"""

    UNKNOWN = 0
    LINEAR = 1
    SUBTENSOR = 2
    SUBSCRIPTS = 3


# We probably want to create a specific file for utility types
LinearIndexType = Union[int, float, np.generic, slice]
IndexType = Union[LinearIndexType, list, np.ndarray]


def get_index_variant(indices: IndexType) -> IndexVariant:
    """Decide on intended indexing variant. No correctness checks."""
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
    elif isinstance(indices, list):
        # TODO this is slightly redundant/inefficient
        key = np.array(indices)
        if len(key.shape) == 1 or key.shape[1] == 1:
            variant = IndexVariant.LINEAR
    return variant


def get_mttkrp_factors(
    U: Union[ttb.ktensor, List[np.ndarray]], n: int, ndims: int
) -> List[np.ndarray]:
    """Apply standard checks and type conversions for mttkrp factors"""
    if isinstance(U, ttb.ktensor):
        U = U.copy()
        # Absorb lambda into one of the factors but not the one that is skipped
        if n == 0:
            U.redistribute(1)
        else:
            U.redistribute(0)

        # Extract the factor matrices
        U = U.factor_matrices

    assert isinstance(
        U, (list, np.ndarray)
    ), "Second argument must be list of numpy.ndarray's or a ktensor"

    assert len(U) == ndims, "List of factor matrices is the wrong length"

    return U


def gather_wrap_dims(
    ndims: int,
    rdims: Optional[np.ndarray] = None,
    cdims: Optional[np.ndarray] = None,
    cdims_cyclic: Optional[Union[Literal["fc"], Literal["bc"], Literal["t"]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
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
                    "Unrecognized value for cdims_cyclic pattern, "
                    'must be "fc" or "bc".'
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
