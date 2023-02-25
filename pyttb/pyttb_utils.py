# Copyright 2022 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.
"""PYTTB shared utilities across tensor types"""
from inspect import signature
from typing import Optional, Tuple, overload

import numpy as np

import pyttb as ttb


def tt_to_dense_matrix(tensorInstance, mode, transpose=False):
    """
    Helper function to unwrap tensor into dense matrix, should replace the core need
    for tenmat

    Parameters
    ----------
    tensorInstance: :class:`pyttb.tensor` or :class:`pyttb.tensor`
        Ktensor->matrix is supported but the inverse is not
    mode: int
        Mode around which to unwrap tensor
    transpose: bool
        Whether or not to tranpose unwrapped tensor

    Returns
    -------
    matrix: :class:`numpy.ndarray`
    """
    siz = np.array(tensorInstance.shape).astype(int)
    old = np.setdiff1d(np.arange(tensorInstance.ndims), mode).astype(int)
    permutation = np.concatenate(([mode], old))
    # This mimics how tenmat handles ktensors
    # TODO check if full can be done after permutation and reshape for efficiency
    if isinstance(tensorInstance, ttb.ktensor):
        tensorInstance = tensorInstance.full()
    tensorInstance = tensorInstance.permute(permutation).reshape(
        (siz[mode], np.prod(siz[old]))
    )
    matrix = tensorInstance.data
    if transpose:
        matrix = np.transpose(matrix)
    return matrix


def tt_from_dense_matrix(matrix, shape, mode, idx):
    """
    Helper function to wrap dense matrix into tensor.
    Inverse of :class:`pyttb.tt_to_dense_matrix`

    Parameters
    ----------
    matrix: :class:`numpy.ndarray`
    mode: int
        Mode around which tensor was unwrapped
    idx: int
        in {0,1}, idx of mode in matrix, s.b. 0 for tranpose=True

    Returns
    -------
    tensorInstance: :class:`pyttb.tensor`
    """
    tensorInstance = ttb.tensor.from_data(matrix)
    if idx == 0:
        tensorInstance = tensorInstance.permute(np.array([1, 0]))
    tensorInstance = tensorInstance.reshape(shape)
    tensorInstance = tensorInstance.permute(
        np.concatenate((np.arange(1, mode + 1), [0], np.arange(mode + 1, len(shape))))
    )
    return tensorInstance


def tt_union_rows(MatrixA, MatrixB):
    """
    Helper function to reproduce functionality of MATLABS intersect(a,b,'rows')

    Parameters
    ----------
    MatrixA: :class:`numpy.ndarray`
    MatrixB: :class:`numpy.ndarray`

    Returns
    -------
    location: :class:`numpy.ndarray` list of intersection indices

    Examples
    --------
    >>> a = np.array([[1,2],[3,4]])
    >>> b = np.array([[0,0],[1,2],[3,4],[0,0]])
    >>> ttb.tt_union_rows(a,b)
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
    location = tt_ismember_rows(
        MatrixBUnique[np.argsort(idxB)], MatrixAUnique[np.argsort(idxA)]
    )
    union = np.vstack(
        (MatrixB[np.sort(idxB[np.where(location < 0)])], MatrixA[np.sort(idxA)])
    )
    return union


@overload
def tt_dimscheck(dims: np.ndarray, N: int, M: None = None) -> Tuple[np.ndarray, None]:
    ...  # pragma: no cover see coveragepy/issues/970


@overload
def tt_dimscheck(dims: np.ndarray, N: int, M: int) -> Tuple[np.ndarray, np.ndarray]:
    ...  # pragma: no cover see coveragepy/issues/970


def tt_dimscheck(
    dims: np.ndarray, N: int, M: Optional[int] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Used to preprocess dimensions for tensor dimensions

    Parameters
    ----------

    Returns
    -------

    """
    # Fix empty case
    if dims.size == 0:
        dims = np.arange(0, N)

    # Fix "minus" case
    if np.max(dims) < 0:
        # Check that all members in range
        if not np.all(np.isin(-dims, np.arange(0, N + 1))):
            assert False, "Invalid magnitude for negative dims selection"
        dims = np.setdiff1d(np.arange(1, N + 1), -dims) - 1

    # Save dimensions of dims
    P = len(dims)

    # Reorder dims from smallest to largest (this matters in particular for the vector
    # multiplicand case, where the order affects the result)
    sidx = np.argsort(dims)
    sdims = dims[sidx]
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


def tt_tenfun(function_handle, *inputs):  # pylint:disable=too-many-branches
    """
    Apply a function to each element in a tensor

    Parameters
    ----------
    function_handle: callable
    inputs: tensor type, or np.array

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
            inputs[i] = ttb.tensor.from_data(an_input)
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
            inputs[i] = ttb.tensor.from_tensor_type(an_input)
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
        Z = ttb.tensor.from_data(data)
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
    Z = ttb.tensor.from_data(data)
    return Z


def tt_setdiff_rows(MatrixA, MatrixB):
    """
    Helper function to reproduce functionality of MATLABS setdiff(a,b,'rows')

    Parameters
    ----------
    MatrixA: :class:`numpy.ndarray`
    MatrixB: :class:`numpy.ndarray`

    Returns
    -------
    location: :class:`numpy.ndarray` list of set difference indices
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
    location = tt_ismember_rows(
        MatrixBUnique[np.argsort(idxB)], MatrixAUnique[np.argsort(idxA)]
    )
    return np.setdiff1d(idxA, location[np.where(location >= 0)])


def tt_intersect_rows(MatrixA, MatrixB):
    """
    Helper function to reproduce functionality of MATLABS intersect(a,b,'rows')

    Parameters
    ----------
    MatrixA: :class:`numpy.ndarray`
    MatrixB: :class:`numpy.ndarray`

    Returns
    -------
    location: :class:`numpy.ndarray` list of intersection indices

    Examples
    --------
    >>> a = np.array([[1,2],[3,4]])
    >>> b = np.array([[0,0],[1,2],[3,4],[0,0]])
    >>> ttb.tt_intersect_rows(a,b)
    array([0, 1])
    >>> ttb.tt_intersect_rows(b,a)
    array([1, 2])
    """
    # TODO ismember and uniqe are very similar in function
    if MatrixA.size > 0:
        MatrixAUnique, idxA = np.unique(MatrixA, axis=0, return_index=True)
    else:
        MatrixAUnique = idxA = np.array([], dtype=int)
    if MatrixB.size > 0:
        MatrixBUnique, idxB = np.unique(MatrixB, axis=0, return_index=True)
    else:
        MatrixBUnique = idxB = np.array([], dtype=int)
    location = tt_ismember_rows(
        MatrixBUnique[np.argsort(idxB)], MatrixAUnique[np.argsort(idxA)]
    )
    return location[np.where(location >= 0)]


def tt_irenumber(t, shape, number_range):  # pylint: disable=unused-argument
    """
    RENUMBER indices for sptensor subsasgn

    Parameters
    ----------
    t: :class:`pyttb.sptensor`
    shape: tuple(int)
    range: tuple, len(range) = modes in tensor. Is key from __setitem__

    Returns
    -------
    newsubs: :class:`numpy.ndarray`
    """
    # TODO shape is unused. Should it be used? I don't particularly understand what
    #  this is meant to be doing
    nz = t.nnz
    if nz == 0:
        newsubs = np.array([])
        return newsubs

    newsubs = t.subs.astype(int)
    for i, r in enumerate(number_range):
        if isinstance(r, slice):
            newsubs[:, i] = (newsubs[:, i])[r]
        elif isinstance(r, int):
            # This appears to be inserting new keys as rows to our subs here
            newsubs = np.insert(newsubs, obj=i, values=r, axis=1)
        else:
            if isinstance(r, list):
                r = np.array(r)
            newsubs[:, i] = r[newsubs[:, i]]
    return newsubs


def tt_assignment_type(x, subs, rhs):
    """
    TT_ASSIGNMENT_TYPE What type of subsagn is this?

    Parameters
    ----------
    x:
    subs:
    rhs:

    Returns
    -------
    objectType
    """
    if type(x) is type(rhs):
        return "subtensor"
    # If subscripts is a tuple that contains an nparray
    if isinstance(subs, tuple) and len(subs) >= 2:
        return "subtensor"
    return "subscripts"


def tt_renumber(subs, shape, number_range):
    """
    RENUMBER indices for sptensor subsref

    [NEWSUBS,NEWSZ] = RENUMBER(SUBS,SZ,RANGE) takes a set of
    original subscripts SUBS with entries from a tensor of size
    SZ. All the entries in SUBS are assumed to be within the
    specified RANGE. These subscripts are then renumbered so that,
    in dimension i, the numbers range from 1:numel(RANGE(i)).

    Parameters
    ----------
    subs: :class:`numpy.ndarray`
    shape: tuple
    range:

    Returns
    -------
    newsubs: :class:`numpy.ndarray`
    newshape: tuple
    """
    newshape = np.array(shape)
    newsubs = subs
    for i in range(0, len(shape)):  # pylint: disable=consider-using-enumerate
        if not number_range[i] == slice(None, None, None):
            if subs.size == 0:
                if not isinstance(number_range[i], slice):
                    if isinstance(number_range[i], (int, float)):
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


def tt_renumberdim(idx, shape, number_range):
    """
    RENUMBERDIM helper function for RENUMBER

    Parameters
    ----------
    idx: :class:`numpy.ndarray`
    shape: int
    number_range: :class:`numpy.ndarray`

    Returns
    -------
    newidx:
    newshape:
    """
    # Determine the size of the new range
    if isinstance(number_range, int):
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


# TODO make more efficient, decide if we want to support the multiple response
#  matlab does
# pylint: disable=line-too-long
# https://stackoverflow.com/questions/22699756/python-version-of-ismember-with-rows-and-index
# For thoughts on how to speed this up
def tt_ismember_rows(search, source):
    """
    Find location of search rows in source array

    Parameters
    ----------
    search: :class:`numpy.ndarray`
    source: :class:`numpy.ndarray`

    Returns
    -------
    results: :class:`numpy.ndarray`
    search.size==results.size,
    if search[0,:] == source[3,:], then results[0] = 3
    if exists i such that search[i,:] != source[j,:] for any j, then results[i] = -1

    Examples
    --------
    >>> a = np.array([[4, 6], [1, 9], [2, 6]])
    >>> b = np.array([[2, 6],[2, 1],[2, 4],[4, 6],[4, 7],[5, 9],[5, 2],[5, 1]])
    >>> results = tt_ismember_rows(a,b)
    >>> print(results)
    [ 3 -1  0]

    """
    results = np.ones(shape=search.shape[0]) * -1
    if search.size == 0:
        return results.astype(int)
    if source.size == 0:
        return results.astype(int)
    (row_idx, col_idx) = np.nonzero(np.all(source == search[:, np.newaxis], axis=2))
    results[row_idx] = col_idx
    return results.astype(int)


def tt_ind2sub(shape: Tuple[int, ...], idx: np.ndarray) -> np.ndarray:
    """
    Multiple subscripts from linear indices.

    Parameters
    ----------
    shape: tuple
    idx: :class:`numpy.ndarray`
    Returns
    -------
    :class:`numpy.ndarray`
    """
    if idx.size == 0:
        return np.empty(shape=(0, len(shape)), dtype=int)

    return np.array(np.unravel_index(idx, shape, order="F")).transpose()


def tt_subsubsref(obj, s):  # pylint: disable=unused-argument
    """
    Helper function for tensor toolbox subsref.

    Parameters
    ----------
    obj: Tensor Data Structure
    s: Reference into tensor

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
    return obj


def tt_intvec2str(v):
    """
    Print integer vector to a string with brackets. Numpy should already handle this so
    it is a placeholder stub

    Parameters
    ----------
    v: :class:`numpy.ndarray` integer vector
    Returns
    -------
    str: formatted string to print
    """
    return np.array2string(v)


def tt_sub2ind(shape, subs):
    """
    Converts multidimensional subscripts to linear indices.

    Parameters
    ----------
    shape: tuple
    Shape of tensor
    subs: :class:`numpy.ndarray`
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


def tt_sizecheck(shape, nargout=True):
    """
    TT_SIZECHECK Checks that the shape is valid.

    TT_SIZECHECK(S) throws an error if S is not a valid shape tuple,
    which means that it is a row vector with strictly postitive,
    real-valued, finite integer values.

    Parameters
    ----------
    shape: tuple
    Shape of tensor
    nargout: bool
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


def tt_subscheck(subs, nargout=True):
    """
    TT_SUBSCHECK Checks for valid subscripts.

    TT_SUBSCHECK(S) throws an error if S is not a valid subscript
    array, which means that S is a matrix of real-valued, finite,
    positive, integer subscripts.

    Parameters
    ----------
    subs: :class:`numpy.ndarray`
    Subs of tensor
    nargout: bool
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
        and (subs > 0).all()
    ):
        ok = True
    else:
        ok = False

    if not ok and not nargout:
        assert False, "Subscripts must be a matrix of real positive integers"
    return ok


def tt_valscheck(vals, nargout=True):
    """
    TT_VALSCHECK Checks for valid values.

    TT_VALSCHECK(S) throws an error if S is not a valid values
    array, which means that S is a column array.

    Parameters
    ----------
    vals: :class:`numpy.ndarray`
    Values of tensor
    nargout: bool
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


def isrow(v):
    """
    ISROW Checks if vector is a row vector.

    ISROW(V) returns True if V is a row vector; otherwise returns False.

    Parameters
    ----------
    v: :class:`numpy.ndarray`
    vector input

    Returns
    -------
    bool
    """
    return v.ndim == 2 and v.shape[0] == 1 and v.shape[1] >= 1


def isvector(a):
    """
    ISVECTOR Checks if vector is a row vector.

    ISVECTOR(A) returns True if A is a vector; otherwise returns False.

    Parameters
    ----------
    a: :class:`numpy.ndarray`

    Returns
    -------
    bool
    """
    return a.ndim == 1 or (a.ndim == 2 and (a.shape[0] == 1 or a.shape[1] == 1))


# TODO: this is a challenge, since it may need to apply to either Python built in types
#  or numpy types
def islogical(a):
    """
    ISLOGICAL Checks if vector is a logical vector.

    ISLOGICAL(A) returns True if A is a logical array; otherwise returns False.

    Parameters
    ----------
    a: :class:`numpy.ndarray`

    Returns
    -------
    bool
    """
    return isinstance(a, bool)
