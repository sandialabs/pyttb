"""Classes and functions for working with dense tensors."""

# Copyright 2022 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from __future__ import annotations

import logging
from collections.abc import Iterable
from itertools import combinations_with_replacement, permutations
from math import factorial
from typing import Any, Callable, List, Literal, Optional, Tuple, Union, overload

import numpy as np
import scipy.sparse.linalg
from numpy_groupies import aggregate as accumarray
from scipy import sparse

import pyttb as ttb
from pyttb.pyttb_utils import (
    IndexVariant,
    get_index_variant,
    get_mttkrp_factors,
    tt_dimscheck,
    tt_ind2sub,
    tt_sub2ind,
    tt_subsubsref,
    tt_tenfun,
)


class tensor:
    """
    TENSOR Class for dense tensors.

    Contains the following data members:

    ``data``: :class:`numpy.ndarray` dense array containing the data elements
    of the tensor.

    Instances of :class:`pyttb.tensor` can be created using `__init__()` or
    the following method:

      * :meth:`from_function`

    Examples
    --------
    For all examples listed below, the following module imports are assumed:

    >>> import pyttb as ttb
    >>> import numpy as np
    """

    __slots__ = ("data", "shape")

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        shape: Optional[Tuple[int, ...]] = None,
        copy: bool = True,
    ):
        """
        Creates a :class:`pyttb.tensor` from a :class:`numpy.ndarray`

        Note that 1D tensors (i.e., when len(shape)==1) contains a data
        array that follow the Numpy convention of being a row vector.

        Parameters
        ----------
        data:
            Tensor source data.
        shape:
            Shape of resulting tensor if not the same as data shape.
        copy:
            Whether to make a copy of provided data or just reference it.

        Examples
        -------
        Create an empty :class:`pyttb.tensor`:

        >>> T = ttb.tensor()
        >>> print(T)
        empty tensor of shape ()
        data = []

        Create a :class:`pyttb.tensor` from a :class:`numpy.ndarray`:

        >>> T = ttb.tensor(np.array([[1,2],[3,4]]))
        >>> print(T)
        tensor of shape (2, 2)
        data[:, :] =
        [[1 2]
         [3 4]]
        """
        if data is None:
            # EMPTY / DEFAULT CONSTRUCTOR
            self.data: np.ndarray = np.array([])
            self.shape: Tuple = ()
            return

        # CONVERT A MULTIDIMENSIONAL ARRAY
        if not issubclass(data.dtype.type, np.number) and not issubclass(
            data.dtype.type, np.bool_
        ):
            assert False, "First argument must be a multidimensional array."

        # Create or check second argument
        if shape is None:
            shape = data.shape
        elif not isinstance(shape, tuple):
            assert False, "Second argument must be a tuple."

        # Make sure the number of elements matches what's been specified
        if len(shape) == 0:
            if data.size > 0:
                assert False, "Empty tensor cannot contain any elements"

        elif np.prod(shape) != data.size:
            assert (
                False
            ), "TTB:WrongSize, Size of data does not match specified size of tensor"

        # Make sure the data is indeed the right shape
        if data.size > 0 and len(shape) > 0:
            # reshaping using Fortran ordering to match Matlab conventions
            data = np.reshape(data, np.array(shape), order="F")

        # Create the tensor
        if copy:
            self.data = data.copy()
        else:
            self.data = data
        self.shape = shape
        return

    @classmethod
    def from_function(
        cls,
        function_handle: Callable[[Tuple[int, ...]], np.ndarray],
        shape: Tuple[int, ...],
    ) -> tensor:
        """
        Construct a :class:`pyttb.tensor` whose data entries are set using
        a function.

        Parameters
        ----------
        function_handle:
            A function that can accept a shape (i.e., :class:`tuple` of
            dimension sizes) and return a :class:`numpy.ndarray` of that shape.
            `numpy.zeros`, `numpy.ones`.
        shape:
            Shape of the resulting tensor.

        Returns
        -------
        Constructed tensor.

        Examples
        --------
        Create a :class:`pyttb.tensor` with entries equal to 1:

        >>> T = ttb.tensor.from_function(np.ones, (2, 3, 4))
        >>> print(T)
        tensor of shape (2, 3, 4)
        data[0, :, :] =
        [[1. 1. 1. 1.]
         [1. 1. 1. 1.]
         [1. 1. 1. 1.]]
        data[1, :, :] =
        [[1. 1. 1. 1.]
         [1. 1. 1. 1.]
         [1. 1. 1. 1.]]
        """
        # Check size
        if not isinstance(shape, tuple):
            assert False, "TTB:BadInput, Shape must be a tuple"

        # Generate data
        data = function_handle(shape)

        # Create the tensor
        return cls(data, shape, copy=False)

    def copy(self) -> tensor:
        """Make a deep copy of a :class:`pyttb.tensor`.

        Returns
        -------
        Copy of original tensor.

        Examples
        --------
        >>> T1 = ttb.tensor(np.ones((3,2)))
        >>> T2 = T1
        >>> T3 = T2.copy()
        >>> T1[0,0] = 3
        >>> T1[0,0] == T2[0,0]
        True
        >>> T1[0,0] == T3[0,0]
        False
        """
        return ttb.tensor(self.data, self.shape, copy=True)

    def __deepcopy__(self, memo):
        return self.copy()

    def collapse(
        self,
        dims: Optional[np.ndarray] = None,
        fun: Callable[[np.ndarray], Union[float, np.ndarray]] = np.sum,
    ) -> Union[float, np.ndarray, tensor]:
        """
        Collapse tensor along specified dimensions.

        Parameters
        ----------
        dims:
            Dimensions to collapse.
        fun:
            Method used to collapse dimensions.

        Returns
        -------
        Collapsed value.

        Examples
        --------
        >>> T = ttb.tensor(np.ones((2,2)))
        >>> T.collapse()
        4.0
        >>> T.collapse(np.array([0]))
        tensor of shape (2,)
        data[:] =
        [2. 2.]
        >>> T.collapse(np.arange(T.ndims), sum)
        4.0
        >>> T.collapse(np.arange(T.ndims), np.prod)
        1.0
        """
        if self.data.size == 0:
            return np.array([])

        if dims is None:
            dims = np.arange(0, self.ndims)

        if dims.size == 0:
            return self.copy()

        dims, _ = tt_dimscheck(self.ndims, dims=dims)
        remdims = np.setdiff1d(np.arange(0, self.ndims), dims)

        # Check for the case where we accumulate over *all* dimensions
        if remdims.size == 0:
            return fun(self.data.flatten("F"))

        ## Calculate the shape of the result
        newshape = tuple(np.array(self.shape)[remdims])

        ## Convert to a matrix where each row is going to be collapsed
        A = ttb.tenmat.from_data(self.data, remdims, dims).double()

        ## Apply the collapse function
        B = np.zeros((A.shape[0], 1))
        for i in range(0, A.shape[0]):
            B[i] = fun(A[i, :])

        ## Form and return the final result
        return ttb.tensor(B, newshape, copy=False)

    def contract(self, i1: int, i2: int) -> Union[np.ndarray, tensor]:
        """
        Contract tensor along two dimensions (array trace).

        Parameters
        ----------
        i1:
            First dimension
        i2:
            Second dimension

        Returns
        -------
        Contracted tensor.

        Examples
        --------
        >>> T = ttb.tensor(np.ones((2,2)))
        >>> T.contract(0, 1)
        2.0
        >>> T = ttb.tensor(np.array([[[1,2],[3,4]],[[5,6],[7,8]]]))
        >>> print(T)
        tensor of shape (2, 2, 2)
        data[0, :, :] =
        [[1 2]
         [3 4]]
        data[1, :, :] =
        [[5 6]
         [7 8]]
        >>> T.contract(0,1)
        tensor of shape (2,)
        data[:] =
        [ 8. 10.]
        >>> T.contract(0,2)
        tensor of shape (2,)
        data[:] =
        [ 7. 11.]
        >>> T.contract(1,2)
        tensor of shape (2,)
        data[:] =
        [ 5. 13.]
        """
        if self.shape[i1] != self.shape[i2]:
            assert False, "Must contract along equally sized dimensions"

        if i1 == i2:
            assert False, "Must contract along two different dimensions"

        # Easy case - returns a scalar
        if self.ndims == 2:
            return np.trace(self.data)

        # Remaining dimensions after trace
        remdims = np.setdiff1d(np.arange(0, self.ndims), np.array([i1, i2])).astype(int)

        # Size for return
        newsize = tuple(np.array(self.shape)[remdims])

        # Total size of remainder
        m = np.prod(newsize)

        # Number of items to add for trace
        n = self.shape[i1]

        # Permute trace dimensions to the end
        x = self.permute(np.concatenate((remdims, np.array([i1, i2]))))

        # Reshape data to be 3D
        data = np.reshape(x.data, (m, n, n), order="F")

        # Add diagonal entries for each slice
        newdata = np.zeros((m, 1))
        for idx in range(0, n):
            newdata += data[:, idx, idx][:, None]

        # Reshape result
        if np.prod(newsize) > 1:
            newdata = np.reshape(newdata, newsize, order="F")

        return ttb.tensor(newdata, newsize, copy=False)

    def double(self) -> np.ndarray:
        """
        Convert `:class:pyttb.tensor` to an `:class:numpy.ndarray` of doubles.

        Returns
        -------
        Copy of tensor data.

        Examples
        --------
        >>> T = ttb.tensor(np.ones((2,2)))
        >>> T.double()
        array([[1., 1.],
               [1., 1.]])
        """
        return self.data.astype(np.float64).copy()

    def exp(self) -> tensor:
        """
        Exponential of the elements of tensor.

        Returns
        -------
        Copy of tensor data with the exponential function applied to data\
            element-wise.

        Examples
        --------
        >>> T = ttb.tensor(np.array([[1, 2], [3, 4]]))
        >>> T.exp().data  # doctest: +ELLIPSIS
        array([[ 2.7182...,  7.3890... ],
               [20.0855..., 54.5981...]])
        """
        return ttb.tensor(np.exp(self.data), copy=False)

    def find(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find subscripts of nonzero elements in a tensor.

        Returns
        -------
        Array of subscripts of the nonzero values in the tensor and a column\
            vector of the corresponding values.

        Examples
        --------
        >>> T = ttb.tensor(np.array([[1,2],[3,4]]))
        >>> print(T)
        tensor of shape (2, 2)
        data[:, :] =
        [[1 2]
         [3 4]]
        >>> T_threshold = T > 2
        >>> subs, vals = T_threshold.find()
        >>> subs.astype(int)
        array([[1, 0],
               [1, 1]])
        >>> vals
        array([[ True],
               [ True]])
        """
        idx = np.nonzero(np.ravel(self.data, order="F"))[0]
        subs = tt_ind2sub(self.shape, idx)
        vals = self.data[tuple(subs.T)][:, None]
        return subs, vals

    def to_sptensor(self) -> ttb.sptensor:
        """
        Contruct a :class:`pyttb.sptensor` from `:class:pyttb.tensor`

        Returns
        -------
        Generated Sparse Tensor

        Examples
        --------
        >>> T = ttb.tensor(np.array([[0,2],[3,0]]))
        >>> print(T)
        tensor of shape (2, 2)
        data[:, :] =
        [[0 2]
         [3 0]]
        >>> S = T.to_sptensor()
        >>> print(S)
        sparse tensor of shape (2, 2) with 2 nonzeros
        [1, 0] = 3
        [0, 1] = 2
        """
        subs, vals = self.find()
        return ttb.sptensor(subs, vals, self.shape, copy=False)

    # TODO: do we need this, now that we have copy() and __deepcopy__()?
    def full(self) -> tensor:
        """
        Convert dense tensor to dense tensor.

        Returns
        -------
        Deep copy
        """
        return ttb.tensor(self.data)

    def innerprod(
        self, other: Union[tensor, ttb.sptensor, ttb.ktensor, ttb.ttensor]
    ) -> float:
        """
        Efficient inner product between a tensor and other `pyttb` tensors
        (`tensor`, `sptensor`, `ktensor`, or `ttensor`).

        Parameters
        ----------
        other:
            Tensor to take an innerproduct with.

        Examples
        --------
        >>> T = ttb.tensor(np.array([[1, 0], [0, 4]]))
        >>> T.innerprod(T)
        17
        >>> S = T.to_sptensor()
        >>> T.innerprod(S)
        17
        """
        if isinstance(other, ttb.tensor):
            if self.shape != other.shape:
                assert False, "Inner product must be between tensors of the same size"
            x = np.reshape(self.data, (self.data.size,), order="F")
            y = np.reshape(other.data, (other.data.size,), order="F")
            return x.dot(y)
        if isinstance(other, (ttb.ktensor, ttb.sptensor, ttb.ttensor)):
            # Reverse arguments and call specializer code
            return other.innerprod(self)
        assert False, "Inner product between tensor and that class is not supported"

    def isequal(self, other: Union[tensor, ttb.sptensor]) -> bool:
        """
        Exact equality for tensors.

        Parameters
        ----------
        other:
            Tensor to compare against.

        Examples
        --------
        >>> T1 = ttb.tensor(2 * np.ones((2,2)))
        >>> T2 = 2 * ttb.tensor(np.ones((2,2)))
        >>> T1.isequal(T2)
        True
        >>> T2[0,0] = 1
        >>> T1.isequal(T2)
        False
        """
        if isinstance(other, ttb.tensor):
            return bool(np.all(self.data == other.data))
        if isinstance(other, ttb.sptensor):
            return bool(np.all(self.data == other.full().data))
        return False

    @overload
    def issymmetric(
        self,
        grps: Optional[np.ndarray],
        version: Optional[Any],
        return_details: Literal[False],
    ) -> bool: ...  # pragma: no cover see coveragepy/issues/970

    @overload
    def issymmetric(
        self,
        grps: Optional[np.ndarray],
        version: Optional[Any],
        return_details: Literal[True],
    ) -> Tuple[
        bool, np.ndarray, np.ndarray
    ]: ...  # pragma: no cover see coveragepy/issues/970

    # TODO: We should probably always return details and let caller drop them
    def issymmetric(  # noqa: PLR0912
        self,
        grps: Optional[np.ndarray] = None,
        version: Optional[Any] = None,
        return_details: bool = False,
    ) -> Union[bool, Tuple[bool, np.ndarray, np.ndarray]]:
        """
        Determine if a dense tensor is symmetric in specified modes.

        Parameters
        ----------
        grps:
            Modes to check for symmetry
        version:
            Any non-None value will call the non-default old version
        return_details:
            Flag to return symmetry details in addition to bool

        Returns
        -------
        If symmetric in modes, optionally all differences and permutations

        Examples
        --------
        >>> T = ttb.tensor(np.ones((2,2)))
        >>> T.issymmetric()
        True
        >>> T.issymmetric(grps=np.arange(T.ndims))
        True
        >>> is_sym, diffs, perms = \
            T.issymmetric(grps=np.arange(T.ndims), version=1, return_details=True)
        >>> print(f"Tensor is symmetric: {is_sym}")
        Tensor is symmetric: True
        >>> print(f"Differences in modes: {diffs}")
        Differences in modes: [[0.]
         [0.]]
        >>> print(f"Permutations: {perms}")
        Permutations: [[0. 1.]
         [1. 0.]]
        """
        n = self.ndims
        sz = np.array(self.shape)

        if grps is None:
            grps = np.arange(0, n)

        if len(grps.shape) == 1:
            grps = np.array([grps])

        # Substantially different routines are called depending on whether the user
        # requests the permutation information. If permutation is required
        # (or requested) the algorithm is much slower
        if version is None and not return_details:
            # Use new algorithm
            for thisgrp in grps:
                # Check tensor dimensions first
                if not np.all(sz[thisgrp[0]] == sz[thisgrp]):
                    return False

                # Construct matrix ind where each row is the multi-index for one
                # element of X
                idx = tt_ind2sub(self.shape, np.arange(0, self.data.size))

                # Find reference index for every element in the tensor - this
                # is to its index in the symmetrized tensor. This puts every
                # element into a 'class' of entries that will be the same under
                # symmetry.
                classidx = idx
                classidx[:, thisgrp] = np.sort(idx[:, thisgrp], axis=1)

                # Compare each element to its class exemplar
                if np.any(self.data.ravel() != self.data[tuple(classidx.transpose())]):
                    return False

            # We survived all the tests!
            return True

        # Use the older algorithm
        else:
            # Check tensor dimensions for compatibility with symmetrization
            for dims in grps:
                for j in dims[1:]:
                    if sz[j] != sz[dims[0]]:
                        return False

            # Check actual symmetry
            cnt = sum(factorial(len(x)) for x in grps)
            all_diffs = np.zeros((cnt, 1))
            all_perms = np.zeros((cnt, n))
            for a_group in grps:
                # Compute the permutations for this group of symmetries
                for p_idx, perm in enumerate(permutations(a_group)):
                    all_perms[p_idx, :] = perm

                    # Do the permutation and record the difference.
                    Y = self.permute(np.array(perm))
                    if np.array_equal(self.data, Y.data):
                        all_diffs[p_idx] = 0
                    else:
                        all_diffs[p_idx] = np.max(
                            np.abs(self.data.ravel() - Y.data.ravel())
                        )

            if return_details is False:
                return bool((all_diffs == 0).all())
            return bool((all_diffs == 0).all()), all_diffs, all_perms

    def logical_and(self, other: Union[float, tensor]) -> tensor:
        """
        Logical and for tensors.

        Parameters
        ----------
        other:
            Value to perform and against.

        Examples
        --------
        >>> T = ttb.tenones((2,2))
        >>> T.logical_and(T).collapse()  # All true
        4.0
        """

        def logical_and(x, y):
            return np.logical_and(x, y).astype(dtype=x.dtype)

        return tt_tenfun(logical_and, self, other)

    def logical_not(self) -> tensor:
        """
        Logical not for tensors.

        Examples
        --------
        >>> T = ttb.tenones((2,2))
        >>> T.logical_not().collapse()  # All false
        0.0
        """
        # Np logical not dtype argument seems to not work here
        return ttb.tensor(np.logical_not(self.data).astype(self.data.dtype), copy=False)

    def logical_or(self, other: Union[float, tensor]) -> tensor:
        """
        Logical or for tensors.

        Parameters
        ----------
        other:
            Value to perform or against.

        Examples
        --------
        >>> T = ttb.tenones((2,2))
        >>> T.logical_or(T.logical_not()).collapse()  # All true
        4.0
        """

        def tensor_or(x, y):
            return np.logical_or(x, y).astype(x.dtype)

        return tt_tenfun(tensor_or, self, other)

    def logical_xor(self, other: Union[float, tensor]) -> tensor:
        """
        Logical xor for tensors.

        Parameters
        ----------
        other:
            Value to perform xor against.

        Examples
        --------
        >>> T = ttb.tenones((2,2))
        >>> T.logical_xor(T.logical_not()).collapse()  # All true
        4.0
        """

        def tensor_xor(x, y):
            return np.logical_xor(x, y).astype(dtype=x.dtype)

        return tt_tenfun(tensor_xor, self, other)

    def mask(self, W: tensor) -> np.ndarray:
        """
        Extract non-zero values at locations specified by mask tensor `W`.

        Parameters
        ----------
        W:
            Mask tensor.

        Returns
        -------
        Array of extracted values.

        Examples
        --------
        >>> T = ttb.tensor(np.array([[1, 2], [3, 4]]))
        >>> W = ttb.tenones((2,2))
        >>> T.mask(W)
        array([1, 3, 2, 4])
        """
        # Error checking
        if np.any(np.array(W.shape) > np.array(self.shape)):
            assert False, "Mask cannot be bigger than the data tensor"

        # Extract locations of nonzeros in W
        wsubs, _ = W.find()

        # Extract those non-zero values
        return self.data[tuple(wsubs.transpose())]

    def mttkrp(self, U: Union[ttb.ktensor, List[np.ndarray]], n: int) -> np.ndarray:
        """
        Matricized tensor times Khatri-Rao product. The matrices used in the
        Khatri-Rao product are passed as a :class:`pyttb.ktensor` (where the
        factor matrices are used) or as a list of :class:`numpy.ndarray` objects.

        Parameters
        ----------
        U:
            Matrices to create the Khatri-Rao product.
        n:
            Mode used to matricize tensor.

        Returns
        -------
        Array containing matrix product.

        Examples
        --------
        >>> T = ttb.tenones((2,2,2))
        >>> U = [np.ones((2,2))] * 3
        >>> T.mttkrp(U, 2)
        array([[4., 4.],
               [4., 4.]])
        """

        # check that we have a tensor that can perform mttkrp
        if self.ndims < 2:
            assert False, "MTTKRP is invalid for tensors with fewer than 2 dimensions"

        U = get_mttkrp_factors(U, n, self.ndims)

        if n == 0:
            R = U[1].shape[1]
        else:
            R = U[0].shape[1]

        # check that the dimensions match
        for i in range(self.ndims):
            if i == n:
                continue
            if U[i].shape[0] != self.shape[i]:
                assert False, f"Entry {i} of list of arrays is wrong size"

        szl = int(np.prod(self.shape[0:n]))
        szr = int(np.prod(self.shape[n + 1 :]))
        szn = self.shape[n]

        if n == 0:
            Ur = ttb.khatrirao(*U[1 : self.ndims], reverse=True)
            Y = np.reshape(self.data, (szn, szr), order="F")
            return Y @ Ur
        if n == self.ndims - 1:
            Ul = ttb.khatrirao(*U[0 : self.ndims - 1], reverse=True)
            Y = np.reshape(self.data, (szl, szn), order="F")
            return Y.T @ Ul
        else:
            Ul = ttb.khatrirao(*U[n + 1 :], reverse=True)
            Ur = np.reshape(
                ttb.khatrirao(*U[0:n], reverse=True), (szl, 1, R), order="F"
            )
            Y = np.reshape(self.data, (-1, szr), order="F")
            Y = Y @ Ul
            Y = np.reshape(Y, (szl, szn, R), order="F")
            V = np.zeros((szn, R))
            for r in range(R):
                V[:, [r]] = Y[:, :, r].T @ Ur[:, :, r]
            return V

    def mttkrps(self, U: Union[ttb.ktensor, List[np.ndarray]]) -> List[np.ndarray]:
        """
        Sequence of MTTKRP calculations for a tensor.

        Result is equivalent to [T.mttkrp(U, k) for k in range(T.ndims)].

        Parameters
        ----------
        U:
            Matrices to create the Khatri-Rao product.

        Returns
        -------
        Array containing matrix product.

        Examples
        --------
        >>> T = ttb.tenones((2,2,2))
        >>> U = [np.ones((2,2))] * 3
        >>> T.mttkrps(U)
        [array([[4., 4.],
               [4., 4.]]), array([[4., 4.],
               [4., 4.]]), array([[4., 4.],
               [4., 4.]])]
        """
        if isinstance(U, ttb.ktensor):
            U = U.factor_matrices
        split_idx = min_split(self.shape)
        V = [np.empty_like(self.data, shape=())] * self.ndims
        K = ttb.khatrirao(*U[split_idx + 1 :], reverse=True)
        W = np.reshape(self.data, (-1, K.shape[0]), order="F").dot(K)
        for k in range(split_idx):
            # Loop entry invariant: W has modes (mk x ... x ms, C)
            V[k] = mttv_mid(W, U[k + 1 : split_idx + 1])
            W = mttv_left(W, U[k])
        V[split_idx] = W
        K = ttb.khatrirao(*U[0 : split_idx + 1], reverse=True)
        W = np.reshape(self.data, (K.shape[0], -1), order="F").transpose().dot(K)
        for k in range(split_idx + 1, self.ndims - 1):
            # Loop invariant: W has modes (mk x .. x md, C)
            V[k] = mttv_mid(W, U[k + 1 :])
            W = mttv_left(W, U[k])
        V[-1] = W
        return V

    @property
    def ndims(self) -> int:
        """
        Number of dimensions of the tensor.

        Examples
        --------
        >>> T = ttb.tenones((2,2))
        >>> T.ndims
        2
        """
        if self.shape == (0,):
            return 0
        return len(self.shape)

    @property
    def nnz(self) -> int:
        """
        Number of non-zero elements in the tensor.

        Examples
        --------
        >>> T = ttb.tenones((2,2,2))
        >>> T.nnz
        8
        """
        return np.count_nonzero(self.data)

    def norm(self) -> float:
        """
        Frobenius norm of the tensor, defined as the square root of the sum of the
        squares of the elements of the tensor.

        Examples
        --------
        >>> T = ttb.tenones((2,2,2,2))
        >>> T.norm()
        4.0
        """
        # default of np.linalg.norm is to vectorize the data and compute the vector
        # norm, which is equivalent to the Frobenius norm for multidimensional arrays.
        # However, the argument 'fro' only works for 1-D and 2-D arrays currently.
        return float(np.linalg.norm(self.data))

    def nvecs(self, n: int, r: int, flipsign: bool = True) -> np.ndarray:
        """
        Compute the leading mode-n vectors of the tensor.

        Computes the `r` leading eigenvectors of Tn*Tn.T (where Tn is the
        mode-`n` matricization/unfolding of self), which provides information
        about the mode-n fibers. In two-dimensions, the `r` leading mode-1
        vectors are the same as the `r` left singular vectors and the `r`
        leading mode-2 vectors are the same as the `r` right singular
        vectors. By default, this method computes the top `r` eigenvectors
        of Tn*Tn.T.

        Parameters
        ----------
        n:
            Mode for tensor matricization.
        r:
            Number of eigenvectors to compute and use.
        flipsign:
            If True, make each column's largest element positive.

        Returns
        -------
        Computed eigenvectors.

        Examples
        --------
        >>> T = ttb.tensor(np.array([[1, 2], [3, 4]]))
        >>> T.nvecs(0,1)  # doctest: +ELLIPSIS
        array([[0.4045...],
               [0.9145...]])
        >>> T.nvecs(0,2)  # doctest: +ELLIPSIS
        array([[ 0.4045...,  0.9145...],
               [ 0.9145..., -0.4045...]])
        """
        Xn = ttb.tenmat.from_tensor_type(self, rdims=np.array([n])).double()
        y = Xn @ Xn.T

        if r < y.shape[0] - 1:
            w, v = scipy.sparse.linalg.eigsh(y, r)
            v = v[:, (-np.abs(w)).argsort()]
            v = v[:, :r]
        else:
            logging.debug(
                "Greater than or equal to tensor.shape[n] - 1 eigenvectors"
                " requires cast to dense to solve"
            )
            w, v = scipy.linalg.eigh(y)
            v = v[:, (-np.abs(w)).argsort()]
            v = v[:, :r]

        if flipsign:
            idx = np.argmax(np.abs(v), axis=0)
            for i in range(v.shape[1]):
                if v[idx[i], i] < 0:
                    v[:, i] *= -1
        return v

    def permute(self, order: np.ndarray) -> tensor:
        """
        Permute tensor dimensions. The result is a tensor that has the
        same values, but the order of the subscripts needed to access
        any particular element are rearranged as specified by `order`.

        Parameters
        ----------
        order:
            New order of tensor dimensions.

        Returns
        -------
        New tensor with permuted dimensions.

        Examples
        --------
        >>> T1 = ttb.tensor(np.array([[1, 2], [3, 4]]))
        >>> T1
        tensor of shape (2, 2)
        data[:, :] =
        [[1 2]
         [3 4]]
        >>> T1.permute(np.array((1,0)))
        tensor of shape (2, 2)
        data[:, :] =
        [[1 3]
         [2 4]]
        """
        if self.ndims != order.size:
            assert False, "Invalid permutation order"

        # If order is empty, return
        if order.size == 0:
            return self.copy()

        # Check for special case of an order-1 object, has no effect
        if (order == 1).all():
            return self.copy()

        # Np transpose does error checking on order, acts as permutation
        return ttb.tensor(np.transpose(self.data, order), copy=False)

    def reshape(self, shape: Tuple[int, ...]) -> tensor:
        """
        Reshape the tensor.

        Parameters
        ----------
        shape:
            New shape

        Examples
        --------
        >>> T1 = ttb.tenones((2,2))
        >>> T1.shape
        (2, 2)
        >>> T2 = T1.reshape((4,1))
        >>> T2.shape
        (4, 1)
        """
        if np.prod(self.shape) != np.prod(shape):
            assert False, "Reshaping a tensor cannot change number of elements"

        return ttb.tensor(np.reshape(self.data, shape, order="F"), shape)

    def scale(
        self,
        factor: Union[np.ndarray, ttb.tensor],
        dims: Union[float, np.ndarray],
    ) -> tensor:
        """
        Scale along specified dimensions for tensors.

        Parameters
        ----------
        factor: Scaling factor
        dims: Dimensions to scale

        Returns
        -------
        Scaled Tensor.

        Examples
        --------
        >>> T = ttb.tenones((3, 4, 5))
        >>> S = np.arange(5)
        >>> Y = T.scale(S, 2)
        >>> Y.data[0, 0, :]
        array([0., 1., 2., 3., 4.])
        >>> S = ttb.tensor(np.arange(5))
        >>> Y = T.scale(S, 2)
        >>> Y.data[0, 0, :]
        array([0., 1., 2., 3., 4.])
        >>> S = ttb.tensor(np.arange(12), shape=(3, 4))
        >>> Y = T.scale(S, [0, 1])
        >>> Y.data[:, :, 0]
        array([[ 0.,  3.,  6.,  9.],
               [ 1.,  4.,  7., 10.],
               [ 2.,  5.,  8., 11.]])
        """
        if isinstance(dims, list):
            dims = np.array(dims)
        elif isinstance(dims, (float, int, np.generic)):
            dims = np.array([dims])

        # TODO update tt_dimscheck overload so I don't need explicit
        #   Nones to appease mypy
        dims, _ = tt_dimscheck(self.ndims, None, dims, None)
        remdims = np.setdiff1d(np.arange(0, self.ndims), dims)

        if not np.array_equal(factor.shape, np.array(self.shape)[dims]):
            raise ValueError(
                f"Scaling factor has shape {factor.shape}, but dimensions "
                f"to scale had shape {np.array(self.shape)[dims]}"
            )
        if isinstance(factor, np.ndarray):
            if len(factor.shape) == 1:
                factor = factor[:, None]
            factor = ttb.tensor(factor, copy=False)
        # TODO this should probably be doable directly as a numpy view
        #   where I think this is currently a copy
        vector_factor = ttb.tenmat.from_tensor_type(
            factor, np.arange(factor.ndims)
        ).double()
        vector_self = ttb.tenmat.from_tensor_type(self, dims, remdims).double()
        # Numpy broadcasting should be equivalent to bsxfun
        result = vector_self * vector_factor
        # TODO why do we need this transpose for things to work?
        if len(dims) == 1:
            result = result.transpose()
        return ttb.tenmat.from_data(result, dims, remdims, self.shape).to_tensor()

    def squeeze(self) -> Union[tensor, float]:
        """
        Removes singleton dimensions from the tensor.

        Returns
        -------
        Tensor or scalar if all dims squeezed.

        Examples
        --------
        >>> T = ttb.tensor(np.array([[[4]]]))
        >>> T.squeeze()
        4
        >>> T = ttb.tensor(np.array([[1, 2, 3]]))
        >>> T.squeeze().data
        array([1, 2, 3])
        """
        shapeArray = np.array(self.shape)
        if np.all(shapeArray > 1):
            return self.copy()
        else:
            idx = np.where(shapeArray > 1)
            if idx[0].size == 0:
                return self.data.item()
            return ttb.tensor(np.squeeze(self.data))

    def symmetrize(  # noqa: PLR0912,PLR0915
        self, grps: Optional[np.ndarray] = None, version: Optional[Any] = None
    ) -> tensor:
        """
        Symmetrize a tensor in the specified modes.

        It is *the same or less* work to just call T = T.symmetrize() then to first
        check if T is symmetric and then symmetrize it, even if T is already symmetric.

        Parameters
        ----------
        grps:
            Modes to check for symmetry.
        version:
            Any non-None value will call the non-default old version.

        Returns
        -------
        Symmetrized tensor.

        Examples
        --------
        >>> T = ttb.tenones((2,2,2))
        >>> T.symmetrize(np.array([0,2]))
        tensor of shape (2, 2, 2)
        data[0, :, :] =
        [[1. 1.]
         [1. 1.]]
        data[1, :, :] =
        [[1. 1.]
         [1. 1.]]
        """
        n = self.ndims
        sz = np.array(self.shape)

        if grps is None:
            grps = np.arange(0, n)

        if len(grps.shape) == 1:
            grps = np.array([grps])

        data = self.data.copy()

        # Use default newer faster version
        if version is None:
            ngrps = len(grps)
            for i in range(0, ngrps):
                # Extract current group
                thisgrp = grps[i]

                # Check tensor dimensions first
                if not np.all(sz[thisgrp[0]] == sz[thisgrp]):
                    assert False, "Dimension mismatch for symmetrization"

                # Check for no overlap in the sets
                if i < ngrps - 1:
                    if not np.intersect1d(thisgrp, grps[i + 1 :, :]).size == 0:
                        assert False, "Cannot have overlapping symmetries"

                # Construct matrix ind where each row is the multi-index for one
                # element of tensor
                idx = tt_ind2sub(self.shape, np.arange(0, data.size))

                # Find reference index for every element in the tensor - this
                # is to its index in the symmetrized tensor. This puts every
                # element into a 'class' of entries that will be the same under
                # symmetry.

                classidx = idx
                classidx[:, thisgrp] = np.sort(idx[:, thisgrp], axis=1)
                linclassidx = tt_sub2ind(self.shape, classidx)

                # Compare each element to its class exemplar
                if np.all(data.ravel() == data[tuple(classidx.transpose())]):
                    continue

                # Take average over all elements in the same class
                classSum = accumarray(linclassidx, data.ravel())
                classNum = accumarray(linclassidx, 1)
                # We ignore this division error state because if we don't have an entry
                # in linclassidx we won't reference the inf or nan in the slice below
                with np.errstate(divide="ignore", invalid="ignore"):
                    avg = classSum / classNum

                newdata = avg[linclassidx]
                data = np.reshape(newdata, self.shape)

            return ttb.tensor(data, copy=False)

        else:  # Original version
            # Check tensor dimensions for compatibility with symmetrization
            ngrps = len(grps)
            for i in range(0, ngrps):
                dims = grps[i]
                for j in dims[1:]:
                    if sz[j] != sz[dims[0]]:
                        assert False, "Dimension mismatch for symmetrization"

            # Check for no overlap in sets
            for i in range(0, ngrps):
                for j in range(i + 1, ngrps):
                    if not np.intersect1d(grps[i, :], grps[j, :]).size == 0:
                        assert False, "Cannot have overlapping symmetries"

            # Create the combinations for each symmetrized subset
            combos = []
            for i in range(0, ngrps):
                combos.append(np.array(list(permutations(grps[i, :]))))
            combos = np.stack(combos)

            # Create all the permuations to be averaged
            combo_lengths = [len(perm) for perm in combos]
            total_perms = int(np.prod(combo_lengths))
            sym_perms = np.tile(np.arange(0, n), [total_perms, 1])
            for i in range(0, ngrps):
                ntimes = np.prod(combo_lengths[0:i], dtype=int)
                ncopies = np.prod(combo_lengths[i + 1 :], dtype=int)
                nelems = len(combos[i])

                perm_idx = 0
                for _ in range(0, ntimes):
                    for k in range(0, nelems):
                        for _ in range(0, ncopies):
                            # TODO: Does this do anything? Matches MATLAB
                            # at very least should be able to flatten
                            sym_perms[perm_idx, grps[i]] = combos[i][k, :]
                            perm_idx += 1

            # Create an average tensor
            Y = ttb.tensor(np.zeros(self.shape), copy=False)
            for i in range(0, total_perms):
                Y += self.permute(sym_perms[i, :])

            Y /= total_perms

            # It's not *exactly* symmetric due to oddities in differently ordered
            # summations and so on, so let's fix that.
            # Idea borrowed from Gergana Bounova:
            # http://www.mit.edu/~gerganaa/downloads/matlab/symmetrize.m
            for i in range(0, total_perms):
                Z = Y.permute(sym_perms[i, :])
                Y.data[:] = np.maximum(Y.data[:], Z.data[:])

            return Y

    def ttm(
        self,
        matrix: Union[np.ndarray, List[np.ndarray]],
        dims: Optional[Union[float, np.ndarray]] = None,
        exclude_dims: Optional[Union[int, np.ndarray]] = None,
        transpose: bool = False,
    ) -> tensor:
        """
        Tensor times matrix.

        Computes the n-mode product of `self` with the matrix `matrix`; i.e.,
        `self x_n matrix`. The integer `n` specifies the dimension (or mode)
        along which the matrix should be multiplied. If `matrix.shape = (J,I)`,
        then the tensor must have `self.shape[n] = I`. The result will be the
        same order and shape as `self` except that the size of dimension `n`
        will be `J`.

        Multiplication with more than one matrix is provided using a list of
        matrices and corresponding dimensions in the tensor to use. Multiplication
        using the transpose of the matrix (or matrices) is also provided.

        The dimensions of the tensor with which to multiply can be provided as
        `dims`, or the dimensions to exclude from `[0, ..., self.ndims]` can be
        specified using `exclude_dims`.

        Parameters
        ----------
        matrix:
            Matrix or matrices to multiple by.
        dims:
            Dimensions to multiply against.
        exclude_dims:
            Use all dimensions but these.
        transpose:
            Transpose matrices during multiplication.

        Returns
        -------
        Tensor product.

        Examples
        --------
        >>> T = ttb.tenones((2,2,2,2))
        >>> A = 2*np.ones((2,1))
        >>> T.ttm([A,A], dims=[0,1], transpose=True)
        tensor of shape (1, 1, 2, 2)
        data[0, 0, :, :] =
        [[16. 16.]
         [16. 16.]]
        >>> T.ttm([A,A], exclude_dims=[0,1], transpose=True)
        tensor of shape (2, 2, 1, 1)
        data[0, 0, :, :] =
        [[16.]]
        data[1, 0, :, :] =
        [[16.]]
        data[0, 1, :, :] =
        [[16.]]
        data[1, 1, :, :] =
        [[16.]]
        """
        if dims is None and exclude_dims is None:
            dims = np.arange(self.ndims)
        elif isinstance(dims, list):
            dims = np.array(dims)
        elif isinstance(dims, (float, int, np.generic)):
            dims = np.array([dims])

        if isinstance(exclude_dims, (float, int)):
            exclude_dims = np.array([exclude_dims])

        if isinstance(matrix, list):
            # Check that the dimensions are valid
            dims, vidx = tt_dimscheck(self.ndims, len(matrix), dims, exclude_dims)

            # Calculate individual products
            Y = self.ttm(matrix[vidx[0]], dims[0], transpose=transpose)
            for k in range(1, dims.size):
                Y = Y.ttm(matrix[vidx[k]], dims[k], transpose=transpose)
            return Y

        if not isinstance(matrix, (np.ndarray, sparse.spmatrix)):
            assert False, f"matrix must be of type numpy.ndarray but got:\n{matrix}"

        dims, _ = tt_dimscheck(self.ndims, dims=dims, exclude_dims=exclude_dims)

        if not (dims.size == 1 and np.isin(dims, np.arange(self.ndims))):
            assert False, "dims must contain values in [0,self.dims)"

        # old version (ver=0)
        shape = np.array(self.shape, dtype=int)
        n = dims[0]
        order = np.array([n, *list(range(0, n)), *list(range(n + 1, self.ndims))])
        newdata = self.permute(order).data
        ids = np.array(list(range(0, n)) + list(range(n + 1, self.ndims)))
        second_dim = 1
        if len(ids) > 0:
            second_dim = np.prod(shape[ids])
        newdata = np.reshape(newdata, (shape[n], second_dim), order="F")
        if transpose:
            newdata = matrix.T @ newdata
            p = matrix.shape[1]
        else:
            newdata = matrix @ newdata
            p = matrix.shape[0]

        newshape = np.array(
            [p, *list(shape[range(0, n)]), *list(shape[range(n + 1, self.ndims)])]
        )
        Y_data = np.reshape(newdata, newshape, order="F")
        Y_data = np.transpose(Y_data, np.argsort(order))
        return ttb.tensor(Y_data, copy=False)

    def ttt(
        self,
        other: tensor,
        selfdims: Optional[Union[int, np.ndarray]] = None,
        otherdims: Optional[Union[int, np.ndarray]] = None,
    ) -> tensor:
        """
        Tensor multiplication (tensor times tensor).

        Computes the contracted product of tensors, self and other, in the
        dimensions specified by the `selfdims` and `otherdims`. The sizes of
        the dimensions specified by `selfdims` and `otherdims` must match;
        that is, `self.shape(selfdims)` must equal `other.shape(otherdims)`.
        If only `selfdims` is provided as input, it is used to specify the
        dimensions for both `self` and `other`.

        Parameters
        ----------
        other:
            Tensor to multiply by.
        selfdims:
            Dimensions to contract self by for multiplication.
        otherdims:
            Dimensions to contract other tensor by for multiplication.

        Returns
        -------
        Tensor product.

        Examples
        --------
        >>> T = ttb.tensor(np.array([[1, 2], [3, 4]]))
        >>> T.ttt(T)
        tensor of shape (2, 2, 2, 2)
        data[0, 0, :, :] =
        [[1 2]
         [3 4]]
        data[1, 0, :, :] =
        [[ 3  6]
         [ 9 12]]
        data[0, 1, :, :] =
        [[2 4]
         [6 8]]
        data[1, 1, :, :] =
        [[ 4  8]
         [12 16]]
        >>> T.ttt(T, 0)
        tensor of shape (2, 2)
        data[:, :] =
        [[10 14]
         [14 20]]
        >>> T.ttt(T, selfdims=0, otherdims=1)
        tensor of shape (2, 2)
        data[:, :] =
        [[ 7 15]
         [10 22]]
        """
        if not isinstance(other, tensor):
            assert False, "other must be of type tensor"

        if selfdims is None:
            selfdims = np.array([], dtype=int)
        elif isinstance(selfdims, int):
            selfdims = np.array([selfdims])
        selfshape = tuple(np.array(self.shape)[selfdims])

        if otherdims is None:
            otherdims = selfdims.copy()
        elif isinstance(otherdims, int):
            otherdims = np.array([otherdims])
        othershape = tuple(np.array(other.shape)[otherdims])

        if np.any(selfshape != othershape):
            assert (
                False
            ), f"Specified dimensions do not match got {selfshape} and {othershape}"

        # Compute the product

        # Avoid transpose by reshaping self and computing result = self * other
        amatrix = ttb.tenmat.from_tensor_type(self, cdims=selfdims)
        bmatrix = ttb.tenmat.from_tensor_type(other, rdims=otherdims)
        cmatrix = amatrix * bmatrix

        # Check whether or not the result is a scalar
        if isinstance(cmatrix, ttb.tenmat):
            return cmatrix.to_tensor()
        return cmatrix

    def ttv(
        self,
        vector: Union[np.ndarray, List[np.ndarray]],
        dims: Optional[Union[int, np.ndarray]] = None,
        exclude_dims: Optional[Union[int, np.ndarray]] = None,
    ) -> Union[float, tensor]:
        """
        Tensor times vector.

        Computes the n-mode product of `self` with the vector `vector`; i.e.,
        `self x_n vector`. The integer `n` specifies the dimension (or mode)
        along which the vector should be multiplied. If `vector.shape = (I,)`,
        then the tensor must have `self.shape[n] = I`. The result will be the
        same order and shape as `self` except that the size of dimension `n`
        will be `J`. The resulting tensor has one less dimension, as dimension
        `n` is removed in the multiplication.

        Multiplication with more than one vector is provided using a list of
        vectors and corresponding dimensions in the tensor to use.

        The dimensions of the tensor with which to multiply can be provided as
        `dims`, or the dimensions to exclude from `[0, ..., self.ndims]` can be
        specified using `exclude_dims`.

        Parameters
        ----------
        vector:
            Vector or vectors to multiple by.
        dims:
            Dimensions to multiply against.
        exclude_dims:
            Use all dimensions but these.

        Returns
        -------
        Tensor product.

        Examples
        --------
        >>> T = ttb.tensor(np.array([[1, 2], [3, 4]]))
        >>> T.ttv(np.ones(2),0)
        tensor of shape (2,)
        data[:] =
        [4. 6.]
        >>> T.ttv(np.ones(2),1)
        tensor of shape (2,)
        data[:] =
        [3. 7.]
        >>> T.ttv([np.ones(2), np.ones(2)])
        10.0
        """
        if dims is None and exclude_dims is None:
            dims = np.array([])
        elif isinstance(dims, (float, int)):
            dims = np.array([dims])

        if isinstance(exclude_dims, (float, int)):
            exclude_dims = np.array([exclude_dims])

        # Check that vector is a list of vectors, if not place single vector as element
        # in list
        if len(vector) > 0 and isinstance(vector[0], (int, float, np.int_, np.float64)):
            return self.ttv(np.array([vector]), dims, exclude_dims)

        # Get sorted dims and index for multiplicands
        dims, vidx = tt_dimscheck(self.ndims, len(vector), dims, exclude_dims)

        # Check that each multiplicand is the right size.
        for i in range(dims.size):
            if vector[vidx[i]].shape != (self.shape[dims[i]],):
                assert False, "Multiplicand is wrong size"

        # Extract the data
        c = self.data.copy()

        # Permute it so that the dimensions we're working with come last
        remdims = np.setdiff1d(np.arange(0, self.ndims), dims)
        if self.ndims > 1:
            c = np.transpose(c, np.concatenate((remdims, dims)))

        # Do each multiply in sequence, doing the highest index first, which is
        # important for vector multiplies.
        n = self.ndims
        sz = np.array(self.shape)[np.concatenate((remdims, dims))]

        for i in range(dims.size - 1, -1, -1):
            c = np.reshape(c, tuple([np.prod(sz[0 : n - 1]), sz[n - 1]]), order="F")
            c = c.dot(vector[vidx[i]])
            n -= 1
        # If needed, convert the final result back to tensor
        if n > 0:
            return ttb.tensor(c, tuple(sz[0:n]), copy=False)
        return c[0]

    def ttsv(
        self,
        vector: np.ndarray,
        skip_dim: Optional[int] = None,
        version: Optional[int] = None,
    ) -> Union[float, np.ndarray, tensor]:
        """
        Tensor times same vector in multiple modes.

        See :meth:`ttv` for details on multiplication of a tensor with a
        vector. When `skip_dim` is provided, multiply the vector by all but
        dimensions except `[0, ..., skip_dim]`.

        Parameters
        ----------
        vector:
            Vector to multiply by.
        skip_dim:
            Initial dimensions of the tensor to skip when multiplying.

        Examples
        --------
        >>> T = ttb.tensor(np.array([[1, 2], [3, 4]]))
        >>> T.ttsv(np.ones(2))
        10.0
        >>> T.ttsv(np.ones(2),0)
        array([3., 7.])
        >>> T.ttsv(np.ones(2),1)
        array([[1, 2],
               [3, 4]])
        """
        # Only two simple cases are supported
        if skip_dim is None:
            exclude_dims = None
            skip_dim = -1  # For easier math later
        elif skip_dim < 0:
            raise ValueError("Invalid modes in ttsv")
        else:
            exclude_dims = np.arange(0, skip_dim + 1)

        if version == 1:  # Calculate the old way
            P = self.ndims
            X = np.array([vector for i in range(P)])
            if skip_dim in (0, 1):  # Return matrix
                result = self.ttv(X, exclude_dims=exclude_dims)
                assert not isinstance(result, float)
                return result.double()
            return self.ttv(X, exclude_dims=exclude_dims)

        if version == 2 or version is None:  # Calculate the new way
            d = self.ndims
            sz = self.shape[0]  # Sizes of all modes must be the same

            dnew = skip_dim + 1  # Number of modes in result
            drem = d - dnew  # Number of modes multiplied out

            y = self.data.copy()
            for i in range(drem, 0, -1):
                yy = np.reshape(y, (sz ** (dnew + i - 1), sz), order="F")
                y = yy.dot(vector)

            # Convert to matrix if 2-way or convert back to tensor if result is >= 3-way
            if dnew == 2:
                return np.reshape(y, [sz, sz], order="F")
            if dnew > 2:
                return ttb.tensor(
                    np.reshape(y, newshape=sz * np.ones(dnew, dtype=int), order="F"),
                    copy=False,
                )

            # extract scalar if needed
            if len(y) == 1:
                y = y.item()

            return y
        assert False, "Invalid value for version; should be None, 1, or 2"

    def __setitem__(self, key, value):
        """
        Subscripted assignment for a tensor.

        We can assign elements to a tensor in three ways.

        Case 1: `T[R1,R2,...,Rn] = Y`, in which case we replace the
        rectangular subtensor (or single element) specified by the ranges
        `R1`,...,`Rn` with `Y`. The right-hand-side can be a scalar, a tensor,
        or a :class:`numpy.ndarray`.

        Case 2a: `T[S] = V`, where `S` is a `p` x `n` array of subscripts and `V` is
        a scalar or a vector containing `p` values.

        Case 2b: `T[I] = V`, where `I` is a set of `p` linear indices and `V` is a
        scalar or a vector containing p values. Resizing is not allowed in this
        case.

        Examples
        --------
        >>> T = tenones((3,4,2))
        >>> # replaces subtensor
        >>> T[0:2,0:2,0] = np.ones((2,2))
        >>> # replaces two elements
        >>> T[np.array([[1, 1, 1], [1, 1, 2]])] = [5, 7]
        >>> # replaces two elements with linear indices
        >>> T[np.array([1, 13])] = [5, 7]
        >>> # grows tensor to accept new element
        >>> T[1,1,2:3] = 1
        >>> T[1,1,4] = 1
        """
        access_type = get_index_variant(key)

        # Case 1: Rectangular Subtensor
        if access_type == IndexVariant.SUBTENSOR:
            return self._set_subtensor(key, value)

        # Case 2a: Subscript indexing
        if access_type == IndexVariant.SUBSCRIPTS:
            return self._set_subscripts(key, value)

        # Case 2b: Linear Indexing
        if access_type == IndexVariant.LINEAR:
            if isinstance(key, list):
                key = np.array(key)
            return self._set_linear(key, value)

        assert False, "Invalid use of tensor setitem"

    def _set_linear(self, key, value):
        idx = key
        if not isinstance(idx, slice) and (idx > np.prod(self.shape)).any():
            assert (
                False
            ), "TTB:BadIndex In assignment X[I] = Y, a tensor X cannot be resized"
        if isinstance(key, (int, float, np.generic)):
            idx = np.array([key])
        elif isinstance(key, slice):
            idx = np.array(range(np.prod(self.shape))[key])
        idx = tt_ind2sub(self.shape, idx)
        if idx.shape[0] == 1:
            self.data[tuple(idx[0, :])] = value
        else:
            actualIdx = tuple(idx.transpose())
            self.data[actualIdx] = value

    def _set_subtensor(self, key, value):  # noqa: PLR0912
        # Extract array of subscripts
        subs = key
        # Will the size change? If so we first need to resize x
        n = self.ndims
        sliceCheck = []
        for element in subs:
            if isinstance(element, slice):
                if element.stop is None:
                    sliceCheck.append(1)
                else:
                    sliceCheck.append(element.stop - 1)
            elif isinstance(element, Iterable):
                if any(
                    not isinstance(entry, (float, int, np.generic)) for entry in element
                ):
                    raise ValueError(
                        f"Entries for setitem must be numeric but recieved, {element}"
                    )
                sliceCheck.append(max(element))
            else:
                sliceCheck.append(element)
        bsiz = np.array(sliceCheck)
        if n == 0:
            newsiz = (bsiz[n:] + 1).astype(int)
        else:
            newsiz = np.concatenate(
                (np.max((self.shape, bsiz[0:n] + 1), axis=0), bsiz[n:] + 1)
            ).astype(int)
        if not np.array_equal(newsiz, self.shape):
            # We need to enlarge x.data.
            newData = np.zeros(shape=tuple(newsiz))
            if self.data.size > 0:
                idx = [slice(None, currentShape) for currentShape in self.shape]
                idx.extend([0] * (len(newsiz) - self.ndims))
                newData[tuple(idx)] = self.data
            self.data = newData

            self.shape = tuple(newsiz)
        if isinstance(value, ttb.tensor):
            self.data[key] = value.data
        else:
            self.data[key] = value

    def _set_subscripts(self, key, value):
        # Extract array of subscripts
        subs = key

        # Will the size change? If so we first need to resize x
        n = self.ndims
        bsiz = np.array(np.max(subs, axis=0))
        if n == 0:
            newsiz = (bsiz[n:] + 1).astype(int)
        else:
            newsiz = np.concatenate(
                (np.max((self.shape, bsiz[0:n] + 1), axis=0), bsiz[n:] + 1)
            ).astype(int)

        if not np.array_equal(newsiz, self.shape):
            # We need to enlarge x.data.
            newData = np.zeros(shape=tuple(newsiz))
            if self.data.size > 0:
                idx = [slice(None, currentShape) for currentShape in self.shape]
                idx.extend([0] * (len(newsiz) - self.ndims))
                newData[tuple(idx)] = self.data
            self.data = newData

            self.shape = tuple(newsiz)

        # Finally we can copy in new data
        if key.shape[0] == 1:  # and len(key.shape) == 1:
            self.data[tuple(key[0, :])] = value
        else:
            self.data[tuple(key.transpose())] = value

    def __getitem__(self, item):  # noqa: PLR0912
        """
        Subscripted reference for tensors.

        We can extract elements or subtensors from a tensor in the
        following ways.

        Case 1a: `y = T[I1,I2,...,In]`, where each `I` is an index, returns a
        scalar.

        Case 1b: `Y = T[R1,R2,...,Rn]`, where one or more `R` is a range and
        the rest are indices, returns a tensor.

        Case 2a: `V = T[S]` where `S` is a `p` x `n` array
        of subscripts, returns a vector of `p` values.

        Case 2b: `V = T[I]` where `I` is a set of `p`
        linear indices, returns a vector of `p` values.

        Any ambiguity results in executing the first valid case. This
        is particularly an issue if `self.ndims == 1`.

        Examples
        --------
        >>> T = tenones((3,4,2,1))
        >>> T[0,0,0,0] # produces a scalar
        1.0
        >>> # produces a tensor of order 1 and size 1
        >>> T[1,1,1,:] # doctest: +NORMALIZE_WHITESPACE
        tensor of shape (1,)
        data[:] =
        [1.]
        >>> # produces a tensor of size 2 x 2 x 1
        >>> T[0:2,[2, 3],1,:] # doctest: +NORMALIZE_WHITESPACE
        tensor of shape (2, 2, 1)
        data[0, :, :] =
        [[1.]
         [1.]]
        data[1, :, :] =
        [[1.]
         [1.]]
        >>> # returns a vector of length 2
        >>> # Equivalent to selecting [0,0,0,0] and [1,1,1,0] separately
        >>> T[np.array([[0, 0, 0, 0], [1, 1, 1, 0]])]
        array([1., 1.])
        >>> T[[0,1,2]] # extracts the first three linearized indices
        array([1., 1., 1.])
        """
        # Case 0: Single Index Linear
        if isinstance(item, (int, float, np.generic, slice)):
            if isinstance(item, (int, float, np.generic)):
                idx = np.array(item)
            elif isinstance(item, slice):
                idx = np.array(range(np.prod(self.shape))[item])
            a = np.squeeze(self.data[tuple(tt_ind2sub(self.shape, idx).transpose())])
            # Todo if row make column?
            return tt_subsubsref(a, idx)
        # Case 1: Rectangular Subtensor
        if isinstance(item, tuple) and len(item) == self.ndims:
            # Copy the subscripts
            region = item

            # Extract the data
            newdata = self.data[region]

            # Determine the subscripts
            newsiz = []  # future new size
            kpdims = []  # dimensions to keep
            rmdims = []  # dimensions to remove

            # Determine the new size and what dimensions to keep
            for i, a_region in enumerate(region):
                if isinstance(a_region, slice):
                    newsiz.append(self.shape[i])
                    kpdims.append(i)
                elif not isinstance(a_region, (int, np.integer)) and len(a_region) > 1:
                    newsiz.append(np.prod(a_region))
                    kpdims.append(i)
                else:
                    rmdims.append(i)

            newsiz = np.array(newsiz, dtype=int)
            kpdims = np.array(kpdims, dtype=int)
            rmdims = np.array(rmdims, dtype=int)

            # If the size is zero, then the result is returned as a scalar
            # otherwise, we convert the result to a tensor
            if newsiz.size == 0:
                a = newdata
            else:
                a = ttb.tensor(newdata, copy=False)
            return a

        # *** CASE 2a: Subscript indexing ***
        is_subscript = (
            isinstance(item, np.ndarray)
            and len(item.shape) == 2
            and item.shape[-1] == self.ndims
        )
        if is_subscript:
            # Extract array of subscripts
            subs = np.array(item)
            a = np.squeeze(self.data[tuple(subs.transpose())])
            # TODO if is row make column?
            return tt_subsubsref(a, subs)

        # Case 2b: Linear Indexing
        if isinstance(item, tuple) and len(item) >= 2:
            assert False, "Linear indexing requires single input array"

        if (isinstance(item, np.ndarray) and len(item.shape) == 1) or (
            isinstance(item, list)
            and all(isinstance(element, (int, np.integer)) for element in item)
        ):
            idx = np.array(item)
            a = np.squeeze(self.data[tuple(tt_ind2sub(self.shape, idx).transpose())])
            # Todo if row make column?
            return tt_subsubsref(a, idx)

        assert False, "Invalid use of tensor getitem"

    def __eq__(self, other):
        """
        Equal for tensors (element-wise).

        Parameters
        ----------
        other: :class:`pyttb.tensor`, float, int

        Returns
        -------
        :class:`pyttb.tensor` of `bool`.

        Examples
        --------
        >>> T = ttb.tensor(np.array([[1, 2], [3, 4]]))
        >>> T == T
        tensor of shape (2, 2)
        data[:, :] =
        [[ True  True]
         [ True  True]]
        >>> T == 1
        tensor of shape (2, 2)
        data[:, :] =
        [[ True False]
         [False False]]
        """

        def tensor_equality(x, y):
            return x == y

        return tt_tenfun(tensor_equality, self, other)

    def __ne__(self, other):
        """
        Not equal (!=) for tensors (element-wise).

        Parameters
        ----------
        other: :class:`pyttb.tensor`, float, int

        Returns
        -------
        :class:`pyttb.tensor` of `bool`.

        Examples
        --------
        >>> T = ttb.tensor(np.array([[1, 2], [3, 4]]))
        >>> T != T
        tensor of shape (2, 2)
        data[:, :] =
        [[False False]
         [False False]]
        >>> T != 1
        tensor of shape (2, 2)
        data[:, :] =
        [[False  True]
         [ True  True]]
        """

        def tensor_not_equal(x, y):
            return x != y

        return tt_tenfun(tensor_not_equal, self, other)

    def __ge__(self, other):
        """
        Greater than or equal (>=) for tensors (element-wise).

        Parameters
        ----------
        other: :class:`pyttb.tensor`, float, int

        Returns
        -------
        :class:`pyttb.tensor` of `bool`.

        Examples
        --------
        >>> T = ttb.tensor(np.array([[1, 2], [3, 4]]))
        >>> T >= T
        tensor of shape (2, 2)
        data[:, :] =
        [[ True  True]
         [ True  True]]
        >>> T >= 1
        tensor of shape (2, 2)
        data[:, :] =
        [[ True  True]
         [ True  True]]
        """

        def greater_or_equal(x, y):
            return x >= y

        return tt_tenfun(greater_or_equal, self, other)

    def __le__(self, other):
        """
        Less than or equal (<=) for tensors (element-wise).

        Parameters
        ----------
        other: :class:`pyttb.tensor`, float, int

        Returns
        -------
        :class:`pyttb.tensor` of `bool`.

        Examples
        --------
        >>> T = ttb.tensor(np.array([[1, 2], [3, 4]]))
        >>> T <= T
        tensor of shape (2, 2)
        data[:, :] =
        [[ True  True]
         [ True  True]]
        >>> T <= 1
        tensor of shape (2, 2)
        data[:, :] =
        [[ True False]
         [False False]]
        """

        def less_or_equal(x, y):
            return x <= y

        return tt_tenfun(less_or_equal, self, other)

    def __gt__(self, other):
        """
        Greater than (>) for tensors (element-wise).

        Parameters
        ----------
        other: :class:`pyttb.tensor`, float, int

        Returns
        -------
        :class:`pyttb.tensor` of `bool`.

        Examples
        --------
        >>> T = ttb.tensor(np.array([[1, 2], [3, 4]]))
        >>> T > T
        tensor of shape (2, 2)
        data[:, :] =
        [[False False]
         [False False]]
        >>> T > 1
        tensor of shape (2, 2)
        data[:, :] =
        [[False  True]
         [ True  True]]
        """

        def greater(x, y):
            return x > y

        return tt_tenfun(greater, self, other)

    def __lt__(self, other):
        """
        Less than (<) for tensors (element-wise).

        Parameters
        ----------
        other: :class:`pyttb.tensor`, float, int

        Returns
        -------
        :class:`pyttb.tensor` of `bool`.

        Examples
        --------
        >>> T = ttb.tensor(np.array([[1, 2], [3, 4]]))
        >>> T < T
        tensor of shape (2, 2)
        data[:, :] =
        [[False False]
         [False False]]
        >>> T < 1
        tensor of shape (2, 2)
        data[:, :] =
        [[False False]
         [False False]]
        """

        def less(x, y):
            return x < y

        return tt_tenfun(less, self, other)

    def __sub__(self, other):
        """
        Binary subtraction (-) for tensors.

        Parameters
        ----------
        other: :class:`pyttb.tensor`, float, int

        Returns
        -------
        :class:`pyttb.tensor`

        Examples
        --------
        >>> T = ttb.tensor(np.array([[1, 2], [3, 4]]))
        >>> T - T
        tensor of shape (2, 2)
        data[:, :] =
        [[0 0]
         [0 0]]
        >>> T - 1
        tensor of shape (2, 2)
        data[:, :] =
        [[0 1]
         [2 3]]
        """

        def minus(x, y):
            return x - y

        return tt_tenfun(minus, self, other)

    def __add__(self, other):
        """
        Binary addition (+) for tensors.

        Parameters
        ----------
        other: :class:`pyttb.tensor`, float, int

        Returns
        -------
        :class:`pyttb.tensor`

        Examples
        --------
        >>> T = ttb.tensor(np.array([[1, 2], [3, 4]]))
        >>> T + T
        tensor of shape (2, 2)
        data[:, :] =
        [[2 4]
         [6 8]]
        >>> T + 1
        tensor of shape (2, 2)
        data[:, :] =
        [[2 3]
         [4 5]]
        """
        # If rhs is sumtensor, treat as such
        if isinstance(other, ttb.sumtensor):
            return other.__add__(self)

        def tensor_add(x, y):
            return x + y

        return tt_tenfun(tensor_add, self, other)

    def __radd__(self, other):
        """
        Right binary addition (+) for tensors

        Parameters
        ----------
        other: :class:`pyttb.tensor`, float, int

        Returns
        -------
        :class:`pyttb.tensor`

        Examples
        --------
        >>> T = ttb.tensor(np.array([[1, 2], [3, 4]]))
        >>> 1 + T
        tensor of shape (2, 2)
        data[:, :] =
        [[2 3]
         [4 5]]
        """
        return self.__add__(other)

    def __pow__(self, power):
        """
        Element-wise Power (**) for tensors.

        Parameters
        ----------
        other::class:`pyttb.tensor`, float, int

        Returns
        -------
        :class:`pyttb.tensor`

        Examples
        --------
        >>> T = ttb.tensor(np.array([[1, 2], [3, 4]]))
        >>> T**2
        tensor of shape (2, 2)
        data[:, :] =
        [[ 1  4]
         [ 9 16]]
        """

        def tensor_pow(x, y):
            return x**y

        return tt_tenfun(tensor_pow, self, power)

    def __mul__(self, other):
        """
        Element-wise multiplication (*) for tensors, self*other

        Parameters
        ----------
        other: :class:`pyttb.tensor`, float, int

        Returns
        -------
        :class:`pyttb.tensor`

        Examples
        --------
        >>> T = ttb.tensor(np.array([[1, 2], [3, 4]]))
        >>> T * T
        tensor of shape (2, 2)
        data[:, :] =
        [[ 1  4]
         [ 9 16]]
        >>> T * 2
        tensor of shape (2, 2)
        data[:, :] =
        [[2 4]
         [6 8]]
        """

        def mul(x, y):
            return x * y

        if isinstance(other, (ttb.ktensor, ttb.sptensor, ttb.ttensor)):
            other = other.full()

        return tt_tenfun(mul, self, other)

    def __rmul__(self, other):
        """
        Element wise right multiplication (*) for tensors, other*self

        Parameters
        ----------
        other: :class:`pyttb.tensor`, float, int

        Returns
        -------
        :class:`pyttb.tensor`

        Examples
        --------
        >>> T = ttb.tensor(np.array([[1, 2], [3, 4]]))
        >>> 2 * T
        tensor of shape (2, 2)
        data[:, :] =
        [[2 4]
         [6 8]]
        """
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        Element-wise left division (/) for tensors, self/other

        Parameters
        ----------
        other: :class:`pyttb.tensor`, float, int

        Returns
        -------
        :class:`pyttb.tensor`

        Examples
        --------
        >>> T = ttb.tensor(np.array([[1, 2], [3, 4]]))
        >>> T / T
        tensor of shape (2, 2)
        data[:, :] =
        [[1. 1.]
         [1. 1.]]
        >>> T / 2
        tensor of shape (2, 2)
        data[:, :] =
        [[0.5 1. ]
         [1.5 2. ]]
        """

        def div(x, y):
            # We ignore the divide by zero errors because np.inf/np.nan is an
            # appropriate representation
            with np.errstate(divide="ignore", invalid="ignore"):
                return x / y

        return tt_tenfun(div, self, other)

    def __rtruediv__(self, other):
        """
        Element wise right division (/) for tensors, other/self

        Parameters
        ----------
        other::class:`pyttb.tensor`, float, int

        Returns
        -------
        :class:`pyttb.tensor`

        Examples
        --------
        >>> T = ttb.tensor(np.array([[1, 2], [3, 4]]))
        >>> np.set_printoptions(precision=8)
        >>> 2 / T  # doctest: +ELLIPSIS
        tensor of shape (2, 2)
        data[:, :] =
        [[2.         1.        ]
         [0.66666... 0.5       ]]
        """

        def div(x, y):
            # We ignore the divide by zero errors because np.inf/np.nan is an
            # appropriate representation
            with np.errstate(divide="ignore", invalid="ignore"):
                return x / y

        return tt_tenfun(div, other, self)

    def __pos__(self):
        """
        Unary plus (+) for tensors.

        Returns
        -------
        Copy of tensor.

        Examples
        --------
        >>> T = ttb.tensor(np.array([[1, 2], [3, 4]]))
        >>> +T
        tensor of shape (2, 2)
        data[:, :] =
        [[1 2]
         [3 4]]
        """
        return self.copy()

    def __neg__(self):
        """
        Unary minus (-) for tensors.

        Returns
        -------
        Copy of negated tensor.

        Examples
        --------
        >>> T = ttb.tensor(np.array([[1, 2], [3, 4]]))
        >>> -T
        tensor of shape (2, 2)
        data[:, :] =
        [[-1 -2]
         [-3 -4]]
        """

        return ttb.tensor(-1 * self.data)

    def __repr__(self):
        """
        String representation of the tensor.

        Returns
        -------
        String displaying shape and data as strings on different lines.

        Examples
        --------
        >>> T = ttb.tensor(np.array([[1, 2], [3, 4]]))
        >>> T
        tensor of shape (2, 2)
        data[:, :] =
        [[1 2]
         [3 4]]
        """
        if self.ndims == 0:
            s = ""
            s += "empty tensor of shape "
            s += str(self.shape)
            s += "\ndata = []"
            return s

        s = ""
        s += f"tensor of shape {self.shape}"

        if self.ndims == 1:
            s += "\ndata"
            if self.ndims == 1:
                s += "[:]"
                s += " =\n"
                s += str(self.data)
                return s
        for i in np.arange(np.prod(self.shape[:-2])):
            s += "\ndata"
            if self.ndims == 2:
                s += "[:, :]"
                s += " =\n"
                s += str(self.data)
            elif self.ndims > 2:
                idx = tt_ind2sub(self.shape[:-2], np.array([i]))
                s += str(idx[0].tolist())[0:-1]
                s += ", :, :]"
                s += " =\n"
                s += str(
                    self.data[
                        tuple(
                            np.concatenate(
                                (idx[0], np.array([slice(None), slice(None)]))
                            )
                        )
                    ]
                )
        # s += '\n'
        return s

    __str__ = __repr__


def tenones(shape: Tuple[int, ...]) -> tensor:
    """
    Creates a tensor of all ones.

    Parameters
    ----------
    shape:
        Shape of resulting tensor.

    Returns
    -------
    Constructed tensor.

    Examples
    --------
    >>> T = ttb.tenones((3,))
    >>> T
    tensor of shape (3,)
    data[:] =
    [1. 1. 1.]
    >>> T = ttb.tenones((3,3))
    >>> T
    tensor of shape (3, 3)
    data[:, :] =
    [[1. 1. 1.]
     [1. 1. 1.]
     [1. 1. 1.]]
    """
    return tensor.from_function(np.ones, shape)


def tenzeros(shape: Tuple[int, ...]) -> tensor:
    """
    Creates a tensor of all zeros.

    Parameters
    ----------
    shape:
        Shape of resulting tensor.

    Returns
    -------
    Constructed tensor.

    Examples
    --------
    >>> T = ttb.tenzeros((3,))
    >>> T
    tensor of shape (3,)
    data[:] =
    [0. 0. 0.]
    >>> T = ttb.tenzeros((3,3))
    >>> T
    tensor of shape (3, 3)
    data[:, :] =
    [[0. 0. 0.]
     [0. 0. 0.]
     [0. 0. 0.]]
    """
    return tensor.from_function(np.zeros, shape)


def tenrand(shape: Tuple[int, ...]) -> tensor:
    """
    Creates a tensor with entries drawn from a uniform
    distribution on the unit interval.

    Parameters
    ----------
    shape:
        Shape of resulting tensor.

    Returns
    -------
    Constructed tensor.

    Examples
    --------
    >>> np.random.seed(1)
    >>> T = ttb.tenrand((3,))
    >>> T
    tensor of shape (3,)
    data[:] =
    [4.170...e-01 7.203...e-01 1.143...e-04]
    """

    # Typing doesn't play nice with partial
    # mypy issue: 1484
    def unit_uniform(pass_through_shape: Tuple[int, ...]) -> np.ndarray:
        return np.random.uniform(low=0, high=1, size=pass_through_shape)

    return tensor.from_function(unit_uniform, shape)


def tendiag(elements: np.ndarray, shape: Optional[Tuple[int, ...]] = None) -> tensor:
    """
    Creates a tensor with elements along super diagonal. If provided shape is too
    small the tensor will be enlarged to accomodate.

    Parameters
    ----------
    elements:
        Elements to set along the diagonal.
    shape:
        Shape of resulting tensor.

    Returns
    -------
    Constructed tensor.

    Examples
    --------
    >>> shape = (3,)
    >>> values = np.ones(shape)
    >>> T1 = ttb.tendiag(values)
    >>> T2 = ttb.tendiag(values, (3, 3, 3))
    >>> T1.isequal(T2)
    True
    """
    # Flatten provided elements
    elements = np.ravel(elements)
    N = len(elements)
    if shape is None:
        constructed_shape = (N,) * N
    else:
        constructed_shape = tuple(max(N, dim) for dim in shape)
    X = tenzeros(constructed_shape)
    subs = np.tile(np.arange(0, N)[:, None], (len(constructed_shape),))
    X[subs] = elements
    return X


def teneye(order: int, size: int) -> tensor:
    """Create identity tensor of specified shape.

    T is an "identity tensor if T.ttsv(x, skip_dim=0) = x for all x such that
    norm(x) == 1.

    An identity tensor only exists if order is even.
    This method is resource intensive
    for even moderate orders or sizes (>=6).

    Parameters
    ----------
    order: Number of dimensions of tensor.
    size: Number of elements in any dimension of the tensor.

    Examples
    --------
    >>> ttb.teneye(2, 3)
    tensor of shape (3, 3)
    data[:, :] =
    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]
    >>> x = np.ones((5,))
    >>> x /= np.linalg.norm(x)
    >>> T = ttb.teneye(4, 5)
    >>> np.allclose(T.ttsv(x, 0), x)
    True

    Returns
    -------
    Identity tensor.
    """
    if order % 2 != 0:
        raise ValueError(f"Order must be even but received {order}")
    idx_iterator = combinations_with_replacement(range(size), order)
    A = tenzeros((size,) * order)
    s = np.zeros((factorial(order), order // 2))
    for _i, indices in enumerate(idx_iterator):
        p = np.array(list(permutations(indices)))
        for j in range(order // 2):
            s[:, j] = p[:, 2 * j - 1] == p[:, 2 * j]
        v = np.sum(np.sum(s, axis=1) == order // 2)
        A[tuple(zip(*p))] = v / factorial(order)
    return A


def mttv_left(W_in: np.ndarray, U1: np.ndarray) -> np.ndarray:
    """
    Contract leading mode in partial MTTKRP W_in using factor matrix U1.
    The leading mode is the mode for which consecutive increases in index address
    elements at consecutive increases in the memory offset.

    Parameters
    ----------
    W_in:
        Has modes in descending order: (m1 x m2 x ... x mN, C). The final mode C is the
        component mode corresponding to the columns in factor matrices.
    U1:
        Factor matrix with modes (m1, C).

    Returns
    -------
        Matrix with modes (m2 x ... x mN, C).
    """
    r = U1.shape[1]
    W_in = np.reshape(W_in, (U1.shape[0], -1, r), order="F")
    W_out = np.zeros_like(W_in, shape=(W_in.shape[1], r))
    # TODO this can be replaced with tensordot and slice,
    #  even better if we can skip slice
    #  W_out = np.dot(W_in.transpose(), U1)[range(r), :, range(r)].transpose()
    for j in range(r):
        W_out[:, j] = W_in[:, :, j].transpose().dot(U1[:, j])
    return W_out


def mttv_mid(W_in: np.ndarray, U_mid: List[np.ndarray]) -> np.ndarray:
    """
    Contract intermediate modes in partial MTTKRP W_in using factor matrices U_mid.

    Parameters
    ----------
    W_in:
        Has modes in descending order: (m1 x m2 x ... x mN, C). The final mode C is the
        component mode corresponding to the columns in factor matrices.
    U_mid:
        Factor matrices with modes (m2, C), (m3, C), ..., (mN, C).

    Returns
    -------
        Matrix with modes (m1, C).
    """
    if len(U_mid) == 0:
        return W_in
    K = ttb.khatrirao(*U_mid, reverse=True)
    r = K.shape[1]
    W_in = np.reshape(W_in, (-1, K.shape[0], r), order="F")
    V = np.zeros_like(W_in, shape=(W_in.shape[0], r))
    for j in range(r):
        V[:, j] = W_in[:, :, j].dot(K[:, j])
    return V


def min_split(shape: Tuple[int, ...]) -> int:
    """Scan for optimal splitting with minimal memory footprint.

    Parameters
    ----------
    shape:
        Shape of original tensor in natural descending order.

    Returns
    -------
        Optimal splitting to minimize partial MTTKRP memory footprint.
        Modes 0:split will contract in left-partial computation and the
        rest will contract in right-partial.
    """
    m_left = shape[0]
    m_right = np.prod(shape[1:])
    idx_min = 0

    # Minimize m_left + m_right
    for idx, s in enumerate(shape[1:], 1):
        # Peel mode idx off right and test placement.
        m_right = m_right // s
        if m_left < m_right:
            # Sum is reduced by placing mode idx on left
            idx_min = idx
            m_left *= s
        else:
            # The sum would be reduced by placing mode s back on the right.
            # Stop collecting modes on the left.
            break
    return idx_min


if __name__ == "__main__":
    import doctest  # pragma: no cover

    doctest.testmod()  # pragma: no cover
