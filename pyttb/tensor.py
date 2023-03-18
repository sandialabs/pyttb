# Copyright 2022 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.
"""Dense Tensor Implementation"""
from __future__ import annotations

import warnings
from itertools import permutations
from math import factorial
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse.linalg
from numpy_groupies import aggregate as accumarray

import pyttb as ttb
from pyttb.pyttb_utils import tt_dimscheck, tt_ind2sub


class tensor:
    """
    TENSOR Class for dense tensors.
    """

    data: np.ndarray
    shape: Tuple

    def __init__(self):
        """
        TENSOR Create empty tensor.
        """

        # EMPTY / DEFAULT CONSTRUCTOR
        self.data = np.array([])
        self.shape = ()

    @classmethod
    def from_data(
        cls, data: np.ndarray, shape: Optional[Tuple[int, ...]] = None
    ) -> tensor:
        """
        Creates a tensor from explicit description. Note that 1D tensors (i.e.,
        when len(shape)==1) contains a data array that follow the Numpy convention
        of being a row vector, which is different than in the Matlab Tensor Toolbox.

        Parameters
        ----------
        data: Tensor source data
        shape: Shape of resulting tensor if not the same as data shape

        Returns
        -------
        Constructed tensor

        Example
        -------
        >>> X = ttb.tensor.from_data(np.ones((2,2)))
        >>> Y = ttb.tensor.from_data(np.ones((2,2)), shape=(4,1))
        """
        # CONVERT A MULTIDIMENSIONAL ARRAY
        if not issubclass(data.dtype.type, np.number) and not issubclass(
            data.dtype.type, np.bool_
        ):
            assert False, "First argument must be a multidimensional array."

        # Create or check second argument
        if shape is None:
            shape = data.shape
        else:
            if not isinstance(shape, tuple):
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
        tensorInstance = cls()
        tensorInstance.data = data.copy()
        tensorInstance.shape = shape
        return tensorInstance

    @classmethod
    def from_tensor_type(
        cls, source: Union[ttb.sptensor, tensor, ttb.ktensor, ttb.tenmat]
    ) -> tensor:
        """
        Converts other tensor types into a dense tensor

        Parameters
        ----------
        source: Tensor type to create dense tensor from

        Returns
        -------
        Constructed tensor

        Example
        -------
        >>> X = ttb.tensor.from_data(np.ones((2,2)))
        >>> Y = ttb.tensor.from_tensor_type(X)
        """
        # CONVERSION/COPY CONSTRUCTORS
        if isinstance(source, tensor):
            # COPY CONSTRUCTOR
            return cls.from_data(source.data.copy(), source.shape)
        if isinstance(
            source,
            (
                ttb.ktensor,
                ttb.ttensor,
                ttb.sptensor,
                ttb.sumtensor,
                ttb.symtensor,
                ttb.symktensor,
            ),
        ):
            # CONVERSION
            t = source.full()
            return cls.from_data(t.data.copy(), t.shape)
        if isinstance(source, ttb.tenmat):
            # RESHAPE TENSOR-AS-MATRIX
            # Here we just reverse what was done in the tenmat constructor.
            # First we reshape the data to be an MDA, then we un-permute
            # it using ipermute.
            shape = source.tshape
            order = np.hstack([source.rindices, source.cindices])
            data = np.reshape(source.data.copy(), np.array(shape)[order], order="F")
            if order.size > 1:
                data = np.transpose(data, np.argsort(order))
            return cls.from_data(data, shape)
        raise ValueError(f"Unsupported type for tensor source, received {type(source)}")

    @classmethod
    def from_function(
        cls,
        function_handle: Callable[[Tuple[int, ...]], np.ndarray],
        shape: Tuple[int, ...],
    ) -> tensor:
        """
        Creates a tensor from a function handle and size

        Parameters
        ----------
        function_handle: Function to generate data to construct tensor
        shape: Shape of resulting tensor

        Returns
        -------
        Constructed tensor

        Example
        -------
        >>> X = ttb.tensor.from_function(lambda a_shape: np.ones(a_shape), (2,2))
        """
        # FUNCTION HANDLE AND SIZE

        # Check size
        if not isinstance(shape, tuple):
            assert False, "TTB:BadInput, Shape must be a tuple"

        # Generate data
        data = function_handle(shape)

        # Create the tensor
        return cls.from_data(data, shape)

    def collapse(
        self,
        dims: Optional[np.ndarray] = None,
        fun: Callable[[np.ndarray], Union[float, np.ndarray]] = np.sum,
    ) -> Union[float, np.ndarray, tensor]:
        """
        Collapse tensor along specified dimensions.

        Parameters
        ----------
        dims: Dimensions to collapse
        fun: Method used to collapse dimensions

        Returns
        -------
        Collapsed value

        Example
        -------
        >>> X = ttb.tensor.from_data(np.ones((2,2)))
        >>> X.collapse()
        4.0
        >>> X.collapse(np.arange(X.ndims), sum)
        4.0
        """
        if self.data.size == 0:
            return np.array([])

        if dims is None:
            dims = np.arange(0, self.ndims)

        if dims.size == 0:
            return ttb.tensor.from_tensor_type(self)

        dims, _ = tt_dimscheck(dims, self.ndims)
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
        return ttb.tensor.from_data(B, newshape)

    def contract(self, i: int, j: int) -> Union[np.ndarray, tensor]:
        """
        Contract tensor along two dimensions (array trace).

        Parameters
        ----------
        i: First dimension
        j: Second dimension

        Returns
        -------
        Contracted tensor

        Example
        -------
        >>> X = ttb.tensor.from_data(np.ones((2,2)))
        >>> X.contract(0, 1)
        2.0
        """
        if self.shape[i] != self.shape[j]:
            assert False, "Must contract along equally sized dimensions"

        if i == j:
            assert False, "Must contract along two different dimensions"

        # Easy case - returns a scalar
        if self.ndims == 2:
            return np.trace(self.data)

        # Remaining dimensions after trace
        remdims = np.setdiff1d(np.arange(0, self.ndims), np.array([i, j])).astype(int)

        # Size for return
        newsize = tuple(np.array(self.shape)[remdims])

        # Total size of remainder
        m = np.prod(newsize)

        # Number of items to add for trace
        n = self.shape[i]

        # Permute trace dimensions to the end
        x = self.permute(np.concatenate((remdims, np.array([i, j]))))

        # Reshape data to be 3D
        data = np.reshape(x.data, (m, n, n), order="F")

        # Add diagonal entries for each slice
        newdata = np.zeros((m, 1))
        for idx in range(0, n):
            newdata += data[:, idx, idx][:, None]

        # Reshape result
        if np.prod(newsize) > 1:
            newdata = np.reshape(newdata, newsize, order="F")

        return ttb.tensor.from_data(newdata, newsize)

    def double(self) -> np.ndarray:
        """
        Convert tensor to an array of doubles

        Returns
        -------
        Copy of tensor data

        Example
        -------
        >>> X = ttb.tensor.from_data(np.ones((2,2)))
        >>> X.double()
        array([[1., 1.],
               [1., 1.]])
        """
        return self.data.astype(np.float_).copy()

    def exp(self) -> tensor:
        """
        Exponential of the elements of tensor

        Returns
        -------
        Copy of tensor data element-wise raised to exponential

        Examples
        --------
        >>> tensor1 = ttb.tensor.from_data(np.array([[1, 2], [3, 4]]))
        >>> tensor1.exp().data  # doctest: +ELLIPSIS
        array([[ 2.7182...,  7.3890... ],
               [20.0855..., 54.5981...]])
        """
        return ttb.tensor.from_data(np.exp(self.data))

    def end(self, k: Optional[int] = None) -> int:
        """
        Last index of indexing expression for tensor

        Parameters
        ----------
        k: dimension for subscripted indexing

        Examples
        --------
        >>> X = ttb.tensor.from_data(np.ones((2,2)))
        >>> X.end()  # linear indexing
        3
        >>> X.end(0)
        1
        """

        if k is not None:  # Subscripted indexing
            return self.shape[k] - 1
        # For linear indexing
        return np.prod(self.shape) - 1

    def find(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        FIND Find subscripts of nonzero elements in a tensor.

        S, V = FIND(X) returns the subscripts of the nonzero values in X and a column
        vector of the values.

        Examples
        --------
        >>> X = ttb.tensor.from_data(np.zeros((3,4,2)))
        >>> larger_entries = X > 0.5
        >>> subs, vals = larger_entries.find()

        See Also
        --------
        TENSOR/SUBSREF, TENSOR/SUBSASGN

        Returns
        -------
        Subscripts and values for non-zero entries
        """
        idx = np.nonzero(np.ravel(self.data, order="F"))[0]
        subs = ttb.tt_ind2sub(self.shape, idx)
        vals = self.data[tuple(subs.T)][:, None]
        return subs, vals

    def full(self) -> tensor:
        """
        Convert dense tensor to dense tensor.

        Returns
        -------
        Deep copy
        """
        return ttb.tensor.from_data(self.data)

    def innerprod(self, other: Union[tensor, ttb.sptensor, ttb.ktensor]) -> float:
        """
        Efficient inner product with a tensor

        Parameters
        ----------
        other: Tensor type to take an innerproduct with

        Examples
        --------
        >>> tensor1 = ttb.tensor.from_data(np.array([[1, 2], [3, 4]]))
        >>> tensor1.innerprod(tensor1)
        30
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
        Exact equality for tensors

        Parameters
        ----------
        other: Tensor to compare against

        Examples
        --------
        >>> X = ttb.tensor.from_data(np.ones((2,2)))
        >>> Y = ttb.tensor.from_data(np.zeros((2,2)))
        >>> X.isequal(Y)
        False
        """
        if isinstance(other, ttb.tensor):
            return bool(np.all(self.data == other.data))
        if isinstance(other, ttb.sptensor):
            return bool(np.all(self.data == other.full().data))
        return False

    # TODO: We should probably always return details and let caller drop them
    # pylint: disable=too-many-branches, too-many-locals
    def issymmetric(
        self,
        grps: Optional[np.ndarray] = None,
        version: Optional[Any] = None,
        return_details: bool = False,
    ) -> Union[bool, Tuple[bool, np.ndarray, np.ndarray]]:
        """
        Determine if a dense tensor is symmetric in specified modes.

        Parameters
        ----------
        grps: Modes to check for symmetry
        version: Flag
            Any non-None value will call the non-default old version
        return_details: Flag to return symmetry details in addition to bool

        Returns
        -------
        If symmetric in modes, optionally all differences and permutations

        Examples
        --------
        >>> X = ttb.tensor.from_data(np.ones((2,2)))
        >>> X.issymmetric()
        True
        >>> X.issymmetric(grps=np.arange(X.ndims))
        True
        >>> is_sym, diffs, perms = \
            X.issymmetric(grps=np.arange(X.ndims), version=1, return_details=True)
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
        if version is None:  # pylint:disable=no-else-return
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

    def logical_and(self, B: Union[float, tensor]) -> tensor:
        """
        Logical and for tensors

        Parameters
        ----------
        B: Value to and against self

        Examples
        --------
        >>> X = ttb.tensor.from_data(np.ones((2,2), dtype=bool))
        >>> X.logical_and(X).collapse()  # All true
        4
        """

        def logical_and(x, y):
            return np.logical_and(x, y)

        return ttb.tt_tenfun(logical_and, self, B)

    def logical_not(self) -> tensor:
        """
        Logical Not For Tensors

        Returns
        -------
        Negated tensor

        Examples
        --------
        >>> X = ttb.tensor.from_data(np.ones((2,2), dtype=bool))
        >>> X.logical_not().collapse()  # All false
        0
        """
        return ttb.tensor.from_data(np.logical_not(self.data))

    def logical_or(self, other: Union[float, tensor]) -> tensor:
        """
        Logical or for tensors

        Parameters
        ----------
        other: Value to perform or against

        Examples
        --------
        >>> X = ttb.tensor.from_data(np.ones((2,2), dtype=bool))
        >>> X.logical_or(X.logical_not()).collapse()  # All true
        4
        """

        def tensor_or(x, y):
            return np.logical_or(x, y)

        return ttb.tt_tenfun(tensor_or, self, other)

    def logical_xor(self, other: Union[float, tensor]) -> tensor:
        """
        Logical xor for tensors

        Parameters
        ----------
        other: Value to perform xor against

        Examples
        --------
        >>> X = ttb.tensor.from_data(np.ones((2,2), dtype=bool))
        >>> X.logical_xor(X.logical_not()).collapse()  # All true
        4
        """

        def tensor_xor(x, y):
            return np.logical_xor(x, y)

        return ttb.tt_tenfun(tensor_xor, self, other)

    def mask(self, W: tensor) -> np.ndarray:
        """
        Extract non-zero values at locations specified by mask tensor

        Parameters
        ----------
        W: Mask tensor

        Returns
        -------
        Extracted values

        Examples
        --------
        >>> W = ttb.tensor.from_data(np.ones((2,2)))
        >>> tensor1 = ttb.tensor.from_data(np.array([[1, 2], [3, 4]]))
        >>> tensor1.mask(W)
        array([1, 3, 2, 4])
        """
        # Error checking
        if np.any(np.array(W.shape) > np.array(self.shape)):
            assert False, "Mask cannot be bigger than the data tensor"

        # Extract locations of nonzeros in W
        wsubs, _ = W.find()

        # Extract those non-zero values
        return self.data[tuple(wsubs.transpose())]

    # pylint: disable=too-many-branches
    def mttkrp(self, U: Union[ttb.ktensor, List[np.ndarray]], n: int) -> np.ndarray:
        """
        Matricized tensor times Khatri-Rao product

        Parameters
        ----------
        U: Matrices to create the Khatri-Rao product
        n: Mode to matricize tensor in

        Returns
        -------
        Matrix product

        Example
        -------
        >>> tensor1 = ttb.tensor.from_data(np.ones((2,2,2)))
        >>> matrices = [np.ones((2,2))] * 3
        >>> tensor1.mttkrp(matrices, 2)
        array([[4., 4.],
               [4., 4.]])
        """

        # check that we have a tensor that can perform mttkrp
        if self.ndims < 2:
            assert False, "MTTKRP is invalid for tensors with fewer than 2 dimensions"

        # extract the list of factor matrices if given a ktensor
        if isinstance(U, ttb.ktensor):
            U = ttb.ktensor.from_tensor_type(U)
            if n == 0:
                U.redistribute(1)
            else:
                U.redistribute(0)
            # Extract the factor matrices
            U = U.factor_matrices

        # check that we have a list (or list extracted from a ktensor)
        if not isinstance(U, list):
            assert False, "Second argument should be a list of arrays or a ktensor"

        # check that list is the correct length
        if len(U) != self.ndims:
            assert False, "Second argument contains the wrong number of arrays"

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
            Ur = ttb.khatrirao(U[1 : self.ndims], reverse=True)
            Y = np.reshape(self.data, (szn, szr), order="F")
            return Y @ Ur
        if n == self.ndims - 1:  # pylint: disable=no-else-return
            Ul = ttb.khatrirao(U[0 : self.ndims - 1], reverse=True)
            Y = np.reshape(self.data, (szl, szn), order="F")
            return Y.T @ Ul
        else:
            Ul = ttb.khatrirao(U[n + 1 :], reverse=True)
            Ur = np.reshape(ttb.khatrirao(U[0:n], reverse=True), (szl, 1, R), order="F")
            Y = np.reshape(self.data, (-1, szr), order="F")
            Y = Y @ Ul
            Y = np.reshape(Y, (szl, szn, R), order="F")
            V = np.zeros((szn, R))
            for r in range(R):
                V[:, [r]] = Y[:, :, r].T @ Ur[:, :, r]
            return V

    @property
    def ndims(self) -> int:
        """
        Return the number of dimensions of a tensor

        Examples
        --------
        >>> X = ttb.tensor.from_data(np.ones((2,2)))
        >>> X.ndims
        2
        """
        if self.shape == (0,):
            return 0
        return len(self.shape)

    @property
    def nnz(self) -> int:
        """
        Number of non-zero elements in tensor

        Examples
        --------
        >>> X = ttb.tensor.from_data(np.ones((2,2)))
        >>> X.nnz
        4
        """
        return np.count_nonzero(self.data)

    def norm(self) -> np.floating:
        """
        Frobenius Norm of Tensor

        Examples
        --------
        >>> X = ttb.tensor.from_data(np.ones((2,2)))
        >>> X.norm()
        2.0
        """
        # default of np.linalg.norm is to vectorize the data and compute the vector
        # norm, which is equivalent to the Frobenius norm for multidimensional arrays.
        # However, the argument 'fro' only works for 1-D and 2-D arrays currently.
        return np.linalg.norm(self.data)

    def nvecs(self, n: int, r: int, flipsign: bool = True) -> np.ndarray:
        """
        Compute the leading mode-n eigenvectors for a tensor

        Parameters
        ----------
        n: Mode to unfold
        r: Number of eigenvectors to compute
        flipsign: Make each eigenvector's largest element positive

        Examples
        --------
        >>> tensor1 = ttb.tensor.from_data(np.array([[1, 2], [3, 4]]))
        >>> tensor1.nvecs(0,1)  # doctest: +ELLIPSIS
        array([[0.4045...],
               [0.9145...]])
        >>> tensor1.nvecs(0,2)  # doctest: +ELLIPSIS
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
            warnings.warn(
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
        Permute tensor dimensions.

        Parameters
        ----------
        order: New order of tensor dimensions

        Returns
        -------
        Updated tensor with shapeNew == shapePrevious[order]

        Examples
        --------
        >>> X = ttb.tensor.from_data(np.ones((2,2)))
        >>> Y = X.permute(np.array((1,0)))
        >>> X.isequal(Y)
        True
        """
        if self.ndims != order.size:
            assert False, "Invalid permutation order"

        # If order is empty, return
        if order.size == 0:
            return ttb.tensor.from_tensor_type(self)

        # Check for special case of an order-1 object, has no effect
        if (order == 1).all():
            return ttb.tensor.from_tensor_type(self)

        # Np transpose does error checking on order, acts as permutation
        return ttb.tensor.from_data(np.transpose(self.data, order))

    def reshape(self, shape: Tuple[int, ...]) -> tensor:
        """
        Reshapes a tensor

        Parameters
        ----------
        shape: New shape

        Examples
        --------
        >>> X = ttb.tensor.from_data(np.ones((2,2)))
        >>> Y = X.reshape((4,1))
        >>> Y.shape
        (4, 1)
        """

        if np.prod(self.shape) != np.prod(shape):
            assert False, "Reshaping a tensor cannot change number of elements"

        return ttb.tensor.from_data(np.reshape(self.data, shape, order="F"), shape)

    def squeeze(self) -> Union[tensor, np.ndarray, float]:
        """
        Removes singleton dimensions from a tensor

        Returns
        -------
        Tensor or scalar if all dims squeezed

        Examples
        --------
        >>> tensor1 = ttb.tensor.from_data(np.array([[[4]]]))
        >>> tensor1.squeeze()
        4
        >>> tensor2 = ttb.tensor.from_data(np.array([[1, 2, 3]]))
        >>> tensor2.squeeze().data
        array([1, 2, 3])

        """
        shapeArray = np.array(self.shape)
        if np.all(shapeArray > 1):  # pylint: disable=no-else-return
            return ttb.tensor.from_tensor_type(self)
        else:
            idx = np.where(shapeArray > 1)
            if idx[0].size == 0:
                return np.squeeze(self.data)[()]
            return ttb.tensor.from_data(np.squeeze(self.data))

    def symmetrize(
        self, grps: Optional[np.ndarray] = None, version: Optional[Any] = None
    ) -> (
        tensor
    ):  # pylint: disable=too-many-branches, too-many-statements, too-many-locals
        """
        Symmetrize a tensor in the specified modes
        Notes
        -----
        It is *the same or less* work to just call X = symmetrize(X) then to first
        check if X is symmetric and then symmetrize it, even if X is already symmetric.

        Parameters
        ----------
        grps: Modes to check for symmetry
        version: Any non-None value will call the non-default old version
        Returns
        -------
        """
        n = self.ndims
        sz = np.array(self.shape)

        if grps is None:
            grps = np.arange(0, n)

        if len(grps.shape) == 1:
            grps = np.array([grps])

        data = self.data.copy()

        # Use default newer faster version
        if version is None:  # pylint: disable=no-else-return
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
                idx = ttb.tt_ind2sub(self.shape, np.arange(0, data.size))

                # Find reference index for every element in the tensor - this
                # is to its index in the symmetrized tensor. This puts every
                # element into a 'class' of entries that will be the same under
                # symmetry.

                classidx = idx
                classidx[:, thisgrp] = np.sort(idx[:, thisgrp], axis=1)
                linclassidx = ttb.tt_sub2ind(self.shape, classidx)

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

            return ttb.tensor.from_data(data)

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
                for j in range(0, ntimes):
                    for k in range(0, nelems):
                        for _ in range(0, ncopies):
                            # TODO: Does this do anything? Matches MATLAB
                            # at very least should be able to flatten
                            sym_perms[perm_idx, grps[i]] = combos[i][k, :]
                            perm_idx += 1

            # Create an average tensor
            Y = ttb.tensor.from_data(np.zeros(self.shape))
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
        transpose: bool = False,
    ) -> tensor:
        """
        Tensor times matrix

        Parameters
        ----------
        matrix: Matrix or matrices to multiple by
        dims: Dimensions to multiply against
        transpose: Transpose matrices during multiplication
        """

        if dims is None:
            dims = np.arange(self.ndims)
        elif isinstance(dims, list):
            dims = np.array(dims)
        elif isinstance(dims, (float, int, np.generic)):
            dims = np.array([dims])

        if isinstance(matrix, list):
            # Check that the dimensions are valid
            dims, vidx = ttb.tt_dimscheck(dims, self.ndims, len(matrix))

            # Calculate individual products
            Y = self.ttm(matrix[vidx[0]], dims[0], transpose)
            for k in range(1, dims.size):
                Y = Y.ttm(matrix[vidx[k]], dims[k], transpose)
            return Y

        if not isinstance(matrix, np.ndarray):
            assert False, f"matrix must be of type numpy.ndarray but got:\n{matrix}"

        if not (dims.size == 1 and np.isin(dims, np.arange(self.ndims))):
            assert False, "dims must contain values in [0,self.dims)"

        # old version (ver=0)
        shape = np.array(self.shape)
        n = dims[0]
        order = np.array([n] + list(range(0, n)) + list(range(n + 1, self.ndims)))
        newdata = self.permute(order).data
        ids = np.array(list(range(0, n)) + list(range(n + 1, self.ndims)))
        newdata = np.reshape(newdata, (shape[n], np.prod(shape[ids])), order="F")
        if transpose:
            newdata = matrix.T @ newdata
            p = matrix.shape[1]
        else:
            newdata = matrix @ newdata
            p = matrix.shape[0]

        newshape = np.array(
            [p] + list(shape[range(0, n)]) + list(shape[range(n + 1, self.ndims)])
        )
        Y_data = np.reshape(newdata, newshape, order="F")
        Y_data = np.transpose(Y_data, np.argsort(order))
        return ttb.tensor.from_data(Y_data)

    def ttt(
        self,
        other: tensor,
        selfdims: Optional[Union[int, np.ndarray]] = None,
        otherdims: Optional[Union[int, np.ndarray]] = None,
    ) -> tensor:
        """
        Tensor multiplication (tensor times tensor)

        Parameters
        ----------
        other: Tensor to multiply by
        selfdims: Dimensions to contract this tensor by for multiplication
        otherdims: Dimensions to contract other tensor by for multiplication
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
        print(amatrix)
        print(bmatrix)
        cmatrix = amatrix * bmatrix

        # Check whether or not the result is a scalar
        if isinstance(cmatrix, ttb.tenmat):
            return ttb.tensor.from_tensor_type(cmatrix)
        return cmatrix

    def ttv(
        self,
        vector: Union[np.ndarray, List[np.ndarray]],
        dims: Optional[Union[int, np.ndarray]] = None,
    ) -> tensor:
        """
        Tensor times vector

        Parameters
        ----------
        vector: Vector(s) to multiply against
        dims: Dimensions to multiply with vector(s)
        """

        if dims is None:
            dims = np.array([])
        elif isinstance(dims, (float, int)):
            dims = np.array([dims])

        # Check that vector is a list of vectors, if not place single vector as element
        # in list
        if len(vector) > 0 and isinstance(vector[0], (int, float, np.int_, np.float_)):
            return self.ttv(np.array([vector]), dims)

        # Get sorted dims and index for multiplicands
        dims, vidx = ttb.tt_dimscheck(dims, self.ndims, len(vector))

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
            return ttb.tensor.from_data(c, tuple(sz[0:n]))
        return c[0]

    def ttsv(
        self,
        vector: Union[np.ndarray, List[np.ndarray]],
        dims: Optional[Union[int, np.ndarray]] = None,
        version: Optional[int] = None,
    ) -> Union[np.ndarray, tensor]:
        """
        Tensor times same vector in multiple modes

        Parameters
        ----------
        vector: Vector(s) to multiply against
        dims: Dimensions to multiply by vector(s)
        """
        # Only two simple cases are supported
        if dims is None:
            dims = 0
        elif dims > 0:
            assert False, "Invalid modes in ttsv"

        if version == 1:  # Calculate the old way
            P = self.ndims
            X = np.array([vector for i in range(P)])
            # pylint: disable=invalid-unary-operand-type
            if dims == 0:
                return self.ttv(X)
            if dims in (-1, -2):  # Return scalar or matrix
                return (self.ttv(X, -np.arange(1, -dims + 1))).double()
            return self.ttv(X, -np.arange(1, -dims + 1))

        if version == 2 or version is None:  # Calculate the new way
            d = self.ndims
            sz = self.shape[0]  # Sizes of all modes must be the same

            # pylint: disable=invalid-unary-operand-type
            dnew = -dims  # Number of modes in result
            drem = d - dnew  # Number of modes multiplied out

            y = self.data
            for i in range(drem, 0, -1):
                yy = np.reshape(y, (sz ** (dnew + i - 1), sz), order="F")
                y = yy.dot(vector)

            # Convert to matrix if 2-way or convert back to tensor if result is >= 3-way
            if dnew == 2:
                return np.reshape(y, [sz, sz], order="F")
            if dnew > 2:
                return ttb.tensor.from_data(
                    np.reshape(y, newshape=sz * np.ones(dnew, dtype=int), order="F")
                )
            return y
        assert False, "Invalid value for version; should be None, 1, or 2"

    def __setitem__(
        self, key, value
    ):  # pylint: disable=too-many-branches, too-many-statements
        """
        SUBSASGN Subscripted assignment for a tensor.

        We can assign elements to a tensor in three ways.

        Case 1: X(R1,R2,...,RN) = Y, in which case we replace the
        rectangular subtensor (or single element) specified by the ranges
        R1,...,RN with Y. The right-hand-side can be a scalar, a tensor, or an
        MDA.

        Case 2a: X(S) = V, where S is a p x n array of subscripts and V is
        a scalar or a vector containing p values.

        Case 2b: X(I) = V, where I is a set of p linear indices and V is a
        scalar or a vector containing p values. Resize is not allowed in this
        case.

        Examples
        X = tensor(rand(3,4,2))
        X(1:2,1:2,1) = ones(2,2) <-- replaces subtensor
        X([1 1 1;1 1 2]) = [5;7] <-- replaces two elements
        X([1;13]) = [5;7] <-- does the same thing
        X(1,1,2:3) = 1 <-- grows tensor
        X(1,1,4) = 1 %<- grows the size of the tensor
        """
        # Figure out if we are doing a subtensor, a list of subscripts or a list of
        # linear indices
        access_type = "error"
        if self.ndims <= 1:
            if isinstance(key, np.ndarray):
                access_type = "subscripts"
            else:
                access_type = "subtensor"
        else:
            if isinstance(key, np.ndarray):
                if len(key.shape) > 1 and key.shape[1] >= self.ndims:
                    access_type = "subscripts"
                elif len(key.shape) == 1 or key.shape[1] == 1:
                    access_type = "linear indices"
            elif isinstance(key, tuple):
                validSubtensor = [
                    isinstance(keyElement, (int, slice)) for keyElement in key
                ]
                if np.all(validSubtensor):
                    access_type = "subtensor"

        # Case 1: Rectangular Subtensor
        if access_type == "subtensor":
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
                        sliceCheck.append(element.stop)
                else:
                    sliceCheck.append(element)
            bsiz = np.array(sliceCheck)
            if n == 0:
                newsiz = (bsiz[n:] + 1).astype(int)
            else:
                newsiz = np.concatenate(
                    (np.max((self.shape, bsiz[0:n] + 1), axis=0), bsiz[n:] + 1)
                ).astype(int)
            if (newsiz != self.shape).any():
                # We need to enlarge x.data.
                newData = np.zeros(shape=tuple(newsiz))
                idx = [slice(None, currentShape) for currentShape in self.shape]
                if self.data.size > 0:
                    newData[tuple(idx)] = self.data
                self.data = newData

                self.shape = tuple(newsiz)
            if isinstance(value, ttb.tensor):
                self.data[key] = value.data
            else:
                self.data[key] = value

            return

        # Case 2a: Subscript indexing
        if access_type == "subscripts":
            # Extract array of subscripts
            subs = key

            # Will the size change? If so we first need to resize x
            n = self.ndims
            if (
                len(subs.shape) == 1
                and len(self.shape) == 1
                and self.shape[0] < subs.shape[0]
            ):
                bsiz = subs
            elif len(subs.shape) == 1:
                bsiz = np.array([np.max(subs, axis=0)])
                key = key.tolist()
            else:
                bsiz = np.array(np.max(subs, axis=0))
            if n == 0:
                newsiz = (bsiz[n:] + 1).astype(int)
            else:
                newsiz = np.concatenate(
                    (np.max((self.shape, bsiz[0:n] + 1), axis=0), bsiz[n:] + 1)
                ).astype(int)

            if (newsiz != self.shape).any():
                # We need to enlarge x.data.
                newData = np.zeros(shape=tuple(newsiz))
                idx = [slice(None, currentShape) for currentShape in self.shape]
                if self.data.size > 0:
                    newData[idx] = self.data
                self.data = newData

                self.shape = tuple(newsiz)

            # Finally we can copy in new data
            if isinstance(key, list):
                self.data[key] = value
            elif key.shape[0] == 1:  # and len(key.shape) == 1:
                self.data[tuple(key[0, :])] = value
            else:
                self.data[tuple(key)] = value
            return

        # Case 2b: Linear Indexing
        if access_type == "linear indices":
            idx = key
            if (idx > np.prod(self.shape)).any():
                assert (
                    False
                ), "TTB:BadIndex In assignment X[I] = Y, a tensor X cannot be resized"
            idx = tt_ind2sub(self.shape, idx)
            if idx.shape[0] == 1:
                self.data[tuple(idx[0, :])] = value
            else:
                actualIdx = tuple(idx.transpose())
                self.data[actualIdx] = value
            return

        assert False, "Invalid use of tensor setitem"

    def __getitem__(self, item):
        """
        SUBSREF Subscripted reference for tensors.

        We can extract elements or subtensors from a tensor in the
        following ways.

        Case 1a: y = X(i1,i2,...,iN), where each in is an index, returns a
        scalar.

        Case 1b: Y = X(R1,R2,...,RN), where one or more Rn is a range and
        the rest are indices, returns a sparse tensor.

        Case 2a: V = X(S) or V = X(S,'extract'), where S is a p x n array
        of subscripts, returns a vector of p values.

        Case 2b: V = X(I) or V = X(I,'extract'), where I is a set of p
        linear indices, returns a vector of p values.

        Any ambiguity results in executing the first valid case. This
        is particularly an issue if ndims(X)==1.

        Examples
        X = tensor(rand(3,4,2,1),[3 4 2 1]);
        X.data <-- returns multidimensional array
        X.size <-- returns size
        X(1,1,1,1) <-- produces a scalar
        X(1,1,1,:) <-- produces a tensor of order 1 and size 1
        X(:,1,1,:) <-- produces a tensor of size 3 x 1
        X(1:2,[2 4],1,:) <-- produces a tensor of size 2 x 2 x 1
        X(1:2,[2 4],1,1) <-- produces a tensor of size 2 x 2
        X([1,1,1,1;3,4,2,1]) <-- returns a vector of length 2
        X = tensor(rand(10,1),10);
        X([1:6]') <-- extracts a subtensor
        X([1:6]','extract') <-- extracts a vector of 6 elements

        Parameters
        ----------

        Returns
        -------
        :class:`pyttb.tensor` or :class:`numpy.ndarray`
        """
        # Case 1: Rectangular Subtensor
        if (
            isinstance(item, tuple)
            and len(item) == self.ndims
            and item[len(item) - 1] != "extract"
        ):
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
                elif not isinstance(a_region, int) and len(a_region) > 1:
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
                a = ttb.tensor.from_data(newdata)
            return a

        # *** CASE 2a: Subscript indexing ***
        if len(item) > 1 and isinstance(item[-1], str) and item[-1] == "extract":
            # Extract array of subscripts
            subs = np.array(item[0])
            a = np.squeeze(self.data[tuple(subs.transpose())])
            # TODO if is row make column?
            return ttb.tt_subsubsref(a, subs)

        # Case 2b: Linear Indexing
        if len(item) >= 2 and not isinstance(item[-1], str):
            assert False, "Linear indexing requires single input array"
        idx = item[0]
        a = np.squeeze(self.data[tuple(ttb.tt_ind2sub(self.shape, idx).transpose())])
        # Todo if row make column?
        return ttb.tt_subsubsref(a, idx)

    def __eq__(self, other):
        """
        Equal for tensors

        Parameters
        ----------
        other: :class:`pyttb.tensor`, float, int

        Returns
        -------
        :class:`pyttb.tensor`
        """

        def tensor_equality(x, y):
            return x == y

        return ttb.tt_tenfun(tensor_equality, self, other)

    def __ne__(self, other):
        """
        Not equal (!=) for tensors

        Parameters
        ----------
        other: :class:`pyttb.tensor`, float, int

        Returns
        -------
        :class:`pyttb.tensor`
        """

        def tensor_not_equal(x, y):
            return x != y

        return ttb.tt_tenfun(tensor_not_equal, self, other)

    def __ge__(self, other):
        """
        Greater than or equal (>=) for tensors

        Parameters
        ----------
        other: :class:`pyttb.tensor`, float, int

        Returns
        -------
        :class:`pyttb.tensor`
        """

        def greater_or_equal(x, y):
            return x >= y

        return ttb.tt_tenfun(greater_or_equal, self, other)

    def __le__(self, other):
        """
        Less than or equal (<=) for tensors

        Parameters
        ----------
        other: :class:`pyttb.tensor`, float, int

        Returns
        -------
        :class:`pyttb.tensor`
        """

        def less_or_equal(x, y):
            return x <= y

        return ttb.tt_tenfun(less_or_equal, self, other)

    def __gt__(self, other):
        """
        Greater than (>) for tensors

        Parameters
        ----------
        other: :class:`pyttb.tensor`, float, int

        Returns
        -------
        :class:`pyttb.tensor`
        """

        def greater(x, y):
            return x > y

        return ttb.tt_tenfun(greater, self, other)

    def __lt__(self, other):
        """
        Less than (<) for tensors

        Parameters
        ----------
        other: :class:`pyttb.tensor`, float, int

        Returns
        -------
        :class:`pyttb.tensor`
        """

        def less(x, y):
            return x < y

        return ttb.tt_tenfun(less, self, other)

    def __sub__(self, other):
        """
        Binary subtraction (-) for tensors

        Parameters
        ----------
        other: :class:`pyttb.tensor`, float, int

        Returns
        -------
        :class:`pyttb.tensor`
        """

        def minus(x, y):
            return x - y

        return ttb.tt_tenfun(minus, self, other)

    def __add__(self, other):
        """
        Binary addition (+) for tensors

        Parameters
        ----------
        other: :class:`pyttb.tensor`, float, int

        Returns
        -------
        :class:`pyttb.tensor`
        """

        # If rhs is sumtensor, treat as such
        if isinstance(other, ttb.sumtensor):  # pragma: no cover
            return other.__add__(self)

        def tensor_add(x, y):
            return x + y

        return ttb.tt_tenfun(tensor_add, self, other)

    def __radd__(self, other):
        """
        Right binary addition (+) for tensors

        Parameters
        ----------
        other: :class:`pyttb.tensor`, float, int

        Returns
        -------
        :class:`pyttb.tensor`
        """

        return self.__add__(other)

    def __pow__(self, power):
        """
        Element Wise Power (**) for tensors

        Parameters
        ----------
        other::class:`pyttb.tensor`, float, int

        Returns
        -------
        :class:`pyttb.tensor`
        """

        def tensor_pow(x, y):
            return x**y

        return ttb.tt_tenfun(tensor_pow, self, power)

    def __mul__(self, other):
        """
        Element wise multiplication (*) for tensors, self*other

        Parameters
        ----------
        other: :class:`pyttb.tensor`, float, int

        Returns
        -------
        :class:`pyttb.tensor`
        """

        def mul(x, y):
            return x * y

        if isinstance(other, (ttb.ktensor, ttb.sptensor, ttb.ttensor)):
            other = other.full()

        return ttb.tt_tenfun(mul, self, other)

    def __rmul__(self, other):
        """
        Element wise right multiplication (*) for tensors, other*self

        Parameters
        ----------
        other: :class:`pyttb.tensor`, float, int

        Returns
        -------
        :class:`pyttb.tensor`
        """
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        Element wise left division (/) for tensors, self/other

        Parameters
        ----------
        other: :class:`pyttb.tensor`, float, int

        Returns
        -------
        :class:`pyttb.tensor`
        """

        def div(x, y):
            # We ignore the divide by zero errors because np.inf/np.nan is an
            # appropriate representation
            with np.errstate(divide="ignore", invalid="ignore"):
                return x / y

        return ttb.tt_tenfun(div, self, other)

    def __rtruediv__(self, other):
        """
        Element wise right division (/) for tensors, other/self

        Parameters
        ----------
        other::class:`pyttb.tensor`, float, int

        Returns
        -------
        :class:`pyttb.tensor`
        """

        def div(x, y):
            # We ignore the divide by zero errors because np.inf/np.nan is an
            # appropriate representation
            with np.errstate(divide="ignore", invalid="ignore"):
                return x / y

        return ttb.tt_tenfun(div, other, self)

    def __pos__(self):
        """
        Unary plus (+) for tensors

        Returns
        -------
        :class:`pyttb.tensor`
            copy of tensor
        """

        return ttb.tensor.from_data(self.data)

    def __neg__(self):
        """
        Unary minus (-) for tensors

        Returns
        -------
        :class:`pyttb.tensor`
            copy of tensor
        """

        return ttb.tensor.from_data(-1 * self.data)

    def __repr__(self):
        """
        String representation of a tensor.

        Returns
        -------
        str
            Contains the shape and data as strings on different lines.
        """
        if self.ndims == 0:
            s = ""
            s += "empty tensor of shape "
            s += str(self.shape)
            s += "\n"
            s += "data = []"
            return s

        s = ""
        s += "tensor of shape "
        s += (" x ").join([str(int(d)) for d in self.shape])
        s += "\n"

        if self.ndims == 1:
            s += "data"
            if self.ndims == 1:
                s += "[:]"
                s += " = \n"
                s += str(self.data)
                s += "\n"
                return s
        for i in np.arange(np.prod(self.shape[:-2])):
            s += "data"
            if self.ndims == 2:
                s += "[:, :]"
                s += " = \n"
                s += str(self.data)
                s += "\n"
            elif self.ndims > 2:
                idx = ttb.tt_ind2sub(self.shape[:-2], np.array([i]))
                s += str(idx[0].tolist())[0:-1]
                s += ", :, :]"
                s += " = \n"
                s += str(
                    self.data[
                        tuple(
                            np.concatenate(
                                (idx[0], np.array([slice(None), slice(None)]))
                            )
                        )
                    ]
                )
                s += "\n"
        # s += '\n'
        return s

    __str__ = __repr__


if __name__ == "__main__":
    import doctest  # pragma: no cover

    doctest.testmod()  # pragma: no cover
