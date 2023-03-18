# Copyright 2022 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.
"""Sparse Tensor Implementation"""
from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, Callable, List, Optional, Tuple, Union, cast, overload

import numpy as np
import scipy.sparse.linalg
from numpy_groupies import aggregate as accumarray
from scipy import sparse

import pyttb as ttb
from pyttb.pyttb_utils import (
    tt_assignment_type,
    tt_dimscheck,
    tt_ind2sub,
    tt_intvec2str,
    tt_sizecheck,
    tt_sub2ind,
    tt_subscheck,
    tt_subsubsref,
    tt_valscheck,
)


def tt_to_sparse_matrix(
    sptensorInstance: sptensor, mode: int, transpose: bool = False
) -> sparse.coo_matrix:
    """
    Helper function to unwrap sptensor into sparse matrix, should replace the core need
    for sptenmat

    Parameters
    ----------
    sptensorInstance: sparse tensor to unwrap
    mode: Mode around which to unwrap tensor
    transpose: Whether or not to tranpose unwrapped tensor

    Returns
    -------
    spmatrix: unwrapped tensor
    """
    old = np.setdiff1d(np.arange(sptensorInstance.ndims), mode).astype(int)
    spmatrix = sptensorInstance.reshape(
        (np.prod(np.array(sptensorInstance.shape)[old]),), old
    ).spmatrix()
    if transpose:
        return spmatrix.transpose()
    return spmatrix


def tt_from_sparse_matrix(
    spmatrix: sparse.coo_matrix, shape: Any, mode: int, idx: int
) -> sptensor:
    """
    Helper function to wrap sparse matrix into sptensor.
    Inverse of :class:`pyttb.tt_to_sparse_matrix`

    Parameters
    ----------
    spmatrix: :class:`Scipy.sparse.coo_matrix`
    mode: int
        Mode around which tensor was unwrapped
    idx: int
        in {0,1}, idx of mode in spmatrix, s.b. 0 for tranpose=True

    Returns
    -------
    sptensorInstance: :class:`pyttb.sptensor`
    """
    siz = np.array(shape)
    old = np.setdiff1d(np.arange(len(shape)), mode).astype(int)
    sptensorInstance = ttb.sptensor.from_tensor_type(sparse.coo_matrix(spmatrix))

    # This expands the compressed dimension back to full size
    sptensorInstance = sptensorInstance.reshape(siz[old], idx)
    # This puts the modes in the right order, reshape places modified modes after the
    # unchanged ones
    sptensorInstance = sptensorInstance.reshape(
        shape,
        np.concatenate([np.arange(1, mode + 1), [0], np.arange(mode + 1, len(shape))]),
    )

    return sptensorInstance


class sptensor:
    """
    SPTENSOR Class for sparse tensors.
    """

    def __init__(self):
        """
        Create an empty sparse tensor

        Returns
        -------
        :class:`pyttb.sptensor`
        """
        # Empty constructor
        self.subs = np.array([], ndmin=2, dtype=int)
        self.vals = np.array([], ndmin=2)
        self.shape = ()

        # TODO: do we want to support an empty sptensor with defined shape?
        # Specifying size
        # if tt_sizecheck(source):
        #    self.subs = np.array([])
        #    self.vals = np.array([])
        #    self.size = source
        #    return

    @classmethod
    def from_data(
        cls, subs: np.ndarray, vals: np.ndarray, shape: Tuple[int, ...]
    ) -> sptensor:
        """
        Construct an sptensor from fully defined SUB, VAL and SIZE matrices.

        Parameters
        ----------
        subs: location of non-zero entries
        vals: values for non-zero entries
        shape: shape of sparse tensor

        Examples
        --------
        Import required modules:

        >>> import pyttb as ttb
        >>> import numpy as np

        Set up input data
        # Create sptensor with explicit data description

        >>> subs = np.array([[1, 2], [1, 3]])
        >>> vals = np.array([[6], [7]])
        >>> shape = (4, 4, 4)
        >>> K0 = ttb.sptensor.from_data(subs,vals, shape)
        """
        sptensorInstance = cls()
        sptensorInstance.subs = subs
        sptensorInstance.vals = vals
        sptensorInstance.shape = shape
        return sptensorInstance

    @classmethod
    def from_tensor_type(
        cls, source: Union[sptensor, ttb.tensor, sparse.coo_matrix]
    ) -> sptensor:
        """
        Contruct an :class:`pyttb.sptensor` from compatible tensor types

        Parameters
        ----------
        source: Source tensor to create sptensor from

        Returns
        -------
        Generated Sparse Tensor
        """
        # Copy Constructor
        if isinstance(source, sptensor):
            return cls().from_data(source.subs.copy(), source.vals.copy(), source.shape)

        # Convert SPTENMAT
        if isinstance(source, ttb.sptenmat):  # pragma: no cover
            raise NotImplementedError

        # Convert Tensor
        if isinstance(source, ttb.tensor):
            subs, vals = source.find()
            return cls().from_data(subs.copy(), vals.copy(), source.shape)

        # Convert SPTENSOR3
        if isinstance(source, ttb.sptensor3):  # pragma: no cover
            raise NotImplementedError

        # Convert Matrix
        # TODO how to handle sparse matrices in general
        if isinstance(source, scipy.sparse.coo_matrix):
            subs = np.vstack((source.row, source.col)).transpose()
            vals = source.data[:, None]
            return ttb.sptensor.from_data(subs, vals, source.shape)

        # Convert MDA
        # TODO what is an MDA?

        assert False, "Invalid Tensor Type To initialize Sptensor"

    @classmethod
    def from_function(
        cls,
        function_handle: Callable[[Tuple[float, float]], np.ndarray],
        shape: Tuple[int, ...],
        nonzeros: float,
    ) -> sptensor:
        """
        Creates a sparse tensor of the specified shape with NZ nonzeros created from
        the specified function handle

        Parameters
        ----------
        function_handle: function that accepts 2 arguments and generates
            :class:`numpy.ndarray` of length nonzeros
        shape: tuple
        nonzeros: int or float

        Returns
        -------
        Generated Sparse Tensor
        """
        # Random Tensor
        assert callable(function_handle), "function_handle must be callable"

        if (nonzeros < 0) or (nonzeros >= np.prod(shape)):
            assert False, (
                "Requested number of non-zeros must be positive "
                "and less than the total size"
            )
        elif nonzeros < 1:
            nonzeros = int(np.ceil(np.prod(shape) * nonzeros))
        else:
            nonzeros = int(np.floor(nonzeros))
        nonzeros = int(nonzeros)

        # Keep iterating until we find enough unique non-zeros or we give up
        subs = np.array([])
        cnt = 0
        while (len(subs) < nonzeros) and (cnt < 10):
            subs = (
                np.random.uniform(size=[nonzeros, len(shape)]).dot(np.diag(shape))
            ).astype(int)
            subs = np.unique(subs, axis=0)
            cnt += 1

        nonzeros = min(nonzeros, subs.shape[0])
        subs = subs[0:nonzeros, :]
        vals = function_handle((nonzeros, 1))

        # Store everything
        return cls().from_data(subs, vals, shape)

    @classmethod
    def from_aggregator(
        cls,
        subs: np.ndarray,
        vals: np.ndarray,
        shape: Optional[Tuple[int, ...]] = None,
        function_handle: Union[str, Callable[[Any], Union[float, np.ndarray]]] = "sum",
    ) -> sptensor:
        """
        Construct an sptensor from fully defined SUB, VAL and shape matrices,
        after an aggregation is applied

        Parameters
        ----------
        subs: location of non-zero entries
        vals: values for non-zero entries
        shape: shape of sparse tensor
        function_handle: Aggregation function, or name of supported
            aggregation function from numpy_groupies

        Returns
        -------
        Generated Sparse Tensor

        Examples
        --------
        >>> subs = np.array([[1, 2], [1, 3]])
        >>> vals = np.array([[6], [7]])
        >>> shape = np.array([4, 4])
        >>> K0 = ttb.sptensor.from_aggregator(subs,vals)
        >>> K1 = ttb.sptensor.from_aggregator(subs,vals,shape)
        >>> function_handle = sum
        >>> K2 = ttb.sptensor.from_aggregator(subs,vals,shape,function_handle)
        """

        tt_subscheck(subs)
        tt_valscheck(vals)
        if subs.size > 1 and vals.shape[0] != subs.shape[0]:
            assert False, "Number of subscripts and values must be equal"

        # Extract the shape
        if shape is not None:
            tt_sizecheck(shape)
        else:
            shape = tuple(np.max(subs, axis=0) + 1)

        # Check for wrong input
        if subs.size > 0 and subs.shape[1] > len(shape):
            assert False, "More subscripts than specified by shape"

        # Check for subscripts out of range
        for j, dim in enumerate(shape):
            if subs.size > 0 and np.max(subs[:, j]) > dim:
                assert False, "Subscript exceeds sptensor shape"

        if subs.size == 0:
            newsubs = np.array([])
            newvals = np.array([])
        else:
            # Identify only the unique indices
            newsubs, loc = np.unique(subs, axis=0, return_inverse=True)
            # Sum the corresponding values
            # Squeeze to convert from column vector to row vector
            newvals = accumarray(
                loc, np.squeeze(vals), size=newsubs.shape[0], func=function_handle
            )

        # Find the nonzero indices of the new values
        nzidx = np.nonzero(newvals)
        newsubs = newsubs[nzidx]
        # None index to convert from row back to column vector
        newvals = newvals[nzidx]
        if newvals.size > 0:
            newvals = newvals[:, None]

        # Store everything
        return cls().from_data(newsubs, newvals, shape)

    # TODO decide if property
    def allsubs(self) -> np.ndarray:
        """
        Generate all possible subscripts for sparse tensor

        Returns
        -------
        s: All possible subscripts for sptensor
        """

        # Generate all possible indices

        # Preallocate (discover any memory issues here!)
        s = np.zeros(shape=(np.prod(self.shape), self.ndims))

        # Generate appropriately sized ones vectors
        o = []
        for n in range(0, self.ndims):
            o.append(np.ones((self.shape[n], 1)))

        # Generate each column of the subscripts in turn
        for n in range(0, self.ndims):
            i = o.copy()
            i[n] = np.expand_dims(np.arange(0, self.shape[n]), axis=1)
            s[:, n] = np.squeeze(ttb.khatrirao(i))

        return s.astype(int)

    def collapse(
        self,
        dims: Optional[np.ndarray] = None,
        fun: Callable[[np.ndarray], Union[float, np.ndarray]] = np.sum,
    ) -> Union[float, np.ndarray, sptensor]:
        """
        Collapse sparse tensor along specified dimensions.

        Parameters
        ----------
        dims: Dimensions to collapse
        fun: Method used to collapse dimensions

        Returns
        -------
        Collapsed value

        Example
        -------
        >>> subs = np.array([[1, 2], [1, 3]])
        >>> vals = np.array([[1], [1]])
        >>> shape = np.array([4, 4])
        >>> X = ttb.sptensor.from_data(subs, vals, shape)
        >>> X.collapse()
        2
        >>> X.collapse(np.arange(X.ndims), sum)
        2
        """
        if dims is None:
            dims = np.arange(0, self.ndims)

        dims, _ = tt_dimscheck(dims, self.ndims)
        remdims = np.setdiff1d(np.arange(0, self.ndims), dims)

        # Check for the case where we accumulate over *all* dimensions
        if remdims.size == 0:
            return fun(self.vals.transpose()[0])

        # Calculate the size of the result
        newsize = np.array(self.shape)[remdims]

        # Check for the case where the result is just a dense vector
        if remdims.size == 1:
            if self.subs.size > 0:
                return accumarray(
                    self.subs[:, remdims].transpose()[0],
                    self.vals.transpose()[0],
                    size=newsize[0],
                    func=fun,
                )
            return np.zeros((newsize[0], 1))

        # Create Result
        if self.subs.size > 0:
            return ttb.sptensor.from_aggregator(
                self.subs[:, remdims], self.vals, tuple(newsize), fun
            )
        return ttb.sptensor.from_data(np.array([]), np.array([]), tuple(newsize))

    def contract(self, i: int, j: int) -> Union[np.ndarray, sptensor, ttb.tensor]:
        """
        Contract tensor along two dimensions (array trace).

        Parameters
        ----------
        i: First dimension
        j: Second dimension

        Returns
        -------
        Contracted sptensor, converted to tensor if sufficiently dense

        Example
        -------
        >>> X = ttb.tensor.from_data(np.ones((2,2)))
        >>> Y = sptensor.from_tensor_type(X)
        >>> Y.contract(0, 1)
        2.0
        """
        if self.shape[i] != self.shape[j]:
            assert False, "Must contract along equally sized dimensions"

        if i == j:
            assert False, "Must contract along two different dimensions"

        # Easy case - returns a scalar
        if self.ndims == 2:
            tfidx = self.subs[:, 0] == self.subs[:, 1]  # find diagonal entries
            return sum(self.vals[tfidx].transpose()[0])

        # Remaining dimensions after contract
        remdims = np.setdiff1d(np.arange(0, self.ndims), np.array([i, j])).astype(int)

        # Size for return
        newsize = tuple(np.array(self.shape)[remdims])

        # Find index of values on diagonal
        indx = np.where(self.subs[:, i] == self.subs[:, j])[0]

        # Let constructor sum entries
        if remdims.size == 1:
            y = ttb.sptensor.from_aggregator(
                self.subs[indx, remdims][:, None], self.vals[indx], newsize
            )
        else:
            y = ttb.sptensor.from_aggregator(
                self.subs[indx, :][:, remdims], self.vals[indx], newsize
            )

        # Check if result should be dense
        if y.nnz > 0.5 * np.prod(y.shape):
            # Final result is a dense tensor
            return ttb.tensor.from_tensor_type(y)
        return y

    def double(self) -> np.ndarray:
        """
        Convert sptensor to dense multidimensional array
        """
        a = np.zeros(self.shape)
        if self.nnz > 0:
            a[tuple(self.subs.transpose())] = self.vals.transpose()[0]
        return a

    def elemfun(self, function_handle: Callable[[np.ndarray], np.ndarray]) -> sptensor:
        """
        Manipulate the non-zero elements of a sparse tensor

        Parameters
        ----------
        function_handle: Function that updates all values.

        Returns
        -------
        Updated sptensor

        Example
        -------
        >>> X = ttb.tensor.from_data(np.ones((2,2)))
        >>> Y = sptensor.from_tensor_type(X)
        >>> Z = Y.elemfun(lambda values: values*2)
        >>> Z.isequal(Y*2)
        True
        """

        vals = function_handle(self.vals)
        idx = np.where(vals > 0)[0]
        if idx.size == 0:
            return ttb.sptensor.from_data(np.array([]), np.array([]), self.shape)
        return ttb.sptensor.from_data(self.subs[idx, :], vals[idx], self.shape)

    def end(self, k: Optional[int] = None) -> int:
        """
        Last index of indexing expression for sparse tensor

        Parameters
        ----------
        k: int Dimension for subscript indexing
        """
        if k is not None:
            return self.shape[k] - 1
        return np.prod(self.shape) - 1

    def extract(self, searchsubs: np.ndarray) -> np.ndarray:
        """
        Extract value for a sptensor.

        Parameters
        ----------
        searchsubs: subscripts to find in sptensor

        See Also
        --------
        :meth:`__getitem__`
        """

        # Check range of requested subscripts
        if len(searchsubs.shape) > 1:
            p = searchsubs.shape[0]
        else:
            searchsubs = np.array(searchsubs[np.newaxis, :])
            p = searchsubs.shape[0]

        # Check that all subscripts are positive and less than the max
        invalid = (searchsubs < 0) | (searchsubs >= np.array(self.shape))
        badloc = np.where(np.sum(invalid, axis=1) > 0)
        if badloc[0].size > 0:
            error_msg = "The following subscripts are invalid: \n"
            badsubs = searchsubs[badloc, :]
            for i in np.arange(0, badloc[0].size):
                error_msg += f"\tsubscript = {tt_intvec2str(badsubs[i, :])} \n"
            assert False, f"{error_msg}" "Invalid subscripts"

        # Set the default answer to zero
        a = np.zeros(shape=(p, 1), dtype=self.vals.dtype)

        # Find which indices already exist and their locations
        loc = ttb.tt_ismember_rows(searchsubs, self.subs)
        # Fill in the non-zero elements in the answer
        nzsubs = np.where(loc >= 0)
        a[nzsubs] = self.vals[loc[nzsubs]]
        return a

    def find(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        FIND Find subscripts of nonzero elements in a sparse tensor.

        Returns
        -------
        subs: Subscripts of nonzero elements
        vals: Values at corresponding subscripts
        """
        return self.subs, self.vals

    def full(self) -> ttb.tensor:
        """
        FULL Convert a sparse tensor to a (dense) tensor.
        """
        # Handle the completely empty (no shape) case
        if len(self.shape) == 0:
            return ttb.tensor()

        # Create a dense zero tensor B that is the same shape as A
        B = ttb.tensor.from_data(np.zeros(shape=self.shape), self.shape)

        if self.subs.size == 0:
            return B

        # Extract the linear indices of entries in A
        idx = tt_sub2ind(self.shape, self.subs)

        # Copy the values of A into B using linear indices
        B[idx.astype(int)] = self.vals.transpose()[0]
        return B

    def innerprod(
        self, other: Union[sptensor, ttb.tensor, ttb.ktensor, ttb.ttensor]
    ) -> float:
        """
        Efficient inner product with a sparse tensor

        Parameters
        ----------
        other: Other tensor to take innerproduct with
        """
        # If all entries are zero innerproduct must be 0
        if self.nnz == 0:
            return 0

        if isinstance(other, ttb.sptensor):
            if self.shape != other.shape:
                assert False, "Sptensors must be same shape for innerproduct"

            if other.nnz == 0:  # other sptensor is all zeros
                return 0

            if self.nnz < other.nnz:
                [subsSelf, valsSelf] = self.find()
                valsOther = other.extract(subsSelf)
            else:
                [subsOther, valsOther] = other.find()
                valsSelf = self.extract(subsOther)
            return valsOther.transpose().dot(valsSelf)

        if isinstance(other, ttb.tensor):
            if self.shape != other.shape:
                assert False, "Sptensor and tensor must be same shape for innerproduct"
            [subsSelf, valsSelf] = self.find()
            valsOther = other[subsSelf, "extract"]
            return valsOther.transpose().dot(valsSelf)

        if isinstance(other, (ttb.ktensor, ttb.ttensor)):  # pragma: no cover
            # Reverse arguments to call ktensor/ttensor implementation
            return other.innerprod(self)

        assert False, f"Inner product between sptensor and {type(other)} not supported"

    def isequal(self, other: Union[sptensor, ttb.tensor]) -> bool:
        """
        Exact equality for sptensors

        Parameters
        ----------
        other: Other tensor to compare against
        """
        if self.shape != other.shape:
            return False
        if isinstance(other, ttb.sptensor):
            return (self - other).nnz == 0
        if isinstance(other, ttb.tensor):
            return other.isequal(self)
        return False

    def logical_and(self, B: Union[float, sptensor, ttb.tensor]) -> sptensor:
        """
        Logical and with self and another object

        Parameters
        ----------
        B: Other value to compare with

        Returns
        ----------
        Indicator tensor
        """
        # Case 1: One argument is a scalar
        if isinstance(B, (int, float)):
            if B == 0:
                C = sptensor.from_data(np.array([]), np.array([]), self.shape)
            else:
                newvals = self.vals == B
                C = sptensor.from_data(self.subs, newvals, self.shape)
            return C
        # Case 2: Argument is a tensor of some sort
        if isinstance(B, sptensor):
            # Check that the shapes match
            if not self.shape == B.shape:
                assert False, "Must be tensors of the same shape"

            def is_length_2(x):
                return len(x) == 2

            C = sptensor.from_aggregator(
                np.vstack((self.subs, B.subs)),
                np.vstack((self.vals, B.vals)),
                self.shape,
                is_length_2,
            )

            return C

        if isinstance(B, ttb.tensor):
            BB = sptensor.from_data(
                self.subs, B[self.subs, "extract"][:, None], self.shape
            )
            C = self.logical_and(BB)
            return C

        # Otherwise
        assert False, "The arguments must be two sptensors or an sptensor and a scalar."

    def logical_not(self) -> sptensor:
        """
        Logical NOT for sptensors

        Returns
        -------
        Sparse tensor with all zero-values marked from original
        sparse tensor
        """
        allsubs = self.allsubs()
        subsIdx = ttb.tt_setdiff_rows(allsubs, self.subs)
        subs = allsubs[subsIdx]
        trueVector = np.ones(shape=(subs.shape[0], 1), dtype=bool)
        return sptensor.from_data(subs, trueVector, self.shape)

    @overload
    def logical_or(self, B: Union[float, ttb.tensor]) -> ttb.tensor:
        ...  # pragma: no cover see coveragepy/issues/970

    @overload
    def logical_or(self, B: sptensor) -> sptensor:
        ...  # pragma: no cover see coveragepy/issues/970

    def logical_or(
        self, B: Union[float, ttb.tensor, sptensor]
    ) -> Union[ttb.tensor, sptensor]:
        """
        Logical OR for sptensor and another value

        Returns
        -------
        Indicator tensor
        """
        # Case 1: Argument is a scalar or tensor
        if isinstance(B, (float, int, ttb.tensor)):
            return self.full().logical_or(B)

        # Case 2: Argument is an sptensor
        if self.shape != B.shape:
            assert False, "Logical Or requires tensors of the same size"

        if isinstance(B, ttb.sptensor):

            def is_length_ge_1(x):
                return len(x) >= 1

            return sptensor.from_aggregator(
                np.vstack((self.subs, B.subs)),
                np.ones((self.subs.shape[0] + B.subs.shape[0], 1)),
                self.shape,
                is_length_ge_1,
            )

        assert False, "Sptensor Logical Or argument must be scalar or sptensor"

    @overload
    def logical_xor(self, other: Union[float, ttb.tensor]) -> ttb.tensor:
        ...  # pragma: no cover see coveragepy/issues/970

    @overload
    def logical_xor(self, other: sptensor) -> sptensor:
        ...  # pragma: no cover see coveragepy/issues/970

    def logical_xor(
        self, other: Union[float, ttb.tensor, sptensor]
    ) -> Union[ttb.tensor, sptensor]:
        """
        Logical XOR for sptensors

        Parameters
        ----------
        other: Other value to xor against

        Returns
        -------
        Indicator tensor
        """
        # Case 1: Argument is a scalar or dense tensor
        if isinstance(other, (float, int, ttb.tensor)):
            return self.full().logical_xor(other)

        # Case 2: Argument is an sptensor
        if isinstance(other, ttb.sptensor):
            # Check shape consistency
            if self.shape != other.shape:
                assert False, "Logical XOR requires tensors of the same size"

            def length1(x):
                return len(x) == 1

            subs = np.vstack((self.subs, other.subs))
            return ttb.sptensor.from_aggregator(
                subs, np.ones((len(subs), 1)), self.shape, length1
            )

        assert False, "The argument must be an sptensor, tensor or scalar"

    def mask(self, W: sptensor) -> np.ndarray:
        """
        Extract values as specified by a mask tensor

        Parameters
        ----------
        W: Mask tensor

        Returns
        -------
        Extracted values
        """
        # Error check
        if len(W.shape) != len(self.shape) or np.any(
            np.array(W.shape) > np.array(self.shape)
        ):
            assert False, "Mask cannot be bigger than the data tensor"

        # Extract locations of nonzeros in W
        wsubs, _ = W.find()

        # Find which values in the mask match nonzeros in X
        idx = ttb.tt_ismember_rows(wsubs, self.subs)

        # Assemble return array
        nvals = wsubs.shape[0]
        vals = np.zeros((nvals, 1))
        vals[idx] = self.vals[idx]
        return vals

    def mttkrp(self, U: Union[ttb.ktensor, List[np.ndarray]], n: int) -> np.ndarray:
        """
        Matricized tensor times Khatri-Rao product for sparse tensor.

        Parameters
        ----------
        U: Matrices to create the Khatri-Rao product
        n: Mode to matricize sptensor in

        Returns
        -------
        Matrix product

        Examples
        --------
        >>> matrix = np.ones((4, 4))
        >>> subs = np.array([[1, 1, 1], [1, 1, 3], [2, 2, 2], [3, 3, 3]])
        >>> vals = np.array([[0.5], [1.5], [2.5], [3.5]])
        >>> shape = (4, 4, 4)
        >>> sptensorInstance = sptensor.from_data(subs, vals, shape)
        >>> sptensorInstance.mttkrp(np.array([matrix, matrix, matrix]), 0)
        array([[0. , 0. , 0. , 0. ],
               [2. , 2. , 2. , 2. ],
               [2.5, 2.5, 2.5, 2.5],
               [3.5, 3.5, 3.5, 3.5]])

        """
        # In the sparse case, it is most efficient to do a series of TTV operations
        # rather than forming the Khatri-Rao product.

        N = self.ndims

        if isinstance(U, ttb.ktensor):
            # Absorb lambda into one of the factors but not the one that is skipped
            if n == 0:
                U.redistribute(1)
            else:
                U.redistribute(0)

            # Extract the factor matrices
            U = U.factor_matrices

        if not isinstance(U, np.ndarray) and not isinstance(U, list):
            assert False, "Second argument must be ktensor or array"

        if len(U) != N:
            assert False, "List is the wrong length"

        if n == 0:
            R = U[1].shape[1]
        else:
            R = U[0].shape[1]

        V = np.zeros((self.shape[n], R))
        for r in range(R):
            # Set up list with appropriate vectors for ttv multiplication
            Z = []
            for i in range(N):
                if i != n:
                    Z.append(U[i][:, r])
                else:
                    Z.append(np.array([]))
            # Perform ttv multiplication
            V[:, r] = self.ttv(Z, -(n + 1)).double()

        return V

    @property
    def ndims(self) -> int:
        """
        NDIMS Number of dimensions of a sparse tensor.
        """
        return len(self.shape)

    @property
    def nnz(self) -> int:
        """
        Number of nonzeros in sparse tensor
        """
        if self.subs.size == 0:
            return 0
        return self.subs.shape[0]

    def norm(self) -> np.floating:
        """
        Compute the Frobenius norm of a sparse tensor.
        """
        return np.linalg.norm(self.vals)

    def nvecs(self, n: int, r: int, flipsign: bool = True) -> np.ndarray:
        """
        Compute the leading mode-n vectors for a sparse tensor.

        Parameters
        ----------
        n: Mode to unfold
        r: Number of eigenvectors to compute
        flipsign: Make each eigenvector's largest element positive
        """
        old = np.setdiff1d(np.arange(self.ndims), n).astype(int)
        # tnt calculation is a workaround for missing sptenmat
        mutatable_sptensor = (
            sptensor.from_tensor_type(self)
            .reshape((np.prod(np.array(self.shape)[old]), 1), old)
            .squeeze()
        )
        if isinstance(mutatable_sptensor, (int, float, np.generic)):
            raise ValueError(
                "Cannot call nvecs on sptensor with only singleton dimensions"
            )
        tnt = mutatable_sptensor.spmatrix().transpose()
        y = tnt.transpose().dot(tnt)
        if r < y.shape[0] - 1:
            _, v = scipy.sparse.linalg.eigs(y, r)
        else:
            warnings.warn(
                "Greater than or equal to sptensor.shape[n] - 1 eigenvectors requires"
                " cast to dense to solve"
            )
            w, v = scipy.linalg.eig(y.toarray())
            v = v[(-np.abs(w)).argsort()]
            v = v[:, :r]

        if flipsign:
            idx = np.argmax(np.abs(v), axis=0)
            for i in range(v.shape[1]):
                if v[idx[i], i] < 0:  # pragma: no cover
                    v[:, i] *= -1
        return v

    def ones(self) -> sptensor:
        """
        Replace nonzero elements of sparse tensor with ones
        """
        oneVals = self.vals.copy()
        oneVals.fill(1)
        return ttb.sptensor.from_data(self.subs, oneVals, self.shape)

    def permute(self, order: np.ndarray) -> sptensor:
        """
        Rearrange the dimensions of a sparse tensor

        Parameters
        ----------
        order: Updated order of dimensions
        """
        # Error check
        if self.ndims != order.size or np.any(
            np.sort(order) != np.arange(0, self.ndims)
        ):
            assert False, "Invalid permutation order"

        # Do the permutation
        if not self.subs.size == 0:
            return ttb.sptensor.from_data(
                self.subs[:, order], self.vals, tuple(np.array(self.shape)[order])
            )
        return ttb.sptensor.from_data(
            self.subs, self.vals, tuple(np.array(self.shape)[order])
        )

    def reshape(
        self,
        new_shape: Tuple[int, ...],
        old_modes: Optional[Union[np.ndarray, int]] = None,
    ) -> sptensor:
        """
        Reshape specified modes of sparse tensor

        Parameters
        ----------
        new_shape: tuple
        old_modes: :class:`Numpy.ndarray`

        Returns
        -------
        :class:`pyttb.sptensor`
        """

        if old_modes is None:
            old_modes = np.arange(0, self.ndims, dtype=int)
            keep_modes = np.array([], dtype=int)
        else:
            keep_modes = np.setdiff1d(np.arange(0, self.ndims, dtype=int), old_modes)

        shapeArray = np.array(self.shape)
        old_shape = shapeArray[old_modes]
        keep_shape = shapeArray[keep_modes]

        if np.prod(new_shape) != np.prod(old_shape):
            assert False, "Reshape must maintain tensor size"

        if self.subs.size == 0:
            return ttb.sptensor.from_data(
                np.array([]),
                np.array([]),
                tuple(np.concatenate((keep_shape, new_shape))),
            )
        if np.isscalar(old_shape):
            old_shape = (old_shape,)
            inds = ttb.tt_sub2ind(old_shape, self.subs[:, old_modes][:, None])
        else:
            inds = ttb.tt_sub2ind(old_shape, self.subs[:, old_modes])
        new_subs = ttb.tt_ind2sub(new_shape, inds)
        return ttb.sptensor.from_data(
            np.concatenate((self.subs[:, keep_modes], new_subs), axis=1),
            self.vals,
            tuple(np.concatenate((keep_shape, new_shape))),
        )

    def scale(self, factor: np.ndarray, dims: Union[float, np.ndarray]) -> sptensor:
        """
        Scale along specified dimensions for sparse tensors

        Parameters
        ----------
        factor: :class:`numpy.ndarray`
        dims: int or :class:`numpy.ndarray`

        Returns
        -------
        :class:`pyttb.sptensor`
        """
        if isinstance(dims, (float, int)):
            dims = np.array([dims])
        dims, _ = ttb.tt_dimscheck(dims, self.ndims)

        if isinstance(factor, ttb.tensor):
            shapeArray = np.array(self.shape)
            if np.any(factor.shape != shapeArray[dims]):
                assert False, "Size mismatch in scale"
            return ttb.sptensor.from_data(
                self.subs,
                self.vals * factor[self.subs[:, dims], "extract"][:, None],
                self.shape,
            )
        if isinstance(factor, ttb.sptensor):
            shapeArray = np.array(self.shape)
            if np.any(factor.shape != shapeArray[dims]):
                assert False, "Size mismatch in scale"
            return ttb.sptensor.from_data(
                self.subs, self.vals * factor.extract(self.subs[:, dims]), self.shape
            )
        if isinstance(factor, np.ndarray):
            shapeArray = np.array(self.shape)
            if factor.shape[0] != shapeArray[dims]:
                assert False, "Size mismatch in scale"
            return ttb.sptensor.from_data(
                self.subs,
                self.vals * factor[self.subs[:, dims].transpose()[0]],
                self.shape,
            )
        assert False, "Invalid scaling factor"

    def spmatrix(self) -> sparse.coo_matrix:
        """
        Converts a two-way sparse tensor to a sparse matrix in
        scipy.sparse.coo_matrix format
        """
        if self.ndims != 2:
            assert False, "Sparse tensor must be two dimensional"

        if self.subs.size == 0:
            return sparse.coo_matrix(self.shape)
        return sparse.coo_matrix(
            (self.vals.transpose()[0], self.subs.transpose()), self.shape
        )

    def squeeze(self) -> Union[sptensor, float]:
        """
        Remove singleton dimensions from a sparse tensor

        Returns
        -------
        :class:`pyttb.sptensor` or float if sptensor is only singleton dimensions
        """
        shapeArray = np.array(self.shape)

        # No singleton dimensions
        if np.all(shapeArray > 1):
            return ttb.sptensor.from_tensor_type(self)
        idx = np.where(shapeArray > 1)[0]
        if idx.size == 0:
            return self.vals[0].copy()
        siz = tuple(shapeArray[idx])
        if self.vals.size == 0:
            return ttb.sptensor.from_data(np.array([]), np.array([]), siz)
        return ttb.sptensor.from_data(self.subs[:, idx], self.vals, siz)

    def subdims(self, region: Sequence[Union[int, np.ndarray, slice]]) -> np.ndarray:
        """
        SUBDIMS Compute the locations of subscripts within a subdimension.

        Parameters
        ----------
        region: :class:`numpy.ndarray` or tuple denoting indexing
            Subset of total sptensor shape in which to find non-zero values

        Returns
        -------
        :class:`numpy.ndarray`
            Index into subs for non-zero values in region

        Examples
        --------
        >>> subs = np.array([[1, 1, 1], [1, 1, 3], [2, 2, 2], [3, 3, 3]])
        >>> vals = np.array([[0.5], [1.5], [2.5], [3.5]])
        >>> shape = (4, 4, 4)
        >>> sp = sptensor.from_data(subs,vals,shape)
        >>> region = [np.array([1]), np.array([1]), np.array([1,3])]
        >>> loc = sp.subdims(region)
        >>> print(loc)
        [0 1]
        >>> region = (1, 1, slice(None, None, None))
        >>> loc = sp.subdims(region)
        >>> print(loc)
        [0 1]
        """
        if len(region) != self.ndims:
            assert False, "Number of subdimensions must equal number of dimensions"

        # Error check that range is valid
        # TODO I think only accepting numeric arrays fixes this

        # TODO we use this empty check a lot, do we want a boolean we store in the
        #  class for this?
        if self.subs.size == 0:
            loc = np.array([])
            return loc

        # Compute the indices of the subscripts that are within the
        # specified range. We start with all indices in loc and
        # pare it down to a final list.

        loc = np.arange(0, len(self.subs))

        for i in range(0, self.ndims):
            # TODO: Consider cleaner typing coercion
            # Find subscripts that match in dimension i
            if isinstance(region[i], (int, np.generic)):
                tf = np.isin(self.subs[loc, i], cast(int, region[i]))
            elif isinstance(region[i], (np.ndarray, list)):
                tf = np.isin(self.subs[loc, i], cast(np.ndarray, region[i]))
            elif isinstance(region[i], slice):
                sliceRegion = range(0, self.shape[i])[region[i]]
                tf = np.isin(self.subs[loc, i], sliceRegion)
            else:
                raise ValueError(
                    f"Unexpected type in region sequence. "
                    f"At index: {i} got {region[i]} with type {type(region[i])}"
                )

            # Pare down the list of indices
            loc = loc[tf]

        return loc

    # pylint: disable=too-many-branches, too-many-locals
    def ttv(
        self,
        vector: Union[np.ndarray, List[np.ndarray]],
        dims: Optional[Union[int, np.ndarray]] = None,
    ) -> Union[sptensor, ttb.tensor]:
        """
        Sparse tensor times vector

        Parameters
        ----------
        vector: Vector(s) to multiply against
        dims: Dimensions to multiply with vector(s)
        """

        if dims is None:
            dims = np.array([])
        elif isinstance(dims, (float, int)):
            dims = np.array([dims])

        # Check that vector is a list of vectors,
        # if not place single vector as element in list
        if len(vector) > 0 and isinstance(vector[0], (int, float, np.int_, np.float_)):
            return self.ttv(np.array([vector]), dims)

        # Get sorted dims and index for multiplicands
        dims, vidx = ttb.tt_dimscheck(dims, self.ndims, len(vector))
        remdims = np.setdiff1d(np.arange(0, self.ndims), dims).astype(int)

        # Check that each multiplicand is the right size.
        for i in range(dims.size):
            if vector[vidx[i]].shape != (self.shape[dims[i]],):
                assert False, "Multiplicand is wrong size"

        # Multiply each value by the appropriate elements of the appropriate vector
        newvals = self.vals.copy()
        subs = self.subs.copy()
        if subs.size == 0:  # No non-zeros in tensor
            newsubs = np.array([], dtype=int)
        else:
            for n, dims_n in enumerate(dims):
                idx = subs[:, dims_n]  # extract indices for dimension n
                w = vector[vidx[n]]  # extract the nth vector
                bigw = w[idx][:, None]  # stretch out the vector
                newvals = newvals * bigw
            newsubs = subs[:, remdims].astype(int)

        # Case 0: If all dimensions were used, then just return the sum
        if remdims.size == 0:
            return np.sum(newvals)

        # Otherwise, figure out new subscripts and accumulate the results.
        newsiz = np.array(self.shape, dtype=int)[remdims]

        # Case 1: Result is a vector
        if remdims.size == 1:
            if newvals.size == 0:
                return ttb.sptensor.from_data(np.array([]), np.array([]), tuple(newsiz))
            c = accumarray(
                newsubs.transpose()[0], newvals.transpose()[0], size=newsiz[0]
            )
            if np.count_nonzero(c) <= 0.5 * newsiz:
                return ttb.sptensor.from_aggregator(
                    np.arange(0, newsiz)[:, None], c, tuple(newsiz)
                )
            return ttb.tensor.from_data(c, tuple(newsiz))

        # Case 2: Result is a multiway array
        c = ttb.sptensor.from_aggregator(newsubs, newvals, tuple(newsiz))

        # Convert to a dense tensor if more than 50% of the result is nonzero.
        if c.nnz > 0.5 * np.prod(newsiz):
            c = ttb.tensor.from_tensor_type(c)

        return c

    # pylint: disable=too-many-branches
    def __getitem__(self, item):
        """
        Subscripted reference for a sparse tensor.

        We can extract elements or subtensors from a sparse tensor in the
        following ways.

        Case 1a: y = X(i1,i2,...,iN), where each in is an index, returns a
        scalar.

        Case 1b: Y = X(R1,R2,...,RN), where one or more Rn is a range and
        the rest are indices, returns a sparse tensor. The elements are
        renumbered here as appropriate.

        Case 2a: V = X(S) or V = X(S,'extract'), where S is a p x n array
        of subscripts, returns a vector of p values.

        Case 2b: V = X(I) or V = X(I,'extract'), where I is a set of p
        linear indices, returns a vector of p values.

        Any ambiguity results in executing the first valid case. This
        is particularily an issue if ndims(X)==1.

        Parameters
        ----------
        item: tuple(int),tuple(slice),:class:`numpy.ndarray`

        Returns
        -------

        :class:`numpy.ndarray` or :class:`pyttb.sptensor`

        Examples
        --------
        >>> subs = np.array([[3,3,3],[1,1,0],[1,2,1]])
        >>> vals = np.array([3,5,1])
        >>> shape = (4,4,4)
        >>> X = sptensor.from_data(subs,vals,shape)
        >>> _ = X[0,1,0] #<-- returns zero
        >>> _ = X[3,3,3] #<-- returns 3
        >>> _ = X[2:3,:,:] #<-- returns 1 x 4 x 4 sptensor
        """
        # This does not work like MATLAB TTB; you must call sptensor.extract to get
        # this functionality: X([1:6]','extract') %<-- extracts a vector of 6 elements

        # TODO IndexError for value outside of indices
        # TODO Key error if item not in container
        # *** CASE 1: Rectangular Subtensor ***
        if isinstance(item, tuple) and len(item) == self.ndims:
            # Extract the subdimensions to be extracted from self
            region = item

            # Pare down the list of subscripts (and values) to only
            # those within the subdimensions specified by region.
            loc = self.subdims(region)
            subs = self.subs[loc, :]
            vals = self.vals[loc]

            # Find the size of the subtensor and renumber the
            # subscripts
            [subs, shape] = ttb.tt_renumber(subs, self.shape, region)

            # Determine the subscripts
            newsiz = []  # (future) new size
            kpdims = []  # dimensions to keep
            rmdims = []  # dimensions to remove

            # Determine the new size and what dimensions to keep
            for i, a_region in enumerate(region):
                if isinstance(a_region, slice):
                    newsiz.append(self.shape[i])
                    kpdims.append(i)
                elif not isinstance(a_region, (int, float)):
                    newsiz.append(np.prod(a_region))
                    kpdims.append(i)
                else:
                    rmdims.append(i)

            newsiz = np.array(newsiz, dtype=int)
            kpdims = np.array(kpdims, dtype=int)
            rmdims = np.array(rmdims, dtype=int)

            # Return a single double value for a zero-order sub-tensor
            if newsiz.size == 0:
                if vals.size == 0:
                    a = np.array([[0]])
                else:
                    a = vals
                return a

            # Assemble the resulting sparse tensor
            # TODO clean up tuple array cast below
            if subs.size == 0:
                a = sptensor.from_data(
                    np.array([]), np.array([]), tuple(np.array(shape)[kpdims])
                )
            else:
                a = sptensor.from_data(
                    subs[:, kpdims], vals, tuple(np.array(shape)[kpdims])
                )
            return a

        # TODO understand how/ why this is used, logic doesn't translate immediately
        # Case 2: EXTRACT

        # *** CASE 2a: Subscript indexing ***
        if (
            isinstance(item, np.ndarray)
            and len(item.shape) == 2
            and item.shape[1] == self.ndims
        ):
            srchsubs = np.array(item)

        # *** CASE 2b: Linear indexing ***
        else:
            # Error checking
            if isinstance(item, list):
                idx = np.array(item)
            elif isinstance(item, np.ndarray):
                idx = item
            else:
                assert False, "Invalid indexing"

            if len(idx.shape) != 1:
                assert False, "Expecting a row index"

            # extract linear indices and convert to subscripts
            srchsubs = tt_ind2sub(self.shape, idx)

        a = self.extract(srchsubs)
        a = tt_subsubsref(a, item)

        return a

    # pylint:disable=too-many-statements, too-many-branches, too-many-locals
    def __setitem__(self, key, value):
        """
        Subscripted assignment for sparse tensor.

        We can assign elements to a sptensor in three ways.

        Case 1: X(R1,R2,...,RN) = Y, in which case we replace the
        rectangular subtensor (or single element) specified by the ranges
        R1,...,RN with Y. The right-hand-side can be a scalar or an
        sptensor.

        Case 2: X(S) = V, where S is a p x n array of subscripts and V is
        a scalar value or a vector containing p values.

        Linear indexing is not supported for sparse tensors.

        Examples
        --------
        X = sptensor([30 40 20]) <-- Create an emtpy 30 x 40 x 20 sptensor
        X(30,40,20) = 7 <-- Assign a single element to be 7
        X([1,1,1;2,2,2]) = 1 <-- Assign a list of elements to the same value
        X(11:20,11:20,11:20) = sptenrand([10,10,10],10) <-- subtensor!
        X(31,41,21) = 7 <-- grows the size of the tensor
        X(111:120,111:120,111:120) = sptenrand([10,10,10],10) <-- grows
        X(1,1,1,1) = 4 <-- increases the number of dimensions from 3 to 4

        X = sptensor([30]) <-- empty one-dimensional tensor
        X([4:6]) = 1 <-- set subtensor to ones (does not increase dimension)
        X([10;12;14]) = (4:6)'  <-- set three elements
        X(31) = 7 <-- grow the first dimension
        X(1,1) = 0 <-- add a dimension, but no nonzeros

        Note regarding singleton dimensions: It is not possible to do, for
        instance, X(1,1:10,1:10) = sptenrand([1 10 10],5). However, it is okay
        to do X(1,1:10,1:10) = squeeze(sptenrand([1 10 10],5)).

        Parameters
        ----------
        key: tuple(int),tuple(slice),:class:`numpy.ndarray`
        value: int,float, :class:`numpy.ndarray`, :class:`pyttb.sptensor`

        """
        # TODO IndexError for value outside of indices
        # TODO Key error if item not in container
        # If empty sptensor and assignment is empty list or empty nparray
        if self.vals.size == 0 and (
            (isinstance(value, np.ndarray) and value.size == 0)
            or (isinstance(value, list) and value == [])
        ):
            return

        # Determine if we are doing a substenor or list of subscripts
        objectType = tt_assignment_type(self, key, value)

        # Case 1: Replace a sub-tensor
        if objectType == "subtensor":  # pylint:disable=too-many-nested-blocks
            # Case I(a): RHS is another sparse tensor
            if isinstance(value, ttb.sptensor):
                # First, Resize the tensor and check the size match with the tensor
                # that's being inserted.
                m = 0
                newsz = []
                for n, key_n in enumerate(key):
                    if isinstance(key_n, slice):
                        if self.ndims <= n:
                            if key_n.stop is None:
                                newsz.append(value.shape[m])
                            else:
                                newsz.append(key_n.stop)
                        else:
                            if key_n.stop is None:
                                newsz.append(max([self.shape[n], value.shape[m]]))
                            else:
                                newsz.append(max([self.shape[n], key_n.stop]))
                        m = m + 1
                    elif isinstance(key_n, (float, int)):
                        if self.ndims <= n:
                            newsz.append(key_n + 1)
                        else:
                            newsz.append(max([self.shape[n], key_n + 1]))
                    else:
                        if len(key_n) != value.shape[m]:
                            assert False, "RHS does not match range size"
                        if self.ndims <= n:
                            newsz.append(max(key_n) + 1)
                        else:
                            newsz.append(max([self.shape[n], max(key_n) + 1]))
                self.shape = tuple(newsz)

                # Expand subs array if there are new modes, i.e., if the order
                # has increased.
                if self.subs.size > 0 and (len(self.shape) > self.subs.shape[1]):
                    self.subs = np.append(
                        self.subs,
                        np.zeros(
                            shape=(
                                self.subs.shape[0],
                                len(self.shape) - self.subs.shape[1],
                            )
                        ),
                        axis=1,
                    )
                # Delete what currently occupies the specified range
                rmloc = self.subdims(key)
                kploc = np.setdiff1d(range(0, self.nnz), rmloc)
                # TODO: evaluate solution for assigning value to empty sptensor
                if len(self.subs.shape) > 1:
                    newsubs = self.subs[kploc.astype(int), :]
                else:
                    newsubs = self.subs[kploc.astype(int)]
                newvals = self.vals[kploc.astype(int)]

                # Renumber the subscripts
                addsubs = ttb.tt_irenumber(value, self.shape, key)
                if newsubs.size > 0 and addsubs.size > 0:
                    self.subs = np.vstack((newsubs, addsubs))
                    self.vals = np.vstack((newvals, value.vals))
                elif newsubs.size > 0:
                    self.subs = newsubs
                    self.vals = newvals
                elif addsubs.size > 0:
                    self.subs = addsubs
                    self.vals = value.vals
                else:
                    self.subs = np.array([], ndmin=2, dtype=int)
                    self.vals = np.array([], ndmin=2)

                return
            # Case I(b): Value is zero or scalar

            # First, resize the tensor, determine new size of existing modes
            newsz = []
            for n in range(0, self.ndims):
                if isinstance(key[n], slice):
                    if key[n].stop is None:
                        newsz.append(self.shape[n])
                    else:
                        newsz.append(max([self.shape[n], key[n].stop]))
                else:
                    newsz.append(max([self.shape[n], key[n] + 1]))

            # Determine size of new modes, if any
            for n in range(self.ndims, len(key)):
                if isinstance(key[n], slice):
                    if key[n].stop is None:
                        assert False, (
                            "Must have well defined slice when expanding sptensor "
                            "shape with setitem"
                        )
                    else:
                        newsz.append(key[n].stop)
                elif isinstance(key[n], np.ndarray):
                    newsz.append(max(key[n]) + 1)
                else:
                    newsz.append(key[n] + 1)
            self.shape = tuple(newsz)

            # Expand subs array if there are new modes, i.e. if the order has increased
            if self.subs.size > 0 and len(self.shape) > self.subs.shape[1]:
                self.subs = np.append(
                    self.subs,
                    np.zeros(
                        shape=(self.subs.shape[0], len(self.shape) - self.subs.shape[1])
                    ),
                    axis=1,
                )

            # Case I(b)i: Zero right-hand side
            if isinstance(value, (int, float)) and value == 0:
                # Delete what currently occupies the specified range
                rmloc = self.subdims(key)
                kploc = np.setdiff1d(range(0, self.nnz), rmloc).astype(int)
                self.subs = self.subs[kploc, :]
                self.vals = self.vals[kploc]
                return

            # Case I(b)ii: Scalar Right Hand Side
            if isinstance(value, (int, float)):
                # Determine number of dimensions (may be larger than current number)
                N = len(key)
                keyCopy = np.array(key)
                # Figure out how many indices are in each dimension
                nssubs = np.zeros((N, 1))
                for n in range(0, N):
                    if isinstance(key[n], slice):
                        # Generate slice explicitly to determine its length
                        keyCopy[n] = np.arange(0, self.shape[n])[key[n]]
                        indicesInN = len(keyCopy[n])
                    else:
                        indicesInN = 1
                    nssubs[n] = indicesInN

                # Preallocate (discover any memory issues here!)
                addsubs = np.zeros((np.prod(nssubs).astype(int), N))

                # Generate appropriately sized ones vectors
                o = []
                for n in range(N):
                    o.append(np.ones((int(nssubs[n]), 1)))

                # Generate each column of the subscripts in turn
                for n in range(N):
                    i = o.copy()
                    if not np.isscalar(keyCopy[n]):
                        i[n] = np.array(keyCopy[n])[:, None]
                    else:
                        i[n] = np.array(keyCopy[n], ndmin=2)
                    addsubs[:, n] = ttb.khatrirao(i).transpose()[:]

                if self.subs.size > 0:
                    # Replace existing values
                    loc = ttb.tt_intersect_rows(self.subs, addsubs)
                    self.vals[loc] = value
                    # pare down list of subscripts to add
                    addsubs = addsubs[ttb.tt_setdiff_rows(addsubs, self.subs)]

                # If there are things to insert then insert them
                if addsubs.size > 0:
                    if self.subs.size > 0:
                        self.subs = np.vstack((self.subs, addsubs.astype(int)))
                        self.vals = np.vstack(
                            (self.vals, value * np.ones((addsubs.shape[0], 1)))
                        )
                    else:
                        self.subs = addsubs.astype(int)
                        self.vals = value * np.ones(addsubs.shape[0])
                return

            assert False, "Invalid assignment value"

        # Case 2: Subscripts
        elif objectType == "subscripts":
            # Case II: Replacing values at specific indices

            newsubs = key
            if len(newsubs.shape) == 1:
                newsubs = np.expand_dims(newsubs, axis=0)
            tt_subscheck(newsubs, nargout=False)

            # Error check on subscripts
            if newsubs.shape[1] < self.ndims:
                assert False, "Invalid subscripts"

            # Check for expanding the order
            if newsubs.shape[1] > self.ndims:
                newshape = list(self.shape)
                for i in range(self.ndims, newsubs.shape[1]):
                    newshape.append(1)
                if self.subs.size > 0:
                    self.subs = np.concatenate(
                        (
                            self.subs,
                            np.ones(
                                (self.shape[0], newsubs.shape[1] - self.ndims),
                                dtype=int,
                            ),
                        ),
                        axis=1,
                    )
                self.shape = tuple(newshape)

            # Copy rhs to newvals
            newvals = value

            if isinstance(newvals, (float, int)):
                newvals = np.expand_dims([newvals], axis=1)

            # Error check the rhs is a column vector. We don't bother to handle any
            # other type with sparse tensors
            tt_valscheck(newvals, nargout=False)

            # Determine number of nonzeros being inserted.
            # (This is determined by number of subscripts)
            newnnz = newsubs.shape[0]

            # Error check on size of newvals
            if newvals.size == 1:
                # Special case where newvals is a single element to be assigned
                # to multiple LHS. Fix to correct size
                newvals = newvals * np.ones((newnnz, 1))

            elif newvals.shape[0] != newnnz:
                # Sizes don't match
                assert False, "Number of subscripts and number of values do not match!"

            # Remove duplicates and print warning if any duplicates were removed
            newsubs, idx = np.unique(newsubs, axis=0, return_index=True)
            if newsubs.shape[0] != newnnz:
                warnings.warn("Duplicate assignments discarded")

            newvals = newvals[idx]

            # Find which subscripts already exist and their locations
            tf = ttb.tt_ismember_rows(newsubs, self.subs)
            loc = np.where(tf >= 0)[0].astype(int)

            # Split into three groups for processing:
            #
            # Group A: Elements that already exist and need to be changed
            # Group B: Elements that already exist and need to be removed
            # Group C: Elements that do not exist and need to be added
            #
            # Note that we are ignoring any new zero elements, because
            # those obviously do not need to be added. Also, it's
            # important to process Group A before Group B because the
            # processing of Group B may change the locations of the
            # remaining elements.

            # TF+1 for logical consideration because 0 is valid index
            # and -1 is our null flag
            idxa = np.logical_and(tf + 1, newvals)[0]
            idxb = np.logical_and(tf + 1, np.logical_not(newvals))[0]
            idxc = np.logical_and(np.logical_not(tf + 1), newvals)[0]

            # Process Group A: Changing values
            if np.sum(idxa) > 0:
                self.vals[tf[idxa]] = newvals[idxa]
            # Proces Group B: Removing Values
            if np.sum(idxb) > 0:
                removesubs = loc[idxb]
                keepsubs = np.setdiff1d(range(0, self.nnz), removesubs)
                self.subs = self.subs[keepsubs, :]
                self.vals = self.vals[keepsubs]
            # Process Group C: Adding new, nonzero values
            if np.sum(idxc) > 0:
                self.subs = np.vstack((self.subs, newsubs[idxc, :]))
                self.vals = np.vstack((self.vals, newvals[idxc]))

            # Resize the tensor
            newshape = []
            for n, dim in enumerate(self.shape):
                smax = max(newsubs[:, n] + 1)
                newshape.append(max(dim, smax))
            self.shape = tuple(newshape)

            return

    def __eq__(self, other):
        """
        Equal comparator for sptensors

        Parameters
        ----------
        other: compare equality of sptensor to other

        Returns
        -------
        :class:`pyttb.sptensor`
        """
        # Case 1: other is a scalar
        if isinstance(other, (float, int)):
            if other == 0:
                return self.logical_not()
            idx = self.vals == other
            return sptensor.from_data(
                self.subs[idx.transpose()[0]],
                True * np.ones((self.subs.shape[0], 1)).astype(bool),
                self.shape,
            )

        # Case 2: other is a tensor type
        # Check sizes
        if self.shape != other.shape:
            assert False, "Size mismatch in sptensor equality"

        # Case 2a: other is a sparse tensor
        if isinstance(other, ttb.sptensor):
            # Find where their zeros intersect
            xzerosubs = ttb.tt_setdiff_rows(self.allsubs(), self.subs)
            otherzerosubs = ttb.tt_setdiff_rows(other.allsubs(), other.subs)
            # zzerosubs = np.isin(xzerosubs, otherzerosubs)
            zzerosubsIdx = ttb.tt_intersect_rows(
                self.allsubs()[xzerosubs], other.allsubs()[otherzerosubs]
            )
            zzerosubs = self.allsubs()[xzerosubs][zzerosubsIdx]

            # Find where their nonzeros intersect
            # TODO consider if intersect rows should return 3 args so we don't have to
            #  call it twice
            nzsubsIdx = ttb.tt_intersect_rows(self.subs, other.subs)
            nzsubs = self.subs[nzsubsIdx]
            iother = ttb.tt_intersect_rows(other.subs, self.subs)
            znzsubs = nzsubs[
                (self.vals[nzsubsIdx] == other.vals[iother]).transpose()[0], :
            ]

            return sptensor.from_data(
                np.vstack((zzerosubs, znzsubs)),
                True
                * np.ones((zzerosubs.shape[0] + znzsubs.shape[0])).astype(bool)[
                    :, None
                ],
                self.shape,
            )

        # Case 2b: other is a dense tensor
        if isinstance(other, ttb.tensor):
            # Find where their zeros interact
            otherzerosubs, _ = (other == 0).find()
            zzerosubs = otherzerosubs[
                (self.extract(otherzerosubs) == 0).transpose()[0], :
            ]

            # Find where their nonzeros intersect
            othervals = other[self.subs, "extract"]
            znzsubs = self.subs[(othervals[:, None] == self.vals).transpose()[0], :]

            return sptensor.from_data(
                np.vstack((zzerosubs, znzsubs)),
                True * np.ones((zzerosubs.shape[0] + znzsubs.shape[0])).astype(bool),
                self.shape,
            )

        assert False, "Sptensor == argument must be scalar or sptensor"

    def __ne__(self, other):
        """
        Not equal comparator (~=) for sptensors

        Parameters
        ----------
        other: compare equality of sptensor to other

        Returns
        -------
        :class:`pyttb.sptensor`
        """
        # Case 1: One argument is a scalar
        if isinstance(other, (float, int)):
            if other == 0:
                return ttb.sptensor.from_data(
                    self.subs, True * np.ones((self.subs.shape[0], 1)), self.shape
                )
            subs1 = self.subs[self.vals.transpose()[0] != other, :]
            subs2Idx = ttb.tt_setdiff_rows(self.allsubs(), self.subs)
            subs2 = self.allsubs()[subs2Idx, :]
            return ttb.sptensor.from_data(
                np.vstack((subs1, subs2)),
                True * np.ones((self.subs.shape[0], 1)).astype(bool),
                self.shape,
            )

        # Case 2: Both x and y are tensors or some sort
        # Check that the sizes match
        if self.shape != other.shape:
            assert False, "Size mismatch"
        # Case 2a: Two sparse tensors
        if isinstance(other, ttb.sptensor):
            # find entries where either x *or* y is nonzero, but not both
            # TODO this is a quick alternative to setxor
            nonUniqueSelf = ttb.tt_intersect_rows(self.subs, other.subs)
            selfIdx = True * np.ones(self.subs.shape[0], dtype=bool)
            selfIdx[nonUniqueSelf] = False
            nonUniqueOther = ttb.tt_intersect_rows(other.subs, self.subs)
            otherIdx = True * np.ones(other.subs.shape[0], dtype=bool)
            otherIdx[nonUniqueOther] = False
            subs1 = np.concatenate((self.subs[selfIdx], other.subs[otherIdx]))
            # subs1 = setxor(self.subs, other.subs,'rows')
            # find entries where both are nonzero, but inequal
            subs2 = ttb.tt_intersect_rows(self.subs, other.subs)
            subs_pad = np.zeros((self.shape[0],)).astype(bool)
            subs_pad[subs2] = (
                self.extract(self.subs[subs2]) != other.extract(self.subs[subs2])
            ).transpose()[0]
            subs2 = self.subs[subs_pad, :]
            # put it all together
            return ttb.sptensor.from_data(
                np.vstack((subs1, subs2)),
                True * np.ones((subs1.shape[0] + subs2.shape[0], 1)).astype(bool),
                self.shape,
            )

        # Case 2b: y is a dense tensor
        if isinstance(other, ttb.tensor):
            # find entries where x is zero but y is nonzero
            unionSubs = ttb.tt_union_rows(
                self.subs, np.array(np.where(other.data == 0)).transpose()
            )
            if unionSubs.shape[0] != np.prod(self.shape):
                subs1Idx = ttb.tt_setdiff_rows(self.allsubs(), unionSubs)
                subs1 = self.allsubs()[subs1Idx]
            else:
                subs1 = np.empty((0, self.subs.shape[1]))
            # find entries where x is nonzero but not equal to y
            subs2 = self.subs[
                self.vals.transpose()[0] != other[self.subs, "extract"], :
            ]
            if subs2.size == 0:
                subs2 = np.empty((0, self.subs.shape[1]))
            # put it all together
            return ttb.sptensor.from_data(
                np.vstack((subs1, subs2)),
                True * np.ones((subs1.shape[0] + subs2.shape[0], 1)).astype(bool),
                self.shape,
            )

        # Otherwise
        assert False, "The arguments must be two sptensors or an sptensor and a scalar."

    def __sub__(self, other):
        """
        MINUS Binary subtraction for sparse tensors.

        Parameters
        ----------
        other: :class:`pyttb.tensor`, :class:`pyttb.sptensor`

        Returns
        -------
        :class:`pyttb.sptensor`
        """
        # Case 1: One argument is a scalar
        # Emulating the sparse matrix case here, which creates and returns
        # a dense result, even if the scalar is zero.

        # Case 1: Second argument is a scalar or a dense tensor
        if isinstance(other, (float, int, ttb.tensor)):
            return self.full() - other

        # Case 2: Both are sparse tensors
        if not isinstance(other, ttb.sptensor) or self.shape != other.shape:
            assert False, "Must be two sparse tensors of the same shape"

        return ttb.sptensor.from_aggregator(
            np.vstack((self.subs, other.subs)),
            np.vstack((self.vals, -1 * other.vals)),
            self.shape,
        )

    def __add__(self, other):
        """
        MINUS Binary addition for sparse tensors.

        Parameters
        ----------
        other: :class:`pyttb.tensor`, :class:`pyttb.sptensor`

        Returns
        -------
        :class:`pyttb.sptensor`
        """
        # If other is sumtensor perform sumtensor add
        if isinstance(other, ttb.sumtensor):  # pragma: no cover
            return other.__add__(self)
        # Otherwise return negated sub
        return self.__sub__(-other)

    def __pos__(self):
        """
        Unary plus (+) for sptensors

        Returns
        -------
        :class:`pyttb.sptensor`, copy of tensor
        """

        return ttb.sptensor.from_tensor_type(self)

    def __neg__(self):
        """
        Unary minus (-) for sptensors

        Returns
        -------
        :class:`pyttb.sptensor`, copy of tensor
        """

        return ttb.sptensor.from_data(self.subs, -1 * self.vals, self.shape)

    def __mul__(self, other):
        """
        Element wise multiplication (*) for sptensors

        Parameters
        ----------
        other: :class:`pyttb.sptensor`, :class:`pyttb.tensor`, float, int

        Returns
        -------
        :class:`pyttb.sptensor`
        """
        if isinstance(other, (float, int, np.number)):
            return ttb.sptensor.from_data(self.subs, self.vals * other, self.shape)

        if (
            isinstance(other, (ttb.sptensor, ttb.tensor, ttb.ktensor))
            and self.shape != other.shape
        ):
            assert False, "Sptensor Multiply requires two tensors of the same shape."

        if isinstance(other, ttb.sptensor):
            idxSelf = ttb.tt_intersect_rows(self.subs, other.subs)
            idxOther = ttb.tt_intersect_rows(other.subs, self.subs)
            return ttb.sptensor.from_data(
                self.subs[idxSelf],
                self.vals[idxSelf] * other.vals[idxOther],
                self.shape,
            )
        if isinstance(other, ttb.tensor):
            csubs = self.subs
            cvals = self.vals * other[csubs, "extract"][:, None]
            return ttb.sptensor.from_data(csubs, cvals, self.shape)
        if isinstance(other, ttb.ktensor):
            csubs = self.subs
            cvals = np.zeros(self.vals.shape)
            R = other.weights.size
            N = self.ndims
            for r in range(R):
                tvals = other.weights[r] * self.vals
                for n in range(N):
                    # Note other[n][:, r] extracts 1-D instead of column vector,
                    # which necessitates [:, None]
                    v = other[n][:, r][:, None]
                    tvals = tvals * v[csubs[:, n]]
                cvals += tvals
            return ttb.sptensor.from_data(csubs, cvals, self.shape)
        assert False, "Sptensor cannot be multiplied by that type of object"

    def __rmul__(self, other):
        """
        Element wise right multiplication (*) for sptensors

        Parameters
        ----------
        other: float, int

        Returns
        -------
        :class:`pyttb.sptensor`
        """
        if isinstance(other, (float, int, np.number)):
            return self.__mul__(other)
        assert False, "This object cannot be multiplied by sptensor"

    # pylint:disable=too-many-branches
    def __le__(self, other):
        """
        Less than or equal (<=) for sptensor

        Parameters
        ----------
        other: :class:`pyttb.sptensor`, :class:`pyttb.tensor`, float, int

        Returns
        -------
        :class:`pyttb.sptensor`
        """
        # TODO le,lt,ge,gt have a lot of code duplication, look at generalizing them
        #  for future maintainabilty
        # Case 1: One argument is a scalar
        if isinstance(other, (float, int)):
            subs1 = self.subs[(self.vals <= other).transpose()[0], :]
            if other >= 0:
                subs2 = self.allsubs()[
                    ttb.tt_setdiff_rows(self.allsubs(), self.subs), :
                ]
                subs = np.vstack((subs1, subs2))
            else:
                subs = subs1
            return ttb.sptensor.from_data(
                subs, True * np.ones((len(subs), 1)), self.shape
            )

        # Case 2: Both x and y are tensors of some sort
        # Check that the sizes match
        if (
            isinstance(other, (ttb.sptensor, ttb.tensor, ttb.ktensor))
            and self.shape != other.shape
        ):
            assert False, "Size mismatch"

        # Case 2a: Two sparse tensors
        if isinstance(other, ttb.sptensor):
            # self not zero, other zero
            if self.subs.size > 0:
                subs1 = self.subs[ttb.tt_setdiff_rows(self.subs, other.subs), :]
                if subs1.size > 0:
                    subs1 = subs1[(self.extract(subs1) < 0).transpose()[0], :]
            else:
                subs1 = np.empty(shape=(0, other.subs.shape[1]))

            # self zero, other not zero
            if other.subs.size > 0:
                subs2 = other.subs[ttb.tt_setdiff_rows(other.subs, self.subs), :]
                if subs2.size > 0:
                    subs2 = subs2[(other.extract(subs2) > 0).transpose()[0], :]
            else:
                subs2 = np.empty(shape=(0, self.subs.shape[1]))

            # self and other not zero
            if self.subs.size > 0:
                subs3 = self.subs[ttb.tt_intersect_rows(self.subs, other.subs), :]
                if subs3.size > 0:
                    subs3 = subs3[
                        (self.extract(subs3) <= other.extract(subs3)).transpose()[0], :
                    ]
            else:
                subs3 = np.empty(shape=(0, other.subs.shape[1]))

            # self and other zero
            xzerosubs = self.allsubs()[
                ttb.tt_setdiff_rows(self.allsubs(), self.subs), :
            ]
            yzerosubs = other.allsubs()[
                ttb.tt_setdiff_rows(other.allsubs(), other.subs), :
            ]
            subs4 = xzerosubs[ttb.tt_intersect_rows(xzerosubs, yzerosubs), :]

            # assemble
            subs = np.vstack((subs1, subs2, subs3, subs4))
            return ttb.sptensor.from_data(
                subs, True * np.ones((len(subs), 1)), self.shape
            )

        # Case 2b: One dense tensor
        if isinstance(other, ttb.tensor):
            # self zero
            subs1, _ = (other >= 0).find()
            subs1 = subs1[ttb.tt_setdiff_rows(subs1, self.subs), :]

            # self nonzero
            subs2 = self.subs[
                self.vals.transpose()[0] <= other[self.subs, "extract"], :
            ]

            # assemble
            subs = np.vstack((subs1, subs2))
            return ttb.sptensor.from_data(
                subs, True * np.ones((len(subs), 1)), self.shape
            )

        # Otherwise
        assert False, "Cannot compare sptensor with that type"

    # pylint:disable=too-many-branches
    def __lt__(self, other):
        """
        Less than (<) for sptensor

        Parameters
        ----------
        other: :class:`pyttb.sptensor`, :class:`pyttb.tensor`, float, int

        Returns
        -------
        :class:`pyttb.sptensor`
        """
        # Case 1: One argument is a scalar
        if isinstance(other, (float, int)):
            subs1 = self.subs[(self.vals < other).transpose()[0], :]
            if other > 0:
                subs2 = self.allsubs()[
                    ttb.tt_setdiff_rows(self.allsubs(), self.subs), :
                ]
                subs = np.vstack((subs1, subs2))
            else:
                subs = subs1
            return ttb.sptensor.from_data(
                subs, True * np.ones((len(subs), 1)), self.shape
            )

        # Case 2: Both x and y are tensors of some sort
        # Check that the sizes match
        if (
            isinstance(other, (ttb.sptensor, ttb.tensor, ttb.ktensor))
            and self.shape != other.shape
        ):
            assert False, "Size mismatch"

        # Case 2a: Two sparse tensors
        if isinstance(other, ttb.sptensor):
            # self not zero, other zero
            if self.subs.size > 0:
                subs1 = self.subs[ttb.tt_setdiff_rows(self.subs, other.subs), :]
                if subs1.size > 0:
                    subs1 = subs1[(self.extract(subs1) < 0).transpose()[0], :]
            else:
                subs1 = np.empty(shape=(0, other.subs.shape[1]))

            # self zero, other not zero
            if other.subs.size > 0:
                subs2 = other.subs[ttb.tt_setdiff_rows(other.subs, self.subs), :]
                if subs2.size > 0:
                    subs2 = subs2[(other.extract(subs2) > 0).transpose()[0], :]
            else:
                subs2 = np.empty(shape=(0, self.subs.shape[1]))

            # self and other not zero
            if self.subs.size > 0:
                subs3 = self.subs[ttb.tt_intersect_rows(self.subs, other.subs), :]
                if subs3.size > 0:
                    subs3 = subs3[
                        (self.extract(subs3) < other.extract(subs3)).transpose()[0], :
                    ]
            else:
                subs3 = np.empty(shape=(0, other.subs.shape[1]))

            # assemble
            subs = np.vstack((subs1, subs2, subs3))
            return ttb.sptensor.from_data(
                subs, True * np.ones((len(subs), 1)), self.shape
            )

        # Case 2b: One dense tensor
        if isinstance(other, ttb.tensor):
            # self zero
            subs1, _ = (other > 0).find()
            subs1 = subs1[ttb.tt_setdiff_rows(subs1, self.subs), :]

            # self nonzero
            subs2 = self.subs[self.vals.transpose()[0] < other[self.subs, "extract"], :]

            # assemble
            subs = np.vstack((subs1, subs2))
            return ttb.sptensor.from_data(
                subs, True * np.ones((len(subs), 1)), self.shape
            )

        # Otherwise
        assert False, "Cannot compare sptensor with that type"

    def __ge__(self, other):
        """
        Greater than or equal (>=) to for sptensor

        Parameters
        ----------
        other: :class:`pyttb.sptensor`, :class:`pyttb.tensor`, float, int

        Returns
        -------
        :class:`pyttb.sptensor`
        """
        # Case 1: Argument is a scalar
        if isinstance(other, (float, int)):
            subs1 = self.subs[(self.vals >= other).transpose()[0], :]
            if other <= 0:
                subs2 = ttb.tt_setdiff_rows(self.allsubs(), self.subs)
                subs = np.vstack((subs1, self.allsubs()[subs2]))
            else:
                subs = subs1
            return ttb.sptensor.from_data(
                subs, True * np.ones((len(subs), 1)), self.shape
            )

        # Case 2: Both x and y are tensors of some sort
        # Check that the sizes match
        if (
            isinstance(other, (ttb.sptensor, ttb.tensor, ttb.ktensor))
            and self.shape != other.shape
        ):
            assert False, "Size mismatch"

        # Case 2a: Two sparse tensors
        if isinstance(other, ttb.sptensor):
            return other.__le__(self)

        # Case 2b: One dense tensor
        if isinstance(other, ttb.tensor):
            # self zero
            subs1, _ = (other <= 0).find()
            subs1 = subs1[ttb.tt_setdiff_rows(subs1, self.subs), :]

            # self nonzero
            subs2 = self.subs[
                (self.vals >= other[self.subs, "extract"][:, None]).transpose()[0], :
            ]

            # assemble
            return ttb.sptensor.from_data(
                np.vstack((subs1, subs2)),
                True * np.ones((len(subs1) + len(subs2), 1)),
                self.shape,
            )

        # Otherwise
        assert False, "Cannot compare sptensor with that type"

    def __gt__(self, other):
        """
        Greater than (>) to for sptensor

        Parameters
        ----------
        other: :class:`pyttb.sptensor`, :class:`pyttb.tensor`, float, int

        Returns
        -------
        :class:`pyttb.sptensor`
        """
        # Case 1: Argument is a scalar
        if isinstance(other, (float, int)):
            subs1 = self.subs[(self.vals > other).transpose()[0], :]
            if other < 0:
                subs2 = ttb.tt_setdiff_rows(self.allsubs(), self.subs)
                subs = np.vstack((subs1, self.allsubs()[subs2]))
            else:
                subs = subs1
            return ttb.sptensor.from_data(
                subs, True * np.ones((len(subs), 1)), self.shape
            )

        # Case 2: Both x and y are tensors of some sort
        # Check that the sizes match
        if (
            isinstance(other, (ttb.sptensor, ttb.tensor, ttb.ktensor))
            and self.shape != other.shape
        ):
            assert False, "Size mismatch"

        # Case 2a: Two sparse tensors
        if isinstance(other, ttb.sptensor):
            return other.__lt__(self)

        # Case 2b: One dense tensor
        if isinstance(other, ttb.tensor):
            # self zero and other < 0
            subs1, _ = (other < 0).find()
            if subs1.size > 0:
                subs1 = subs1[ttb.tt_setdiff_rows(subs1, self.subs), :]

            # self and other nonzero
            subs2 = self.subs[
                (self.vals > other[self.subs, "extract"][:, None]).transpose()[0], :
            ]

            # assemble
            return ttb.sptensor.from_data(
                np.vstack((subs1, subs2)),
                True * np.ones((len(subs1) + len(subs2), 1)),
                self.shape,
            )

        # Otherwise
        assert False, "Cannot compare sptensor with that type"

    # pylint:disable=too-many-statements, too-many-branches, too-many-locals
    def __truediv__(self, other):
        """
        Division for sparse tensors (sptensor/other).

        Parameters
        ----------
        other

        Returns
        -------

        """

        # Divide by a scalar -> result is sparse
        if isinstance(other, (float, int)):
            # Inline mrdivide
            newsubs = self.subs
            # We ignore the divide by zero errors because np.inf/np.nan is an
            # appropriate representation
            with np.errstate(divide="ignore", invalid="ignore"):
                newvals = self.vals / other
            if other == 0:
                nansubsidx = ttb.tt_setdiff_rows(self.allsubs(), newsubs)
                nansubs = self.allsubs()[nansubsidx]
                newsubs = np.vstack((newsubs, nansubs))
                newvals = np.vstack((newvals, np.nan * np.ones((nansubs.shape[0], 1))))
            return ttb.sptensor.from_data(newsubs, newvals, self.shape)

        # Tensor divided by a tensor
        if (
            isinstance(other, (ttb.sptensor, ttb.tensor, ttb.ktensor))
            and self.shape != other.shape
        ):
            assert False, "Sptensor division requires tensors of the same shape"

        # Two sparse tensors
        if isinstance(other, ttb.sptensor):
            # Find where their zeros are
            if self.subs.size == 0:
                SelfZeroSubs = self.allsubs()
            else:
                SelfZeroSubsIdx = ttb.tt_setdiff_rows(self.allsubs(), self.subs)
                SelfZeroSubs = self.allsubs()[SelfZeroSubsIdx]
            if other.subs.size == 0:
                OtherZeroSubs = other.allsubs()
            else:
                OtherZeroSubsIdx = ttb.tt_setdiff_rows(other.allsubs(), other.subs)
                OtherZeroSubs = other.allsubs()[OtherZeroSubsIdx]

            # Both nonzero
            if self.subs.size > 0 and other.subs.size > 0:
                idxSelf = ttb.tt_intersect_rows(self.subs, other.subs)
                idxOther = ttb.tt_intersect_rows(other.subs, self.subs)
                newsubs = self.subs[idxSelf, :]
                newvals = self.vals[idxSelf] / other.vals[idxOther]
            else:
                newsubs = np.empty((0, len(self.shape)))
                newvals = np.empty((0, 1))

            # Self nonzero and other zero
            if self.subs.size > 0:
                moresubs = ttb.tt_intersect_rows(self.subs, OtherZeroSubs)
                morevals = np.empty((moresubs.shape[0], 1))
                morevals.fill(np.nan)
                if moresubs.size > 0:
                    newsubs = np.vstack((newsubs, SelfZeroSubs[moresubs, :]))
                    newvals = np.vstack((newvals, morevals))

            # other nonzero and self zero
            if other.subs.size > 0:
                moresubs = ttb.tt_intersect_rows(other.subs, SelfZeroSubs)
                morevals = np.empty((moresubs.shape[0], 1))
                morevals.fill(0)
                if moresubs.size > 0:
                    newsubs = np.vstack((newsubs, OtherZeroSubs[moresubs, :]))
                    newvals = np.vstack((newvals, morevals))

            # Both zero
            moresubs = ttb.tt_intersect_rows(SelfZeroSubs, OtherZeroSubs)
            morevals = np.empty((SelfZeroSubs[moresubs, :].shape[0], 1))
            morevals.fill(np.nan)
            if moresubs.size > 0:
                newsubs = np.vstack((newsubs, SelfZeroSubs[moresubs, :]))
                newvals = np.vstack((newvals, morevals))

            return ttb.sptensor.from_data(newsubs, newvals, self.shape)

        if isinstance(other, ttb.tensor):
            csubs = self.subs
            cvals = self.vals / other[csubs, "extract"][:, None]
            return ttb.sptensor.from_data(csubs, cvals, self.shape)
        if isinstance(other, ttb.ktensor):
            # TODO consider removing epsilon and generating nans consistent with above
            epsilon = np.finfo(float).eps
            subs = self.subs
            vals = np.zeros(self.vals.shape)
            R = (other.weights).size
            N = self.ndims
            for r in range(R):
                tvals = np.ones(((vals).size, 1)).dot(other.weights[r])
                for n in range(N):
                    v = other[n][:, r][:, None]
                    tvals = tvals * v[subs[:, n]]
                vals += tvals
            return ttb.sptensor.from_data(
                self.subs, self.vals / np.maximum(epsilon, vals), self.shape
            )
        assert False, "Invalid arguments for sptensor division"

    def __rtruediv__(self, other):
        """
        Right Division for sparse tensors (other/sptensor).

        Parameters
        ----------
        other

        Returns
        -------

        """
        # Scalar divided by a tensor -> result is dense
        if isinstance(other, (float, int)):
            return other / self.full()
        assert False, "Dividing that object by an sptensor is not supported"

    def __repr__(self):  # pragma: no cover
        """
        String representation of a sparse tensor.

        Returns
        -------
        str
            Contains the shape, subs and vals as strings on different lines.
        """
        nz = self.nnz
        if nz == 0:
            s = "All-zero sparse tensor of shape "
            if self.ndims == 0:
                s += str(self.shape)
                return s
            s += (" x ").join([str(int(d)) for d in self.shape])
            return s

        s = "Sparse tensor of shape "
        s += (" x ").join([str(int(d)) for d in self.shape])
        s += f" with {nz} nonzeros \n"

        # Stop insane printouts
        if nz > 10000:
            r = input("Are you sure you want to print all nonzeros? (Y/N)")
            if r.upper() != "Y":
                return s
        for i in range(0, self.subs.shape[0]):
            s += "\t"
            s += "["
            idx = self.subs[i, :]
            s += str(idx.tolist())[1:]
            s += " = "
            s += str(self.vals[i][0])
            if i < self.subs.shape[0] - 1:
                s += "\n"
        return s

    __str__ = __repr__

    def ttm(
        self,
        matrices: Union[np.ndarray, List[np.ndarray]],
        dims: Optional[Union[float, np.ndarray]] = None,
        transpose: bool = False,
    ):
        """
        Sparse tensor times matrix.

        Parameters
        ----------
        matrices: A matrix or list of matrices
        dims: :class:`Numpy.ndarray`, int
        transpose: Transpose matrices to be multiplied

        Returns
        -------

        """
        if dims is None:
            dims = np.arange(self.ndims)
        elif isinstance(dims, list):
            dims = np.array(dims)
        elif isinstance(dims, (float, int, np.generic)):
            dims = np.array([dims])

        # Handle list of matrices
        if isinstance(matrices, list):
            # Check dimensions are valid
            [dims, vidx] = tt_dimscheck(dims, self.ndims, len(matrices))
            # Calculate individual products
            Y = self.ttm(matrices[vidx[0]], dims[0], transpose=transpose)
            for i in range(1, dims.size):
                Y = Y.ttm(matrices[vidx[i]], dims[i], transpose=transpose)
            return Y

        # Check matrices
        if len(matrices.shape) != 2:
            assert False, "Sptensor.ttm: second argument must be a matrix"

        # Flip matrices if transposed
        if transpose:
            matrices = matrices.transpose()

        # Ensure this is the terminal single dimension case
        if not (dims.size == 1 and np.isin(dims, np.arange(self.ndims))):
            assert False, "dims must contain values in [0,self.dims)"
        final_dim: int = dims[0]

        # Compute the product

        # Check that sizes match
        if self.shape[final_dim] != matrices.shape[1]:
            assert False, "Matrix shape doesn't match tensor shape"

        # Compute the new size
        siz = np.array(self.shape)
        siz[final_dim] = matrices.shape[0]

        # Compute self[mode]'
        Xnt = tt_to_sparse_matrix(self, final_dim, True)

        # Reshape puts the reshaped things after the unchanged modes, transpose then
        # puts it in front
        idx = 0

        # Convert to sparse matrix and do multiplication; generally result is sparse
        Z = Xnt.dot(matrices.transpose())

        # Rearrange back into sparse tensor of correct shape
        Ynt = tt_from_sparse_matrix(Z, siz, final_dim, idx)

        if not isinstance(Z, np.ndarray) and Z.nnz <= 0.5 * np.prod(siz):
            return Ynt
        # TODO evaluate performance loss by casting into sptensor then tensor.
        #  I assume minimal since we are already using spare matrix representation
        return ttb.tensor.from_tensor_type(Ynt)
