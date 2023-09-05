"""Sparse Tensor Implementation"""
# Copyright 2022 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable, Sequence
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
    overload,
)

import numpy as np
import scipy.sparse.linalg
from numpy_groupies import aggregate as accumarray
from scipy import sparse

import pyttb as ttb
from pyttb.pyttb_utils import (
    IndexVariant,
    get_index_variant,
    tt_dimscheck,
    tt_ind2sub,
    tt_intersect_rows,
    tt_intvec2str,
    tt_irenumber,
    tt_ismember_rows,
    tt_renumber,
    tt_setdiff_rows,
    tt_sizecheck,
    tt_sub2ind,
    tt_subscheck,
    tt_subsubsref,
    tt_union_rows,
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
    sptensorInstance:
        Sparse tensor to unwrap
    mode:
        Mode around which to unwrap tensor
    transpose:
        Whether or not to transpose unwrapped tensor

    Returns
    -------
    spmatrix:
        Unwrapped tensor
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
    spmatrix:
    mode:
        Mode around which tensor was unwrapped
    idx:
        in {0,1}, idx of mode in spmatrix, s.b. 0 for tranpose=True

    Returns
    -------
    sptensorInstance: :class:`pyttb.sptensor`
    """
    siz = np.array(shape)
    old = np.setdiff1d(np.arange(len(shape)), mode).astype(int)
    if not isinstance(spmatrix, sparse.coo_matrix):
        spmatrix = sparse.coo_matrix(spmatrix)
    subs = np.vstack((spmatrix.row, spmatrix.col)).transpose()
    vals = spmatrix.data[:, None]
    sptensorInstance = ttb.sptensor(subs, vals, spmatrix.shape, copy=False)

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

    __slots__ = ("subs", "vals", "shape")

    def __init__(
        self,
        subs: Optional[np.ndarray] = None,
        vals: Optional[np.ndarray] = None,
        shape: Optional[Tuple[int, ...]] = None,
        copy=True,
    ):
        """
        Construct an sptensor from fully defined SUB, VAL and SIZE matrices.
        This does no validation to optimize for speed when components are known.
        For default initializer with error checking see
        :func:`~pyttb.sptensor.sptensor.from_aggregator`.

        Parameters
        ----------
        subs:
            Location of non-zero entries
        vals:
            Values for non-zero entries
        shape:
            Shape of sparse tensor
        copy:
            Whether to make a copy of provided data or just reference it

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
        >>> K0 = ttb.sptensor(subs,vals, shape)
        >>> empty_sptensor = ttb.sptensor(shape=shape)
        """

        if subs is None and vals is None:
            # Empty constructor
            self.subs = np.array([], ndmin=2, dtype=int)
            self.vals = np.array([], ndmin=2)
            self.shape: Union[Tuple[()], Tuple[int, ...]] = ()
            if shape is not None:
                if not tt_sizecheck(shape):
                    raise ValueError(f"Invalid shape provided: {shape}")
                self.shape = tuple(shape)
            return
        if subs is None or vals is None or shape is None:
            raise ValueError(
                "For non-empty sptensors subs, vals, and shape must be provided"
            )

        if copy:
            self.subs = subs.copy()
            self.vals = vals.copy()
            self.shape = shape
            return
        self.subs = subs
        self.vals = vals
        self.shape = shape
        return

    @classmethod
    def from_function(
        cls,
        function_handle: Callable[[Tuple[int, ...]], np.ndarray],
        shape: Tuple[int, ...],
        nonzeros: float,
    ) -> sptensor:
        """
        Creates a sparse tensor of the specified shape with NZ nonzeros created from
        the specified function handle

        Parameters
        ----------
        function_handle:
            Function that accepts 2 arguments and generates
            :class:`numpy.ndarray` of length nonzeros
        shape:
            Shape of generated tensor
        nonzeros:
            Number of nonzeros in generated tensor

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
        return cls(subs, vals, shape, copy=False)

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
        subs:
            Location of non-zero entries
        vals:
            Values for non-zero entries
        shape:
            Shape of sparse tensor
        function_handle:
            Aggregation function, or name of supported
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
            if subs.size > 0 and np.max(subs[:, j]) >= dim:
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
        return cls(newsubs, newvals, shape, copy=False)

    def copy(self) -> sptensor:
        """Make a deep copy of a :class:`pyttb.sptensor`.

        Returns
        -------
        Copy of original sptensor.

        Examples
        --------
        >>> first = ttb.sptensor(shape=(2,2))
        >>> first[0,0] = 1
        >>> second = first
        >>> third = second.copy()
        >>> first[0,0] = 3
        >>> first[0,0] == second[0,0]
        True
        >>> first[0,0] == third[0,0]
        False
        """
        return ttb.sptensor(self.subs, self.vals, self.shape, copy=True)

    def __deepcopy__(self, memo):
        return self.copy()

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
            s[:, n] = np.squeeze(ttb.khatrirao(*i))

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
        dims:
            Dimensions to collapse
        fun:
            Method used to collapse dimensions

        Returns
        -------
        Collapsed value

        Example
        -------
        >>> subs = np.array([[1, 2], [1, 3]])
        >>> vals = np.array([[1], [1]])
        >>> shape = np.array([4, 4])
        >>> X = ttb.sptensor(subs, vals, shape)
        >>> X.collapse()
        2
        >>> X.collapse(np.arange(X.ndims), sum)
        2
        """
        if dims is None:
            dims = np.arange(0, self.ndims)

        dims, _ = tt_dimscheck(self.ndims, dims=dims)
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
            return np.zeros((newsize[0],))

        # Create Result
        if self.subs.size > 0:
            return ttb.sptensor.from_aggregator(
                self.subs[:, remdims], self.vals, tuple(newsize), fun
            )
        return ttb.sptensor(np.array([]), np.array([]), tuple(newsize), copy=False)

    def contract(self, i: int, j: int) -> Union[np.ndarray, sptensor, ttb.tensor]:
        """
        Contract tensor along two dimensions (array trace).

        Parameters
        ----------
        i:
            First dimension
        j:
            Second dimension

        Returns
        -------
        Contracted sptensor, converted to tensor if sufficiently dense

        Example
        -------
        >>> X = ttb.tensor(np.ones((2,2)))
        >>> Y = X.to_sptensor()
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
            return y.to_tensor()
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
        function_handle:
            Function that updates all values.

        Returns
        -------
        Updated sptensor

        Example
        -------
        >>> X = ttb.tensor(np.ones((2,2)))
        >>> Y = X.to_sptensor()
        >>> Z = Y.elemfun(lambda values: values*2)
        >>> Z.isequal(Y*2)
        True
        """

        vals = function_handle(self.vals)
        idx = np.where(vals > 0)[0]
        if idx.size == 0:
            return ttb.sptensor(np.array([]), np.array([]), self.shape, copy=False)
        return ttb.sptensor(self.subs[idx, :], vals[idx], self.shape, copy=False)

    def extract(self, searchsubs: np.ndarray) -> np.ndarray:
        """
        Extract value for a sptensor.

        Parameters
        ----------
        searchsubs:
            subscripts to find in sptensor

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
        loc = tt_ismember_rows(searchsubs, self.subs)
        # Fill in the non-zero elements in the answer
        nzsubs = np.where(loc >= 0)
        non_zeros = self.vals[loc[nzsubs]]
        if non_zeros.size > 0:
            a[nzsubs] = non_zeros
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

    def to_tensor(self) -> ttb.tensor:
        """Convenience method to convert to tensor.
        Same as :meth:`pyttb.sptensor.full`
        """
        return self.full()

    def full(self) -> ttb.tensor:
        """
        FULL Convert a sparse tensor to a (dense) tensor.
        """
        # Handle the completely empty (no shape) case
        if len(self.shape) == 0:
            return ttb.tensor()

        # Create a dense zero tensor B that is the same shape as A
        B = ttb.tensor(np.zeros(shape=self.shape), copy=False)

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
        other:
            Other tensor to take innerproduct with
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
            return valsOther.transpose().dot(valsSelf).item()

        if isinstance(other, ttb.tensor):
            if self.shape != other.shape:
                assert False, "Sptensor and tensor must be same shape for innerproduct"
            [subsSelf, valsSelf] = self.find()
            valsOther = other[subsSelf]
            return valsOther.transpose().dot(valsSelf).item()

        if isinstance(other, (ttb.ktensor, ttb.ttensor)):  # pragma: no cover
            # Reverse arguments to call ktensor/ttensor implementation
            return other.innerprod(self)

        assert False, f"Inner product between sptensor and {type(other)} not supported"

    def isequal(self, other: Union[sptensor, ttb.tensor]) -> bool:
        """
        Exact equality for sptensors

        Parameters
        ----------
        other:
            Other tensor to compare against
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
        B:
            Other value to compare with

        Returns
        ----------
        Indicator tensor
        """
        # Case 1: One argument is a scalar
        if isinstance(B, (int, float)):
            if B == 0:
                C = sptensor(shape=self.shape)
            else:
                newvals = self.vals == B
                C = sptensor(self.subs, newvals, self.shape)
            return C
        # Case 2: Argument is a tensor of some sort
        if isinstance(B, sptensor):
            # Check that the shapes match
            if not self.shape == B.shape:
                assert False, "Must be tensors of the same shape"

            C = sptensor.from_aggregator(
                np.vstack((self.subs, B.subs)),
                np.vstack((self.vals, B.vals)),
                self.shape,
                lambda x: len(x) == 2,
            )

            return C

        if isinstance(B, ttb.tensor):
            BB = sptensor(self.subs, B[self.subs][:, None], self.shape)
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
        subsIdx = tt_setdiff_rows(allsubs, self.subs)
        subs = allsubs[subsIdx]
        trueVector = np.ones(shape=(subs.shape[0], 1), dtype=bool)
        return sptensor(subs, trueVector, self.shape)

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
            return sptensor.from_aggregator(
                np.vstack((self.subs, B.subs)),
                np.ones((self.subs.shape[0] + B.subs.shape[0], 1)),
                self.shape,
                lambda x: len(x) >= 1,
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
        other:
            Other value to xor against

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

            subs = np.vstack((self.subs, other.subs))
            return ttb.sptensor.from_aggregator(
                subs, np.ones((len(subs), 1)), self.shape, lambda x: len(x) == 1
            )

        assert False, "The argument must be an sptensor, tensor or scalar"

    def mask(self, W: sptensor) -> np.ndarray:
        """
        Extract values as specified by a mask tensor

        Parameters
        ----------
        W:
            Mask tensor

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
        idx = tt_ismember_rows(wsubs, self.subs)

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
        U:
            Matrices to create the Khatri-Rao product
        n:
            Mode to matricize sptensor in

        Returns
        -------
        Matrix product

        Examples
        --------
        >>> matrix = np.ones((4, 4))
        >>> subs = np.array([[1, 1, 1], [1, 1, 3], [2, 2, 2], [3, 3, 3]])
        >>> vals = np.array([[0.5], [1.5], [2.5], [3.5]])
        >>> shape = (4, 4, 4)
        >>> sptensorInstance = sptensor(subs, vals, shape)
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
            V[:, r] = self.ttv(Z, exclude_dims=n).double()

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
        n:
            Mode to unfold
        r:
            Number of eigenvectors to compute
        flipsign:
            Make each eigenvector's largest element positive
        """
        old = np.setdiff1d(np.arange(self.ndims), n).astype(int)
        # tnt calculation is a workaround for missing sptenmat
        mutatable_sptensor = (
            self.copy().reshape((np.prod(np.array(self.shape)[old]), 1), old).squeeze()
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
            logging.debug(
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
        return ttb.sptensor(self.subs, oneVals, self.shape)

    def permute(self, order: np.ndarray) -> sptensor:
        """
        Rearrange the dimensions of a sparse tensor

        Parameters
        ----------
        order:
            Updated order of dimensions
        """
        # Error check
        if self.ndims != order.size or np.any(
            np.sort(order) != np.arange(0, self.ndims)
        ):
            assert False, "Invalid permutation order"

        # Do the permutation
        if not self.subs.size == 0:
            return ttb.sptensor(
                self.subs[:, order], self.vals, tuple(np.array(self.shape)[order])
            )
        return ttb.sptensor(self.subs, self.vals, tuple(np.array(self.shape)[order]))

    def reshape(
        self,
        new_shape: Tuple[int, ...],
        old_modes: Optional[Union[np.ndarray, int]] = None,
    ) -> sptensor:
        """
        Reshape specified modes of sparse tensor

        Parameters
        ----------
        new_shape:
        old_modes:

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
            return ttb.sptensor(
                np.array([]),
                np.array([]),
                tuple(np.concatenate((keep_shape, new_shape))),
                copy=False,
            )
        if np.isscalar(old_shape):
            old_shape = (old_shape,)
            inds = tt_sub2ind(old_shape, self.subs[:, old_modes][:, None])
        else:
            inds = tt_sub2ind(old_shape, self.subs[:, old_modes])
        new_subs = tt_ind2sub(new_shape, inds)
        return ttb.sptensor(
            np.concatenate((self.subs[:, keep_modes], new_subs), axis=1),
            self.vals,
            tuple(np.concatenate((keep_shape, new_shape))),
        )

    def scale(
        self,
        factor: Union[np.ndarray, ttb.tensor, ttb.sptensor],
        dims: Union[float, np.ndarray],
    ) -> sptensor:
        """
        Scale along specified dimensions for sparse tensors

        Parameters
        ----------
        factor: Scaling factor
        dims: Dimensions to scale

        Returns
        -------
        :class:`pyttb.sptensor`
        """
        if isinstance(dims, (float, int)):
            dims = np.array([dims])
        dims, _ = tt_dimscheck(self.ndims, dims=dims)

        if isinstance(factor, ttb.tensor):
            shapeArray = np.array(self.shape)
            if not np.array_equal(factor.shape, shapeArray[dims]):
                assert False, "Size mismatch in scale"
            return ttb.sptensor(
                self.subs,
                self.vals * factor[self.subs[:, dims]][:, None],
                self.shape,
            )
        if isinstance(factor, ttb.sptensor):
            shapeArray = np.array(self.shape)
            if not np.array_equal(factor.shape, shapeArray[dims]):
                assert False, "Size mismatch in scale"
            return ttb.sptensor(
                self.subs, self.vals * factor.extract(self.subs[:, dims]), self.shape
            )
        if isinstance(factor, np.ndarray):
            shapeArray = np.array(self.shape)
            if factor.shape[0] != shapeArray[dims]:
                assert False, "Size mismatch in scale"
            return ttb.sptensor(
                self.subs,
                self.vals * factor[self.subs[:, dims].transpose()[0]][:, None],
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
            return self.copy()
        idx = np.where(shapeArray > 1)[0]
        if idx.size == 0:
            return self.vals[0].copy()
        siz = tuple(shapeArray[idx])
        if self.vals.size == 0:
            return ttb.sptensor(np.array([]), np.array([]), siz, copy=False)
        return ttb.sptensor(self.subs[:, idx], self.vals, siz)

    def subdims(self, region: Sequence[Union[int, np.ndarray, slice]]) -> np.ndarray:
        """
        SUBDIMS Compute the locations of subscripts within a subdimension.

        Parameters
        ----------
        region:
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
        >>> sp = sptensor(subs,vals,shape)
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

    def ttv(  # noqa: PLR0912
        self,
        vector: Union[np.ndarray, List[np.ndarray]],
        dims: Optional[Union[int, np.ndarray]] = None,
        exclude_dims: Optional[Union[int, np.ndarray]] = None,
    ) -> Union[sptensor, ttb.tensor]:
        """
        Sparse tensor times vector

        Parameters
        ----------
        vector:
            Vector(s) to multiply against
        dims:
            Dimensions to multiply with vector(s)
        exclude_dims:
            Use all dimensions but these
        """

        if dims is None and exclude_dims is None:
            dims = np.array([])
        elif isinstance(dims, (float, int)):
            dims = np.array([dims])

        if isinstance(exclude_dims, (float, int)):
            exclude_dims = np.array([exclude_dims])

        # Check that vector is a list of vectors,
        # if not place single vector as element in list
        if len(vector) > 0 and isinstance(vector[0], (int, float, np.int_, np.float_)):
            return self.ttv(np.array([vector]), dims, exclude_dims)

        # Get sorted dims and index for multiplicands
        dims, vidx = tt_dimscheck(self.ndims, len(vector), dims, exclude_dims)
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
                return ttb.sptensor(shape=tuple(newsiz))
            c = accumarray(
                newsubs.transpose()[0], newvals.transpose()[0], size=newsiz[0]
            )
            if np.count_nonzero(c) <= 0.5 * newsiz:
                return ttb.sptensor.from_aggregator(
                    np.arange(0, newsiz)[:, None], c, tuple(newsiz)
                )
            return ttb.tensor(c, tuple(newsiz), copy=False)

        # Case 2: Result is a multiway array
        c = ttb.sptensor.from_aggregator(newsubs, newvals, tuple(newsiz))

        # Convert to a dense tensor if more than 50% of the result is nonzero.
        if c.nnz > 0.5 * np.prod(newsiz):
            c = c.to_tensor()

        return c

    def __getitem__(self, item):  # noqa: PLR0912, PLR0915
        """
        Subscripted reference for a sparse tensor.

        We can extract elements or subtensors from a sparse tensor in the
        following ways.

        Case 1a: y = X(i1,i2,...,iN), where each in is an index, returns a
        scalar.

        Case 1b: Y = X(R1,R2,...,RN), where one or more Rn is a range and
        the rest are indices, returns a sparse tensor. The elements are
        renumbered here as appropriate.

        Case 2a: V = X(S) where S is a p x n array
        of subscripts, returns a vector of p values.

        Case 2b: V = X(I) where I is a set of p
        linear indices, returns a vector of p values.

        Any ambiguity results in executing the first valid case. This
        is particularily an issue if ndims(X)==1.

        Parameters
        ----------
        item:

        Returns
        -------

        :class:`numpy.ndarray` or :class:`pyttb.sptensor`

        Examples
        --------
        >>> subs = np.array([[3,3,3],[1,1,0],[1,2,1]])
        >>> vals = np.array([3,5,1])
        >>> shape = (4,4,4)
        >>> X = sptensor(subs,vals,shape)
        >>> print(X[0,1,0])
        0
        >>> print(X[3,3,3])
        3
        >>> _ = X[2:3,:,:] #<-- returns 1 x 4 x 4 sptensor
        """
        # TODO IndexError for value outside of indices
        # TODO Key error if item not in container
        # *** CASE 1: Rectangular Subtensor ***
        if isinstance(item, tuple) and len(item) == self.ndims:
            # Extract the subdimensions to be extracted from self
            region = []
            for dim, value in enumerate(item):
                if isinstance(value, (int, np.integer)) and value < 0:
                    value = self.shape[dim] + value  # noqa: PLW2901
                region.append(value)

            # Pare down the list of subscripts (and values) to only
            # those within the subdimensions specified by region.
            loc = self.subdims(region)
            # Handle slicing an sptensor with no entries
            if self.subs.size == 0:
                subs = self.subs.copy()
            else:
                subs = self.subs[loc, :]
            if self.vals.size == 0:
                vals = self.vals.copy()
            else:
                vals = self.vals[loc]

            # Find the size of the subtensor and renumber the
            # subscripts
            [subs, shape] = tt_renumber(subs, self.shape, region)

            # Determine the subscripts
            newsiz = []  # (future) new size
            kpdims = []  # dimensions to keep
            rmdims = []  # dimensions to remove

            # Determine the new size and what dimensions to keep
            for i, a_region in enumerate(region):
                if isinstance(a_region, slice):
                    newsiz.append(self.shape[i])
                    kpdims.append(i)
                elif not isinstance(a_region, (int, float, np.integer)):
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
                    a = np.array(0, dtype=vals.dtype)
                else:
                    a = vals
                if a.size == 1:
                    a = a.item()
                return a

            # Assemble the resulting sparse tensor
            # TODO clean up tuple array cast below
            if subs.size == 0:
                a = sptensor(shape=tuple(np.array(shape)[kpdims]))
            else:
                a = sptensor(subs[:, kpdims], vals, tuple(np.array(shape)[kpdims]))
            return a

        # Case 2:

        # *** CASE 2a: Subscript indexing ***
        if (
            isinstance(item, np.ndarray)
            and len(item.shape) == 2
            and item.shape[1] == self.ndims
        ):
            srchsubs = np.array(item)

        # *** CASE 2b: Linear indexing ***
        else:
            # TODO: Copy pasted from tensor DRY up
            if isinstance(item, (int, float, np.generic, slice)):
                if isinstance(item, (int, float, np.generic)):
                    if item < 0:
                        item = np.prod(self.shape) + item
                    idx = np.array([item])
                elif isinstance(item, slice):
                    idx = np.array(range(np.prod(self.shape))[item])
            elif isinstance(item, np.ndarray) or (
                isinstance(item, list)
                and all(isinstance(element, (int, np.integer)) for element in item)
            ):
                idx = np.array(item)
            else:
                assert False, "Invalid indexing"

            if len(idx.shape) != 1:
                assert False, "Expecting a row index"

            # extract linear indices and convert to subscripts
            srchsubs = tt_ind2sub(self.shape, idx)

        a = self.extract(srchsubs)
        a = tt_subsubsref(a, item)

        return a

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
        >>> X = sptensor(shape=(30, 40, 20)) # <-- Create an empty 30 x 40 x 20 sptensor
        >>> X[29, 39, 19] = 7 # <-- Assign a single element to be 7
        >>> X[np.array([[1,1,1], [2,2,2]])] = 1 # <-- Assign a list of elements
        >>> X[11:20,11:20,11:20] = ttb.sptenrand((10,10,10),nonzeros=10)
        >>> X[31,41,21] = 7 # <-- grows the size of the tensor
        >>> # Grow tensor
        >>> X[111:120,111:120,111:120] = ttb.sptenrand((10,10,10),nonzeros=10)
        >>> X[1,1,1,1] = 4 # <-- increases the number of dimensions from 3 to 4

        >>> X = ttb.sptensor(shape=(30,)) # <-- empty one-dimensional tensor
        >>> X[4:6] = 1 # <-- set subtensor to ones (does not increase dimension)
        >>> X[np.array([[10], [12], [14]])] = np.array([[5], [6], [7]])
        >>> X[31] = 7 # <-- grow the ONLY dimension
        >>> X[1,1] = 0 # <-- add a dimension, but no nonzeros

        Note regarding singleton dimensions: It is not possible to do, for
        instance, X(1,1:10,1:10) = sptenrand([1 10 10],5). However, it is okay
        to do X(1,1:10,1:10) = squeeze(sptenrand([1 10 10],5)).

        Parameters
        ----------
        key:
        value:

        """
        # TODO IndexError for value outside of indices
        # TODO Key error if item not in container
        # If empty sptensor and assignment is empty list or empty nparray
        if self.vals.size == 0 and (
            (isinstance(value, np.ndarray) and value.size == 0)
            or (isinstance(value, list) and value == [])
        ):
            return None

        access_type = get_index_variant(key)

        # Case 1: Replace a sub-tensor
        if access_type == IndexVariant.SUBTENSOR:
            updated_key = []
            for dim, entry in enumerate(key):
                if isinstance(entry, (int, np.integer)) and entry < 0:
                    entry = self.shape[dim] + entry  # noqa: PLW2901
                updated_key.append(entry)
            return self._set_subtensor(updated_key, value)
        # Case 2: Subscripts
        if access_type == IndexVariant.SUBSCRIPTS:
            return self._set_subscripts(key, value)
        if access_type == IndexVariant.LINEAR and len(self.shape) == 1:
            if isinstance(key, slice):
                key = np.arange(0, self.shape[0])[key, None]
            else:
                key = np.array([[key]])
            return self._set_subscripts(key, value)
        raise ValueError("Unknown assignment type")  # pragma: no cover

    def _set_subscripts(self, key, value):  # noqa: PLR0912
        # Case II: Replacing values at specific indices
        newsubs = key
        tt_subscheck(newsubs, nargout=False)

        # Error check on subscripts
        if newsubs.shape[1] < self.ndims:
            assert False, "Invalid subscripts"

        # Check for expanding the order
        if newsubs.shape[1] > self.ndims:
            newshape = list(self.shape)
            grow_size = newsubs.shape[1] - self.ndims
            newshape.extend([1] * grow_size)
            if self.subs.size > 0:
                self.subs = np.concatenate(
                    (
                        self.subs,
                        np.ones(
                            (self.subs.shape[0], grow_size),
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
        tf = tt_ismember_rows(newsubs, self.subs)
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
            if self.subs.size > 0:
                self.subs = np.vstack((self.subs, newsubs[idxc, :]))
                self.vals = np.vstack((self.vals, newvals[idxc]))
            else:
                self.subs = newsubs[idxc, :]
                self.vals = newvals[idxc]

        # Resize the tensor
        newshape = []
        for n, dim in enumerate(self.shape):
            smax = max(newsubs[:, n] + 1)
            newshape.append(max(dim, smax))
        self.shape = tuple(newshape)

    def _set_subtensor(self, key, value):  # noqa: PLR0912, PLR0915
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
                    elif key_n.stop is None:
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
            addsubs = tt_irenumber(value, self.shape, key)
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
            elif isinstance(key[n], Iterable):
                newsz.append(max([self.shape[n], max(key[n]) + 1]))
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
            elif isinstance(key[n], (np.ndarray, Iterable)):
                newsz.append(max(key[n]) + 1)
            else:
                newsz.append(key[n] + 1)
        self.shape = tuple(newsz)

        # Expand subs array if there are new modes, i.e. if the order has increased
        if self.subs.size > 0 and len(self.shape) > self.subs.shape[1]:
            self.subs = np.append(
                self.subs,
                np.zeros(
                    shape=(self.subs.shape[0], len(self.shape) - self.subs.shape[1]),
                    dtype=int,
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
            keyCopy = [None] * N
            # Figure out how many indices are in each dimension
            nssubs = np.zeros((N, 1))
            for n in range(0, N):
                if isinstance(key[n], slice):
                    # Generate slice explicitly to determine its length
                    keyCopy[n] = np.arange(0, self.shape[n])[key[n]]
                    indicesInN = len(keyCopy[n])
                elif isinstance(key[n], Iterable):
                    keyCopy[n] = key[n]
                    indicesInN = len(key[n])
                else:
                    keyCopy[n] = key[n]
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
                addsubs[:, n] = ttb.khatrirao(*i).transpose()[:]

            if self.subs.size > 0:
                # Replace existing values
                loc = tt_intersect_rows(self.subs, addsubs)
                self.vals[loc] = value
                # pare down list of subscripts to add
                addsubs = addsubs[tt_setdiff_rows(addsubs, self.subs)]

            # If there are things to insert then insert them
            if addsubs.size > 0:
                if self.subs.size > 0:
                    self.subs = np.vstack((self.subs, addsubs.astype(int)))
                    self.vals = np.vstack(
                        (self.vals, value * np.ones((addsubs.shape[0], 1)))
                    )
                else:
                    self.subs = addsubs.astype(int)
                    self.vals = value * np.ones((addsubs.shape[0], 1))
            return

        assert False, "Invalid assignment value"

    def __eq__(self, other):
        """
        Equal comparator for sptensors

        Parameters
        ----------
        other:
            Compare equality of sptensor to other

        Returns
        -------
        :class:`pyttb.sptensor`
        """
        # Case 1: other is a scalar
        if isinstance(other, (float, int)):
            if other == 0:
                return self.logical_not()
            idx = self.vals == other
            return sptensor(
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
            xzerosubs = tt_setdiff_rows(self.allsubs(), self.subs)
            otherzerosubs = tt_setdiff_rows(other.allsubs(), other.subs)
            # zzerosubs = np.isin(xzerosubs, otherzerosubs)
            zzerosubsIdx = tt_intersect_rows(
                self.allsubs()[xzerosubs], other.allsubs()[otherzerosubs]
            )
            zzerosubs = self.allsubs()[xzerosubs][zzerosubsIdx]

            # Find where their nonzeros intersect
            # TODO consider if intersect rows should return 3 args so we don't have to
            #  call it twice
            nzsubsIdx = tt_intersect_rows(self.subs, other.subs)
            nzsubs = self.subs[nzsubsIdx]
            iother = tt_intersect_rows(other.subs, self.subs)
            znzsubs = nzsubs[
                (self.vals[nzsubsIdx] == other.vals[iother]).transpose()[0], :
            ]

            return sptensor(
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
            othervals = other[self.subs]
            znzsubs = self.subs[(othervals[:, None] == self.vals).transpose()[0], :]

            return sptensor(
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
        other:
            Compare equality of sptensor to other

        Returns
        -------
        :class:`pyttb.sptensor`
        """
        # Case 1: One argument is a scalar
        if isinstance(other, (float, int)):
            if other == 0:
                return ttb.sptensor(
                    self.subs, True * np.ones((self.subs.shape[0], 1)), self.shape
                )
            subs1 = self.subs[self.vals.transpose()[0] != other, :]
            subs2Idx = tt_setdiff_rows(self.allsubs(), self.subs)
            subs2 = self.allsubs()[subs2Idx, :]
            return ttb.sptensor(
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
            nonUniqueSelf = tt_intersect_rows(self.subs, other.subs)
            selfIdx = True * np.ones(self.subs.shape[0], dtype=bool)
            selfIdx[nonUniqueSelf] = False
            nonUniqueOther = tt_intersect_rows(other.subs, self.subs)
            otherIdx = True * np.ones(other.subs.shape[0], dtype=bool)
            otherIdx[nonUniqueOther] = False
            subs1 = np.concatenate((self.subs[selfIdx], other.subs[otherIdx]))
            # subs1 = setxor(self.subs, other.subs,'rows')
            # find entries where both are nonzero, but inequal
            subs2 = tt_intersect_rows(self.subs, other.subs)
            subs_pad = np.zeros((self.shape[0],)).astype(bool)
            subs_pad[subs2] = (
                self.extract(self.subs[subs2]) != other.extract(self.subs[subs2])
            ).transpose()[0]
            subs2 = self.subs[subs_pad, :]
            # put it all together
            return ttb.sptensor(
                np.vstack((subs1, subs2)),
                True * np.ones((subs1.shape[0] + subs2.shape[0], 1)).astype(bool),
                self.shape,
            )

        # Case 2b: y is a dense tensor
        if isinstance(other, ttb.tensor):
            # find entries where x is zero but y is nonzero
            unionSubs = tt_union_rows(
                self.subs, np.array(np.where(other.data == 0)).transpose()
            )
            if unionSubs.shape[0] != np.prod(self.shape):
                subs1Idx = tt_setdiff_rows(self.allsubs(), unionSubs)
                subs1 = self.allsubs()[subs1Idx]
            else:
                subs1 = np.empty((0, self.subs.shape[1]))
            # find entries where x is nonzero but not equal to y
            subs2 = self.subs[self.vals.transpose()[0] != other[self.subs], :]
            if subs2.size == 0:
                subs2 = np.empty((0, self.subs.shape[1]))
            # put it all together
            return ttb.sptensor(
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

        return self.copy()

    def __neg__(self):
        """
        Unary minus (-) for sptensors

        Returns
        -------
        :class:`pyttb.sptensor`, copy of tensor
        """

        return ttb.sptensor(self.subs, -1 * self.vals, self.shape)

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
            return ttb.sptensor(self.subs, self.vals * other, self.shape)

        if (
            isinstance(other, (ttb.sptensor, ttb.tensor, ttb.ktensor))
            and self.shape != other.shape
        ):
            assert False, "Sptensor Multiply requires two tensors of the same shape."

        if isinstance(other, ttb.sptensor):
            idxSelf = tt_intersect_rows(self.subs, other.subs)
            idxOther = tt_intersect_rows(other.subs, self.subs)
            return ttb.sptensor(
                self.subs[idxSelf],
                self.vals[idxSelf] * other.vals[idxOther],
                self.shape,
            )
        if isinstance(other, ttb.tensor):
            csubs = self.subs
            cvals = self.vals * other[csubs][:, None]
            return ttb.sptensor(csubs, cvals, self.shape)
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
                    v = other.factor_matrices[n][:, r][:, None]
                    tvals = tvals * v[csubs[:, n]]
                cvals += tvals
            return ttb.sptensor(csubs, cvals, self.shape)
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

    def __le__(self, other):  # noqa: PLR0912
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
                subs2 = self.allsubs()[tt_setdiff_rows(self.allsubs(), self.subs), :]
                subs = np.vstack((subs1, subs2))
            else:
                subs = subs1
            return ttb.sptensor(subs, True * np.ones((len(subs), 1)), self.shape)

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
                subs1 = self.subs[tt_setdiff_rows(self.subs, other.subs), :]
                if subs1.size > 0:
                    subs1 = subs1[(self.extract(subs1) < 0).transpose()[0], :]
            else:
                subs1 = np.empty(shape=(0, other.subs.shape[1]))

            # self zero, other not zero
            if other.subs.size > 0:
                subs2 = other.subs[tt_setdiff_rows(other.subs, self.subs), :]
                if subs2.size > 0:
                    subs2 = subs2[(other.extract(subs2) > 0).transpose()[0], :]
            else:
                subs2 = np.empty(shape=(0, self.subs.shape[1]))

            # self and other not zero
            if self.subs.size > 0:
                subs3 = self.subs[tt_intersect_rows(self.subs, other.subs), :]
                if subs3.size > 0:
                    subs3 = subs3[
                        (self.extract(subs3) <= other.extract(subs3)).transpose()[0], :
                    ]
            else:
                subs3 = np.empty(shape=(0, other.subs.shape[1]))

            # self and other zero
            xzerosubs = self.allsubs()[tt_setdiff_rows(self.allsubs(), self.subs), :]
            yzerosubs = other.allsubs()[tt_setdiff_rows(other.allsubs(), other.subs), :]
            subs4 = xzerosubs[tt_intersect_rows(xzerosubs, yzerosubs), :]

            # assemble
            subs = np.vstack((subs1, subs2, subs3, subs4))
            return ttb.sptensor(subs, True * np.ones((len(subs), 1)), self.shape)

        # Case 2b: One dense tensor
        if isinstance(other, ttb.tensor):
            # self zero
            subs1, _ = (other >= 0).find()
            subs1 = subs1[tt_setdiff_rows(subs1, self.subs), :]

            # self nonzero
            subs2 = self.subs[self.vals.transpose()[0] <= other[self.subs], :]

            # assemble
            subs = np.vstack((subs1, subs2))
            return ttb.sptensor(subs, True * np.ones((len(subs), 1)), self.shape)

        # Otherwise
        assert False, "Cannot compare sptensor with that type"

    def __lt__(self, other):  # noqa: PLR0912
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
                subs2 = self.allsubs()[tt_setdiff_rows(self.allsubs(), self.subs), :]
                subs = np.vstack((subs1, subs2))
            else:
                subs = subs1
            return ttb.sptensor(subs, True * np.ones((len(subs), 1)), self.shape)

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
                subs1 = self.subs[tt_setdiff_rows(self.subs, other.subs), :]
                if subs1.size > 0:
                    subs1 = subs1[(self.extract(subs1) < 0).transpose()[0], :]
            else:
                subs1 = np.empty(shape=(0, other.subs.shape[1]))

            # self zero, other not zero
            if other.subs.size > 0:
                subs2 = other.subs[tt_setdiff_rows(other.subs, self.subs), :]
                if subs2.size > 0:
                    subs2 = subs2[(other.extract(subs2) > 0).transpose()[0], :]
            else:
                subs2 = np.empty(shape=(0, self.subs.shape[1]))

            # self and other not zero
            if self.subs.size > 0:
                subs3 = self.subs[tt_intersect_rows(self.subs, other.subs), :]
                if subs3.size > 0:
                    subs3 = subs3[
                        (self.extract(subs3) < other.extract(subs3)).transpose()[0], :
                    ]
            else:
                subs3 = np.empty(shape=(0, other.subs.shape[1]))

            # assemble
            subs = np.vstack((subs1, subs2, subs3))
            return ttb.sptensor(subs, True * np.ones((len(subs), 1)), self.shape)

        # Case 2b: One dense tensor
        if isinstance(other, ttb.tensor):
            # self zero
            subs1, _ = (other > 0).find()
            subs1 = subs1[tt_setdiff_rows(subs1, self.subs), :]

            # self nonzero
            subs2 = self.subs[self.vals.transpose()[0] < other[self.subs], :]

            # assemble
            subs = np.vstack((subs1, subs2))
            return ttb.sptensor(subs, True * np.ones((len(subs), 1)), self.shape)

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
                subs2 = tt_setdiff_rows(self.allsubs(), self.subs)
                subs = np.vstack((subs1, self.allsubs()[subs2]))
            else:
                subs = subs1
            return ttb.sptensor(subs, True * np.ones((len(subs), 1)), self.shape)

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
            subs1 = subs1[tt_setdiff_rows(subs1, self.subs), :]

            # self nonzero
            subs2 = self.subs[
                (self.vals >= other[self.subs][:, None]).transpose()[0],
                :,
            ]

            # assemble
            return ttb.sptensor(
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
                subs2 = tt_setdiff_rows(self.allsubs(), self.subs)
                subs = np.vstack((subs1, self.allsubs()[subs2]))
            else:
                subs = subs1
            return ttb.sptensor(subs, True * np.ones((len(subs), 1)), self.shape)

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
                subs1 = subs1[tt_setdiff_rows(subs1, self.subs), :]

            # self and other nonzero
            subs2 = self.subs[
                (self.vals > other[self.subs][:, None]).transpose()[0],
                :,
            ]

            # assemble
            return ttb.sptensor(
                np.vstack((subs1, subs2)),
                True * np.ones((len(subs1) + len(subs2), 1)),
                self.shape,
            )

        # Otherwise
        assert False, "Cannot compare sptensor with that type"

    def __truediv__(self, other):  # noqa: PLR0912, PLR0915
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
                nansubsidx = tt_setdiff_rows(self.allsubs(), newsubs)
                nansubs = self.allsubs()[nansubsidx]
                newsubs = np.vstack((newsubs, nansubs))
                newvals = np.vstack((newvals, np.nan * np.ones((nansubs.shape[0], 1))))
            return ttb.sptensor(newsubs, newvals, self.shape)

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
                SelfZeroSubsIdx = tt_setdiff_rows(self.allsubs(), self.subs)
                SelfZeroSubs = self.allsubs()[SelfZeroSubsIdx]
            if other.subs.size == 0:
                OtherZeroSubs = other.allsubs()
            else:
                OtherZeroSubsIdx = tt_setdiff_rows(other.allsubs(), other.subs)
                OtherZeroSubs = other.allsubs()[OtherZeroSubsIdx]

            # Both nonzero
            if self.subs.size > 0 and other.subs.size > 0:
                idxSelf = tt_intersect_rows(self.subs, other.subs)
                idxOther = tt_intersect_rows(other.subs, self.subs)
                newsubs = self.subs[idxSelf, :]
                newvals = self.vals[idxSelf] / other.vals[idxOther]
            else:
                newsubs = np.empty((0, len(self.shape)))
                newvals = np.empty((0, 1))

            # Self nonzero and other zero
            if self.subs.size > 0:
                moresubs = tt_intersect_rows(self.subs, OtherZeroSubs)
                morevals = np.empty((moresubs.shape[0], 1))
                morevals.fill(np.nan)
                if moresubs.size > 0:
                    newsubs = np.vstack((newsubs, SelfZeroSubs[moresubs, :]))
                    newvals = np.vstack((newvals, morevals))

            # other nonzero and self zero
            if other.subs.size > 0:
                moresubs = tt_intersect_rows(other.subs, SelfZeroSubs)
                morevals = np.empty((moresubs.shape[0], 1))
                morevals.fill(0)
                if moresubs.size > 0:
                    newsubs = np.vstack((newsubs, OtherZeroSubs[moresubs, :]))
                    newvals = np.vstack((newvals, morevals))

            # Both zero
            moresubs = tt_intersect_rows(SelfZeroSubs, OtherZeroSubs)
            morevals = np.empty((SelfZeroSubs[moresubs, :].shape[0], 1))
            morevals.fill(np.nan)
            if moresubs.size > 0:
                newsubs = np.vstack((newsubs, SelfZeroSubs[moresubs, :]))
                newvals = np.vstack((newvals, morevals))

            return ttb.sptensor(newsubs, newvals, self.shape)

        if isinstance(other, ttb.tensor):
            csubs = self.subs
            cvals = self.vals / other[csubs][:, None]
            return ttb.sptensor(csubs, cvals, self.shape)
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
                    v = other.factor_matrices[n][:, r][:, None]
                    tvals = tvals * v[subs[:, n]]
                vals += tvals
            return ttb.sptensor(
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

        s = f"Sparse tensor of shape {self.shape}"
        s += f" with {nz} nonzeros\n"

        # Stop insane printouts
        if nz > 10000:
            r = input("Are you sure you want to print all nonzeros? (Y/N)")
            if r.upper() != "Y":
                return s
        for i in range(0, self.subs.shape[0]):
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
        exclude_dims: Optional[Union[float, np.ndarray]] = None,
        transpose: bool = False,
    ) -> Union[ttb.tensor, sptensor]:
        """
        Sparse tensor times matrix.

        Parameters
        ----------
        matrices:
            A matrix or list of matrices
        dims:
            Dimensions to multiply against
        exclude_dims:
            Use all dimensions but these
        transpose:
            Transpose matrices to be multiplied

        Returns
        -------

        """
        if dims is None and exclude_dims is None:
            dims = np.arange(self.ndims)
        elif isinstance(dims, list):
            dims = np.array(dims)
        elif isinstance(dims, (float, int, np.generic)):
            dims = np.array([dims])

        if isinstance(exclude_dims, (float, int)):
            exclude_dims = np.array([exclude_dims])

        # Handle list of matrices
        if isinstance(matrices, list):
            # Check dimensions are valid
            [dims, vidx] = tt_dimscheck(self.ndims, len(matrices), dims, exclude_dims)
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

        # FIXME: This made typing happy but shouldn't be possible
        if not isinstance(dims, np.ndarray):  # pragma: no cover
            raise ValueError("Dims should be an array here")

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
        return Ynt.to_tensor()

    @overload
    def squash(self, return_inverse: Literal[False]) -> sptensor:
        ...  # pragma: no cover see coveragepy/issues/970

    @overload
    def squash(self, return_inverse: Literal[True]) -> Tuple[sptensor, Dict]:
        ...  # pragma: no cover see coveragepy/issues/970

    def squash(
        self, return_inverse: bool = False
    ) -> Union[sptensor, Tuple[sptensor, Dict]]:
        """
        Remove empty slices from a sparse tensor.

        Parameters
        ----------
        return_inverse:
            Return mapping from new tensor to old tensor subscripts.

        Examples
        --------
        >>> X = ttb.sptenrand((2, 2, 2), nonzeros=3)
        >>> Y = X.squash()
        >>> Y, inverse = X.squash(True)
        >>> np.array_equal(X.subs[:, 0], inverse[0][Y.subs[:, 0]])
        True

        Returns
        -------
        Copy of current sparse tensor with empty slices removed.
        """
        ndims = self.ndims
        subs = np.zeros(self.subs.shape, dtype=int)
        shape = []
        idx_map = {}
        for n in range(ndims):
            unique_subs, inverse = np.unique(self.subs[:, n], return_inverse=True)
            subs[:, n] = inverse
            shape.append(len(subs))
            idx_map[n] = unique_subs
        squashed_tensor = sptensor(subs, self.vals, tuple(shape))
        if return_inverse:
            return squashed_tensor, idx_map
        return squashed_tensor


def sptenrand(
    shape: Tuple[int, ...],
    density: Optional[float] = None,
    nonzeros: Optional[float] = None,
) -> sptensor:
    """
    Create sptensor with entries drawn from a uniform distribution on the unit interval

    Parameters
    ----------
    shape:
        Shape of resulting tensor
    density:
        Density of resulting sparse tensor
    nonzeros:
        Number of nonzero entries in resulting sparse tensor

    Returns
    -------
    Constructed tensor

    Example
    -------
    >>> X = ttb.sptenrand((2,2), nonzeros=1)
    >>> Y = ttb.sptenrand((2,2), density=0.25)
    """
    if density is None and nonzeros is None:
        raise ValueError("Must set either density or nonzeros")

    if density is not None and nonzeros is not None:
        raise ValueError("Must set either density or nonzeros but not both")

    if density is not None and not 0 < density <= 1:
        raise ValueError(f"Density must be a fraction (0, 1] but received {density}")

    if isinstance(density, float):
        valid_nonzeros = float(np.prod(shape) * density)
    elif isinstance(nonzeros, (int, float)):
        valid_nonzeros = nonzeros
    else:  # pragma: no cover
        raise ValueError(
            f"Incorrect types for density:{density} and nonzeros:{nonzeros}"
        )

    # Typing doesn't play nice with partial
    # mypy issue: 1484
    def unit_uniform(pass_through_shape: Tuple[int, ...]) -> np.ndarray:
        return np.random.uniform(low=0, high=1, size=pass_through_shape)

    return ttb.sptensor.from_function(unit_uniform, shape, valid_nonzeros)


def sptendiag(
    elements: np.ndarray, shape: Optional[Tuple[int, ...]] = None
) -> sptensor:
    """
    Creates a sparse tensor with elements along super diagonal
    If provided shape is too small the tensor will be enlarged to accomodate

    Parameters
    ----------
    elements:
        Elements to set along the diagonal
    shape:
        Shape of resulting tensor

    Returns
    -------
    Constructed tensor

    Example
    -------
    >>> shape = (2,)
    >>> values = np.ones(shape)
    >>> X = ttb.sptendiag(values)
    >>> Y = ttb.sptendiag(values, (2, 2))
    >>> X.isequal(Y)
    True
    """
    # Flatten provided elements
    elements = np.ravel(elements)
    N = len(elements)
    if shape is None:
        constructed_shape = (N,) * N
    else:
        constructed_shape = tuple(max(N, dim) for dim in shape)
    subs = np.tile(np.arange(0, N).transpose(), (len(constructed_shape), 1)).transpose()
    return sptensor.from_aggregator(subs, elements.reshape((N, 1)), constructed_shape)
