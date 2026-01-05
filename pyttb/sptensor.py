"""Classes and functions for working with sparse tensors."""

# Copyright 2025 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Iterable, Sequence
from math import prod
from operator import ge, gt, le, lt
from typing import (
    Any,
    Literal,
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
    OneDArray,
    Shape,
    gather_wrap_dims,
    get_index_variant,
    get_mttkrp_factors,
    np_to_python,
    parse_one_d,
    parse_shape,
    to_memory_order,
    tt_dimscheck,
    tt_ind2sub,
    tt_intersect_rows,
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


class sptensor:  # noqa: PLW1641
    """
    SPTENSOR Class for sparse tensors.

    Contains the following data members:

    ``subs``: subscripts of the nonzero values in the sparse tensor.
    Row `k` specifies the subscripts of the `k`-th value in `vals`.

    ``vals``: data elements of the sparese tensor.

    ``shape``: sizes of the dimensions of the sparse tensor.

    Instances of :class:`pyttb.sptensor` can be created using :meth:`__init__` or
    the following methods:

      * :meth:`from_function`
      * :meth:`from_aggregator`

    Examples
    --------
    For all examples listed below, the following module imports are assumed:

    >>> import pyttb as ttb
    >>> import numpy as np
    """

    __slots__ = ("shape", "subs", "vals")

    def __init__(
        self,
        subs: np.ndarray | None = None,
        vals: np.ndarray | None = None,
        shape: Shape | None = None,
        copy: bool = True,
    ):
        """Construct a :class:`pyttb.sptensor`.

        Constructed from a set of `subs` (subscripts),
        `vals` (values), and `shape`. No validation is performed. For
        initializer with error checking see :meth:`from_aggregator`.

        Parameters
        ----------
        subs:
            Subscripts of nonzero entries.
        vals:
            Values for nonzero entries.
        shape:
            Shape of sparse tensor.
        copy:
            Whether to make a copy of provided data or just reference it.

        Examples
        --------
        Create an empty :class:`pyttb.sptensor`:

        >>> shape = (4, 4, 4)
        >>> S = ttb.sptensor(shape=shape)
        >>> S
        empty sparse tensor of shape (4, 4, 4) with order F

        Create a :class:`pyttb.sptensor` from subscripts and values:

        >>> subs = np.array([[1, 2, 1], [1, 3, 1]])
        >>> vals = np.array([[6], [7]])
        >>> S = ttb.sptensor(subs, vals, shape)
        >>> S
        sparse tensor of shape (4, 4, 4) with 2 nonzeros and order F
        [1, 2, 1] = 6
        [1, 3, 1] = 7
        """
        if subs is None and vals is None:
            # Empty constructor
            self.subs = np.array([], ndmin=2, dtype=int)
            self.vals = np.array([], ndmin=2)
            self.shape: tuple[()] | tuple[int, ...] = ()
            if shape is not None:
                shape = parse_shape(shape)
                # TODO do we need sizecheck or should that wrap in our share parser?
                if not tt_sizecheck(shape):
                    raise ValueError(f"Invalid shape provided: {shape}")
                self.shape = tuple(shape)
            return
        if subs is None or vals is None:
            raise ValueError("If subs or vals are provided they must both be provided.")

        if shape is None:
            shape = parse_shape(np.max(subs, axis=0) + 1)
        else:
            shape = parse_shape(shape)

        if subs.size > 0:
            assert subs.shape[1] == len(shape) and np.all(
                (np.max(subs, axis=0) + 1) <= shape
            ), (
                f"Shape provided was incorrect to fit all subscripts; "
                f"max subscripts are "
                f"{tuple(np.max(subs, axis=0) + 1)}"
            )
        else:
            # In case user provides an empty array in weird format
            subs = np.array([], ndmin=2, dtype=int)

        if vals.size == 0:
            # In case user provides an empty array in weird format
            vals = np.array([], dtype=vals.dtype, ndmin=2)
        elif len(vals.shape) == 1:
            # Enforce column array
            vals = vals.reshape((vals.shape[0], 1))
        elif len(vals.shape) > 2:
            raise ValueError("Values should be a column vector")

        if copy:
            self.subs = subs.copy("K")
            self.vals = vals.copy("K")
            self.shape = shape
            return
        self.subs = subs
        self.vals = vals
        self.shape = shape
        return

    @classmethod
    def from_function(
        cls,
        function_handle: Callable[[tuple[int, ...]], np.ndarray],
        shape: Shape,
        nonzeros: float,
    ) -> sptensor:
        """Construct a :class:`pyttb.sptensor`.

        Constructed with nonzeros set using a
        function. The subscripts of the nonzero elements of the sparse tensor
        are generated randomly using `numpy`, so calling `numpy.random.seed()`
        before using this method will provide reproducible results.

        Parameters
        ----------
        function_handle:
            A function that can accept a shape (i.e., :class:`tuple` of
            dimension sizes) and return a :class:`numpy.ndarray` of that shape.
            Example functions include `numpy.random.random_sample`,
            `numpy.zeros`, and `numpy.ones`.
        shape:
            Shape of generated sparse tensor.
        nonzeros:
            Number of nonzeros in generated sparse tensor.

        Examples
        --------
        Create a :class:`pyttb.sptensor` with entries taken from a uniform
        random distribution:

        >>> np.random.seed(1)
        >>> S = ttb.sptensor.from_function(np.random.random_sample, (2, 3, 4), 5)
        >>> print(S)  # doctest: +ELLIPSIS
        sparse tensor of shape (2, 3, 4) with 5 nonzeros and order F
        [0, 1, 3] = 0.4478...
        [0, 2, 0] = 0.9085...
        [1, 2, 0] = 0.2936...
        [1, 2, 1] = 0.2877...
        [1, 2, 2] = 0.1300...

        Create a :class:`pyttb.sptensor` with entries equal to 1:

        >>> np.random.seed(1)
        >>> S = ttb.sptensor.from_function(np.ones, (2, 3, 4), 5)
        >>> print(S)
        sparse tensor of shape (2, 3, 4) with 5 nonzeros and order F
        [0, 1, 3] = 1.0
        [0, 2, 0] = 1.0
        [1, 2, 0] = 1.0
        [1, 2, 1] = 1.0
        [1, 2, 2] = 1.0
        """
        assert callable(function_handle), "function_handle must be callable"

        shape = parse_shape(shape)
        if (nonzeros < 0) or (nonzeros >= prod(shape)):
            assert False, (
                "Requested number of nonzeros must be positive "
                "and less than the total size"
            )
        elif nonzeros < 1:
            nonzeros = int(np.ceil(prod(shape) * nonzeros))
        else:
            nonzeros = int(np.floor(nonzeros))
        nonzeros = int(nonzeros)

        # Keep iterating until we find enough unique nonzeros or we give up
        subs = np.array([])
        cnt = 0
        while (len(subs) < nonzeros) and (cnt < 10):
            subs = (
                np.random.uniform(size=[nonzeros, len(shape)]).dot(np.diag(shape))
            ).astype(int)
            subs = np.unique(subs, axis=0)
            cnt += 1

        nonzeros = int(min(nonzeros, subs.shape[0]))
        subs = subs[0:nonzeros, :]
        vals = function_handle((nonzeros, 1))

        # Store everything
        return cls(subs, vals, shape, copy=False)

    @classmethod
    def from_aggregator(
        cls,
        subs: np.ndarray,
        vals: np.ndarray,
        shape: Shape | None = None,
        function_handle: str | Callable[[Any], float | np.ndarray] = "sum",
    ) -> sptensor:
        """Construct a :class:`pyttb.sptensor`.

        Constructed from a set of `subs` (subscripts),
        `vals` (values), and `shape` after an aggregation function is applied
        to the values.

        Parameters
        ----------
        subs:
            Subscripts of nonzero entries.
        vals:
            Values for nonzero entries.
        shape:
            Shape of sparse tensor.
        function_handle:
            Aggregation function, or name of supported
            aggregation function from :class:`numpy_groupies`.

        Examples
        --------
        Create a :class:`pyttb.sptensor` with some duplicate subscripts and use
        an aggregator function. The default aggregator is `sum`. The shape of the
        sparse tensor is inferred from the subscripts.

        >>> subs = np.array([[1, 2], [1, 3], [1, 3]])
        >>> vals = np.array([[6], [7], [8]])
        >>> shape = (4, 4)
        >>> S = ttb.sptensor.from_aggregator(subs, vals)
        >>> print(S)
        sparse tensor of shape (2, 4) with 2 nonzeros and order F
        [1, 2] = 6
        [1, 3] = 15

        Create another :class:`pyttb.sptensor` but specify the shape
        explicitly.

        >>> S = ttb.sptensor.from_aggregator(subs, vals, shape)
        >>> print(S)
        sparse tensor of shape (4, 4) with 2 nonzeros and order F
        [1, 2] = 6
        [1, 3] = 15

        Create another :class:`pyttb.sptensor` but aggregate using the mean of
        values corresponding to duplicate subscripts.

        >>> S3 = ttb.sptensor.from_aggregator(
        ...     subs, vals, shape, function_handle=np.mean
        ... )
        >>> print(S3)
        sparse tensor of shape (4, 4) with 2 nonzeros and order F
        [1, 2] = 6.0
        [1, 3] = 7.5
        """
        tt_subscheck(subs, False)
        tt_valscheck(vals, False)
        if subs.size > 1 and vals.shape[0] != subs.shape[0]:
            assert False, "Number of subscripts and values must be equal"

        # Extract the shape
        if shape is not None:
            shape = parse_shape(shape)
            tt_sizecheck(shape, False)
        else:
            shape = parse_shape(np.max(subs, axis=0) + 1)

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
                loc.flatten(),
                np.squeeze(vals),
                size=newsubs.shape[0],
                func=function_handle,
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
        """
        Return a deep copy of the :class:`pyttb.sptensor`.

        Examples
        --------
        Create a :class:`pyttb.sptensor` (S1) and make a deep copy. Verify
        the deep copy (S3) is not just a reference (like S2) to the original.

        >>> S1 = ttb.sptensor(shape=(2, 2))
        >>> S1[0, 0] = 1
        >>> S2 = S1
        >>> S3 = S1.copy()
        >>> S1[0, 0] = 3
        >>> S1[0, 0] == S2[0, 0]
        True
        >>> S1[0, 0] == S3[0, 0]
        False
        """
        return ttb.sptensor(self.subs, self.vals, self.shape, copy=True)

    @property
    def order(self) -> Literal["F"]:
        """Return the data layout of the underlying storage."""
        return "F"

    def _matches_order(self, array: np.ndarray) -> bool:
        """Check if provided array matches tensor memory layout."""
        if array.flags["C_CONTIGUOUS"] and self.order == "C":
            return True
        if array.flags["F_CONTIGUOUS"] and self.order == "F":
            return True
        return False

    def __deepcopy__(self, memo):
        """Return deep copy of this sptensor."""
        return self.copy()

    def allsubs(self) -> np.ndarray:
        """
        Generate all possible subscripts for the :class:`pyttb.sptensor`.

        Examples
        --------
        Create an empty :class:`pyttb.sptensor` and generate all subscripts:

        >>> S = ttb.sptensor(shape=(2, 2))
        >>> S.allsubs()
        array([[0, 0],
               [0, 1],
               [1, 0],
               [1, 1]])
        """
        # Preallocate (discover any memory issues here!)
        if len(self.shape) == 0:
            return np.empty(shape=(1, 0), dtype=int)

        s = np.zeros(shape=(prod(self.shape), self.ndims))

        # Generate appropriately sized ones vectors
        o = []
        for n in range(self.ndims):
            o.append(np.ones((self.shape[n], 1)))

        # Generate each column of the subscripts in turn
        for n in range(self.ndims):
            i: list[np.ndarray] = o.copy()
            i[n] = np.expand_dims(np.arange(0, self.shape[n]), axis=1)
            s[:, n] = np.squeeze(ttb.khatrirao(*i))

        return s.astype(int)

    @overload
    def collapse(
        self,
        dims: None,
        function_handle: Callable[[np.ndarray], float | np.ndarray],
    ) -> float: ...  # pragma: no cover see coveragepy/issues/970

    @overload
    def collapse(
        self,
        dims: OneDArray,
        function_handle: Callable[[np.ndarray], float | np.ndarray] = sum,
    ) -> np.ndarray | sptensor: ...  # pragma: no cover see coveragepy/issues/970

    def collapse(
        self,
        dims: OneDArray | None = None,
        function_handle: Callable[[np.ndarray], float | np.ndarray] = sum,
    ) -> float | np.ndarray | sptensor:
        """
        Collapse the :class:`pyttb.sptensor` along specified dimensions.

        Parameters
        ----------
        dims:
            Dimensions to collapse.
        function_handle:
            Function used to collapse dimensions.

        Examples
        --------
        Create a :class:`pyttb.sptensor` with two elements:

        >>> subs = np.array([[0, 0, 0], [0, 1, 0]])
        >>> vals = np.array([[6.0], [7.0]])
        >>> shape = (1, 2, 1)
        >>> S = ttb.sptensor(subs, vals, shape)

        Collapse across all dimensions, resulting in a scalar value:

        >>> S.collapse()
        13.0

        Collapse across a single dimension, resulting in a
        :class:`pyttb.sptensor`:

        >>> S.collapse(dims=np.array([0]))
        sparse tensor of shape (2, 1) with 2 nonzeros and order F
        [0, 0] = 6.0
        [1, 0] = 7.0

        Collapse across all but one dimension, resulting in a
        :class:`numpy.ndarray`:

        >>> S.collapse(dims=np.array([0, 2]))
        array([6., 7.])
        """
        dims, _ = tt_dimscheck(self.ndims, dims=dims)
        remdims = np.setdiff1d(np.arange(0, self.ndims), dims)

        # Check for the case where we accumulate over *all* dimensions
        if remdims.size == 0:
            result = function_handle(self.vals.transpose()[0])
            if isinstance(result, np.generic):
                result = result.item()
            return result

        # Calculate the size of the result
        newsize = np.array(self.shape)[remdims]

        # Check for the case where the result is just a dense vector
        if remdims.size == 1:
            if self.subs.size > 0:
                return accumarray(
                    self.subs[:, remdims].transpose()[0],
                    self.vals.transpose()[0],
                    size=newsize[0],
                    func=function_handle,
                )
            # TODO think about if this makes sense
            # complicates return typing
            return np.zeros((newsize[0],))

        # Create Result
        if self.subs.size > 0:
            return ttb.sptensor.from_aggregator(
                self.subs[:, remdims], self.vals, tuple(newsize), function_handle
            )
        return ttb.sptensor(np.array([]), np.array([]), tuple(newsize), copy=False)

    def contract(self, i_0: int, i_1: int) -> np.ndarray | sptensor | ttb.tensor:
        """Contract the :class:`pyttb.sptensor` along two dimensions.

        If the
        result is sufficiently dense, it is returned as a
        :class:`pyttb.tensor`.

        Parameters
        ----------
        i_0:
            First dimension.
        i_1:
            Second dimension.

        Examples
        --------
        Create a :class:`pyttb.sptensor` from a :class:`pyttb.tensor` and
        contract, resulting in a dense tensor, since the result is dense:

        >>> T = ttb.tensor(np.ones((2, 2, 2)))
        >>> S = T.to_sptensor()
        >>> S.contract(0, 1)
        tensor of shape (2,) with order F
        data[:] =
        [2. 2.]

        Create a :class:`pyttb.sptensor` and contract, resulting in
        a :class:`pyttb.sptensor` since the result is sparse:

        >>> subs = np.array([[1, 1, 1], [2, 2, 2]])
        >>> vals = np.array([[0.5], [1.5]])
        >>> shape = (4, 4, 4)
        >>> S = ttb.sptensor(subs, vals, shape)
        >>> S.contract(1, 2)
        sparse tensor of shape (4,) with 2 nonzeros and order F
        [1] = 0.5
        [2] = 1.5
        """
        if self.shape[i_0] != self.shape[i_1]:
            assert False, "Must contract along equally sized dimensions"

        if i_0 == i_1:
            assert False, "Must contract along two different dimensions"

        # Easy case - returns a scalar
        if self.ndims == 2:
            tfidx = self.subs[:, 0] == self.subs[:, 1]  # find diagonal entries
            return sum(self.vals[tfidx].transpose()[0])

        # Remaining dimensions after contract
        remdims = np.setdiff1d(np.arange(0, self.ndims), np.array([i_0, i_1])).astype(
            int
        )

        # Size for return
        newsize = tuple(np.array(self.shape)[remdims])

        # Find index of values on diagonal
        indx = np.where(self.subs[:, i_0] == self.subs[:, i_1])[0]

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
        if y.nnz > 0.5 * prod(y.shape):
            # Final result is a dense tensor
            return y.to_tensor()
        return y

    def double(self, immutable: bool = False) -> np.ndarray:
        """
        Convert the :class:`pyttb.sptensor` to a :class:`numpy.ndarray`.

        Parameters
        ----------
        immutable: Whether or not the returned data cam be mutated. May enable
            additional optimizations.

        Examples
        --------
        Create a :class:`pyttb.sptensor` with two elements and convert it to a
        :class:`numpy.ndarray`:

        >>> S = ttb.sptensor()
        >>> S[0, 1] = 1.5
        >>> S
        sparse tensor of shape (1, 2) with 1 nonzeros and order F
        [0, 1] = 1.5
        >>> S.double()
        array([[0. , 1.5]])
        """
        a = np.zeros(self.shape, order=self.order)
        if self.nnz > 0:
            a[tuple(self.subs.transpose())] = self.vals.transpose()[0]
        if immutable:
            a.flags.writeable = False
        return a

    def elemfun(self, function_handle: Callable[[np.ndarray], np.ndarray]) -> sptensor:
        """Apply a function to the nonzero elements of the :class:`pyttb.sptensor`.

        Returns a copy of the sparse tensor, with the
        updated values.

        Parameters
        ----------
        function_handle:
            Function to apply to all values.

        Examples
        --------
        Create a the :class:`pyttb.sptensor` and multiply each nonzero element
        by 2:

        >>> S1 = ttb.sptensor()
        >>> S1[2, 2, 2] = 1.5
        >>> S2 = S1.elemfun(lambda values: values * 2)
        >>> S2
        sparse tensor of shape (3, 3, 3) with 1 nonzeros and order F
        [2, 2, 2] = 3.0
        """
        vals = function_handle(self.vals)
        idx = np.where(vals > 0)[0]
        if idx.size == 0:
            return ttb.sptensor(np.array([]), np.array([]), self.shape, copy=False)
        return ttb.sptensor(self.subs[idx, :], vals[idx], self.shape, copy=False)

    def extract(self, searchsubs: np.ndarray) -> np.ndarray:
        """
        Extract value from the :class:`pyttb.sptensor`.

        Parameters
        ----------
        searchsubs:
            subscripts to find.

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
                error_msg += f"\tsubscript = {np.array2string(badsubs[i, :])} \n"
            assert False, f"{error_msg}Invalid subscripts"

        # Set the default answer to zero
        a = np.zeros(shape=(p, 1), dtype=self.vals.dtype, order=self.order)

        # Find which indices already exist and their locations
        valid, loc = tt_ismember_rows(searchsubs, self.subs)
        # Fill in the nonzero elements in the answer
        non_zeros = self.vals[loc[valid]]
        if np.sum(valid) > 0:
            a[valid] = non_zeros
        return a

    def find(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Find subscripts of nonzero elements in the :class:`pyttb.sptensor`.

        Examples
        --------
        >>> S = ttb.sptensor()
        >>> S[0, 1] = 1
        >>> S.find()
        (array([[0, 1]]), array([[1.]]))
        """
        return self.subs, self.vals

    def to_tensor(self) -> ttb.tensor:
        """Convert to dense tensor.

        Same as :meth:`pyttb.sptensor.full`.
        """
        return self.full()

    def full(self) -> ttb.tensor:
        """
        Convert the :class:`pyttb.sptensor` to a :class:`pyttb.tensor`.

        Examples
        --------
        Create a :class:`pyttb.sptensor` and convert it to a
        :class:`pyttb.tensor`:

        >>> S = ttb.sptensor()
        >>> S[1, 1] = 1
        >>> S.to_tensor()
        tensor of shape (2, 2) with order F
        data[:, :] =
        [[0. 0.]
         [0. 1.]]
        """
        # Handle the completely empty (no shape) case
        if len(self.shape) == 0:
            return ttb.tensor()

        # Create a dense zero tensor B that is the same shape as A
        B = ttb.tensor(np.zeros(shape=self.shape, order=self.order), copy=False)

        if self.subs.size == 0:
            return B

        # Extract the linear indices of entries in A
        idx = tt_sub2ind(self.shape, self.subs)

        # Copy the values of A into B using linear indices
        B[idx.astype(int)] = self.vals.transpose()[0]
        return B

    def to_sptenmat(
        self,
        rdims: np.ndarray | None = None,
        cdims: np.ndarray | None = None,
        cdims_cyclic: Literal["fc"] | Literal["bc"] | Literal["t"] | None = None,
    ) -> ttb.sptenmat:
        """Construct a :class:`pyttb.sptenmat` from a :class:`pyttb.sptensor`.

        Parameters
        ----------
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

        Examples
        --------
        Create a :class:`pyttb.sptensor`.

        >>> subs = np.array([[1, 2, 1], [1, 3, 1]])
        >>> vals = np.array([[6], [7]])
        >>> tshape = (4, 4, 4)
        >>> S = ttb.sptensor(subs, vals, tshape)

        Convert to a :class:`pyttb.sptenmat` unwrapping around the first dimension.
            Either allow for implicit column or explicit column dimension
            specification.

        >>> ST1 = S.to_sptenmat(rdims=np.array([0]))
        >>> ST2 = S.to_sptenmat(rdims=np.array([0]), cdims=np.array([1, 2]))
        >>> ST1.isequal(ST2)
        True

        Convert using cyclic column ordering. For the three mode case _fc_ is the same
            result.

        >>> ST3 = S.to_sptenmat(rdims=np.array([0]), cdims_cyclic="fc")
        >>> ST3  # doctest: +NORMALIZE_WHITESPACE
        sptenmat corresponding to a sptensor of shape (4, 4, 4) with 2 nonzeros \
        and order F
        rdims = [ 0 ] (modes of sptensor corresponding to rows)
        cdims = [ 1, 2 ] (modes of sptensor corresponding to columns)
            [1, 6] = 6
            [1, 7] = 7

        Backwards cyclic reverses the order.

        >>> ST4 = S.to_sptenmat(rdims=np.array([0]), cdims_cyclic="bc")
        >>> ST4  # doctest: +NORMALIZE_WHITESPACE
        sptenmat corresponding to a sptensor of shape (4, 4, 4) with 2 nonzeros \
        and order F
        rdims = [ 0 ] (modes of sptensor corresponding to rows)
        cdims = [ 2, 1 ] (modes of sptensor corresponding to columns)
            [1, 9] = 6
            [1, 13] = 7
        """
        n = self.ndims
        alldims = np.array([range(n)])

        rdims, cdims = gather_wrap_dims(self.ndims, rdims, cdims, cdims_cyclic)
        dims = np.hstack([rdims, cdims], dtype=int)
        if not len(dims) == n or not (alldims == np.sort(dims)).all():
            assert False, (
                "Incorrect specification of dimensions, the sorted "
                "concatenation of rdims and cdims must be range(source.ndims)."
            )

        rsize = np.array(self.shape)[rdims]
        csize = np.array(self.shape)[cdims]

        if rsize.size == 0:
            ridx: np.ndarray[tuple[int, ...], np.dtype[Any]] = np.zeros(
                (self.nnz, 1), dtype=int
            )
        elif self.subs.size == 0:
            ridx = np.array([], dtype=int)
        else:
            ridx = tt_sub2ind(rsize, self.subs[:, rdims])
        ridx = ridx.reshape((ridx.size, 1)).astype(int)

        if csize.size == 0:
            cidx: np.ndarray[tuple[int, ...], np.dtype[Any]] = np.zeros(
                (self.nnz, 1), dtype=int
            )
        elif self.subs.size == 0:
            cidx = np.array([], dtype=int)
        else:
            cidx = tt_sub2ind(csize, self.subs[:, cdims])
        cidx = cidx.reshape((cidx.size, 1)).astype(int)

        return ttb.sptenmat(
            np.hstack([ridx, cidx], dtype=int),
            self.vals.copy("K"),
            rdims.astype(int),
            cdims.astype(int),
            self.shape,
        )

    def innerprod(
        self, other: sptensor | ttb.tensor | ttb.ktensor | ttb.ttensor
    ) -> float:
        """Compute inner product of the :class:`pyttb.sptensor` with another tensor.

        Parameters
        ----------
        other:
            Other tensor to compute inner product with.

        Examples
        --------
        Create a :class:`pyttb.sptensor`:

        >>> S = ttb.sptensor()
        >>> S[0, 0] = 1
        >>> S[1, 1] = 2
        >>> S
        sparse tensor of shape (2, 2) with 2 nonzeros and order F
        [0, 0] = 1.0
        [1, 1] = 2.0

        Compute inner product with :class:`pyttb.tensor` of all ones that is
        the same shape as `S`:

        >>> T = ttb.tenones(S.shape)
        >>> S.innerprod(T)
        3.0

        Compute inner product with rank-1 :class:`pyttb.ktensor` of all ones
        that is the same shape as `S`:

        >>> factor_matrices = [np.ones((s, 1)) for s in S.shape]
        >>> K = ttb.ktensor(factor_matrices)
        >>> S.innerprod(K)
        3.0
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
                valsOther = other[subsSelf]
            else:
                [subsOther, valsOther] = other.find()
                valsSelf = self[subsOther]
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

    def isequal(self, other: sptensor | ttb.tensor) -> bool:
        """Determine if the :class:`pyttb.sptensor` is equal to another tensor.

        Equal when all elements are exactly the same in both tensors.

        Parameters
        ----------
        other:
           Other tensor to compare against.

        Examples
        --------
        Create a :class:`pyttb.sptensor`:

        >>> S = ttb.sptensor()
        >>> S[0, 0] = 1
        >>> S[1, 1] = 2
        >>> S
        sparse tensor of shape (2, 2) with 2 nonzeros and order F
        [0, 0] = 1.0
        [1, 1] = 2.0

        Compare with a tensor that should be equal:

        >>> T = S.to_tensor()
        >>> S.isequal(T)
        True

        Compare with a tensor that should not be equal:

        >>> T[0, 0] = T[0, 0] + 1
        >>> S.isequal(T)
        False
        """
        if self.shape != other.shape:
            return False
        if isinstance(other, ttb.sptensor):
            if self.nnz != other.nnz:
                return False
            return (self - other).nnz == 0
        if isinstance(other, ttb.tensor):
            return other.isequal(self)
        return False

    def logical_and(self, other: float | sptensor | ttb.tensor) -> sptensor:
        """
        Logical AND between the :class:`pyttb.sptensor` and another object.

        Parameters
        ----------
        other:
           Other object to compute with.

        Examples
        --------
        Create a :class:`pyttb.sptensor`:

        >>> S = ttb.sptensor()
        >>> S[0, 0] = 1
        >>> S[1, 1] = 2
        >>> S
        sparse tensor of shape (2, 2) with 2 nonzeros and order F
        [0, 0] = 1.0
        [1, 1] = 2.0

        Compute logical AND with a :class:`pyttb.tensor` that has the same
        nonzero pattern but different values:

        >>> T = S.to_tensor()
        >>> T[0, 0] = T[0, 0] + 1
        >>> S.logical_and(T)
        sparse tensor of shape (2, 2) with 2 nonzeros and order F
        [0, 0] = 1.0
        [1, 1] = 1.0

        Compute logical AND with a scalar value:

        >>> S.logical_and(1.0)
        sparse tensor of shape (2, 2) with 2 nonzeros and order F
        [0, 0] = 1.0
        [1, 1] = 1.0
        """
        # Case 1: One argument is a scalar
        if isinstance(other, (int, float)):
            if other == 0:
                C = sptensor(shape=self.shape)
            else:
                newvals = np.ones_like(self.vals)
                C = sptensor(self.subs, newvals, self.shape)
            return C
        # Case 2: Argument is a tensor of some sort
        if isinstance(other, sptensor):
            # Check that the shapes match
            if not self.shape == other.shape:
                assert False, "Must be tensors of the same shape"

            C = sptensor.from_aggregator(
                np.vstack((self.subs, other.subs)),
                np.vstack((self.vals, other.vals)),
                self.shape,
                lambda x: len(x) == 2,
            )
            C.vals = C.vals.astype(self.vals.dtype)

            return C

        if isinstance(other, ttb.tensor):
            BB = sptensor(self.subs, other[self.subs][:, None], self.shape)
            C = self.logical_and(BB)
            return C

        # Otherwise
        assert False, "The arguments must be two sptensors or an sptensor and a scalar."

    def logical_not(self) -> sptensor:
        """
        Logical NOT for the :class:`pyttb.sptensor`.

        Examples
        --------
        Create a :class:`pyttb.sptensor` and compute logical NOT:

        >>> S = ttb.sptensor()
        >>> S[0, 0] = 1
        >>> S[1, 1] = 2
        >>> S
        sparse tensor of shape (2, 2) with 2 nonzeros and order F
        [0, 0] = 1.0
        [1, 1] = 2.0
        >>> S.logical_not()
        sparse tensor of shape (2, 2) with 2 nonzeros and order F
        [0, 1] = 1.0
        [1, 0] = 1.0
        """
        allsubs = self.allsubs()
        subsIdx = tt_setdiff_rows(allsubs, self.subs)
        subs = allsubs[subsIdx]
        trueVector = np.ones(shape=(subs.shape[0], 1), dtype=self.vals.dtype)
        return sptensor(subs, trueVector, self.shape)

    @overload
    def logical_or(
        self, other: float | ttb.tensor
    ) -> ttb.tensor: ...  # pragma: no cover see coveragepy/issues/970

    @overload
    def logical_or(
        self, other: sptensor
    ) -> sptensor: ...  # pragma: no cover see coveragepy/issues/970

    def logical_or(self, other: float | ttb.tensor | sptensor) -> ttb.tensor | sptensor:
        """
        Logical OR between the :class:`pyttb.sptensor` and another object.

        Parameters
        ----------
        other:
            Other object to compute with.

        Examples
        --------
        Create a :class:`pyttb.sptensor` and compute logical OR with itself:

        >>> S = ttb.sptensor()
        >>> S[0, 0] = 1
        >>> S[1, 1] = 2
        >>> S.logical_or(S)
        sparse tensor of shape (2, 2) with 2 nonzeros and order F
        [0, 0] = 1.0
        [1, 1] = 1.0

        Compute logical OR with a :class:`pyttb.tensor` that has the same
        nonzero pattern:

        >>> T = S.to_tensor()
        >>> S.logical_or(T)
        tensor of shape (2, 2) with order F
        data[:, :] =
        [[1. 0.]
         [0. 1.]]

        Compute logical OR with a scalar value:

        >>> S.logical_or(1)
        tensor of shape (2, 2) with order F
        data[:, :] =
        [[1. 1.]
         [1. 1.]]
        """
        # Case 1: Argument is a scalar or tensor
        if isinstance(other, (float, int, ttb.tensor)):
            return self.full().logical_or(other)

        # Case 2: Argument is an sptensor
        if self.shape != other.shape:
            assert False, "Logical Or requires tensors of the same size"

        if isinstance(other, ttb.sptensor):
            C = sptensor.from_aggregator(
                np.vstack((self.subs, other.subs)),
                np.ones((self.subs.shape[0] + other.subs.shape[0], 1)),
                self.shape,
                lambda x: len(x) >= 1,
            )
            C.vals = C.vals.astype(self.vals.dtype)
            return C

        assert False, "Sptensor Logical Or argument must be scalar or sptensor"

    @overload
    def logical_xor(
        self, other: float | ttb.tensor
    ) -> ttb.tensor: ...  # pragma: no cover see coveragepy/issues/970

    @overload
    def logical_xor(
        self, other: sptensor
    ) -> sptensor: ...  # pragma: no cover see coveragepy/issues/970

    def logical_xor(
        self, other: float | ttb.tensor | sptensor
    ) -> ttb.tensor | sptensor:
        """
        Logical XOR between the :class:`pyttb.sptensor` and another object.

        Parameters
        ----------
        other:
            Other object to compute with.

        Examples
        --------
        Create a :class:`pyttb.sptensor` and compute logical XOR with itself:

        >>> S = ttb.sptensor()
        >>> S[0, 0] = 1
        >>> S[1, 1] = 2
        >>> S.logical_xor(S)
        empty sparse tensor of shape (2, 2) with order F

        Compute logical XOR with :class:`pyttb.tensor` that has a different
        nonzero pattern:

        >>> T = S.to_tensor()
        >>> T[1, 0] = 1.0
        >>> S.logical_xor(T)
        tensor of shape (2, 2) with order F
        data[:, :] =
        [[0. 0.]
         [1. 0.]]

        Compute logical XOR with a scalar value:

        >>> S.logical_xor(1)
        tensor of shape (2, 2) with order F
        data[:, :] =
        [[0. 1.]
         [1. 0.]]
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
            result = ttb.sptensor.from_aggregator(
                subs, np.ones((len(subs), 1)), self.shape, lambda x: len(x) == 1
            )
            result.vals = result.vals.astype(self.vals.dtype)
            return result

        assert False, "The argument must be an sptensor, tensor or scalar"

    def mask(self, W: sptensor) -> np.ndarray:
        """Extract values of the :class:`pyttb.sptensor` as specified by `W`.

        The values in the sparse tensor corresponding to ones (1) in `W`
        will be returned as a column vector.

        Parameters
        ----------
        W:
            Mask tensor.

        Examples
        --------
        Create a :class:`pyttb.sptensor`:

        >>> S = ttb.sptensor()
        >>> S[0, 0] = 1
        >>> S[1, 1] = 2
        >>> S
        sparse tensor of shape (2, 2) with 2 nonzeros and order F
        [0, 0] = 1.0
        [1, 1] = 2.0

        Create mask :class:`pyttb.sptensor` and extract nonzero values
        from `S`:

        >>> W = ttb.sptensor()
        >>> W[0, 0] = 1
        >>> W[1, 1] = 1
        >>> S.mask(W)
        array([[1.],
               [2.]])

        Create mask :class:`pyttb.sptensor` and extract some nonzero
        values and some zero values:

        >>> W = ttb.sptensor()
        >>> W[0, 0] = 1
        >>> W[1, 0] = 1
        >>> S.mask(W)
        array([[1.],
               [0.]])
        """
        # Error check
        if len(W.shape) != len(self.shape) or np.any(
            np.array(W.shape) > np.array(self.shape)
        ):
            assert False, "Mask cannot be bigger than the data tensor"

        # Extract locations of nonzeros in W
        wsubs, _ = W.find()

        # Find which values in the mask match nonzeros in X
        valid, idx = tt_ismember_rows(wsubs, self.subs)
        matching_indices = idx[valid]

        # Assemble return array
        nvals = wsubs.shape[0]
        vals = np.zeros((nvals, 1))
        vals[matching_indices] = self.vals[matching_indices]
        return vals

    def mttkrp(
        self, U: ttb.ktensor | Sequence[np.ndarray], n: int | np.integer
    ) -> np.ndarray:
        """Matricized tensor times Khatri-Rao product using :class:`pyttb.sptensor`.

        This is an efficient form of the matrix
        product that avoids explicitly computing the matricized sparse tensor
        and the large intermediate Khatri-Rao product arrays.

        If the input includes a list of 2-D arrays (factor_matrices), this
        computes a matrix product of the mode-`n` matricization of the sparse
        tensor with the Khatri-Rao product of all arrays in the list except
        the `n` th. The length of the list of arrays must equal the number of
        dimensions of the sparse tensor. The shapes of each array must have
        leading dimensions equal to the dimensions of the sparse tensor and
        the same second dimension.

        If the input is a :class:`pyttb.ktensor`, this
        computes a matrix product of the mode-`n` matricization of the sparse
        tensor with the Khatri-Rao product formed by the `factor_matrices` and
        `weights` from the ktensor, excluding the `n` th factor matrix and
        corresponding weight. The shape of the ktensor must be compatible with
        the shape of the sparse tensor.

        Parameters
        ----------
        U:
            Factor matrix or list of factor matrices.
        n:
            Mode used to matricize the :class:`pyttb.sptensor`.

        Examples
        --------
        Create list of factor matrices:

        >>> A = np.ones((4, 4))
        >>> U = [A, A, A]

        Create a :class:`pyttb.sptensor` and compute the matricized tensor
        times Khatri-Rao product between it and the factor matrices:

        >>> subs = np.array([[1, 1, 1], [1, 1, 3], [2, 2, 2], [3, 3, 3]])
        >>> vals = np.array([[0.5], [1.5], [2.5], [3.5]])
        >>> shape = (4, 4, 4)
        >>> S = ttb.sptensor(subs, vals, shape)
        >>> S.mttkrp(U, 0)
        array([[0. , 0. , 0. , 0. ],
               [2. , 2. , 2. , 2. ],
               [2.5, 2.5, 2.5, 2.5],
               [3.5, 3.5, 3.5, 3.5]])
        """
        # In the sparse case, it is most efficient to do a series of TTV operations
        # rather than forming the Khatri-Rao product.

        U = get_mttkrp_factors(U, n, self.ndims)

        if n == 0:
            R = U[1].shape[1]
        else:
            R = U[0].shape[1]

        V = np.zeros((self.shape[n], R), order=self.order)
        for r in range(R):
            # Set up list with appropriate vectors for ttv multiplication
            Z = []
            for i in range(self.ndims):
                if i != n:
                    Z.append(U[i][:, r])
                else:
                    Z.append(np.array([], order=self.order))
            # Perform ttv multiplication
            ttv = self.ttv(Z, exclude_dims=int(n))
            # TODO is is possible to hit the float condition here?
            if isinstance(ttv, float):  # pragma: no cover
                V[:, r] = ttv
            else:
                V[:, r] = ttv.double()

        return V

    @property
    def ndims(self) -> int:
        """
        Number of dimensions of the :class:`pyttb.sptensor`.

        Examples
        --------
        Create a :class:`pyttb.sptensor` and return the number of dimensions:

        >>> S = ttb.sptensor(shape=(1, 2, 3, 4, 5, 6))
        >>> S
        empty sparse tensor of shape (1, 2, 3, 4, 5, 6) with order F
        >>> S.ndims
        6
        """
        return len(self.shape)

    @property
    def nnz(self) -> int:
        """
        Number of nonzero values in the :class:`pyttb.sptensor`.

        Examples
        --------
        Create a :class:`pyttb.sptensor` from a :class:`pyttb.tensor`
        containing a 10x10 diagonal identity matrix and return the number
        of nonzeros:

        >>> S = ttb.tensor(np.eye(10)).to_sptensor()
        >>> S.nnz
        10
        """
        if self.subs.size == 0:
            return 0
        return self.subs.shape[0]

    def norm(self) -> float:
        """Compute the norm of the :class:`pyttb.sptensor`.

        Frobenius norm, or square root of the sum of
        squares of entries.

        Examples
        --------
        Create a :class:`pyttb.sptensor` from a diagonal matrix and compute
        its norm:

        >>> S = ttb.tensor(np.diag([1.0, 2.0, 3.0, 4.0])).to_sptensor()
        >>> S
        sparse tensor of shape (4, 4) with 4 nonzeros and order F
        [0, 0] = 1.0
        [1, 1] = 2.0
        [2, 2] = 3.0
        [3, 3] = 4.0
        >>> S.norm()  # doctest: +ELLIPSIS
        5.47722557...
        """
        return np.linalg.norm(self.vals).item()

    def nvecs(self, n: int, r: int, flipsign: bool = True) -> np.ndarray:
        """
        Compute the leading mode-n vectors of the :class:`pyttb.sptensor`.

        Computes the `r` leading eigenvectors of `Sn*Sn.T` (where `Sn` is the
        mode-`n` matricization/unfolding of a sparse tensor `S`), which
        provides information about the mode-`n` fibers. In two-dimensions,
        the `r` leading mode-1 vectors are the same as the `r` left singular
        vectors and the `r` leading mode-2 vectors are the same as the `r`
        right singular vectors. By default, this method computes the top `r`
        eigenvectors of `Sn*Sn.T`. The output product for sparse tensors is
        not formed, making this operation very efficient when the tensor is
        very sparse.

        Parameters
        ----------
        n:
            Mode to use for matricization.
        r:
            Number of eigenvectors to compute and use.
        flipsign:
            If True, make each column's largest element positive.

        Examples
        --------
        Create a :class:`pyttb.sptensor`:

        >>> subs = np.array([[0, 0], [0, 1], [1, 0]])
        >>> vals = np.array([[1.0], [2.0], [3.0]])
        >>> shape = (2, 2)
        >>> S = ttb.sptensor(subs, vals, shape)

        Compute two mode-0 leading eigenvectors of `S`, making sign of largest
        element of each eigenvector positive (i.e., `flipsign` =True).

        >>> S.nvecs(0, 2, flipsign=True)  # doctest: +ELLIPSIS
        array([[-0.4718...,  0.8816...],
               [ 0.8816...,  0.4718...]])

        Compute the same `nvecs` of `S`, but do not adjust the sign of the
        largest element of each eigenvector.

        >>> S.nvecs(0, 2, flipsign=False)  # doctest: +ELLIPSIS
        array([[ 0.4718..., -0.8816...],
               [-0.8816..., -0.4718...]])
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
        return to_memory_order(v, self.order)

    def ones(self) -> sptensor:
        """
        Replace nonzero values of the :class:`pyttb.sptensor` with ones (1).

        Examples
        --------
        Create a :class:`pyttb.sptensor`:

        >>> subs = np.array([[0, 0], [0, 1], [1, 0]])
        >>> vals = np.array([[1.0], [2.0], [3.0]])
        >>> shape = (2, 2)
        >>> S = ttb.sptensor(subs, vals, shape)
        >>> S
        sparse tensor of shape (2, 2) with 3 nonzeros and order F
        [0, 0] = 1.0
        [0, 1] = 2.0
        [1, 0] = 3.0

        Replace the nonzero values of `S` with the value 1:

        >>> S.ones()
        sparse tensor of shape (2, 2) with 3 nonzeros and order F
        [0, 0] = 1.0
        [0, 1] = 1.0
        [1, 0] = 1.0
        """
        oneVals = self.vals.copy("K")
        oneVals.fill(1)
        return ttb.sptensor(self.subs, oneVals, self.shape)

    def permute(self, order: OneDArray) -> sptensor:
        """Permute the :class:`pyttb.sptensor` dimensions.

        The result is a new
        sparse tensor that has the same values, but the order of the
        subscripts needed to access any particular element are rearranged
        as specified by `order`.

        Parameters
        ----------
        order:
            New order of tensor dimensions.

        Examples
        --------
        Create a :class:`pyttb.sptensor`:

        >>> subs = np.array([[0, 0], [0, 1], [1, 0]])
        >>> vals = np.array([[1.0], [2.0], [3.0]])
        >>> shape = (2, 2)
        >>> S = ttb.sptensor(subs, vals, shape)
        >>> S
        sparse tensor of shape (2, 2) with 3 nonzeros and order F
        [0, 0] = 1.0
        [0, 1] = 2.0
        [1, 0] = 3.0

        Permute the order of the dimensions by reversing them:

        >>> S1 = S.permute(np.array((1, 0)))
        >>> S1
        sparse tensor of shape (2, 2) with 3 nonzeros and order F
        [0, 0] = 1.0
        [1, 0] = 2.0
        [0, 1] = 3.0
        """
        order = parse_one_d(order)
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
        new_shape: Shape,
        old_modes: np.ndarray | int | None = None,
    ) -> sptensor:
        """Reshape the :class:`pyttb.sptensor` to the `new_shape`.

        If `old_modes` is specified, reshape only those modes of
        the sparse tensor, moving newly reshaped modes to the end of the
        subscripts; otherwise use all modes. The product of the new shape
        must equal the product of the old shape.

        Parameters
        ----------
        new_shape:
            New shape.
        old_modes:
            Modes used for reshaping.

        Examples
        --------
        Create a :class:`pyttb.sptensor` from a :class:`pyttb.tensor`:

        >>> S = ttb.tensor(np.arange(9) + 1, shape=(1, 3, 3)).to_sptensor()
        >>> S
        sparse tensor of shape (1, 3, 3) with 9 nonzeros and order F
        [0, 0, 0] = 1
        [0, 1, 0] = 2
        [0, 2, 0] = 3
        [0, 0, 1] = 4
        [0, 1, 1] = 5
        [0, 2, 1] = 6
        [0, 0, 2] = 7
        [0, 1, 2] = 8
        [0, 2, 2] = 9

        Reshape to a 2-way :class:`pyttb.sptensor`:

        >>> S.reshape((1, 9))
        sparse tensor of shape (1, 9) with 9 nonzeros and order F
        [0, 0] = 1
        [0, 1] = 2
        [0, 2] = 3
        [0, 3] = 4
        [0, 4] = 5
        [0, 5] = 6
        [0, 6] = 7
        [0, 7] = 8
        [0, 8] = 9

        Reshape the first two dimensions and move to the end of the subscripts.
        The first two subscripts are reshaped from (1,3) to (3,1) and moved
        after the remaining subscript (i.e., corresponding to mode 2).

        >>> S.reshape(new_shape=(3, 1), old_modes=np.array((1, 0)))
        sparse tensor of shape (3, 3, 1) with 9 nonzeros and order F
        [0, 0, 0] = 1
        [0, 1, 0] = 2
        [0, 2, 0] = 3
        [1, 0, 0] = 4
        [1, 1, 0] = 5
        [1, 2, 0] = 6
        [2, 0, 0] = 7
        [2, 1, 0] = 8
        [2, 2, 0] = 9
        """
        if old_modes is None:
            old_modes = np.arange(0, self.ndims, dtype=int)
            keep_modes = np.array([], dtype=int)
        else:
            keep_modes = np.setdiff1d(np.arange(0, self.ndims, dtype=int), old_modes)

        shapeArray = np.array(self.shape)
        old_shape = shapeArray[old_modes]
        keep_shape = shapeArray[keep_modes]
        new_shape = parse_shape(new_shape)

        if prod(new_shape) != prod(old_shape):
            assert False, "Reshape must maintain tensor size"

        if self.subs.size == 0:
            return ttb.sptensor(
                np.array([]),
                np.array([]),
                np.concatenate((keep_shape, new_shape)),
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
            np.concatenate((keep_shape, new_shape)),
        )

    def scale(
        self,
        factor: np.ndarray | ttb.tensor | ttb.sptensor,
        dims: OneDArray,
    ) -> sptensor:
        """
        Scale the :class:`pyttb.sptensor` along specified dimensions.

        Parameters
        ----------
        factor:
            Scaling factor.
        dims:
            Dimensions to scale.

        Examples
        --------
        Create a :class:`pyttb.sptensor` from a :class:`pyttb.tensor`:

        >>> S = ttb.tensor(np.arange(9) + 1, shape=(1, 3, 3)).to_sptensor()
        >>> S
        sparse tensor of shape (1, 3, 3) with 9 nonzeros and order F
        [0, 0, 0] = 1
        [0, 1, 0] = 2
        [0, 2, 0] = 3
        [0, 0, 1] = 4
        [0, 1, 1] = 5
        [0, 2, 1] = 6
        [0, 0, 2] = 7
        [0, 1, 2] = 8
        [0, 2, 2] = 9

        Mode 2 is of length 3. Create a scaling factor array of length 3 and
        scale along mode 2:

        >>> scaling_factor = np.array([1, 2, 3])
        >>> S.scale(scaling_factor, np.array([2]))
        sparse tensor of shape (1, 3, 3) with 9 nonzeros and order F
        [0, 0, 0] = 1
        [0, 1, 0] = 2
        [0, 2, 0] = 3
        [0, 0, 1] = 8
        [0, 1, 1] = 10
        [0, 2, 1] = 12
        [0, 0, 2] = 21
        [0, 1, 2] = 24
        [0, 2, 2] = 27
        """
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
                self.subs, self.vals * factor[self.subs[:, dims]], self.shape
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
        """Convert 2-way :class:`pyttb.sptensor` to :class:`scipy.sparse.coo_matrix`.

        Examples
        --------
        Create a 2-way :class:`pyttb.sptensor`:

        >>> S = ttb.tendiag([1, 2]).to_sptensor()
        >>> S
        sparse tensor of shape (2, 2) with 2 nonzeros and order F
        [0, 0] = 1.0
        [1, 1] = 2.0

        Convert it to a :class:`scipy.sparse.coo_matrix`:

        >>> M = S.spmatrix()
        >>> type(M)
        <class 'scipy.sparse._coo.coo_matrix'>
        >>> M.toarray()
        array([[1., 0.],
               [0., 2.]])
        """
        if self.ndims != 2:
            assert False, "Sparse tensor must be two dimensional"

        if self.subs.size == 0:
            return sparse.coo_matrix(self.shape)
        return sparse.coo_matrix(
            (self.vals.transpose()[0], self.subs.transpose()), self.shape
        )

    def squeeze(self) -> sptensor | float:
        """Remove singleton dimensions from the :class:`pyttb.sptensor`.

        Examples
        --------
        Create a :class:`pyttb.sptensor` with a single element and squeeze
        all the dimensions:

        >>> S = ttb.sptensor(np.array([[0, 0, 0, 0, 0]]), np.array([[3.14]]))
        >>> S.squeeze()
        3.14

        Create a :class:`pyttb.sptensor` with and interior singleton dimension
        and squeeze it out:

        >>> S = ttb.sptensor(np.array([[0, 0, 0], [1, 0, 1]]), np.array([[1.0], [2.0]]))
        >>> S
        sparse tensor of shape (2, 1, 2) with 2 nonzeros and order F
        [0, 0, 0] = 1.0
        [1, 0, 1] = 2.0
        >>> S.squeeze()
        sparse tensor of shape (2, 2) with 2 nonzeros and order F
        [0, 0] = 1.0
        [1, 1] = 2.0
        """
        shapeArray = np.array(self.shape)

        # No singleton dimensions
        if np.all(shapeArray > 1):
            return self.copy()
        idx = np.where(shapeArray > 1)[0]
        if idx.size == 0:
            return self.vals.item()
        siz = tuple(shapeArray[idx])
        if self.vals.size == 0:
            return ttb.sptensor(np.array([]), np.array([]), siz, copy=False)
        return ttb.sptensor(self.subs[:, idx], self.vals, siz)

    def subdims(self, region: Sequence[int | np.ndarray | slice]) -> np.ndarray:
        """
        Compute the locations of subscripts within a subdimension.

        Finds the locations of the subscripts in the :class:`pyttb.sptensor`
        that are within the range specified by `region`. For example, if
        `region` is `[1, np.array([1,2]), np.array([1,2]])`, then the locations
        of all elements of the sparse tensor that have a first subscript equal
        to 1, a second subscript equal to 1 or 2, and a third subscript equal
        to 1 or 2 are returned.

        Parameters
        ----------
        region:
            Subset of subscripts in which to find nonzero values.

        Examples
        --------
        Create a :class:`pyttb.sptensor`:

        >>> subs = np.array([[1, 1, 1], [1, 1, 3], [2, 2, 2], [2, 3, 2]])
        >>> vals = np.array([[0.5], [1.5], [2.5], [3.5]])
        >>> shape = (4, 4, 4)
        >>> S = ttb.sptensor(subs, vals, shape)

        Define a region with subscripts 1 in mode 0, 1 in mode 1, and either
        1 or 3 in mode 2, then find the location of the subscripts of the
        `S` for that region:

        >>> region = [1, 1, np.array([1, 3])]
        >>> subs_loc = S.subdims(region)
        >>> print(subs_loc)
        [0 1]
        >>> S.subs[subs_loc]
        array([[1, 1, 1],
               [1, 1, 3]])

        Use :meth:`slice` to define part of the region. In this case,
        allow any subscript in mode 1:

        >>> region = (2, slice(None, None, None), 2)
        >>> subs_loc = S.subdims(region)
        >>> print(subs_loc)
        [2 3]
        >>> S.subs[subs_loc]
        array([[2, 2, 2],
               [2, 3, 2]])
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

        for i in range(self.ndims):
            # TODO: Consider cleaner typing coercion
            # Find subscripts that match in dimension i
            if isinstance(region[i], (int, np.generic)):
                tf = np.isin(self.subs[loc, i], cast("int", region[i]))
            elif isinstance(region[i], (np.ndarray, list)):
                tf = np.isin(self.subs[loc, i], cast("np.ndarray", region[i]))
            elif isinstance(region[i], slice):
                sliceRegion = range(self.shape[i])[region[i]]
                tf = np.isin(self.subs[loc, i], sliceRegion)
            else:
                raise ValueError(
                    f"Unexpected type in region sequence. "
                    f"At index: {i} got {region[i]} with type {type(region[i])}"
                )

            # Pare down the list of indices
            loc = loc[tf]

        return loc

    def ttv(
        self,
        vector: np.ndarray | Sequence[np.ndarray],
        dims: OneDArray | None = None,
        exclude_dims: OneDArray | None = None,
    ) -> sptensor | ttb.tensor | float:
        """
        Multiplication of the :class:`pyttb.sptensor` with a vector.

        Computes the n-mode product of the :class:`pyttb.sptensor` with a
        vector. Let `n` specify the dimension (or mode) along which the
        vector should be multiplied. If the vector has `shape = (I,)`, then
        the sparse tensor must have `shape[n] = I`. The result has one less
        dimension, as dimension `n` is removed in the multiplication.

        Multiplication with more than one vector is provided using a list of
        vectors and corresponding dimensions in the sparse tensor to use.

        The dimensions of the sparse tensor with which to multiply can be provided as
        `dims`, or the dimensions to exclude from `[0, ..., self.ndims]` can be
        specified using `exclude_dims`.

        Parameters
        ----------
        vector:
            Vector or vectors to multiply by.
        dims:
            Dimensions to multiply against.
        exclude_dims:
            Use all dimensions but these.

        Examples
        --------
        Create a 2-way :class:`pyttb.sptensor` that is relatively dense:

        >>> subs = np.array([[0, 0], [0, 1], [1, 0]])
        >>> vals = np.array([[1.0], [2.0], [3.0]])
        >>> shape = (2, 2)
        >>> S = ttb.sptensor(subs, vals, shape)
        >>> S
        sparse tensor of shape (2, 2) with 3 nonzeros and order F
        [0, 0] = 1.0
        [0, 1] = 2.0
        [1, 0] = 3.0

        Compute the product of `S` with a vector of ones across mode 0. The
        result is a :class:`pyttb.tensor`:

        >>> S.ttv(np.ones(2), 0)
        tensor of shape (2,) with order F
        data[:] =
        [4. 2.]

        Create a 3-way :class:`pyttb.sptensor` that is much more sparse:

        >>> subs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 1]])
        >>> vals = np.array([[1.0], [2.0], [3.0]])
        >>> shape = (2, 2, 2)
        >>> S1 = ttb.sptensor(subs, vals, shape)

        Compute the product of `S1` with a vector of ones across mode 1. The
        result is a :class:`pyttb.sptensor`:

        >>> S1.ttv(np.ones(2), 1)
        sparse tensor of shape (2, 2) with 2 nonzeros and order F
        [0, 0] = 3.0
        [1, 1] = 3.0

        Compute the product of `S1` with multiple vectors across all
        dimensions. When all dimensions will be included in the product,
        `dims` does not need to be specified. The result is a scalar value.

        >>> vectors = [(i + 1) * np.ones(2) for i in range(len(S1.shape))]
        >>> vectors
        [array([1., 1.]), array([2., 2.]), array([3., 3.])]
        >>> S1.ttv(vectors)
        36.0
        """
        # Check that vector is a list of vectors,
        # if not place single vector as element in list
        if len(vector) > 0 and isinstance(vector[0], (int, float, np.int_, np.float64)):
            return self.ttv(np.array([vector]), dims, exclude_dims)

        # Get sorted dims and index for multiplicands
        dims, vidx = tt_dimscheck(self.ndims, len(vector), dims, exclude_dims)
        remdims = np.setdiff1d(np.arange(0, self.ndims), dims).astype(int)

        # Check that each multiplicand is the right size.
        for i in range(dims.size):
            if vector[vidx[i]].shape != (self.shape[dims[i]],):
                assert False, "Multiplicand is wrong size"

        # Multiply each value by the appropriate elements of the appropriate vector
        newvals = self.vals.copy("K")
        subs = self.subs.copy("K")
        if subs.size == 0:  # No nonzeros in tensor
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
            return np.sum(newvals).item()

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
                    np.arange(0, newsiz)[:, None], c.reshape((len(c), 1)), tuple(newsiz)
                )
            return ttb.tensor(c, tuple(newsiz), copy=False)

        # Case 2: Result is a multiway array
        c = ttb.sptensor.from_aggregator(newsubs, newvals, tuple(newsiz))

        # Convert to a dense tensor if more than 50% of the result is nonzero.
        if c.nnz > 0.5 * prod(newsiz):
            c = c.to_tensor()

        return c

    def __getitem__(self, item):  # noqa: PLR0912, PLR0915
        """
        Subscripted reference for the :class:`pyttb.sptensor`.

        We can extract elements or subtensors from a sparse tensor in the
        following ways.

        Case 1a: `y = S[I1,I2,...,In]`, where each `I` is an subscript, returns a
        scalar.

        Case 1b: `Y = S[R1,R2,...,Rn]`, where one or more `R` is a range and
        the rest are subsctiprs, returns a sparse tensor. The elements are
        renumbered here as appropriate.

        Case 2a: `V = S[M] where `M` is a `p` x `n` array of subscripts, returns
        a vector of `p` values.

        Case 2b: `V = S[I]` where `I` is a set of `p`
        linear subscripts, returns a vector of `p` values.

        Any ambiguity results in executing the first valid case. This
        is particularly an issue if `self.ndims == 1`.

        Examples
        --------
        Create a 3-way :class:`pyttb.sptensor`:

        >>> subs = np.array([[3, 3, 3], [1, 1, 0], [1, 2, 1]])
        >>> vals = np.array([[3], [5], [1]])
        >>> shape = (4, 4, 4)
        >>> S = ttb.sptensor(subs, vals, shape)

        Use a single subscript (Case 1a):

        >>> print(S[1, 2, 1])
        1

        Use a range of subscripts (Case 1b):

        >>> S[3, 3, :]
        sparse tensor of shape (4,) with 1 nonzeros and order F
        [3] = 3

        Use an array of subscripts (Case 2a):

        >>> M = np.array([[1, 1, 0], [1, 1, 1]])
        >>> print(S[M])
        [[5]
         [0]]

        Use linear subscripting, including negative subscript for offsets from
        the end of the linear subscripts into the sparse tensor data (Case 2b):

        >>> print(S[[5, -1]])
        [[5]
         [3]]
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
                subs = self.subs.copy("K")
            else:
                subs = self.subs[loc, :]
            if self.vals.size == 0:
                vals = self.vals.copy("K")
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
                        item = prod(self.shape) + item
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
        Subscripted assignment for the :class:`pyttb.sptensor`.

        We can assign elements to a sparse tensor in the following ways.

        Case 1: `S[R1,R2,...,Rn] = Y`, in which case we replace the
        rectangular subtensor (or single element) specified by the ranges
        `R1`,...,`Rn` with `Y`. The right-hand-side can be a scalar or an
        sparse tensor.

        Case 2: `S[M] = V`, where `M` is a `p` x `n` array of subscripts
        and `V` is a scalar value or a vector containing `p` values.

        Assignment using linear subscripting is not supported for sparse
        tensors.

        Examples
        --------
        Create a 3-way :class:`pyttb.sptensor`:

        >>> S = ttb.sptensor(shape=(3, 4, 5))

        Set a single element using subscripts or a tuple:

        >>> S[0, 0, 0] = 1
        >>> S[(0, 0, 0)] = 1
        >>> S
        sparse tensor of shape (3, 4, 5) with 1 nonzeros and order F
        [0, 0, 0] = 1.0
        >>> S
        sparse tensor of shape (3, 4, 5) with 1 nonzeros and order F
        [0, 0, 0] = 1.0

        Set a range of elements using a single value:

        >>> S[0, 0, 1:3] = 2
        >>> S
        sparse tensor of shape (3, 4, 5) with 3 nonzeros and order F
        [0, 0, 0] = 1.0
        [0, 0, 1] = 2.0
        [0, 0, 2] = 2.0

        Set a range of elements using a :class:`pyttb.sptensor`:

        >>> S[0:1, 1:3, 3:4] = 3 * ttb.tenones((1, 2, 1)).to_sptensor()
        >>> S
        sparse tensor of shape (3, 4, 5) with 5 nonzeros and order F
        [0, 0, 0] = 1.0
        [0, 0, 1] = 2.0
        [0, 0, 2] = 2.0
        [0, 1, 3] = 3.0
        [0, 2, 3] = 3.0

        Grow the sparse tensor by assigning an element with a subscript
        outside the current shape:

        >>> S[3, 4, 5] = 4
        >>> S
        sparse tensor of shape (4, 5, 6) with 6 nonzeros and order F
        [0, 0, 0] = 1.0
        [0, 0, 1] = 2.0
        [0, 0, 2] = 2.0
        [0, 1, 3] = 3.0
        [0, 2, 3] = 3.0
        [3, 4, 5] = 4.0

        Assign one or more values using an array of subscripts and a vector
        of values:

        >>> S[S.subs] = 5 * np.ones((S.vals.shape[0], 1))
        >>> S
        sparse tensor of shape (4, 5, 6) with 6 nonzeros and order F
        [0, 0, 0] = 5.0
        [0, 0, 1] = 5.0
        [0, 0, 2] = 5.0
        [0, 1, 3] = 5.0
        [0, 2, 3] = 5.0
        [3, 4, 5] = 5.0

        Note regarding singleton dimensions: It is not possible to do, for
        instance, `S[1,1:10,1:10] = ttb.sptenrand((1,10,10),nonzeros=5)`.
        However, it is okay to do
        `S[1,1:10,1:10] = ttb.sptenrand((1,10,10),nonzeros=5).squeeze()`.
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
        _, tf = tt_ismember_rows(newsubs, self.subs)
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
        # Process Group B: Removing Values
        if np.sum(idxb) > 0:
            removesubs = loc[idxb]
            keepsubs = np.setdiff1d(range(self.nnz), removesubs)
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
            kploc = np.setdiff1d(range(self.nnz), rmloc)
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
        for n in range(self.ndims):
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
            kploc = np.setdiff1d(range(self.nnz), rmloc).astype(int)
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
            for n in range(N):
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
        Element-wise equal operator (==).

        Parameters
        ----------
        other:
            Other object to compare with.

        Examples
        --------
        Compare the :class:`pyttb.sptensor` to itself, returning all `True`
        values:

        >>> S = ttb.sptensor(shape=(2, 2))
        >>> S[1, 1] = 1.0
        >>> S == S
        sparse tensor of shape (2, 2) with 4 nonzeros and order F
        [0, 0] = 1.0
        [0, 1] = 1.0
        [1, 0] = 1.0
        [1, 1] = 1.0

        Compare with a scalar value, returning only a single `True` value:

        >>> S == 1
        sparse tensor of shape (2, 2) with 1 nonzeros and order F
        [1, 1] = 1.0
        """
        # Case 1: other is a scalar
        if isinstance(other, (float, int)):
            if other == 0:
                return self.logical_not()
            idx = self.vals == other
            subs = np.empty(shape=(0, self.ndims), dtype=int)
            if self.nnz > 0:
                subs = self.subs[idx.transpose()[0]]
            return sptensor(
                subs,
                True * np.ones((self.subs.shape[0], 1)).astype(self.vals.dtype),
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
            equal_subs = self.vals[nzsubsIdx] == other.vals[iother]
            znzsubs = np.empty(shape=(0, other.ndims), dtype=int)
            if equal_subs.size > 0:
                znzsubs = nzsubs[(equal_subs).transpose()[0], :]

            return sptensor(
                np.vstack((zzerosubs, znzsubs)),
                True
                * np.ones(zzerosubs.shape[0] + znzsubs.shape[0]).astype(
                    self.vals.dtype
                )[:, None],
                self.shape,
            )

        # Case 2b: other is a dense tensor
        if isinstance(other, ttb.tensor):
            # Find where their zeros interact
            otherzerosubs, _ = (other == 0).find()
            zzerosubs = otherzerosubs[(self[otherzerosubs] == 0).transpose()[0], :]

            # Find where their nonzeros intersect
            znzsubs = np.empty(shape=(0, other.ndims), dtype=int)
            if self.nnz > 0:
                othervals = other[self.subs]
                znzsubs = self.subs[(othervals[:, None] == self.vals).transpose()[0], :]

            return sptensor(
                np.vstack((zzerosubs, znzsubs)),
                True
                * np.ones((zzerosubs.shape[0] + znzsubs.shape[0], 1)).astype(
                    self.vals.dtype
                ),
                self.shape,
            )

        assert False, "Comparison allowed with sptensor, tensor, or scalar only."

    def __ne__(self, other):  # noqa: PLR0912
        """
        Element-wise not equal operator (!=).

        Parameters
        ----------
        other:
            Other object to compare with.

        Examples
        --------
        Compare a :class:`pyttb.sptensor` to itself, returning no `True` values:

        >>> S = ttb.sptensor(shape=(2, 2))
        >>> S[1, 1] = 1.0
        >>> S != S
        empty sparse tensor of shape (2, 2) with order F

        Compare with a scalar value: :

        >>> S != 1
        sparse tensor of shape (2, 2) with 3 nonzeros and order F
        [0, 0] = 1.0
        [0, 1] = 1.0
        [1, 0] = 1.0
        """
        # Case 1: One argument is a scalar
        if isinstance(other, (float, int)):
            if other == 0:
                return ttb.sptensor(
                    self.subs, True * np.ones((self.subs.shape[0], 1)), self.shape
                )
            subs1 = np.empty(shape=(0, self.ndims), dtype=int)
            if self.nnz > 0:
                subs1 = self.subs[self.vals.transpose()[0] != other, :]
            subs2Idx = tt_setdiff_rows(self.allsubs(), self.subs)
            subs2 = self.allsubs()[subs2Idx, :]
            return ttb.sptensor(
                np.vstack((subs1, subs2)),
                True * np.ones((subs2.shape[0], 1)).astype(self.vals.dtype),
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
            self_subs = np.empty(shape=(0, other.ndims), dtype=int)
            if selfIdx.size > 0 and self.nnz > 0:
                self_subs = self.subs[selfIdx]
            other_subs = np.empty(shape=(0, other.ndims), dtype=int)
            if otherIdx.size > 0 and other.nnz > 0:
                other_subs = other.subs[otherIdx]
            subs1 = np.concatenate((self_subs, other_subs))
            # subs1 = setxor(self.subs, other.subs,'rows')
            # find entries where both are nonzero, but inequal
            subs2 = np.empty(shape=(0, other.ndims), dtype=int)
            if self.nnz != 0 and other.nnz != 0:
                subs2 = tt_intersect_rows(self.subs, other.subs)
                subs_pad = np.zeros((self.subs.shape[0],)).astype(bool)
                subs_pad[subs2] = (
                    self.extract(self.subs[subs2]) != other.extract(self.subs[subs2])
                ).transpose()[0]
                subs2 = self.subs[subs_pad, :]
            # put it all together
            return ttb.sptensor(
                np.vstack((subs1, subs2)),
                True
                * np.ones((subs1.shape[0] + subs2.shape[0], 1)).astype(self.vals.dtype),
                self.shape,
            )

        # Case 2b: y is a dense tensor
        if isinstance(other, ttb.tensor):
            # find entries where x is zero but y is nonzero
            unionSubs = tt_union_rows(
                self.subs, np.array(np.where(other.data == 0)).transpose()
            )
            if unionSubs.shape[0] != prod(self.shape):
                subs1Idx = tt_setdiff_rows(self.allsubs(), unionSubs)
                subs1 = self.allsubs()[subs1Idx]
            else:
                subs1 = np.empty((0, self.ndims))
            # find entries where x is nonzero but not equal to y
            subs2 = np.empty((0, self.ndims))
            if self.nnz > 0:
                subs2 = self.subs[self.vals.transpose()[0] != other[self.subs], :]
            if subs2.size == 0:
                subs2 = np.empty((0, self.ndims))
            # put it all together
            return ttb.sptensor(
                np.vstack((subs1, subs2)),
                True
                * np.ones((subs1.shape[0] + subs2.shape[0], 1)).astype(self.vals.dtype),
                self.shape,
            )

        # Otherwise
        assert False, "Comparison allowed with sptensor, tensor, or scalar only."

    def __sub__(self, other):
        """
        Binary subtraction operator (-).

        Parameters
        ----------
        other:
            Object to subtract from the sparse tensor.

        Examples
        --------
        Subtract a :class:`pyttb.sptensor` from itself, returning a sparse
        tensor:

        >>> S = ttb.sptensor(shape=(2, 2))
        >>> S[1, 1] = 1.0
        >>> S - S
        empty sparse tensor of shape (2, 2) with order F

        Subtract a scalar value, returning a dense tensor:

        >>> S - 1
        tensor of shape (2, 2) with order F
        data[:, :] =
        [[-1. -1.]
         [-1.  0.]]
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

        if self.nnz == 0:
            return -other.copy()
        elif other.nnz == 0:
            return self.copy()
        return ttb.sptensor.from_aggregator(
            np.vstack((self.subs, other.subs)),
            np.vstack((self.vals, -1 * other.vals)),
            self.shape,
        )

    def __add__(self, other):
        """
        Binary addition operator (+).

        Parameters
        ----------
        other:
            Object to add to the sparse tensor.

        Examples
        --------
        Add a :class:`pyttb.sptensor` to itself, returning a sparse tensor:

        >>> S = ttb.sptensor(shape=(2, 2))
        >>> S[1, 1] = 1.0
        >>> S + S
        sparse tensor of shape (2, 2) with 1 nonzeros and order F
        [1, 1] = 2.0

        Add a scalar value, returning a dense tensor:

        >>> S + 1
        tensor of shape (2, 2) with order F
        data[:, :] =
        [[1. 1.]
         [1. 2.]]
        """
        # If other is sumtensor perform sumtensor add
        if isinstance(other, ttb.sumtensor):
            return other.__add__(self)
        # Otherwise return negated sub
        return self.__sub__(-other)

    def __radd__(self, other):
        """
        Right binary addition operator (+).

        Parameters
        ----------
        other:
            Object to add to the sparse tensor.

        Examples
        --------
        Add a scalar value, returning a dense tensor:

        >>> S = ttb.sptensor(shape=(2, 2))
        >>> S[1, 1] = 1.0
        >>> 1 + S
        tensor of shape (2, 2) with order F
        data[:, :] =
        [[1. 1.]
         [1. 2.]]
        """
        return self.__add__(other)

    def __pos__(self):
        """
        Unary plus operator (+).

        Examples
        --------
        Create a :class:`pyttb.sptensor`:

        >>> S = ttb.sptensor(shape=(2, 2, 2))
        >>> S[1, 1, 1] = 1
        >>> S
        sparse tensor of shape (2, 2, 2) with 1 nonzeros and order F
        [1, 1, 1] = 1.0

        Apply the + operator:

        >>> +S
        sparse tensor of shape (2, 2, 2) with 1 nonzeros and order F
        [1, 1, 1] = 1.0
        """
        return self.copy()

    def __neg__(self):
        """
        Unary minus operator (-).

        Examples
        --------
        Create a :class:`pyttb.sptensor`:

        >>> S = ttb.sptensor(shape=(2, 2, 2))
        >>> S[1, 1, 1] = 1
        >>> S
        sparse tensor of shape (2, 2, 2) with 1 nonzeros and order F
        [1, 1, 1] = 1.0

        Apply the + operator:

        >>> -S
        sparse tensor of shape (2, 2, 2) with 1 nonzeros and order F
        [1, 1, 1] = -1.0
        """
        return ttb.sptensor(self.subs, -1 * self.vals, self.shape)

    def __mul__(self, other):
        """
        Element-wise multiplication operator (*).

        Parameters
        ----------
        other:
            Object to multiply with the sparsee tensor.

        Examples
        --------
        Multiply a :class:`pyttb.sptensor` by a scalar:

        >>> S = ttb.sptensor(shape=(2, 2))
        >>> S[1, 1] = 1.0
        >>> S * 3
        sparse tensor of shape (2, 2) with 1 nonzeros and order F
        [1, 1] = 3.0

        Multiply two sparse tensors with no overlap in subscripts of
        nonzeros, resulting in an empty sparse tensor:

        >>> S2 = ttb.sptensor(shape=(2, 2))
        >>> S2[1, 0] = 1.0
        >>> S * S2
        empty sparse tensor of shape (2, 2) with order F
        """
        if isinstance(other, (float, int, np.number)):
            return ttb.sptensor(self.subs, self.vals * other, self.shape)

        if (
            isinstance(other, (ttb.sptensor, ttb.tensor, ttb.ktensor))
            and self.shape != other.shape
        ):
            assert False, "Sptensor multiply requires two tensors of the same shape."

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
        Element-wise right multiplication operator (*).

        Parameters
        ----------
        other:
            Object to multiple with sparse tensor.

        Examples
        --------
        Multiple scalar by a :class:`pyttb.sptensor`:

        >>> S = ttb.sptensor(shape=(2, 2))
        >>> S[1, 1] = 1.0
        >>> 3 * S
        sparse tensor of shape (2, 2) with 1 nonzeros and order F
        [1, 1] = 3.0
        """
        if isinstance(other, (float, int, np.number)):
            return self.__mul__(other)
        assert False, "This object cannot be multiplied by sptensor"

    def _compare(self, other, operator, opposite_operator, include_zero=False):  # noqa: PLR0912
        """Generalized Comparison operation.

        Parameters
        ----------
        operator:
            Primary comparison.
        opposite_operator:
            There is some symmetry around zero encoded in the logic.
            This isn't just logical not.
        include_zero:
            Whether or not to treat matching zeros as true.
        """
        if operator not in (ge, gt, le, lt):
            raise ValueError(
                "Internal comparison operator called for unsupported operator"
                f" {operator=}"
            )
        # Case 1: One argument is a scalar
        if isinstance(other, (float, int)):
            subs1 = np.empty(shape=(0, self.ndims), dtype=int)
            if self.nnz > 0:
                subs1 = self.subs[(operator(self.vals, other)).transpose()[0], :]
            if opposite_operator(other, 0):
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
                    subs1 = subs1[
                        np.logical_not(
                            opposite_operator(self.extract(subs1), 0)
                        ).transpose()[0],
                        :,
                    ]
            else:
                subs1 = np.empty(shape=(0, other.ndims), dtype=int)

            # self zero, other not zero
            if other.subs.size > 0:
                subs2 = other.subs[tt_setdiff_rows(other.subs, self.subs), :]
                if subs2.size > 0:
                    subs2 = subs2[
                        np.logical_not(operator(other.extract(subs2), 0)).transpose()[
                            0
                        ],
                        :,
                    ]
            else:
                subs2 = np.empty(shape=(0, self.ndims), dtype=int)

            # self and other not zero
            if self.subs.size > 0:
                subs3 = self.subs[tt_intersect_rows(self.subs, other.subs), :]
                if subs3.size > 0:
                    subs3 = subs3[
                        operator(self.extract(subs3), other.extract(subs3)).transpose()[
                            0
                        ],
                        :,
                    ]
            else:
                subs3 = np.empty(shape=(0, other.ndims), dtype=int)

            if include_zero:
                # self and other zero
                xzerosubs = self.allsubs()[
                    tt_setdiff_rows(self.allsubs(), self.subs), :
                ]
                yzerosubs = other.allsubs()[
                    tt_setdiff_rows(other.allsubs(), other.subs), :
                ]
                subs4 = xzerosubs[tt_intersect_rows(xzerosubs, yzerosubs), :]

                # assemble
                subs = np.vstack((subs1, subs2, subs3, subs4), dtype=int)
            else:
                # assemble
                subs = np.vstack((subs1, subs2, subs3), dtype=int)
            return ttb.sptensor(subs, True * np.ones((len(subs), 1)), self.shape)

        # Case 2b: One dense tensor
        if isinstance(other, ttb.tensor):
            # self zero
            subs1, _ = opposite_operator(other, 0).find()
            subs1 = subs1[tt_setdiff_rows(subs1, self.subs), :]

            # self nonzero
            subs2 = np.empty(shape=(0, self.ndims), dtype=int)
            if self.nnz > 0:
                subs2 = self.subs[
                    operator(self.vals.transpose()[0], other[self.subs]), :
                ]

            # assemble
            subs = np.vstack((subs1, subs2))
            return ttb.sptensor(subs, True * np.ones((len(subs), 1)), self.shape)

        # Otherwise
        assert False, "Comparison allowed with sptensor, tensor, or scalar only."

    def __le__(self, other):
        """
        Less than or equal operator (<=).

        Parameters
        ----------
        other:
            Object to compare with.

        Examples
        --------
        Compare a :class:`pyttb.sptensor` with itself:

        >>> S = ttb.sptensor(shape=(2, 2))
        >>> S[1, 1] = 1.0
        >>> S <= S
        sparse tensor of shape (2, 2) with 4 nonzeros and order F
        [1, 1] = 1.0
        [0, 0] = 1.0
        [0, 1] = 1.0
        [1, 0] = 1.0

        Compare with a scalar:

        >>> S <= -1
        empty sparse tensor of shape (2, 2) with order F
        """
        return self._compare(other, le, ge, True)

    def __lt__(self, other):
        """
        Less than operator (<).

        Parameters
        ----------
        other:
            Object to compare with.

        Examples
        --------
        Compare a :class:`pyttb.sptensor` with itself:

        >>> S = ttb.sptensor(shape=(2, 2))
        >>> S[1, 1] = 1.0
        >>> S < S
        empty sparse tensor of shape (2, 2) with order F

        Compare with a scalar:

        >>> S < 1
        sparse tensor of shape (2, 2) with 3 nonzeros and order F
        [0, 0] = 1.0
        [0, 1] = 1.0
        [1, 0] = 1.0
        """
        return self._compare(other, lt, gt)

    def __ge__(self, other):
        """
        Greater than or equal operator (>=).

        Parameters
        ----------
        other:
            Object to compare with.

        Examples
        --------
        Compare a :class:`pyttb.sptensor` with itself:

        >>> S = ttb.sptensor(shape=(2, 2))
        >>> S[1, 1] = 1.0
        >>> S >= S
        sparse tensor of shape (2, 2) with 4 nonzeros and order F
        [1, 1] = 1.0
        [0, 0] = 1.0
        [0, 1] = 1.0
        [1, 0] = 1.0

        Compare with a scalar:

        >>> S >= 1
        sparse tensor of shape (2, 2) with 1 nonzeros and order F
        [1, 1] = 1.0
        """
        return self._compare(other, ge, le, True)

    def __gt__(self, other):
        """
        Greater than operator (>).

        Parameters
        ----------
        other:
            Object to compare with.

        Examples
        --------
        Compare a :class:`pyttb.sptensor` with itself:

        >>> S = ttb.sptensor(shape=(2, 2))
        >>> S[1, 1] = 1.0
        >>> S > S
        empty sparse tensor of shape (2, 2) with order F

        Compare with a scalar:

        >>> S > 0
        sparse tensor of shape (2, 2) with 1 nonzeros and order F
        [1, 1] = 1.0
        """
        return self._compare(other, gt, lt)

    def __truediv__(self, other):  # noqa: PLR0912, PLR0915
        """Element-wise left division operator (/).

        Comparisons with empty tensors raise an exception.

        Parameters
        ----------
        other:
            Object to divide from the sparse tensor.

        Examples
        --------
        Divide a :class:`pyttb.sptensor` by a :class:`pyttb.sptensor`:

        >>> S = ttb.sptensor(shape=(2, 2))
        >>> S[1, 1] = 2.0
        >>> S2 = ttb.sptensor(shape=(2, 2))
        >>> S2[1, 1] = 4.0
        >>> S / S2
        sparse tensor of shape (2, 2) with 4 nonzeros and order F
        [1, 1] = 0.5
        [0, 0] = nan
        [0, 1] = nan
        [1, 0] = nan

        Divide by a scalar:

        >>> S / 3  # doctest: +ELLIPSIS
        sparse tensor of shape (2, 2) with 1 nonzeros and order F
        [1, 1] = 0.66666...
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
        Element-wise right division operator (/).

        Parameters
        ----------
        other:
            Object to divide sparse tensor by.

        Examples
        --------
        Divide a scalar by a :class:`pyttb.sptensor`:

        >>> S = ttb.sptensor(shape=(2, 2))
        >>> S[:, :] = 2.0
        >>> 1 / S
        tensor of shape (2, 2) with order F
        data[:, :] =
        [[0.5 0.5]
         [0.5 0.5]]
        """
        # Scalar divided by a tensor -> result is dense
        if isinstance(other, (float, int)):
            return other / self.full()
        assert False, "Dividing that object by an sptensor is not supported"

    def __repr__(self):  # pragma: no cover
        """Return string representation of a :class:`pyttb.sptensor`.

        Examples
        --------
        Create a :class:`pyttb.sptensor` and print it as a string:

        >>> S = ttb.sptensor(shape=(2, 2))
        >>> S[:, :] = 1.0
        >>> print(S)
        sparse tensor of shape (2, 2) with 4 nonzeros and order F
        [0, 0] = 1.0
        [0, 1] = 1.0
        [1, 0] = 1.0
        [1, 1] = 1.0
        """
        nz = self.nnz
        if nz == 0:
            s = (
                f"empty sparse tensor of shape {np_to_python(self.shape)!r}"
                " with order F"
            )
            return s

        s = f"sparse tensor of shape {np_to_python(self.shape)!r}"
        s += f" with {nz} nonzeros and order {self.order}\n"

        # Stop insane printouts
        if nz > 10000:
            r = input("Are you sure you want to print all nonzeros? (Y/N)")
            if r.upper() != "Y":
                return s
        for i in range(self.subs.shape[0]):
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
        matrices: np.ndarray | Sequence[np.ndarray],
        dims: OneDArray | None = None,
        exclude_dims: OneDArray | None = None,
        transpose: bool = False,
    ) -> ttb.tensor | sptensor:
        """
        Multiplication of a :class:`pyttb.sptensor` with a matrix.

        Computes the n-mode product of the :class:`pyttb.sptensor` with a
        matrix (i.e., array). Let `n` specify the dimension (or mode) along
        which the matrix should be multiplied. If the matrix has `shape = (I,J)`,
        then the sparse tensor must have `shape[n] = I`. If the matrix has
        `shape = (J,I)`, you can set `transpose=True` to multiply with the
        transpose of the matrix. The result has `shape[n] = J`.

        Multiplication with more than one matrix is provided using a list of
        matrices and corresponding dimensions in the sparse tensor to use.

        The dimensions of the sparse tensor with which to multiply can be provided as
        `dims`, or the dimensions to exclude from `[0, ..., self.ndims]` can be
        specified using `exclude_dims`.

        Parameters
        ----------
        matrices:
            A matrix or list of matrices.
        dims:
            Dimensions to multiply against.
        exclude_dims:
            Use all dimensions but these.
        transpose:
            Transpose matrices to be multiplied.

        Examples
        --------
        Create a :class:`pyttb.sptensor` with a region of elements set to 1:

        >>> S = ttb.sptensor(shape=(2, 2, 2, 2))
        >>> S[:, 0:1, :, 0:1] = 1

        Compute the product of `S` with multiple matrices of ones along the
        first two dimensions, transposing the matrices when multiplying:

        >>> A = 2 * np.ones((2, 1))
        >>> S.ttm([A, A], dims=[0, 1], transpose=True)
        tensor of shape (1, 1, 2, 2) with order F
        data[:, :, 0, 0] =
        [[8.]]
        data[:, :, 1, 0] =
        [[8.]]
        data[:, :, 0, 1] =
        [[0.]]
        data[:, :, 1, 1] =
        [[0.]]

        Compute sparse tensor matrix product specifying which two tensor
        dimensions to exclude in the multiplication:

        >>> S.ttm([A, A], exclude_dims=[0, 1], transpose=True)
        tensor of shape (2, 2, 1, 1) with order F
        data[:, :, 0, 0] =
        [[8. 0.]
         [8. 0.]]
        """
        # Handle list of matrices
        if isinstance(matrices, Sequence):
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

        # This is slightly inefficient for the looping above
        # consider short circuit
        dims, _ = tt_dimscheck(self.ndims, None, dims, exclude_dims)

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
        Xnt = self.to_sptenmat(np.array([final_dim]), cdims_cyclic="t")

        # Convert to sparse matrix and do multiplication; generally result is sparse
        Z = Xnt.double().dot(matrices.transpose())

        # Rearrange back into sparse tensor of correct shape
        Ynt = ttb.sptenmat.from_array(Z, Xnt.rdims, Xnt.cdims, tuple(siz)).to_sptensor()

        if not isinstance(Z, np.ndarray) and Z.nnz <= 0.5 * prod(siz):
            return Ynt
        # TODO evaluate performance loss by casting into sptensor then tensor.
        #  I assume minimal since we are already using sparse matrix representation
        return Ynt.to_tensor()

    @overload
    def squash(
        self, return_inverse: Literal[False]
    ) -> sptensor: ...  # pragma: no cover see coveragepy/issues/970

    @overload
    def squash(
        self, return_inverse: Literal[True]
    ) -> tuple[sptensor, dict]: ...  # pragma: no cover see coveragepy/issues/970

    def squash(self, return_inverse: bool = False) -> sptensor | tuple[sptensor, dict]:
        """
        Remove empty slices from a :class:`pyttb.sptensor`.

        Parameters
        ----------
        return_inverse:
            Return mapping from new tensor to old tensor subscripts.

        Examples
        --------
        Create a :class:`pyttb.sptensor` with a few entries and squash empty
        slices:

        >>> S = ttb.sptensor(shape=(10, 10, 10))
        >>> S[0, 1, 2] = 1
        >>> S[0, 1, 3] = 2
        >>> S
        sparse tensor of shape (10, 10, 10) with 2 nonzeros and order F
        [0, 1, 2] = 1.0
        [0, 1, 3] = 2.0
        >>> S.squash()
        sparse tensor of shape (2, 2, 2) with 2 nonzeros and order F
        [0, 0, 0] = 1.0
        [0, 0, 1] = 2.0

        Squash and return the inverse subscript mapping, checking that the
        mapping in all dimensions is correct:

        >>> S2, inverse = S.squash(True)
        >>> for i in range(S.ndims):
        ...     np.array_equal(S.subs[:, i], inverse[i][S2.subs[:, i]])
        True
        True
        True
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
    shape: Shape,
    density: float | None = None,
    nonzeros: float | None = None,
) -> sptensor:
    """Create a :class:`pyttb.sptensor` with random entries and indices.

    Entries drawn from a uniform
    distribution on the unit interval and indices selected using a uniform
    distribution. You can specify the density or number of nonzeros in the
    resulting sparse tensor but not both.

    Parameters
    ----------
    shape:
        Shape of resulting sparse tensor.
    density:
        Density of resulting sparse tensor.
    nonzeros:
        Number of nonzero entries in resulting sparse tensor.

    Examples
    --------
    Create a :class:`pyttb.sptensor`, specifying the number of nonzeros:

    >>> S = ttb.sptenrand((2, 2), nonzeros=1)

    Create a :class:`pyttb.sptensor`, specifying the density of nonzeros:

    >>> S2 = ttb.sptenrand((2, 2), density=0.25)
    """
    if density is None and nonzeros is None:
        raise ValueError("Must set either density or nonzeros")

    if density is not None and nonzeros is not None:
        raise ValueError("Must set either density or nonzeros but not both")

    if density is not None and not 0 < density <= 1:
        raise ValueError(f"Density must be a fraction (0, 1] but received {density}")

    shape = parse_shape(shape)
    if isinstance(density, float):
        # TODO this should be an int
        valid_nonzeros = float(prod(shape) * density)
    elif isinstance(nonzeros, (int, float)):
        valid_nonzeros = nonzeros
    else:  # pragma: no cover
        raise ValueError(
            f"Incorrect types for density:{density} and nonzeros:{nonzeros}"
        )

    # Typing doesn't play nice with partial
    # mypy issue: 1484
    def unit_uniform(pass_through_shape: tuple[int, ...]) -> np.ndarray:
        return np.random.uniform(low=0, high=1, size=pass_through_shape)

    return ttb.sptensor.from_function(unit_uniform, shape, valid_nonzeros)


def sptendiag(elements: OneDArray, shape: Shape | None = None) -> sptensor:
    """Create a :class:`pyttb.sptensor` with elements along the super diagonal.

    If provided shape is too small the sparse tensor will be enlarged to
    accommodate.

    Parameters
    ----------
    elements:
        Elements to set along the diagonal.
    shape:
        Shape of the resulting sparse tensor.

    Examples
    --------
    Create a :class:`pyttb.sptensor` by specifying the super diagonal with a
    1-D array that has 2 elements, which will create a 2x2 sparse tensor:

    >>> shape = (2,)
    >>> values = np.ones(shape)
    >>> S = ttb.sptendiag(values)

    Create a 2x2 :class:`pyttb.sptensor`, specifying the correct shape, and
    verify that it is equal to `S`:

    >>> S2 = ttb.sptendiag(values, (2, 2))
    >>> S.isequal(S2)
    True
    """
    # Flatten provided elements
    elements = parse_one_d(elements)
    N = len(elements)
    if shape is None:
        constructed_shape = (N,) * N
    else:
        shape = parse_shape(shape)
        constructed_shape = tuple(max(N, dim) for dim in shape)
    subs = np.tile(np.arange(0, N).transpose(), (len(constructed_shape), 1)).transpose()
    return sptensor.from_aggregator(subs, elements.reshape((N, 1)), constructed_shape)


if __name__ == "__main__":
    import doctest  # pragma: no cover

    doctest.testmod()  # pragma: no cover
