"""Classes and functions for working with matricized sparse tensors."""

# Copyright 2025 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy_groupies import aggregate as accumarray
from scipy import sparse

import pyttb as ttb
from pyttb.pyttb_utils import (
    gather_wrap_dims,
    np_to_python,
    tt_ind2sub,
)


class sptenmat:
    """Store sparse tensor as a sparse matrix."""

    __slots__ = ("cdims", "rdims", "subs", "tshape", "vals")

    def __init__(  # noqa: PLR0913
        self,
        subs: np.ndarray | None = None,
        vals: np.ndarray | None = None,
        rdims: np.ndarray | None = None,
        cdims: np.ndarray | None = None,
        tshape: tuple[int, ...] = (),
        copy: bool = True,
    ):
        """Construct a :class:`pyttb.sptenmat`.

        Constructed from a set of 2D subscripts (subs)
        and values (vals) along with the mappings of the row (rdims) and column
        indices (cdims) and the shape of the original tensor (tshape).

        If you already have an sparse tensor see :meth:`pyttb.sptensor.to_sptenmat`.

        Parameters
        ----------
        subs:
            Location of non-zero entries, in sptenmat.
        vals:
            Values for non-zero entries, in sptenmat.
        rdims:
            Mapping of row indices.
        cdims:
            Mapping of column indices.
        tshape:
            Shape of the original tensor.
        copy:
            Whether to make a copy of provided data or just reference it.
            Skips error checking when just setting reference.

        Examples
        --------
        Create an empty :class:`pyttb.sptenmat`:

        >>> S = ttb.sptenmat()
        >>> S # doctest: +NORMALIZE_WHITESPACE
        sptenmat corresponding to a sptensor of shape () with 0 nonzeros and order F
        rdims = [  ] (modes of sptensor corresponding to rows)
        cdims = [  ] (modes of sptensor corresponding to columns)

        Create a :class:`pyttb.sptenmat` from subscripts, values, and unwrapping
            dimensions:

        >>> subs = np.array([[1, 6], [1, 7]])
        >>> vals = np.array([[6], [7]])
        >>> tshape = (4, 4, 4)
        >>> S = ttb.sptenmat(\
            subs,\
            vals,\
            rdims=np.array([0]),\
            cdims=np.array([1,2]),\
            tshape=tshape\
        )
        >>> S # doctest: +NORMALIZE_WHITESPACE
        sptenmat corresponding to a sptensor of shape (4, 4, 4) with 2 nonzeros \
        and order F
        rdims = [ 0 ] (modes of sptensor corresponding to rows)
        cdims = [ 1, 2 ] (modes of sptensor corresponding to columns)
            [1, 6] = 6
            [1, 7] = 7
        """
        # Empty case
        if rdims is None and cdims is None:
            assert subs is None and vals is None, (
                "Must provide rdims or cdims with values"
            )
            self.subs = np.array([], ndmin=2, dtype=int)
            self.vals = np.array([], ndmin=2)
            self.rdims = np.array([], dtype=int)
            self.cdims = np.array([], dtype=int)
            self.tshape: tuple[()] | tuple[int, ...] = ()
            return

        if subs is None:
            subs = np.array([], ndmin=2, dtype=int)
        if vals is None:
            vals = np.array([], ndmin=2)

        n = len(tshape)
        alldims = np.array([range(n)])
        # Error check
        rdims, cdims = gather_wrap_dims(n, rdims, cdims)
        # if rdims or cdims is empty, hstack will output an array of float not int
        if rdims.size == 0:
            dims = cdims.copy("K")
        elif cdims.size == 0:
            dims = rdims.copy("K")
        else:
            dims = np.hstack([rdims, cdims], dtype=int)
        assert len(dims) == n and (alldims == np.sort(dims)).all(), (
            "Incorrect specification of dimensions, the sorted concatenation of "
            "rdims and cdims must be range(len(tshape))."
        )
        assert subs.size == 0 or np.prod(np.array(tshape)[rdims]) >= np.max(
            subs[:, 0]
        ), "Invalid row index."
        assert subs.size == 0 or np.prod(np.array(tshape)[cdims]) >= np.max(
            subs[:, 1]
        ), "Invalid column index."

        # Sum any duplicates
        newsubs = subs
        newvals = vals
        if vals.size == 0:
            assert vals.size == 0, "Empty subs requires empty vals"
            newsubs = np.array([])
            newvals = np.array([])
        elif copy:
            # Identify only the unique indices
            newsubs, loc = np.unique(subs, axis=0, return_inverse=True)
            # Sum the corresponding values
            # Squeeze to convert from column vector to row vector
            newvals = accumarray(
                loc.flatten(), np.squeeze(vals, axis=1), size=newsubs.shape[0], func=sum
            )

        if copy:
            # Find the nonzero indices of the new values
            nzidx = np.nonzero(newvals)
            newsubs = newsubs[nzidx]
            # None index to convert from row back to column vector
            newvals = newvals[nzidx]
            if newvals.size > 0:
                newvals = newvals[:, None]

            self.tshape = tshape
            self.rdims = rdims.copy("K").astype(int)
            self.cdims = cdims.copy("K").astype(int)
            self.subs = newsubs
            self.vals = newvals
        else:
            self.tshape = tshape
            self.rdims = rdims
            self.cdims = cdims
            self.subs = newsubs
            self.vals = newvals

    @classmethod
    def from_array(
        cls,
        array: sparse.coo_matrix | np.ndarray,
        rdims: np.ndarray | None = None,
        cdims: np.ndarray | None = None,
        tshape: tuple[int, ...] = (),
    ):
        """Construct a :class:`pyttb.sptenmat`.

        Constructed from a coo_matrix
        along with the mappings of the row (rdims) and column
        indices (cdims) and the shape of the original tensor (tshape).

        Parameters
        ----------
        array:
            Representation of sparse tensor data (sparse or dense).
        rdims:
            Mapping of row indices.
        cdims:
            Mapping of column indices.
        tshape:
            Shape of the original tensor.

        Examples
        --------
        Create a :class:`pyttb.sptenmat` from a sparse matrix and unwrapping
            dimensions. Infer column dimensions from row dimensions specification.

        >>> data = np.array([6, 7])
        >>> rows = np.array([1, 1])
        >>> cols = np.array([6, 7])
        >>> sparse_matrix = sparse.coo_matrix((data, (rows, cols)))
        >>> tshape = (4, 4, 4)
        >>> S = ttb.sptenmat.from_array(\
            sparse_matrix,\
            rdims=np.array([0]),\
            tshape=tshape\
        )
        >>> S # doctest: +NORMALIZE_WHITESPACE
        sptenmat corresponding to a sptensor of shape (4, 4, 4) with 2 nonzeros \
        and order F
        rdims = [ 0 ] (modes of sptensor corresponding to rows)
        cdims = [ 1, 2 ] (modes of sptensor corresponding to columns)
            [1, 6] = 6
            [1, 7] = 7
        """
        vals = None
        if isinstance(array, np.ndarray):
            vals = np.expand_dims(array[array.nonzero()], axis=1)
        elif sparse.issparse(array):
            vals = np.expand_dims(array.tocoo(False).data, axis=1)
        else:
            raise ValueError(
                f"Expected sparse matrix or array but received: {type(array)}"
            )
        subs = np.vstack(array.nonzero()).transpose()
        return ttb.sptenmat(subs, vals, rdims, cdims, tshape)

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

    def copy(self) -> sptenmat:
        """
        Return a deep copy of the :class:`pyttb.sptenmat`.

        Examples
        --------
        Create a :class:`pyttb.sptenmat` (ST1) and make a deep copy. Verify
        the deep copy (ST3) is not just a reference (like ST2) to the original.

        >>> S1 = ttb.sptensor(shape=(2, 2))
        >>> S1[0, 0] = 1
        >>> ST1 = S1.to_sptenmat(np.array([0]))
        >>> ST2 = ST1
        >>> ST3 = ST1.copy()
        >>> ST1[0, 0] = 3
        >>> ST1.to_sptensor().isequal(ST2.to_sptensor())
        True
        >>> ST1.to_sptensor().isequal(ST3.to_sptensor())
        False
        """
        return sptenmat(
            self.subs,
            self.vals,
            self.rdims,
            self.cdims,
            self.tshape,
            copy=True,
        )

    def __deepcopy__(self, memo):
        """Return deepcopy of this sptenmat."""
        return self.copy()

    def to_sptensor(self) -> ttb.sptensor:
        """Construct a :class:`pyttb.sptensor` from :class:`pyttb.sptenmat`.

        Examples
        --------
        >>> S1 = ttb.sptensor(shape=(2, 2, 2))
        >>> S1[0, 0, 0] = 1
        >>> S1  # doctest: +NORMALIZE_WHITESPACE
        sparse tensor of shape (2, 2, 2) with 1 nonzeros and order F
        [0, 0, 0] = 1.0
        >>> ST1 = S1.to_sptenmat(np.array([0]))
        >>> ST1  # doctest: +NORMALIZE_WHITESPACE
        sptenmat corresponding to a sptensor of shape (2, 2, 2) with 1 nonzeros \
        and order F
        rdims = [ 0 ] (modes of sptensor corresponding to rows)
        cdims = [ 1, 2 ] (modes of sptensor corresponding to columns)
            [0, 0] = 1.0
        >>> ST1.to_sptensor()  # doctest: +NORMALIZE_WHITESPACE
        sparse tensor of shape (2, 2, 2) with 1 nonzeros and order F
        [0, 0, 0] = 1.0
        """
        vals = None
        subs = None
        if self.subs.size > 0:
            tshape = np.array(self.tshape)
            rdims = tt_ind2sub(tshape[self.rdims], self.subs[:, 0])
            cdims = tt_ind2sub(tshape[self.cdims], self.subs[:, 1])
            subs = np.zeros(
                (rdims.shape[0], rdims.shape[1] + cdims.shape[1]), dtype=int
            )
            subs[:, self.rdims] = rdims
            subs[:, self.cdims] = cdims
            vals = self.vals
        return ttb.sptensor(subs, vals, self.tshape)

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Return the shape of a :class:`pyttb.sptenmat`.

        Examples
        --------
        >>> ttb.sptenmat().shape  # empty sptenmat
        ()
        >>> S1 = ttb.sptensor(shape=(2, 2, 2))
        >>> S1[0, 0, 0] = 1
        >>> ST1 = S1.to_sptenmat(np.array([0]))
        >>> ST1.shape
        (2, 4)
        """
        if self.tshape == ():
            return ()
        else:
            m = np.prod(np.array(self.tshape)[self.rdims])
            n = np.prod(np.array(self.tshape)[self.cdims])
            return int(m), int(n)

    def double(self, immutable: bool = False) -> sparse.coo_matrix:  # noqa: ARG002
        """
        Convert a :class:`pyttb.sptenmat` to a COO :class:`scipy.sparse.coo_matrix`.

        Parameters
        ----------
        immutable: Parameter for compatibility but coo_matrix doesn't allow assignment.

        Examples
        --------
        >>> S1 = ttb.sptensor(shape=(2, 2, 2))
        >>> S1[0, 0, 0] = 1
        >>> ST1 = S1.to_sptenmat(np.array([0]))
        >>> COO = ST1.double()
        >>> type(COO)  # doctest: +NORMALIZE_WHITESPACE
        <class 'scipy.sparse._coo.coo_matrix'>
        >>> COO.nnz  # doctest: +NORMALIZE_WHITESPACE
        1
        >>> COO.toarray()  # doctest: +NORMALIZE_WHITESPACE
        array([[1., 0., 0., 0.],
            [0., 0., 0., 0.]])
        """
        if self.subs.size == 0:
            return sparse.coo_matrix(self.shape)
        return sparse.coo_matrix(
            (self.vals.transpose()[0], self.subs.transpose()), self.shape
        )

    def full(self) -> ttb.tenmat:
        """
        Convert a :class:`pyttb.sptenmat` to a (dense) :class:`pyttb.tenmat`.

        Examples
        --------
        >>> S1 = ttb.sptensor(shape=(2, 2, 2))
        >>> S1[0, 0, 0] = 1
        >>> ST1 = S1.to_sptenmat(np.array([0]))
        >>> ST1.full()  # doctest: +NORMALIZE_WHITESPACE
        matrix corresponding to a tensor of shape (2, 2, 2) with order F
        rindices = [ 0 ] (modes of tensor corresponding to rows)
        cindices = [ 1, 2 ] (modes of tensor corresponding to columns)
        data[:, :] =
        [[1. 0. 0. 0.]
         [0. 0. 0. 0.]]
        """
        # Create empty dense tenmat
        result = ttb.tenmat(
            np.zeros(self.shape, order=self.order), self.rdims, self.cdims, self.tshape
        )
        # Assign nonzero values
        result[tuple(self.subs.transpose())] = np.squeeze(self.vals)
        return result

    @property
    def nnz(self) -> int:
        """
        Number of nonzero values in the :class:`pyttb.sptenmat`.

        Examples
        --------
        >>> S1 = ttb.sptensor(shape=(2, 2, 2))
        >>> S1[0, 0, 0] = 1
        >>> ST1 = S1.to_sptenmat(np.array([0]))
        >>> ST1.nnz
        1
        """
        return len(self.vals)

    def norm(self) -> float:
        """Compute the norm of the :class:`pyttb.sptenmat`.

        Frobenius norm, or square root of the sum of
        squares of entries.

        Examples
        --------
        >>> S1 = ttb.sptensor(shape=(2, 2, 2))
        >>> S1[0, 0, 0] = 1
        >>> ST1 = S1.to_sptenmat(np.array([0]))
        >>> ST1.norm()
        1.0
        """
        return np.linalg.norm(self.vals).item()

    def isequal(self, other: sptenmat) -> bool:
        """
        Exact equality for :class:`pyttb.sptenmat`.

        Examples
        --------
        >>> S1 = ttb.sptensor(shape=(2, 2, 2))
        >>> S1[0, 0, 0] = 1
        >>> ST1 = S1.to_sptenmat(np.array([0]))
        >>> ST2 = ttb.sptenmat()
        >>> ST1.isequal(ST2)
        False
        >>> ST1.isequal(ST1)
        True
        """
        if not isinstance(other, ttb.sptenmat):
            raise ValueError(
                f"Can only compares against other sptenmat but received: {type(other)}"
            )
        return (
            np.array_equal(self.vals, other.vals)
            and np.array_equal(self.subs, other.subs)
            and self.tshape == other.tshape
            and np.array_equal(self.cdims, other.cdims)
            and np.array_equal(self.rdims, other.rdims)
        )

    def __pos__(self):
        """
        Unary plus operator (+).

        Examples
        --------
        >>> S1 = ttb.sptensor(shape=(2, 2, 2))
        >>> S1[0, 0, 0] = 1
        >>> ST1 = S1.to_sptenmat(np.array([0]))
        >>> +ST1  # doctest: +NORMALIZE_WHITESPACE
        sptenmat corresponding to a sptensor of shape (2, 2, 2) with 1 nonzeros \
        and order F
        rdims = [ 0 ] (modes of sptensor corresponding to rows)
        cdims = [ 1, 2 ] (modes of sptensor corresponding to columns)
            [0, 0] = 1.0
        """
        return self.copy()

    def __neg__(self):
        """
        Unary minus operator (-).

        Examples
        --------
        >>> S1 = ttb.sptensor(shape=(2, 2, 2))
        >>> S1[0, 0, 0] = 1
        >>> ST1 = S1.to_sptenmat(np.array([0]))
        >>> -ST1  # doctest: +NORMALIZE_WHITESPACE
        sptenmat corresponding to a sptensor of shape (2, 2, 2) with 1 nonzeros \
        and order F
        rdims = [ 0 ] (modes of sptensor corresponding to rows)
        cdims = [ 1, 2 ] (modes of sptensor corresponding to columns)
            [0, 0] = -1.0
        """
        result = self.copy()
        result.vals *= -1
        return result

    def __setitem__(self, key, value):  # noqa: PLR0912
        """
        Subscripted assignment for the :class:`pyttb.sptenmat`.

        Examples
        --------
        Create an empty :class:`pyttb.sptenmat`.

        >>> ST = ttb.sptenmat(rdims=np.array([0]), tshape=(2, 2, 2))
        >>> ST  # doctest: +NORMALIZE_WHITESPACE
        sptenmat corresponding to a sptensor of shape (2, 4) with 0 nonzeros and order F
        rdims = [ 0 ] (modes of sptensor corresponding to rows)
        cdims = [ 1, 2 ] (modes of sptensor corresponding to columns)

        Insert a new value into it.

        >>> ST[0, 0] = 1.0
        >>> ST  # doctest: +NORMALIZE_WHITESPACE
        sptenmat corresponding to a sptensor of shape (2, 2, 2) with 1 nonzeros \
        and order F
        rdims = [ 0 ] (modes of sptensor corresponding to rows)
        cdims = [ 1, 2 ] (modes of sptensor corresponding to columns)
            [0, 0] = 1.0

        Update an existing value in it.

        >>> ST[0, 0] = 2.0
        >>> ST  # doctest: +NORMALIZE_WHITESPACE
        sptenmat corresponding to a sptensor of shape (2, 2, 2) with 1 nonzeros \
        and order F
        rdims = [ 0 ] (modes of sptensor corresponding to rows)
        cdims = [ 1, 2 ] (modes of sptensor corresponding to columns)
            [0, 0] = 2.0
        """
        if not isinstance(key, tuple):
            raise IndexError("Sptenmat takes two arguments as a 2D array")
        if len(key) != 2:
            raise IndexError(
                f"Wrong number of indices. Expected 2 received: {len(key)}"
            )
        rsubs = key[0]
        if isinstance(rsubs, slice):
            rsubs = np.arange(0, self.shape[0])[rsubs]
        rsubs = np.asarray(rsubs, dtype=int)
        if rsubs.shape == ():
            rsubs = np.array([rsubs])

        csubs = key[1]
        if isinstance(csubs, slice):
            csubs = np.arange(0, self.shape[1])[csubs]
        csubs = np.asarray(csubs, dtype=int)
        if csubs.shape == ():
            csubs = np.array([csubs])

        if isinstance(value, (int, float, np.floating)):
            value = value * np.ones((len(csubs) * len(rsubs), 1))
        value = np.asarray(value)

        newsubs = []
        newvals = []

        k = -1

        # Loop over row and column indices, finding appropriate row index for (i,j)
        for j in range(len(csubs)):
            indxc = np.array([], dtype=int)
            if self.subs.size > 0:
                indxc = np.where(self.subs[:, 1] == csubs[j])[0]
            for i in range(len(rsubs)):
                indxr = np.array([], dtype=int)
                if self.subs.size > 0:
                    indxr = np.where(self.subs[indxc, 0] == rsubs[i])[0]
                indx = indxc[indxr]

                k += 1

                if indx.size == 0:
                    newsubs.append(np.hstack([rsubs[i], csubs[j]]))
                    newvals.append(value[k])
                else:
                    self.vals[indx] = value[k]

        # If there are new values to append, then add them on and sort
        if len(newvals) != 0:
            if self.subs.size > 0:
                self.subs = np.vstack((self.subs, newsubs))
                self.vals = np.vstack((self.vals, newvals))
            else:
                self.subs = np.vstack(newsubs)
                self.vals = np.vstack(newvals)
            sort_idx = np.lexsort(self.subs.transpose()[::-1])
            self.subs = self.subs[sort_idx]
            self.vals = self.vals[sort_idx]

    def __repr__(self):
        """Return string representation of a :class:`pyttb.sptenmat`.

        Examples
        --------
        >>> ttb.sptenmat()  # doctest: +NORMALIZE_WHITESPACE
        sptenmat corresponding to a sptensor of shape () with 0 nonzeros \
        and order F
        rdims = [  ] (modes of sptensor corresponding to rows)
        cdims = [  ] (modes of sptensor corresponding to columns)
        >>> S1 = ttb.sptensor(shape=(2, 2, 2))
        >>> S1[0, 0, 0] = 1
        >>> ST1 = S1.to_sptenmat(np.array([0]))
        >>> ST1  # doctest: +NORMALIZE_WHITESPACE
        sptenmat corresponding to a sptensor of shape (2, 2, 2) with 1 nonzeros \
        and order F
        rdims = [ 0 ] (modes of sptensor corresponding to rows)
        cdims = [ 1, 2 ] (modes of sptensor corresponding to columns)
            [0, 0] = 1.0

        Returns
        -------
        str
            Contains the shape, row indices (rindices), column indices (cindices)
                and data as strings on different lines.
        """
        s = ""
        s += "sptenmat corresponding to a sptensor of shape "
        if self.vals.size == 0:
            s += str(np_to_python(self.shape))
        else:
            s += f"{np_to_python(self.tshape)!r}"
        s += f" with {self.vals.size} nonzeros and order {self.order}\n"

        s += "rdims = "
        s += "[ " + (", ").join([str(int(d)) for d in self.rdims]) + " ] "
        s += "(modes of sptensor corresponding to rows)\n"

        s += "cdims = "
        s += "[ " + (", ").join([str(int(d)) for d in self.cdims]) + " ] "
        s += "(modes of sptensor corresponding to columns)\n"

        # Stop insane printouts
        if self.vals.size > 10000:  # pragma: no cover
            r = input("Are you sure you want to print all nonzeros? (Y/N)")
            if r.upper() != "Y":
                return s

        # An empty ndarray with minimum dimensions still has a shape
        if self.subs.size > 0:
            for i in range(self.subs.shape[0]):
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


if __name__ == "__main__":
    import doctest  # pragma: no cover

    doctest.testmod()  # pragma: no cover
