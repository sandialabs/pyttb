# Copyright 2022 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

"""Classes and functions for working with Kruskal tensors."""
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from numpy_groupies import aggregate as accumarray
from scipy import sparse

import pyttb as ttb
from pyttb.pyttb_utils import gather_wrap_dims, tt_ind2sub


class sptenmat:
    """
    SPTENMAT Store sparse tensor as a sparse matrix.

    """

    __slots__ = ("tshape", "rdims", "cdims", "subs", "vals")

    def __init__(  # noqa: PLR0913
        self,
        subs: Optional[np.ndarray] = None,
        vals: Optional[np.ndarray] = None,
        rdims: Optional[np.ndarray] = None,
        cdims: Optional[np.ndarray] = None,
        tshape: Tuple[int, ...] = (),
    ):
        """
        Construct a :class:`pyttb.sptenmat` from a set of 2D subscripts (subs)
        and values (vals) along with the mappings of the row (rdims) and column
        indices (cdims) and the shape of the original tensor (tshape).

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

        Examples
        --------
        Create an empty :class:`pyttb.sptenmat`:

        >>> S = ttb.sptenmat()
        >>> S # doctest: +NORMALIZE_WHITESPACE
        sptenmat corresponding to a sptensor of shape () with 0 nonzeros
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
        sptenmat corresponding to a sptensor of shape (4, 4, 4) with 2 nonzeros
        rdims = [ 0 ] (modes of sptensor corresponding to rows)
        cdims = [ 1, 2 ] (modes of sptensor corresponding to columns)
            [1, 6] = 6
            [1, 7] = 7
        """
        # Empty case
        if rdims is None and cdims is None:
            assert (
                subs is None and vals is None
            ), "Must provide rdims or cdims with values"
            self.subs = np.array([], ndmin=2, dtype=int)
            self.vals = np.array([], ndmin=2)
            self.rdims = np.array([], dtype=int)
            self.cdims = np.array([], dtype=int)
            self.tshape: Union[Tuple[()], Tuple[int, ...]] = ()
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
            dims = cdims.copy()
        elif cdims.size == 0:
            dims = rdims.copy()
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
        if vals.size == 0:
            assert vals.size == 0, "Empty subs requires empty vals"
            newsubs = np.array([])
            newvals = np.array([])
        else:
            # Identify only the unique indices
            newsubs, loc = np.unique(subs, axis=0, return_inverse=True)
            # Sum the corresponding values
            # Squeeze to convert from column vector to row vector
            newvals = accumarray(
                loc.flatten(), np.squeeze(vals, axis=1), size=newsubs.shape[0], func=sum
            )

        # Find the nonzero indices of the new values
        nzidx = np.nonzero(newvals)
        newsubs = newsubs[nzidx]
        # None index to convert from row back to column vector
        newvals = newvals[nzidx]
        if newvals.size > 0:
            newvals = newvals[:, None]

        self.tshape = tshape
        self.rdims = rdims.copy().astype(int)
        self.cdims = cdims.copy().astype(int)
        self.subs = newsubs
        self.vals = newvals

    @classmethod
    def from_array(
        cls,
        array: Union[sparse.coo_matrix, np.ndarray],
        rdims: Optional[np.ndarray] = None,
        cdims: Optional[np.ndarray] = None,
        tshape: Tuple[int, ...] = (),
    ):
        """
        Construct a :class:`pyttb.sptenmat` from a coo_matrix
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
        sptenmat corresponding to a sptensor of shape (4, 4, 4) with 2 nonzeros
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

    def copy(self) -> sptenmat:
        """
        Return a deep copy of the :class:`pyttb.sptenmat`.

        Examples
        --------
        Create a :class:`pyttb.sptenmat` (ST1) and make a deep copy. Verify
        the deep copy (ST3) is not just a reference (like ST2) to the original.

        >>> S1 = ttb.sptensor(shape=(2,2))
        >>> S1[0,0] = 1
        >>> ST1 = S1.to_sptenmat(np.array([0]))
        >>> ST2 = ST1
        >>> ST3 = ST1.copy()
        >>> ST1[0,0] = 3
        >>> ST1.to_sptensor().isequal(ST2.to_sptensor())
        True
        >>> ST1.to_sptensor().isequal(ST3.to_sptensor())
        False
        """
        return sptenmat(
            self.subs.copy(),
            self.vals.copy(),
            self.rdims.copy(),
            self.cdims.copy(),
            self.tshape,
        )

    def __deepcopy__(self, memo):
        return self.copy()

    def to_sptensor(self) -> ttb.sptensor:
        """
        Contruct a :class:`pyttb.sptensor` from `:class:pyttb.sptenmat`
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
    def shape(self) -> Tuple[int, ...]:
        """
        Return the shape of a sptenmat
        """
        if self.tshape == ():
            return ()
        else:
            m = np.prod(np.array(self.tshape)[self.rdims])
            n = np.prod(np.array(self.tshape)[self.cdims])
            return m, n

    def double(self) -> sparse.coo_matrix:
        """
        Convert a :class:`pyttb.sptenmat` to a COO :class:`scipy.sparse.coo_matrix`.
        """
        if self.subs.size == 0:
            return sparse.coo_matrix(self.shape)
        return sparse.coo_matrix(
            (self.vals.transpose()[0], self.subs.transpose()), self.shape
        )

    def full(self) -> ttb.tenmat:
        """
        Convert a :class:`pyttb.sptenmat` to a (dense) :class:`pyttb.tenmat`.
        """
        # Create empty dense tenmat
        result = ttb.tenmat(np.zeros(self.shape), self.rdims, self.cdims, self.tshape)
        # Assign nonzero values
        result[tuple(self.subs.transpose())] = np.squeeze(self.vals)
        return result

    @property
    def nnz(self) -> int:
        """
        Number of nonzero values in the :class:`pyttb.sptenmat`.
        """
        return len(self.vals)

    def norm(self) -> np.floating:
        """
        Compute the norm (i.e., Frobenius norm, or square root of the sum of
        squares of entries) of the :class:`pyttb.sptenmat`.
        """
        return np.linalg.norm(self.vals)

    def isequal(self, other: sptenmat) -> bool:
        """
        Exact equality for :class:`pyttb.sptenmat`
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
        """
        return self.copy()

    def __neg__(self):
        """
        Unary minus operator (-).
        """
        result = self.copy()
        result.vals *= -1
        return result

    def __setitem__(self, key, value):  # noqa: PLR0912
        """
        Subscripted assignment for the :class:`pyttb.sptenmat`.
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
        """
        String representation of a sptenmat.

        Returns
        -------
        str
            Contains the shape, row indices (rindices), column indices (cindices)
                and data as strings on different lines.
        """
        s = ""
        s += "sptenmat corresponding to a sptensor of shape "
        if self.vals.size == 0:
            s += str(self.shape)
        else:
            s += f"{self.tshape!r}"
        s += " with " + str(self.vals.size) + " nonzeros"
        s += "\n"

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
