# Copyright 2022 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

"""Classes and functions for working with Kruskal tensors."""
from __future__ import annotations

from typing import Literal, Optional, Tuple, Union

import numpy as np
from numpy_groupies import aggregate as accumarray
from scipy import sparse

import pyttb as ttb
from pyttb.pyttb_utils import tt_sub2ind


class sptenmat(object):
    """
    SPTENMAT Store sparse tensor as a sparse matrix.

    """

    __slots__ = ("tshape", "rdims", "cdims", "subs", "vals")

    def __init__(self):
        """
        Construct an empty :class:`pyttb.sptenmat`
        """
        self.tshape = ()
        self.rdims = np.array([])
        self.cdims = np.array([])
        self.subs = np.array([], ndmin=2, dtype=int)
        self.vals = np.array([], ndmin=2)

    @classmethod
    def from_data(  # noqa: PLR0913
        cls,
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
            Location of non-zero entries
        vals:
            Values for non-zero entries
        rdims:
            Mapping of row indices
        cdims:
            Mapping of column indices
        tshape:
            Shape of the original tensor
        """
        if subs is None:
            subs = np.array([], ndmin=2, dtype=int)
        if vals is None:
            vals = np.array([], ndmin=2)
        if rdims is None:
            rdims = np.array([], dtype=int)
        if cdims is None:
            cdims = np.array([], dtype=int)

        n = len(tshape)
        alldims = np.array([range(n)])
        # Error check
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
        if subs.size == 0:
            assert vals.size == 0, "Empty subs requires empty vals"
            newsubs = np.array([])
            newvals = np.array([])
        else:
            # Identify only the unique indices
            newsubs, loc = np.unique(subs, axis=0, return_inverse=True)
            # Sum the corresponding values
            # Squeeze to convert from column vector to row vector
            newvals = accumarray(loc, np.squeeze(vals), size=newsubs.shape[0], func=sum)

        # Find the nonzero indices of the new values
        nzidx = np.nonzero(newvals)
        newsubs = newsubs[nzidx]
        # None index to convert from row back to column vector
        newvals = newvals[nzidx]
        if newvals.size > 0:
            newvals = newvals[:, None]

        sptenmatInstance = cls()
        sptenmatInstance.tshape = tshape
        sptenmatInstance.rdims = rdims.copy().astype(int)
        sptenmatInstance.cdims = cdims.copy().astype(int)
        sptenmatInstance.subs = newsubs
        sptenmatInstance.vals = newvals
        return sptenmatInstance

    @classmethod
    def from_tensor_type(  # noqa: PLR0912
        cls,
        source: Union[ttb.sptensor, ttb.sptenmat],
        rdims: Optional[np.ndarray] = None,
        cdims: Optional[np.ndarray] = None,
        cdims_cyclic: Optional[Union[Literal["fc"], Literal["bc"]]] = None,
    ):
        valid_sources = (sptenmat, ttb.sptensor)
        assert isinstance(source, valid_sources), (
            "Can only generate sptenmat from "
            f"{[src.__name__ for src in valid_sources]} but received {type(source)}."
        )
        # Copy Constructor
        if isinstance(source, sptenmat):
            return cls().from_data(
                source.subs.copy(),
                source.vals.copy(),
                source.rdims.copy(),
                source.cdims.copy(),
                source.tshape,
            )

        if isinstance(source, ttb.sptensor):
            n = source.ndims
            alldims = np.array([range(n)])

            if rdims is not None and cdims is None:
                # Single row mapping
                if len(rdims) == 1 and cdims_cyclic is not None:
                    if cdims_cyclic == "fc":
                        # cdims = [rdims+1:n, 1:rdims-1];
                        cdims = np.array(
                            [i for i in range(rdims[0] + 1, n)]
                            + [i for i in range(rdims[0])]
                        )
                    elif cdims_cyclic == "bc":
                        # cdims = [rdims-1:-1:1, n:-1:rdims+1];
                        cdims = np.array(
                            [i for i in range(rdims[0] - 1, -1, -1)]
                            + [i for i in range(n - 1, rdims[0], -1)]
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
            dims = np.hstack([rdims, cdims], dtype=int)
            if not len(dims) == n or not (alldims == np.sort(dims)).all():
                assert False, (
                    "Incorrect specification of dimensions, the sorted "
                    "concatenation of rdims and cdims must be range(source.ndims)."
                )

            rsize = np.array(source.shape)[rdims]
            csize = np.array(source.shape)[cdims]

            if rsize.size == 0:
                ridx = np.zeros((source.nnz, 1))
            elif source.subs.size == 0:
                ridx = np.array([], dtype=int)
            else:
                ridx = tt_sub2ind(rsize, source.subs[:, rdims])
            ridx = ridx.reshape((ridx.size, 1)).astype(int)

            if csize.size == 0:
                cidx = np.zeros((source.nnz, 1))
            elif source.subs.size == 0:
                cidx = np.array([], dtype=int)
            else:
                cidx = tt_sub2ind(csize, source.subs[:, cdims])
            cidx = cidx.reshape((cidx.size, 1)).astype(int)

            return cls().from_data(
                np.hstack([ridx, cidx], dtype=int),
                source.vals.copy(),
                rdims,
                cdims,
                source.shape,
            )

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Return the shape of a sptenmat

        Returns
        -------
        tuple
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
        result = ttb.tenmat.from_data(
            np.zeros(self.shape), self.rdims, self.cdims, self.tshape
        )
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

    def __pos__(self):
        """
        Unary plus operator (+).
        """
        return self.from_tensor_type(self)

    def __neg__(self):
        """
        Unary minus operator (-).
        """
        result = self.from_tensor_type(self)
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
