# Copyright 2022 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

"""Classes and functions for working with Kruskal tensors."""


import numpy as np
from numpy_groupies import aggregate as accumarray

import pyttb as ttb
from pyttb.pyttb_utils import tt_sub2ind


class sptenmat(object):
    """
    SPTENMAT Store sparse tensor as a sparse matrix.

    """

    def __init__(self, *args):  # pragma:no cover
        """
        Construct an empty :class:`pyttb.sptenmat`

        The constructor takes no arguments and returns an empty
        :class:`pyttb.sptenmat`.
        """
        self.tshape = ()
        self.rdims = np.array([])
        self.cdims = np.array([])
        self.subs = np.array([])
        self.vals = np.array([])

    @classmethod
    def from_data(cls, subs, vals, rdims, cdims, tshape):  # noqa: PLR0913
        """
        Construct a :class:`pyttb.sptenmat` from a set of 2D subscripts (subs)
        and values (vals) along with the mappings of the row (rdims) and column
        indices (cdims) and the shape of the original tensor (tshape).

        Parameters
        ----------
        subs: :class:`numpy.ndarray`, required
            Location of non-zero entries
        vals: :class:`numpy.ndarray`, required
            Values for non-zero entries
        rdims: :class:`numpy.ndarray`, required
            Mapping of row indices
        cdims: :class:`numpy.ndarray`, required
            Mapping of column indices
        tshape: :class:`tuple`, required
            Shape of the original tensor

        """
        n = len(tshape)
        alldims = np.array([range(n)])
        # Error check
        if rdims.size == 0:
            dims = cdims.copy()
        elif cdims.size == 0:
            dims = rdims.copy()
        else:
            dims = np.hstack([rdims, cdims])
        if not len(dims) == n or not (alldims == np.sort(dims)).all():
            assert False, (
                "Incorrect specification of dimensions, the sorted concatenation of "
                "rdims and cdims must be range(len(tshape))."
            )
        elif subs.size > 1 and np.prod(np.array(tshape)[rdims]) < np.max(subs[:, 0]):
            assert False, "Invalid row index."
        elif subs.size > 1 and np.prod(np.array(tshape)[cdims]) < np.max(subs[:, 1]):
            assert False, "Invalid column index."

        # Sum any duplicates
        if subs.size == 0:
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
        sptenmatInstance.rdims = rdims.copy()
        sptenmatInstance.cdims = cdims.copy()
        sptenmatInstance.subs = newsubs
        sptenmatInstance.vals = newvals
        return sptenmatInstance

    @classmethod
    def from_tensor_type(  # noqa: PLR0912
        cls, source, rdims=None, cdims=None, cdims_cyclic=None
    ):
        # Copy Contructor
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
            tshape = source.shape

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
                            'must be "t", "fc" or "bc".'
                        )
                else:
                    # Multiple row mapping
                    cdims = np.setdiff1d(alldims, rdims)

            elif rdims is None and cdims is not None:
                rdims = np.setdiff1d(alldims, cdims)

            # if rdims or cdims is empty, hstack will output an array of float not int
            if rdims.size == 0:
                dims = cdims.copy()
            elif cdims.size == 0:
                dims = rdims.copy()
            else:
                dims = np.hstack([rdims, cdims])
            if not len(dims) == n or not (alldims == np.sort(dims)).all():
                assert False, (
                    "Incorrect specification of dimensions, the sorted "
                    "concatenation of rdims and cdims must be range(source.ndims)."
                )

            rsize = np.array(source.shape)[rdims]
            csize = np.array(source.shape)[cdims]

            if rdims.size == 0:
                ridx = np.ones((source.nnz, 1))
            elif source.subs.size == 0:
                ridx = np.array([])
            else:
                ridx = tt_sub2ind(rsize, source.subs[:, rdims])
            ridx = ridx.reshape((ridx.size, 1))

            if cdims.size == 0:
                cidx = np.ones((source.nnz, 1))
            elif source.subs.size == 0:
                cidx = np.array([])
            else:
                cidx = tt_sub2ind(csize, source.subs[:, cdims])
            cidx = cidx.reshape((cidx.size, 1))

            return cls().from_data(
                np.hstack([ridx, cidx]), source.vals.copy(), rdims, cdims, source.shape
            )

    @property
    def shape(self):
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
            return (m, n)

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
            s += (" x ").join([str(int(d)) for d in self.tshape])
        s += " with " + str(self.vals.size) + " nonzeros"
        s += "\n"

        s += "rdims = "
        s += "[ " + (", ").join([str(int(d)) for d in self.rdims]) + " ] "
        s += "(modes of sptensor corresponding to rows)\n"

        s += "cdims = "
        s += "[ " + (", ").join([str(int(d)) for d in self.cdims]) + " ] "
        s += "(modes of sptensor corresponding to columns)\n"

        # Stop insane printouts
        if self.vals.size > 10000:
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
