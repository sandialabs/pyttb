# Copyright 2022 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

import numpy as np

import pyttb as ttb

from .pyttb_utils import *


class tenmat(object):
    """
    TENMAT Store tensor as a matrix.

    """

    def __init__(self):
        """
        TENSOR Create empty tensor.
        """

        # Case 0a: Empty Contructor
        self.tshape = ()
        self.rindices = np.array([])
        self.cindices = np.array([])
        self.data = np.array([])

    @classmethod
    def from_data(cls, data, rdims, cdims=None, tshape=None):
        # CONVERT A MULTIDIMENSIONAL ARRAY

        # Verify that data is a numeric numpy.ndarray
        if not isinstance(data, np.ndarray) or not issubclass(
            data.dtype.type, np.number
        ):
            assert False, "First argument must be a numeric numpy.ndarray."

        # data is empty, return empty tenmat unless rdims, cdims, or tshape are not empty
        if data.size == 0:
            if not rdims.size == 0 or not cdims.size == 0 or not tshape == ():
                assert (
                    False
                ), "When data is empty, rdims, cdims, and tshape must also be empty."
            else:
                return cls()

        # data is 1d array, must convert to 2d array for tenmat
        if len(data.shape) == 1:
            if tshape is None:
                assert False, "tshape must be specified when data is 1d array."
            else:
                # make data a 2d array with shape (1, data.shape[0]), i.e., a row vector
                data = np.reshape(data.copy(), (1, data.shape[0]), order="F")

        # data is ndarray and only rdims is specified
        if cdims is None:
            return ttb.tenmat.from_tensor_type(ttb.tensor.from_data(data), rdims)

        # use data.shape for tshape if not provided
        if tshape is None:
            tshape = data.shape
        elif not isinstance(tshape, tuple):
            assert False, "tshape must be a tuple."

        # check that data.shape and tshape agree
        if not np.prod(data.shape) == np.prod(tshape):
            assert (
                False
            ), "Incorrect dimensions specified: products of data.shape and tuple do not match"

        # check that data.shape and product of dimensions agree
        if not np.prod(np.array(tshape)[rdims]) * np.prod(
            np.array(tshape)[cdims]
        ) == np.prod(data.shape):
            assert (
                False
            ), "data.shape does not match shape specified by rdims, cdims, and tshape."

        return ttb.tenmat.from_tensor_type(
            ttb.tensor.from_data(data, tshape), rdims, cdims
        )

    @classmethod
    def from_tensor_type(cls, source, rdims=None, cdims=None, cdims_cyclic=None):
        # Case 0b: Copy Contructor
        if isinstance(source, tenmat):
            # Create tenmat
            tenmatInstance = cls()
            tenmatInstance.tshape = source.tshape
            tenmatInstance.rindices = source.rindices.copy()
            tenmatInstance.cindices = source.cindices.copy()
            tenmatInstance.data = source.data.copy()
            return tenmatInstance

        # Case III: Convert a tensor to a tenmat
        if isinstance(source, ttb.tensor):
            n = source.ndims
            alldims = np.array([range(n)])
            tshape = source.shape

            # Verify inputs
            if rdims is None and cdims is None:
                assert False, "Either rdims or cdims or both must be specified."
            if rdims is not None and not sum(np.in1d(rdims, alldims)) == len(rdims):
                assert False, "Values in rdims must be in [0, source.ndims]."
            if cdims is not None and not sum(np.in1d(cdims, alldims)) == len(cdims):
                assert False, "Values in cdims must be in [0, source.ndims]."

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
                        assert (
                            False
                        ), 'Unrecognized value for cdims_cyclic pattern, must be "fc" or "bc".'

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
                assert (
                    False
                ), "Incorrect specification of dimensions, the sorted concatenation of rdims and cdims must be range(source.ndims)."

            rprod = 1 if rdims.size == 0 else np.prod(np.array(tshape)[rdims])
            cprod = 1 if cdims.size == 0 else np.prod(np.array(tshape)[cdims])
            data = np.reshape(source.permute(dims).data, (rprod, cprod), order="F")

            # Create tenmat
            tenmatInstance = cls()
            tenmatInstance.tshape = tshape
            tenmatInstance.rindices = rdims.copy()
            tenmatInstance.cindices = cdims.copy()
            tenmatInstance.data = data.copy()
            return tenmatInstance

    def ctranspose(self):
        """
        Complex conjugate transpose for tenmat.

        Parameters
        ----------

        Returns
        -------
        :class:`pyttb.tenmat`
        """

        tenmatInstance = tenmat()
        tenmatInstance.rindices = self.cindices.copy()
        tenmatInstance.cindices = self.rindices.copy()
        tenmatInstance.tshape = self.tshape
        tenmatInstance.data = self.data.conj().T.copy()
        return tenmatInstance

    def double(self):
        """
        Convert tenmat to an array of doubles

        Returns
        -------
        :class:`numpy.ndarray`
            copy of tenmat data
        """
        return self.data.astype(np.float_).copy()

    def end(self, k):
        """
        Last index of indexing expression for tenmat

        Parameters
        ----------
        k: int
            dimension for subscripted indexing

        Returns
        -------
        int: index
        """

        return self.data.shape[k] - 1

    @property
    def ndims(self):
        """
        Return the number of dimensions of a tenmat

        Returns
        -------
        int
        """
        return len(self.shape)

    def norm(self):
        """
        Frobenius norm of a tenmat.

        Parameters
        ----------

        Returns
        -------
        Returns
        -------
        float
        """
        # default of np.linalg.norm is to vectorize the data and compute the vector norm, which is equivalent to
        # the Frobenius norm for multidimensional arrays. However, the argument 'fro' only workks for 1-D and 2-D
        # arrays currently.
        return np.linalg.norm(self.data)

    @property
    def shape(self):
        """
        Return the shape of a tenmat

        Returns
        -------
        tuple
        """
        if self.data.shape == (0,):
            return ()
        else:
            return self.data.shape

    def __setitem__(self, key, value):
        """
        SUBSASGN Subscripted assignment for a tensor.
        """
        self.data[key] = value
        return

    def __getitem__(self, item):
        """
        SUBSREF Subscripted reference for tenmat.

        Parameters
        ----------
        item:

        Returns
        -------
        :class:`numpy.ndarray`, float, int
        """
        return self.data[item]

    def __mul__(self, other):
        """
        Multiplies two tenmat objects.

        Parameters
        ----------
        other: :class:`pyttb.tenmat`

        Returns
        -------
        :class:`pyttb.tenmat`
        """
        # One argument is a scalar
        if np.isscalar(other):
            Z = ttb.tenmat.from_tensor_type(self)
            Z.data = Z.data * other
            return Z
        elif isinstance(other, tenmat):
            # Check that data shapes are compatible
            if not self.shape[1] == other.shape[0]:
                assert (
                    False
                ), "tenmat shape mismatch: number or columns of left operand must match number of rows of right operand."

            tshape = tuple(
                np.hstack(
                    (
                        np.array(self.tshape)[self.rindices],
                        np.array(other.tshape)[other.cindices],
                    )
                )
            )

            if tshape == ():
                return (self.data @ other.data)[0, 0]
            else:
                tenmatInstance = tenmat()
                tenmatInstance.tshape = tshape
                tenmatInstance.rindices = np.arange(len(self.rindices))
                tenmatInstance.cindices = np.arange(len(other.cindices)) + len(
                    self.rindices
                )
                tenmatInstance.data = self.data @ other.data
                return tenmatInstance
        else:
            assert (
                False
            ), "tenmat multiplication only valid with scalar or tenmat objects."

    def __rmul__(self, other):
        """
        Multiplies two tenmat objects.

        Parameters
        ----------
        other: :class:`pyttb.tenmat`

        Returns
        -------
        :class:`pyttb.tenmat`
        """
        return self.__mul__(other)

    def __add__(self, other):
        pass
        """
        Binary addition (+) for tenmats

        Parameters
        ----------
        other: :class:`pyttb.tenmat`, float, int

        Returns
        -------
        :class:`pyttb.tenmat`
        """

        # One argument is a scalar
        if np.isscalar(other):
            Z = ttb.tenmat.from_tensor_type(self)
            Z.data = Z.data + other
            return Z
        elif isinstance(other, tenmat):
            # Check that data shapes agree
            if not self.shape == other.shape:
                assert False, "tenmat shape mismatch."

            Z = ttb.tenmat.from_tensor_type(self)
            Z.data = Z.data + other.data
            return Z
        else:
            assert False, "tenmat addition only valid with scalar or tenmat objects."

    def __radd__(self, other):
        """
        Reverse binary addition (+) for tenmats

        Parameters
        ----------
        other: :class:`pyttb.tenmat`, float, int

        Returns
        -------
        :class:`pyttb.tenmat`
        """
        return self.__add__(other)

    def __sub__(self, other):
        pass
        """
        Binary subtraction (-) for tenmats

        Parameters
        ----------
        other: :class:`pyttb.tenmat`, float, int

        Returns
        -------
        :class:`pyttb.tenmat`
        """

        # One argument is a scalar
        if np.isscalar(other):
            Z = ttb.tenmat.from_tensor_type(self)
            Z.data = Z.data - other
            return Z
        elif isinstance(other, tenmat):
            # Check that data shapes agree
            if not self.shape == other.shape:
                assert False, "tenmat shape mismatch."

            Z = ttb.tenmat.from_tensor_type(self)
            Z.data = Z.data - other.data
            return Z
        else:
            assert False, "tenmat subtraction only valid with scalar or tenmat objects."

    def __rsub__(self, other):
        """
        Reverse binary subtraction (-) for tenmats

        Parameters
        ----------
        other: :class:`pyttb.tenmat`, float, int

        Returns
        -------
        :class:`pyttb.tenmat`
        """

        # One argument is a scalar
        if np.isscalar(other):
            Z = ttb.tenmat.from_tensor_type(self)
            Z.data = other - Z.data
            return Z
        elif isinstance(other, tenmat):
            # Check that data shapes agree
            if not self.shape == other.shape:
                assert False, "tenmat shape mismatch."

            Z = ttb.tenmat.from_tensor_type(self)
            Z.data = other.data - Z.data
            return Z
        else:
            assert False, "tenmat subtraction only valid with scalar or tenmat objects."

    def __pos__(self):
        """
        Unary plus (+) for tenmats

        Returns
        -------
        :class:`pyttb.tenmat`
            copy of tenmat
        """

        T = ttb.tenmat.from_tensor_type(self)

        return T

    def __neg__(self):
        """
        Unary minus (-) for tenmats

        Returns
        -------
        :class:`pyttb.tenmat`
            copy of tenmat
        """

        T = ttb.tenmat.from_tensor_type(self)
        T.data = -1 * T.data

        return T

    def __repr__(self):
        """
        String representation of a tenmat.

        Returns
        -------
        str
            Contains the shape, row indices (rindices), column indices (cindices) and data as strings on different lines.
        """
        s = ""
        s += "matrix corresponding to a tensor of shape "
        if self.data.size == 0:
            s += str(self.shape)
        else:
            s += (" x ").join([str(int(d)) for d in self.tshape])
        s += "\n"

        s += "rindices = "
        s += "[ " + (", ").join([str(int(d)) for d in self.rindices]) + " ] "
        s += "(modes of tensor corresponding to rows)\n"

        s += "cindices = "
        s += "[ " + (", ").join([str(int(d)) for d in self.cindices]) + " ] "
        s += "(modes of tensor corresponding to columns)\n"

        if self.data.size == 0:
            s += "data = []\n"
        else:
            s += "data[:, :] = \n"
            s += str(self.data)
            s += "\n"

        return s

    __str__ = __repr__
