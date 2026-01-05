"""Classes and functions for working with matricized dense tensors."""

# Copyright 2025 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from __future__ import annotations

import logging
from math import prod
from typing import Literal

import numpy as np

import pyttb as ttb
from pyttb.pyttb_utils import (
    Shape,
    gather_wrap_dims,
    np_to_python,
    parse_shape,
    to_memory_order,
)


class tenmat:
    """Store tensor as a matrix."""

    __slots__ = ("cindices", "data", "rindices", "tshape")

    def __init__(  # noqa: PLR0912
        self,
        data: np.ndarray | None = None,
        rdims: np.ndarray | None = None,
        cdims: np.ndarray | None = None,
        tshape: Shape | None = None,
        copy: bool = True,
    ):
        """Construct a :class:`pyttb.tenmat` from explicit components.

        If you already have a tensor see :meth:`pyttb.tensor.to_tenmat`.

        Parameters
        ----------
        data:
            Flattened tensor data.
        rdims:
            Which dimensions of original tensor map to rows.
        cdims:
            Which dimensions of original tensor map to columns.
        tshape:
            Original tensor shape.
        copy:
            Whether to make a copy of provided data or just reference it.

        Examples
        --------
        Create an empty :class:`pyttb.tenmat`.

        >>> ttb.tenmat()  # doctest: +NORMALIZE_WHITESPACE
        matrix corresponding to a tensor of shape () with order F
        rindices = [  ] (modes of tensor corresponding to rows)
        cindices = [  ] (modes of tensor corresponding to columns)
        data = []

        Create tensor shaped data.

        >>> tshape = (2, 2, 2)
        >>> data = np.reshape(np.arange(prod(tshape), dtype=np.double), tshape)
        >>> data  # doctest: +NORMALIZE_WHITESPACE
        array([[[0., 1.],
                [2., 3.]],
               [[4., 5.],
                [6., 7.]]])

        Manually matrize the tensor.

        >>> flat_data = np.reshape(data, (2, 4), order="F")
        >>> flat_data  # doctest: +NORMALIZE_WHITESPACE
        array([[0., 2., 1., 3.],
               [4., 6., 5., 7.]])

        Encode matrication into :class:`pyttb.tenmat`.

        >>> tm = ttb.tenmat(flat_data, rdims=np.array([0]), tshape=tshape)

        Extract original tensor shaped data.

        >>> tm.to_tensor().double()  # doctest: +NORMALIZE_WHITESPACE
        array([[[0., 1.],
                [2., 3.]],
               [[4., 5.],
                [6., 7.]]])
        """
        # Case 0a: Empty Constructor
        # data is empty, return empty tenmat unless rdims, cdims, or tshape are
        # not empty
        if data is None or (isinstance(data, np.ndarray) and data.size == 0):
            cdims_empty = cdims is None or cdims.size == 0
            rdims_empty = rdims is None or rdims.size == 0
            tshape_empty = tshape is None or tshape == ()
            assert rdims_empty and cdims_empty and tshape_empty, (
                "When data is empty, rdims, cdims, and tshape must also be empty."
            )

            self.tshape: tuple[()] | tuple[int, ...] = ()
            self.rindices = np.array([])
            self.cindices = np.array([])
            self.data = np.array([], ndmin=2, order=self.order)
            return

        # Verify that data is a numeric numpy.ndarray
        assert isinstance(data, np.ndarray) and issubclass(
            data.dtype.type, np.number
        ), "First argument must be a numeric numpy.ndarray."

        # data is 1d array, must convert to 2d array for tenmat
        if len(data.shape) == 1:
            if tshape is None:
                assert False, "tshape must be specified when data is 1d array."
            else:
                # make data a 2d array with shape (1, data.shape[0]), i.e., a row vector
                data = np.reshape(data.copy("K"), (1, data.shape[0]), order=self.order)

        if len(data.shape) != 2:
            raise ValueError(
                f"Data must be a matrix or vector but had {len(data.shape)} dimensions"
            )

        # use data.shape for tshape if not provided
        if tshape is None:
            tshape = data.shape
        tshape = parse_shape(tshape)

        # check that data.shape and tshape agree
        if prod(data.shape) != prod(tshape):
            assert False, (
                "Incorrect dimensions specified: products of data.shape and tuple do "
                "not match"
            )

        n = len(tshape)
        alldims = np.array([range(n)])
        rdims, cdims = gather_wrap_dims(n, rdims, cdims)

        # check that data.shape and product of dimensions agree
        if not np.prod(np.array(tshape)[rdims]) * np.prod(
            np.array(tshape)[cdims]
        ) == prod(data.shape):
            assert False, (
                "data.shape does not match shape specified by rdims, cdims, and tshape."
            )

        # if rdims or cdims is empty, hstack will output an array of float not int
        if rdims.size == 0:
            dims = cdims.copy("K")
        elif cdims.size == 0:
            dims = rdims.copy("K")
        else:
            dims = np.hstack([rdims, cdims])
        if not len(dims) == n or not (alldims == np.sort(dims)).all():
            assert False, (
                "Incorrect specification of dimensions, the sorted concatenation "
                "of rdims and cdims must be range(source.ndims)."
            )

        self.tshape = tshape
        self.rindices = rdims.copy("K")
        self.cindices = cdims.copy("K")

        if not copy and not self._matches_order(data):
            logging.warning(
                f"Selected no copy, but input data isn't {self.order} ordered "
                "so must copy."
            )
            copy = True
        self.data = to_memory_order(data, self.order, copy=copy)
        return

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

    def copy(self) -> tenmat:
        """
        Return a deep copy of the :class:`pyttb.tenmat`.

        Examples
        --------
        Create a :class:`pyttb.tenmat` (TM1) and make a deep copy. Verify
        the deep copy (TM3) is not just a reference (like TM2) to the original.

        >>> T1 = ttb.tensor(np.ones((3, 2)))
        >>> TM1 = T1.to_tenmat(np.array([0]))
        >>> TM2 = TM1
        >>> TM3 = TM1.copy()
        >>> TM1[0, 0] = 3

        # Item to convert numpy boolean to python boolena for nicer printing

        >>> (TM1[0, 0] == TM2[0, 0]).item()
        True
        >>> (TM1[0, 0] == TM3[0, 0]).item()
        False
        """
        # Create tenmat
        return ttb.tenmat(
            self.data, self.rindices, self.cindices, self.tshape, copy=True
        )

    def __deepcopy__(self, memo):
        """Return deep copy of this tenmat."""
        return self.copy()

    def to_tensor(self, copy: bool = True) -> ttb.tensor:
        """
        Return :class:`pyttb.tenmat` data as a :class:`pyttb.tensor`.

        Parameters
        ----------
        copy:
            Whether to make a copy of provided data or just reference it.

        Examples
        --------
        Create tensor shaped data.

        >>> tshape = (2, 2, 2)
        >>> data = np.reshape(np.arange(np.prod(tshape), dtype=np.double), tshape)
        >>> data  # doctest: +NORMALIZE_WHITESPACE
        array([[[0., 1.],
                [2., 3.]],
               [[4., 5.],
                [6., 7.]]])

        Manually matrize the tensor.

        >>> flat_data = np.reshape(data, (2, 4), order="F")
        >>> flat_data  # doctest: +NORMALIZE_WHITESPACE
        array([[0., 2., 1., 3.],
               [4., 6., 5., 7.]])

        Encode matrication into :class:`pyttb.tenmat`.

        >>> tm = ttb.tenmat(flat_data, rdims=np.array([0]), tshape=tshape)

        Extract original tensor shaped data.

        >>> tm.to_tensor()  # doctest: +NORMALIZE_WHITESPACE
        tensor of shape (2, 2, 2) with order F
        data[:, :, 0] =
        [[0. 2.]
         [4. 6.]]
        data[:, :, 1] =
        [[1. 3.]
         [5. 7.]]
        """
        # RESHAPE TENSOR-AS-MATRIX
        # Here we just reverse what was done in the tenmat constructor.
        # First we reshape the data to be an MDA, then we un-permute
        # it using ipermute.
        shape = self.tshape
        order = np.hstack([self.rindices, self.cindices])
        data = self.data
        if copy:
            data = self.data.copy("K")
        data = np.reshape(data, np.array(shape)[order], order=self.order)
        if order.size > 1:
            if not copy:
                logging.warning(
                    "This tenmat cannot be trivially unwrapped into tensor "
                    "so must copy."
                )
            data = to_memory_order(np.transpose(data, np.argsort(order)), self.order)
        return ttb.tensor(data, shape, copy=False)

    def ctranspose(self) -> tenmat:
        """
        Complex conjugate transpose for :class:`pyttb.tenmat`.

        Examples
        --------
        Create :class:`pyttb.tensor` then convert to :class:`pyttb.tenmat`.

        >>> T = ttb.tenones((2, 2, 2))
        >>> TM = T.to_tenmat(rdims=np.array([0]))
        >>> TM  # doctest: +NORMALIZE_WHITESPACE
        matrix corresponding to a tensor of shape (2, 2, 2) with order F
        rindices = [ 0 ] (modes of tensor corresponding to rows)
        cindices = [ 1, 2 ] (modes of tensor corresponding to columns)
        data[:, :] =
        [[1. 1. 1. 1.]
         [1. 1. 1. 1.]]
        >>> TM.ctranspose()  # doctest: +NORMALIZE_WHITESPACE
        matrix corresponding to a tensor of shape (2, 2, 2) with order F
        rindices = [ 1, 2 ] (modes of tensor corresponding to rows)
        cindices = [ 0 ] (modes of tensor corresponding to columns)
        data[:, :] =
        [[1. 1.]
         [1. 1.]
         [1. 1.]
         [1. 1.]]
        """
        return tenmat(
            self.data.conj().T,
            self.cindices,
            self.rindices,
            self.tshape,
            copy=True,
        )

    def double(self, immutable: bool = False) -> np.ndarray:
        """
        Convert a :class:`pyttb.tenmat` to an array of doubles.

        Parameters
        ----------
        immutable: Whether or not the returned data cam be mutated. May enable
            additional optimizations.

        Examples
        --------
        >>> T = ttb.tenones((2, 2, 2))
        >>> TM = T.to_tenmat(rdims=np.array([0]))
        >>> TM  # doctest: +NORMALIZE_WHITESPACE
        matrix corresponding to a tensor of shape (2, 2, 2) with order F
        rindices = [ 0 ] (modes of tensor corresponding to rows)
        cindices = [ 1, 2 ] (modes of tensor corresponding to columns)
        data[:, :] =
        [[1. 1. 1. 1.]
         [1. 1. 1. 1.]]
        >>> TM.double()  # doctest: +NORMALIZE_WHITESPACE
        array([[1., 1., 1., 1.],
               [1., 1., 1., 1.]])
        """
        double = to_memory_order(self.data, self.order, copy=not immutable).astype(
            np.float64
        )
        if immutable:
            double.flags.writeable = False
        elif np.shares_memory(double, self.data):
            double = double.copy("K")
        return double

    @property
    def ndims(self) -> int:
        """Return the number of dimensions of a :class:`pyttb.tenmat`.

        Examples
        --------
        >>> TM = ttb.tenmat()  # empty tenmat
        >>> TM.ndims
        0

        >>> TM = ttb.tenones((2, 2, 2)).to_tenmat(np.array([0]))
        >>> TM.ndims
        2
        """
        return len(self.shape)

    def norm(self) -> float:
        """Frobenius norm of a :class:`pyttb.tenmat`.

        Examples
        --------
        >>> T = ttb.tenones((2, 2, 2))
        >>> TM = T.to_tenmat(rdims=np.array([0]))
        >>> TM  # doctest: +NORMALIZE_WHITESPACE
        matrix corresponding to a tensor of shape (2, 2, 2) with order F
        rindices = [ 0 ] (modes of tensor corresponding to rows)
        cindices = [ 1, 2 ] (modes of tensor corresponding to columns)
        data[:, :] =
        [[1. 1. 1. 1.]
         [1. 1. 1. 1.]]
        >>> TM.norm()  # doctest: +ELLIPSIS
        2.82...
        """
        # default of np.linalg.norm is to vectorize the data and compute the vector
        # norm, which is equivalent to the Frobenius norm for multidimensional arrays.
        # However, the argument 'fro' only works for 1-D and 2-D
        # arrays currently.
        return float(np.linalg.norm(self.data))

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of a :class:`pyttb.tenmat`.

        Examples
        --------
        >>> TM = ttb.tenmat()  # empty tenmat
        >>> TM.shape
        ()

        >>> TM = ttb.tenones((2, 2, 2)).to_tenmat(np.array([0]))
        >>> TM.shape
        (2, 4)
        """
        if self.data.size == 0:
            return ()
        return self.data.shape

    def isequal(self, other: tenmat) -> bool:
        """
        Exact equality for :class:`pyttb.tenmat`.

        Examples
        --------
        >>> TM1 = ttb.tenmat()  # empty tenmat
        >>> TM2 = ttb.tenones((2, 2, 2)).to_tenmat(np.array([0]))
        >>> TM1.isequal(TM2)
        False
        >>> TM1.isequal(TM1)
        True
        """
        if not isinstance(other, ttb.tenmat):
            raise ValueError(
                f"Can only compares against other tenmat but received: {type(other)}"
            )
        return (
            np.array_equal(self.data, other.data)
            and self.tshape == other.tshape
            and np.array_equal(self.rindices, other.rindices)
            and np.array_equal(self.cindices, other.cindices)
        )

    def __setitem__(self, key, value):
        """
        Subscripted assignment for a :class:`pyttb.tenmat`.

        Examples
        --------
        >>> TM = ttb.tenones((2, 2, 2)).to_tenmat(np.array([0]))
        >>> TM  # doctest: +NORMALIZE_WHITESPACE
        matrix corresponding to a tensor of shape (2, 2, 2) with order F
        rindices = [ 0 ] (modes of tensor corresponding to rows)
        cindices = [ 1, 2 ] (modes of tensor corresponding to columns)
        data[:, :] =
        [[1. 1. 1. 1.]
         [1. 1. 1. 1.]]
        >>> TM[0, 0] = 2.0
        >>> TM  # doctest: +NORMALIZE_WHITESPACE
        matrix corresponding to a tensor of shape (2, 2, 2) with order F
        rindices = [ 0 ] (modes of tensor corresponding to rows)
        cindices = [ 1, 2 ] (modes of tensor corresponding to columns)
        data[:, :] =
        [[2. 1. 1. 1.]
         [1. 1. 1. 1.]]
        """
        self.data[key] = value

    def __getitem__(self, item):
        """
        Subscripted reference for :class:`pyttb.tenmat`.

        Examples
        --------
        >>> TM = ttb.tenones((2, 2, 2)).to_tenmat(np.array([0]))
        >>> print(TM[0, 0])
        1.0

        Returns
        -------
        :class:`numpy.ndarray`, float, int
        """
        return self.data[item]

    def __mul__(self, other):
        """
        Multiplies two :class:`pyttb.tenmat` objects.

        Parameters
        ----------
        other: :class:`pyttb.tenmat`

        Examples
        --------
        >>> TM = ttb.tenones((2, 2)).to_tenmat(np.array([0]))
        >>> TM * TM  # doctest: +NORMALIZE_WHITESPACE
        matrix corresponding to a tensor of shape (2, 2) with order F
        rindices = [ 0 ] (modes of tensor corresponding to rows)
        cindices = [ 1 ] (modes of tensor corresponding to columns)
        data[:, :] =
        [[2. 2.]
         [2. 2.]]

        Returns
        -------
        :class:`pyttb.tenmat`
        """
        # One argument is a scalar
        if np.isscalar(other):
            Z = self.copy()
            Z.data = Z.data * other
            return Z
        if isinstance(other, tenmat):
            # Check that data shapes are compatible
            if not self.shape[1] == other.shape[0]:
                assert False, (
                    "tenmat shape mismatch: number or columns of left operand must "
                    "match number of rows of right operand."
                )

            tshape = tuple(
                np.hstack(
                    (
                        np.array(self.tshape)[self.rindices],
                        np.array(other.tshape)[other.cindices],
                    )
                )
            )

            if not tshape:
                return (self.data @ other.data)[0, 0]
            tenmatInstance = tenmat(
                np.matmul(self.data, other.data, order=self.order),
                np.arange(len(self.rindices)),
                np.arange(len(other.cindices)) + len(self.rindices),
                tshape,
                copy=False,
            )
            return tenmatInstance
        assert False, "tenmat multiplication only valid with scalar or tenmat objects."

    def __rmul__(self, other):
        """
        Multiplies two :class:`pyttb.tenmat` objects.

        Parameters
        ----------
        other: :class:`pyttb.tenmat`

        Examples
        --------
        >>> TM = ttb.tenones((2, 2)).to_tenmat(np.array([0]))
        >>> TM * TM  # doctest: +NORMALIZE_WHITESPACE
        matrix corresponding to a tensor of shape (2, 2) with order F
        rindices = [ 0 ] (modes of tensor corresponding to rows)
        cindices = [ 1 ] (modes of tensor corresponding to columns)
        data[:, :] =
        [[2. 2.]
         [2. 2.]]

        Returns
        -------
        :class:`pyttb.tenmat`
        """
        return self.__mul__(other)

    def __add__(self, other):
        """
        Binary addition (+) for :class:`pyttb.tenmat`.

        Parameters
        ----------
        other: :class:`pyttb.tenmat`, float, int

        Examples
        --------
        >>> TM = ttb.tenones((2, 2)).to_tenmat(np.array([0]))
        >>> TM + TM  # doctest: +NORMALIZE_WHITESPACE
        matrix corresponding to a tensor of shape (2, 2) with order F
        rindices = [ 0 ] (modes of tensor corresponding to rows)
        cindices = [ 1 ] (modes of tensor corresponding to columns)
        data[:, :] =
        [[2. 2.]
         [2. 2.]]
        >>> TM + 1.0  # doctest: +NORMALIZE_WHITESPACE
        matrix corresponding to a tensor of shape (2, 2)  with order F
        rindices = [ 0 ] (modes of tensor corresponding to rows)
        cindices = [ 1 ] (modes of tensor corresponding to columns)
        data[:, :] =
        [[2. 2.]
         [2. 2.]]

        Returns
        -------
        :class:`pyttb.tenmat`
        """
        # One argument is a scalar
        if np.isscalar(other):
            Z = self.copy()
            Z.data = Z.data + other
            return Z
        if isinstance(other, tenmat):
            # Check that data shapes agree
            if not self.shape == other.shape:
                assert False, "tenmat shape mismatch."

            Z = self.copy()
            Z.data = Z.data + other.data
            return Z
        assert False, "tenmat addition only valid with scalar or tenmat objects."

    def __radd__(self, other):
        """
        Right binary addition (+) for :class:`pyttb.tenmat`.

        Parameters
        ----------
        other: :class:`pyttb.tenmat`, float, int

        Examples
        --------
        >>> TM = ttb.tenones((2, 2)).to_tenmat(np.array([0]))
        >>> 1.0 + TM  # doctest: +NORMALIZE_WHITESPACE
        matrix corresponding to a tensor of shape (2, 2) with order F
        rindices = [ 0 ] (modes of tensor corresponding to rows)
        cindices = [ 1 ] (modes of tensor corresponding to columns)
        data[:, :] =
        [[2. 2.]
         [2. 2.]]

        Returns
        -------
        :class:`pyttb.tenmat`
        """
        return self.__add__(other)

    def __sub__(self, other):
        """
        Binary subtraction (-) for :class:`pyttb.tenmat`.

        Parameters
        ----------
        other: :class:`pyttb.tenmat`, float, int

        Examples
        --------
        >>> TM = ttb.tenones((2, 2)).to_tenmat(np.array([0]))
        >>> TM - TM  # doctest: +NORMALIZE_WHITESPACE
        matrix corresponding to a tensor of shape (2, 2) with order F
        rindices = [ 0 ] (modes of tensor corresponding to rows)
        cindices = [ 1 ] (modes of tensor corresponding to columns)
        data[:, :] =
        [[0. 0.]
         [0. 0.]]
        >>> TM - 1.0  # doctest: +NORMALIZE_WHITESPACE
        matrix corresponding to a tensor of shape (2, 2) with order F
        rindices = [ 0 ] (modes of tensor corresponding to rows)
        cindices = [ 1 ] (modes of tensor corresponding to columns)
        data[:, :] =
        [[0. 0.]
         [0. 0.]]

        Returns
        -------
        :class:`pyttb.tenmat`
        """
        # One argument is a scalar
        if np.isscalar(other):
            Z = self.copy()
            Z.data = Z.data - other
            return Z
        if isinstance(other, tenmat):
            # Check that data shapes agree
            if not self.shape == other.shape:
                assert False, "tenmat shape mismatch."

            Z = self.copy()
            Z.data = Z.data - other.data
            return Z
        assert False, "tenmat subtraction only valid with scalar or tenmat objects."

    def __rsub__(self, other):
        """
        Right binary subtraction (-) for :class:`pyttb.tenmat`.

        Parameters
        ----------
        other: :class:`pyttb.tenmat`, float, int

        Examples
        --------
        >>> TM = ttb.tenones((2, 2)).to_tenmat(np.array([0]))
        >>> 1.0 - TM  # doctest: +NORMALIZE_WHITESPACE
        matrix corresponding to a tensor of shape (2, 2) with order F
        rindices = [ 0 ] (modes of tensor corresponding to rows)
        cindices = [ 1 ] (modes of tensor corresponding to columns)
        data[:, :] =
        [[0. 0.]
         [0. 0.]]

        Returns
        -------
        :class:`pyttb.tenmat`
        """
        # One argument is a scalar
        if np.isscalar(other):
            Z = self.copy()
            Z.data = other - Z.data
            return Z
        if isinstance(other, tenmat):
            # Check that data shapes agree
            if not self.shape == other.shape:
                assert False, "tenmat shape mismatch."

            Z = self.copy()
            Z.data = other.data - Z.data
            return Z
        assert False, "tenmat subtraction only valid with scalar or tenmat objects."

    def __pos__(self):
        """
        Unary plus (+) for :class:`pyttb.tenmat`.

        Examples
        --------
        >>> TM = ttb.tenones((2, 2)).to_tenmat(np.array([0]))
        >>> +TM  # doctest: +NORMALIZE_WHITESPACE
        matrix corresponding to a tensor of shape (2, 2) with order F
        rindices = [ 0 ] (modes of tensor corresponding to rows)
        cindices = [ 1 ] (modes of tensor corresponding to columns)
        data[:, :] =
        [[1. 1.]
         [1. 1.]]

        Returns
        -------
        :class:`pyttb.tenmat`
            copy of tenmat
        """
        T = self.copy()

        return T

    def __neg__(self):
        """
        Unary minus (-) for :class:`pyttb.tenmat`.

        Examples
        --------
        >>> TM = ttb.tenones((2, 2)).to_tenmat(np.array([0]))
        >>> -TM  # doctest: +NORMALIZE_WHITESPACE
        matrix corresponding to a tensor of shape (2, 2) with order F
        rindices = [ 0 ] (modes of tensor corresponding to rows)
        cindices = [ 1 ] (modes of tensor corresponding to columns)
        data[:, :] =
        [[-1. -1.]
         [-1. -1.]]

        Returns
        -------
        :class:`pyttb.tenmat`
            Copy of original tenmat with negated data.
        """
        T = self.copy()
        T.data = -1 * T.data

        return T

    def __repr__(self):
        """Return string representation of a :class:`pyttb.tenmat`.

        Examples
        --------
        Print an empty :class:`pyttb.tenmat`.

        >>> ttb.tenmat()  # doctest: +NORMALIZE_WHITESPACE
        matrix corresponding to a tensor of shape () with order F
        rindices = [  ] (modes of tensor corresponding to rows)
        cindices = [  ] (modes of tensor corresponding to columns)
        data = []

        Print a non-empty :class:`pyttb.tenmat`.

        >>> TM = ttb.tenones((2, 2)).to_tenmat(np.array([0]))
        >>> TM  # doctest: +NORMALIZE_WHITESPACE
        matrix corresponding to a tensor of shape (2, 2) with order F
        rindices = [ 0 ] (modes of tensor corresponding to rows)
        cindices = [ 1 ] (modes of tensor corresponding to columns)
        data[:, :] =
        [[1. 1.]
         [1. 1.]]

        Returns
        -------
        str
            Contains the shape, row indices (rindices), column indices (cindices) and
            data as strings on different lines.
        """
        s = ""
        s += "matrix corresponding to a tensor of shape "
        s += str(np_to_python(self.tshape))
        s += f" with order {self.order}"
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


if __name__ == "__main__":
    import doctest  # pragma: no cover

    doctest.testmod()  # pragma: no cover
