"""Matricized Tensor Representation"""

# Copyright 2022 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

import pyttb as ttb
from pyttb.pyttb_utils import gather_wrap_dims


class tenmat:
    """
    TENMAT Store tensor as a matrix.

    """

    __slots__ = ("tshape", "rindices", "cindices", "data")

    def __init__(  # noqa: PLR0912, PLR0913
        self,
        data: Optional[np.ndarray] = None,
        rdims: Optional[np.ndarray] = None,
        cdims: Optional[np.ndarray] = None,
        tshape: Optional[Tuple[int, ...]] = None,
        copy: bool = True,
    ):
        """
        Construct a :class:`pyttb.tenmat` from explicit components.
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

        >>> ttb.tenmat() # doctest: +NORMALIZE_WHITESPACE
        matrix corresponding to a tensor of shape ()
        rindices = [  ] (modes of tensor corresponding to rows)
        cindices = [  ] (modes of tensor corresponding to columns)
        data = []

        Create tensor shaped data.

        >>> tshape = (2, 2, 2)
        >>> data = np.reshape(np.arange(np.prod(tshape), dtype=np.double), tshape)
        >>> data # doctest: +NORMALIZE_WHITESPACE
        array([[[0., 1.],
                [2., 3.]],
               [[4., 5.],
                [6., 7.]]])

        Manually matrize the tensor.

        >>> flat_data = np.reshape(data, (2,4), order="F")
        >>> flat_data # doctest: +NORMALIZE_WHITESPACE
        array([[0., 2., 1., 3.],
               [4., 6., 5., 7.]])

        Encode matrication into :class:`pyttb.tenmat`.

        >>> tm = ttb.tenmat(flat_data, rdims=np.array([0]), tshape=tshape)

        Extract original tensor shaped data.

        >>> tm.to_tensor().double() # doctest: +NORMALIZE_WHITESPACE
        array([[[0., 1.],
                [2., 3.]],
               [[4., 5.],
                [6., 7.]]])
        """

        # Case 0a: Empty Contructor
        # data is empty, return empty tenmat unless rdims, cdims, or tshape are
        # not empty
        if data is None or (isinstance(data, np.ndarray) and data.size == 0):
            cdims_empty = cdims is None or cdims.size == 0
            rdims_empty = rdims is None or rdims.size == 0
            tshape_empty = tshape is None or tshape == ()
            assert (
                rdims_empty and cdims_empty and tshape_empty
            ), "When data is empty, rdims, cdims, and tshape must also be empty."

            self.tshape: Union[Tuple[()], Tuple[int, ...]] = ()
            self.rindices = np.array([])
            self.cindices = np.array([])
            self.data = np.array([], ndmin=2, order="F")
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
                data = np.reshape(data.copy(), (1, data.shape[0]), order="F")

        if len(data.shape) != 2:
            raise ValueError(
                f"Data must be a matrix or vector but had {len(data.shape)} dimensions"
            )

        # use data.shape for tshape if not provided
        if tshape is None:
            tshape = data.shape
        elif not isinstance(tshape, tuple):
            assert False, "tshape must be a tuple."

        # check that data.shape and tshape agree
        if np.prod(data.shape) != np.prod(tshape):
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
        ) == np.prod(data.shape):
            assert (
                False
            ), "data.shape does not match shape specified by rdims, cdims, and tshape."

        # if rdims or cdims is empty, hstack will output an array of float not int
        if rdims.size == 0:
            dims = cdims.copy()
        elif cdims.size == 0:
            dims = rdims.copy()
        else:
            dims = np.hstack([rdims, cdims])
        if not len(dims) == n or not (alldims == np.sort(dims)).all():
            assert False, (
                "Incorrect specification of dimensions, the sorted concatenation "
                "of rdims and cdims must be range(source.ndims)."
            )

        self.tshape = tshape
        self.rindices = rdims.copy()
        self.cindices = cdims.copy()
        if copy:
            self.data = data.copy()
        else:
            self.data = data
        return

    def copy(self) -> tenmat:
        """
        Return a deep copy of the :class:`pyttb.tenmat`.

        Examples
        --------
        Create a :class:`pyttb.tenmat` (TM1) and make a deep copy. Verify
        the deep copy (TM3) is not just a reference (like TM2) to the original.

        >>> T1 = ttb.tensor(np.ones((3,2)))
        >>> TM1 = T1.to_tenmat(np.array([0]))
        >>> TM2 = TM1
        >>> TM3 = TM1.copy()
        >>> TM1[0,0] = 3

        # Item to convert numpy boolean to python boolena for nicer printing

        >>> (TM1[0,0] == TM2[0,0]).item()
        True
        >>> (TM1[0,0] == TM3[0,0]).item()
        False
        """
        # Create tenmat
        return ttb.tenmat(
            self.data, self.rindices, self.cindices, self.tshape, copy=True
        )

    def __deepcopy__(self, memo):
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
        >>> data # doctest: +NORMALIZE_WHITESPACE
        array([[[0., 1.],
                [2., 3.]],
               [[4., 5.],
                [6., 7.]]])

        Manually matrize the tensor.

        >>> flat_data = np.reshape(data, (2,4), order="F")
        >>> flat_data # doctest: +NORMALIZE_WHITESPACE
        array([[0., 2., 1., 3.],
               [4., 6., 5., 7.]])

        Encode matrication into :class:`pyttb.tenmat`.

        >>> tm = ttb.tenmat(flat_data, rdims=np.array([0]), tshape=tshape)

        Extract original tensor shaped data.

        >>> tm.to_tensor() # doctest: +NORMALIZE_WHITESPACE
        tensor of shape (2, 2, 2)
        data[0, :, :] =
        [[0. 1.]
         [2. 3.]]
        data[1, :, :] =
        [[4. 5.]
         [6. 7.]]
        """
        # RESHAPE TENSOR-AS-MATRIX
        # Here we just reverse what was done in the tenmat constructor.
        # First we reshape the data to be an MDA, then we un-permute
        # it using ipermute.
        shape = self.tshape
        order = np.hstack([self.rindices, self.cindices])
        data = self.data
        if copy:
            data = self.data.copy()
        data = np.reshape(data, np.array(shape)[order], order="F")
        if order.size > 1:
            data = np.transpose(data, np.argsort(order))
        return ttb.tensor(data, shape, copy=False)

    def ctranspose(self) -> tenmat:
        """
        Complex conjugate transpose for :class:`pyttb.tenmat`.

        Examples
        --------
        Create :class:`pyttb.tensor` then convert to :class:`pyttb.tenmat`.

        >>> T = ttb.tenones((2,2,2))
        >>> TM = T.to_tenmat(rdims=np.array([0]))
        >>> TM # doctest: +NORMALIZE_WHITESPACE
        matrix corresponding to a tensor of shape (2, 2, 2)
        rindices = [ 0 ] (modes of tensor corresponding to rows)
        cindices = [ 1, 2 ] (modes of tensor corresponding to columns)
        data[:, :] =
        [[1. 1. 1. 1.]
         [1. 1. 1. 1.]]
        >>> TM.ctranspose() # doctest: +NORMALIZE_WHITESPACE
        matrix corresponding to a tensor of shape (2, 2, 2)
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

    def double(self) -> np.ndarray:
        """
        Convert a :class:`pyttb.tenmat` to an array of doubles.

        Examples
        --------
        >>> T = ttb.tenones((2,2,2))
        >>> TM = T.to_tenmat(rdims=np.array([0]))
        >>> TM # doctest: +NORMALIZE_WHITESPACE
        matrix corresponding to a tensor of shape (2, 2, 2)
        rindices = [ 0 ] (modes of tensor corresponding to rows)
        cindices = [ 1, 2 ] (modes of tensor corresponding to columns)
        data[:, :] =
        [[1. 1. 1. 1.]
         [1. 1. 1. 1.]]
        >>> TM.double() # doctest: +NORMALIZE_WHITESPACE
        array([[1., 1., 1., 1.],
               [1., 1., 1., 1.]])

        Returns
        -------
        Copy of tenmat data.
        """
        return self.data.astype(np.float64).copy()

    @property
    def ndims(self) -> int:
        """Return the number of dimensions of a :class:`pyttb.tenmat`.

        Examples
        --------
        >>> TM = ttb.tenmat() # empty tenmat
        >>> TM.ndims
        0

        >>> TM = ttb.tenones((2,2,2)).to_tenmat(np.array([0]))
        >>> TM.ndims
        2
        """
        return len(self.shape)

    def norm(self) -> float:
        """
        Frobenius norm of a :class:`pyttb.tenmat`.

        Examples
        --------
        >>> T = ttb.tenones((2,2,2))
        >>> TM = T.to_tenmat(rdims=np.array([0]))
        >>> TM # doctest: +NORMALIZE_WHITESPACE
        matrix corresponding to a tensor of shape (2, 2, 2)
        rindices = [ 0 ] (modes of tensor corresponding to rows)
        cindices = [ 1, 2 ] (modes of tensor corresponding to columns)
        data[:, :] =
        [[1. 1. 1. 1.]
         [1. 1. 1. 1.]]
        >>> TM.norm() # doctest: +ELLIPSIS
        2.82...
        """
        # default of np.linalg.norm is to vectorize the data and compute the vector
        # norm, which is equivalent to the Frobenius norm for multidimensional arrays.
        # However, the argument 'fro' only works for 1-D and 2-D
        # arrays currently.
        return float(np.linalg.norm(self.data))

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of a :class:`pyttb.tenmat`.

        Examples
        --------
        >>> TM = ttb.tenmat() # empty tenmat
        >>> TM.shape
        ()

        >>> TM = ttb.tenones((2,2,2)).to_tenmat(np.array([0]))
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
        >>> TM1 = ttb.tenmat() # empty tenmat
        >>> TM2 = ttb.tenones((2,2,2)).to_tenmat(np.array([0]))
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
        >>> TM = ttb.tenones((2,2,2)).to_tenmat(np.array([0]))
        >>> TM # doctest: +NORMALIZE_WHITESPACE
        matrix corresponding to a tensor of shape (2, 2, 2)
        rindices = [ 0 ] (modes of tensor corresponding to rows)
        cindices = [ 1, 2 ] (modes of tensor corresponding to columns)
        data[:, :] =
        [[1. 1. 1. 1.]
         [1. 1. 1. 1.]]
        >>> TM[0, 0] = 2.
        >>> TM # doctest: +NORMALIZE_WHITESPACE
        matrix corresponding to a tensor of shape (2, 2, 2)
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
        >>> TM = ttb.tenones((2,2,2)).to_tenmat(np.array([0]))
        >>> TM[0, 0]
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
        >>> TM = ttb.tenones((2,2)).to_tenmat(np.array([0]))
        >>> TM * TM # doctest: +NORMALIZE_WHITESPACE
        matrix corresponding to a tensor of shape (2, 2)
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
                self.data @ other.data,
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
        >>> TM = ttb.tenones((2,2)).to_tenmat(np.array([0]))
        >>> TM * TM # doctest: +NORMALIZE_WHITESPACE
        matrix corresponding to a tensor of shape (2, 2)
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
        >>> TM = ttb.tenones((2,2)).to_tenmat(np.array([0]))
        >>> TM + TM # doctest: +NORMALIZE_WHITESPACE
        matrix corresponding to a tensor of shape (2, 2)
        rindices = [ 0 ] (modes of tensor corresponding to rows)
        cindices = [ 1 ] (modes of tensor corresponding to columns)
        data[:, :] =
        [[2. 2.]
         [2. 2.]]
        >>> TM + 1.0 # doctest: +NORMALIZE_WHITESPACE
        matrix corresponding to a tensor of shape (2, 2)
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
        >>> TM = ttb.tenones((2,2)).to_tenmat(np.array([0]))
        >>> 1.0 + TM # doctest: +NORMALIZE_WHITESPACE
        matrix corresponding to a tensor of shape (2, 2)
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
        >>> TM = ttb.tenones((2,2)).to_tenmat(np.array([0]))
        >>> TM - TM # doctest: +NORMALIZE_WHITESPACE
        matrix corresponding to a tensor of shape (2, 2)
        rindices = [ 0 ] (modes of tensor corresponding to rows)
        cindices = [ 1 ] (modes of tensor corresponding to columns)
        data[:, :] =
        [[0. 0.]
         [0. 0.]]
        >>> TM - 1.0 # doctest: +NORMALIZE_WHITESPACE
        matrix corresponding to a tensor of shape (2, 2)
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
        >>> TM = ttb.tenones((2,2)).to_tenmat(np.array([0]))
        >>> 1.0 - TM # doctest: +NORMALIZE_WHITESPACE
        matrix corresponding to a tensor of shape (2, 2)
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
        >>> TM = ttb.tenones((2,2)).to_tenmat(np.array([0]))
        >>> +TM # doctest: +NORMALIZE_WHITESPACE
        matrix corresponding to a tensor of shape (2, 2)
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
        >>> TM = ttb.tenones((2,2)).to_tenmat(np.array([0]))
        >>> -TM # doctest: +NORMALIZE_WHITESPACE
        matrix corresponding to a tensor of shape (2, 2)
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
        """
        String representation of a :class:`pyttb.tenmat`.

        Examples
        --------
        Print an empty :class:`pyttb.tenmat`.

        >>> ttb.tenmat() # doctest: +NORMALIZE_WHITESPACE
        matrix corresponding to a tensor of shape ()
        rindices = [  ] (modes of tensor corresponding to rows)
        cindices = [  ] (modes of tensor corresponding to columns)
        data = []

        Print a non-empty :class:`pyttb.tenmat`.

        >>> TM = ttb.tenones((2,2)).to_tenmat(np.array([0]))
        >>> TM # doctest: +NORMALIZE_WHITESPACE
        matrix corresponding to a tensor of shape (2, 2)
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
        s += str(self.tshape)
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
