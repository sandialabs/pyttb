"""Classes and functions for working with implicit sums of tensors."""

# Copyright 2025 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from __future__ import annotations

import warnings
from copy import deepcopy
from textwrap import indent
from typing import TYPE_CHECKING, Literal

import pyttb as ttb
from pyttb.pyttb_utils import np_to_python

if TYPE_CHECKING:
    import numpy as np


class sumtensor:
    """Class for implicit sum of other tensors."""

    def __init__(
        self,
        tensors: list[ttb.tensor | ttb.sptensor | ttb.ktensor | ttb.ttensor]
        | None = None,
        copy: bool = True,
    ):
        """Create a :class:`pyttb.sumtensor` from a collection of tensors.

        Each provided tensor is explicitly retained. All provided tensors
        must have the same shape but can be combinations of types.

        Parameters
        ----------
        tensors:
            Tensor source data.
        copy:
            Whether to make a copy of provided data or just reference it.

        Examples
        --------
        Create an empty :class:`pyttb.tensor`:

        >>> T1 = ttb.tenones((3, 4, 5))
        >>> T2 = ttb.sptensor(shape=(3, 4, 5))
        >>> S = ttb.sumtensor([T1, T2])
        """
        if tensors is None:
            tensors = []
        assert isinstance(tensors, list), (
            "Collection of tensors must be provided as a list "
            f"but received: {type(tensors)}"
        )
        assert all(tensors[0].shape == tensor_i.shape for tensor_i in tensors[1:]), (
            "All tensors must be the same shape"
        )
        if copy:
            tensors = deepcopy(tensors)
        self.parts = tensors

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

    def copy(self) -> sumtensor:
        """Make a deep copy of a :class:`pyttb.sumtensor`.

        Returns
        -------
        Copy of original sumtensor.

        Examples
        --------
        >>> T1 = ttb.tensor(np.ones((3, 2)))
        >>> S1 = ttb.sumtensor([T1, T1])
        >>> S2 = S1
        >>> S3 = S2.copy()
        >>> S1.parts[0][0, 0] = 3
        >>> S1.parts[0][0, 0] == S2.parts[0][0, 0]
        True
        >>> S1.parts[0][0, 0] == S3.parts[0][0, 0]
        False
        """
        return ttb.sumtensor(self.parts, copy=True)

    def __deepcopy__(self, memo):
        """Return deepcopy of this sumtensor."""
        return self.copy()

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of a :class:`pyttb.sumtensor`."""
        if len(self.parts) == 0:
            return ()
        return self.parts[0].shape

    def __repr__(self):
        """Return string representation of the sumtensor.

        Returns
        -------
        String displaying shape and constituent parts.

        Examples
        --------
        >>> T1 = ttb.tenones((2, 2))
        >>> T2 = ttb.sptensor(shape=(2, 2))
        >>> ttb.sumtensor([T1, T2])  # doctest: +NORMALIZE_WHITESPACE
        sumtensor of shape (2, 2) with 2 parts:
        Part 0:
            tensor of shape (2, 2) with order F
            data[:, :] =
            [[1. 1.]
             [1. 1.]]
        Part 1:
            empty sparse tensor of shape (2, 2) with order F
        """
        if len(self.parts) == 0:
            return "Empty sumtensor"
        s = (
            f"sumtensor of shape {np_to_python(self.shape)} "
            f"with {len(self.parts)} parts:"
        )
        for i, part in enumerate(self.parts):
            s += f"\nPart {i}: \n"
            s += indent(str(part), prefix="\t")
        return s

    __str__ = __repr__

    @property
    def ndims(self) -> int:
        """
        Number of dimensions of the sumtensor.

        Examples
        --------
        >>> T1 = ttb.tenones((2, 2))
        >>> S = ttb.sumtensor([T1, T1])
        >>> S.ndims
        2
        """
        return self.parts[0].ndims

    def __pos__(self):
        """
        Unary plus (+) for tensors.

        Returns
        -------
        Copy of sumtensor.

        Examples
        --------
        >>> T = ttb.tensor(np.array([[1, 2], [3, 4]]))
        >>> S = ttb.sumtensor([T, T])
        >>> S2 = +S
        """
        return self.copy()

    def __neg__(self):
        """
        Unary minus (-) for tensors.

        Returns
        -------
        Copy of negated sumtensor.

        Examples
        --------
        >>> T = ttb.tensor(np.array([[1, 2], [3, 4]]))
        >>> S = ttb.sumtensor([T, T])
        >>> S2 = -S
        >>> S2.parts[0].isequal(-1 * S.parts[0])
        True
        """
        return ttb.sumtensor([-part for part in self.parts], copy=False)

    def __add__(self, other):
        """
        Binary addition (+) for sumtensors.

        Parameters
        ----------
        other: :class:`pyttb.tensor`, :class:`pyttb.sptensor`
            :class:`pyttb.ktensor`, :class:`pyttb.ttensor`, or list
            containing those classes

        Returns
        -------
        :class:`pyttb.sumtensor`

        Examples
        --------
        >>> T = ttb.tenones((2, 2))
        >>> S = ttb.sumtensor([T])
        >>> len(S.parts)
        1
        >>> S2 = S + T
        >>> len(S2.parts)
        2
        >>> S3 = S2 + [T, T]
        >>> len(S3.parts)
        4
        """
        updated_parts = self.parts.copy()
        if isinstance(other, (ttb.tensor, ttb.sptensor, ttb.ktensor, ttb.ttensor)):
            updated_parts.append(other)
        elif isinstance(other, list) and all(
            isinstance(part, (ttb.tensor, ttb.sptensor, ttb.ktensor, ttb.ttensor))
            for part in other
        ):
            updated_parts.extend(other)
        else:
            raise TypeError(
                "Sumtensor only supports collections of tensor, sptensor, ktensor, "
                f"and ttensor but received: {type(other)}"
            )
        return ttb.sumtensor(updated_parts, copy=False)

    def __radd__(self, other):
        """
        Right Binary addition (+) for sumtensors.

        Parameters
        ----------
        other: :class:`pyttb.tensor`, :class:`pyttb.sptensor`
            :class:`pyttb.ktensor`, :class:`pyttb.ttensor`, or list
            containing those classes

        Returns
        -------
        :class:`pyttb.sumtensor`

        Examples
        --------
        >>> T = ttb.tenones((2, 2))
        >>> S = ttb.sumtensor([T])
        >>> len(S.parts)
        1
        >>> S2 = T + S
        >>> len(S2.parts)
        2
        >>> S3 = [T, T] + S2
        >>> len(S3.parts)
        4
        """
        return self.__add__(other)

    def to_tensor(self) -> ttb.tensor:
        """Return sumtensor converted to dense tensor.

        Same as :meth:`pyttb.sumtensor.full`.
        """
        return self.full()

    def full(self) -> ttb.tensor:
        """
        Convert a :class:`pyttb.sumtensor` to a :class:`pyttb.tensor`.

        Returns
        -------
        Re-assembled dense tensor.

        Examples
        --------
        >>> T = ttb.tenones((2, 2))
        >>> S = ttb.sumtensor([T, T])
        >>> print(S.full())  # doctest: +NORMALIZE_WHITESPACE
        tensor of shape (2, 2) with order F
        data[:, :] =
        [[2. 2.]
         [2. 2.]]
        <BLANKLINE>
        """
        result = self.parts[0].full()
        for part in self.parts[1:]:
            result += part
        return result

    def double(self, immutable: bool = False) -> np.ndarray:
        """
        Convert :class:`pyttb.tensor` to an :class:`numpy.ndarray` of doubles.

        Parameters
        ----------
        immutable: Whether or not the returned data cam be mutated. May enable
            additional optimizations.

        Examples
        --------
        >>> T = ttb.tenones((2, 2))
        >>> S = ttb.sumtensor([T, T])
        >>> S.double()
        array([[2., 2.],
               [2., 2.]])
        """
        return self.full().double(immutable)

    def innerprod(
        self, other: ttb.tensor | ttb.sptensor | ttb.ktensor | ttb.ttensor
    ) -> float:
        """Efficient inner product between a sumtensor and other `pyttb` tensors.

        Parameters
        ----------
        other:
            Tensor to take an innerproduct with.

        Examples
        --------
        >>> T1 = ttb.tensor(np.array([[1.0, 0.0], [0.0, 4.0]]))
        >>> T2 = T1.to_sptensor()
        >>> S = ttb.sumtensor([T1, T2])
        >>> T1.innerprod(T1)
        17.0
        >>> T1.innerprod(T2)
        17.0
        >>> S.innerprod(T1)
        34.0
        """
        result = self.parts[0].innerprod(other)
        for part in self.parts[1:]:
            result += part.innerprod(other)
        return result

    def mttkrp(
        self, U: ttb.ktensor | list[np.ndarray], n: int | np.integer
    ) -> np.ndarray:
        """Matricized tensor times Khatri-Rao product.

        The matrices used in the
        Khatri-Rao product are passed as a :class:`pyttb.ktensor` (where the
        factor matrices are used) or as a list of :class:`numpy.ndarray` objects.

        Parameters
        ----------
        U:
            Matrices to create the Khatri-Rao product.
        n:
            Mode used to matricize tensor.

        Returns
        -------
        Array containing matrix product.

        Examples
        --------
        >>> T1 = ttb.tenones((2, 2, 2))
        >>> T2 = T1.to_sptensor()
        >>> S = ttb.sumtensor([T1, T2])
        >>> U = [np.ones((2, 2))] * 3
        >>> T1.mttkrp(U, 2)
        array([[4., 4.],
               [4., 4.]])
        >>> S.mttkrp(U, 2)
        array([[8., 8.],
               [8., 8.]])
        """
        result = self.parts[0].mttkrp(U, n)
        for part in self.parts[1:]:
            result += part.mttkrp(U, n)
        return result

    def ttv(
        self,
        vector: np.ndarray | list[np.ndarray],
        dims: int | np.ndarray | None = None,
        exclude_dims: int | np.ndarray | None = None,
    ) -> float | sumtensor:
        """
        Tensor times vector.

        Computes the n-mode product of `parts` with the vector `vector`; i.e.,
        `self x_n vector`. The integer `n` specifies the dimension (or mode)
        along which the vector should be multiplied. If `vector.shape = (I,)`,
        then the sumtensor must have `self.shape[n] = I`. The result will be the
        same order and shape as `self` except that the size of dimension `n`
        will be `J`. The resulting parts of the sum tensor have one less dimension,
        as dimension `n` is removed in the multiplication.

        Multiplication with more than one vector is provided using a list of
        vectors and corresponding dimensions in the tensor to use.

        The dimensions of the tensor with which to multiply can be provided as
        `dims`, or the dimensions to exclude from `[0, ..., self.ndims]` can be
        specified using `exclude_dims`.

        Parameters
        ----------
        vector:
            Vector or vectors to multiple by.
        dims:
            Dimensions to multiply against.
        exclude_dims:
            Use all dimensions but these.

        Returns
        -------
        Sumtensor containing individual products or a single sum if every
            product is a single value.

        Examples
        --------
        >>> T = ttb.tensor(np.array([[1, 2], [3, 4]]))
        >>> S = ttb.sumtensor([T, T])
        >>> T.ttv(np.ones(2), 0)
        tensor of shape (2,) with order F
        data[:] =
        [4. 6.]
        >>> S.ttv(np.ones(2), 0)  # doctest: +NORMALIZE_WHITESPACE
        sumtensor of shape (2,) with 2 parts:
        Part 0:
             tensor of shape (2,) with order F
             data[:] =
             [4. 6.]
        Part 1:
             tensor of shape (2,) with order F
             data[:] =
             [4. 6.]
        >>> T.ttv([np.ones(2), np.ones(2)])
        10.0
        >>> S.ttv([np.ones(2), np.ones(2)])
        20.0
        """
        new_parts = []
        scalar_sum = 0.0
        for part in self.parts:
            result = part.ttv(vector, dims, exclude_dims)
            if isinstance(result, float):
                scalar_sum += result
            else:
                new_parts.append(result)
        if len(new_parts) == 0:
            return scalar_sum
        assert scalar_sum == 0.0
        return ttb.sumtensor(new_parts, copy=False)

    def norm(self) -> float:
        """Compatibility Interface. Just returns 0."""
        warnings.warn(
            "Sumtensor doesn't actually support norm. Returning 0 for compatibility."
        )
        return 0.0


if __name__ == "__main__":
    import doctest  # pragma: no cover

    doctest.testmod()  # pragma: no cover
