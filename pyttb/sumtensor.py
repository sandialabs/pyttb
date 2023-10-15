"""Sum Tensor Class Placeholder"""
# Copyright 2022 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.
from __future__ import annotations

from copy import deepcopy
from textwrap import indent
from typing import List, Tuple, Union

import numpy as np

import pyttb as ttb


class sumtensor:
    """
    SUMTENSOR Class for implicit sum of other tensors.

    """

    def __init__(
        self,
        tensors: List[Union[ttb.tensor, ttb.sptensor, ttb.ktensor, ttb.ttensor]],
        copy: bool = True,
    ):
        """
        Creates a :class:`pyttb.sumtensor` from a collection of tensors.
        Each provided tensor is explicitly retained. All provided tensors
        must have the same shape but can be combinations of types.

        Parameters
        ----------
        tensors:
            Tensor source data.
        copy:
            Whether to make a copy of provided data or just reference it.

        Examples
        -------
        Create an empty :class:`pyttb.tensor`:

        >>> T1 = ttb.tenones((3,4,5))
        >>> T2 = ttb.sptensor(shape=(3,4,5))
        >>> S = ttb.sumtensor([T1, T2])
        """
        assert all(
            tensors[0].shape == tensor_i.shape for tensor_i in tensors[1:]
        ), "All tensors must be the same shape"
        if copy:
            tensors = deepcopy(tensors)
        self.parts = tensors

    def copy(self) -> sumtensor:
        """Make a deep copy of a :class:`pyttb.sumtensor`.

        Returns
        -------
        Copy of original sumtensor.

        Examples
        --------
        >>> T1 = ttb.tensor(np.ones((3,2)))
        >>> S1 = ttb.sumtensor([T1, T1])
        >>> S2 = S1
        >>> S3 = S2.copy()
        >>> S1.parts[0][0,0] = 3
        >>> S1.parts[0][0,0] == S2.parts[0][0,0]
        True
        >>> S1.parts[0][0,0] == S3.parts[0][0,0]
        False
        """
        return ttb.sumtensor(self.parts, copy=True)

    def __deepcopy__(self, memo):
        return self.copy()

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.parts[0].shape

    def __repr__(self):
        """
        String representation of the sumtensor.

        Returns
        -------
        String displaying shape and constituent parts.

        Examples
        --------
        >>> T1 = ttb.tenones((2,2))
        >>> T2 = ttb.sptensor(shape=(2,2))
        >>> ttb.sumtensor([T1, T2]) # doctest: +NORMALIZE_WHITESPACE
        sumtensor of shape (2, 2) with 2 parts:
        Part 0:
            tensor of shape (2, 2)
            data[:, :] =
            [[1. 1.]
            [1. 1.]]
        Part 1:
            All-zero sparse tensor of shape 2 x 2
        """
        s = f"sumtensor of shape {self.shape} with {len(self.parts)} parts:"
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
        >>> S = ttb.sumtensor([T1,T1])
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
        >>> T = ttb.tenones((2,2))
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
        >>> T = ttb.tenones((2,2))
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

    def full(self) -> ttb.tensor:
        """
        Convert a :class:`pyttb.sumtensor` to a :class:`pyttb.tensor`.

        Returns
        -------
        Re-assembled dense tensor.

        Examples
        --------
        >>> T = ttb.tenones((2,2))
        >>> S = ttb.sumtensor([T, T])
        >>> print(S.full()) # doctest: +NORMALIZE_WHITESPACE
        tensor of shape (2, 2)
        data[:, :] =
        [[2. 2.]
         [2. 2.]]
        <BLANKLINE>
        """
        result = self.parts[0].full()
        for part in self.parts[1:]:
            result += part
        return result

    def double(self) -> np.ndarray:
        """
        Convert `:class:pyttb.tensor` to an `:class:numpy.ndarray` of doubles.

        Returns
        -------
        Copy of tensor data.

        Examples
        --------
        >>> T = ttb.tenones((2,2))
        >>> S = ttb.sumtensor([T, T])
        >>> S.double()
        array([[2., 2.],
               [2., 2.]])
        """
        return self.full().double()
