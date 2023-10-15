"""Sum Tensor Class Placeholder"""
# Copyright 2022 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.
from __future__ import annotations

from copy import deepcopy
from textwrap import indent
from typing import List, Tuple, Union

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
