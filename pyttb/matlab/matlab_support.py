"""A limited number of utilities to support users coming from MATLAB."""

# Copyright 2025 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.
from __future__ import annotations

import numpy as np

from pyttb.tensor import tensor

from .matlab_utilities import _matlab_array_str

PRINT_CLASSES = tensor | np.ndarray


def matlab_print(
    data: tensor | np.ndarray,
    format: str | None = None,
    name: str | None = None,
):
    """Print data in a format more similar to MATLAB.

    Parameters
    ----------
    data:
        Object to print.
    format:
        Numerical formatting.
    name:
        Name to display for the data.

    Raises
    ------
    ValueError
        If ``data`` is not a :class:`pyttb.tensor` or :class:`numpy.ndarray`.

    Examples
    --------
    Print a three-way :class:`pyttb.tensor` in MATLAB style::

        >>> from pyttb.matlab.matlab_support import matlab_print
        >>> T = ttb.tensor(np.arange(8), (2, 2, 2))
        >>> matlab_print(T)  # doctest: +NORMALIZE_WHITESPACE
        This is a tensor of shape 2 x 2 x 2
            data(:,:, 0) =
                0 2
                1 3
        <BLANKLINE>
            data(:,:, 1) =
                4 6
                5 7
        <BLANKLINE>

    Give the tensor a name and control the numerical format::

        >>> matlab_print(T, format="5.1f", name="T")  # doctest: +NORMALIZE_WHITESPACE
        T is a tensor of shape 2 x 2 x 2
            T(:,:, 0) =
                  0.0   2.0
                  1.0   3.0
        <BLANKLINE>
            T(:,:, 1) =
                  4.0   6.0
                  5.0   7.0
        <BLANKLINE>
    """
    if not isinstance(data, (tensor, np.ndarray)):
        raise ValueError(
            f"matlab_print only supports inputs of type {PRINT_CLASSES} but got"
            f" {type(data)}."
        )
    if isinstance(data, np.ndarray):
        print(_matlab_array_str(data, format, name))
        return
    print(data._matlab_str(format, name))
