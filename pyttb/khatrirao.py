"""Khatri-Rao Product Implementation."""

# Copyright 2025 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.
from __future__ import annotations

import numpy as np


def khatrirao(*matrices: np.ndarray, reverse: bool = False) -> np.ndarray:
    """
    Khatri-Rao product of matrices.

    Computes the Khatri-Rao product of matrices that have the same number of
    columns. For two matrices ``A`` and ``B`` the result is the column-wise Kronecker
    product ``[kron(A[:, 0], B[:, 0]), ..., kron(A[:, n], B[:, n])]``.

    Parameters
    ----------
    *matrices:
        Matrices to take the product of.
    reverse:
        Set to true to calculate product in reverse.

    Returns
    -------
    Matrix with the same number of columns as the inputs and as many rows as the
    product of the numbers of rows of the inputs.

    Raises
    ------
    ValueError
        If a single list is passed instead of separate matrices, or if ``reverse``
        is not a bool.

    Examples
    --------
    Compute the Khatri-Rao product of two matrices with two columns each::

        >>> np.random.seed(1)
        >>> A = np.random.normal(size=(5, 2))
        >>> B = np.random.normal(size=(4, 2))
        >>> khatrirao(A, B).shape
        (20, 2)
        >>> np.allclose(khatrirao(B, A, reverse=True), khatrirao(A, B))
        True

    Pass multiple matrices, or a list via argument unpacking::

        >>> C = np.random.normal(size=(3, 2))
        >>> khatrirao(A, B, C).shape
        (60, 2)
        >>> np.allclose(khatrirao(*[A, B, C]), khatrirao(A, B, C))
        True
    """
    # Determine if list of matrices of multiple matrix arguments
    if len(matrices) == 1 and isinstance(matrices[0], list):
        raise ValueError(
            "Khatrirao interface has changed. Instead of "
            " `khatrirao([matrix_a, matrix_b])` please update to use argument "
            "unpacking `khatrirao(*[matrix_a, matrix_b])`. This reduces ambiguity "
            "in usage moving forward. "
        )

    if not isinstance(reverse, bool):
        raise ValueError(f"Expected a bool for reverse but received {reverse}")

    # Error checking on input and set matrix order
    if reverse is True:
        matrices = tuple(reversed(matrices))
    if not all(len(matrix.shape) == 2 for matrix in matrices):
        assert False, "Each argument must be a matrix"

    ncolFirst = matrices[0].shape[1]
    if not all(matrix.shape[1] == ncolFirst for matrix in matrices):
        assert False, "All matrices must have the same number of columns."

    # Computation
    P = matrices[0]
    for i in matrices[1:]:
        P = np.reshape(i, (-1, 1, ncolFirst)) * np.reshape(
            P, (1, -1, ncolFirst), order="F"
        )
    return np.reshape(P, (-1, ncolFirst), order="F")
