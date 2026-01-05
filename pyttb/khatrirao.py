"""Khatri-Rao Product Implementation."""

# Copyright 2025 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.
from __future__ import annotations

import numpy as np


def khatrirao(*matrices: np.ndarray, reverse: bool = False) -> np.ndarray:
    """
    KHATRIRAO Khatri-Rao product of matrices.

    KHATRIRAO(A,B) computes the Khatri-Rao product of matrices A and
    B that have the same number of columns.  The result is the
    column-wise Kronecker product
    [KRON(A(:,1),B(:,1)) ... KRON(A(:,n),B(:,n))]

    Parameters
    ----------
    matrices:
        Collection of matrices to take the product of
    reverse:
        Set to true to calculate product in reverse

    Examples
    --------
    >>> A = np.random.normal(size=(5, 2))
    >>> B = np.random.normal(size=(5, 2))
    >>> _ = khatrirao(A, B)  # <-- Khatri-Rao of A and B
    >>> _ = khatrirao(B, A, reverse=True)  # <-- same thing as above
    >>> _ = khatrirao(A, A, B)  # <-- passing multiple items
    >>> _ = khatrirao(B, A, A, reverse=True)  # <-- same as above
    >>> _ = khatrirao(*[A, A, B])  # <-- passing a list via unpacking items
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
