"""Higher Order SVD Implementation."""

# Copyright 2025 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from __future__ import annotations

import warnings

import numpy as np
import scipy

import pyttb as ttb
from pyttb.pyttb_utils import OneDArray, parse_one_d


def hosvd(  # noqa: PLR0912,PLR0913,PLR0915
    input_tensor: ttb.tensor,
    tol: float,
    verbosity: float = 1,
    dimorder: OneDArray | None = None,
    sequential: bool = True,
    ranks: OneDArray | None = None,
) -> ttb.ttensor:
    """Compute sequentially-truncated higher-order SVD (Tucker).

    Computes a Tucker decomposition with relative error
    specified by tol, i.e., it computes a ttensor M such that
    ||X-M||/||X|| <= tol.

    Parameters
    ----------
    input_tensor:
        Tensor to factor
    tol:
        Relative error to stop at
    verbosity:
        Print level. Values above 0 print a summary, above 2 also print norms and
        thresholds, and above 5 also print eigenvalue sums per mode; values of 0 or
        less are silent.
    dimorder:
        Order to loop through dimensions
    sequential:
        Use sequentially-truncated version
    ranks:
        Specify ranks to consider rather than computing

    Returns
    -------
    :
        Resulting :class:`pyttb.ttensor` from the HOSVD factorization.

    Raises
    ------
    ValueError
        If ``ranks`` does not have one entry per mode, or ``dimorder`` is not a
        permutation of the modes.

    See Also
    --------
    :func:`pyttb.tucker_als`: Tucker decomposition via alternating least squares.

    Examples
    --------
    Create a :class:`pyttb.ttensor` with a ``2 x 2 x 2`` core and orthonormal factor
    matrices, and generate a noisy dense data tensor from it by sampling each entry
    from a normal distribution with standard deviation one and mean equal to the
    corresponding entry of the :class:`pyttb.ttensor`::

        >>> np.random.seed(1)
        >>> factors = [np.linalg.qr(np.random.rand(n, 2))[0] for n in (5, 4, 3)]
        >>> core = ttb.tensor(30 * np.random.rand(2, 2, 2))
        >>> M_true = ttb.ttensor(core, factors)
        >>> X = ttb.tensor(np.random.normal(loc=M_true.full().data, scale=1.0))
        >>> X.shape
        (5, 4, 3)

    Compute a Tucker decomposition with a relative error of at most ``tol``. The
    ranks are chosen automatically and match the multilinear rank of the model::

        >>> tol = 0.25
        >>> disable_printing = -1
        >>> M = hosvd(X, tol, verbosity=disable_printing)
        >>> M.core.shape
        (2, 2, 2)
        >>> ((M.full() - X).norm() / X.norm()) < tol
        True

    Specify the ranks of the decomposition rather than a tolerance::

        >>> M = hosvd(X, tol, verbosity=disable_printing, ranks=[2, 2, 1])
        >>> M.core.shape
        (2, 2, 1)

    **Tutorial**

    See the :doc:`/tutorial/algorithm_hosvd` tutorial for more examples.
    """
    # In tucker als this is N
    d = input_tensor.ndims

    if ranks is None:
        ranks = np.zeros((d,), dtype=int)
    else:
        ranks = parse_one_d(ranks)

    if len(ranks) != d:
        raise ValueError(
            "Ranks must be a sequence of length tensor ndims."
            f" Ndims: {d} but got ranks: {ranks}."
        )

    # Set up dimorder if not specified (this is copy past from tucker_als
    if dimorder is None:
        dimorder = np.arange(d)
    else:
        dimorder = parse_one_d(dimorder)
        if tuple(range(d)) != tuple(sorted(dimorder)):
            raise ValueError(
                "Dimorder must be a list or permutation of range(tensor.ndims)"
            )

    # TODO should unify printing throughout. Probably easier to use python logging
    #  levels
    if verbosity > 0:
        print("Computing HOSVD...\n")

    normxsqr = (input_tensor**2).collapse()
    eigsumthresh = ((tol**2) * normxsqr) / d

    if verbosity > 2:
        print(
            f"||X||^2 = {normxsqr: g}\n"
            f"tol = {tol: g}\n"
            f"eigenvalue sum threshold = tol^2 ||X||^2 / d = {eigsumthresh: g}"
        )

    # Main Loop
    factor_matrices = [np.empty(1)] * d
    # Copy input tensor, shrinks every step for sequential
    Y = input_tensor.copy()

    for k in dimorder:
        # Compute Gram matrix
        Yk = Y.to_tenmat(np.array([k])).double()
        Z = np.dot(Yk, Yk.transpose())

        # Compute eigenvalue decomposition
        D, V = scipy.linalg.eigh(Z)
        pi = np.argsort(-D, kind="quicksort")
        eigvec = D[pi]

        # If rank not provided compute it.
        if ranks[k] == 0:
            eigsum = np.cumsum(eigvec[::-1])
            eigsum = eigsum[::-1]
            # +1 because the 0th element corresponds to rank 1
            ranks[k] = np.where(eigsum > eigsumthresh)[0][-1] + 1

            if verbosity > 5:
                print("Reverse cumulative sum of evals of Gram matrix:")
                for i, a_sum in enumerate(eigsum):
                    rank = i + 1
                    print_msg = f"{rank: d}: {a_sum: 6.4f}"
                    if rank == ranks[k]:
                        print_msg += " <-- Cutoff"
                    print(print_msg)

        # Extract factor matrix b picking leading eigenvectors of V
        factor_matrices[k] = V[:, pi[0 : ranks[k]]]

        # Shrink!
        if sequential:
            Y = Y.ttm(factor_matrices[k].transpose(), int(k))
    # Extract final core
    if sequential:
        G = Y
    else:
        G = Y.ttm(factor_matrices, transpose=True)

    result = ttb.ttensor(G, factor_matrices, copy=False)

    if verbosity > 0:
        diffnormsqr = ((input_tensor - result.full()) ** 2).collapse()
        relnorm = np.sqrt(diffnormsqr / normxsqr)
        print(f"Shape of core: {G.shape}")
        if relnorm <= tol:
            print(f"||X-T||/||X|| = {relnorm: g} <={tol: f} (tol)")
        else:
            print(
                "Tolerance not satisfied!! "
                f"||X-T||/||X|| = {relnorm: g} >="
                f"{tol: f} (tol)"
            )
            warnings.warn("Specified tolerance was not achieved")
    return result
