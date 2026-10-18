"""Tucker decomposition via Alternating Least Squares."""

# Copyright 2025 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from __future__ import annotations

from numbers import Real
from typing import TYPE_CHECKING, Literal

import numpy as np

from pyttb.pyttb_utils import OneDArray, parse_one_d
from pyttb.ttensor import ttensor

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pyttb as ttb


def tucker_als(  # noqa: PLR0912, PLR0913, PLR0915
    input_tensor: ttb.tensor,
    rank: OneDArray,
    stoptol: float = 1e-4,
    maxiters: int = 1000,
    dimorder: OneDArray | None = None,
    init: Literal["random", "nvecs"] | Sequence[np.ndarray] | ttb.ttensor = "random",
    printitn: int = 1,
) -> tuple[ttensor, list[np.ndarray | None], dict]:
    """Compute Tucker decomposition with alternating least squares.

    Parameters
    ----------
    input_tensor:
        Tensor to decompose.
    rank:
        Rank of the decomposition(s)
    stoptol:
        Tolerance used for termination - when the change in the fitness function
        in successive iterations drops below this value, the iterations terminate
    maxiters:
        Maximum number of iterations
    dimorder:
        Order to loop through dimensions (default: [range(tensor.ndims)])
    init:
        Initial guess (default: "random")
         * "random": initialize the factor matrices with values chosen from a
            uniform distribution on [0, 1)
         * "nvecs": initialize the factor matrices using the eigenvectors of the
            outer product of the matricized input tensor
         * list of :class:`numpy.ndarray`: one initial factor matrix per mode, each
            of shape ``(input_tensor.shape[n], rank[n])``; the entry for the first
            mode in ``dimorder`` is not used
         * :class:`pyttb.ttensor`: use its factor matrices as the initial guess,
            which must match the shape of the input tensor and the input rank
    printitn:
        Number of iterations to perform before printing iteration status:
        0 for no status printing

    Returns
    -------
    M:
        Resulting ttensor from Tucker-ALS factorization
    M_init:
        List of the initial factor matrices. For ``"random"`` and ``"nvecs"`` the
        entry for the first mode in ``dimorder`` is ``None``, because that mode is
        solved for first.
    M_info:
        Information about the computation. Dictionary keys:

        * `params` : dict of the input parameters with keys `stoptol`, `maxiters`,
            `printitn`, `dimorder`
        * `iters`: number of iterations performed
        * `normresidual`: norm of the difference between the input tensor and the
            Tucker model
        * `fit`: value of the fitness function (fraction of tensor data explained by
            the model)

    Raises
    ------
    ValueError
        If ``stoptol``, ``maxiters``, or ``printitn`` are not valid scalars, if
        ``dimorder`` is not a permutation of the modes, if ``init`` has the wrong
        length or factor shapes, or if ``init`` is not supported.

    See Also
    --------
    :func:`pyttb.hosvd`: Tucker decomposition via higher-order SVD.

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

    Compute a Tucker decomposition with a ``2 x 2 x 1`` core::

        >>> M, M_init, M_info = ttb.tucker_als(X, [2, 2, 1], printitn=0)
        >>> M.core.shape
        (2, 2, 1)
        >>> [factor.shape for factor in M.factor_matrices]
        [(5, 2), (4, 2), (3, 1)]
        >>> sorted(M_info.keys())
        ['fit', 'iters', 'normresidual', 'params']

    Use a single rank for every mode::

        >>> M_rank1, M_rank1_init, M_rank1_info = ttb.tucker_als(X, 1, printitn=0)
        >>> M_rank1.core.shape
        (1, 1, 1)

    **Tutorial**

    See the :doc:`/tutorial/algorithm_tucker_als` tutorial for more examples.
    """
    N = input_tensor.ndims
    normX = input_tensor.norm()

    # TODO: These argument checks look common with CP-ALS factor out
    if not isinstance(stoptol, Real):
        raise ValueError(
            f"stoptol must be a real valued scalar but received: {stoptol}"
        )
    if not isinstance(maxiters, Real) or maxiters < 0:
        raise ValueError(
            f"maxiters must be a non-negative real valued scalar but received: "
            f"{maxiters}"
        )
    if not isinstance(printitn, Real):
        raise ValueError(
            f"printitn must be a real valued scalar but received: {printitn}"
        )

    rank = parse_one_d(rank)
    if len(rank) == 1:
        rank = rank.repeat(N)

    # Set up dimorder if not specified
    if dimorder is None:
        dimorder = np.arange(N)
    else:
        dimorder = parse_one_d(dimorder)
        if tuple(range(N)) != tuple(sorted(dimorder)):
            raise ValueError("Dimorder must be a permutation of range(tensor.ndims)")

    if isinstance(init, ttensor):
        init = init.factor_matrices
    if isinstance(init, (list, tuple)):
        Uinit = list(init)
        if len(init) != N:
            raise ValueError(
                f"Init needs to be of length tensor.ndim (which was {N}) but only got "
                f"length {len(init)}."
            )
        for n in dimorder[1::]:
            correct_shape = (input_tensor.shape[n], rank[n])
            if Uinit[n].shape != correct_shape:
                raise ValueError(
                    f"Init factor {n} had incorrect shape. Expected {correct_shape} "
                    f"but got {Uinit[n].shape}"
                )
    elif isinstance(init, str) and init.lower() == "random":
        Uinit = [None] * N
        # Observe that we don't need to calculate an initial guess for the
        # first index in dimorder because that will be solved for in the first
        # inner iteration.
        for n in dimorder[1::]:
            Uinit[n] = np.random.uniform(0, 1, (input_tensor.shape[n], rank[n]))
    elif isinstance(init, str) and init.lower() in ("nvecs", "eigs"):
        # Compute an orthonormal basis for the dominant
        # Rn-dimensional left singular subspace of
        # X_(n) (0 <= n < N).
        Uinit = [None] * N
        for n in dimorder[1::]:
            if printitn > 0:
                print(f" Computing {rank[n]} leading e-vector for factor {n}.\n")
            Uinit[n] = input_tensor.nvecs(n, rank[n])
    else:
        raise ValueError(
            f"The selected initialization method is not supported. Provided: {init}"
        )

    # Set up for iterations - initializing U and the fit.
    U = Uinit.copy()
    fit = 0

    if printitn > 0:
        print("\nTucker Alternating Least-Squares:\n")

    # Main loop: Iterate until convergence
    for iteration in range(maxiters):
        fitold = fit

        # Iterate over all N modes of the tensor
        for n in dimorder:
            Utilde = input_tensor.ttm(U, exclude_dims=n, transpose=True)
            # Maximize norm(Utilde x_n W') wrt W and
            # maintain orthonormality of W
            U[n] = Utilde.nvecs(n, rank[n])

        # Assemble the current approximation
        core = Utilde.ttm(U, n, transpose=True)

        # Compute fit
        normresidual = np.sqrt(abs(normX**2 - core.norm() ** 2))
        fit = 1 - (normresidual / normX)  # fraction explained by model
        fitchange = abs(fitold - fit)

        if (printitn > 0) and (divmod(iteration, printitn)[1] == 0):
            print(f" Iter {iteration}: fit = {fit:e} fitdelta = {fitchange:7.1e}")

        # Check for convergence
        if fitchange < stoptol:
            break

    solution = ttensor(core, U, copy=False)

    output = {
        "params": {
            "stoptol": stoptol,
            "maxiters": maxiters,
            "printitn": printitn,
            "dimorder": dimorder,
        },
        "iters": iteration,
        "normresidual": normresidual,
        "fit": fit,
    }

    return solution, Uinit, output
