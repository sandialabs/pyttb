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
    import pyttb as ttb


def tucker_als(  # noqa: PLR0912, PLR0913, PLR0915
    input_tensor: ttb.tensor,
    rank: OneDArray,
    stoptol: float = 1e-4,
    maxiters: int = 1000,
    dimorder: OneDArray | None = None,
    init: Literal["random"] | Literal["nvecs"] | ttb.ktensor = "random",
    printitn: int = 1,
) -> tuple[ttensor, ttensor, dict]:
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
    dimorder:
        Order to loop through dimensions (default: [range(tensor.ndims)])
    maxiters:
        Maximum number of iterations
    init:
        Initial guess (default: "random")
         * "random": initialize using a :class:`pyttb.ttensor` with values chosen from
            a Normal distribution with mean 0 and standard deviation 1
         * "nvecs": initialize factor matrices of a :class:`pyttb.ttensor` using the
            eigenvectors of the outer product of the matricized input tensor
         * :class:`pyttb.ttensor`: initialize using a specific :class:`pyttb.ttensor`
            as input - must be the same shape as the input tensor and have the same
            rank as the input rank

    printitn:
        Number of iterations to perform before printing iteration status:
        0 for no status printing

    Returns
    -------
    M:
        Resulting ttensor from Tucker-ALS factorization
    Minit:
        Initial guess
    output:
        Information about the computation. Dictionary keys:

        * `params` : tuple of (stoptol, maxiters, printitn, dimorder)
        * `iters`: number of iterations performed
        * `normresidual`: norm of the difference between the input tensor and ktensor
            factorization
        * `fit`: value of the fitness function (fraction of tensor data explained by
            the model)

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

    if isinstance(init, list):
        Uinit = init
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
        "params": (stoptol, maxiters, printitn, dimorder),
        "iters": iteration,
        "normresidual": normresidual,
        "fit": fit,
    }

    return solution, Uinit, output
