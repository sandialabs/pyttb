"""CP Decomposition via Alternating Least Squares."""

# Copyright 2025 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from __future__ import annotations

from typing import Literal

import numpy as np

import pyttb as ttb
from pyttb.pyttb_utils import OneDArray, parse_one_d


def cp_als(  # noqa: PLR0912,PLR0913,PLR0915
    input_tensor: ttb.tensor | ttb.sptensor | ttb.ttensor | ttb.sumtensor,
    rank: int,
    stoptol: float = 1e-4,
    maxiters: int = 1000,
    dimorder: OneDArray | None = None,
    optdims: OneDArray | None = None,
    init: Literal["random"] | Literal["nvecs"] | ttb.ktensor = "random",
    printitn: int = 1,
    fixsigns: bool = True,
) -> tuple[ttb.ktensor, ttb.ktensor, dict]:
    """Compute CP decomposition with alternating least squares.

    Parameters
    ----------
    input_tensor:
        Tensor to decompose
    rank:
        Rank of the decomposition
    stoptol:
        Tolerance used for termination - when the change in the fitness function
        in successive iterations drops below this value, the iterations terminate
    dimorder:
        Order to loop through dimensions (default: [range(tensor.ndims)])
    optdims:
        Whether factor for corresponding mode should be optimized or not.
        Defaults to optimizing all modes ([range(tensor.ndims)]).
    maxiters:
        Maximum number of iterations
    init:
        Initial guess (default: "random")
         * "random": initialize using a :class:`pyttb.ktensor` with values chosen
            from a Normal distribution with mean 0 and standard deviation 1
         * "nvecs": initialize factor matrices of a :class:`pyttb.ktensor` using
            the eigenvectors of the outer product of the matricized input tensor
         * :class:`pyttb.ktensor`: initialize using a specific
            :class:`pyttb.ktensor` as input - must be the same shape as the input
            tensor and have the same rank as the input rank
    printitn:
        Number of iterations to perform before printing iteration status - 0 for
        no status printing
    fixsigns:
        Align the signs of the columns of the factorization to align with the
        input tensor data

    Returns
    -------
    M:
        Resulting ktensor from CP-ALS factorization
    Minit:
        Initial guess
    output:
        Information about the computation. Dictionary keys:
         * `params` : tuple of (stoptol, maxiters, printitn, dimorder)
         * `iters`: number of iterations performed
         * `normresidual`: norm of the difference between the input tensor
            and ktensor factorization
         * `fit`: value of the fitness function (fraction of tensor data
            explained by the model)

    Examples
    --------
    Create a rank-2 :class:`pyttb.ktensor` and generate a noisy dense data tensor
    from it by sampling each entry from a normal distribution with standard
    deviation one and mean equal to the corresponding entry of the
    :class:`pyttb.ktensor`::

        >>> np.random.seed(1)
        >>> factors = [np.random.rand(n, 2) for n in (5, 4, 3)]
        >>> factors = [A / A.sum(axis=0) for A in factors]  # columns sum to one
        >>> K = ttb.ktensor(factors, weights=np.array([25.0, 15.0]))
        >>> X = ttb.tensor(np.random.normal(loc=K.full().data, scale=1.0))
        >>> X.shape
        (5, 4, 3)

    Compute a rank-2 decomposition using the default values (``"random"``
    initialization). Random initialization causes slight perturbation in the
    intermediate results, so the printed iteration status is not shown here::

        >>> M, Minit, output = ttb.cp_als(X, 2, printitn=0)
        >>> M.shape
        (5, 4, 3)
        >>> M.ncomponents
        2
        >>> round(float(output["normresidual"]), 1)  # norm of the residual
        5.3
        >>> round(float(output["fit"]), 2)  # fraction of the data explained
        0.48

    Use the ``"nvecs"`` initialization::

        >>> M_nvecs, _, output = ttb.cp_als(X, 2, init="nvecs", printitn=0)
        >>> round(float(output["fit"]), 2)
        0.48

    Use a specific :class:`pyttb.ktensor` as the initial guess, here the initial
    guess used by the first call::

        >>> M_rerun, _, output = ttb.cp_als(X, 2, init=Minit, printitn=0)
        >>> round(float(output["fit"]), 2)
        0.48
    """
    # Extract number of dimensions and norm of tensor
    N = input_tensor.ndims
    normX = input_tensor.norm()

    # Set up dimorder if not specified
    if dimorder is None:
        dimorder = np.arange(N)
    else:
        dimorder = parse_one_d(dimorder)
    if tuple(range(N)) != tuple(sorted(dimorder)):
        assert False, "Dimorder must be a list or permutation of range(tensor.ndims)"

    # Set up optdims if not specified
    if optdims is None:
        optdims = np.arange(N)
    else:
        optdims = parse_one_d(optdims)

    # Error checking
    assert rank > 0, "Number of components requested must be positive"

    # Set up and error checking on initial guess
    if isinstance(init, ttb.ktensor):
        # User provided an initial ktensor; validate it
        assert init.ndims == N, f"Initial guess does not have {N} modes"
        assert init.ncomponents == rank, (
            f"Initial guess does not have {rank} components"
        )
        for n in dimorder:
            if init.factor_matrices[n].shape != (input_tensor.shape[n], rank):
                assert False, f"Mode {n} of the initial guess is the wrong size"
    elif isinstance(init, str) and init.lower() == "random":
        factor_matrices = []
        for n in range(N):
            factor_matrices.append(
                np.random.uniform(0, 1, (input_tensor.shape[n], rank))
            )
        init = ttb.ktensor(factor_matrices)
    elif isinstance(init, str) and init.lower() == "nvecs":
        assert not isinstance(input_tensor, ttb.sumtensor), (
            "Sumtensor doesn't support nvecs"
        )
        factor_matrices = []
        for n in range(N):
            factor_matrices.append(input_tensor.nvecs(n, rank))
        init = ttb.ktensor(factor_matrices)
    else:
        assert False, "The selected initialization method is not supported"

    # Set up for iterates and fit
    U = init.copy().factor_matrices
    fit: float | np.ndarray = 0.0
    normresidual: float | np.ndarray

    # Reduce dimorder to only those modes we will optimize
    dimorder_in = dimorder  # save for output
    dimorder = [int(d) for d in dimorder if d in optdims]

    # Store the last MTTKRP result to accelerate fitness computation
    U_mttkrp = np.zeros((input_tensor.shape[dimorder[-1]], rank))

    if printitn > 0:
        print("CP_ALS:")

    # Main Loop: Iterate until convergence

    UtU = np.zeros((rank, rank, N))
    for n in range(N):
        UtU[:, :, n] = U[n].T @ U[n]

    for iteration in range(maxiters):
        fitold = fit

        # Iterate over optimized modes in specified order
        for n in dimorder:
            # Calculate Unew = X_(n) * khatrirao(all U except n, 'r').
            Unew = input_tensor.mttkrp(U, n)

            # Save the last MTTKRP result for fitness check.
            if n == dimorder[-1]:
                U_mttkrp = Unew

            # Compute the matrix of coefficients for linear system
            Y = np.prod(UtU, axis=2, where=[i != n for i in range(N)])
            # don't try to solve linear system with Y = 0
            if (Y == 0).all():
                Unew = np.zeros(Unew.shape)
            else:
                Unew = np.linalg.solve(Y.T, Unew.T).T
            # TODO: should we have issparse implemented? I am not sure
            #  when the following will occur
            # if issparse(Unew):
            #    Unew = full(Unew)   # for the case R=1

            # Normalize each vector to prevent singularities in coefmatrix
            if iteration == 0:
                weights = np.sqrt(sum(Unew**2, 0))  # 2-norm
            else:
                weights = np.maximum(np.max(np.abs(Unew), 0), 1)  # max-norm

            # if weights are 0, do not divide
            if not (weights == 0).all():
                Unew = Unew / weights

            U[n] = Unew
            UtU[:, :, n] = U[n].T @ U[n]

        M = ttb.ktensor(U, weights)

        # This is equivalent to innerprod(X,P).
        iprod = np.sum(
            np.sum(M.factor_matrices[dimorder[-1]] * U_mttkrp, 0) * weights, 0
        )
        if normX == 0:
            normresidual = M.norm() ** 2 - 2 * iprod
            fit = normresidual
        else:
            # the following input to np.sqrt can be negative due to rounding and
            # truncation errors, so np.abs is used
            normresidual = np.sqrt(np.abs(normX**2 + M.norm() ** 2 - 2 * iprod))
            fit = 1 - (normresidual / normX)  # fraction explained by model

        fitchange = np.abs(fitold - fit)

        # Check for convergence
        if (iteration > 0) and (fitchange < stoptol):
            flag = 0
        else:
            flag = 1

        if (printitn > 0) and ((divmod(iteration, printitn)[1] == 0) or (flag == 0)):
            print(f" Iter {iteration}: f = {fit:e} f-delta = {fitchange:7.1e}")

        # Check for convergence
        if flag == 0:
            break

    # Clean up final result

    # Arrange the final tensor so that the columns are normalized.
    M.arrange()
    # Fix the signs if requested
    if fixsigns:
        M = M.fixsigns()

    if printitn > 0:
        if normX == 0:
            normresidual = M.norm() ** 2 - 2 * input_tensor.innerprod(M)
            fit = normresidual
        else:
            normresidual = np.sqrt(
                np.abs(normX**2 + M.norm() ** 2 - 2 * input_tensor.innerprod(M))
            )
            fit = 1 - (normresidual / normX)  # fraction explained by model
        print(f" Final f = {fit:e}")

    output = {
        "params": {
            "stoptol": stoptol,
            "maxiters": maxiters,
            "dimorder": dimorder_in,
            "optdims": optdims,
            "printitn": printitn,
            "fixsigns": fixsigns,
        },
        "iters": iteration,
        "normresidual": normresidual,
        "fit": fit,
    }

    return M, init, output


if __name__ == "__main__":
    import doctest  # pragma: no cover

    doctest.testmod()  # pragma: no cover
