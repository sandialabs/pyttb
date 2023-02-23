# Copyright 2022 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

import numpy as np

import pyttb as ttb
from pyttb.pyttb_utils import *


def cp_als(
    tensor,
    rank,
    stoptol=1e-4,
    maxiters=1000,
    dimorder=None,
    init="random",
    printitn=1,
    fixsigns=True,
):
    """
    Compute CP decomposition with alternating least squares

    Parameters
    ----------
    tensor: :class:`pyttb.tensor` or :class:`pyttb.sptensor` or :class:`pyttb.ktensor`
    rank: int
        Rank of the decomposition
    stoptol: float
        Tolerance used for termination - when the change in the fitness function in successive iterations drops
        below this value, the iterations terminate (default: 1e-4)
    dimorder: list
        Order to loop through dimensions (default: [range(tensor.ndims)])
    maxiters: int
        Maximum number of iterations (default: 1000)
    init: str or :class:`pyttb.ktensor`
        Initial guess (default: "random")

             * "random": initialize using a :class:`pyttb.ktensor` with values chosen from a Normal distribution with mean 1 and standard deviation 0
             * "nvecs": initialize factor matrices of a :class:`pyttb.ktensor` using the eigenvectors of the outer product of the matricized input tensor
             * :class:`pyttb.ktensor`: initialize using a specific :class:`pyttb.ktensor` as input - must be the same shape as the input tensor and have the same rank as the input rank

    printitn: int
        Number of iterations to perform before printing iteration status - 0 for no status printing (default: 1)
    fixsigns: bool
        Align the signs of the columns of the factorization to align with the input tensor data (default: True)

    Returns
    -------
    M: :class:`pyttb.ktensor`
        Resulting ktensor from CP-ALS factorization
    Minit: :class:`pyttb.ktensor`
        Initial guess
    output: dict
        Information about the computation. Dictionary keys:

            * `params` : tuple of (stoptol, maxiters, printitn, dimorder)
            * `iters`: number of iterations performed
            * `normresidual`: norm of the difference between the input tensor and ktensor factorization
            * `fit`: value of the fitness function (fraction of tensor data explained by the model)

    Example
    -------
    Random initialization causes slight pertubation in intermediate results.
    `...` is our place holder for these numeric values.
    Example using default values ("random" initialization):

    >>> weights = np.array([1., 2.])
    >>> fm0 = np.array([[1., 2.], [3., 4.]])
    >>> fm1 = np.array([[5., 6.], [7., 8.]])
    >>> K = ttb.ktensor.from_data(weights, [fm0, fm1])
    >>> np.random.seed(1)
    >>> M, Minit, output = ttb.cp_als(K.full(), 2) # doctest: +ELLIPSIS
    CP_ALS:
     Iter 0: f = ... f-delta = ...
     Iter 1: f = ... f-delta = ...
     Final f = ...
    >>> print(M) # doctest: +ELLIPSIS
    ktensor of shape 2 x 2
    weights=[108.4715... 8.6114...]
    factor_matrices[0] =
    [[0.4187... 0.3989...]
     [0.9080... 0.9169...]]
    factor_matrices[1] =
    [[0.6188... 0.2581...]
     [0.7854... 0.9661...]]
    >>> print(Minit) # doctest: +ELLIPSIS
    ktensor of shape 2 x 2
    weights=[1. 1.]
    factor_matrices[0] =
    [[4.1702...e-01 7.2032...e-01]
     [1.1437...e-04 3.0233...e-01]]
    factor_matrices[1] =
    [[0.1467... 0.0923...]
     [0.1862... 0.3455...]]
    >>> print(output)
    {'params': (0.0001, 1000, 1, [0, 1]), 'iters': 1, 'normresidual': ..., 'fit': ...}

    Example using "nvecs" initialization:

    >>> M, Minit, output = ttb.cp_als(K.full(), 2, init="nvecs") # doctest: +ELLIPSIS
    CP_ALS:
     Iter 0: f = ... f-delta = ...
     Iter 1: f = ... f-delta = ...
     Final f = ...

    Example using :class:`pyttb.ktensor` initialization:

    >>> M, Minit, output = ttb.cp_als(K.full(), 2, init=K) # doctest: +ELLIPSIS
    CP_ALS:
     Iter 0: f = ... f-delta = ...
     Iter 1: f = ... f-delta = ...
     Final f = ...
    """

    # Extract number of dimensions and norm of tensor
    N = tensor.ndims
    normX = tensor.norm()

    # Set up dimorder if not specified
    if not dimorder:
        dimorder = list(range(N))
    else:
        if not isinstance(dimorder, list):
            assert False, "Dimorder must be a list"
        elif tuple(range(N)) != tuple(sorted(dimorder)):
            assert (
                False
            ), "Dimorder must be a list or permutation of range(tensor.ndims)"

    # Error checking
    assert rank > 0, "Number of components requested must be positive"

    # Set up and error checking on initial guess
    if isinstance(init, ttb.ktensor):
        # User provided an initial ktensor; validate it
        assert init.ndims == N, "Initial guess does not have {} modes".format(N)
        assert (
            init.ncomponents == rank
        ), "Initial guess does not have {} components".format(rank)
        for n in dimorder:
            if init.factor_matrices[n].shape != (tensor.shape[n], rank):
                assert False, "Mode {} of the initial guess is the wrong size".format(n)
    elif isinstance(init, str) and init.lower() == "random":
        factor_matrices = []
        for n in range(N):
            factor_matrices.append(np.random.uniform(0, 1, (tensor.shape[n], rank)))
        init = ttb.ktensor.from_factor_matrices(factor_matrices)
    elif isinstance(init, str) and init.lower() == "nvecs":
        factor_matrices = []
        for n in range(N):
            factor_matrices.append(tensor.nvecs(n, rank))
        init = ttb.ktensor.from_factor_matrices(factor_matrices)
    else:
        assert False, "The selected initialization method is not supported"

    # Set up for iterates and fit
    U = init.copy().factor_matrices
    fit = 0

    # Store the last MTTKRP result to accelerate fitness computation
    U_mttkrp = np.zeros((tensor.shape[dimorder[-1]], rank))

    if printitn > 0:
        print("CP_ALS:")

    # Main Loop: Iterate until convergence

    UtU = np.zeros((rank, rank, N))
    for n in range(N):
        UtU[:, :, n] = U[n].T @ U[n]

    for iter in range(maxiters):
        fitold = fit

        # Iterate over all N modes of the tensor
        for n in dimorder:
            # Calculate Unew = X_(n) * khatrirao(all U except n, 'r').
            Unew = tensor.mttkrp(U, n)

            # Save the last MTTKRP result for fitness check.
            if n == dimorder[-1]:
                U_mttkrp = Unew

            # Compute the matrix of coefficients for linear system
            Y = np.prod(UtU, axis=2, where=[i != n for i in range(N)])
            # don't try to solve linear system with Y = 0
            if (Y == np.zeros(Y.shape)).all():
                Unew = np.zeros(Unew.shape)
            else:
                Unew = np.linalg.solve(Y.T, Unew.T).T
            # TODO: should we have issparse implemented? I am not sure when the following will occur
            # if issparse(Unew):
            #    Unew = full(Unew)   # for the case R=1

            # Normalize each vector to prevent singularities in coefmatrix
            if iter == 0:
                weights = np.sqrt(sum(Unew**2, 0))  # 2-norm
            else:
                weights = np.maximum(np.max(np.abs(Unew), 0), 1)  # max-norm

            # if weights are 0, do not divide
            if not (weights == np.zeros(weights.shape)).all():
                Unew = Unew / weights

            U[n] = Unew
            UtU[:, :, n] = U[n].T @ U[n]

        M = ttb.ktensor.from_data(weights, U)

        # This is equivalent to innerprod(X,P).
        iprod = np.sum(
            np.sum(M.factor_matrices[dimorder[-1]] * U_mttkrp, 0) * weights, 0
        )
        if normX == 0:
            normresidual = M.norm() ** 2 - 2 * iprod
            fit = normresidual
        else:
            # the following input to np.sqrt can be negative due to rounding and truncation errors, so np.abs is used
            normresidual = np.sqrt(np.abs(normX**2 + M.norm() ** 2 - 2 * iprod))
            fit = 1 - (normresidual / normX)  # fraction explained by model

        fitchange = np.abs(fitold - fit)

        # Check for convergence
        if (iter > 0) and (fitchange < stoptol):
            flag = 0
        else:
            flag = 1

        if (divmod(iter, printitn)[1] == 0) or (printitn > 0 and flag == 0):
            print(f" Iter {iter}: f = {fit:e} f-delta = {fitchange:7.1e}")

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
            normresidual = M.norm() ** 2 - 2 * tensor.innerprod(M)
            fit = normresidual
        else:
            normresidual = np.sqrt(
                np.abs(normX**2 + M.norm() ** 2 - 2 * tensor.innerprod(M))
            )
            fit = 1 - (normresidual / normX)  # fraction explained by model
        print(f" Final f = {fit:e}")

    output = {}
    output["params"] = (stoptol, maxiters, printitn, dimorder)
    output["iters"] = iter
    output["normresidual"] = normresidual
    output["fit"] = fit

    return M, init, output


if __name__ == "__main__":
    import doctest  # pragma: no cover

    doctest.testmod()  # pragma: no cover
