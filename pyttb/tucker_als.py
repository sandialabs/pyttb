from numbers import Real

import numpy as np

from pyttb.ttensor import ttensor


def tucker_als(
    tensor, rank, stoptol=1e-4, maxiters=1000, dimorder=None, init="random", printitn=1
):
    """
    Compute Tucker decomposition with alternating least squares

    Parameters
    ----------
    tensor: :class:`pyttb.tensor`
    rank: int, list[int]
        Rank of the decomposition(s)
    stoptol: float
        Tolerance used for termination - when the change in the fitness function in successive iterations drops
        below this value, the iterations terminate (default: 1e-4)
    dimorder: list
        Order to loop through dimensions (default: [range(tensor.ndims)])
    maxiters: int
        Maximum number of iterations (default: 1000)
    init: str or list[np.ndarray]
        Initial guess (default: "random")

             * "random": initialize using a :class:`pyttb.ttensor` with values chosen from a Normal distribution with mean 1 and standard deviation 0
             * "nvecs": initialize factor matrices of a :class:`pyttb.ttensor` using the eigenvectors of the outer product of the matricized input tensor
             * :class:`pyttb.ttensor`: initialize using a specific :class:`pyttb.ttensor` as input - must be the same shape as the input tensor and have the same rank as the input rank

    printitn: int
        Number of iterations to perform before printing iteration status - 0 for no status printing (default: 1)

    Returns
    -------
    M: :class:`pyttb.ttensor`
        Resulting ttensor from Tucker-ALS factorization
    Minit: :class:`pyttb.ttensor`
        Initial guess
    output: dict
        Information about the computation. Dictionary keys:

            * `params` : tuple of (stoptol, maxiters, printitn, dimorder)
            * `iters`: number of iterations performed
            * `normresidual`: norm of the difference between the input tensor and ktensor factorization
            * `fit`: value of the fitness function (fraction of tensor data explained by the model)

    """
    N = tensor.ndims
    normX = tensor.norm()

    # TODO: These argument checks look common with CP-ALS factor out
    if not isinstance(stoptol, Real):
        raise ValueError(
            f"stoptol must be a real valued scalar but received: {stoptol}"
        )
    if not isinstance(maxiters, Real) or maxiters < 0:
        raise ValueError(
            f"maxiters must be a non-negative real valued scalar but received: {maxiters}"
        )
    if not isinstance(printitn, Real):
        raise ValueError(
            f"printitn must be a real valued scalar but received: {printitn}"
        )

    if isinstance(rank, Real) or len(rank) == 1:
        rank = rank * np.ones(N, dtype=int)

    # Set up dimorder if not specified
    if not dimorder:
        dimorder = list(range(N))
    else:
        if not isinstance(dimorder, list):
            raise ValueError("Dimorder must be a list")
        elif tuple(range(N)) != tuple(sorted(dimorder)):
            raise ValueError(
                "Dimorder must be a list or permutation of range(tensor.ndims)"
            )

    if isinstance(init, list):
        Uinit = init
        if len(init) != N:
            raise ValueError(
                f"Init needs to be of length tensor.ndim (which was {N}) but only got length {len(init)}."
            )
        for n in dimorder[1::]:
            correct_shape = (tensor.shape[n], rank[n])
            if Uinit[n].shape != correct_shape:
                raise ValueError(
                    f"Init factor {n} had incorrect shape. Expected {correct_shape} but got {Uinit[n].shape}"
                )
    elif isinstance(init, str) and init.lower() == "random":
        Uinit = [None] * N
        # Observe that we don't need to calculate an initial guess for the
        # first index in dimorder because that will be solved for in the first
        # inner iteration.
        for n in range(1, N):
            Uinit[n] = np.random.uniform(0, 1, (tensor.shape[n], rank[n]))
    elif isinstance(init, str) and init.lower() in ("nvecs", "eigs"):
        # Compute an orthonormal basis for the dominant
        # Rn-dimensional left singular subspace of
        # X_(n) (0 <= n < N).
        Uinit = [None] * N
        for n in dimorder[1::]:
            print(f" Computing {rank[n]} leading e-vector for factor {n}.\n")
            Uinit[n] = tensor.nvecs(n, rank[n])
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
    for iter in range(maxiters):
        fitold = fit

        # Iterate over all N modes of the tensor
        for n in dimorder:
            # TODO proposal to change ttm to include_dims and exclude_dims to resolve -0 ambiguity
            dims = np.arange(0, tensor.ndims)
            dims = dims[dims != n]
            Utilde = tensor.ttm(U, dims, True)
            print(f"Utilde[{n}] = {Utilde}")
            # Maximize norm(Utilde x_n W') wrt W and
            # maintain orthonormality of W
            U[n] = Utilde.nvecs(n, rank[n])

        # Assemble the current approximation
        core = Utilde.ttm(U, n, True)

        # Compute fit
        normresidual = np.sqrt(abs(normX**2 - core.norm() ** 2))
        fit = 1 - (normresidual / normX)  # fraction explained by model
        fitchange = abs(fitold - fit)

        if iter % printitn == 0:
            print(f" Iter {iter}: fit = {fit:e} fitdelta = {fitchange:7.1e}\n")

        # Check for convergence
        if fitchange < stoptol:
            break

    solution = ttensor.from_data(core, U)

    output = {}
    output["params"] = (stoptol, maxiters, printitn, dimorder)
    output["iters"] = iter
    output["normresidual"] = normresidual
    output["fit"] = fit

    return solution, Uinit, output
