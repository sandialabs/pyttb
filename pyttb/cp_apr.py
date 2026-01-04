"""Non-negative CP decomposition with alternating Poisson regression."""

# Copyright 2025 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from __future__ import annotations

import time
import warnings
from typing import Literal, overload

import numpy as np
from numpy_groupies import aggregate as accumarray

import pyttb as ttb


def cp_apr(  # noqa: PLR0913
    input_tensor: ttb.tensor | ttb.sptensor,
    rank: int,
    algorithm: Literal["mu"] | Literal["pdnr"] | Literal["pqnr"] = "mu",
    stoptol: float = 1e-4,
    stoptime: float = 1e6,
    maxiters: int = 1000,
    init: ttb.ktensor | Literal["random"] = "random",
    maxinneriters: int = 10,
    epsDivZero: float = 1e-10,
    printitn: int = 1,
    printinneritn: int = 0,
    kappa: float = 0.01,
    kappatol: float = 1e-10,
    epsActive: float = 1e-8,
    mu0: float = 1e-5,
    precompinds: bool = True,
    inexact: bool = True,
    lbfgsMem: int = 3,
) -> tuple[ttb.ktensor, ttb.ktensor, dict]:
    """
    Compute non-negative CP with alternating Poisson regression.

    Parameters
    ----------
    input_tensor: :class:`pyttb.tensor` or :class:`pyttb.sptensor`
    rank: int
        Rank of the decomposition
    algorithm: str
        in {'mu', 'pdnr, 'pqnr'}
    stoptol: float
        Tolerance on overall KKT violation
    stoptime: float
        Maximum number of seconds to run
    maxiters: int
        Maximum number of iterations
    init: str or :class:`pyttb.ktensor`
        Initial guess
    maxinneriters: int
        Maximum inner iterations per outer iteration
    epsDivZero: float
        Safeguard against divide by zero
    printitn: int
        Print every n outer iterations, 0 for none
    printinneritn: int
        Print every n inner iterations
    kappa: int
        MU ALGORITHM PARAMETER: Offset to fix complementary slackness
    kappatol:
        MU ALGORITHM PARAMETER: Tolerance on complementary slackness
    epsActive: float
        PDNR & PQNR ALGORITHM PARAMETER: Bertsekas tolerance for active set
    mu0: float
        PDNR ALGORITHM PARAMETER: Initial Damping Parameter
    precompinds: bool
        PDNR & PQNR ALGORITHM PARAMETER: Precompute sparse tensor indices
    inexact: bool
        PDNR ALGORITHM PARAMETER: Compute inexact Newton steps
    lbfgsMem: int
        PQNR ALGORITHM PARAMETER: Precompute sparse tensor indices

    Returns
    -------
    M: :class:`pyttb.ktensor`
        Resulting ktensor from CP APR
    Minit: :class:`pyttb.ktensor`
        Initial Guess
    output: dict
        Additional output #TODO document this more appropriately

    """
    # Extract the number of modes in tensor X
    N = input_tensor.ndims

    assert rank > 0, "Number of components requested must be positive"

    # Check that the data is non-negative.
    tmp = input_tensor < 0.0
    assert tmp.nnz == 0, (
        "Data tensor must be nonnegative for Poisson-based factorization"
    )

    # Set up an initial guess for the factor matrices.
    if isinstance(init, ttb.ktensor):
        # User provided an initial ktensor; validate it
        assert init.ndims == N, "Initial guess does not have the right number of modes"
        assert init.ncomponents == rank, (
            "Initial guess does not have the right number of components"
        )
        for n in range(N):
            if init.shape[n] != input_tensor.shape[n]:
                assert False, f"Mode {n} of the initial guess is the wrong size"
            if np.min(init.factor_matrices[n]) < 0.0:
                assert False, f"Initial guess has negative element in mode {n}"
        if np.min(init.weights) < 0:
            assert False, "Initial guess has a negative ktensor weight"

    elif init.lower() == "random":
        factor_matrices = []
        for n in range(N):
            factor_matrices.append(
                np.random.uniform(0, 1, (input_tensor.shape[n], rank))
            )
        init = ttb.ktensor(factor_matrices)
    else:
        raise ValueError(
            f"Initial guess supports ktensor or `random` but received {init}"
        )

    # Call solver based on the couce of algorithm parameter, passing all the other
    # input parameters
    if algorithm.lower() == "mu":
        M, output = tt_cp_apr_mu(
            input_tensor,
            rank,
            init,
            stoptol,
            stoptime,
            maxiters,
            maxinneriters,
            epsDivZero,
            printitn,
            printinneritn,
            kappa,
            kappatol,
        )
        output["algorithm"] = "mu"
        output["params"]["algorithm"] = "mu"
    elif algorithm.lower() == "pdnr":
        M, output = tt_cp_apr_pdnr(
            input_tensor,
            rank,
            init,
            stoptol,
            stoptime,
            maxiters,
            maxinneriters,
            epsDivZero,
            printitn,
            printinneritn,
            epsActive,
            mu0,
            precompinds,
            inexact,
        )
        output["algorithm"] = "pdnr"
        output["params"]["algorithm"] = "pdnr"
    elif algorithm.lower() == "pqnr":
        M, output = tt_cp_apr_pqnr(
            input_tensor,
            rank,
            init,
            stoptol,
            stoptime,
            maxiters,
            maxinneriters,
            epsDivZero,
            printitn,
            printinneritn,
            epsActive,
            lbfgsMem,
            precompinds,
        )
        output["algorithm"] = "pqnr"
        output["params"]["algorithm"] = "pqnr"
    else:
        assert False, "{algorithm} is not a supported cp_als algorithm"

    return M, init, output


def tt_cp_apr_mu(  # noqa: PLR0912,PLR0913,PLR0915
    input_tensor: ttb.tensor | ttb.sptensor,
    rank: int,
    init: ttb.ktensor,
    stoptol: float,
    stoptime: float,
    maxiters: int,
    maxinneriters: int,
    epsDivZero: float,
    printitn: int,
    printinneritn: int,
    kappa: float,
    kappatol: float,
) -> tuple[ttb.ktensor, dict]:
    """
    Compute nonnegative CP with alternating Poisson regression.

    Parameters
    ----------
    input_tensor: :class:`pyttb.tensor` or :class:`pyttb.sptensor`
    rank: int
        Rank of the decomposition
    init: :class:`pyttb.ktensor`
        Initial guess
    stoptol: float
        Tolerance on overall KKT violation
    stoptime: float
        Maximum number of seconds to run
    maxiters: int
        Maximum number of iterations
    maxinneriters: int
        Maximum inner iterations per outer iteration
    epsDivZero: float
        Safeguard against divide by zero
    printitn: int
        Print every n outer iterations, 0 for none
    printinneritn: int
        Print every n inner iterations
    kappa: int
        MU ALGORITHM PARAMETER: Offset to fix complementary slackness
    kappatol:
        MU ALGORITHM PARAMETER: Tolerance on complementary slackness

    Notes
    -----
    REFERENCE: E. C. Chi and T. G. Kolda. On Tensors, Sparsity, and
    Nonnegative Factorizations, arXiv:1112.2414 [math.NA], December 2011,
    URL: http://arxiv.org/abs/1112.2414. Submitted for publication.

    """
    N = input_tensor.ndims

    # TODO I vote no duplicate error checking, copy error checking from cp_apr for
    #  initial guess here if disagree

    # Initialize output arrays
    # fnEvals = np.zeros((maxiters,))
    kktViolations = -np.ones((maxiters,))
    # TODO we initialize nInnerIters of size max outer iters?
    nInnerIters = np.zeros((maxiters,))
    # nzeros = np.zeros((maxiters,))
    nViolations = np.zeros((maxiters,))
    nTimes = np.zeros((maxiters,))

    # Set up for iteration - initializing M and Phi.
    M = init.copy()
    M.normalize(normtype=1)
    Phi = []  # np.zeros((N,))#cell(N,1)
    for n in range(N):
        # TODO prepopulation Phi instead of append should be faster
        Phi.append(np.zeros(M.factor_matrices[n].shape))
    kktModeViolations = np.zeros((N,))

    if printitn > 0:
        print("CP_APR:")

    # Start the wall clock timer.
    start = time.time()

    # PDN-R and PQN-R benefit from precomputing sparse indices of X for each mode
    # subproblem. However, MU execution time barely changes, so the precomputer option
    # is not offered.

    # Main loop: Iterate until convergence
    for iteration in range(maxiters):
        isConverged = True
        for n in range(N):
            # Make adjustments to entries of M[n] that are violating complementary
            # slackness conditions.
            # TODO both these zeros were 1 in matlab
            if iteration > 0:
                V = (Phi[n] > 0) & (M.factor_matrices[n] < kappatol)
                if np.any(V):
                    nViolations[iteration] += 1
                    M.factor_matrices[n][V > 0] += kappa

            # Shift the weight from lambda to mode n
            M.redistribute(mode=n)

            # Calculate product of all matrices but the n-th
            # Sparse case only calculates entries corresponding to nonzeros in X
            Pi = calculate_pi(input_tensor, M, rank, n, N)

            # Do the multiplicative updates
            for i in range(maxinneriters):
                # Count the inner iterations
                nInnerIters[iteration] += 1

                # Calculate matrix for multiplicative update
                Phi[n] = calculate_phi(input_tensor, M, rank, n, Pi, epsDivZero)

                # Check for convergence
                kktModeViolations[n] = np.max(
                    np.abs(
                        vectorize_for_mu(np.minimum(M.factor_matrices[n], 1 - Phi[n]))
                    )
                )
                if kktModeViolations[n] < stoptol:
                    break
                isConverged = False

                # Do the multiplicative update
                # TODO cannot update M[n] in this way
                M.factor_matrices[n] *= Phi[n]

                # Print status
                if (printinneritn > 0) and (divmod(i, printinneritn)[1] == 0):
                    print(
                        "\t\tMode = {n}, Inner Iter = {i}, "
                        f"KKT violation = {kktModeViolations[n]}"
                    )

            # Shift weight from mode n back to lambda
            M.normalize(normtype=1, mode=n)

        kktViolations[iteration] = np.max(kktModeViolations)
        if (printitn > 0) and (divmod(iteration, printitn)[1] == 0):
            print(
                f"\tIter {iteration}: Inner Its = {nInnerIters[iteration]} "
                f"KKT violation = {kktViolations[iteration]}, "
                f"nViolations = {nViolations[iteration]}"
            )

        nTimes[iteration] = time.time() - start

        # Check for convergence
        if isConverged:
            if printitn > 0:
                print("Exiting because all subproblems reached KKT tol.")
            break
        if nTimes[iteration] > stoptime:
            if printitn > 0:
                print("Exiting because time limit exceeded.")
            break

    t_stop = time.time() - start

    # Clean up final result
    M.normalize(sort=True, normtype=1)

    obj = tt_loglikelihood(input_tensor, M)

    if printitn > 0:
        normTensor = input_tensor.norm()
        normresidual = np.sqrt(
            normTensor**2 + M.norm() ** 2 - 2 * input_tensor.innerprod(M)
        )
        fit = 1 - (normresidual / normTensor)  # fraction explained by model
        print("===========================================")
        print(f" Final log-likelihood = {obj}")
        print(f" Final least squares fit = {fit}")
        print(f" Final KKT violation = {kktViolations[iteration]}")
        print(f" Total inner iterations = {sum(nInnerIters)}")
        print(f" Total execution time = {t_stop} secs")

    output = {
        "params": {
            "stoptol": stoptol,
            "stoptime": stoptime,
            "maxiters": maxiters,
            "maxinneriters": maxinneriters,
            "epsDivZero": epsDivZero,
            "printitn": printitn,
            "printinneritn": printinneritn,
            "kappa": kappa,
            "kappatol": kappatol,
        },
        "kktViolations": kktViolations[: iteration + 1],
        "nInnerIters": nInnerIters[: iteration + 1],
        "nViolations": nViolations[: iteration + 1],
        "nTotalIters": np.sum(nInnerIters),
        "times": nTimes[: iteration + 1],
        "totalTime": t_stop,
        "obj": obj,
    }

    return M, output


def tt_cp_apr_pdnr(  # noqa: PLR0912,PLR0913,PLR0915
    input_tensor: ttb.tensor | ttb.sptensor,
    rank: int,
    init: ttb.ktensor,
    stoptol: float,
    stoptime: float,
    maxiters: int,
    maxinneriters: int,
    epsDivZero: float,
    printitn: int,
    printinneritn: int,
    epsActive: float,
    mu0: float,
    precompinds: bool,
    inexact: bool,
) -> tuple[ttb.ktensor, dict]:
    """Compute nonnegative CP with alternating Poisson regression.

    Computes an estimate of the best rank-R
    CP model of a tensor X using an alternating Poisson regression.
    The algorithm solves "row subproblems" in each alternating subproblem,
    using a Hessian of size R^2.

    Parameters
    ----------
    input_tensor: Union[:class:`pyttb.tensor`,:class:`pyttb.sptensor`]
    rank: int
        Rank of the decomposition
    init: str or :class:`pyttb.ktensor`
        Initial guess
    stoptol: float
        Tolerance on overall KKT violation
    stoptime: float
        Maximum number of seconds to run
    maxiters: int
        Maximum number of iterations
    maxinneriters: int
        Maximum inner iterations per outer iteration
    epsDivZero: float
        Safeguard against divide by zero
    printitn: int
        Print every n outer iterations, 0 for none
    printinneritn: int
        Print every n inner iterations
    epsActive: float
        PDNR & PQNR ALGORITHM PARAMETER: Bertsekas tolerance for active set
    mu0: float
        PDNR ALGORITHM PARAMETER: Initial Damping Parameter
    precompinds: bool
        PDNR & PQNR ALGORITHM PARAMETER: Precompute sparse tensor indices
    inexact: bool
        PDNR ALGORITHM PARAMETER: Compute inexact Newton steps

    Returns
    -------
    # TODO detail return dictionary

    Notes
    -----
    REFERENCE: Samantha Hansen, Todd Plantenga, Tamara G. Kolda.
    Newton-Based Optimization for Nonnegative Tensor Factorizations,
    arXiv:1304.4964 [math.NA], April 2013,
    URL: http://arxiv.org/abs/1304.4964. Submitted for publication.

    """
    # Extract the number of modes in tensor X
    N = input_tensor.ndims

    # If the initial guess has any rows of all zero elements, then modify so the row
    # subproblem is not taking log(0). Values will be restored to zero later if the
    # unfolded X for the row has no zeros.
    for n in range(N):
        rowsum = np.sum(init.factor_matrices[n], axis=1)
        tmpIdx = np.where(rowsum == 0)[0]
        if tmpIdx.size != 0:
            init.factor_matrices[n][tmpIdx, 0] = 1e-8

    # Start with the initial guess, normalized using the vector L1 norm
    M = init.copy()
    M.normalize(normtype=1)

    # Sparse tensor flag affects how Pi and Phi are computed.
    isSparse = isinstance(input_tensor, ttb.sptensor)

    # Initialize output arrays
    fnEvals = np.zeros((maxiters, 1))
    fnVals = np.zeros((maxiters, 1))
    kktViolations = -np.ones((maxiters, 1))
    nInnerIters = np.zeros((maxiters, 1))
    nzeros = np.zeros((maxiters, 1))
    times = np.zeros((maxiters, 1))

    if printitn > 0:
        print("CP_PDNR (alternating Poisson regression using damped Newton)")

    dispLineWarn = printinneritn > 0

    # Start the wall clock timer.
    start = time.time()

    if isinstance(input_tensor, ttb.sptensor) and isSparse and precompinds:
        # Precompute sparse index sets for all the row subproblems.
        # Takes more memory but can cut execution time significantly in some cases.
        if printitn > 0:
            print("\tPrecomuting sparse index sets...")
        sparseIx = []
        for n in range(N):
            num_rows = M.factor_matrices[n].shape[0]
            row_indices = []
            for jj in range(num_rows):
                row_indices.append(np.where(input_tensor.subs[:, n] == jj)[0])
            sparseIx.append(row_indices)

        if printitn > 0:
            print("done")

    e_vec = np.ones((1, rank))

    rowsubprobStopTol = stoptol

    # Main loop: iterate until convergence or a max threshold is reached
    for iteration in range(maxiters):
        isConverged = True
        kktModeViolations = np.zeros((N,))
        countInnerIters = np.zeros((N,))

        # Alternate thru each factor matrix, A_1, A_2, ..., A_N.
        for n in range(N):
            # Shift the weight from lambda to mode n.
            M.redistribute(mode=n)

            # calculate khatri-rao product of all matrices but the n-th
            if isinstance(input_tensor, ttb.tensor) and isSparse is False:
                # Data is not a sparse tensor.
                Pi = tt_calcpi_prowsubprob(input_tensor, M, rank, n, N, isSparse)
                X_mat = input_tensor.to_tenmat(
                    np.array([n], order=input_tensor.order), copy=False
                ).data

            num_rows = M.factor_matrices[n].shape[0]
            isRowNOTconverged = np.zeros((num_rows,))

            # Loop over the row subproblems in mode n.
            for jj in range(num_rows):
                # Initialize the damped Hessian parameter for the row subproblem.
                mu = mu0

                # Get data values for row jj of matricized mode n,
                if isinstance(input_tensor, ttb.sptensor) and isSparse:
                    # Data is a sparse tensor
                    if not precompinds:
                        sparse_indices = np.where(input_tensor.subs[:, n] == jj)[0]
                    else:
                        sparse_indices = sparseIx[n][jj]

                    if sparse_indices.size == 0:
                        # The row jj of matricized tensor X in mode n is empty
                        M.factor_matrices[n][jj, :] = 0
                        continue

                    x_row = input_tensor.vals[sparse_indices]

                    # Calculate just the columns of Pi needed for this row.
                    Pi = tt_calcpi_prowsubprob(
                        input_tensor, M, rank, n, N, isSparse, sparse_indices
                    )

                else:
                    x_row = X_mat[jj, :]

                # Get current values of the row subproblem variables.
                m_row = M.factor_matrices[n][jj, :]

                # Iteratively solve the row subproblem with projected Newton steps.
                if inexact and iteration == 1:
                    innerIterMaximum = 2
                else:
                    innerIterMaximum = maxinneriters

                f_new = 0.0
                for i in range(innerIterMaximum):
                    # Calculate the gradient.
                    [phi_row, ups_row] = calc_partials(
                        isSparse, Pi, epsDivZero, x_row, m_row
                    )
                    gradM = (e_vec - phi_row).transpose()

                    # Compute the row subproblem kkt_violation.

                    # Note experiments in the original paper used:
                    # kkt_violation = \
                    #    np.norm(np.abs(np.minimum(m_row, gradM.transpose())))

                    # We now use \| KKT \|_{inf}:
                    kkt_violation = np.max(
                        np.abs(np.minimum(m_row, gradM.transpose()[0]))
                    )

                    # Report largest row subproblem initial violation
                    if i == 0 and kkt_violation > kktModeViolations[n]:
                        kktModeViolations[n] = kkt_violation

                    if (printinneritn > 0) and (divmod(i, printinneritn) == 0):
                        print(
                            f"\tMode = {n}, Row = {jj}, InnerIt = {i}",
                            end="",
                        )

                        if i == 0:
                            print(", RowKKT = {kkt_violation}")
                        else:
                            print(f", RowKKT = {kkt_violation}, RowObj = {-f_new}")

                    # Check for row subproblem convergence.
                    if kkt_violation < stoptol:
                        break
                    # Not converged, so m_row will be modified.
                    isRowNOTconverged[jj] = 1

                    # Calculate the search direction
                    # TODO clean up reshaping gradM to row
                    search_dir, predicted_red = get_search_dir_pdnr(
                        Pi, ups_row, rank, gradM.transpose()[0], m_row, mu, epsActive
                    )

                    # Perform a projected linesearch and update variables.
                    # Start from a unit step length, decrease by 1/2,
                    # stop with sufficicent decrease of 1.0e-4 or at most 10 steps.
                    (
                        m_rowNew,
                        f_old,
                        f_unit,
                        f_new,
                        num_evals,
                    ) = tt_linesearch_prowsubprob(
                        search_dir.transpose()[0],
                        gradM.transpose(),
                        m_row,
                        1,
                        1 / 2,
                        10,
                        1.0e-4,
                        isSparse,
                        x_row,
                        Pi,
                        phi_row,
                        dispLineWarn,
                    )
                    fnEvals[iteration] += num_evals
                    m_row = m_rowNew

                    # Update damping parameter mu based on the unit step length, which
                    # is returned in f_unit
                    actual_red = f_old - f_unit
                    rho = actual_red / -predicted_red
                    if predicted_red == 0:
                        mu *= 10
                    elif rho < 1 / 4:
                        mu *= 7 / 2
                    elif rho > 3 / 4:
                        mu *= 2 / 7

                M.factor_matrices[n][jj, :] = m_row
                countInnerIters[n] += i

            # Test if all row subproblems have converged, which means that no variables
            # in this row were changed.
            if np.sum(isRowNOTconverged) != 0:
                isConverged = False

            # Shift weight from mode n back to lambda.
            M.normalize(mode=n, normtype=1)

            # Total number of inner iterations for a given outer iteration, totalled
            # across all modes and all row subproblems in each mode
            nInnerIters[iteration] += countInnerIters[n]

        # Save output items for the outer iteration.
        num_zero = np.intp(0)
        for n in range(N):
            num_zero += np.count_nonzero(M.factor_matrices[n] == 0)  # [0].size

        nzeros[iteration] = num_zero
        kktViolations[iteration] = np.max(kktModeViolations)

        if inexact:
            rowsubprobStopTol = np.maximum(stoptol, kktViolations[iteration]) / 100.0

            # Print outer iteration status.
            if (printitn > 0) and (divmod(iteration, printitn)[1] == 0):
                fnVals[iteration] = -tt_loglikelihood(input_tensor, M)
                print(
                    f"{iteration}. Ttl Inner Its: {nInnerIters[iteration]}, "
                    f"KKT viol = {kktViolations[iteration]}, obj = {fnVals[iteration]}"
                    f", nz: {num_zero}"
                )

        times[iteration] = time.time() - start

        # Check for convergence
        if isConverged and not inexact:
            break
        if isConverged and inexact and rowsubprobStopTol <= stoptol:
            break
        if times[iteration] > stoptime:
            print("EXiting because time limit exceeded")
            break

    t_stop = time.time() - start

    # Clean up final result
    M.normalize(sort=True, normtype=1)

    obj = tt_loglikelihood(input_tensor, M)

    if printitn > 0:
        normTensor = input_tensor.norm()
        normresidual = np.sqrt(
            normTensor**2 + M.norm() ** 2 - 2 * input_tensor.innerprod(M)
        )
        fit = 1 - (normresidual / normTensor)  # fraction explained by model
        print("===========================================")
        print(f" Final log-likelihood = {obj}")
        print(f" Final least squares fit = {fit}")
        print(f" Final KKT violation = {kktViolations[iteration]}")
        print(f" Total inner iterations = {sum(nInnerIters)}")
        print(f" Total execution time = {t_stop} secs")

    output = {
        "params": {
            "stoptol": stoptol,
            "stoptime": stoptime,
            "maxiters": maxiters,
            "maxinneriters": maxinneriters,
            "epsDivZero": epsDivZero,
            "printitn": printitn,
            "printinneritn": printinneritn,
            "epsActive": epsActive,
            "mu0": mu0,
            "precompinds": precompinds,
            "inexact": inexact,
        },
        "kktViolations": kktViolations[: iteration + 1],
        "obj": obj,
        "fnEvals": fnEvals[: iteration + 1],
        "fnVals": fnVals[: iteration + 1],
        "nInnerIters": nInnerIters[: iteration + 1],
        "nZeros": nzeros[: iteration + 1],
        "times": times[: iteration + 1],
        "totalTime": t_stop,
    }

    return M, output


def tt_cp_apr_pqnr(  # noqa: PLR0912,PLR0913,PLR0915
    input_tensor: ttb.tensor | ttb.sptensor,
    rank: int,
    init: ttb.ktensor,
    stoptol: float,
    stoptime: float,
    maxiters: int,
    maxinneriters: int,
    epsDivZero: float,
    printitn: int,
    printinneritn: int,
    epsActive: float,
    lbfgsMem: int,
    precompinds: bool,
) -> tuple[ttb.ktensor, dict]:
    """
    Compute nonnegative CP with alternating Poisson regression.

    tt_cp_apr_pdnr computes an estimate of the best rank-R
    CP model of a tensor X using an alternating Poisson regression.
    The algorithm solves "row subproblems" in each alternating subproblem,
    using a Hessian of size R^2.
    The function is typically called by cp_apr.

    The model is solved by nonlinear optimization, and the code literally
    minimizes the negative of log-likelihood.  However, printouts to the
    console reverse the sign to show maximization of log-likelihood.

    Parameters
    ----------
    input_tensor:
        Tensor to decompose
    rank: int
        Rank of the decomposition
    init:
        Initial guess
    stoptol:
        Tolerance on overall KKT violation
    stoptime:
        Maximum number of seconds to run
    maxiters:
        Maximum number of iterations
    maxinneriters:
        Maximum inner iterations per outer iteration
    epsDivZero:
        Safeguard against divide by zero
    printitn:
        Print every n outer iterations, 0 for none
    printinneritn:
        Print every n inner iterations
    epsActive:
        PDNR & PQNR ALGORITHM PARAMETER: Bertsekas tolerance for active set
    lbfgsMem:
        Number of vector pairs to store for L-BFGS
    precompinds:
        Precompute sparse tensor indices

    Returns
    -------
    # TODO detail return dictionary

    Notes
    -----
    REFERENCE: Samantha Hansen, Todd Plantenga, Tamara G. Kolda.
    Newton-Based Optimization for Nonnegative Tensor Factorizations,
    arXiv:1304.4964 [math.NA], April 2013,
    URL: http://arxiv.org/abs/1304.4964. Submitted for publication.

    """
    # TODO first ~100 lines are identical to PDNR, consider abstracting just the
    #  algorithm portion
    # Extract the number of modes in data tensor
    N = input_tensor.ndims

    # If the initial guess has any rows of all zero elements, then modify so the row
    # subproblem is not taking log(0). Values will be restored to zero later if the
    # unfolded X for the row has no zeros.
    for n in range(N):
        rowsum = np.sum(init.factor_matrices[n], axis=1)
        tmpIdx = np.where(rowsum == 0)[0]
        if tmpIdx.size != 0:
            init.factor_matrices[n][tmpIdx, 0] = 1e-8

    # Start with the initial guess, normalized using the vector L1 norm
    M = init.copy()
    M.normalize(normtype=1)

    # Sparse tensor flag affects how Pi and Phi are computed.
    isSparse = isinstance(input_tensor, ttb.sptensor)

    # Initialize output arrays
    fnEvals = np.zeros((maxiters, 1))
    fnVals = np.zeros((maxiters, 1))
    kktViolations = -np.ones((maxiters, 1))
    nInnerIters = np.zeros((maxiters, 1))
    nzeros = np.zeros((maxiters, 1))
    times = np.zeros((maxiters, 1))

    if printitn > 0:
        print("CP_PQNR (alternating Poisson regression using quasi-Newton)")

    dispLineWarn = printinneritn > 0

    # Start the wall clock timer.
    start = time.time()

    if isinstance(input_tensor, ttb.sptensor) and precompinds:
        # Precompute sparse index sets for all the row subproblems.
        # Takes more memory but can cut execution time significantly in some cases.
        if printitn > 0:
            print("\tPrecomuting sparse index sets...")
        sparseIx = []
        for n in range(N):
            num_rows = M.factor_matrices[n].shape[0]
            row_indices = []
            for jj in range(num_rows):
                row_indices.append(np.where(input_tensor.subs[:, n] == jj)[0])
            sparseIx.append(row_indices)

        if printitn > 0:
            print("done")

    # Main loop: iterate until convergence or a max threshold is reached
    for iteration in range(maxiters):
        isConverged = True
        kktModeViolations = np.zeros((N,))
        countInnerIters = np.zeros((N,))

        # Alternate thru each factor matrix, A_1, A_2, ..., A_N.
        for n in range(N):
            # Shift the weight from lambda to mode n.
            M.redistribute(mode=n)

            # calculate khatri-rao product of all matrices but the n-th
            if not isinstance(input_tensor, ttb.sptensor) and not isSparse:
                # Data is not a sparse tensor.
                Pi = tt_calcpi_prowsubprob(input_tensor, M, rank, n, N, isSparse)
                X_mat = input_tensor.to_tenmat(
                    np.array([n], order=input_tensor.order), copy=False
                ).data

            num_rows = M.factor_matrices[n].shape[0]
            isRowNOTconverged = np.zeros((num_rows,))

            # Loop over the row subproblems in mode n.
            for jj in range(num_rows):
                # Get data values for row jj of matricized mode n,
                if isinstance(input_tensor, ttb.sptensor) and isSparse:
                    # Data is a sparse tensor
                    if not precompinds:
                        sparse_indices = np.where(input_tensor.subs[:, n] == jj)[0]
                    else:
                        sparse_indices = sparseIx[n][jj]

                    if sparse_indices.size == 0:
                        # The row jj of matricized tensor X in mode n is empty
                        M.factor_matrices[n][jj, :] = 0
                        continue

                    x_row = input_tensor.vals[sparse_indices]

                    # Calculate just the columns of Pi needed for this row.
                    Pi = tt_calcpi_prowsubprob(
                        input_tensor, M, rank, n, N, isSparse, sparse_indices
                    )

                else:
                    x_row = X_mat[jj, :]

                # Get current values of the row subproblem variables.
                m_row = M.factor_matrices[n][jj, :]

                # Initialize L-BFGS storage for the row subproblem.
                delm = np.zeros((rank, lbfgsMem))
                delg = np.zeros((rank, lbfgsMem))
                rho = np.zeros((lbfgsMem,))
                lbfgsPos = 0
                m_rowOLD = np.empty((), dtype=m_row.dtype)
                gradOLD = np.empty((), dtype=m_row.dtype)

                # Iteratively solve the row subproblem with projected quasi-Newton steps
                for i in range(maxinneriters):
                    # Calculate the gradient.
                    gradM, phi_row = calc_grad(isSparse, Pi, epsDivZero, x_row, m_row)
                    if i == 0:
                        # Note from MATLAB tensortoolbox
                        # Original cp_aprPQN_row code (and plb_row) does a gradient
                        # step to prime the L-BFGS approximation.  However, it means
                        # a row subproblem that already converged wastes time
                        # doing a gradient step before checking KKT conditions.
                        # TODO: fix in a future release.
                        m_rowOLD = m_row
                        gradOLD = gradM
                        m_row, _, _, f_new, num_evals = tt_linesearch_prowsubprob(
                            -gradM.transpose(),
                            gradM.transpose(),
                            m_rowOLD,
                            1,
                            1 / 2,
                            10,
                            1e-4,
                            isSparse,
                            x_row,
                            Pi,
                            phi_row,
                            dispLineWarn,
                        )
                        fnEvals[iteration] += num_evals
                        gradM, phi_row = calc_grad(
                            isSparse, Pi, epsDivZero, x_row, m_row
                        )

                    # Compute the row subproblem kkt_violation.

                    # Note experiments in the original paper used:
                    # kkt_violation = \
                    #    np.norm(np.abs(np.minimum(m_row, gradM.transpose())))

                    # We now use \| KKT \|_{inf}:
                    kkt_violation = np.max(np.abs(np.minimum(m_row, gradM)))

                    # Report largest row subproblem initial violation
                    if i == 0 and kkt_violation > kktModeViolations[n]:
                        kktModeViolations[n] = kkt_violation

                    if (printinneritn > 0) and (divmod(i, printinneritn) == 0):
                        print(
                            f"\tMode = {n}, Row = {jj}, InnerIt = {i}",
                            end="",
                        )

                        if i == 0:
                            print(f", RowKKT = {kkt_violation}")
                        else:
                            print(f", RowKKT = {kkt_violation}, RowObj = {-f_new}")

                    # Check for row subproblem convergence.
                    if kkt_violation < stoptol:
                        break
                    # Not converged, so m_row will be modified.
                    isRowNOTconverged[jj] = 1

                    # Update the L-BFGS approximation.
                    tmp_delm: np.ndarray = m_row - m_rowOLD
                    tmp_delg: np.ndarray = gradM - gradOLD
                    tmp_delm_dot = tmp_delm.dot(tmp_delg.transpose())
                    if not np.any(tmp_delm_dot == 0):
                        tmp_rho = 1 / tmp_delm_dot
                        delm[:, lbfgsPos] = tmp_delm
                        delg[:, lbfgsPos] = tmp_delg
                        rho[lbfgsPos] = tmp_rho
                    else:
                        # Rho is required to be positive; if not, then skip the L-BFGS
                        # update pair. The recommended safeguard for full BFGS is
                        # Powell damping, but not clear how to damp in 2-loop L-BFGS
                        if dispLineWarn:
                            warnings.warn(
                                "WARNING: skipping L-BFGS update, rho would be "
                                f"1 / {tmp_delm * tmp_delg}"
                            )
                        # Roll back lbfgsPos since it will increment later.
                        if lbfgsPos == 0:
                            if rho[lbfgsMem - 1] > 0:
                                lbfgsPos = lbfgsMem - 1
                            else:
                                # Fatal error, should not happen.
                                assert False, "ERROR: L-BFGS first iterate is bad"

                        else:
                            lbfgsPos -= 1

                    # Calculate search direction
                    search_dir = get_search_dir_pqnr(
                        m_row,
                        gradM,
                        epsActive,
                        delm,
                        delg,
                        rho,
                        lbfgsPos,
                        i,
                        dispLineWarn,
                    )

                    lbfgsPos = np.mod(lbfgsPos, lbfgsMem)

                    m_rowOLD = m_row
                    gradOLD = gradM

                    # Perform a projected linesearch and update variables.
                    # Start from a unit step length, decrease by 1/2,
                    # stop with sufficicent decrease of 1.0e-4 or at most 10 steps.
                    m_row, _, _, f_new, num_evals = tt_linesearch_prowsubprob(
                        search_dir.transpose()[0],
                        gradOLD.transpose(),
                        m_rowOLD,
                        1,
                        1 / 2,
                        10,
                        1.0e-4,
                        isSparse,
                        x_row,
                        Pi,
                        phi_row,
                        dispLineWarn,
                    )
                    fnEvals[iteration] += num_evals

                M.factor_matrices[n][jj, :] = m_row
                countInnerIters[n] += i

            # Test if all row subproblems have converged, which means that no variables
            # in this row were changed.
            if np.sum(isRowNOTconverged) != 0:
                isConverged = False

            # Shift weight from mode n back to lambda.
            M.normalize(mode=n, normtype=1)

            # Total number of inner iterations for a given outer iteration,
            # totalled across all modes and all row subproblems in each mode
            nInnerIters[iteration] += countInnerIters[n]

        # Save output items for the outer iteration.
        num_zero = np.intp(0)
        for n in range(N):
            num_zero += np.count_nonzero(M.factor_matrices[n] == 0)  # [0].size

        nzeros[iteration] = num_zero
        kktViolations[iteration] = np.max(kktModeViolations)

        # Print outer iteration status.
        if (printitn > 0) and (divmod(iteration, printitn)[1] == 0):
            fnVals[iteration] = -tt_loglikelihood(input_tensor, M)
            print(
                f"{iteration}. Ttl Inner Its: {nInnerIters[iteration]}, KKT viol = "
                f"{kktViolations[iteration]}, obj = {fnVals[iteration]}, nz: {num_zero}"
            )

        times[iteration] = time.time() - start

        # Check for convergence
        if isConverged:
            break
        if times[iteration] > stoptime:
            print("Exiting because time limit exceeded")
            break

    t_stop = time.time() - start

    # Clean up final result
    M.normalize(sort=True, normtype=1)

    obj = tt_loglikelihood(input_tensor, M)

    if printitn > 0:
        normTensor = input_tensor.norm()
        normresidual = np.sqrt(
            normTensor**2 + M.norm() ** 2 - 2 * input_tensor.innerprod(M)
        )
        fit = 1 - (normresidual / normTensor)  # fraction explained by model
        print("===========================================")
        print(f" Final log-likelihood = {obj}")
        print(f" Final least squares fit = {fit}")
        print(f" Final KKT violation = {kktViolations[iteration]}")
        print(f" Total inner iterations = {sum(nInnerIters)}")
        print(f" Total execution time = {t_stop} secs")

    output = {
        "params": {
            "stoptol": stoptol,
            "stoptime": stoptime,
            "maxiters": maxiters,
            "maxinneriters": maxinneriters,
            "epsDivZero": epsDivZero,
            "printitn": printitn,
            "printinneritn": printinneritn,
            "epsActive": epsActive,
            "lbfgsMem": lbfgsMem,
            "precompinds": precompinds,
        },
        "kktViolations": kktViolations[: iteration + 1],
        "obj": obj,
        "fnEvals": fnEvals[: iteration + 1],
        "fnVals": fnVals[: iteration + 1],
        "nInnerIters": nInnerIters[: iteration + 1],
        "nZeros": nzeros[: iteration + 1],
        "times": times[: iteration + 1],
        "totalTime": t_stop,
    }

    return M, output


# PDNR helper functions


@overload
def tt_calcpi_prowsubprob(
    Data: ttb.sptensor,
    Model: ttb.ktensor,
    rank: int,
    factorIndex: int,
    ndims: int,
    isSparse: Literal[True],
    sparse_indices: np.ndarray,
) -> np.ndarray: ...  # pragma: no cover see coveragepy/issues/970


@overload
def tt_calcpi_prowsubprob(
    Data: ttb.tensor,
    Model: ttb.ktensor,
    rank: int,
    factorIndex: int,
    ndims: int,
    isSparse: Literal[False],
) -> np.ndarray: ...  # pragma: no cover see coveragepy/issues/970


def tt_calcpi_prowsubprob(  # noqa: PLR0913
    Data: ttb.sptensor | ttb.tensor,
    Model: ttb.ktensor,
    rank: int,
    factorIndex: int,
    ndims: int,
    isSparse: bool = False,
    sparse_indices: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute Pi for a row subproblem.

    Parameters
    ----------
    Data:
        Tensor to compute subproblem for.
    isSparse:
        Flag to determine if sparse subproblem.
    Model:
        Current decomposition.
    rank:
        Rank of solution.
    factorIndex:
        Which factor to solve
    ndims:
        Number of dimensions
    sparse_indices:
        Indices of row subproblem nonzero elements

    Returns
    -------
    Pi: :class:`numpy.ndarray`

    See Also
    --------
    :class:`pyttb.calculate_pi`

    """
    # TODO: this can probably be merged with general calculate pi,
    #  where default for sparse_indices is slice(None,None,None)
    #  alternatively, sparse_indices seems redundant with isSparse
    if isSparse and sparse_indices is not None and isinstance(Data, ttb.sptensor):
        # Data is a sparse tensor. Compute Pi for the row problem specified by
        # sparse_indices
        num_row_nnz = len(sparse_indices)

        Pi = np.ones((num_row_nnz, rank))
        for i in np.setdiff1d(np.arange(ndims), factorIndex).astype(int):
            Pi *= Model.factor_matrices[i][Data.subs[sparse_indices, i], :]
    else:
        Pi = ttb.khatrirao(
            *(
                Model.factor_matrices[:factorIndex]
                + Model.factor_matrices[factorIndex + 1 : ndims + 1]
            ),
            reverse=True,
        )

    return Pi


def calc_partials(
    isSparse: bool,
    Pi: np.ndarray,
    epsilon: float,
    data_row: np.ndarray,
    model_row: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Compute derivative quantities for a PDNR row subproblem.

    Parameters
    ----------
    isSparse:
        Flag if sparse subproblem.
    Pi:
    epsilon:
        Prevent division by zero.
    data_row:
        Row of data for subproblem.
    model_row:
        Row of model for subproblem.

    Returns
    -------
    phi_row: :class:`numpy.ndarray`
        gradient of row subproblem, except for a constant \n
        :math:`phi\_row[r] = \sum_{j=1}^{J_n}\frac{x_j\pi_{rj}}{\sum_i^R b_i\pi_{ij}}`
    ups_row: :class:`numpy.ndarray`
        intermediate quantity (upsilon) used for second derivatives  \n
        :math:`ups\_row[j] = \frac{x_j}{\left(\sum_i^R b_i\pi_{ij}\right)^2}`

    """
    if isSparse:
        data_row = data_row.transpose()[0]
    v = model_row.dot(Pi.transpose())
    w = data_row.transpose() / np.maximum(v, epsilon)
    phi_row = w.dot(Pi)
    u = v**2
    ups_row = data_row.transpose() / np.maximum(u, epsilon)
    return phi_row, ups_row


def get_search_dir_pdnr(  # noqa: PLR0913
    Pi: np.ndarray,
    ups_row: np.ndarray,
    rank: int,
    gradModel: np.ndarray,
    model_row: np.ndarray,
    mu: float,
    epsActSet: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the search direction using a two-metric projection with damped Hessian.

    Parameters
    ----------
    Pi:
    ups_row:
        intermediate quantity (upsilon) used for second derivatives
    rank:
        number of variables for the row subproblem
    gradModel:
        gradient vector for the row subproblem
    model_row:
        vector of variables for the row subproblem
    mu:
        damping parameter
    epsActSet:
        Bertsekas tolerance for active set determination

    Returns
    -------
    search_dir: :class:`numpy.ndarray`
        search direction vector
    pred_red: :class:`numpy.ndarray`
        predicted reduction in quadratic model
    """
    search_dir = np.zeros((rank, 1))
    projGradStep = (model_row - gradModel.transpose()) * (
        model_row - (gradModel.transpose() > 0).astype(float)
    )
    wk = np.linalg.norm(model_row - projGradStep)

    # Determine active and free variables
    num_free = 0
    free_indices_tmp = np.zeros((rank,)).astype(int)
    for r in range(rank):
        if (model_row[r] <= np.minimum(epsActSet, wk)) and (gradModel[r] > 0):
            # Variable is not free (belongs to set A or G)
            if model_row[r] != 0:
                # Variable moves according to the gradient (set G).
                search_dir[r] = -gradModel[r]

        else:
            # Variable is free (set F).
            num_free += 1
            free_indices_tmp[num_free - 1] = r

    free_indices = free_indices_tmp[0:num_free]

    # Compute the Hessian for free variables.
    Hessian_free = get_hessian(ups_row, Pi, free_indices)
    grad_free = -gradModel[free_indices]

    # Compute the damped Newton search direction over free variables
    # TODO verify this is appropriate representation of matlab's method,
    #  s.b. because hessian is square, and addition should ensure full rank,
    #  try.catch handles singular matrix
    try:
        search_dir[free_indices] = np.linalg.solve(
            Hessian_free + (mu * np.eye(num_free)), grad_free
        )[:, None]
    except np.linalg.LinAlgError:
        warnings.warn("CP_APR: Damped Hessian is nearly singular\n")
        # TODO: note this may be a typo in matlab see line 1107
        search_dir = -gradModel

    # Calculate expected reduction in the quadratic model of the objective.
    # TODO: double check if left or right multiplication has an speed effect,
    #  memory layout
    q = (
        search_dir[free_indices]
        .transpose()
        .dot(Hessian_free + (mu * np.eye(num_free)))
        .dot(search_dir[free_indices])
    )
    pred_red = (search_dir[free_indices].transpose().dot(gradModel[free_indices])) + (
        0.5 * q
    )
    if pred_red > 0:
        warnings.warn("CP_APR: Expected decrease in objective is positive\n")
        search_dir = -gradModel

    return search_dir, pred_red


def tt_linesearch_prowsubprob(  # noqa: PLR0913
    direction: np.ndarray,
    grad: np.ndarray,
    model_old: np.ndarray,
    step_len: float,
    step_red: float,
    max_steps: int,
    suff_decr: float,
    isSparse: bool,
    data_row: np.ndarray,
    Pi: np.ndarray,
    phi_row: np.ndarray,
    display_warning: bool,
) -> tuple[np.ndarray, float, float, float, int]:
    """Perform a line search on a row subproblem.

    Parameters
    ----------
    direction:
        search direction
    grad:
        gradient vector a model_old
    model_old:
        current variable values
    step_len:
        initial step length, which is the maximum possible step length
    step_red:
        step reduction factor (suggest 1/2)
    max_steps:
        maximum number of steps to try (suggest 10)
    suff_decr:
        sufficient decrease for convergence (suggest 1.0e-4)
    isSparse:
        sparsity flag for computing the objective
    data_row:
        row subproblem data, for computing the objective
    Pi:
        Pi matrix, for computing the objective
    phi_row:
        1-grad, more accurate if failing over to multiplicative update
    display_warning:
        Flag to display warning messages or not

    Returns
    -------
    m_new:  :class:`numpy.ndarray`
        new (improved) model values
    num_evals: int
        number of times objective was evaluated
    f_old: float
        objective value at model_old
    f_1: float
        objective value at model_old + step_len*direction
    f_new: float
        objective value at model_new
    """
    minDescentTol = 1.0e-7
    smallStepTol = 1.0e-7

    stepSize = step_len

    # Evaluate the current objective value
    f_old = -tt_loglikelihood_row(isSparse, data_row, model_old, Pi)
    num_evals = 1
    count = 1

    while count <= max_steps:
        # Compute a new step and project it onto the positive orthant.
        model_new = model_old + stepSize * direction
        model_new *= model_new > 0

        # Check that it is a descent direction.
        gDotd = np.sum(grad * (model_new - model_old))
        if (gDotd > 0) or (np.sum(model_new) < minDescentTol):
            # Don't evaluate the objective if not a descent direction
            # or if all the elements of model_new are close to zero
            f_new = np.inf
            if count == 1:
                f_1 = f_new

            stepSize *= step_red
            count += 1
        else:
            # Evaluate objective function at new iterate
            f_new = -tt_loglikelihood_row(isSparse, data_row, model_new, Pi)
            num_evals += 1
            if count == 1:
                f_1 = f_new

            # Check for sufficient decrease.
            if f_new <= (f_old + suff_decr * gDotd):
                break
            stepSize *= step_red
            count += 1

    if np.isinf(f_1):
        # Unit step failed; return a value that yields ared =0
        f_1 = f_old

    if ((count >= max_steps) and (f_new > f_old)) or (np.sum(model_new) < smallStepTol):
        # Fall back on a multiplicative update step (scaled steepest descent).
        # Experiments indicate it works better than a unit step in the direction
        # of steepest descent, which would be the following:
        # m_new = m_old - (step_len * grad);     # steepest descent
        # A simple update formula follows, but suffers from round-off error
        # when phi_row is tiny:
        # m_new = m_old - (m_old .* grad);
        # Use this for best accuracy:
        model_new = model_old * phi_row  # multiplicative update

        # Project to the constraints and reevaluate the subproblem objective
        model_new *= model_new > 0
        f_new = -tt_loglikelihood_row(isSparse, data_row, model_new, Pi)
        num_evals += 1

        # Let the caller know the search direction made no progress.
        f_1 = f_old

        if display_warning:
            warnings.warn(
                "CP_APR: Line search failed, using multiplicative update step"
            )

    return model_new, f_old, f_1, f_new, num_evals


def get_hessian(
    upsilon: np.ndarray, Pi: np.ndarray, free_indices: np.ndarray
) -> np.ndarray:
    """Return the Hessian for one PDNR row subproblem of Model[n].

    Only for just the rows and columns corresponding to the free variables.

    Parameters
    ----------
    upsilon:
        intermediate quantity (upsilon) used for second derivatives
    Pi:
    free_indices:

    Returns
    -------
    Hessian: :class:`numpy.ndarray`
        Sub-block of full Hessian identified by free-indices

    """
    num_free = len(free_indices)
    H = np.zeros((num_free, num_free))
    for i in range(num_free):
        for j in range(num_free):
            c = free_indices[i]
            d = free_indices[j]
            val = np.sum(upsilon.transpose() * Pi[:, c] * Pi[:, d])
            H[(i, j), (j, i)] = val
    return H


def tt_loglikelihood_row(
    isSparse: bool,
    data_row: np.ndarray,
    model_row: np.ndarray,
    Pi: np.ndarray,
) -> float:
    """Compute log-likelihood of one row subproblem.

    Parameters
    ----------
    isSparse:
        Sparsity flag
    data_row:
        vector of data values
    model_row:
        vector of model values
    Pi:

    Notes
    -----
    The row subproblem for a given mode includes one row of matricized tensor
    data (x) and one row of the model (m) in the same matricized mode.
    Then
    (dense case)
    m:  R-length vector
    x:  J-length vector
    Pi: R x J matrix
    (sparse case)
    m:  R-length vector
    x:  p-length vector, where p = nnz in row of matricized data tensor
    Pi: R x p matrix
    F = - (sum_r m_r - sum_j x_j * log (m * Pi_j)
    where Pi_j denotes the j^th column of Pi
    NOTE: Rows of Pi' must sum to one

    Returns
    -------
    loglikelihood: float
        See notes for description

    """
    term1 = -np.sum(model_row)
    if isSparse:
        term2 = np.sum(data_row.transpose() * np.log(model_row.dot(Pi.transpose())))
    else:
        b_pi = model_row.dot(Pi.transpose())
        skip_zeros = data_row != 0
        term2 = np.dot(data_row[skip_zeros], np.log(b_pi[skip_zeros])).item()
    loglikelihood = term1 + term2
    return loglikelihood


# PQNR helper functions
def get_search_dir_pqnr(  # noqa: PLR0913
    model_row: np.ndarray,
    gradModel: np.ndarray,
    epsActSet: float,
    delta_model: np.ndarray,
    delta_grad: np.ndarray,
    rho: np.ndarray,
    lbfgs_pos: int,
    iters: int,
    disp_warn: bool,
) -> np.ndarray:
    """
    Compute the search direction by projecting with L-BFGS.

    Parameters
    ----------
    model_row:
        current variable values
    gradModel:
        gradient at model_row
    epsActSet:
        Bertsekas tolerance for active set determination
    delta_model:
        L-BFGS array of variable deltas
    delta_grad:
        L-BFGS array of gradient deltas
    rho:
    lbfgs_pos:
        pointer into L-BFGS arrays
    iters:
    disp_warn:

    Returns
    -------
    direction: :class:`numpy.ndarray`
        Search direction based on current L-BFGS and grad

    Notes
    -----
    Adapted from MATLAB code of Dongmin Kim and Suvrit Sra written in 2008.
    Modified extensively to solve row subproblems and use a better linesearch;
    for details see REFERENCE: Samantha Hansen, Todd Plantenga, Tamara G. Kolda.
    Newton-Based Optimization for Nonnegative Tensor Factorizations,
    arXiv:1304.4964 [math.NA], April 2013,
    URL: http://arxiv.org/abs/1304.4964. Submitted for publication.

    """
    lbfgsSize = delta_model.shape[1]

    # Determine active and free variables.
    # TODO: is the below relevant?
    # If epsActSet is zero, then the following works:
    # fixedVars = find((m_row == 0) & (grad' > 0));
    # For the general case this works but is less clear and assumes m_row > 0:
    # fixedVars = find((grad' > 0) & (m_row <= min(epsActSet,grad')));
    projGradStep = (model_row - gradModel.transpose()) * (
        model_row - (gradModel.transpose() > 0).astype(float)
    )
    wk = np.linalg.norm(model_row - projGradStep)
    fixedVars = np.logical_and(gradModel > 0, (model_row <= np.minimum(epsActSet, wk)))

    direction = -gradModel
    direction[fixedVars] = 0

    if delta_model[:, lbfgs_pos].transpose().dot(delta_grad[:, lbfgs_pos]) == 0.0:
        # Cannot proceed with this L-BFGS data; most likely the iteration has
        # converged, so this is rarely seen.
        if disp_warn:
            warnings.warn("WARNING: L-BFGS update is orthogonal, using gradient")
        return direction

    alpha = np.ones((lbfgsSize,))
    k = lbfgs_pos

    # Perform an L-BFGS two-loop recursion to compute the search direction.
    for _ in range(np.minimum(iters, lbfgsSize)):
        alpha[k] = rho[k] * (delta_model[:, k].transpose().dot(direction))
        direction -= alpha[k] * (delta_grad[:, k])
        # TODO check mod
        k = (
            lbfgsSize - np.mod(1 - k, lbfgsSize) - 1
        )  # -1 accounts for numpy indexing starting at 0 not 1

    coef = (
        1
        / rho[lbfgs_pos]
        / delta_grad[:, lbfgs_pos].transpose().dot(delta_grad[:, lbfgs_pos])
    )
    direction *= coef

    for _ in range(np.minimum(iters, lbfgsSize)):
        k = np.mod(k, lbfgsSize)  # + 1
        b = rho[k] * (delta_grad[:, k].transpose().dot(direction))
        direction += (alpha[k] - b) * (delta_model[:, k])

    direction[fixedVars] = 0

    return direction


def calc_grad(
    isSparse: bool,
    Pi: np.ndarray,
    eps_div_zero: float,
    data_row: np.ndarray,
    model_row: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the gradient for a PQNR row subproblem.

    Parameters
    ----------
    isSparse:
    Pi:
    eps_div_zero:
    data_row:
    model_row:

    Returns
    -------
    phi_row:
    grad_row:

    """
    # TODO: note this is duplicated exactly from calc_partials, should
    #  combine to one function
    if isSparse:
        data_row = data_row.transpose()[0]
    v = model_row.dot(Pi.transpose())
    w = data_row / np.maximum(v, eps_div_zero)
    phi_row = w.dot(Pi)
    # print("V: {}, W :{}".format(v,w))
    # u = v**2
    # ups_row = data_row.transpose() / np.maximum(u, epsilon)
    grad_row = (np.ones(phi_row.shape) - phi_row).transpose()
    return grad_row, phi_row


# TODO verify what pi is
# Mu helper functions
def calculate_pi(
    Data: ttb.sptensor | ttb.tensor,
    Model: ttb.ktensor,
    rank: int,
    factorIndex: int,
    ndims: int,
) -> np.ndarray:
    """Calculate Pi matrix.

    Parameters
    ----------
    Data:
    Model:
    rank:
    factorIndex:
    ndims:

    Returns
    -------
    Pi:
    """
    if isinstance(Data, ttb.sptensor):
        Pi = np.ones((Data.nnz, rank))
        for i in np.setdiff1d(np.arange(ndims), factorIndex).astype(int):
            Pi *= Model.factor_matrices[i][Data.subs[:, i], :]
    else:
        Pi = ttb.khatrirao(
            *(
                Model.factor_matrices[:factorIndex]
                + Model.factor_matrices[factorIndex + 1 :]
            ),
            reverse=True,
        )

    return Pi


def calculate_phi(  # noqa: PLR0913
    Data: ttb.sptensor | ttb.tensor,
    Model: ttb.ktensor,
    rank: int,
    factorIndex: int,
    Pi: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Calculate Phi.

    Parameters
    ----------
    Data:
    Model:
    rank:
    factorIndex:
    Pi:
    epsilon:

    """
    if isinstance(Data, ttb.sptensor):
        Phi = -np.ones((Data.shape[factorIndex], rank))
        xsubs = Data.subs[:, factorIndex]
        v = np.sum(Model.factor_matrices[factorIndex][xsubs, :] * Pi, axis=1)
        wvals = Data.vals / np.maximum(v, epsilon)[:, None]
        for r in range(rank):
            Yr = accumarray(
                xsubs,
                np.squeeze(wvals * Pi[:, r][:, None]),
                size=Data.shape[factorIndex],
            )
            Phi[:, r] = Yr
    else:
        Xn = Data.to_tenmat(np.array([factorIndex], order=Data.order), copy=False).data
        V = Model.factor_matrices[factorIndex].dot(Pi.transpose())
        W = Xn / np.maximum(V, epsilon)
        Y = W.dot(Pi)
        Phi = Y

    return Phi


def tt_loglikelihood(Data: ttb.tensor | ttb.sptensor, Model: ttb.ktensor) -> float:
    """
    Compute log-likelihood of data with model.

    Parameters
    ----------
    Data:
    Model:

    Returns
    -------
    loglikelihood:
        - (sum_i m_i - x_i * log_i) with i as a multiindex across all tensor dimensions

    Notes
    -----
    We define for any x 0*log(x)=0, such that if our true data is 0 the loglikelihood
        is the value of the model.

    """
    N = Data.ndims

    assert isinstance(Model, ttb.ktensor), "Model must be a ktensor"

    Model.normalize(weight_factor=0, normtype=1)
    if isinstance(Data, ttb.sptensor):
        xsubs = Data.subs
        A = Model.factor_matrices[0][xsubs[:, 0], :]
        for n in range(1, N):
            A *= Model.factor_matrices[n][xsubs[:, n], :]
        return float(
            np.sum(Data.vals * np.log(np.sum(A, axis=1))[:, None])
            - np.sum(Model.factor_matrices[0])
        )
    dX = Data.to_tenmat(np.array([1], order=Data.order), copy=False).data
    dM = Model.to_tenmat(np.array([1], order=Model.order), copy=False).data
    f = 0
    for i in range(dX.shape[0]):
        for j in range(dX.shape[1]):
            if dX[i, j] == 0:
                pass
            else:
                f += dX[i, j] * np.log(dM[i, j])

    f -= np.sum(Model.factor_matrices[0])
    return float(f)


def vectorize_for_mu(matrix: np.ndarray) -> np.ndarray:
    """Unravel matrix into vector.

    Parameters
    ----------
    matrix:

    Returns
    -------
    matrix:
        Unraveled matrix len(matrix.shape)==1
    """
    return matrix.ravel()
