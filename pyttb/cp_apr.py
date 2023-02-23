# Copyright 2022 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

import time
import warnings

import numpy as np
from numpy_groupies import aggregate as accumarray

import pyttb as ttb

from .pyttb_utils import *


def cp_apr(
    tensor,
    rank,
    algorithm="mu",
    stoptol=1e-4,
    stoptime=1e6,
    maxiters=1000,
    init="random",
    maxinneriters=10,
    epsDivZero=1e-10,
    printitn=1,
    printinneritn=0,
    kappa=0.01,
    kappatol=1e-10,
    epsActive=1e-8,
    mu0=1e-5,
    precompinds=True,
    inexact=True,
    lbfgsMem=3,
):
    """
    Compute non-negative CP with alternating Poisson regression.

    Parameters
    ----------
    tensor: :class:`pyttb.tensor` or :class:`pyttb.sptensor`
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
    N = tensor.ndims

    assert rank > 0, "Number of components requested must be positive"

    # Check that the data is non-negative.
    tmp = tensor < 0.0
    assert (
        tmp.nnz == 0
    ), "Data tensor must be nonnegative for Poisson-based factorization"

    # Set up an initial guess for the factor matrices.
    if isinstance(init, ttb.ktensor):
        # User provided an initial ktensor; validate it
        assert init.ndims == N, "Initial guess does not have the right number of modes"
        assert (
            init.ncomponents == rank
        ), "Initial guess does not have the right number of componenets"
        for n in range(N):
            if init.shape[n] != tensor.shape[n]:
                assert False, "Mode {} of the initial guess is the wrong size".format(n)
            if np.min(init.factor_matrices[n]) < 0.0:
                assert False, "Initial guess has negative element in mode {}".format(n)
        if np.min(init.weights) < 0:
            assert False, "Initial guess has a negative ktensor weight"

    elif init.lower() == "random":
        factor_matrices = []
        for n in range(N):
            factor_matrices.append(np.random.uniform(0, 1, (tensor.shape[n], rank)))
        init = ttb.ktensor.from_factor_matrices(factor_matrices)

    # Call solver based on the couce of algorithm parameter, passing all the other input parameters
    if algorithm.lower() == "mu":
        M, output = tt_cp_apr_mu(
            tensor,
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
    elif algorithm.lower() == "pdnr":
        M, output = tt_cp_apr_pdnr(
            tensor,
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
    elif algorithm.lower() == "pqnr":
        M, output = tt_cp_apr_pqnr(
            tensor,
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
    else:
        assert False, "{} is not a supported cp_als algorithm".format(algorithm)

    return M, init, output


def tt_cp_apr_mu(
    tensor,
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
):
    """
    Compute nonnegative CP with alternating Poisson regression.

    Parameters
    ----------
    tensor: :class:`pyttb.tensor` or :class:`pyttb.sptensor`
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

    Returns
    -------

    Notes
    -----
    REFERENCE: E. C. Chi and T. G. Kolda. On Tensors, Sparsity, and
    Nonnegative Factorizations, arXiv:1112.2414 [math.NA], December 2011,
    URL: http://arxiv.org/abs/1112.2414. Submitted for publication.

    """
    N = tensor.ndims

    # TODO I vote no duplicate error checking, copy error checking from cp_apr for initial guess here if disagree

    # Initialize output arrays
    # fnEvals = np.zeros((maxiters,))
    kktViolations = -np.ones((maxiters,))
    # TODO we initialize nInnerIters of size max outer iters?
    nInnerIters = np.zeros((maxiters,))
    # nzeros = np.zeros((maxiters,))
    nViolations = np.zeros((maxiters,))
    nTimes = np.zeros((maxiters,))

    # Set up for iteration - initializing M and Phi.
    # TODO replace with copy
    M = ttb.ktensor.from_tensor_type(init)
    M.normalize(normtype=1)
    Phi = []  # np.zeros((N,))#cell(N,1)
    for n in range(N):
        # TODO prepopulation Phi instead of appen should be faster
        Phi.append(np.zeros(M[n].shape))
    kktModeViolations = np.zeros((N,))

    if printitn > 0:
        print("\nCP_APR:\n")

    # Start the wall clock timer.
    start = time.time()

    # PDN-R and PQN-R benefit from precomputing sparse indices of X for each mode subproblem.
    # However, MU execution time barely changes, so the precomputer option is not offered.

    # Main loop: Iterate until convergence
    for iter in range(maxiters):
        isConverged = True
        for n in range(N):
            # Make adjustments to entries of M[n] that are violating complementary slackness conditions.
            # TODO both these zeros were 1 in matlab
            if iter > 0:
                V = (Phi[n] > 0) & (M[n] < kappatol)
                if np.any(V):
                    nViolations[iter] += 1
                    M.factor_matrices[n][V > 0] += kappa

            # Shift the weight from lambda to mode n
            M.redistribute(mode=n)

            # Calculate product of all matrices but the n-th
            # Sparse case only calculates entries corresponding to nonzeros in X
            Pi = calculatePi(tensor, M, rank, n, N)

            # Do the multiplicative updates
            for i in range(maxinneriters):
                # Count the inner iterations
                nInnerIters[iter] += 1

                # Calculate matrix for multiplicative update
                Phi[n] = calculatePhi(tensor, M, rank, n, Pi, epsDivZero)

                # Check for convergence
                kktModeViolations[n] = np.max(
                    np.abs(vectorizeForMu(np.minimum(M.factor_matrices[n], 1 - Phi[n])))
                )
                if kktModeViolations[n] < stoptol:
                    break
                else:
                    isConverged = False

                # Do the multiplicative update
                # TODO cannot update M[n] in this way
                M.factor_matrices[n] *= Phi[n]

                # Print status
                if printinneritn != 0 and divmod(i, printinneritn)[1] == 0:
                    print(
                        "\t\tMode = {}, Inner Iter = {}, KKT violation = {}\n".format(
                            n, i, kktModeViolations[n]
                        )
                    )

            # Shift weight from mode n back to lambda
            M.normalize(normtype=1, mode=n)

        kktViolations[iter] = np.max(kktModeViolations)
        if divmod(iter, printitn)[1] == 0:
            print(
                "\tIter {}: Inner Its = {} KKT violation = {}, nViolations = {}".format(
                    iter, nInnerIters[iter], kktViolations[iter], nViolations[iter]
                )
            )

        nTimes[iter] = time.time() - start

        # Check for convergence
        if isConverged:
            if printitn > 0:
                print("Exiting because all subproblems reached KKT tol.\n")
            break
        if nTimes[iter] > stoptime:
            if printitn > 0:
                print("Exiting because time limit exceeded.\n")
            break

    t_stop = time.time() - start

    # Clean up final result
    M.normalize(sort=True, normtype=1)

    obj = tt_loglikelihood(tensor, M)

    if printitn > 0:
        normTensor = tensor.norm()
        normresidual = np.sqrt(
            normTensor**2 + M.norm() ** 2 - 2 * tensor.innerprod(M)
        )
        fit = 1 - (normresidual / normTensor)  # fraction explained by model
        print("===========================================\n")
        print(" Final log-likelihood = {} \n".format(obj))
        print(" Final least squares fit = {} \n".format(fit))
        print(" Final KKT violation = {}\n".format(kktViolations[iter]))
        print(" Total inner iterations = {}\n".format(sum(nInnerIters)))
        print(" Total execution time = {} secs\n".format(t_stop))

    output = {}
    output["params"] = (
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
    output["kktViolations"] = kktViolations[: iter + 1]
    output["nInnerIters"] = nInnerIters[: iter + 1]
    output["nViolations"] = nViolations[: iter + 1]
    output["nTotalIters"] = np.sum(nInnerIters)
    output["times"] = nTimes[: iter + 1]
    output["totalTime"] = t_stop
    output["obj"] = obj

    return M, output


def tt_cp_apr_pdnr(
    tensor,
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
):
    """
    Compute nonnegative CP with alternating Poisson regression
    computes an estimate of the best rank-R
    CP model of a tensor X using an alternating Poisson regression.
    The algorithm solves "row subproblems" in each alternating subproblem,
    using a Hessian of size R^2.

    Parameters
    ----------
    # TODO it looks like this method of define union helps the typ hinting better than or
    tensor: Union[:class:`pyttb.tensor`,:class:`pyttb.sptensor`]
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
    N = tensor.ndims

    # If the initial guess has any rows of all zero elements, then modify so the row subproblem is not taking log(0).
    # Values will be restored to zero later if the unfolded X for the row has no zeros.
    for n in range(N):
        rowsum = np.sum(init[n], axis=1)
        tmpIdx = np.where(rowsum == 0)[0]
        if tmpIdx.size != 0:
            init[n][tmpIdx, 0] = 1e-8

    # Start with the initial guess, normalized using the vector L1 norm
    # TODO replace with copy
    M = ttb.ktensor.from_tensor_type(init)
    M.normalize(normtype=1)

    # Sparse tensor flag affects how Pi and Phi are computed.
    if isinstance(tensor, ttb.sptensor):
        isSparse = True
    else:
        isSparse = False

    # Initialize output arrays
    fnEvals = np.zeros((maxiters, 1))
    fnVals = np.zeros((maxiters, 1))
    kktViolations = -np.ones((maxiters, 1))
    nInnerIters = np.zeros((maxiters, 1))
    nzeros = np.zeros((maxiters, 1))
    times = np.zeros((maxiters, 1))

    if printitn > 0:
        print("\nCP_PDNR (alternating Poisson regression using damped Newton)\n")

    dispLineWarn = printinneritn > 0

    # Start the wall clock timer.
    start = time.time()

    if isSparse and precompinds:
        # Precompute sparse index sets for all the row subproblems.
        # Takes more memory but can cut exectuion time significantly in some cases.
        if printitn > 0:
            print("\tPrecomuting sparse index sets...")
        sparseIx = []
        for n in range(N):
            num_rows = M[n].shape[0]
            row_indices = []
            for jj in range(num_rows):
                row_indices.append(np.where(tensor.subs[:, n] == jj)[0])
            sparseIx.append(row_indices)

        if printitn > 0:
            print("done\n")

    e_vec = np.ones((1, rank))

    rowsubprobStopTol = stoptol

    # Main loop: iterate until convergence or a max threshold is reached
    for iter in range(maxiters):
        isConverged = True
        kktModeViolations = np.zeros((N,))
        countInnerIters = np.zeros((N,))

        # Alternate thru each factor matrix, A_1, A_2, ..., A_N.
        for n in range(N):
            # Shift the weight from lambda to mode n.
            M.redistribute(mode=n)

            # calculate khatri-rao product of all matrices but the n-th
            if isSparse == False:
                # Data is not a sparse tensor.
                Pi = ttb.tt_calcpi_prowsubprob(tensor, M, rank, n, N, isSparse)
                X_mat = ttb.tt_to_dense_matrix(tensor, n)

            num_rows = M[n].shape[0]
            isRowNOTconverged = np.zeros((num_rows,))

            # Loop over the row subproblems in mode n.
            for jj in range(num_rows):
                # Initialize the damped Hessian parameter for the row subproblem.
                mu = mu0

                # Get data values for row jj of matricized mode n,
                if isSparse:
                    # Data is a sparse tensor
                    if not precompinds:
                        sparse_indices = np.where(tensor.subs[:, n] == jj)[0]
                    else:
                        sparse_indices = sparseIx[n][jj]

                    if sparse_indices.size == 0:
                        # The row jj of matricized tensor X in mode n is empty
                        M.factor_matrices[n][jj, :] = 0
                        continue

                    x_row = tensor.vals[sparse_indices]

                    # Calculate just the columns of Pi needed for this row.
                    Pi = ttb.tt_calcpi_prowsubprob(
                        tensor, M, rank, n, N, isSparse, sparse_indices
                    )

                else:
                    x_row = X_mat[jj, :]

                # Get current values of the row subproblem variables.
                m_row = M[n][jj, :]

                # Iteratively solve the row subproblem with projected Newton steps.
                if inexact and iter == 1:
                    innerIterMaximum = 2
                else:
                    innerIterMaximum = maxinneriters

                for i in range(innerIterMaximum):
                    # Calculate the gradient.
                    [phi_row, ups_row] = calc_partials(
                        isSparse, Pi, epsDivZero, x_row, m_row
                    )
                    gradM = (e_vec - phi_row).transpose()

                    # Compute the row subproblem kkt_violation.

                    # Note experiments in the original paper used:
                    # kkt_violation = np.norm(np.abs(np.minimum(m_row, gradM.transpose())))

                    # We now use \| KKT \|_{inf}:
                    kkt_violation = np.max(
                        np.abs(np.minimum(m_row, gradM.transpose()[0]))
                    )

                    # Report largest row subproblem initial violation
                    if i == 0 and kkt_violation > kktModeViolations[n]:
                        kktModeViolations[n] = kkt_violation

                    if printinneritn > 0 and np.mod(i, printinneritn) == 0:
                        print("\tMode = {}, Row = {}, InnerIt = {}".format(n, jj, i))

                        if i == 0:
                            print(", RowKKT = {}\n".format(kkt_violation))
                        else:
                            print(
                                ", RowKKT = {}, RowObj = {}\n".format(
                                    kkt_violation, -f_new
                                )
                            )

                    # Check for row subproblem convergence.
                    if kkt_violation < stoptol:
                        break
                    else:
                        # Not converged, so m_row will be modified.
                        isRowNOTconverged[jj] = 1

                    # Calculate the search direction
                    # TODO clean up reshaping gradM to row
                    search_dir, predicted_red = getSearchDirPdnr(
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
                    ) = ttb.tt_linesearch_prowsubprob(
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
                    fnEvals[iter] += num_evals
                    m_row = m_rowNew

                    # Update damping parameter mu based on the unit step length, which is returned in f_unit
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

            # Test if all row subproblems have converged, which means that no varibales in this more were changed.
            if np.sum(isRowNOTconverged) != 0:
                isConverged = False

            # Shift weight from mode n back to lambda.
            M.normalize(mode=n, normtype=1)

            # Total number of inner iterations for a given outer iteration, totalled across all modes and all
            # row subproblems in each mode
            nInnerIters[iter] += countInnerIters[n]

        # Save output items for the outer iteration.
        num_zero = 0
        for n in range(N):
            num_zero += np.count_nonzero(M[n] == 0)  # [0].size

        nzeros[iter] = num_zero
        kktViolations[iter] = np.max(kktModeViolations)

        if inexact:
            rowsubprobStopTol = np.maximum(stoptol, kktViolations[iter]) / 100.0

            # Print outer iteration status.
            if printitn > 0 and np.mod(iter, printitn) == 0:
                fnVals[iter] = -tt_loglikelihood(tensor, M)
                print(
                    "{}. Ttl Inner Its: {}, KKT viol = {}, obj = {}, nz: {}\n".format(
                        iter,
                        nInnerIters[iter],
                        kktViolations[iter],
                        fnVals[iter],
                        num_zero,
                    )
                )

        times[iter] = time.time() - start

        # Check for convergence
        if isConverged and inexact == False:
            break
        if isConverged and inexact and rowsubprobStopTol <= stoptol:
            break
        if times[iter] > stoptime:
            print("EXiting because time limit exceeded\n")
            break

    t_stop = time.time() - start

    # Clean up final result
    M.normalize(sort=True, normtype=1)

    obj = tt_loglikelihood(tensor, M)

    if printitn > 0:
        normTensor = tensor.norm()
        normresidual = np.sqrt(
            normTensor**2 + M.norm() ** 2 - 2 * tensor.innerprod(M)
        )
        fit = 1 - (normresidual / normTensor)  # fraction explained by model
        print("===========================================\n")
        print(" Final log-likelihood = {} \n".format(obj))
        print(" Final least squares fit = {} \n".format(fit))
        print(" Final KKT violation = {}\n".format(kktViolations[iter]))
        print(" Total inner iterations = {}\n".format(sum(nInnerIters)))
        print(" Total execution time = {} secs\n".format(t_stop))

    output = {}
    output["params"] = (
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
    output["kktViolations"] = kktViolations[: iter + 1]
    output["obj"] = obj
    output["fnEvals"] = fnEvals[: iter + 1]
    output["fnVals"] = fnVals[: iter + 1]
    output["nInnerIters"] = nInnerIters[: iter + 1]
    output["nZeros"] = nzeros[: iter + 1]
    output["times"] = times[: iter + 1]
    output["totalTime"] = t_stop

    return M, output


def tt_cp_apr_pqnr(
    tensor,
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
):
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


    mu0: float
        PDNR ALGORITHM PARAMETER: Initial Damping Parameter
    precompinds: bool
        PDNR & PQNR ALGORITHM PARAMETER: Precompute sparse tensor indices
    inexact: bool
        PDNR ALGORITHM PARAMETER: Compute inexact Newton steps

    Parameters
    ----------
    tensor: Union[:class:`pyttb.tensor`,:class:`pyttb.sptensor`]
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
    lbfgsMem: int
        Number of vector pairs to store for L-BFGS
    precompinds

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
    # TODO first ~100 lines are identical to PDNR, consider abstracting just the algorithm portion
    # Extract the number of modes in data tensor
    N = tensor.ndims

    # If the initial guess has any rows of all zero elements, then modify so the row subproblem is not taking log(0).
    # Values will be restored to zero later if the unfolded X for the row has no zeros.
    for n in range(N):
        rowsum = np.sum(init[n], axis=1)
        tmpIdx = np.where(rowsum == 0)[0]
        if tmpIdx.size != 0:
            init[n][tmpIdx, 0] = 1e-8

    # Start with the initial guess, normalized using the vector L1 norm
    # TODO replace with copy
    M = ttb.ktensor.from_tensor_type(init)
    M.normalize(normtype=1)

    # Sparse tensor flag affects how Pi and Phi are computed.
    if isinstance(tensor, ttb.sptensor):
        isSparse = True
    else:
        isSparse = False

    # Initialize output arrays
    fnEvals = np.zeros((maxiters, 1))
    fnVals = np.zeros((maxiters, 1))
    kktViolations = -np.ones((maxiters, 1))
    nInnerIters = np.zeros((maxiters, 1))
    nzeros = np.zeros((maxiters, 1))
    times = np.zeros((maxiters, 1))

    if printitn > 0:
        print("\nCP_PQNR (alternating Poisson regression using quasi-Newton)\n")

    dispLineWarn = printinneritn > 0

    # Start the wall clock timer.
    start = time.time()

    if isSparse and precompinds:
        # Precompute sparse index sets for all the row subproblems.
        # Takes more memory but can cut exectuion time significantly in some cases.
        if printitn > 0:
            print("\tPrecomuting sparse index sets...")
        sparseIx = []
        for n in range(N):
            num_rows = M[n].shape[0]
            row_indices = []
            for jj in range(num_rows):
                row_indices.append(np.where(tensor.subs[:, n] == jj)[0])
            sparseIx.append(row_indices)

        if printitn > 0:
            print("done\n")

    # Main loop: iterate until convergence or a max threshold is reached
    for iter in range(maxiters):
        isConverged = True
        kktModeViolations = np.zeros((N,))
        countInnerIters = np.zeros((N,))

        # Alternate thru each factor matrix, A_1, A_2, ..., A_N.
        for n in range(N):
            # Shift the weight from lambda to mode n.
            M.redistribute(mode=n)

            # calculate khatri-rao product of all matrices but the n-th
            if isSparse == False:
                # Data is not a sparse tensor.
                Pi = ttb.tt_calcpi_prowsubprob(tensor, M, rank, n, N, isSparse)
                X_mat = ttb.tt_to_dense_matrix(tensor, n)

            num_rows = M[n].shape[0]
            isRowNOTconverged = np.zeros((num_rows,))

            # Loop over the row subproblems in mode n.
            for jj in range(num_rows):
                # Get data values for row jj of matricized mode n,
                if isSparse:
                    # Data is a sparse tensor
                    if not precompinds:
                        sparse_indices = np.where(tensor.subs[:, n] == jj)[0]
                    else:
                        sparse_indices = sparseIx[n][jj]

                    if sparse_indices.size == 0:
                        # The row jj of matricized tensor X in mode n is empty
                        M.factor_matrices[n][jj, :] = 0
                        continue

                    x_row = tensor.vals[sparse_indices]

                    # Calculate just the columns of Pi needed for this row.
                    Pi = ttb.tt_calcpi_prowsubprob(
                        tensor, M, rank, n, N, isSparse, sparse_indices
                    )

                else:
                    x_row = X_mat[jj, :]

                # Get current values of the row subproblem variables.
                m_row = M[n][jj, :]

                # Initialize L-BFGS storage for the row subproblem.
                delm = np.zeros((rank, lbfgsMem))
                delg = np.zeros((rank, lbfgsMem))
                rho = np.zeros((lbfgsMem,))
                lbfgsPos = 0
                m_rowOLD = []
                gradOLD = []

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
                        m_row, f, f_unit, f_new, num_evals = tt_linesearch_prowsubprob(
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
                        fnEvals[iter] += num_evals
                        gradM, phi_row = calc_grad(
                            isSparse, Pi, epsDivZero, x_row, m_row
                        )

                    # Compute the row subproblem kkt_violation.

                    # Note experiments in the original paper used:
                    # kkt_violation = np.norm(np.abs(np.minimum(m_row, gradM.transpose())))

                    # We now use \| KKT \|_{inf}:
                    kkt_violation = np.max(np.abs(np.minimum(m_row, gradM)))
                    # print("Intermediate Printing m_row: {}\n and gradM{}".format(m_row, gradM))

                    # Report largest row subproblem initial violation
                    if i == 0 and kkt_violation > kktModeViolations[n]:
                        kktModeViolations[n] = kkt_violation

                    if printinneritn > 0 and np.mod(i, printinneritn) == 0:
                        print("\tMode = {}, Row = {}, InnerIt = {}".format(n, jj, i))

                        if i == 0:
                            print(", RowKKT = {}\n".format(kkt_violation))
                        else:
                            print(
                                ", RowKKT = {}, RowObj = {}\n".format(
                                    kkt_violation, -f_new
                                )
                            )

                    # Check for row subproblem convergence.
                    if kkt_violation < stoptol:
                        break
                    else:
                        # Not converged, so m_row will be modified.
                        isRowNOTconverged[jj] = 1

                    # Update the L-BFGS approximation.
                    tmp_delm = m_row - m_rowOLD
                    tmp_delg = gradM - gradOLD
                    tmp_delm_dot = tmp_delm.dot(tmp_delg.transpose())
                    if not np.any(tmp_delm_dot == 0):
                        tmp_rho = 1 / tmp_delm_dot
                        delm[:, lbfgsPos] = tmp_delm
                        delg[:, lbfgsPos] = tmp_delg
                        rho[lbfgsPos] = tmp_rho
                    else:
                        # Rho is required to be postive; if not, then skip the L-BFGS update pair. The recommended
                        # safeguard for full BFGS is Powell damping, but not clear how to damp in 2-loop L-BFGS
                        if dispLineWarn:
                            warnings.warn(
                                "WARNING: skipping L-BFGS update, rho whould be 1 / {}".format(
                                    tmp_delm * tmp_delg
                                )
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
                    search_dir = getSearchDirPqnr(
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
                    m_row, f, f_unit, f_new, num_evals = ttb.tt_linesearch_prowsubprob(
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
                    fnEvals[iter] += num_evals

                M.factor_matrices[n][jj, :] = m_row
                countInnerIters[n] += i

            # Test if all row subproblems have converged, which means that no varibales in this more were changed.
            if np.sum(isRowNOTconverged) != 0:
                isConverged = False

            # Shift weight from mode n back to lambda.
            M.normalize(mode=n, normtype=1)

            # Total number of inner iterations for a given outer iteration, totalled across all modes and all
            # row subproblems in each mode
            nInnerIters[iter] += countInnerIters[n]

        # Save output items for the outer iteration.
        num_zero = 0
        for n in range(N):
            num_zero += np.count_nonzero(M[n] == 0)  # [0].size

        nzeros[iter] = num_zero
        kktViolations[iter] = np.max(kktModeViolations)

        # Print outer iteration status.
        if printitn > 0 and np.mod(iter, printitn) == 0:
            fnVals[iter] = -tt_loglikelihood(tensor, M)
            print(
                "{}. Ttl Inner Its: {}, KKT viol = {}, obj = {}, nz: {}\n".format(
                    iter, nInnerIters[iter], kktViolations[iter], fnVals[iter], num_zero
                )
            )

        times[iter] = time.time() - start

        # Check for convergence
        if isConverged:
            break
        if times[iter] > stoptime:
            print("Exiting because time limit exceeded\n")
            break

    t_stop = time.time() - start

    # Clean up final result
    M.normalize(sort=True, normtype=1)

    obj = tt_loglikelihood(tensor, M)

    if printitn > 0:
        normTensor = tensor.norm()
        normresidual = np.sqrt(
            normTensor**2 + M.norm() ** 2 - 2 * tensor.innerprod(M)
        )
        fit = 1 - (normresidual / normTensor)  # fraction explained by model
        print("===========================================\n")
        print(" Final log-likelihood = {} \n".format(obj))
        print(" Final least squares fit = {} \n".format(fit))
        print(" Final KKT violation = {}\n".format(kktViolations[iter]))
        print(" Total inner iterations = {}\n".format(sum(nInnerIters)))
        print(" Total execution time = {} secs\n".format(t_stop))

    output = {}
    output["params"] = (
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
    output["kktViolations"] = kktViolations[: iter + 1]
    output["obj"] = obj
    output["fnEvals"] = fnEvals[: iter + 1]
    output["fnVals"] = fnVals[: iter + 1]
    output["nInnerIters"] = nInnerIters[: iter + 1]
    output["nZeros"] = nzeros[: iter + 1]
    output["times"] = times[: iter + 1]
    output["totalTime"] = t_stop

    return M, output


# PDNR helper functions
def tt_calcpi_prowsubprob(
    Data, Model, rank, factorIndex, ndims, isSparse=False, sparse_indices=None
):
    """
    Compute Pi for a row subproblem.

    Parameters
    ----------
    Data :class:`pyttb.sptensor` or :class:`pyttb.tensor`
    isSparse: bool
    Model :class:`pyttb.ktensor`
    rank: int
    factorIndex: int
    ndims: int
    sparse_indices: list
        Indices of row subproblem nonzero elements

    Returns
    -------
    Pi: :class:`numpy.ndarray`

    See Also
    --------
    :class:`pyttb.calculatePi`

    """
    # TODO: this can probably be merged with general calculate pi, where default for sparse_indices is slice(None,None,None)
    if isSparse:
        # Data is a sparse tensor. Compute Pi for the row problem specified by sparse_indices
        num_row_nnz = len(sparse_indices)

        Pi = np.ones((num_row_nnz, rank))
        for i in np.setdiff1d(np.arange(ndims), factorIndex).astype(int):
            Pi *= Model[i][Data.subs[sparse_indices, i], :]
    else:
        Pi = ttb.khatrirao(
            Model.factor_matrices[:factorIndex]
            + Model.factor_matrices[factorIndex + 1 : ndims + 1],
            reverse=True,
        )

    return Pi


def calc_partials(isSparse, Pi, epsilon, data_row, model_row):
    """
    Compute derivative quantities for a PDNR row subproblem.

    Parameters
    ----------
    isSparse: bool
    Pi: :class:`numpy.ndarray`
    epsilon: float
        Prevent division by zero
    data_row: :class:`numpy.ndarray`
    model_row: :class:`numpy.ndarray`

    Returns
    -------
    phi_row: :class:`numpy.ndarray`
        gradient of row subproblem, except for a constant \n
        :math:`phi\_row[r] = \sum_{j=1}^{J_n}\\frac{x_j\pi_{rj}}{\sum_i^R b_i\pi_{ij}}`
    ups_row: :class:`numpy.ndarray`
        intermediate quantity (upsilon) used for second derivatives  \n
        :math:`ups\_row[j] = \\frac{x_j}{\\left(\sum_i^R b_i\pi_{ij}\\right)^2}`

    """
    if isSparse:
        data_row = data_row.transpose()[0]
    v = model_row.dot(Pi.transpose())
    w = data_row.transpose() / np.maximum(v, epsilon)
    phi_row = w.dot(Pi)
    u = v**2
    ups_row = data_row.transpose() / np.maximum(u, epsilon)
    return phi_row, ups_row


def getSearchDirPdnr(Pi, ups_row, rank, gradModel, model_row, mu, epsActSet):
    """
    Compute the search direction for PDNR using a two-metric projection with damped Hessian

    Parameters
    ----------
    Pi: :class:`numpy.ndarray`
    ups_row: :class:`numpy.ndarray`
        intermediate quantity (upsilon) used for second derivatives
    rank: int
        number of variables for the row subproblem
    gradModel:  :class:`numpy.ndarray`
        gradient vector for the row subproblem
    model_row: :class:`numpy.ndarray`
        vector of variables for the row subproblem
    mu: float
        damping parameter
    epsActSet: float
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
    Hessian_free = getHessian(ups_row, Pi, free_indices)
    grad_free = -gradModel[free_indices]

    # Compute the damped Newton search direction over free variables
    # TODO verify this is appropriate representation of matlab's method, s.b. because hessian is square, and addition
    # should ensure full rank, try.catch handles singular matrix
    try:
        search_dir[free_indices] = np.linalg.solve(
            Hessian_free + (mu * np.eye(num_free)), grad_free
        )[:, None]
    except np.linalg.LinAlgError:
        warnings.warn("CP_APR: Damped Hessian is nearly singular\n")
        # TODO: note this may be a typo in matlab see line 1107
        search_dir = -gradModel

    # Calculate expected reduction in the quadratic model of the objective.
    # TODO: double check if left or right multiplication has an speed effect, memory layout
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


def tt_linesearch_prowsubprob(
    direction,
    grad,
    model_old,
    step_len,
    step_red,
    max_steps,
    suff_decr,
    isSparse,
    data_row,
    Pi,
    phi_row,
    display_warning,
):
    """
    Perform a line search on a row subproblem

    Parameters
    ----------
    direction:  :class:`numpy.ndarray`
        search direction
    grad:  :class:`numpy.ndarray`
        gradient vector a model_old
    model_old:  :class:`numpy.ndarray`
        current variable values
    step_len: float
        initial step length, which is the maximum possible step length
    step_red: float
        step reduction factor (suggest 1/2)
    max_steps: int
        maximum number of steps to try (suggest 10)
    suff_decr: float
        sufficent decrease for convergence (suggest 1.0e-4)
    isSparse: bool
        sparsity flag for computing the objective
    data_row:  :class:`numpy.ndarray`
        row subproblem data, for computing the objective
    Pi:  :class:`numpy.ndarray`
        Pi matrix, for computing the objective
    phi_row:  :class:`numpy.ndarray`
        1-grad, more accurate if failing over to multiplicative update
    display_warning: bool
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

    # Evalute the current objective value
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
            f_new = -ttb.tt_loglikelihood_row(isSparse, data_row, model_new, Pi)
            num_evals += 1
            if count == 1:
                f_1 = f_new

            # Check for sufficient decrease.
            if f_new <= (f_old + suff_decr * gDotd):
                break
            else:
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
        f_new = -ttb.tt_loglikelihood_row(isSparse, data_row, model_new, Pi)
        num_evals += 1

        # Let the caller know the search direction made no progress.
        f_1 = f_old

        if display_warning:
            warnings.warn(
                "CP_APR: Line search failed, using multiplicative update step"
            )

    return model_new, f_old, f_1, f_new, num_evals


def getHessian(upsilon, Pi, free_indices):
    """
    Return the Hessian for one PDNR row subproblem of Model[n], for just the rows and columns corresponding to the free variables

    Parameters
    ----------
    upsilon: :class:`numpy.ndarray`
        intermediate quantity (upsilon) used for second derivatives
    Pi: :class:`numpy.ndarray`
    free_indices: list

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


def tt_loglikelihood_row(isSparse, data_row, model_row, Pi):
    """
    Compute log-likelihood of one row subproblem

    Parameters
    ----------
    isSparse: bool
        Sparsity flag
    data_row: :class:`numpy.ndarray`
        vector of data values
    model_row: :class:`numpy.ndarray`
        vector of model values
    Pi: :class:`numpy.ndarray`

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
        term2 = 0
        for i in range(len(data_row)):
            if data_row[i] != 0:
                # Define zero times log(anything) to be zero
                term2 += data_row[i] * np.log(b_pi[i])
    loglikelihood = term1 + term2
    return loglikelihood


# PQNR helper functions
def getSearchDirPqnr(
    model_row,
    gradModel,
    epsActSet,
    delta_model,
    delta_grad,
    rho,
    lbfgs_pos,
    iters,
    disp_warn,
):
    """
    Compute the search direction by projecting with L-BFGS.

    Parameters
    ----------
    model_row: :class:`numpy.ndarray`
        current variable values
    gradModel: :class:`numpy.ndarray`
        gradient at model_row
    epsActSet: float
        Bertsekas tolerance for active set determination
    delta_model: :class:`numpy.ndarray`
        L-BFGS array of variable deltas
    delta_grad: :class:`numpy.ndarray`
        L-BFGS array of gradient deltas
    rho:
    lbfgs_pos: int
        pointer into L-BFGS arrays
    iters:
    disp_warn: bool

    Returns
    -------
    direction: :class:`numpy.ndarray`
        Search direction based on current L-BFGS and grad

    Notes
    -----
    Adapted from MATLAB code of Dongmin Kim and Suvrit Sra written in 2008.
    Modified extensively to solve row subproblems and use a better linesearch;  for details see
    REFERENCE: Samantha Hansen, Todd Plantenga, Tamara G. Kolda.
    Newton-Based Optimization for Nonnegative Tensor Factorizations,
    arXiv:1304.4964 [math.NA], April 2013,
    URL: http://arxiv.org/abs/1304.4964. Submitted for publication.

    """

    lbfgsSize = delta_model.shape[1]

    # Determine active and free variables.
    # TODO: is the bellow relevant?
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
        # Cannot proceed with this L-BFGS data; most likely the iteration has converged, so this is rarely seen.
        if disp_warn:
            warnings.warn("WARNING: L-BFGS update is orthogonal, using gradient")
        return direction

    alpha = np.ones((lbfgsSize,))
    k = lbfgs_pos

    # Perform an L-BFGS two-loop recursion to compute the search direction.
    for i in range(np.minimum(iters, lbfgsSize)):
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

    for i in range(np.minimum(iters, lbfgsSize)):
        k = np.mod(k, lbfgsSize)  # + 1
        b = rho[k] * (delta_grad[:, k].transpose().dot(direction))
        direction += (alpha[k] - b) * (delta_model[:, k])

    direction[fixedVars] = 0

    return direction


def calc_grad(isSparse, Pi, eps_div_zero, data_row, model_row):
    """
    Compute the gradient for a PQNR row subproblem

    Parameters
    ----------
    isSparse: bool
    Pi: :class:`numpy.ndarray`
    eps_div_zero: float
    data_row: :class:`numpy.ndarray`
    model_row: :class:`numpy.ndarray`

    Returns
    -------
    phi_row: :class:`numpy.ndarray`
    grad_row: :class:`numpy.ndarray`

    """
    # TODO: note this is duplicated exactly from calc_partials, should combine to one function
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


# Mu helper functions
def calculatePi(Data, Model, rank, factorIndex, ndims):
    """
    Helper function to calculate Pi matrix
    # TODO verify what pi is

    Parameters
    ----------
    Data: :class:`pyttb.sptensor` or :class:`pyttb.tensor`
    Model: :class:`pyttb.ktensor`
    rank: int
    factorIndex: int
    ndims: int

    Returns
    -------
    Pi: :class:`numpy.ndarray`
    """
    if isinstance(Data, ttb.sptensor):
        Pi = np.ones((Data.nnz, rank))
        for i in np.setdiff1d(np.arange(ndims), factorIndex).astype(int):
            Pi *= Model[i][Data.subs[:, i], :]
    else:
        Pi = ttb.khatrirao(
            Model.factor_matrices[:factorIndex]
            + Model.factor_matrices[factorIndex + 1 :],
            reverse=True,
        )

    return Pi


def calculatePhi(Data, Model, rank, factorIndex, Pi, epsilon):
    """

    Parameters
    ----------
    Data: :class:`pyttb.sptensor` or :class:`pyttb.tensor`
    Model: :class:`pyttb.ktensor`
    rank: int
    factorIndex: int
    Pi: :class:`numpy.ndarray`
    epsilon: float

    Returns
    -------

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
        Xn = ttb.tt_to_dense_matrix(Data, factorIndex)
        V = Model.factor_matrices[factorIndex].dot(Pi.transpose())
        W = Xn / np.maximum(V, epsilon)
        Y = W.dot(Pi)
        Phi = Y

    return Phi


def tt_loglikelihood(Data, Model):
    """
    Compute log-likelihood of data with model.

    Parameters
    ----------
    Data: :class:`pyttb.sptensor` or :class:`pyttb.tensor`
    Model: :class:`pyttb.ktensor`

    Returns
    -------
    loglikelihood: float
        - (sum_i m_i - x_i * log_i) where i is a multiindex across all tensor dimensions.

    Notes
    -----
    We define for any x 0*log(x)=0, such that if our true data is 0 the loglikelihood is the value of the model.

    """
    N = Data.ndims

    assert isinstance(Model, ttb.ktensor), "Model must be a ktensor"

    Model.normalize(weight_factor=0, normtype=1)
    if isinstance(Data, ttb.sptensor):
        xsubs = Data.subs
        A = Model.factor_matrices[0][xsubs[:, 0], :]
        for n in range(1, N):
            A *= Model.factor_matrices[n][xsubs[:, n], :]
        return np.sum(Data.vals * np.log(np.sum(A, axis=1))[:, None]) - np.sum(
            Model.factor_matrices[0]
        )
    else:
        dX = ttb.tt_to_dense_matrix(Data, 1)
        dM = ttb.tt_to_dense_matrix(Model, 1)
        f = 0
        for i in range(dX.shape[0]):
            for j in range(dX.shape[1]):
                if dX[i, j] == 0:
                    pass
                else:
                    f += dX[i, j] * np.log(dM[i, j])

        f -= np.sum(Model.factor_matrices[0])
        return f


def vectorizeForMu(matrix):
    """
    Helper Function to unravel matrix into vector

    Parameters
    ----------
    matrix: :class:`numpy.ndarray`

    Returns
    -------
    matrix: :class:`numpy.ndarray`
        len(matrix.shape)==1
    """
    return matrix.ravel()
