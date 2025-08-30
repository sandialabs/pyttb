# Copyright 2025 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.
from __future__ import annotations

import numpy as np
import pytest

import pyttb as ttb
from pyttb.cp_apr import (
    calc_partials,
    calculate_phi,
    calculate_pi,
    get_hessian,
    get_search_dir_pdnr,
    get_search_dir_pqnr,
    tt_calcpi_prowsubprob,
    tt_linesearch_prowsubprob,
    tt_loglikelihood,
    tt_loglikelihood_row,
    vectorize_for_mu,
)


@pytest.fixture()
def sample_tensor1():
    weights = np.array([1.0, 2.0])
    fm0 = np.array([[0.0, 0.0], [3.0, 4.0]])
    fm1 = np.array([[0.0, 6.0], [7.0, 8.0]])
    factor_matrices = [fm0, fm1]
    ktensorInstance = ttb.ktensor(factor_matrices, weights)
    tensorInstance = ktensorInstance.full()
    return tensorInstance, ktensorInstance


@pytest.fixture()
def sample_tensor2():
    weights = np.array([1.0, 2.0])
    fm0 = np.array([[1.0, 1.0], [3.0, 4.0]])
    fm1 = np.array([[1.0, 6.0], [7.0, 8.0]])
    factor_matrices = [fm0, fm1]
    ktensorInstance = ttb.ktensor(factor_matrices, weights)
    tensorInstance = ktensorInstance.full()
    return tensorInstance, ktensorInstance


@pytest.fixture()
def default_init_ktensor():
    weights = np.array([1.0, 1.0])
    fm0 = np.array([[1.0, 1.0], [1.0, 1.0]])
    fm1 = np.array([[1.0, 1.0], [1.0, 1.0]])
    factor_matrices = [fm0, fm1]
    ktensorInstance = ttb.ktensor(factor_matrices=factor_matrices, weights=weights)
    return ktensorInstance


@pytest.fixture()
def rand_ktensor():
    fm0 = np.array([[0.69646919, 0.28613933], [0.22685145, 0.55131477]])
    fm1 = np.array([[0.71946897, 0.42310646], [0.9807642, 0.68482974]])
    factor_matrices = [fm0, fm1]
    ktensorInstance = ttb.ktensor(factor_matrices=factor_matrices)
    return ktensorInstance


def test_vectorizeForMu():
    matrix = np.array([[1, 2], [3, 4]])
    vector = np.array([1, 2, 3, 4])
    assert np.array_equal(vectorize_for_mu(matrix), vector)


def test_loglikelihood():
    # Test case when both model and data are zero, we define 0*log(0) = 0
    weights = np.array([1.0, 2.0])
    fm0 = np.array([[0.0, 0.0], [3.0, 4.0]])
    fm1 = np.array([[0.0, 6.0], [7.0, 8.0]])
    factor_matrices = [fm0, fm1]
    ktensorInstance = ttb.ktensor(factor_matrices, weights)
    tensorInstance = ktensorInstance.full()
    sptensorInstance = tensorInstance.to_sptensor()

    # Generate explicit answer
    vector = tensorInstance.data.ravel()
    vector2 = [element * np.log(element) for element in vector if element > 0]
    vector = [element for element in vector if element > 0]
    explicitAnswer = -np.sum(np.array(vector) - np.array(vector2))
    assert np.isclose(
        explicitAnswer, tt_loglikelihood(sptensorInstance, ktensorInstance)
    )
    assert np.isclose(explicitAnswer, tt_loglikelihood(tensorInstance, ktensorInstance))

    # Test case for randomly selected model and data
    np.random.seed(123)
    n = 4
    weights = np.abs(np.random.normal(size=(n,)))
    factor_matrices = []
    for i in range(n):
        factor_matrices.append(np.abs(np.random.normal(size=(5, n))))
    ktensorInstance = ttb.ktensor(factor_matrices, weights)
    tensorInstance = ttb.tensor(np.abs(np.random.normal(size=ktensorInstance.shape)))
    sptensorInstance = tensorInstance.to_sptensor()

    vector = ktensorInstance.full().data.ravel()
    data = tensorInstance.data.ravel()
    vector2 = []
    for idx, element in np.ndenumerate(vector):
        if element == 0:
            vector2.append(0)
        else:
            vector2.append(data[idx] * np.log(element))
    explicitAnswer = -np.sum(np.array(vector) - np.array(vector2))
    assert np.isclose(
        explicitAnswer, tt_loglikelihood(sptensorInstance, ktensorInstance)
    )
    assert np.isclose(explicitAnswer, tt_loglikelihood(tensorInstance, ktensorInstance))


def test_calculatePi():
    # Test simple case
    weights = np.array([1.0, 2.0])
    fm0 = np.array([[0.0, 0.0], [3.0, 4.0]])
    fm1 = np.array([[0.0, 6.0], [7.0, 8.0]])
    factor_matrices = [fm0, fm1]
    ktensorInstance = ttb.ktensor(factor_matrices, weights)
    tensorInstance = ktensorInstance.full()
    sptensorInstance = tensorInstance.to_sptensor()
    answer = np.array([[0, 6], [7, 8]])
    assert np.all(
        np.isclose(
            calculate_pi(tensorInstance, ktensorInstance, 2, 0, tensorInstance.ndims),
            answer,
        )
    )
    assert np.all(
        np.isclose(
            calculate_pi(
                sptensorInstance, ktensorInstance, 2, 0, sptensorInstance.ndims
            ),
            answer,
        )
    )

    """
    # Test case for randomly selected model and data
    np.random.seed(123)
    n = 4
    weights = np.abs(np.random.normal(size=(n,)))
    factor_matrices = []
    for i in range(n):
        factor_matrices.append(np.abs(np.random.normal(size=(5, n))))
    ktensorInstance = ttb.ktensor(factor_matrices, weights)
    tensorInstance = ttb.tensor(np.abs(np.random.normal(size=ktensorInstance.shape)))
    sptensorInstance = tensorInstance.to_sptensor()

    print(tensorInstance.shape)
    print(sptensorInstance.shape)
    print(calculate_pi(tensorInstance, ktensorInstance, n, 0, tensorInstance.ndims).shape)
    print(calculate_pi(sptensorInstance, ktensorInstance, n, 0, sptensorInstance.ndims).shape)
    print(np.all(np.isclose(calculate_pi(tensorInstance, ktensorInstance, n, 0, tensorInstance.ndims),
               calculate_pi(sptensorInstance, ktensorInstance, n, 0, sptensorInstance.ndims))))
    assert True
    """


def test_calculatePhi():
    # Test simple case
    weights = np.array([1.0, 2.0])
    fm0 = np.array([[0.0, 0.0], [3.0, 4.0]])
    fm1 = np.array([[0.0, 6.0], [7.0, 8.0]])
    factor_matrices = [fm0, fm1]
    ktensorInstance = ttb.ktensor(factor_matrices, weights)
    tensorInstance = ktensorInstance.full()
    sptensorInstance = tensorInstance.to_sptensor()
    answer = np.array([[0, 0], [11.226415094339623, 24.830188679245282]])
    Pi = calculate_pi(sptensorInstance, ktensorInstance, 2, 0, tensorInstance.ndims)
    assert np.isclose(
        calculate_phi(sptensorInstance, ktensorInstance, 2, 0, Pi, 1e-12), answer
    ).all()
    assert np.isclose(
        calculate_phi(tensorInstance, ktensorInstance, 2, 0, Pi, 1e-12), answer
    ).all()


def test_cpapr_mu(capsys, sample_tensor1, default_init_ktensor):
    # Test simple case
    tensorInstance, ktensorSolnInstance = sample_tensor1
    ktensorInitInstance = default_init_ktensor
    M, _, _ = ttb.cp_apr(
        input_tensor=tensorInstance, rank=2, init=ktensorInitInstance, printinneritn=1
    )

    # Consume the cp_apr diagnostic printing
    capsys.readouterr()
    assert np.isclose(M.full().data, ktensorSolnInstance.full().data).all()

    # Assert given an initial guess of the final answer yields immediate convergence
    M, _, output = ttb.cp_apr(
        input_tensor=tensorInstance, rank=2, init=ktensorSolnInstance
    )
    capsys.readouterr()
    assert output["nTotalIters"] == 2

    # Assert that params from previous run can be provided as input
    M, _, output2 = ttb.cp_apr(
        input_tensor=tensorInstance,
        rank=2,
        init=ktensorSolnInstance,
        **output["params"],
    )
    capsys.readouterr()
    assert output["params"] == output2["params"]

    # Edge cases
    # Confirm timeout works
    _ = ttb.cp_apr(
        input_tensor=tensorInstance, rank=2, init=ktensorInitInstance, stoptime=-1
    )
    out, _ = capsys.readouterr()
    assert "time limit exceeded" in out


@pytest.mark.filterwarnings(
    "ignore::UserWarning"
)  # Ignores UserWarning "CP_APR: Line search failed, using multiplicative update step"
def test_cpapr_pdnr(capsys, sample_tensor1, default_init_ktensor):
    # Test simple case
    tensorInstance, ktensorSolnInstance = sample_tensor1
    ktensorInitInstance = default_init_ktensor
    M, _, _ = ttb.cp_apr(
        input_tensor=tensorInstance,
        rank=2,
        init=ktensorInitInstance,
        algorithm="pdnr",
        printinneritn=1,
        inexact=False,
    )

    # Consume the cp_apr diagnostic printing
    capsys.readouterr()
    assert np.isclose(M.full().data, ktensorSolnInstance.full().data, rtol=1e-04).all()

    # Try solve with sptensor
    sptensorInstance = tensorInstance.to_sptensor()
    M, _, _ = ttb.cp_apr(
        input_tensor=tensorInstance, rank=2, init=ktensorInitInstance, algorithm="pdnr"
    )
    capsys.readouterr()
    assert np.isclose(M.full().data, ktensorSolnInstance.full().data, rtol=1e-04).all()

    # Do not precompute indices
    M, _, output = ttb.cp_apr(
        input_tensor=sptensorInstance,
        rank=2,
        init=ktensorInitInstance,
        algorithm="pdnr",
        precompinds=False,
    )
    capsys.readouterr()
    assert np.isclose(M.full().data, ktensorSolnInstance.full().data, rtol=1e-04).all()

    # Assert that params from previous run can be provided as input
    M, _, output2 = ttb.cp_apr(input_tensor=tensorInstance, rank=2, **output["params"])
    capsys.readouterr()
    assert output["params"] == output2["params"]

    # Edge cases
    # Confirm timeout works
    _ = ttb.cp_apr(
        input_tensor=tensorInstance,
        rank=2,
        init=ktensorInitInstance,
        algorithm="pdnr",
        stoptime=-1,
    )
    out, _ = capsys.readouterr()
    assert "time limit exceeded" in out


@pytest.mark.filterwarnings(
    "ignore::UserWarning"
)  # Ignores UserWarning "CP_APR: Line search failed, using multiplicative update step"
def test_cpapr_pqnr(
    capsys,
    sample_tensor1,
    sample_tensor2,
    default_init_ktensor,
    rand_ktensor,
):
    # Test case known to fail
    tensorInstance, ktensorSolnInstance = sample_tensor1
    ktensorInitInstance = default_init_ktensor

    with pytest.raises(AssertionError) as excinfo:
        M, _, _ = ttb.cp_apr(
            input_tensor=tensorInstance,
            rank=2,
            init=ktensorInitInstance,
            algorithm="pqnr",
        )
    assert "ERROR: L-BFGS first iterate is bad" in str(excinfo)
    capsys.readouterr()

    # Test simple case
    tensorInstance, ktensorSolnInstance = sample_tensor2
    ktensorInitInstance = rand_ktensor
    M, _, _ = ttb.cp_apr(
        input_tensor=tensorInstance,
        rank=2,
        init=ktensorInitInstance,
        algorithm="pqnr",
        printinneritn=1,
    )
    capsys.readouterr()
    assert np.isclose(M.full().data, ktensorSolnInstance.full().data, rtol=1e-01).all()

    # Try solve with sptensor
    sptensorInstance = tensorInstance.to_sptensor()
    M, _, _ = ttb.cp_apr(
        input_tensor=sptensorInstance,
        rank=2,
        init=ktensorInitInstance,
        algorithm="pqnr",
    )
    capsys.readouterr()
    assert np.isclose(M.full().data, ktensorSolnInstance.full().data, rtol=1e-01).all()

    # Do not precompute indices
    M, _, output = ttb.cp_apr(
        input_tensor=sptensorInstance,
        rank=2,
        init=ktensorInitInstance,
        algorithm="pqnr",
        precompinds=False,
    )
    capsys.readouterr()
    assert np.isclose(M.full().data, ktensorSolnInstance.full().data, rtol=1e-01).all()

    # Assert that params from previous run can be provided as input
    M, _, output2 = ttb.cp_apr(
        input_tensor=sptensorInstance,
        rank=2,
        init=ktensorInitInstance,
        **output["params"],
    )
    capsys.readouterr()
    assert output["params"] == output2["params"]

    # Edge cases
    # Confirm timeout works
    _ = ttb.cp_apr(
        input_tensor=tensorInstance,
        rank=2,
        init=ktensorInitInstance,
        algorithm="pqnr",
        stoptime=-1,
    )
    out, _ = capsys.readouterr()
    assert "time limit exceeded" in out


# PDNR tests below


def test_calculatepi_prowsubprob():
    # Test simple case
    weights = np.array([1.0, 2.0])
    fm0 = np.array([[0.0, 0.0], [3.0, 4.0]])
    fm1 = np.array([[0.0, 6.0], [7.0, 8.0]])
    factor_matrices = [fm0, fm1]
    ktensorInstance = ttb.ktensor(factor_matrices, weights)
    tensorInstance = ktensorInstance.full()
    sptensorInstance = tensorInstance.to_sptensor()
    answer = np.array([[0, 6], [7, 8]])
    # Reproduce calculate pi with the appropriate inputs
    assert np.all(
        np.isclose(
            tt_calcpi_prowsubprob(
                tensorInstance, ktensorInstance, 2, 0, tensorInstance.ndims
            ),
            answer,
        )
    )
    assert np.all(
        np.isclose(
            tt_calcpi_prowsubprob(
                sptensorInstance,
                ktensorInstance,
                2,
                0,
                sptensorInstance.ndims,
                True,
                np.arange(sptensorInstance.subs.shape[0]),
            ),
            answer,
        )
    )


def test_calc_partials():
    # Test simple case
    weights = np.array([1.0, 2.0])
    fm0 = np.array([[0.0, 0.0], [3.0, 4.0]])
    fm1 = np.array([[0.0, 6.0], [7.0, 8.0]])
    factor_matrices = [fm0, fm1]
    ktensorInstance = ttb.ktensor(factor_matrices, weights)
    tensorInstance = ktensorInstance.full()
    # print(tensorInstance[:, 0])
    sptensorInstance = tensorInstance.to_sptensor()
    np.array([[0, 6], [7, 8]])
    Pi = calculate_pi(sptensorInstance, ktensorInstance, 2, 0, tensorInstance.ndims)
    # TODO: These are just verifying same functionality as matlab
    phi, ups = calc_partials(
        False,
        Pi,
        1e-12,
        tensorInstance[0, :].data,
        ktensorInstance.factor_matrices[0][0, :],
    )
    assert np.isclose(phi, np.array([0, 0])).all()
    assert np.isclose(ups, np.array([0, 0])).all()

    phi, ups = calc_partials(
        False,
        Pi,
        1e-12,
        tensorInstance[1, :].data,
        ktensorInstance.factor_matrices[0][0, :],
    )
    assert np.isclose(phi, 1e14 * np.array([5.95, 9.68])).all()
    assert np.isclose(ups, 1e13 * np.array([4.8, 8.5])).all()

    phi, ups = calc_partials(
        True, Pi, 1e-12, sptensorInstance.vals, ktensorInstance.factor_matrices[0][0, :]
    )
    assert np.isclose(phi, 1e14 * np.array([5.95, 9.68])).all()
    assert np.isclose(ups, 1e13 * np.array([4.8, 8.5])).all()

    # Compare against explicit calculation reproduced from (3.1) in reference
    # Note phi_row = 1 - [\nabla_1f_{row}(b),.., \nabla_{rank}f_{row}(b)]
    # This is meant to be an inefficient yet explicit implementation
    rank = 5
    length = 7
    m_row = np.random.normal(size=(rank,))
    x_row = np.random.normal(size=(length,))
    Pi = np.random.normal(size=(length, rank))
    hessM = np.zeros(shape=(length,))
    gradM = np.zeros(shape=(rank,))

    eps_div_zero = 1e-12
    # Test \nabla_{r}f_{row}(b)
    for r in range(rank):
        grad_sum = 0
        for j in range(Pi.shape[0]):
            denominator = 0
            for i in range(rank):
                # Note Pi indices are flipped, not sure if paper definition is transposed of ours
                denominator += m_row[i] * Pi[j, i]
            grad_sum += (x_row[j] * Pi[j, r]) / np.maximum(denominator, eps_div_zero)
            hessian_sum = (x_row[j]) / np.maximum(denominator**2, eps_div_zero)
            # Test \nabla^2_{rs}f_{row}(b)
            hessM[j] = hessian_sum

        gradM[r] = grad_sum

    phi, ups = calc_partials(False, Pi, eps_div_zero, x_row, m_row)
    assert np.allclose(phi, gradM)
    assert np.allclose(ups, hessM)


def test_getHessian():
    # Test simple case
    weights = np.array([1.0, 2.0])
    fm0 = np.array([[0.0, 0.0], [3.0, 4.0]])
    fm1 = np.array([[0.0, 6.0], [7.0, 8.0]])
    factor_matrices = [fm0, fm1]
    ktensorInstance = ttb.ktensor(factor_matrices, weights)
    tensorInstance = ktensorInstance.full()
    # print(tensorInstance[:, 0])
    free_indices = [0, 1]
    rank = 2
    sptensorInstance = tensorInstance.to_sptensor()
    Pi = calculate_pi(sptensorInstance, ktensorInstance, rank, 0, tensorInstance.ndims)
    phi, ups = calc_partials(
        False,
        Pi,
        1e-12,
        tensorInstance[1, :].data,
        ktensorInstance.factor_matrices[0][0, :],
    )
    Hessian = get_hessian(ups, Pi, free_indices)
    assert np.allclose(Hessian, Hessian.transpose())

    # Element indexed simple test
    rank = 5
    length = 7
    m_row = np.random.normal(size=(rank,))
    x_row = np.random.normal(size=(length,))
    Pi = np.random.normal(size=(length, rank))
    phi, ups = calc_partials(False, Pi, 1e-12, x_row, m_row)
    free_indices = [0, 1]
    Hessian = get_hessian(ups, Pi, free_indices)

    #
    num_free = len(free_indices)
    H = np.zeros((num_free, num_free))
    for i in range(num_free):
        for j in range(num_free):
            c = free_indices[i]
            d = free_indices[j]
            val = 0
            for k in range(Pi.shape[0]):
                val += ups[k] * Pi[k, c] * Pi[k, d]
            H[(i, j), (j, i)] = val

    assert np.allclose(H, Hessian)


def test_getSearchDirPdnr():
    # Test simple case
    weights = np.array([1.0, 2.0])
    fm0 = np.array([[0.0, 0.0], [3.0, 4.0]])
    fm1 = np.array([[0.0, 6.0], [7.0, 8.0]])
    factor_matrices = [fm0, fm1]
    ktensorInstance = ttb.ktensor(factor_matrices, weights)
    tensorInstance = ktensorInstance.full()
    # print(tensorInstance[:, 0])
    sptensorInstance = tensorInstance.to_sptensor()
    data_row = tensorInstance[1, :].data
    model_row = ktensorInstance.factor_matrices[0][0, :]
    Pi = calculate_pi(sptensorInstance, ktensorInstance, 2, 0, tensorInstance.ndims)
    phi, ups = calc_partials(False, Pi, 1e-12, data_row, model_row)
    search, pred = get_search_dir_pdnr(Pi, ups, 2, phi, model_row, 0.1, 1e-6)
    # TODO validate this projection formulation
    projGradStep = (model_row - ups.transpose()) * (
        model_row - (ups.transpose() > 0).astype(float)
    )
    wk = np.linalg.norm(model_row - projGradStep)
    epsilon_k = np.minimum(1e-6, wk)
    free_indices = []
    # Validates formulation of (3.5)
    for r in range(len(search)):
        if model_row[r] == 0 and ups[r] > 0:  # in set A
            assert search[r] == 0
        elif 0 < model_row[r] <= epsilon_k and ups[r] > 0:  # in set G
            assert search[r] == ups[r]
        else:
            free_indices.append(r)
    Hessian_free = get_hessian(ups, Pi, free_indices)
    direction = np.linalg.solve(
        Hessian_free + (0.1 * np.eye(len(free_indices))), -ups[free_indices]
    )[:, None]
    for i in free_indices:
        assert search[i] == direction[i]


def test_tt_loglikelihood_row():
    # Test simple case
    weights = np.array([1.0, 2.0])
    fm0 = np.array([[0.0, 0.0], [3.0, 4.0]])
    fm1 = np.array([[0.0, 6.0], [7.0, 8.0]])
    factor_matrices = [fm0, fm1]
    ktensorInstance = ttb.ktensor(factor_matrices, weights)
    tensorInstance = ktensorInstance.full()
    # print(tensorInstance[:, 0])
    sptensorInstance = tensorInstance.to_sptensor()
    Pi = calculate_pi(sptensorInstance, ktensorInstance, 2, 0, tensorInstance.ndims)
    tt_loglikelihood_row(
        False, tensorInstance[1, :].data, tensorInstance[1, :].data, Pi
    )
    # print(loglikelihood)


def test_tt_linesearch_prowsubprob():
    # Test simple case
    weights = np.array([1.0, 2.0])
    fm0 = np.array([[0.0, 0.0], [3.0, 4.0]])
    fm1 = np.array([[0.0, 6.0], [7.0, 8.0]])
    factor_matrices = [fm0, fm1]
    ktensorInstance = ttb.ktensor(factor_matrices, weights)
    tensorInstance = ktensorInstance.full()
    # print(tensorInstance[:, 0])
    sptensorInstance = tensorInstance.to_sptensor()
    Pi = calculate_pi(sptensorInstance, ktensorInstance, 2, 0, tensorInstance.ndims)
    phi, ups = calc_partials(
        False,
        Pi,
        1e-12,
        tensorInstance[1, :].data,
        ktensorInstance.factor_matrices[0][0, :],
    )
    search, pred = get_search_dir_pdnr(
        Pi, ups, 2, phi, tensorInstance[1, :].data, 0.1, 1e-6
    )
    with pytest.warns(Warning) as record:
        tt_linesearch_prowsubprob(
            search.transpose()[0],
            phi.transpose(),
            tensorInstance[1, :].data,
            1,
            1 / 2,
            10,
            1.0e-4,
            False,
            tensorInstance[1, :].data,
            Pi,
            phi,
            True,
        )
    assert "CP_APR: Line search failed, using multiplicative update step" in str(
        record[0].message
    )


def test_getSearchDirPqnr():
    # Test simple case
    weights = np.array([1.0, 2.0])
    fm0 = np.array([[0.0, 0.0], [3.0, 4.0]])
    fm1 = np.array([[0.0, 6.0], [7.0, 8.0]])
    factor_matrices = [fm0, fm1]
    ktensorInstance = ttb.ktensor(factor_matrices, weights)
    tensorInstance = ktensorInstance.full()
    # print(tensorInstance[:, 0])
    sptensorInstance = tensorInstance.to_sptensor()
    data_row = tensorInstance[1, :].data
    model_row = ktensorInstance.factor_matrices[0][0, :]
    Pi = calculate_pi(sptensorInstance, ktensorInstance, 2, 0, tensorInstance.ndims)
    phi, ups = calc_partials(False, Pi, 1e-12, data_row, model_row)
    delta_model = np.random.normal(size=(2, model_row.shape[0]))
    delta_grad = np.random.normal(size=(2, phi.shape[0]))
    search, pred = get_search_dir_pqnr(
        model_row, phi, 1e-6, delta_model, delta_grad, phi, 1, 5, False
    )
    # This only verifies that for the right shaped input nothing crashes. Doesn't verify correctness
    assert True


def test_cp_apr_negative_tests():
    dense_tensor = ttb.tensor(np.ones((2, 2, 2)))
    bad_weights = np.array([8.0])
    bad_factors = [np.array([[1.0]])] * 3
    bad_initial_guess_shape = ttb.ktensor(bad_factors, bad_weights)
    with pytest.raises(AssertionError):
        ttb.cp_apr(dense_tensor, init=bad_initial_guess_shape, rank=1)
    good_weights = np.array([8.0] * 3)
    good_factor = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    bad_initial_guess_factors = ttb.ktensor([-1.0 * good_factor] * 3, good_weights)
    with pytest.raises(AssertionError):
        ttb.cp_apr(dense_tensor, init=bad_initial_guess_factors, rank=3)
    bad_initial_guess_weight = ttb.ktensor([good_factor] * 3, -1.0 * good_weights)
    with pytest.raises(AssertionError):
        ttb.cp_apr(dense_tensor, init=bad_initial_guess_weight, rank=3)

    with pytest.raises(AssertionError):
        ttb.cp_apr(dense_tensor, rank=1, algorithm="UNSUPPORTED_ALG")

    with pytest.raises(ValueError):
        ttb.cp_apr(dense_tensor, rank=1, init="Bad initial guess")
