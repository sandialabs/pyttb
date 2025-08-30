from __future__ import annotations

import numpy as np
import pytest

import pyttb as ttb
from pyttb.create_problem import (
    BaseProblem,
    CPProblem,
    ExistingSolution,
    MissingData,
    TuckerProblem,
    create_problem,
    generate_data,
    generate_data_sparse,
    generate_solution,
)


class TestDataclasses:
    def test_problemparams(self):
        arbitrary_shape = (2, 2, 2)
        with pytest.raises(ValueError):
            number_larger_than_one = 2.0
            BaseProblem(arbitrary_shape, noise=number_larger_than_one)
        with pytest.raises(ValueError):
            number_less_than_zero = -2.0
            BaseProblem(arbitrary_shape, noise=number_less_than_zero)

    def test_missingdata(self):
        arbitrary_shape = (2, 2, 2)
        with pytest.raises(ValueError):
            number_larger_than_one = 2.0
            MissingData(missing_ratio=number_larger_than_one)
        with pytest.raises(ValueError):
            number_less_than_zero = -2.0
            MissingData(missing_ratio=number_less_than_zero)

        with pytest.raises(ValueError):
            number_larger_than_zero = 1.0
            arbitrary_missing_pattern = ttb.tenones(arbitrary_shape)
            MissingData(
                missing_ratio=number_larger_than_zero,
                missing_pattern=arbitrary_missing_pattern,
            )

        missing_params = MissingData(missing_ratio=0.1)
        assert missing_params.has_missing()
        with pytest.raises(ValueError):
            missing_params.raise_symmetric()

        missing_params = MissingData(sparse_model=True)
        with pytest.raises(ValueError):
            missing_params.raise_symmetric()

        missing_params = MissingData()
        assert not missing_params.has_missing()
        missing_params.raise_symmetric()

        missing_params = MissingData()
        assert missing_params.get_pattern(arbitrary_shape) is None

    def test_missingdata_logging(self, caplog):
        arbitrary_shape = (2, 2, 2)
        missing_params = MissingData(missing_ratio=0.1, sparse_model=True)
        missing_params.get_pattern(arbitrary_shape)
        assert "missing elements" in caplog.text

    def test_existing_solution(self, sample_ktensor_2way):
        solution = sample_ktensor_2way
        existing_solution = ExistingSolution(solution)
        assert existing_solution.solution is solution
        assert existing_solution.noise == 0.1

        with pytest.raises(ValueError):
            value_less_than_zero = -0.1
            ExistingSolution(solution, noise=value_less_than_zero)

        with pytest.raises(ValueError):
            value_greater_than_one = 1.1
            ExistingSolution(solution, noise=value_greater_than_one)


def test_generate_solution_cp():
    # Smoke test with defaults
    shape = (2, 2, 2)
    cp_params = CPProblem(shape)
    model = generate_solution(cp_params)
    assert isinstance(model, ttb.ktensor)
    assert model.shape == shape

    # TODO could test with different generators and enforce that they actually get used


def test_generate_data_cp():
    # Smoke test with defaults
    shape = (2, 2, 2)
    cp_params = CPProblem(shape)
    model = generate_solution(cp_params)
    data = generate_data(model, cp_params)
    assert isinstance(data, ttb.tensor)
    assert data.shape == model.shape


def test_generate_solution_tucker():
    # Smoke test with defaults
    shape = (2, 2, 2)
    tucker_params = TuckerProblem(shape)
    model = generate_solution(tucker_params)
    assert isinstance(model, ttb.ttensor)
    assert model.shape == shape

    # Smoke test with a tensor core generator
    shape = (2, 2, 2)
    tucker_params = TuckerProblem(shape, core_generator=ttb.tenrand)
    model = generate_solution(tucker_params)
    assert isinstance(model, ttb.ttensor)
    assert model.shape == shape
    # TODO could test with different generators and enforce that they actually get used


def test_generate_data_tucker():
    # Smoke test with defaults
    shape = (2, 2, 2)
    tucker_params = TuckerProblem(shape)
    model = generate_solution(tucker_params)
    data = generate_data(model, tucker_params)
    assert isinstance(data, ttb.tensor)
    assert data.shape == model.shape


def test_create_problem_smoke():
    shape = (2, 2, 2)
    cp_params = CPProblem(shape)
    missing_params = MissingData()
    soln, data = create_problem(cp_params, missing_params)
    assert soln.full().shape == data.shape

    existing_params = ExistingSolution(soln)
    missing_params = MissingData()
    soln, data = create_problem(existing_params, missing_params)
    assert soln.full().shape == data.shape
    assert soln is existing_params.solution, "Solution should be the same object"

    cp_params.symmetric = [(0, 1)]
    soln, data = create_problem(cp_params, missing_params)
    assert soln.full().shape == data.shape

    with pytest.raises(ValueError):
        empty_num_factors = BaseProblem(shape)
        create_problem(empty_num_factors, missing_params)
    with pytest.raises(ValueError):
        inconsistent_num_factors = BaseProblem(shape, num_factors=[2, 2])
        create_problem(inconsistent_num_factors, missing_params)
    with pytest.raises(ValueError):
        bad_problem_type = BaseProblem(shape, num_factors=3)
        create_problem(bad_problem_type, missing_params)

    # TODO hit edge cases and symmetric


def test_create_problem_smoke_sparse():
    shape = (2, 2, 2)
    cp_params = CPProblem(
        shape, sparse_generation=0.99, factor_generator=np.random.random
    )
    missing_params = MissingData()
    soln, data = create_problem(cp_params, missing_params)
    assert soln.full().shape == data.shape

    with pytest.raises(ValueError):
        missing_AND_sparse_generation = MissingData(missing_ratio=0.1)
        create_problem(cp_params, missing_AND_sparse_generation)
    # TODO hit edge cases and symmetric


def test_create_problem_smoke_missing():
    shape = (4, 5, 6)
    cp_params = CPProblem(shape, factor_generator=np.random.random)
    missing_params = MissingData(missing_ratio=0.8)
    soln, data = create_problem(cp_params, missing_params)
    assert soln.full().shape == data.shape

    missing_params = MissingData(missing_ratio=0.8, sparse_model=True)
    soln, data = create_problem(cp_params, missing_params)
    assert soln.full().shape == data.shape

    with pytest.raises(ValueError):
        bad_pattern_shape = np.ones([dim + 1 for dim in soln.shape])
        missing_params = MissingData(missing_pattern=bad_pattern_shape)
        create_problem(cp_params, missing_params)

    with pytest.raises(ValueError):
        bad_pattern_type = np.ones(soln.shape)
        missing_params = MissingData(missing_pattern=bad_pattern_type)
        create_problem(cp_params, missing_params)


def test_generate_data_sparse_value_errors():
    """Test that generate_data_sparse raises expected ValueErrors."""
    shape = (3, 3, 3)

    # Test negative weights
    factor_matrices = [np.random.random((3, 2)) for _ in range(3)]
    negative_weights = np.array([-1.0, 1.0])  # One negative weight
    solution = ttb.ktensor(factor_matrices, negative_weights)
    problem_params = CPProblem(shape, sparse_generation=0.5)

    with pytest.raises(ValueError):
        generate_data_sparse(solution, problem_params)

    # Test negative factor matrices
    factor_matrices = [np.random.random((3, 2)) for _ in range(3)]
    factor_matrices[0][0, 0] = -1.0  # Make one element negative
    positive_weights = np.array([1.0, 1.0])
    solution = ttb.ktensor(factor_matrices, positive_weights)
    problem_params = CPProblem(shape, sparse_generation=0.5)

    with pytest.raises(ValueError):
        generate_data_sparse(solution, problem_params)

    # Test missing sparse_generation
    factor_matrices = [np.random.random((3, 2)) for _ in range(3)]
    positive_weights = np.array([1.0, 1.0])
    solution = ttb.ktensor(factor_matrices, positive_weights)
    problem_params = CPProblem(shape, sparse_generation=None)

    with pytest.raises(ValueError):
        generate_data_sparse(solution, problem_params)
