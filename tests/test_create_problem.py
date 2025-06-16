import numpy as np
import pytest

import pyttb as ttb
from pyttb.create_problem import (
    CPProblem,
    DataParams,
    MissingData,
    TuckerProblem,
    create_problem,
    generate_data,
    generate_solution,
)


class TestDataclasses:
    def test_dataparams(self):
        with pytest.raises(ValueError):
            number_larger_than_one = 2.0
            DataParams(noise=number_larger_than_one)
        with pytest.raises(ValueError):
            number_less_than_zero = -2.0
            DataParams(noise=number_less_than_zero)

    def test_missingdata(self):
        with pytest.raises(ValueError):
            number_larger_than_one = 2.0
            MissingData(missing_ratio=number_larger_than_one)
        with pytest.raises(ValueError):
            number_less_than_zero = -2.0
            MissingData(missing_ratio=number_less_than_zero)

        missing_params = MissingData(missing_ratio=0.1)
        assert missing_params.has_missing()
        with pytest.raises(ValueError):
            missing_params.raise_symmetric()
        with pytest.raises(ValueError):
            missing_params.raise_symmetric()
        missing_params = MissingData()
        assert not missing_params.has_missing()
        missing_params.raise_symmetric()


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
    data = generate_data(model, cp_params, data_params=DataParams())
    assert isinstance(data, ttb.tensor)
    assert data.shape == model.shape


def test_generate_solution_tucker():
    # Smoke test with defaults
    shape = (2, 2, 2)
    tucker_params = TuckerProblem(shape)
    model = generate_solution(tucker_params)
    assert isinstance(model, ttb.ttensor)
    assert model.shape == shape

    # TODO could test with different generators and enforce that they actually get used


def test_generate_data_tucker():
    # Smoke test with defaults
    shape = (2, 2, 2)
    tucker_params = TuckerProblem(shape)
    model = generate_solution(tucker_params)
    data = generate_data(model, tucker_params, data_params=DataParams())
    assert isinstance(data, ttb.tensor)
    assert data.shape == model.shape


def test_create_problem_smoke():
    shape = (2, 2, 2)
    cp_params = CPProblem(shape)
    data_params = DataParams()
    missing_params = MissingData()
    soln, data = create_problem(cp_params, missing_params, data_params)
    assert soln.full().shape == data.shape

    # TODO hit edge cases and symmetric


def test_create_problem_smoke_sparse():
    shape = (2, 2, 2)
    cp_params = CPProblem(
        shape, sparse_generation=0.99, factor_generator=np.random.random
    )
    data_params = DataParams()
    missing_params = MissingData()
    soln, data = create_problem(cp_params, missing_params, data_params)
    assert soln.full().shape == data.shape

    # TODO hit edge cases and symmetric


def test_create_problem_smoke_missing():
    shape = (4, 5, 6)
    cp_params = CPProblem(shape, factor_generator=np.random.random)
    data_params = DataParams()
    missing_params = MissingData(missing_ratio=0.8)
    soln, data = create_problem(cp_params, missing_params, data_params)
    assert soln.full().shape == data.shape

    missing_params = MissingData(missing_ratio=0.8, sparse_model=True)
    soln, data = create_problem(cp_params, missing_params, data_params)
    assert soln.full().shape == data.shape
