import pyttb as ttb
from pyttb.create_problem import (
    CPProblem,
    DataParams,
    TuckerProblem,
    generate_data,
    generate_solution,
)


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
