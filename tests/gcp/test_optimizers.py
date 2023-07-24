from __future__ import annotations

import numpy as np
import pytest

import pyttb as ttb
from pyttb.gcp import samplers
from pyttb.gcp.handles import gaussian, gaussian_grad
from pyttb.gcp.optimizers import LBFGSB, SGD, Adagrad, Adam

global f_est
f_est = 0.0


def diverging_function_handle(data: np.ndarray, model: np.ndarray) -> np.ndarray:
    global f_est
    f_est += 1.0
    return f_est * data


@pytest.fixture()
def generate_problem():
    dense_data = ttb.tenones((2, 2))
    dense_data[0, 1] = 0.0
    dense_data[1, 0] = 0.0
    sampler = samplers.GCPSampler(dense_data)
    model = ttb.ktensor([np.ones((2, 2))] * 2)
    return dense_data, model, sampler


def test_sgd(generate_problem):
    dense_data, model, sampler = generate_problem

    solver = SGD(max_iters=2, epoch_iters=1)
    result, info = solver.solve(
        model, dense_data, gaussian, gaussian_grad, sampler=sampler
    )
    assert isinstance(info, dict)
    assert (model.full() - dense_data).norm() > (result.full() - dense_data).norm()

    # Force inf grad
    with pytest.raises(ValueError):
        inf_data = np.inf * ttb.tenones(dense_data.shape)
        solver.solve(model, inf_data, gaussian, gaussian_grad, sampler=sampler)

    # Force bad step
    result, info = solver.solve(
        model, dense_data, diverging_function_handle, gaussian_grad, sampler=sampler
    )
    assert model.isequal(result)
    assert solver._nfails == min(solver._max_iters, solver._max_fails + 1)


def test_adam(generate_problem):
    dense_data, model, sampler = generate_problem

    solver = Adam(max_iters=1, epoch_iters=1)
    result, info = solver.solve(
        model, dense_data, gaussian, gaussian_grad, sampler=sampler
    )
    assert isinstance(info, dict)
    assert (model.full() - dense_data).norm() > (result.full() - dense_data).norm()

    # Force bad step
    solver = Adam(max_iters=1, epoch_iters=1)
    result, info = solver.solve(
        model, dense_data, diverging_function_handle, gaussian_grad, sampler=sampler
    )
    assert model.isequal(result)
    assert [np.testing.assert_array_equal(mk, np.zeros_like(mk)) for mk in solver._m]
    assert [np.testing.assert_array_equal(vk, np.zeros_like(vk)) for vk in solver._v]
    assert solver._nfails == min(solver._max_iters, solver._max_fails + 1)


def test_adagrad(generate_problem):
    dense_data, model, sampler = generate_problem

    solver = Adagrad(max_iters=1, epoch_iters=1)
    result, info = solver.solve(
        model, dense_data, gaussian, gaussian_grad, sampler=sampler
    )
    assert isinstance(info, dict)
    assert (model.full() - dense_data).norm() > (result.full() - dense_data).norm()

    # Force bad step
    solver = Adagrad(max_iters=1, epoch_iters=1)
    result, info = solver.solve(
        model, dense_data, diverging_function_handle, gaussian_grad, sampler=sampler
    )
    assert model.isequal(result)
    assert solver._gnormsum == 0.0
    assert solver._nfails == min(solver._max_iters, solver._max_fails + 1)


def test_lbfgsb(generate_problem):
    dense_data, model, sampler = generate_problem

    solver = LBFGSB(maxiter=2)
    result, info = solver.solve(model, dense_data, gaussian, gaussian_grad)
    assert isinstance(info, dict)
    assert not model.isequal(result)
    assert (model.full() - dense_data).norm() > (result.full() - dense_data).norm()
