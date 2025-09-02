# Copyright 2024 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import pyttb as ttb
from pyttb.decompositions.cp.gcp import samplers
from pyttb.decompositions.cp.gcp.handles import gaussian, gaussian_grad
from pyttb.decompositions.cp.gcp.optimizers import LBFGSB, SGD, Adagrad, Adam

if TYPE_CHECKING:
    from pyttb.gcp.fg_setup import function_type


def diverging_function_handle_factory(function_handle: function_type) -> function_type:
    """Returns a function that diverges when the gradient is correct."""

    def diverging_function_handle(data: np.ndarray, model: np.ndarray) -> np.ndarray:
        return -1.0 * function_handle(data, model)

    return diverging_function_handle


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

    solver = SGD(max_iters=3, epoch_iters=1)
    result, info = solver.solve(
        model, dense_data, gaussian, gaussian_grad, sampler=sampler
    )
    assert isinstance(info, dict)
    assert (model.full() - dense_data).norm() > (result.full() - dense_data).norm()

    # Force inf grad
    with pytest.raises(ValueError):
        inf_data = np.inf * ttb.tenones(dense_data.shape)
        solver.solve(model, inf_data, gaussian, gaussian_grad, sampler=sampler)

    # Force bad step to check rejection
    result, info = solver.solve(
        model,
        dense_data,
        diverging_function_handle_factory(gaussian),
        gaussian_grad,
        sampler=sampler,
    )
    assert model.isequal(result)
    assert solver._nfails == min(solver._max_iters, solver._max_fails + 1)


def test_adam(generate_problem):
    dense_data, model, sampler = generate_problem

    solver = Adam(max_iters=3, epoch_iters=1)
    result, info = solver.solve(
        model, dense_data, gaussian, gaussian_grad, sampler=sampler
    )
    assert isinstance(info, dict)
    assert (model.full() - dense_data).norm() > (result.full() - dense_data).norm()

    # Force bad step to check rejection
    solver = Adam(max_iters=3, epoch_iters=1)
    result, info = solver.solve(
        model,
        dense_data,
        diverging_function_handle_factory(gaussian),
        gaussian_grad,
        sampler=sampler,
    )
    assert model.isequal(result)
    assert [np.testing.assert_array_equal(mk, np.zeros_like(mk)) for mk in solver._m]
    assert [np.testing.assert_array_equal(vk, np.zeros_like(vk)) for vk in solver._v]
    assert solver._nfails == min(solver._max_iters, solver._max_fails + 1)


def test_adagrad(generate_problem):
    dense_data, model, sampler = generate_problem

    solver = Adagrad(max_iters=3, epoch_iters=1)
    result, info = solver.solve(
        model, dense_data, gaussian, gaussian_grad, sampler=sampler
    )
    assert isinstance(info, dict)
    assert (model.full() - dense_data).norm() > (result.full() - dense_data).norm()

    # Force bad step to check rejection
    solver = Adagrad(max_iters=3, epoch_iters=1)
    result, info = solver.solve(
        model,
        dense_data,
        diverging_function_handle_factory(gaussian),
        gaussian_grad,
        sampler=sampler,
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


def test_lbfgsb_callback(generate_problem):
    dense_data, model, sampler = generate_problem

    # Test default callback
    maxiter = 2
    solver = LBFGSB(maxiter=maxiter)
    _, info = solver.solve(model, dense_data, gaussian, gaussian_grad)
    assert (
        not "callback" in info["callback"].keys()
    )  # No nested callback (wasn't defined)
    assert info["callback"]["time_trace"].shape == (info["nit"],)
    assert np.all(info["callback"]["time_trace"] > 0)

    # Test reuse of optimizer with callback
    assert solver._solver_kwargs["callback"] is None  # Unregistered from previous call
    # Inject non-empty callback structure from previous to solver kwargs
    with pytest.raises(TypeError):
        solver._solver_kwargs["callback"] = info["callback"]
        _, info = solver.solve(model, dense_data, gaussian, gaussian_grad)

    # Test user-defined callback
    class Callback:
        def __init__(self, rows: int = 0, cols: int = 0):
            self.i = 0
            self.xk = np.zeros((rows, cols))

        def __call__(self, xk):
            self.xk[self.i, :] = xk
            self.i += 1

    maxiter = 2
    callback = Callback(maxiter, np.prod(model.shape) * model.ncomponents)
    solver = LBFGSB(maxiter=maxiter, callback=callback)
    _, info = solver.solve(model, dense_data, gaussian, gaussian_grad)
    callback_external = vars(callback)
    callback_internal = vars(info["callback"]["callback"])
    assert callback_external["i"] == callback_internal["i"]
    np.testing.assert_array_equal(callback_external["xk"], callback_internal["xk"])
