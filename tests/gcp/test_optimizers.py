from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

import pyttb as ttb
from pyttb.gcp import samplers
from pyttb.gcp.handles import gaussian, gaussian_grad
from pyttb.gcp.optimizers import SGD, Adagrad, Adam


def test_sgd():
    num_zeros = 2
    num_nonzeros = 2
    dense_data = ttb.tenones((2, 2))
    dense_data[0, 1] = 0.0
    dense_data[1, 0] = 0.0
    sampler = samplers.GCPSampler(dense_data, num_zeros, num_nonzeros)
    model = ttb.ktensor([np.ones((2, 2))] * 2)

    solver = SGD(max_iters=2, epoch_iters=1)
    result, info = solver.solve(model, dense_data, gaussian, gaussian_grad, sampler)
    assert isinstance(info, dict)
    assert (model.full() - dense_data).norm() > (result.full() - dense_data).norm()

    # Force inf grad
    with pytest.raises(ValueError):
        inf_data = np.inf * dense_data
        solver.solve(model, inf_data, gaussian, gaussian_grad, sampler)


def test_adam():
    num_zeros = 2
    num_nonzeros = 2
    dense_data = ttb.tenones((2, 2))
    dense_data[0, 1] = 0.0
    dense_data[1, 0] = 0.0
    sampler = samplers.GCPSampler(dense_data, num_zeros, num_nonzeros)
    model = ttb.ktensor([np.ones((2, 2))] * 2)

    solver = Adam(max_iters=1, epoch_iters=1)
    result, info = solver.solve(model, dense_data, gaussian, gaussian_grad, sampler)
    assert isinstance(info, dict)
    assert (model.full() - dense_data).norm() > (result.full() - dense_data).norm()


def test_adagrad():
    num_zeros = 2
    num_nonzeros = 2
    dense_data = ttb.tenones((2, 2))
    dense_data[0, 1] = 0.0
    dense_data[1, 0] = 0.0
    sampler = samplers.GCPSampler(dense_data, num_zeros, num_nonzeros)
    model = ttb.ktensor([np.ones((2, 2))] * 2)

    solver = Adagrad(max_iters=1, epoch_iters=1)
    result, info = solver.solve(model, dense_data, gaussian, gaussian_grad, sampler)
    assert isinstance(info, dict)
    assert (model.full() - dense_data).norm() > (result.full() - dense_data).norm()
