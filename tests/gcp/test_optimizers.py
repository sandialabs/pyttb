from __future__ import annotations

import numpy as np
import pytest

import pyttb as ttb
from pyttb.gcp import samplers
from pyttb.gcp.optimizers import SGD
from pyttb.gcp.handles import gaussian, gaussian_grad


def test_sgd():
    num_zeros = 2
    num_nonzeros = 2
    dense_data = ttb.tenones((2, 2))
    dense_data[0, 1] = 0.0
    dense_data[1, 0] = 0.0
    sampler = samplers.GCPSampler(dense_data, num_zeros, num_nonzeros)
    model = ttb.ktensor([np.ones((2, 2))] * 2)

    solver = SGD(max_iters=1, epoch_iters=1)
    result, info = solver.solve(model, dense_data, gaussian, gaussian_grad, sampler)
