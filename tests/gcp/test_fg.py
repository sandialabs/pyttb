from __future__ import annotations

import numpy as np
import pytest

import pyttb as ttb
from pyttb.gcp.fg import evaluate


def test_evaluate():
    # No function or gradient handle
    with pytest.raises(ValueError):
        model = ttb.ktensor()
        data = ttb.tensor()
        evaluate(model, data)

    def no_op(data, model):
        """Function handle that does nothing"""
        return data

    # Just function handle
    model = ttb.ktensor([np.ones((2, 2))] * 2)
    data = ttb.tenzeros((2, 2))
    f = evaluate(model, data, function_handle=no_op)
    assert isinstance(f, float)
    assert f == np.sum(data.data)

    weight = np.zeros_like(data.data)
    f = evaluate(model, data, weights=weight, function_handle=no_op)
    assert isinstance(f, float)
    assert f == 0.0

    # Just gradient handle
    model = ttb.ktensor([np.ones((k, 2)) for k in (2, 3, 4)])
    data = ttb.tenzeros(model.shape)
    g = evaluate(model, data, gradient_handle=no_op)
    assert all(isinstance(g_i, np.ndarray) for g_i in g)

    weight = np.zeros_like(data.data)
    g = evaluate(model, data, weights=weight, gradient_handle=no_op)
    assert all(isinstance(g_i, np.ndarray) for g_i in g)
    assert all(np.all(g_i == 0.0) for g_i in g)

    # Both handles
    f, g = evaluate(
        model,
        data,
        function_handle=no_op,
        gradient_handle=no_op,
        weights=weight,
    )
    assert isinstance(f, float)
    assert all(isinstance(g_i, np.ndarray) for g_i in g)
