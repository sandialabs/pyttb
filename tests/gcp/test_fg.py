from __future__ import annotations

from inspect import Parameter, signature
from math import exp, log, pi

import numpy as np
import pytest
import scipy

import pyttb as ttb
from pyttb.gcp import fg_setup
from pyttb.gcp.fg import evaluate
from pyttb.gcp.handles import Objectives


def test_evaluate():
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


def test_evaluate_sptensor():
    model = ttb.import_data("tests/data/ktensor_low_rank_small.tns")
    data = ttb.import_data("tests/data/sptensor_low_rank_small.tns")
    EPS = 1e-04

    # Computed with MATLAB TTB v3.5
    function_values = [
        5.114502408482430,
        -61.434016498399011,
        -125.977893445433750,
        -26.214062308801779,
        9751.193889205471351,
        -7705.253241233131121,
        -3844.118857614557328,
        1.659013125911521,
        18.038214072232726,
        438.017472632638373,
    ]

    # Computed with MATLAB TTB v3.5
    gradient_values = [
        1.475451151518274,
        6.490763041535352,
        16.467247284617468,
        5.749316318105722,
        8205.858044313505161,
        114972799424.041717529296875,
        57486399712.019142150878906,
        0.403829517433526,
        6.952818383681525,
        5748639985.372480392456055,
    ]

    for an_objective, function_val, gradient_val in zip(
        Objectives, function_values, gradient_values
    ):
        # Just function handle
        fh, gh, lb = fg_setup.setup(an_objective, additional_parameter=0.1)
        f = evaluate(model, data, function_handle=fh)
        assert isinstance(f, float)
        assert f == pytest.approx(function_val, rel=EPS)

        weight = np.zeros_like(data.double().data)
        f = evaluate(model, data, weights=weight, function_handle=fh)
        assert isinstance(f, float)
        assert f == 0.0

        # Just gradient handle
        g = evaluate(model, data, gradient_handle=gh)
        assert all(isinstance(g_i, np.ndarray) for g_i in g)
        assert np.linalg.norm(np.vstack(g)) == pytest.approx(gradient_val, rel=EPS)

        weight = np.zeros_like(data.double().data)
        g = evaluate(model, data, weights=weight, gradient_handle=gh)
        assert all(isinstance(g_i, np.ndarray) for g_i in g)
        assert all(np.all(g_i == 0.0) for g_i in g)

        # Both handles
        f, g = evaluate(
            model,
            data,
            function_handle=fh,
            gradient_handle=gh,
            weights=weight,
        )
        assert isinstance(f, float)
        assert all(isinstance(g_i, np.ndarray) for g_i in g)
