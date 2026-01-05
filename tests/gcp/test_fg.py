# Copyright 2024 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from __future__ import annotations

import numpy as np
import pytest

import pyttb as ttb
from pyttb.gcp import fg_setup
from pyttb.gcp.fg import evaluate
from pyttb.gcp.handles import Objectives


def test_evaluate():
    # No function or gradient handle
    with pytest.raises(ValueError):
        model = ttb.ktensor()
        data = ttb.tensor()
        evaluate(model, data)

    def no_op(data, model):  # noqa: ARG001
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
    # Construct an sptensor from fully defined SUB, VAL and SIZE matrices.
    shape = (4, 6, 8)
    subs = np.array(
        [
            [0, 3, 0],
            [0, 3, 5],
            [0, 3, 6],
            [0, 5, 5],
            [0, 5, 6],
            [1, 0, 1],
            [1, 0, 3],
            [1, 0, 7],
            [1, 3, 3],
            [1, 4, 1],
            [1, 5, 1],
            [1, 5, 3],
            [1, 5, 7],
            [3, 0, 7],
            [3, 1, 0],
            [3, 1, 7],
            [3, 4, 7],
        ]
    )

    vals = np.array(
        [
            [1],
            [5],
            [9],
            [1],
            [1],
            [6],
            [5],
            [3],
            [1],
            [1],
            [6],
            [8],
            [3],
            [4],
            [1],
            [2],
            [1],
        ]
    )
    data = ttb.sptensor(subs=subs, vals=vals, shape=shape)

    #  Create a :class:`pyttb.ktensor` from weights and a list of factor matrices:
    weights = np.array([33.0, 17.0, 8.0])
    fm0 = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    fm1 = np.array(
        [
            [14 / 33, 0.0, 0.5],
            [0.0, 0.0, 0.375],
            [0.0, 0.0, 0.0],
            [1 / 33, 15 / 17, 0.0],
            [1 / 33, 0.0, 0.125],
            [17 / 33, 2 / 17, 0.0],
        ]
    )
    fm2 = np.array(
        [
            [0.0, 1 / 17, 0.125],
            [13 / 33, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [14 / 33, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 6 / 17, 0.0],
            [0.0, 10 / 17, 0.0],
            [6 / 33, 0.0, 0.875],
        ]
    )
    model = ttb.ktensor(factor_matrices=[fm0, fm1, fm2], weights=weights)

    EPS = 1e-01

    # Computed with MATLAB TTB v3.5
    function_values = {
        "GAUSSIAN": 5.114502408482430,
        "BERNOULLI_ODDS": -61.434016498399011,
        "BERNOULLI_LOGIT": -125.9778934454338,
        "POISSON": -26.214062308801779,
        "POISSON_LOG": 9751.193889205471,
        "RAYLEIGH": -7705.253241233131,
        "GAMMA": -3844.118857614557,
        "HUBER": 1.659013125911521,
        "NEGATIVE_BINOMIAL": 18.038214072232726,
        "BETA": 438.0174726326384,
        "ZT_POISSON": -3911.3193230730776122,
    }

    # Computed with MATLAB TTB v3.5
    gradient_values = {
        "GAUSSIAN": 1.475451151518274,
        "BERNOULLI_ODDS": 6.490763041535352,
        "BERNOULLI_LOGIT": 16.467247284617468,
        "POISSON": 5.749316318105722,
        "POISSON_LOG": 8205.858044313505,
        "RAYLEIGH": 114972799424.0417,
        "GAMMA": 57486399712.01914,
        "HUBER": 0.403829517433526,
        "NEGATIVE_BINOMIAL": 6.952818383681525,
        "BETA": 5748639985.372480,
        "ZT_POISSON": 57486401131.6107635498046875,
    }

    for an_objective in Objectives:
        # Just function handle
        function_val = function_values[an_objective.name]
        fh, gh, lb = fg_setup.setup(an_objective, additional_parameter=0.1)
        f = evaluate(model, data, function_handle=fh)
        assert isinstance(f, float)
        assert f == pytest.approx(function_val, rel=EPS)

        weight = np.zeros_like(data.double().data)
        f = evaluate(model, data, weights=weight, function_handle=fh)
        assert isinstance(f, float)
        assert f == 0.0

        # Just gradient handle
        gradient_val = gradient_values[an_objective.name]
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
