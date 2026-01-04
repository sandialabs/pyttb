# Copyright 2024 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from __future__ import annotations

import numpy as np
import pytest

import pyttb as ttb
from pyttb.gcp.fg_est import estimate, estimate_helper


def test_estimate_helper():
    fm0 = np.array([[1.0, 3.0], [2.0, 4.0]])
    fm1 = np.array([[5.0, 8.0], [6.0, 9.0], [7.0, 10.0]])
    fm2 = np.array([[11.0, 15.0], [12.0, 16.0], [13.0, 17.0], [14.0, 18.0]])
    factor_matrices = [fm0, fm1, fm2]
    model = ttb.ktensor(factor_matrices)
    full = model.full()
    shape = full.shape
    all_indices = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                all_indices.append([i, j, k])
    all_indices = np.array(all_indices)
    values, _ = estimate_helper(factor_matrices, np.array(all_indices))
    np.testing.assert_array_equal(full[all_indices], values)
    # TODO should probably test Zexploded but that's a pain

    values, Z = estimate_helper(factor_matrices, np.array([]))
    assert values.size == 0
    assert len(Z) == 0


def test_estimate():
    fm0 = np.array([[1.0, 3.0], [2.0, 4.0]])
    fm1 = np.array([[5.0, 8.0], [6.0, 9.0], [7.0, 10.0]])
    fm2 = np.array([[11.0, 15.0], [12.0, 16.0], [13.0, 17.0], [14.0, 18.0]])
    factor_matrices = [fm0, fm1, fm2]
    model = ttb.ktensor(factor_matrices)
    data_subs = np.array([[0, 0, 0], [1, 1, 1]])
    data_vals = np.ones((2,))
    weights = np.ones_like(data_vals)

    def no_op(data, model):  # noqa: ARG001
        """Function handle that does nothing"""
        return data

    # No function or gradient handle
    with pytest.raises(ValueError):
        estimate(model, data_subs, data_vals, weights)

    # Fail lambda check
    with pytest.warns():
        bad_model = model.copy()
        bad_model.weights += 5
        estimate(bad_model, data_subs, data_vals, weights, function_handle=no_op)

    # Just function handle
    f = estimate(model, data_subs, data_vals, weights, function_handle=no_op)
    assert isinstance(f, float)
    assert f == np.sum(data_vals)

    zero_weights = np.zeros_like(data_vals)
    f = estimate(model, data_subs, data_vals, zero_weights, function_handle=no_op)
    assert isinstance(f, float)
    assert f == 0.0

    # With our no_op function the crng should have no impact
    crng = np.arange(len(data_vals), dtype=int)
    f = estimate(model, data_subs, data_vals, weights, function_handle=no_op, crng=crng)
    assert isinstance(f, float)
    assert f == np.sum(data_vals)

    # Just gradient handle
    g = estimate(model, data_subs, data_vals, weights, gradient_handle=no_op)
    assert all(isinstance(g_i, np.ndarray) for g_i in g)

    nsamples = data_subs.shape[0]
    g = estimate(model, data_subs, data_vals, zero_weights, gradient_handle=no_op)
    assert all(isinstance(g_i, np.ndarray) for g_i in g)
    assert all(
        np.all(g_i[data_subs[:, i], np.arange(nsamples)] == 0.0)
        for i, g_i in enumerate(g)
    )

    g = estimate(
        model, data_subs, data_vals, zero_weights, gradient_handle=no_op, crng=crng
    )
    assert all(isinstance(g_i, np.ndarray) for g_i in g)
    assert all(
        np.all(g_i[data_subs[:, i], np.arange(nsamples)] == 0.0)
        for i, g_i in enumerate(g)
    )

    # Both handles
    f, g = estimate(
        model,
        data_subs,
        data_vals,
        zero_weights,
        function_handle=no_op,
        gradient_handle=no_op,
    )
    assert isinstance(f, float)
    assert all(isinstance(g_i, np.ndarray) for g_i in g)
