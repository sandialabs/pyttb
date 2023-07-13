from __future__ import annotations

from math import exp, log, pi

import numpy as np
import pytest
import scipy

import pyttb as ttb
from pyttb.gcp import handles
from pyttb.gcp.handles import EPS


@pytest.fixture()
def sample_data_model():
    data = ttb.tenrand((4, 4))[:]
    model = ttb.tenrand((4, 4))[:]
    return data, model


def test_gaussian(sample_data_model):
    data, model = sample_data_model
    result = handles.gaussian(data, model)
    expected = (model - data) ** 2
    assert np.array_equal(result, expected)


def test_gaussian_grad(sample_data_model):
    data, model = sample_data_model
    result = handles.gaussian_grad(data, model)
    expected = 2 * (model - data)
    assert np.array_equal(result.data, expected)


def test_bernoulli_odds(sample_data_model):
    data, model = sample_data_model
    result = handles.bernoulli_odds(data, model)
    expected = np.zeros_like(result)
    for i, _ in enumerate(expected):
        expected[i] = log(model[i] + 1) - data[i] * log(model[i] + EPS)
    assert np.array_equal(result, expected)


def test_bernoulli_odds_grad(sample_data_model):
    data, model = sample_data_model
    result = handles.bernoulli_odds_grad(data, model)
    expected = np.zeros_like(result)
    for i, _ in enumerate(expected):
        expected[i] = 1.0 / (model[i] + 1) - data[i] / (model[i] + EPS)
    assert np.array_equal(result, expected)


def test_bernoulli_logit(sample_data_model):
    data, model = sample_data_model
    result = handles.bernoulli_logit(data, model)
    expected = np.zeros_like(result)
    for i, _ in enumerate(expected):
        expected[i] = log(exp(model[i]) + 1) - data[i] * model[i]
    assert np.array_equal(result, expected)


def test_bernoulli_logit_grad(sample_data_model):
    data, model = sample_data_model
    result = handles.bernoulli_logit_grad(data, model)
    expected = np.zeros_like(result)
    for i, _ in enumerate(expected):
        expected[i] = exp(model[i]) / (exp(model[i]) + 1) - data[i]
    assert np.array_equal(result, expected)


def test_poisson(sample_data_model):
    data, model = sample_data_model
    result = handles.poisson(data, model)
    expected = np.zeros_like(result)
    for i, _ in enumerate(expected):
        expected[i] = model[i] - data[i] * log(model[i] + EPS)
    assert np.array_equal(result, expected)


def test_poisson_grad(sample_data_model):
    data, model = sample_data_model
    result = handles.poisson_grad(data, model)
    expected = np.zeros_like(result)
    for i, _ in enumerate(expected):
        expected[i] = 1 - data[i] / (model[i] + EPS)
    assert np.array_equal(result, expected)


def test_poisson_log(sample_data_model):
    data, model = sample_data_model
    result = handles.poisson_log(data, model)
    expected = np.zeros_like(result)
    for i, _ in enumerate(expected):
        expected[i] = exp(model[i]) - data[i] * model[i]
    assert np.array_equal(result, expected)


def test_poisson_log_grad(sample_data_model):
    data, model = sample_data_model
    result = handles.poisson_log_grad(data, model)
    expected = np.zeros_like(result)
    for i, _ in enumerate(expected):
        expected[i] = exp(model[i]) - data[i]
    assert np.array_equal(result, expected)


def test_rayleigh(sample_data_model):
    data, model = sample_data_model
    result = handles.rayleigh(data, model)
    expected = np.zeros_like(result)
    for i, _ in enumerate(expected):
        expected[i] = (
            2 * log(model[i] + EPS) + (pi / 4) * (data[i] / (model[i] + EPS)) ** 2
        )
    assert np.array_equal(result, expected)


def test_rayleigh_grad(sample_data_model):
    data, model = sample_data_model
    result = handles.rayleigh_grad(data, model)
    expected = np.zeros_like(result)
    for i, _ in enumerate(expected):
        expected[i] = (
            2 / (model[i] + EPS) - (pi / 2) * data[i] ** 2 / (model[i] + EPS) ** 3
        )
    assert np.array_equal(result, expected)


def test_gamma(sample_data_model):
    data, model = sample_data_model
    result = handles.gamma(data, model)
    expected = np.zeros_like(result)
    for i, _ in enumerate(expected):
        expected[i] = data[i] / (model[i] + EPS) + log(model[i] + EPS)
    assert np.array_equal(result, expected)


def test_gamma_grad(sample_data_model):
    data, model = sample_data_model
    result = handles.gamma_grad(data, model)
    expected = np.zeros_like(result)
    for i, _ in enumerate(expected):
        expected[i] = -data[i] / (model[i] + EPS) ** 2 + 1 / (model[i] + EPS)
    assert np.array_equal(result, expected)


def test_huber(sample_data_model):
    data, model = sample_data_model
    result = handles.huber(data, model, threshold=0.1)
    # TODO maybe just use scipy and use manual to verify?
    expected = 2 * scipy.special.huber(0.1, model - data)
    # Scipy seems to do something clever leading to imperfect match
    assert np.allclose(result, expected), f"Expected: {expected}\n Got: {result}"


def test_huber_grad(sample_data_model):
    data, model = sample_data_model
    threshold = 0.1
    result = handles.huber_grad(data, model, threshold=threshold)
    expected = -2 * (data - model) * (np.abs(data - model) < threshold) - (
        (2 * threshold * np.sign(data - model)) * (np.abs(data - model) >= threshold)
    )
    assert np.array_equal(result, expected), f"Expected: {expected}\n Got: {result}"


def test_negative_binomial(sample_data_model):
    data, model = sample_data_model
    num_trials = 5
    result = handles.negative_binomial(data, model, num_trials)
    expected = np.zeros_like(result)
    for i, _ in enumerate(expected):
        expected[i] = (num_trials + data[i]) * log(model[i] + 1) - data[i] * log(
            model[i] + EPS
        )
    assert np.array_equal(result, expected)


def test_negative_binomial_grad(sample_data_model):
    data, model = sample_data_model
    num_trials = 5
    result = handles.negative_binomial_grad(data, model, num_trials)
    expected = np.zeros_like(result)
    for i, _ in enumerate(expected):
        expected[i] = (num_trials + 1) / (model[i] + 1) - data[i] / (model[i] + EPS)
    assert np.array_equal(result, expected)


def test_beta(sample_data_model):
    data, model = sample_data_model
    beta = 0.5
    result = handles.beta(data, model, beta)
    expected = np.zeros_like(result)
    for i, _ in enumerate(expected):
        expected[i] = (1 / beta) * (model[i] + EPS) ** beta - (1 / (beta - 1)) * data[
            i
        ] * (model[i] + EPS) ** (beta - 1)
    assert np.array_equal(result, expected)


def test_beta_grad(sample_data_model):
    data, model = sample_data_model
    beta = 0.5
    result = handles.beta_grad(data, model, beta)
    expected = np.zeros_like(result)
    for i, _ in enumerate(expected):
        expected[i] = (model[i] + EPS) ** (beta - 1) - data[i] * (model[i] + EPS) ** (
            beta - 2
        )
    assert np.array_equal(result, expected)
