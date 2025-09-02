"""Implementation of the different function and gradient handles for GCP OPT."""

# Copyright 2025 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pyttb as ttb

# Epsilon values for distributions
EPS = 1e-10


class Objectives(Enum):
    """Valid objective functions for GCP."""

    GAUSSIAN = 0
    BERNOULLI_ODDS = 1
    BERNOULLI_LOGIT = 2
    POISSON = 3
    POISSON_LOG = 4
    RAYLEIGH = 5
    GAMMA = 6
    HUBER = 7
    NEGATIVE_BINOMIAL = 8
    BETA = 9
    ZT_POISSON = 10


def gaussian(data: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Return objective function for gaussian distributions."""
    return (model - data) ** 2


def gaussian_grad(data: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Return gradient function for gaussian distributions."""
    return 2 * (model - data)


def bernoulli_odds(data: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Return objective function for bernoulli distributions."""
    return np.log(model + 1) - data * np.log(model + EPS)


def bernoulli_odds_grad(data: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Return gradient function for bernoulli distributions."""
    return 1.0 / (model + 1) - data / (model + EPS)


def bernoulli_logit(data: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Return objective function for bernoulli logit distributions."""
    return np.log(np.exp(model) + 1) - data * model


def bernoulli_logit_grad(data: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Return gradient function for bernoulli logit distributions."""
    return np.exp(model) / (np.exp(model) + 1) - data


def poisson(data: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Return objective function for poisson distributions."""
    return model - data * np.log(model + EPS)


def poisson_grad(data: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Return gradient function for poisson distributions."""
    return 1 - data / (model + EPS)


def poisson_log(data: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Return objective function for log poisson distributions."""
    return np.exp(model) - data * model


def poisson_log_grad(data: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Return gradient function for log poisson distributions."""
    return np.exp(model) - data


def rayleigh(data: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Return objective function for rayleigh distributions."""
    return 2 * np.log(model + EPS) + (np.pi / 4) * (data / (model + EPS)) ** 2


def rayleigh_grad(data: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Return gradient function for rayleigh distributions."""
    return 2 / (model + EPS) - (np.pi / 2) * data**2 / (model + EPS) ** 3


def gamma(data: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Return objective function for gamma distributions."""
    return data / (model + EPS) + np.log(model + EPS)


def gamma_grad(data: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Return gradient function for gamma distributions."""
    return -data / (model + EPS) ** 2 + 1 / (model + EPS)


def huber(data: ttb.tensor, model: ttb.tensor, threshold: float) -> np.ndarray:
    """Return objective function for huber loss."""
    abs_diff = np.abs(data - model)
    below_threshold = abs_diff < threshold
    return abs_diff**2 * below_threshold + (
        2 * threshold * abs_diff - threshold**2
    ) * np.logical_not(below_threshold)


def huber_grad(data: ttb.tensor, model: ttb.tensor, threshold: float) -> np.ndarray:
    """Return gradient function for huber loss."""
    abs_diff = np.abs(data - model)
    below_threshold = abs_diff < threshold
    return -2 * (data - model) * below_threshold - (
        2 * threshold * np.sign(data - model)
    ) * np.logical_not(below_threshold)


# FIXME: Num trials should be enforced as integer here and in MATLAB
# requires updating our regression test values to calculate MATLAB integer version
def negative_binomial(
    data: np.ndarray, model: np.ndarray, num_trials: float
) -> np.ndarray:
    """Return objective function for negative binomial distributions."""
    return (num_trials + data) * np.log(model + 1) - data * np.log(model + EPS)


def negative_binomial_grad(
    data: np.ndarray, model: np.ndarray, num_trials: float
) -> np.ndarray:
    """Return gradient function for negative binomial distributions."""
    return (num_trials + 1) / (1 + model) - data / (model + EPS)


def beta(data: np.ndarray, model: np.ndarray, b: float) -> np.ndarray:
    """Return objective function for beta distributions."""
    return (1 / b) * (model + EPS) ** b - (1 / (b - 1)) * data * (model + EPS) ** (
        b - 1
    )


def beta_grad(data: np.ndarray, model: np.ndarray, b: float) -> np.ndarray:
    """Return gradient function for beta distributions."""
    return (model + EPS) ** (b - 1) - data * (model + EPS) ** (b - 2)


def ztp(data: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Return objective function for zero-truncated poisson distributions."""
    return poisson(data, model) + np.log(1 - np.exp(-model) + EPS)


def ztp_grad(data: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Return gradient function for zero-truncated poisson distributions."""
    return poisson_grad(data, model) + 1 / ((np.exp(model) - 1) + EPS)
