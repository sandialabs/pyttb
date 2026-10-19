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
    """Valid objective functions for GCP.

    * ``GAUSSIAN``: Gaussian (least squares) loss, for real-valued data.
    * ``BERNOULLI_ODDS``: Bernoulli loss with the odds link, for binary data.
    * ``BERNOULLI_LOGIT``: Bernoulli loss with the logit link, for binary data.
    * ``POISSON``: Poisson loss with the identity link, for nonnegative count data.
    * ``POISSON_LOG``: Poisson loss with the log link, for count data.
    * ``RAYLEIGH``: Rayleigh loss, for positive continuous data.
    * ``GAMMA``: Gamma loss, for positive continuous data.
    * ``HUBER``: Huber loss, for real-valued data with outliers; requires a
      ``threshold``.
    * ``NEGATIVE_BINOMIAL``: Negative binomial loss, for overdispersed count data;
      requires ``num_trials``.
    * ``BETA``: Beta divergence loss; requires the exponent ``b``.
    * ``ZT_POISSON``: Zero-truncated Poisson loss, for count data with no zeros.
    """

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
    """Return objective function for gaussian distributions.

    Evaluated elementwise as::

        (model - data) ** 2

    Parameters
    ----------
    data:
        Data values.
    model:
        Model values for the same entries.

    Returns
    -------
    Elementwise objective values, with the same shape as ``data``.
    """
    return (model - data) ** 2


def gaussian_grad(data: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Return gradient function for gaussian distributions.

    Evaluated elementwise as::

        2 * (model - data)

    Parameters
    ----------
    data:
        Data values.
    model:
        Model values for the same entries.

    Returns
    -------
    Elementwise gradient values, with the same shape as ``data``.
    """
    return 2 * (model - data)


def bernoulli_odds(data: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Return objective function for bernoulli distributions.

    Evaluated elementwise as::

        log(model + 1) - data * log(model + EPS)

    Parameters
    ----------
    data:
        Data values.
    model:
        Model values for the same entries.

    Returns
    -------
    Elementwise objective values, with the same shape as ``data``.
    """
    return np.log(model + 1) - data * np.log(model + EPS)


def bernoulli_odds_grad(data: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Return gradient function for bernoulli distributions.

    Evaluated elementwise as::

        1 / (model + 1) - data / (model + EPS)

    Parameters
    ----------
    data:
        Data values.
    model:
        Model values for the same entries.

    Returns
    -------
    Elementwise gradient values, with the same shape as ``data``.
    """
    return 1.0 / (model + 1) - data / (model + EPS)


def bernoulli_logit(data: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Return objective function for bernoulli logit distributions.

    Evaluated elementwise as::

        log(exp(model) + 1) - data * model

    Parameters
    ----------
    data:
        Data values.
    model:
        Model values for the same entries.

    Returns
    -------
    Elementwise objective values, with the same shape as ``data``.
    """
    return np.log(np.exp(model) + 1) - data * model


def bernoulli_logit_grad(data: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Return gradient function for bernoulli logit distributions.

    Evaluated elementwise as::

        exp(model) / (exp(model) + 1) - data

    Parameters
    ----------
    data:
        Data values.
    model:
        Model values for the same entries.

    Returns
    -------
    Elementwise gradient values, with the same shape as ``data``.
    """
    return np.exp(model) / (np.exp(model) + 1) - data


def poisson(data: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Return objective function for poisson distributions.

    Evaluated elementwise as::

        model - data * log(model + EPS)

    Parameters
    ----------
    data:
        Data values.
    model:
        Model values for the same entries.

    Returns
    -------
    Elementwise objective values, with the same shape as ``data``.
    """
    return model - data * np.log(model + EPS)


def poisson_grad(data: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Return gradient function for poisson distributions.

    Evaluated elementwise as::

        1 - data / (model + EPS)

    Parameters
    ----------
    data:
        Data values.
    model:
        Model values for the same entries.

    Returns
    -------
    Elementwise gradient values, with the same shape as ``data``.
    """
    return 1 - data / (model + EPS)


def poisson_log(data: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Return objective function for log poisson distributions.

    Evaluated elementwise as::

        exp(model) - data * model

    Parameters
    ----------
    data:
        Data values.
    model:
        Model values for the same entries.

    Returns
    -------
    Elementwise objective values, with the same shape as ``data``.
    """
    return np.exp(model) - data * model


def poisson_log_grad(data: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Return gradient function for log poisson distributions.

    Evaluated elementwise as::

        exp(model) - data

    Parameters
    ----------
    data:
        Data values.
    model:
        Model values for the same entries.

    Returns
    -------
    Elementwise gradient values, with the same shape as ``data``.
    """
    return np.exp(model) - data


def rayleigh(data: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Return objective function for rayleigh distributions.

    Evaluated elementwise as::

        2 * log(model + EPS) + (pi / 4) * (data / (model + EPS)) ** 2

    Parameters
    ----------
    data:
        Data values.
    model:
        Model values for the same entries.

    Returns
    -------
    Elementwise objective values, with the same shape as ``data``.
    """
    return 2 * np.log(model + EPS) + (np.pi / 4) * (data / (model + EPS)) ** 2


def rayleigh_grad(data: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Return gradient function for rayleigh distributions.

    Evaluated elementwise as::

        2 / (model + EPS) - (pi / 2) * data**2 / (model + EPS) ** 3

    Parameters
    ----------
    data:
        Data values.
    model:
        Model values for the same entries.

    Returns
    -------
    Elementwise gradient values, with the same shape as ``data``.
    """
    return 2 / (model + EPS) - (np.pi / 2) * data**2 / (model + EPS) ** 3


def gamma(data: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Return objective function for gamma distributions.

    Evaluated elementwise as::

        data / (model + EPS) + log(model + EPS)

    Parameters
    ----------
    data:
        Data values.
    model:
        Model values for the same entries.

    Returns
    -------
    Elementwise objective values, with the same shape as ``data``.
    """
    return data / (model + EPS) + np.log(model + EPS)


def gamma_grad(data: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Return gradient function for gamma distributions.

    Evaluated elementwise as::

        -data / (model + EPS) ** 2 + 1 / (model + EPS)

    Parameters
    ----------
    data:
        Data values.
    model:
        Model values for the same entries.

    Returns
    -------
    Elementwise gradient values, with the same shape as ``data``.
    """
    return -data / (model + EPS) ** 2 + 1 / (model + EPS)


def huber(data: ttb.tensor, model: ttb.tensor, threshold: float) -> np.ndarray:
    """Return objective function for huber loss.

    Evaluated elementwise as::

        d**2                               if d < threshold
        2 * threshold * d - threshold**2   otherwise

    where ``d = |data - model|``.

    Parameters
    ----------
    data:
        Data values.
    model:
        Model values for the same entries.
    threshold:
        Threshold on the absolute difference between data and model at which the
        loss changes from quadratic to linear.

    Returns
    -------
    Elementwise objective values, with the same shape as ``data``.
    """
    abs_diff = np.abs(data - model)
    below_threshold = abs_diff < threshold
    return abs_diff**2 * below_threshold + (
        2 * threshold * abs_diff - threshold**2
    ) * np.logical_not(below_threshold)


def huber_grad(data: ttb.tensor, model: ttb.tensor, threshold: float) -> np.ndarray:
    """Return gradient function for huber loss.

    Evaluated elementwise as::

        -2 * (data - model)                   if |data - model| < threshold
        -2 * threshold * sign(data - model)   otherwise

    Parameters
    ----------
    data:
        Data values.
    model:
        Model values for the same entries.
    threshold:
        Threshold on the absolute difference between data and model at which the
        loss changes from quadratic to linear.

    Returns
    -------
    Elementwise gradient values, with the same shape as ``data``.
    """
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
    """Return objective function for negative binomial distributions.

    Evaluated elementwise as::

        (num_trials + data) * log(model + 1) - data * log(model + EPS)

    Parameters
    ----------
    data:
        Data values.
    model:
        Model values for the same entries.
    num_trials:
        Number of trials of the negative binomial distribution.

    Returns
    -------
    Elementwise objective values, with the same shape as ``data``.
    """
    return (num_trials + data) * np.log(model + 1) - data * np.log(model + EPS)


def negative_binomial_grad(
    data: np.ndarray, model: np.ndarray, num_trials: float
) -> np.ndarray:
    """Return gradient function for negative binomial distributions.

    Evaluated elementwise as::

        (num_trials + 1) / (1 + model) - data / (model + EPS)

    Parameters
    ----------
    data:
        Data values.
    model:
        Model values for the same entries.
    num_trials:
        Number of trials of the negative binomial distribution.

    Returns
    -------
    Elementwise gradient values, with the same shape as ``data``.
    """
    return (num_trials + 1) / (1 + model) - data / (model + EPS)


def beta(data: np.ndarray, model: np.ndarray, b: float) -> np.ndarray:
    """Return objective function for beta distributions.

    Evaluated elementwise as::

        (1 / b) * (model + EPS) ** b - (1 / (b - 1)) * data * (model + EPS) ** (b - 1)

    Parameters
    ----------
    data:
        Data values.
    model:
        Model values for the same entries.
    b:
        Exponent of the beta divergence (``b`` is neither 0 nor 1).

    Returns
    -------
    Elementwise objective values, with the same shape as ``data``.
    """
    return (1 / b) * (model + EPS) ** b - (1 / (b - 1)) * data * (model + EPS) ** (
        b - 1
    )


def beta_grad(data: np.ndarray, model: np.ndarray, b: float) -> np.ndarray:
    """Return gradient function for beta distributions.

    Evaluated elementwise as::

        (model + EPS) ** (b - 1) - data * (model + EPS) ** (b - 2)

    Parameters
    ----------
    data:
        Data values.
    model:
        Model values for the same entries.
    b:
        Exponent of the beta divergence (``b`` is neither 0 nor 1).

    Returns
    -------
    Elementwise gradient values, with the same shape as ``data``.
    """
    return (model + EPS) ** (b - 1) - data * (model + EPS) ** (b - 2)


def ztp(data: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Return objective function for zero-truncated poisson distributions.

    Evaluated elementwise as::

        poisson(data, model) + log(1 - exp(-model) + EPS)

    Parameters
    ----------
    data:
        Data values.
    model:
        Model values for the same entries.

    Returns
    -------
    Elementwise objective values, with the same shape as ``data``.
    """
    return poisson(data, model) + np.log(1 - np.exp(-model) + EPS)


def ztp_grad(data: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Return gradient function for zero-truncated poisson distributions.

    Evaluated elementwise as::

        poisson_grad(data, model) + 1 / (exp(model) - 1 + EPS)

    Parameters
    ----------
    data:
        Data values.
    model:
        Model values for the same entries.

    Returns
    -------
    Elementwise gradient values, with the same shape as ``data``.
    """
    return poisson_grad(data, model) + 1 / ((np.exp(model) - 1) + EPS)
