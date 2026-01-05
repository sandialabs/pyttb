"""Prepare Function and Gradient Handles for GCP OPT."""

# Copyright 2025 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import numpy as np

import pyttb as ttb
from pyttb.gcp import handles
from pyttb.gcp.handles import Objectives

function_type = Callable[[np.ndarray, np.ndarray], np.ndarray]
fg_return = tuple[function_type, function_type, float]


def setup(  # noqa: PLR0912,PLR0915
    objective: Objectives,
    data: ttb.tensor | ttb.sptensor | None = None,
    additional_parameter: float | None = None,
) -> fg_return:
    """Collect the function and gradient handles for GCP.

    Parameters
    ----------
    objective:
        Objective function to gather handles for.
    data:
        Tensor to check for consistency with desired objective function.
    additional_parameter:
        Additional constant argument provided to objective function if necessary.

    Returns
    -------
    Function handle, gradient handle, and lower bound.
    """
    if objective == Objectives.GAUSSIAN:
        function_handle = handles.gaussian
        gradient_handle = handles.gaussian_grad
        lower_bound = -np.inf
    elif objective == Objectives.BERNOULLI_ODDS:
        if data is not None and not valid_binary(data):
            raise ValueError(f"{objective.name} requires a binary tensor")
        function_handle = handles.bernoulli_odds
        gradient_handle = handles.bernoulli_odds_grad
        lower_bound = 0.0
    elif objective == Objectives.BERNOULLI_LOGIT:
        if data is not None and not valid_binary(data):
            raise ValueError(f"{objective.name} requires a binary tensor")
        function_handle = handles.bernoulli_logit
        gradient_handle = handles.bernoulli_logit_grad
        lower_bound = -np.inf
    elif objective == Objectives.POISSON:
        if data is not None and not valid_natural(data):
            raise ValueError(f"{objective.name} requires a count tensor")
        function_handle = handles.poisson
        gradient_handle = handles.poisson_grad
        lower_bound = 0.0
    elif objective == Objectives.POISSON_LOG:
        if data is not None and not valid_natural(data):
            raise ValueError(f"{objective.name} requires a count tensor")
        function_handle = handles.poisson_log
        gradient_handle = handles.poisson_log_grad
        lower_bound = -np.inf
    elif objective == Objectives.RAYLEIGH:
        if data is not None and not valid_nonneg(data):
            raise ValueError(f"{objective.name} requires a non-negative tensor")
        function_handle = handles.rayleigh
        gradient_handle = handles.rayleigh_grad
        lower_bound = 0.0
    elif objective == Objectives.GAMMA:
        if data is not None and not valid_nonneg(data):
            raise ValueError(f"{objective.name} requires a non-negative tensor")
        function_handle = handles.gamma
        gradient_handle = handles.gamma_grad
        lower_bound = 0.0
    elif objective == Objectives.HUBER:
        if additional_parameter is None:
            raise ValueError(
                f"{objective.name} requires additional parameter for `threshold`"
            )
        function_handle = partial(handles.huber, threshold=additional_parameter)
        gradient_handle = partial(handles.huber_grad, threshold=additional_parameter)
        lower_bound = -np.inf
    elif objective == Objectives.NEGATIVE_BINOMIAL:
        if data is not None and not valid_nonneg(data):
            raise ValueError(f"{objective.name} requires a non-negative tensor")
        if additional_parameter is None:
            raise ValueError(
                f"{objective.name} requires additional parameter for `num_trials`"
            )
        function_handle = partial(
            handles.negative_binomial, num_trials=additional_parameter
        )
        gradient_handle = partial(
            handles.negative_binomial_grad, num_trials=additional_parameter
        )
        lower_bound = 0
    elif objective == Objectives.BETA:
        if data is not None and not valid_nonneg(data):
            raise ValueError(f"{objective.name} requires a non-negative tensor")
        if additional_parameter is None:
            raise ValueError(f"{objective.name} requires additional parameter for `b`")
        function_handle = partial(handles.beta, b=additional_parameter)
        gradient_handle = partial(handles.beta_grad, b=additional_parameter)
        lower_bound = 0
    elif objective == Objectives.ZT_POISSON:
        if data is not None and not valid_natural(data):
            raise ValueError(f"{objective.name} requires a count tensor")
        function_handle = handles.ztp
        gradient_handle = handles.ztp_grad
        lower_bound = 0.0
    else:
        raise ValueError(f" Unknown objective: {objective}")

    return function_handle, gradient_handle, lower_bound


def valid_nonneg(data: ttb.tensor | ttb.sptensor) -> bool:
    """Check if provided data is valid non-negative tensor."""
    if isinstance(data, ttb.sptensor):
        return bool(np.all(data.vals > 0))
    return bool(np.all(data.data > 0))


def valid_binary(data: ttb.tensor | ttb.sptensor) -> bool:
    """Check if provided data is valid binary tensor."""
    if isinstance(data, ttb.sptensor):
        return bool(np.all(data.vals == 1))
    return bool(np.all(np.isin(np.unique(data.data), [0, 1])))


def valid_natural(data: ttb.tensor | ttb.sptensor) -> bool:
    """Check if provided data is valid natural number tensor."""
    if isinstance(data, ttb.sptensor):
        vals = data.vals
    else:
        vals = data.data
    return bool(np.all(vals % 1 == 0))
