"""Evaluate Function And Gradient Handles"""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple, Union, overload

import numpy as np

import pyttb as ttb
from pyttb.gcp.fg_setup import function_type


@overload
def evaluate(
    model: ttb.ktensor,
    data: Union[ttb.tensor, ttb.sptensor],
    weights: Optional[np.ndarray],
    function_handle: Literal[None],
    gradient_handle: function_type,
) -> List[np.ndarray]: ...  # pragma: no cover see coveragepy/issues/970


@overload
def evaluate(
    model: ttb.ktensor,
    data: Union[ttb.tensor, ttb.sptensor],
    weights: Optional[np.ndarray],
    function_handle: function_type,
    gradient_handle: Literal[None],
) -> float: ...  # pragma: no cover see coveragepy/issues/970


@overload
def evaluate(
    model: ttb.ktensor,
    data: Union[ttb.tensor, ttb.sptensor],
    weights: Optional[np.ndarray],
    function_handle: function_type,
    gradient_handle: function_type,
) -> Tuple[float, List[np.ndarray]]: ...  # pragma: no cover see coveragepy/issues/970


def evaluate(
    model: ttb.ktensor,
    data: Union[ttb.tensor, ttb.sptensor],
    weights: Optional[np.ndarray] = None,
    function_handle: Optional[function_type] = None,
    gradient_handle: Optional[function_type] = None,
) -> Union[float, List[np.ndarray], Tuple[float, List[np.ndarray]]]:
    """Evaluate an objective function and/or gradient function

    Parameters
    ----------
    model:
        Current decomposition.
    data:
        Source tensor to decompose.
    weights:
        Weighted values for returned tensor. Can be used as a mask.
    function_handle:
        Objective function.
    gradient_handle:
        Gradient definition.

    Returns
    -------
    Objective function value and/or gradient function value with respect to model.
    """
    if function_handle is None and gradient_handle is None:
        raise ValueError(
            "Either a function handle, or a gradient handle must be provided."
        )

    if isinstance(data, ttb.sptensor):
        data = data.full()

    full_model = model.full()
    # TODO should we early check shapes?
    # I don't think we always get vectorization for free in python
    # we should be able to operate on underlying np arrays directly though
    F: Optional[float] = None
    G: Optional[List[np.ndarray]] = None
    if function_handle is not None:
        Y = function_handle(data.data, full_model.data)
        if weights is not None:
            Y *= weights
        F = float(np.sum(Y))

    if gradient_handle is not None:
        Y = gradient_handle(data.data, full_model.data)
        if weights is not None:
            Y *= weights
        G = ttb.tensor(Y, copy=False).mttkrps(model.factor_matrices)

    if F is not None and G is not None:
        return F, G
    if F is not None:
        return F
    if G is not None:
        return G
    raise ValueError(
        "No valid outputs for either function or gradient handles"
    )  # pragma: no cover
