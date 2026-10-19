"""Evaluate Function And Gradient Handles."""

# Copyright 2025 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import numpy as np

import pyttb as ttb

if TYPE_CHECKING:
    from pyttb.gcp.fg_setup import function_type


@overload
def evaluate(
    model: ttb.ktensor,
    data: ttb.tensor | ttb.sptensor,
    weights: np.ndarray | None,
    function_handle: Literal[None],
    gradient_handle: function_type,
) -> list[np.ndarray]: ...  # pragma: no cover see coveragepy/issues/970


@overload
def evaluate(
    model: ttb.ktensor,
    data: ttb.tensor | ttb.sptensor,
    weights: np.ndarray | None,
    function_handle: function_type,
    gradient_handle: Literal[None],
) -> float: ...  # pragma: no cover see coveragepy/issues/970


@overload
def evaluate(
    model: ttb.ktensor,
    data: ttb.tensor | ttb.sptensor,
    weights: np.ndarray | None,
    function_handle: function_type,
    gradient_handle: function_type,
) -> tuple[float, list[np.ndarray]]: ...  # pragma: no cover see coveragepy/issues/970


def evaluate(
    model: ttb.ktensor,
    data: ttb.tensor | ttb.sptensor,
    weights: np.ndarray | None = None,
    function_handle: function_type | None = None,
    gradient_handle: function_type | None = None,
) -> float | list[np.ndarray] | tuple[float, list[np.ndarray]]:
    """Evaluate an objective function and/or gradient function.

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
    Objective function value if only ``function_handle`` is given, gradient (a list
    with one array per factor matrix) if only ``gradient_handle`` is given, or a
    tuple of both if both are given.

    Examples
    --------
    Evaluate the Gaussian objective and gradient of a rank-1 model of a tensor of
    ones::

        >>> from pyttb.gcp.fg_setup import Objectives, setup
        >>> function_handle, gradient_handle, _ = setup(Objectives.GAUSSIAN)
        >>> X = ttb.tenones((2, 2, 2))
        >>> M = ttb.ktensor([np.full((2, 1), 2.0)] * 3)
        >>> F, G = evaluate(M, X, None, function_handle, gradient_handle)
        >>> float(F)
        392.0
        >>> [g.shape for g in G]
        [(2, 1), (2, 1), (2, 1)]
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
    F: float | None = None
    G: list[np.ndarray] | None = None
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
