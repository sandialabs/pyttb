"""Evaluate Functions And Gradients based on Subsamples."""

# Copyright 2025 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
from scipy.sparse import csr_array

if TYPE_CHECKING:
    import pyttb as ttb
    from pyttb.gcp.fg_setup import function_type


@overload
def estimate(
    model: ttb.ktensor,
    data_subs: np.ndarray,
    data_vals: np.ndarray,
    weights: np.ndarray,
    function_handle: Literal[None],
    gradient_handle: function_type,
    lambda_check: bool = True,
    crng: np.ndarray | None = None,
) -> list[np.ndarray]: ...  # pragma: no cover see coveragepy/issues/970


@overload
def estimate(
    model: ttb.ktensor,
    data_subs: np.ndarray,
    data_vals: np.ndarray,
    weights: np.ndarray,
    function_handle: function_type,
    gradient_handle: Literal[None] = None,
    lambda_check: bool = False,
    crng: np.ndarray | None = None,
) -> float: ...  # pragma: no cover see coveragepy/issues/970


@overload
def estimate(
    model: ttb.ktensor,
    data_subs: np.ndarray,
    data_vals: np.ndarray,
    weights: np.ndarray,
    function_handle: function_type,
    gradient_handle: function_type,
    lambda_check: bool,
    crng: np.ndarray | None,
) -> tuple[float, list[np.ndarray]]: ...  # pragma: no cover see coveragepy/issues/970


def estimate(  # noqa: PLR0913
    model: ttb.ktensor,
    data_subs: np.ndarray,
    data_vals: np.ndarray,
    weights: np.ndarray,
    function_handle: function_type | None = None,
    gradient_handle: function_type | None = None,
    lambda_check: bool = True,
    crng: np.ndarray | None = None,
) -> float | list[np.ndarray] | tuple[float, list[np.ndarray]]:
    """Estimate the GCP function and gradient with a subsample.

    Parameters
    ----------
    model:
        Current decomposition.
    data_subs:
        Subscripts of data sample.
    data_vals:
        Values of data sample.
    function_handle:
        Handle to evaluate objective function.
    gradient_handle:
        Handle to evaluate gradient of objective function.
    lambda_check:
        Whether or not to check decomposition weights are all ones.
        (Which is assumed in implementation details)
    crng:
        Range of indices for correct/adjustment when zeros are sampled accidentally.

    Returns
    -------
        Estimated objective function value and/or estimated gradient value with
        respect to the model.
    """
    if function_handle is None and gradient_handle is None:
        raise ValueError(
            "Either a function handle, or a gradient handle must be provided."
        )

    if lambda_check and any(model.weights != 1.0):
        warnings.warn("Normalizing model to have all 1's for weights")
        model = model.normalize(0)
    model_vals, Zexp = estimate_helper(model.factor_matrices, data_subs)

    F: float | None = None
    G: list[np.ndarray] | None = None

    if function_handle is not None:
        Y = function_handle(data_vals, model_vals)
        if crng is not None:
            Y[crng] -= function_handle(np.zeros_like(crng), model_vals[crng])
        F = np.sum(weights * Y)

    if gradient_handle is not None:
        # Compute sample y values
        Y = weights * gradient_handle(data_vals, model_vals)
        if crng is not None:
            Y[crng] -= weights[crng] * gradient_handle(
                np.zeros_like(crng), model_vals[crng]
            )
        G = [np.empty(())] * model.ndims
        nsamples = data_subs.shape[0]
        for k in range(model.ndims):
            # The row of each element is the row index to accumulate in the gradient.
            # The columns are the corresponding samples. They are in order because they
            # match the vector of samples to be multiplied on the right.
            S = csr_array(
                (Y, (data_subs[:, k], np.arange(nsamples))),
                shape=(model.shape[k], nsamples),
            )
            G[k] = S.dot(Zexp[k])

    if F is not None and G is not None:
        return F, G
    if F is not None:
        return F
    if G is not None:
        return G
    raise ValueError(
        "No valid outputs for either function or gradient handles"
    )  # pragma: no cover


def estimate_helper(
    factors: list[np.ndarray], subs: np.ndarray
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Extract model values at sample locations and exploded Zk's.

    Parameters
    ----------
    factors:
        Factor matrices from model.
    subs:
        Subscripts to extract from model.

    Returns
    -------
        Model values at subs and exploded Zk's
    """
    Zexp: list[np.ndarray] = []
    if subs.size == 0:
        return np.array([]), Zexp

    ndim = subs.shape[1]

    # Create exploded U's from the model factor matrices
    Uexp = [np.empty((), dtype=factors[0].dtype)] * ndim
    for k in range(ndim):
        Uexp[k] = factors[k][subs[:, k], :]

    # After this pass, Zexp[k] = Hadarmard product of Uexp[0] through
    # Uexp[k-1] for k = 1,...,ndim
    Zexp = [np.empty(())] * ndim
    Zexp[1] = Uexp[0].copy("K")
    for k in range(2, ndim):
        Zexp[k] = Zexp[k - 1] * Uexp[k - 1]

    # After this pass, Zexp[k] = Hadarmard product of Uexcp[0] through
    # Uexp[d], except Uexp[k] for k = 0, ..., ndim
    Zexp[0] = Uexp[ndim - 1].copy("K")
    for k in range(ndim - 2, 0, -1):
        Zexp[k] *= Zexp[0]
        Zexp[0] *= Uexp[k]

    mvals = np.sum(Zexp[ndim - 1] * Uexp[ndim - 1], axis=1)
    return mvals, Zexp
