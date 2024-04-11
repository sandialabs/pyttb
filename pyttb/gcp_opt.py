"""Generalized CP Decomposition"""

from __future__ import annotations

import logging
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np

import pyttb as ttb
from pyttb.gcp.fg_setup import function_type, setup
from pyttb.gcp.handles import Objectives
from pyttb.gcp.optimizers import LBFGSB, StochasticSolver
from pyttb.gcp.samplers import GCPSampler


def gcp_opt(  # noqa:  PLR0912,PLR0913
    data: Union[ttb.tensor, ttb.sptensor],
    rank: int,
    objective: Union[Objectives, Tuple[function_type, function_type, float]],
    optimizer: Union[StochasticSolver, LBFGSB],
    init: Union[Literal["random"], ttb.ktensor, List[np.ndarray]] = "random",
    mask: Optional[Union[ttb.tensor, np.ndarray]] = None,
    sampler: Optional[GCPSampler] = None,
    printitn: int = 1,
) -> Tuple[ttb.ktensor, ttb.ktensor, Dict]:
    """Fits Generalized CP decomposition with user-specified function.

    Parameters
    ----------
    data:
        Tensor to decompose.
    rank:
        Rank of desired CP decomposition.
    objective:
        Objective function to minimize for the CP decomposition. Either a pre-defined
        objective or a tuple of function_handle, gradient_handle, and lower_bound.
    optimizer:
        Optimizer class for solving the decompistion problem defined.
    init:
        Initial solution to the problem.
    mask:
        A binary mask to note missing rather than sparse data.
        (Only valid for dense, LBFGSB solves)
    sampler:
        Class that defined sampling strategy for stochastic solves.
    printitn:
        Controls verbosity of printing throughout the solve

    Returns
    -------
        Solution, Initial Guess, Dictionary of meta data
    """
    if not isinstance(objective, Objectives):
        # TODO probably do some runtime type validation here to make
        #  sure tuple is correct
        if len(objective) != 3:
            raise ValueError(
                "Objective must either be an Objectives enum or a tuple containing a "
                "function handle, gradient_handle and lower bound."
            )

    if isinstance(objective, Objectives):
        # TODO not clear how to pass in other params to setup for ex huber
        function_handle, gradient_handle, lower_bound = setup(objective, data)
    else:
        function_handle, gradient_handle, lower_bound = objective

    if not isinstance(data, (ttb.tensor, ttb.sptensor)):
        raise ValueError("Input data must be tensor or sptensor.")

    tensor_size = int(np.prod(data.shape))

    if isinstance(data, ttb.tensor) and isinstance(mask, ttb.tensor):
        data *= mask
        nmissing = tensor_size - mask.nnz
    elif isinstance(data, ttb.sptensor) and mask is not None:
        raise ValueError("Cannot specify missing entries for sparse tensors")
    else:
        nmissing = 0

    # Create initial guess
    M0 = _get_initial_guess(data, rank, init)

    if not isinstance(optimizer, (StochasticSolver, LBFGSB)):
        raise ValueError("Must select a supported optimizer.")

    if isinstance(data, ttb.sptensor) and isinstance(optimizer, LBFGSB):
        raise ValueError("For sparse tensor must use: ADAM, SGD, or ADAGRAD.")

    if isinstance(optimizer, StochasticSolver) and mask is not None:
        raise ValueError("Mask isn't supported for stochastic solves")

    # Welcome Message
    if printitn > 0:
        # TODO capture full verbosity from MATLAB
        optimizer_name = type(optimizer).__name__
        objective_name = "user-provided"
        if isinstance(objective, Objectives):
            objective_name = objective.name
        welcome_msg = (
            f"\nGCP-OPT-{optimizer_name} (Generalized CP Tensor Decomposition)\n"
            f"\nTensor shape: {data.shape} ({tensor_size} total entries)\n"
            f"GCP rank: {rank}\nGeneralized function type: {objective_name}"
        )
        if nmissing > 0:
            welcome_msg += (
                f"Missing entries: {nmissing} ({100*nmissing/tensor_size:.2g}%)"
            )
        logging.info(welcome_msg)

    if isinstance(optimizer, StochasticSolver):
        result, info = optimizer.solve(
            M0, data, function_handle, gradient_handle, lower_bound, sampler
        )
    else:
        if isinstance(mask, ttb.tensor):
            mask = mask.data
        assert isinstance(data, ttb.tensor)
        result, info = optimizer.solve(
            M0, data, function_handle, gradient_handle, lower_bound, mask
        )
    return result, M0, info


def _get_initial_guess(
    data: Union[ttb.tensor, ttb.sptensor],
    rank: int,
    init: Union[Literal["random"], ttb.ktensor, List[np.ndarray]],
) -> ttb.ktensor:
    """Get initial guess for gcp_opt

    Returns
    -------
        Normalized ktensor.
    """
    # TODO might be nice to merge with ALS/other CP methods
    if isinstance(init, list):
        return ttb.ktensor(init).normalize("all")
    if isinstance(init, ttb.ktensor):
        init.normalize("all")
        return init
    if init == "random":
        factor_matrices = []
        for n in range(data.ndims):
            factor_matrices.append(np.random.uniform(0, 1, (data.shape[n], rank)))
        M0 = ttb.ktensor(factor_matrices)
        M0 *= data.norm() / M0.norm()
        M0.normalize("all")
        return M0
    raise ValueError(f"Unexpected input for init received: {init}")
