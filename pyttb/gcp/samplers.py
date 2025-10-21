"""Implementation of various sampling approaches for GCP OPT."""

# Copyright 2025 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import partial
from math import ceil
from typing import cast

import numpy as np

import pyttb as ttb
from pyttb.pyttb_utils import tt_sub2ind
from pyttb.sptensor import sptensor
from pyttb.tensor import tensor

sample_type = tuple[np.ndarray, np.ndarray, np.ndarray]
sampler_type = Callable[[tensor | sptensor], sample_type]


@dataclass
class StratifiedCount:
    """Contains stratified sampling counts."""

    num_zeros: int
    num_nonzeros: int


class Samplers(Enum):
    """Implemented Samplers."""

    UNIFORM = 0
    SEMISTRATIFIED = 1
    STRATIFIED = 2


class GCPSampler:
    """Contains Gradient and Function Sampling Details."""

    def __init__(  # noqa: PLR0913
        self,
        data: ttb.tensor | ttb.sptensor,
        function_sampler: Samplers | None = None,
        function_samples: int | StratifiedCount | None = None,
        gradient_sampler: Samplers | None = None,
        gradient_samples: int | StratifiedCount | None = None,
        max_iters: int = 1000,
        over_sample_rate: float = 1.1,
    ):
        """Create sampler.

        Parameters
        ----------
        data:
            Tensor we will be sampling. Allows for automated decisions and sanity
            checks.
        function_sampler:
            Type of sampling used for evaluating function estimates.
        function_samples:
            How many samples to take of the function every iteration.
        gradient_sampler:
            Type of sampling used for evaluating gradient estimates.
        gradient_samples:
            How many samples to take of the gradient every iteration.
        max_iters:
            Maximum number of iterations to normalize number of samples.
        over_sample_rate:
            Ratio of extra samples to take to account for bad draws.
        """
        self._crng = np.array([], dtype=int)
        # TODO add interface for arbitrary callable with no args that returns ndarray
        if function_sampler is None:
            if isinstance(data, ttb.sptensor):
                function_sampler = Samplers.STRATIFIED
            else:
                function_sampler = Samplers.UNIFORM

        tensor_size = int(np.prod(data.shape))
        num_nonzeros = data.nnz
        num_zeros = tensor_size - num_nonzeros

        self._prepare_function_sampler(
            data,
            function_sampler,
            num_zeros,
            num_nonzeros,
            over_sample_rate,
            function_samples,
        )
        if gradient_sampler is None:
            if isinstance(data, ttb.sptensor):
                gradient_sampler = Samplers.STRATIFIED
            else:
                gradient_sampler = Samplers.UNIFORM
        self._prepare_gradient_sampler(
            data,
            gradient_sampler,
            num_zeros,
            num_nonzeros,
            over_sample_rate,
            gradient_samples,
            max_iters,
        )

    def _prepare_function_sampler(  # noqa: PLR0913
        self,
        data: ttb.tensor | ttb.sptensor,
        function_sampler: Samplers,
        num_zeros: int,
        num_nonzeros: int,
        over_sample_rate: float,
        function_samples: int | StratifiedCount | None,
    ):
        if function_sampler == Samplers.STRATIFIED:
            if not isinstance(data, ttb.sptensor):
                raise ValueError(
                    "Stratified sampling is only supported for sptensor data."
                )
            if function_samples is None:
                ftmp = int(max(ceil(num_nonzeros / 100), 10**5))
                function_samples = StratifiedCount(
                    num_nonzeros=min(ftmp, num_nonzeros),
                    num_zeros=min(ftmp, num_nonzeros, num_zeros),
                )
            elif isinstance(function_samples, int):
                function_samples = StratifiedCount(
                    num_nonzeros=function_samples, num_zeros=function_samples
                )
            elif not isinstance(function_samples, StratifiedCount):
                raise ValueError(
                    "Function samples should be an int or StratifiedCount but "
                    f" received: {function_samples}"
                )
            xnzidx = np.sort(tt_sub2ind(data.shape, data.subs))
            self._fsampler = partial(
                stratified,
                nz_idx=xnzidx,
                num_nonzeros=function_samples.num_nonzeros,
                num_zeros=function_samples.num_zeros,
                over_sample_rate=over_sample_rate,
            )
        elif function_sampler == Samplers.UNIFORM:
            if function_samples is None:
                tensor_size = int(np.prod(data.shape))
                function_samples = min(max(ceil(tensor_size / 10), 10**6), tensor_size)
            if not isinstance(function_samples, int):
                raise ValueError(
                    "Uniform sampling only accepts integers for number of samples"
                )
            self._fsampler = partial(
                uniform,
                samples=function_samples,
            )
        else:
            raise ValueError("Invalid choice for function_sampler")

    def _prepare_gradient_sampler(  # noqa: PLR0912,PLR0913
        self,
        data: ttb.tensor | ttb.sptensor,
        gradient_sampler: Samplers,
        num_zeros: int,
        num_nonzeros: int,
        over_sample_rate: float,
        gradient_samples: int | StratifiedCount | None,
        max_iters: int,
    ):
        if gradient_sampler in (Samplers.STRATIFIED, Samplers.SEMISTRATIFIED):
            if gradient_samples is None:
                gtmp = int(max(1000, ceil(3 * num_nonzeros / max_iters)))
                gradient_samples = StratifiedCount(
                    num_nonzeros=int(min(gtmp, num_nonzeros)),
                    num_zeros=int(min(gtmp, num_nonzeros, num_zeros)),
                )
            elif isinstance(gradient_samples, int):
                gradient_samples = StratifiedCount(
                    num_nonzeros=gradient_samples, num_zeros=gradient_samples
                )
            elif not isinstance(gradient_samples, StratifiedCount):
                raise ValueError(
                    "Gradient samples should be an int or StratifiedCount but "
                    f" received: {gradient_samples}"
                )
            if gradient_sampler == Samplers.SEMISTRATIFIED:
                self._gsampler: sampler_type = partial(
                    semistrat,
                    num_nonzeros=gradient_samples.num_nonzeros,
                    num_zeros=gradient_samples.num_zeros,
                )
                self._crng = np.arange(gradient_samples.num_nonzeros)
            else:
                if not isinstance(data, ttb.sptensor):
                    raise ValueError(
                        "Stratified sampling is only supported for sptensor data."
                    )
                # TODO store value if computed to avoid duplicate work
                xnzidx = np.sort(tt_sub2ind(data.shape, data.subs))
                self._gsampler = partial(
                    stratified,
                    nz_idx=xnzidx,
                    num_nonzeros=gradient_samples.num_nonzeros,
                    num_zeros=gradient_samples.num_zeros,
                    over_sample_rate=over_sample_rate,
                )
        elif gradient_sampler == Samplers.UNIFORM:
            tensor_size = int(np.prod(data.shape))
            if gradient_samples is None:
                gradient_samples = int(
                    min(max(1000, ceil(10 * tensor_size / max_iters)), tensor_size)
                )
            if not isinstance(gradient_samples, int):
                raise ValueError(
                    "Uniform sampling only accepts integers for number of samples"
                )
            if isinstance(data, ttb.sptensor):
                exp_nonzeros = gradient_samples * num_nonzeros / tensor_size
                exp_zeros = gradient_samples * num_zeros / tensor_size
                xnzidx = np.sort(tt_sub2ind(data.shape, data.subs))
                # NOTE: Must use lambda over partial because we need late binding,
                # every draw should first uniquely sample num_nonzeros
                self._gsampler = lambda data: stratified(
                    data=cast("ttb.sptensor", data),
                    nz_idx=xnzidx,
                    num_nonzeros=np.random.poisson(exp_nonzeros),
                    num_zeros=np.random.poisson(exp_zeros),
                    over_sample_rate=over_sample_rate,
                )
            else:
                self._gsampler = partial(uniform, samples=gradient_samples)

        else:
            raise ValueError("Invalid choice for function_sampler")

    def function_sample(self, data: ttb.tensor | ttb.sptensor) -> sample_type:
        """Draw a sample from the objective function."""
        return self._fsampler(data)

    def gradient_sample(self, data: ttb.tensor | ttb.sptensor) -> sample_type:
        """Draw a sample from the gradient function."""
        return self._gsampler(data)

    @property
    def crng(self) -> np.ndarray:
        """Correction Range for possibly miss-sampled zeros."""
        return self._crng


def nonzeros(
    data: ttb.sptensor, samples: int, with_replacement: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Sample nonzeros from a sparse tensor.

    Parameters
    ----------
    data:
        Tensor to sample.
    samples:
        Number of samples to collect
    with_replacement:
        Whether or not to sample with replacement.

    Returns
    -------
        Subscripts and values for samples.
    """
    nnz = data.nnz

    # Select nonzeros
    if samples == nnz:
        nidx: np.ndarray = np.arange(0, nnz, dtype=int)
    elif with_replacement or samples < nnz:
        nidx = np.random.choice(nnz, size=samples, replace=with_replacement)
    else:
        raise ValueError("Tensor doesn't have enough nonzeros to sample")
    subs = data.subs[nidx, :]
    vals = data.vals[nidx]
    return subs, vals.squeeze(1)


def zeros(
    data: ttb.sptensor,
    nz_idx: np.ndarray,
    samples: int,
    over_sample_rate: float = 1.1,
    with_replacement=True,
) -> np.ndarray:
    """Sample zeros from a sparse tensor.

    Parameters
    ----------
    data:
        Tensor to sample.
    nz_idx:
        Sorted linear indices of the nonzeros in tensor.
    samples:
        Number of samples to retrieve.
    over_sample_rate:
        Oversampling rate to allow success if a samples is large relative to data.nnz
    with_replacement:
        Whether or not to sample with replacement.
    """
    data_size = np.prod(data.shape)
    nnz = len(nz_idx)
    num_zeros = data_size - nnz

    if over_sample_rate < 1.1:
        raise ValueError(
            f"Over sampling rate must be >= 1.1 but got {over_sample_rate}"
        )

    # Determine number of samples to generate
    # We need to oversample to account for potential duplicates and for
    # nonzeros we may pick

    if not with_replacement and samples > num_zeros:
        raise ValueError(
            "Cannot sample more than the total number of zeros without replacement"
        )

    # Save requested number of zeros
    samples_requested = samples

    # First determine the number of samples to take accounting for some will be
    # nonzeros and discarded.
    ntmp = np.ceil(samples * data_size / num_zeros)

    if not with_replacement and ntmp >= data_size:
        raise ValueError("Need too many zero samples for this to work")

    # Second determine number of samples given that some will be duplicates,
    # via coupon collector problem. This only matters if sampling with replacement.from
    if not with_replacement:
        ntmp = np.ceil(data_size * np.log(1 / (1 - (ntmp / data_size))))

    # Finally, add a margin of safety by oversampling
    samples = int(np.ceil(over_sample_rate * ntmp))

    # Generate actual samples, removing duplicates, nonzeros and excess
    tmpsubs = (
        np.ceil(
            np.random.uniform(0, 1, (samples, data.ndims)) * np.array(data.shape),
        ).astype(int)
        - 1
    )

    if not with_replacement:
        tmpsubs = np.unique(tmpsubs, axis=0)

    # Select out just the zeros
    tmpidx = tt_sub2ind(data.shape, tmpsubs)
    iszero = np.logical_not(np.isin(tmpidx, nz_idx))
    tmpsubs = tmpsubs[iszero, :]

    # Trim back to desired numb of samples
    samples = min(tmpsubs.shape[0], samples_requested)

    # Final return values
    if samples < samples_requested:
        logging.warning(
            "Unable to get number of zero samples requested"
            " Requested: %d but obtained: %d.",
            samples_requested,
            samples,
        )

    return tmpsubs[:samples, :]


def uniform(data: ttb.tensor, samples: int) -> sample_type:
    """Uniformly samples indices from a tensor.

    Parameters
    ----------
    data:
        Tensor to sample.
    samples:
        Number of samples to take.

    Returns
    -------
        Subscripts of samples, values at those subscripts, and weight of samples.
    """
    subs = (
        np.ceil(
            np.random.uniform(0, 1, (samples, data.ndims)) * np.array(data.shape),
        ).astype(int)
        - 1
    )
    vals = data[subs]
    wgts = (np.prod(data.shape) / samples) * np.ones((samples,))
    return subs, vals, wgts


def semistrat(data: ttb.sptensor, num_nonzeros: int, num_zeros: int) -> sample_type:
    """Sample nonzero and zero entries from a sparse tensor.

    Parameters
    ----------
    data:
        Tensor to sample.
    num_nonzeros:
        Number of nonzero samples requested.
    num_zeros:
        Number of zero samples requested.

    Returns
    -------
    Subscripts, values, and weights of samples (Nonzeros then zeros).
    """
    [nonzero_subs, nonzero_vals] = nonzeros(data, num_nonzeros, with_replacement=True)
    nonzero_weights = (data.nnz / num_nonzeros) * np.ones((num_nonzeros,))

    # Uniformly sample unconfirmed zeros
    zero_subs = np.ceil(
        np.random.uniform(0, 1, (num_zeros, data.ndims)) * (np.array(data.shape) - 1),
    ).astype(int)
    zero_vals = np.zeros((num_zeros,))
    zero_weights = (np.prod(data.shape) / num_zeros) * np.ones((num_zeros,))

    all_subs = np.vstack((nonzero_subs, zero_subs))
    all_vals = np.concatenate((nonzero_vals, zero_vals))
    all_weights = np.concatenate((nonzero_weights, zero_weights))
    return all_subs, all_vals, all_weights


def stratified(
    data: ttb.sptensor | ttb.tensor,
    nz_idx: np.ndarray,
    num_nonzeros: int,
    num_zeros: int,
    over_sample_rate: float = 1.1,
) -> sample_type:
    """Sample nonzero and zero entries from a sparse tensor.

    Parameters
    ----------
    data:
        Tensor to sample.
    nz_idx:
        Sorted linear indices of non-zero entries in tensor.
    num_nonzeros:
        Number of nonzero samples requested.
    num_zeros:
        Number of zero samples requested.
    over_sample_rate:
        Rate of oversampling to account for bad random draws.

    Returns
    -------
    Subscripts, values, and weights of samples (Nonzeros then zeros).
    """
    assert isinstance(data, ttb.sptensor), (
        "For stratified sampling Sparse Tensor must be provided"
    )
    [nonzero_subs, nonzero_vals] = nonzeros(data, num_nonzeros, with_replacement=True)
    nonzero_weights = np.ones((num_nonzeros,))
    if num_nonzeros > 0:
        nonzero_weights *= data.nnz / num_nonzeros

    zero_subs = zeros(data, nz_idx, num_zeros, over_sample_rate, with_replacement=True)
    zero_vals = np.zeros((num_zeros,))
    data_nonzero_count = np.prod(data.shape) - data.nnz
    zero_weights = np.ones((num_zeros,))
    if num_zeros > 0:
        zero_weights *= data_nonzero_count / num_zeros

    all_subs = np.vstack((nonzero_subs, zero_subs))
    all_vals = np.concatenate((nonzero_vals, zero_vals))
    all_weights = np.concatenate((nonzero_weights, zero_weights))
    return all_subs, all_vals.squeeze(), all_weights
