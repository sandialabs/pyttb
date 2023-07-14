"""Implementation of various sampling approaches for GCP OPT"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

import pyttb as ttb


def nonzeros(
    data: ttb.sptensor, samples: int, with_replacement: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
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
        nidx = np.arange(0, nnz)
    elif with_replacement or samples < nnz:
        nidx = np.random.choice(nnz, size=samples, replace=with_replacement)
    else:
        raise ValueError("Tensor doesn't have enough nonzeros to sample")
    subs = data.subs[nidx, :]
    vals = data.vals[nidx]
    return subs, vals


def zeros(
    data: ttb.sptensor,
    nz_idx: np.ndarray,
    samples: int,
    over_sample_rate: float = 1.1,
    with_replacement=True,
) -> np.ndarray:
    """Samples zeros from a sparse tensor

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
    tmpsubs = np.ceil(
        np.random.uniform(0, 1, (samples, data.ndims)) * (np.array(data.shape) - 1),
    ).astype(int)

    if not with_replacement:
        tmpsubs = np.unique(tmpsubs, axis=0)

    # Select out just the zeros
    tmpidx = ttb.tt_sub2ind(data.shape, tmpsubs)
    iszero = np.logical_not(np.in1d(tmpidx, nz_idx))
    tmpsubs = tmpsubs[iszero, :]

    # Trim back to desired numb of samples
    samples = min(tmpsubs.shape[0], samples_requested)

    # Final return values
    if samples < samples_requested:
        logging.warning(
            "Unable to get number of zero samples requested"
            "Requested: %d but obtained: %d.",
            samples_requested,
            samples,
        )

    return tmpsubs[:samples, :]


def uniform(
    data: ttb.tensor, samples: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniformly samples indices from a tensor

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
    subs = np.ceil(
        np.random.uniform(0, 1, (samples, data.ndims)) * (np.array(data.shape) - 1),
    ).astype(int)
    vals = data[subs]
    wgts = (np.prod(data.shape) / samples) * np.ones((samples,))
    return subs, vals, wgts
