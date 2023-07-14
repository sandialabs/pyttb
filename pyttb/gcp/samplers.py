"""Implementation of various sampling approaches for GCP OPT"""

from __future__ import annotations

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
