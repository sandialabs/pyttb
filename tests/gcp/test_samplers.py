from __future__ import annotations

from math import exp, log, pi

import numpy as np
import pytest
import scipy

import pyttb as ttb
from pyttb.gcp import samplers
from pyttb.gcp.handles import EPS


def test_nonzeros():
    data = ttb.sptenrand((2, 2), nonzeros=2)

    # Sample all values
    subs, vals = samplers.nonzeros(data, 2, False)
    np.testing.assert_array_equal(subs, data.subs)
    np.testing.assert_array_equal(vals, data.vals)

    # Sample subset
    subs, vals = samplers.nonzeros(data, 1, False)
    assert np.all(np.isin(vals, data.vals))
    assert len(vals) == 1

    # Sample with replacement
    subs, vals = samplers.nonzeros(data, 4)
    assert np.all(np.isin(vals, data.vals))
    assert len(vals) == 4

    with pytest.raises(ValueError):
        # Sample too many without replacement
        samplers.nonzeros(data, 4, False)


def test_zeros():
    data = ttb.sptenrand((2, 2), nonzeros=2)
    lin_idx = np.sort(ttb.tt_sub2ind((2, 2), data.subs))
    subs = samplers.zeros(data, lin_idx, 1)
    assert len(subs.shape) == 2

    subs = samplers.zeros(data, lin_idx, 1, with_replacement=False)
    assert len(subs.shape) == 2

    # Negative tests
    # Too small over sample rate
    with pytest.raises(ValueError):
        samplers.zeros(data, lin_idx, 1, over_sample_rate=0.0)

    # Too many samples without replacement
    with pytest.raises(ValueError):
        samplers.zeros(data, lin_idx, 3, with_replacement=False)

    # Too many samples without replacement (over sampling)
    with pytest.raises(ValueError):
        samplers.zeros(data, lin_idx, 2, with_replacement=False)