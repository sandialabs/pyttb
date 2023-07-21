from __future__ import annotations

import numpy as np
import pytest

import pyttb as ttb
from pyttb.gcp import samplers


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
    shape = (2, 2)
    data = ttb.sptenrand(shape, nonzeros=2)
    lin_idx = np.sort(ttb.tt_sub2ind(shape, data.subs))
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


def test_uniform():
    data = ttb.tenrand((4, 4))
    subs, vals, wgts = samplers.uniform(data, 2)
    assert np.all(np.isin(vals, data.data))
    assert len(subs) == 2
    assert len(wgts) == 2


def test_semistrat():
    data = ttb.sptenrand((4, 4), nonzeros=2)
    num_zeros = 2
    num_nonzeros = 2
    subs, vals, wgts = samplers.semistrat(data, num_nonzeros, num_zeros)
    assert len(subs.shape) == 2
    assert len(vals) == num_zeros + num_nonzeros
    assert len(wgts) == 4
    assert np.all(vals[:num_nonzeros] != 0.0)


def test_stratified():
    shape = (4, 4)
    num_zeros = 2
    num_nonzeros = 2
    data = ttb.sptenrand(shape, nonzeros=2)
    lin_idx = np.sort(ttb.tt_sub2ind(shape, data.subs))
    subs, vals, wgts = samplers.stratified(data, lin_idx, num_nonzeros, num_zeros)
    assert len(subs.shape) == 2
    assert len(vals) == num_zeros + num_nonzeros
    assert len(wgts) == num_zeros + num_nonzeros
    assert np.all(vals[:num_nonzeros] != 0.0)
    assert np.all(vals[-num_zeros:] == 0.0)
