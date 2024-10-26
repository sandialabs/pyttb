# Copyright 2024 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from __future__ import annotations

import numpy as np
import pytest

import pyttb as ttb
from pyttb.gcp import samplers
from pyttb.pyttb_utils import tt_sub2ind


def check_sample_output(
    subs: np.ndarray,
    vals: np.ndarray,
    wgts: np.ndarray,
    num_zeros: int,
    num_nonzeros: int,
):
    """Verify shapes of values returned from sampler"""
    assert len(subs.shape) == 2
    assert len(vals) == num_zeros + num_nonzeros
    assert len(wgts) == num_zeros + num_nonzeros


def test_nonzeros():
    data = ttb.sptenrand((2, 2), nonzeros=2)

    # Sample all values
    subs, vals = samplers.nonzeros(data, 2, False)
    np.testing.assert_array_equal(subs, data.subs)
    np.testing.assert_array_equal(vals, data.vals.squeeze())

    # Sample subset
    subs, vals = samplers.nonzeros(data, 1, False)
    assert np.all(np.isin(vals, data.vals.squeeze()))
    assert len(vals) == 1

    # Sample with replacement
    subs, vals = samplers.nonzeros(data, 4)
    assert np.all(np.isin(vals, data.vals.squeeze()))
    assert len(vals) == 4

    with pytest.raises(ValueError):
        # Sample too many without replacement
        samplers.nonzeros(data, 4, False)


def test_zeros():
    shape = (2, 2)
    data = ttb.sptenrand(shape, nonzeros=2)
    lin_idx = np.sort(tt_sub2ind(shape, data.subs))
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
    check_sample_output(subs, vals, wgts, num_zeros, num_nonzeros)
    assert np.all(vals[:num_nonzeros] != 0.0)


def test_stratified():
    shape = (4, 4)
    num_zeros = 2
    num_nonzeros = 2
    data = ttb.sptenrand(shape, nonzeros=2)
    lin_idx = np.sort(tt_sub2ind(shape, data.subs))
    subs, vals, wgts = samplers.stratified(data, lin_idx, num_nonzeros, num_zeros)
    check_sample_output(subs, vals, wgts, num_zeros, num_nonzeros)
    assert np.all(vals[:num_nonzeros] != 0.0)
    assert np.all(vals[-num_zeros:] == 0.0)


def test_gcp_sampler():
    num_zeros = 2
    num_nonzeros = 2
    # Dense data defaults
    dense_data = ttb.tenones((2, 2))
    dense_data[0, 1] = 0.0
    dense_data[1, 0] = 0.0
    sampler = samplers.GCPSampler(dense_data)
    subs, vals, wgts = sampler.function_sample(dense_data)
    check_sample_output(subs, vals, wgts, num_zeros, num_nonzeros)
    subs, vals, wgts = sampler.gradient_sample(dense_data)
    check_sample_output(subs, vals, wgts, num_zeros, num_nonzeros)
    assert sampler.crng.size == 0
    assert np.issubdtype(sampler.crng.dtype, np.integer)

    # Sparse data defaults
    sparse_data = dense_data.to_sptensor()
    sampler = samplers.GCPSampler(sparse_data)
    subs, vals, wgts = sampler.function_sample(sparse_data)
    check_sample_output(subs, vals, wgts, num_zeros, num_nonzeros)
    subs, vals, wgts = sampler.gradient_sample(sparse_data)
    check_sample_output(subs, vals, wgts, num_zeros, num_nonzeros)
    assert sampler.crng.size == 0
    assert np.issubdtype(sampler.crng.dtype, np.integer)

    # Sparse stratified integer sample count
    sampler = samplers.GCPSampler(
        sparse_data,
        function_samples=2,
        gradient_samples=2,
    )
    subs, vals, wgts = sampler.function_sample(sparse_data)
    check_sample_output(subs, vals, wgts, num_zeros, num_nonzeros)
    subs, vals, wgts = sampler.gradient_sample(sparse_data)
    check_sample_output(subs, vals, wgts, num_zeros, num_nonzeros)
    assert sampler.crng.size == 0
    assert np.issubdtype(sampler.crng.dtype, np.integer)

    # Sparse stratified stratified count
    sampler = samplers.GCPSampler(
        sparse_data,
        function_samples=samplers.StratifiedCount(2, 2),
        gradient_samples=samplers.StratifiedCount(2, 2),
    )
    subs, vals, wgts = sampler.function_sample(sparse_data)
    check_sample_output(subs, vals, wgts, num_zeros, num_nonzeros)
    subs, vals, wgts = sampler.gradient_sample(sparse_data)
    check_sample_output(subs, vals, wgts, num_zeros, num_nonzeros)
    assert sampler.crng.size == 0
    assert np.issubdtype(sampler.crng.dtype, np.integer)

    # Sparse uniform sampling
    sampler = samplers.GCPSampler(
        sparse_data,
        function_sampler=samplers.Samplers.UNIFORM,
        gradient_sampler=samplers.Samplers.UNIFORM,
    )
    subs, vals, wgts = sampler.function_sample(sparse_data)
    check_sample_output(subs, vals, wgts, num_zeros, num_nonzeros)
    sampler.gradient_sample(sparse_data)
    # We skip verifying sptensor UNIFORM gradient samples since it can vary in shape
    assert sampler.crng.size == 0
    assert np.issubdtype(sampler.crng.dtype, np.integer)

    # Sparse semi-stratified sampling
    sampler = samplers.GCPSampler(
        sparse_data,
        function_sampler=samplers.Samplers.UNIFORM,
        gradient_sampler=samplers.Samplers.SEMISTRATIFIED,
    )
    subs, vals, wgts = sampler.function_sample(sparse_data)
    check_sample_output(subs, vals, wgts, num_zeros, num_nonzeros)
    subs, vals, wgts = sampler.gradient_sample(sparse_data)
    check_sample_output(subs, vals, wgts, num_zeros, num_nonzeros)
    assert sampler.crng.size != 0
    assert np.issubdtype(sampler.crng.dtype, np.integer)

    # Negative tests

    # No dense stratified function
    with pytest.raises(ValueError):
        samplers.GCPSampler(
            dense_data,
            function_sampler=samplers.Samplers.STRATIFIED,
        )
    # No dense stratified gradient
    with pytest.raises(ValueError):
        samplers.GCPSampler(
            dense_data,
            gradient_sampler=samplers.Samplers.STRATIFIED,
        )
    # Incorrect function samples type stratified
    with pytest.raises(ValueError):
        samplers.GCPSampler(sparse_data, function_samples=(2, 2))
    # Incorrect gradient samples type stratified
    with pytest.raises(ValueError):
        samplers.GCPSampler(sparse_data, gradient_samples=(2, 2))
    # Incorrect function samples type uniform
    with pytest.raises(ValueError):
        samplers.GCPSampler(dense_data, function_samples=(2, 2))
    # Incorrect gradient samples type uniform
    with pytest.raises(ValueError):
        samplers.GCPSampler(dense_data, gradient_samples=(2, 2))
    # Bad function sampler type
    with pytest.raises(ValueError):
        samplers.GCPSampler(dense_data, function_sampler="Something bad")
    # Bad gradient sampler type
    with pytest.raises(ValueError):
        samplers.GCPSampler(dense_data, gradient_sampler="Something bad")
