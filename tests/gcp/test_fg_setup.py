from __future__ import annotations

from inspect import Parameter, signature
from math import exp, log, pi

import numpy as np
import pytest
import scipy

import pyttb as ttb
from pyttb.gcp import fg_setup
from pyttb.gcp.handles import Objectives


def test_setup_success():
    # Loop over happy case for all known objectives
    for an_objective in Objectives:
        fh, gh, lb = fg_setup.setup(an_objective, additional_parameter=0.1)

        # Make sure handles are distinct callables
        assert callable(fh)
        assert callable(gh)
        assert fh is not gh

        # Make sure they take two and only two arguments
        # our use of partial makes testing this a bit awkward
        fh_non_default_params = 0
        for a_param in signature(fh).parameters.values():
            if a_param.default is Parameter.empty:
                fh_non_default_params += 1
        assert fh_non_default_params == 2
        gh_non_default_params = 0
        for a_param in signature(fh).parameters.values():
            if a_param.default is Parameter.empty:
                gh_non_default_params += 1
        assert gh_non_default_params == 2

        # Make sure lower bound is a scalar
        assert np.isscalar(lb)


def test_setup_wrong_data():
    bad_data = -0.1 * ttb.tenones((2, 2))
    for objective in (
        Objectives.BERNOULLI_ODDS,
        Objectives.BERNOULLI_LOGIT,
        Objectives.POISSON,
        Objectives.POISSON_LOG,
        Objectives.RAYLEIGH,
        Objectives.GAMMA,
        Objectives.NEGATIVE_BINOMIAL,
        Objectives.BETA,
    ):
        with pytest.raises(ValueError):
            fg_setup.setup(objective, bad_data)


def test_setup_missing_param():
    for objective in (
        Objectives.HUBER,
        Objectives.NEGATIVE_BINOMIAL,
        Objectives.BETA,
    ):
        with pytest.raises(ValueError):
            fg_setup.setup(objective)


def test_setup_bad_objective():
    with pytest.raises(ValueError):
        fg_setup.setup("Please Use Enum")


def test_valid_nonneg():
    non_negative_dense = ttb.tenones((2, 2))
    assert fg_setup.valid_nonneg(non_negative_dense)

    negative_dense = -1 * non_negative_dense
    assert not fg_setup.valid_nonneg(negative_dense)

    non_negative_sparse = ttb.sptenrand((2, 2), nonzeros=3)
    non_negative_sparse.vals = np.abs(non_negative_sparse.vals)
    assert fg_setup.valid_nonneg(non_negative_sparse)

    negative_sparse = -1 * non_negative_sparse
    assert not fg_setup.valid_nonneg(negative_sparse)


def test_valid_binary():
    binary_dense = ttb.tenones((2, 2))
    assert fg_setup.valid_binary(binary_dense)

    arbitrary_dense = 4 * binary_dense
    assert not fg_setup.valid_binary(arbitrary_dense)

    binary_sparse = ttb.sptenrand((2, 2), nonzeros=3)
    binary_sparse.vals = 1
    assert fg_setup.valid_binary(binary_sparse)

    arbitrary_sparse = 5 * binary_sparse
    assert not fg_setup.valid_binary(arbitrary_sparse)


def test_valid_natural():
    natural_dense = 2 * ttb.tenones((2, 2))
    assert fg_setup.valid_natural(natural_dense)

    arbitrary_dense = 0.5 + natural_dense
    assert not fg_setup.valid_natural(arbitrary_dense)

    natural_sparse = ttb.sptenrand((2, 2), nonzeros=3)
    natural_sparse.vals = 2
    assert fg_setup.valid_natural(natural_sparse)

    arbitrary_sparse = 1.1 * natural_sparse
    assert not fg_setup.valid_natural(arbitrary_sparse)
