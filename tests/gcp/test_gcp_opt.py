from __future__ import annotations

import numpy as np
import pytest

import pyttb as ttb
from pyttb.gcp import samplers
from pyttb.gcp.handles import Objectives, gaussian, gaussian_grad
from pyttb.gcp.optimizers import LBFGSB, SGD, Adagrad, Adam


class TestGcpOpt:
    def test_external_solves(
        self,
    ):
        dense_data = ttb.tenones((2, 2))
        dense_data[0, 1] = 0.0
        dense_data[1, 0] = 0.0
        rank = 2
        optimizer = LBFGSB(maxiter=2)
        result_gauss, initial_guess, info = ttb.gcp_opt(
            dense_data, rank, Objectives.GAUSSIAN, optimizer
        )
        assert not result_gauss.isequal(initial_guess)
        assert all(initial_guess.weights == 1.0)

        # Test with missing data
        mask = ttb.tenones(dense_data.shape)
        mask[0] = 0
        result_gauss, initial_guess, info = ttb.gcp_opt(
            dense_data, rank, Objectives.GAUSSIAN, optimizer, mask=mask
        )
        assert not result_gauss.isequal(initial_guess)
        assert all(initial_guess.weights == 1.0)

    def test_stochastic_solves(
        self,
    ):
        dense_data = ttb.tenones((2, 2))
        dense_data[0, 1] = 0.0
        dense_data[1, 0] = 0.0
        rank = 2

        np.random.seed(1)
        optimizer = SGD(max_iters=2, epoch_iters=1)
        result_gauss, initial_guess, info = ttb.gcp_opt(
            dense_data, rank, Objectives.GAUSSIAN, optimizer
        )
        assert not result_gauss.isequal(initial_guess)
        assert all(initial_guess.weights == 1.0)

        # Providing an initial guess skips the rng to generate initial ktensor
        np.random.seed(1)
        optimizer = SGD(max_iters=2, epoch_iters=1)
        result_gauss, initial_guess, info = ttb.gcp_opt(
            dense_data, rank, Objectives.GAUSSIAN, optimizer, initial_guess
        )
        assert not result_gauss.isequal(initial_guess)
        assert all(initial_guess.weights == 1.0)

        # Test custom objective equivalent to GAUSSIAN
        np.random.seed(1)
        optimizer = SGD(max_iters=2, epoch_iters=1)
        result, initial_guess_custom, info = ttb.gcp_opt(
            dense_data,
            rank,
            (gaussian, gaussian_grad, -np.inf),
            optimizer,
            initial_guess,
        )
        assert not result.isequal(initial_guess)
        assert initial_guess_custom.isequal(initial_guess)
        assert all(
            np.allclose(obj_factor, custom_factor)
            for obj_factor, custom_factor in zip(
                result.factor_matrices, result_gauss.factor_matrices
            )
        )

        # Test non-normalized initial guess
        np.random.seed(1)
        optimizer = SGD(max_iters=2, epoch_iters=1)
        non_norm_guess = initial_guess.copy()
        non_norm_guess.weights *= 2
        result_gauss, initial_guess, info = ttb.gcp_opt(
            dense_data, rank, Objectives.GAUSSIAN, optimizer, non_norm_guess
        )
        assert not result_gauss.isequal(initial_guess)
        assert all(initial_guess.weights == 1.0)

        # Test just providing factor matrices
        np.random.seed(1)
        optimizer = SGD(max_iters=2, epoch_iters=1)
        result_gauss, initial_guess, info = ttb.gcp_opt(
            dense_data,
            rank,
            Objectives.GAUSSIAN,
            optimizer,
            initial_guess.factor_matrices,
        )
        assert not result_gauss.isequal(initial_guess)
        assert all(initial_guess.weights == 1.0)

    def test_invalid_optimizer_options(
        self,
    ):
        dense_data = ttb.tenones((2, 2))
        dense_data[0, 1] = 0.0
        dense_data[1, 0] = 0.0
        rank = 2

        # No mask with stochastic solve
        with pytest.raises(ValueError):
            optimizer = SGD(max_iters=2, epoch_iters=1)
            ttb.gcp_opt(
                dense_data,
                rank,
                Objectives.GAUSSIAN,
                optimizer,
                mask=ttb.tenones(dense_data.shape),
            )

        # LBFGSB only supports dense
        with pytest.raises(ValueError):
            sparse_data = ttb.sptensor.from_tensor_type(dense_data)
            ttb.gcp_opt(sparse_data, rank, Objectives.GAUSSIAN, LBFGSB())

    def test_general_invalid_options(
        self,
    ):
        dense_data = ttb.tenones((2, 2))
        dense_data[0, 1] = 0.0
        dense_data[1, 0] = 0.0
        rank = 2
        optimizer = SGD(max_iters=2, epoch_iters=1)

        # Incorrect customer objective
        with pytest.raises(ValueError):
            ttb.gcp_opt(dense_data, rank, (1, 2), optimizer)

        # Sptensor with mask
        with pytest.raises(ValueError):
            ttb.gcp_opt(
                ttb.sptensor(),
                rank,
                Objectives.GAUSSIAN,
                optimizer,
                mask=np.ones((2, 2)),
            )

        # Non-tensor data
        with pytest.raises(ValueError):
            result, initial_guess_custom, info = ttb.gcp_opt(
                [], rank, (gaussian, gaussian_grad, -np.inf), optimizer
            )

        # Invalid optimizer choices
        with pytest.raises(ValueError):
            ttb.gcp_opt(dense_data, rank, Objectives.GAUSSIAN, "Not an optimizer")
        # Invalid Init
        with pytest.raises(ValueError):
            optimizer = SGD(max_iters=2, epoch_iters=1)
            ttb.gcp_opt(
                dense_data,
                rank,
                Objectives.GAUSSIAN,
                optimizer,
                init="Not a supported choice",
            )
