"""Optimizer Implementations for GCP"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from math import inf
from typing import Dict, List, Tuple, Union

import numpy as np

import pyttb as ttb
from pyttb.gcp.fg_est import estimate
from pyttb.gcp.fg_setup import function_type
from pyttb.gcp.samplers import GCPSampler


# pylint: disable=too-many-instance-attributes
class StochasticSolver(ABC):
    """Interface for Stochastic GCP Solvers"""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        rate: float = 1e-3,
        decay: float = 0.1,
        max_fails: int = 1,
        epoch_iters: int = 1000,
        f_est_tol: float = -inf,
        max_iters: int = 1000,
        printitn: int = 1,
        lower_bound: float = -np.inf,
    ):
        self._rate = rate
        self._decay = decay
        self._max_fails = max_fails
        self._epoch_iters = epoch_iters
        self._f_est_tol = f_est_tol
        self._max_iters = max_iters
        self._printitn = printitn
        self._nfails = 0
        self._lb = lower_bound

    @abstractmethod
    def update_step(
        self, model: ttb.ktensor, gradient: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], float]:
        """Generate update step"""

    @abstractmethod
    def set_failed_epoch(self):
        """Set internal state on failed epoch"""

    # pylint: disable=too-many-locals
    def solve(
        self,
        initial_model: ttb.ktensor,
        data: Union[ttb.tensor, ttb.sptensor],
        function_handle: function_type,
        gradient_handle: function_type,
        sampler: GCPSampler,
    ) -> Tuple[ttb.ktensor, Dict]:
        """Run solver until completion"""
        solver_start = time.monotonic()

        # Extract samples for estimating function value - these never change
        f_subs, f_vals, f_wgts = sampler.function_sample(data)

        # Compute initial estimated function value
        f_est = estimate(
            initial_model, f_subs, f_vals, f_wgts, function_handle, lambda_check=False
        )

        # Setup loop variables
        model = initial_model.copy()
        self._nfails = 0

        best_model = model.copy()
        f_est_prev = f_est

        # Tracing the progress in the function value by epoch
        fest_trace = np.zeros((self._max_iters + 1,))
        step_trace = np.zeros((self._max_iters + 1,))
        time_trace = np.zeros((self._max_iters + 1,))
        fest_trace[0] = f_est

        if self._printitn > 0:
            logging.info("Begin Main loop\nInitial f-est: %e\n", f_est)
        # Note in MATLAB this time also includes the time for setting up samplers
        time_trace[0] = time.monotonic() - solver_start
        main_start = time.monotonic()

        n_epoch = 0  # In case range short circuits
        for n_epoch in range(self._max_iters):
            for iteration in range(self._epoch_iters):
                # Select subset for stochastic gradient
                g_subs, g_vals, g_wgts = sampler.gradient_sample(data)

                # Compute gradients for each mode
                g_est = estimate(
                    model,
                    g_subs,
                    g_vals,
                    g_wgts,
                    None,  # Functional handle unused, just to make typing happy
                    gradient_handle=gradient_handle,
                    lambda_check=False,
                    crng=sampler.crng,
                )

                # Check for inf
                if np.any(np.isinf(g_est)):
                    raise ValueError(
                        f"Infinite gradient encountered! (epoch = {n_epoch}, "
                        f"iter = {iteration}"
                    )
                model.factor_matrices, step = self.update_step(model, g_est)
            # Estimate objective function value
            f_est = estimate(
                model, f_subs, f_vals, f_wgts, function_handle, lambda_check=False
            )
            # Save trace
            fest_trace[n_epoch + 1] = f_est
            step_trace[n_epoch + 1] = step

            # Check convergence
            failed_epoch = f_est > f_est_prev
            self._nfails += failed_epoch

            f_est_tol_test = f_est < self._f_est_tol

            # Reporting
            if self._printitn > 0 and (
                n_epoch % self._printitn == 0 or failed_epoch or f_est_tol_test
            ):
                msg = f"Epoch {n_epoch}: f-est = {f_est}, step = {step}"
                if failed_epoch:
                    msg += (
                        f", nfails = {self._nfails} (resetting to solution from "
                        "last epoch)"
                    )

            if failed_epoch:
                # Reset to best solution so far
                model = best_model.copy()
                f_est = f_est_prev
                self.set_failed_epoch()
            else:
                best_model = model.copy()
                f_est_prev = f_est

            # Save time
            time_trace[n_epoch + 1] = time.monotonic() - solver_start

            if (self._nfails > self._max_fails) or f_est_tol_test:
                break
        main_time = time.monotonic() - main_start

        info = {
            "f_est_trace": fest_trace[0 : n_epoch + 1],
            "step_trace": step_trace[0 : n_epoch + 1],
            "time_trace": time_trace[0 : n_epoch + 1],
            "n_epoch": n_epoch,
        }

        if self._printitn > 0:
            # TODO print setup time which include sampler setup time external to here
            msg = (
                "End Main Loop\n"
                f"Final f-est: {f_est}"
                f"Main loop time: {main_time}"
                f"Total iterations: {n_epoch*self._epoch_iters}"
            )
            logging.info(msg)

        return model, info


class SGD(StochasticSolver):
    """General Stochastic Gradient Descent"""

    def update_step(
        self, model: ttb.ktensor, gradient: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], float]:
        step = self._decay**self._nfails * self._rate
        factor_matrices = [
            np.maximum(self._lb, factor - step * grad)
            for factor, grad in zip(model.factor_matrices, gradient)
        ]
        return factor_matrices, step

    def set_failed_epoch(self):
        # No additional internal state for SGD
        pass


class Adam(StochasticSolver):
    """Adam Optimizer"""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        rate: float = 1e-3,
        decay: float = 0.1,
        max_fails: int = 1,
        epoch_iters: int = 1000,
        f_est_tol: float = -inf,
        max_iters: int = 1000,
        printitn: int = 1,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        super().__init__(
            rate,
            decay,
            max_fails,
            epoch_iters,
            f_est_tol,
            max_iters,
            printitn,
        )
        self._total_iterations = 0
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._epsilon = epsilon
        self._m: List[np.ndarray] = []
        self._m_prev: List[np.ndarray] = []
        self._v: List[np.ndarray] = []
        self._v_prev: List[np.ndarray] = []

    def set_failed_epoch(
        self,
    ):
        self._total_iterations -= self._epoch_iters
        self._m = self._m_prev.copy()
        self._v = self._v_prev.copy()

    def update_step(
        self, model: ttb.ktensor, gradient: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], float]:
        if self._total_iterations == 0:
            for shape_i in model.shape:
                self._m.append(
                    np.zeros_like(
                        model.factor_matrices[0], shape=(shape_i, model.ncomponents)
                    )
                )
                self._v.append(
                    np.zeros_like(
                        model.factor_matrices[0], shape=(shape_i, model.ncomponents)
                    )
                )
        self._total_iterations += self._epoch_iters
        step = self._decay**self._nfails * self._rate
        self._m_prev = self._m.copy()
        self._v_prev = self._v.copy()
        self._m = [
            self._beta_1 * mk + (1 - self._beta_1) * gk
            for mk, gk in zip(self._m, gradient)
        ]
        self._v = [
            self._beta_2 * vk + (1 - self._beta_2) * gk**2
            for vk, gk in zip(self._v, gradient)
        ]
        mhat = [mk / (1 - self._beta_1**self._total_iterations) for mk in self._m]
        vhat = [vk / (1 - self._beta_2**self._total_iterations) for vk in self._v]
        factor_matrices = [
            np.maximum(self._lb, factor_k - step * mhk / (np.sqrt(vhk) + self._epsilon))
            for factor_k, mhk, vhk in zip(model.factor_matrices, mhat, vhat)
        ]
        return factor_matrices, step


class Adagrad(StochasticSolver):
    """Adagrad Optimizer"""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        rate: float = 1e-3,
        decay: float = 0.1,
        max_fails: int = 1,
        epoch_iters: int = 1000,
        f_est_tol: float = -inf,
        max_iters: int = 1000,
        printitn: int = 1,
    ):
        super().__init__(
            rate,
            decay,
            max_fails,
            epoch_iters,
            f_est_tol,
            max_iters,
            printitn,
        )
        self._gnormsum = 0.0

    def set_failed_epoch(
        self,
    ):
        self._gnormsum = 0.0

    def update_step(
        self, model: ttb.ktensor, gradient: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], float]:
        self._gnormsum += np.sum([np.sum(gk**2) for gk in gradient])
        step = 1.0 / np.sqrt(self._gnormsum)
        factor_matrices = [
            np.maximum(self._lb, factor_k - step * gk)
            for factor_k, gk in zip(model.factor_matrices, gradient)
        ]
        return factor_matrices, step
