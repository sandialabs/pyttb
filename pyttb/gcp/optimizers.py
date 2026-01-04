"""Optimizer Implementations for GCP."""

# Copyright 2025 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from math import inf
from typing import TYPE_CHECKING, TypedDict

import numpy as np
from scipy.optimize import fmin_l_bfgs_b

import pyttb as ttb
from pyttb.gcp.fg import evaluate
from pyttb.gcp.fg_est import estimate
from pyttb.gcp.samplers import GCPSampler

if TYPE_CHECKING:
    from collections.abc import Callable

    from pyttb.gcp.fg_setup import function_type


class StochasticSolver(ABC):
    """Interface for Stochastic GCP Solvers."""

    def __init__(  # noqa: PLR0913
        self,
        rate: float = 1e-3,
        decay: float = 0.1,
        max_fails: int = 1,
        epoch_iters: int = 1000,
        f_est_tol: float = -inf,
        max_iters: int = 1000,
        printitn: int = 1,
    ):
        """General Setup for Stochastic Solvers.

        Parameters
        ----------
        rate:
            Rate of descent, proportional to step size.
        decay:
            How much to decrease step size on failed epochs.
        max_fails:
            How many failed epochs before terminating the solve.
        epoch_iters:
            Number of steps to take per epoch.
        f_est_tol:
            Tolerance for function estimate changes to terminate solve.
        max_iters:
            Maximum number of epochs.
        printitn:
            Controls verbosity of information during solve.
        """
        self._rate = rate
        self._decay = decay
        self._max_fails = max_fails
        self._epoch_iters = epoch_iters
        self._f_est_tol = f_est_tol
        self._max_iters = max_iters
        self._printitn = printitn
        self._nfails = 0

    @abstractmethod
    def update_step(
        self,
        model: ttb.ktensor,
        gradient: list[np.ndarray],
        lower_bound: float,
    ) -> tuple[list[np.ndarray], float]:
        """Calculate the update step for the solver.

        Parameters
        ----------
        model:
            Current decomposition.
        gradient:
            Gradient calculation.
        lower_bound:
            Minimum value for the decomposition.

        Returns
        -------
            Update to be applied to decomposition (to be applied by caller).
            Step size used.
        """

    @abstractmethod
    def set_failed_epoch(self):
        """Set internal state on failed epoch."""

    def solve(  # noqa: PLR0913
        self,
        initial_model: ttb.ktensor,
        data: ttb.tensor | ttb.sptensor,
        function_handle: function_type,
        gradient_handle: function_type,
        lower_bound: float = -np.inf,
        sampler: GCPSampler | None = None,
    ) -> tuple[ttb.ktensor, dict]:
        """Run solver until completion.

        Parameters
        ----------
        initial_model:
            Beginning solution.
        data:
            Tensor to solve for.
        function_handle:
            Callable to sample objective values.
        gradient_handle:
            Callable to sample gradient values.
        lower_bound:
            Lower bound on model values.
        sampler:
            Sampler to select which values to evaluate or take gradients from.

        Returns
        -------
            Final answer and dictionary of details.
        """
        if sampler is None:
            sampler = GCPSampler(data)
        solver_start = time.perf_counter()

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
        time_trace[0] = time.perf_counter() - solver_start
        main_start = time.perf_counter()

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
                if any(np.any(np.isinf(g_est_i)) for g_est_i in g_est):
                    raise ValueError(
                        f"Infinite gradient encountered! (epoch = {n_epoch}, "
                        f"iter = {iteration}"
                    )
                model.factor_matrices, step = self.update_step(
                    model, g_est, lower_bound
                )
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
                logging.info(msg)

            if failed_epoch:
                # Reset to best solution so far
                model = best_model.copy()
                f_est = f_est_prev
                self.set_failed_epoch()
            else:
                best_model = model.copy()
                f_est_prev = f_est

            # Save time
            time_trace[n_epoch + 1] = time.perf_counter() - solver_start

            if (self._nfails > self._max_fails) or f_est_tol_test:
                break
        main_time = time.perf_counter() - main_start

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
                f"Final f-est: {f_est: 10.4e}\n"
                f"Main loop time: {main_time: .2f}\n"
                f"Total iterations: {n_epoch * self._epoch_iters}"
            )
            logging.info(msg)

        return model, info


class SGD(StochasticSolver):
    """General Stochastic Gradient Descent."""

    def update_step(  # noqa: D102
        self, model: ttb.ktensor, gradient: list[np.ndarray], lower_bound: float
    ) -> tuple[list[np.ndarray], float]:
        step = self._decay**self._nfails * self._rate
        factor_matrices = [
            np.maximum(lower_bound, factor - step * grad)
            for factor, grad in zip(model.factor_matrices, gradient, strict=False)
        ]
        return factor_matrices, step

    def set_failed_epoch(self):  # noqa: D102
        # No additional internal state for SGD
        pass


class Adam(StochasticSolver):
    """Adam Optimizer."""

    def __init__(  # noqa: PLR0913
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
        """General Setup for Adam Solver.

        Parameters
        ----------
        rate:
            Rate of descent, proportional to step size.
        decay:
            How much to decrease step size on failed epochs.
        max_fails:
            How many failed epochs before terminating the solve.
        epoch_iters:
            Number of steps to take per epoch.
        f_est_tol:
            Tolerance for function estimate changes to terminate solve.
        max_iters:
            Maximum number of epochs.
        printitn:
            Controls verbosity of information during solve.
        beta_1:
            Adam specific momentum parameter beta_1.
        beta_2:
            Adam specific momentum parameter beta_2.
        epsilon:
            Adam specific momentum parameter to avoid division by zero.
        """
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
        self._m: list[np.ndarray] = []
        self._m_prev: list[np.ndarray] = []
        self._v: list[np.ndarray] = []
        self._v_prev: list[np.ndarray] = []

    def set_failed_epoch(  # noqa: D102
        self,
    ):
        self._total_iterations -= self._epoch_iters
        self._m = self._m_prev.copy()
        self._v = self._v_prev.copy()

    def update_step(  # noqa: D102
        self, model: ttb.ktensor, gradient: list[np.ndarray], lower_bound: float
    ) -> tuple[list[np.ndarray], float]:
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
            for mk, gk in zip(self._m, gradient, strict=False)
        ]
        self._v = [
            self._beta_2 * vk + (1 - self._beta_2) * gk**2
            for vk, gk in zip(self._v, gradient, strict=False)
        ]
        mhat = [mk / (1 - self._beta_1**self._total_iterations) for mk in self._m]
        vhat = [vk / (1 - self._beta_2**self._total_iterations) for vk in self._v]
        factor_matrices = [
            np.maximum(
                lower_bound, factor_k - step * mhk / (np.sqrt(vhk) + self._epsilon)
            )
            for factor_k, mhk, vhk in zip(
                model.factor_matrices, mhat, vhat, strict=False
            )
        ]
        return factor_matrices, step


class Adagrad(StochasticSolver):
    """Adagrad Optimizer."""

    def __init__(  # noqa: PLR0913
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

    def set_failed_epoch(  # noqa: D102
        self,
    ):
        self._gnormsum = 0.0

    def update_step(  # noqa: D102
        self, model: ttb.ktensor, gradient: list[np.ndarray], lower_bound: float
    ) -> tuple[list[np.ndarray], float]:
        self._gnormsum += np.sum([np.sum(gk**2) for gk in gradient])
        step = 1.0 / np.sqrt(self._gnormsum)
        factor_matrices = [
            np.maximum(lower_bound, factor_k - step * gk)
            for factor_k, gk in zip(model.factor_matrices, gradient, strict=False)
        ]
        return factor_matrices, step


# If we use more scipy optimizers in the future we should generalize this
class LBFGSB:
    """Simple wrapper around scipy lbfgsb.

    NOTE: If used for publications please see scipy documentation for adding citation
    for the implementation.
    """

    def __init__(  # noqa: PLR0913
        self,
        m: int | None = None,
        factr: float = 1e7,
        pgtol: float | None = None,
        epsilon: float | None = None,
        iprint: int | None = None,
        disp: int | None = None,
        maxfun: int | None = None,
        maxiter: int = 1000,
        callback: Callable[[np.ndarray], None] | None = None,
        maxls: int | None = None,
    ):
        """Prepare all hyper-parameters for solver.

        See scipy for details and standard defaults.
        A variety of defaults are set specifically for gcp opt.
        """

        class ArgType(TypedDict):
            m: int | None
            factr: float
            pgtol: float | None
            epsilon: float | None
            iprint: int | None
            disp: int | None
            maxfun: int | None
            maxiter: int
            callback: Callable[[np.ndarray], None] | None
            maxls: int | None

        self._solver_kwargs: ArgType = {
            "m": m,
            "factr": factr,
            "pgtol": pgtol,
            "epsilon": epsilon,
            "iprint": iprint,
            "disp": disp,
            "maxfun": maxfun,
            "maxiter": maxiter,
            "callback": callback,
            "maxls": maxls,
        }

    def _non_empty_kwargs(self):
        # Prune None values so we don't have to maintain compatibility with scipy
        return {
            key: value
            for key, value in self._solver_kwargs.items()
            if value is not None
        }

    def solve(  # noqa: PLR0913
        self,
        initial_model: ttb.ktensor,
        data: ttb.tensor,
        function_handle: function_type,
        gradient_handle: function_type,
        lower_bound: float = -np.inf,
        mask: np.ndarray | None = None,
    ) -> tuple[ttb.ktensor, dict]:
        """Solves the defined optimization problem."""
        model = initial_model.copy()

        def lbfgsb_func_grad(vector: np.ndarray):
            model.update(np.arange(initial_model.ndims), vector)
            func_val, grads = evaluate(
                model,
                data,
                mask,
                function_handle,
                gradient_handle,
            )
            return func_val, ttb.ktensor(grads, copy=False).tovec(False)

        x0 = model.tovec(False)
        if "pgtol" not in self._solver_kwargs:
            self._solver_kwargs["pgtol"] = 1e-4 * np.prod(data.shape)

        # Set callback function that returns time trace by default
        monitor = LBFGSB.Monitor(
            self._solver_kwargs["maxiter"],
            self._solver_kwargs.get("callback", None),  # callback may be pruned in ctor
        )
        self._solver_kwargs["callback"] = monitor

        final_vector, final_f, lbfgsb_info = fmin_l_bfgs_b(
            lbfgsb_func_grad,
            x0,
            fprime=None,
            approx_grad=False,
            bounds=[(lower_bound, np.inf)] * len(x0),
            **self._non_empty_kwargs(),
        )
        model.update(np.arange(initial_model.ndims), final_vector)

        lbfgsb_info["final_f"] = final_f
        lbfgsb_info["callback"] = vars(monitor)
        # Unregister monitor in case of reuse
        self._solver_kwargs["callback"] = monitor.callback

        # TODO big print output
        return model, lbfgsb_info

    class Monitor(dict):
        """Monitor LBFGSB Timings."""

        def __init__(
            self,
            maxiter: int,
            callback: Callable[[np.ndarray], None] | None = None,  # type: ignore
        ):
            self.startTime = time.perf_counter()
            self.time_trace = np.zeros((maxiter,))
            self.iter = 0
            self._callback = callback

        def __call__(self, xk: np.ndarray) -> None:
            """Update monitor."""
            if self._callback is not None:
                self._callback(xk)
            self.time_trace[self.iter] = time.perf_counter() - self.startTime
            self.iter += 1

        @property
        def callback(self):
            """Return stored callback."""
            return self._callback

        @property
        def __dict__(self):
            """Monitor Entries."""
            if not self._callback:
                return {"time_trace": self.time_trace}
            else:
                return {"time_trace": self.time_trace, "callback": self._callback}
