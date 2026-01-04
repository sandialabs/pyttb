"""Create test problems for tensor factorizations."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast, overload

import numpy as np
from numpy_groupies import aggregate as accumarray

import pyttb as ttb
from pyttb.pyttb_utils import Shape, parse_shape

solution_generator = Callable[[tuple[int, ...]], np.ndarray]
core_generator_t = Callable[[tuple[int, ...]], ttb.tensor | ttb.sptensor | np.ndarray]


def randn(shape: tuple[int, ...]) -> np.ndarray:
    """Stub for MATLAB randn.

    TODO move somewhere shareable.
    """
    return np.random.normal(0, 1, size=shape)


@dataclass
class BaseProblem:
    """Parameters general to all solutions.

    Attributes
    ----------
    shape:
        Tensor shape for generated problem.
    factor_generator:
        Method to generate factor matrices.
    symmetric:
        List of modes that should be symmetric.
        For instance, `[(1,2), (3,4)]` specifies that
        modes 1 and 2 have identical factor matrices, and modes 3 and 4
        also have identical factor matrices.
    num_factors:
        Number of factors.
    noise:
        Amount of Gaussian noise to add to solution.
        If data is sparse noise is only added to nonzero entries.
    """

    shape: Shape = field(metadata={"doc": "A shape"})
    factor_generator: solution_generator = randn
    symmetric: list[tuple[int, int]] | None = None
    num_factors: int | list[int] | None = None
    noise: float = 0.10

    def __post_init__(self):
        self.shape = ttb.pyttb_utils.parse_shape(self.shape)
        if not 0.0 <= self.noise <= 1.0:
            raise ValueError(f"Noise must be in [0,1] but got {self.noise}")


@dataclass
class CPProblem(BaseProblem):
    """Parameters specifying CP Solutions.

    Attributes
    ----------
    shape:
        Tensor shape for generated problem.
    factor_generator:
        Method to generate factor matrices.
    symmetric:
        List of modes that should be symmetric.
        For instance, `[(1,2), (3,4)]` specifies that
        modes 1 and 2 have identical factor matrices, and modes 3 and 4
        also have identical factor matrices.
    num_factors:
        Number of factors.
    noise:
        Amount of Gaussian noise to add to solution.
        If data is sparse noise is only added to nonzero entries.
    weight_generator:
        Method to generate weights for ktensor solution.
    """

    # NOTE inherited attributes are manually copy pasted, keep aligned between problems

    num_factors: int = 2
    weight_generator: solution_generator = np.random.random
    # TODO: This is in DataParams in MATLAB, but only works for CP problems so
    # feels more reasonable here
    sparse_generation: float | None = None


@dataclass
class TuckerProblem(BaseProblem):
    """Parameters specifying Tucker Solutions.

    Attributes
    ----------
    shape:
        Tensor shape for generated problem.
    factor_generator:
        Method to generate factor matrices.
    symmetric:
        List of modes that should be symmetric.
        For instance, `[(1,2), (3,4)]` specifies that
        modes 1 and 2 have identical factor matrices, and modes 3 and 4
        also have identical factor matrices.
    num_factors:
        Number of factors.
    noise:
        Amount of Gaussian noise to add to solution.
        If data is sparse noise is only added to nonzero entries.
    core_generator:
        Method to generate weights for ttensor solution.
    """

    # TODO post_init set to [2, 2, 2]
    num_factors: list[int] | None = None
    core_generator: core_generator_t = randn

    def __post_init__(self):
        super().__post_init__()
        self.num_factors = self.num_factors or [2, 2, 2]


@dataclass
class ExistingSolution:
    """Parameters for using an existing tensor solution.

    Attributes
    ----------
    solution:
        Pre-existing tensor solution (ktensor or ttensor).
    noise:
        Amount of Gaussian noise to add to solution.
        If data is sparse noise is only added to nonzero entries.
    """

    solution: ttb.ktensor | ttb.ttensor
    noise: float = 0.10

    def __post_init__(self):
        if not 0.0 <= self.noise <= 1.0:
            raise ValueError(f"Noise must be in [0,1] but got {self.noise}")

    @property
    def symmetric(self) -> None:
        """Get the symmetric modes from the solution."""
        # ExistingSolution doesn't support symmetry constraints
        return None


@dataclass
class ExistingTuckerSolution(ExistingSolution):
    """Parameters for using an existing Tucker tensor solution.

    Attributes
    ----------
    solution:
        Pre-existing ttensor solution.
    noise:
        Amount of Gaussian noise to add to solution.
        If data is sparse noise is only added to nonzero entries.
    """

    solution: ttb.ttensor


@dataclass
class ExistingCPSolution(ExistingSolution):
    """Parameters for using an existing CP tensor solution.

    Attributes
    ----------
    solution:
        Pre-existing ktensor solution.
    noise:
        Amount of Gaussian noise to add to solution.
        If data is sparse noise is only added to nonzero entries.
    sparse_generation:
        Generate a sparse tensor that can be scaled so that the
        column factors and weights are stochastic. Provide a number
        of nonzeros to be inserted. A value in range [0,1) will be
        interpreted as a ratio.
    """

    solution: ttb.ktensor
    sparse_generation: float | None = None


@dataclass
class MissingData:
    """Parameters to control missing data.

    Attributes
    ----------
    missing_ratio:
        Proportion of missing data.
    missing_pattern:
        An explicit tensor representing missing data locations.
    sparse_model:
        Whether to generate sparse rather than dense missing data pattern.
        Only useful for large tensors that don't easily fit in memory and
        when missing ratio > 0.8.
    """

    missing_ratio: float = 0.0
    missing_pattern: ttb.sptensor | ttb.tensor | None = None
    sparse_model: bool = False

    def __post_init__(self):
        if not 0.0 <= self.missing_ratio <= 1.0:
            raise ValueError(
                f"Missing ratio must be in [0,1] but got {self.missing_ratio}"
            )
        if self.missing_ratio > 0.0 and self.missing_pattern is not None:
            raise ValueError(
                "Can't set ratio and explicit pattern to specify missing data. "
                "Select one or the other."
            )

    def has_missing(self) -> bool:
        """Check if any form of missing data is requested."""
        return self.missing_ratio > 0.0 or self.missing_pattern is not None

    def raise_symmetric(self):
        """Raise for unsupported symmetry request."""
        if self.missing_ratio:
            raise ValueError("Can't generate a symmetric problem with missing data.")
        if self.sparse_model:
            raise ValueError("Can't generate sparse symmetric problem.")

    def get_pattern(self, shape: Shape) -> None | ttb.tensor | ttb.sptensor:
        """Generate a tensor pattern of missing data."""
        if self.missing_pattern is not None:
            if self.missing_pattern.shape != shape:
                raise ValueError(
                    "Missing pattern and problem shapes are not compatible."
                )
            return self.missing_pattern

        if self.missing_ratio == 0.0:
            # All usages of this are internal, should we just rule out this situation?
            return None
        if self.missing_ratio < 0.8 and self.sparse_model:
            logging.warning(
                "Setting sparse to false because there are"
                " fewer than 80% missing elements."
            )
        return _create_missing_data_pattern(
            shape, self.missing_ratio, self.sparse_model
        )


def _create_missing_data_pattern(
    shape: Shape, missing_ratio: float, sparse_model: bool = False
) -> ttb.tensor | ttb.sptensor:
    """Create a randomly missing element indicator tensor.

    Creates a binary tensor of specified size with 0's indication missing data
    and 1's indicating valid data. Will only return a tensor that has at least
    one entry per N-1 dimensional slice.
    """
    shape = parse_shape(shape)
    ndim = len(shape)
    P = math.prod(shape)
    Q = math.ceil((1 - missing_ratio) * P)
    W: ttb.tensor | ttb.sptensor

    # Create tensor
    ## Keep iterating until tensor is created or we give up.
    # TODO: make range configurable?
    for _ in range(20):
        if sparse_model:
            # Start with 50% more than Q random subs
            # Note in original matlab to work out expected value of a*Q to guarantee
            # Q unique entries
            subs = np.unique(
                np.floor(
                    np.random.random((int(np.ceil(1.5 * Q)), len(shape))).dot(
                        np.diag(shape)
                    )
                ),
                axis=0,
            ).astype(int)
            # Check if there are too many unique subs
            if len(subs) > Q:
                # TODO: check if note from matlab still relevant
                # Note in original matlab: unique orders the subs and would bias toward
                # first subs with lower values, so we sample to cut back
                idx = np.random.permutation(subs.shape[0])
                subs = subs[idx[:Q]]
            elif subs.shape[0] < Q:
                logging.warning(
                    f"Only generated {subs.shape[0]} of {Q} desired subscripts"
                )
            W = ttb.sptensor(
                subs,
                np.ones(
                    (len(subs), 1),
                ),
                shape=shape,
            )
        else:
            # Compute the linear indices of the missing entries.
            idx = np.random.permutation(P)
            idx = idx[:Q]
            W = ttb.tenzeros(shape)
            W[idx] = 1
        # return W

        # Check if W has any empty slices
        isokay = True
        for n in range(ndim):
            all_but_n = np.arange(W.ndims)
            all_but_n = np.delete(all_but_n, n)
            collapse_W = W.collapse(all_but_n)
            if isinstance(collapse_W, np.ndarray):
                isokay &= bool(np.all(collapse_W))
            else:
                isokay &= bool(np.all(collapse_W.double()))

        # Quit if okay
        if isokay:
            break

    if not isokay:
        raise ValueError(
            f"After {iter} iterations, cannot produce a tensor with"
            f"{missing_ratio * 100} missing data without an empty slice."
        )
    return W


@overload
def create_problem(
    problem_params: CPProblem, missing_params: MissingData | None = None
) -> tuple[
    ttb.ktensor, ttb.tensor | ttb.sptensor
]: ...  # pragma: no cover see coveragepy/issues/970


@overload
def create_problem(
    problem_params: TuckerProblem,
    missing_params: MissingData | None = None,
) -> tuple[ttb.ttensor, ttb.tensor]: ...  # pragma: no cover see coveragepy/issues/970


@overload
def create_problem(
    problem_params: ExistingSolution,
    missing_params: MissingData | None = None,
) -> tuple[
    ttb.ktensor | ttb.ttensor, ttb.tensor | ttb.sptensor
]: ...  # pragma: no cover see coveragepy/issues/970


def create_problem(
    problem_params: CPProblem | TuckerProblem | ExistingSolution,
    missing_params: MissingData | None = None,
) -> tuple[ttb.ktensor | ttb.ttensor, ttb.tensor | ttb.sptensor]:
    """Generate a problem and solution.

    Arguments
    ---------
    problem_params:
        Parameters related to the problem to generate, or an existing solution.
    missing_params:
        Parameters to control missing data in the generated data/solution.

    Examples
    --------
    Base example params

    >>> shape = (5, 4, 3)

    Generate a CP problem

    >>> cp_specific_params = CPProblem(shape=shape, num_factors=3, noise=0.1)
    >>> no_missing_data = MissingData()
    >>> solution, data = create_problem(cp_specific_params, no_missing_data)
    >>> diff = (solution.full() - data).norm() / solution.full().norm()
    >>> bool(np.isclose(diff, 0.1))
    True

    Generate Tucker Problem

    >>> tucker_specific_params = TuckerProblem(shape, num_factors=[3, 3, 2], noise=0.1)
    >>> solution, data = create_problem(tucker_specific_params, no_missing_data)
    >>> diff = (solution.full() - data).norm() / solution.full().norm()
    >>> bool(np.isclose(diff, 0.1))
    True

    Use existing solution

    >>> factor_matrices = [np.random.random((dim, 3)) for dim in shape]
    >>> weights = np.random.random(3)
    >>> existing_ktensor = ttb.ktensor(factor_matrices, weights)
    >>> existing_params = ExistingSolution(existing_ktensor, noise=0.1)
    >>> solution, data = create_problem(existing_params, no_missing_data)
    >>> assert solution is existing_ktensor

    Generate sparse count data from CP solution
    If we assume each model parameter is the input to a Poisson process, then
    we can generate a sparse test problems. This requires that all the factor
    matrices and lambda be nonnegative. The default factor generator ('randn')
    won't work since it produces both positive and negative values.

    >>> shape = (20, 15, 10)
    >>> num_factors = 4
    >>> A = []
    >>> for n in range(len(shape)):
    ...     A.append(np.random.rand(shape[n], num_factors))
    ...     for r in range(num_factors):
    ...         p = np.random.permutation(np.arange(shape[n]))
    ...         idx = p[1 : round(0.2 * shape[n])]
    ...         A[n][idx, r] *= 10
    >>> S = ttb.ktensor(A)
    >>> _ = S.normalize(sort=True)
    >>> existing_params = ExistingCPSolution(S, noise=0.0, sparse_generation=500)
    >>> solution, data = create_problem(existing_params)
    """
    if missing_params is None:
        missing_params = MissingData()

    if problem_params.symmetric is not None:
        missing_params.raise_symmetric()

    solution = generate_solution(problem_params)

    data: ttb.tensor | ttb.sptensor
    if (
        isinstance(problem_params, (CPProblem, ExistingCPSolution))
        and problem_params.sparse_generation is not None
    ):
        if missing_params.has_missing():
            raise ValueError(
                f"Can't combine missing data {MissingData.__name__} and "
                f" sparse generation {CPProblem.__name__}."
            )
        solution = cast("ttb.ktensor", solution)
        solution, data = generate_data_sparse(solution, problem_params)
    elif missing_params.has_missing():
        pattern = missing_params.get_pattern(solution.shape)
        data = generate_data(solution, problem_params, pattern)
    else:
        data = generate_data(solution, problem_params)
    return solution, data


def generate_solution_factors(base_params: BaseProblem) -> list[np.ndarray]:
    """Generate the factor matrices for either type of solution."""
    # Get shape of final tensor
    shape = cast("tuple[int, ...]", base_params.shape)

    # Get shape of factors
    if isinstance(base_params.num_factors, int):
        nfactors = [base_params.num_factors] * len(shape)
    elif base_params.num_factors is not None:
        nfactors = base_params.num_factors
    else:
        raise ValueError("Num_factors shouldn't be none.")
    if len(nfactors) != len(shape):
        raise ValueError(
            "Num_factors should be the same dimensions as shape but got"
            f"{nfactors} and {shape}"
        )
    factor_matrices = []
    for shape_i, nfactors_i in zip(shape, nfactors, strict=False):
        factor_matrices.append(base_params.factor_generator((shape_i, nfactors_i)))

    if base_params.symmetric is not None:
        for grp in base_params.symmetric:
            for j in range(1, len(grp)):
                factor_matrices[grp[j]] = factor_matrices[grp[0]]

    return factor_matrices


@overload
def generate_solution(
    problem_params: TuckerProblem,
) -> ttb.ttensor: ...


@overload
def generate_solution(
    problem_params: CPProblem,
) -> ttb.ktensor: ...


@overload
def generate_solution(
    problem_params: ExistingSolution,
) -> ttb.ktensor | ttb.ttensor: ...


def generate_solution(
    problem_params: CPProblem | TuckerProblem | ExistingSolution,
) -> ttb.ktensor | ttb.ttensor:
    """Generate problem solution."""
    if isinstance(problem_params, ExistingSolution):
        return problem_params.solution
    factor_matrices = generate_solution_factors(problem_params)
    # Create final model
    if isinstance(problem_params, TuckerProblem):
        nfactors = cast("list[int]", problem_params.num_factors)
        generated_core = problem_params.core_generator(tuple(nfactors))
        if isinstance(generated_core, (ttb.tensor, ttb.sptensor)):
            core = generated_core
        else:
            core = ttb.tensor(generated_core)
        return ttb.ttensor(core, factor_matrices)
    elif isinstance(problem_params, CPProblem):
        weights = problem_params.weight_generator((problem_params.num_factors,))
        return ttb.ktensor(factor_matrices, weights)
    raise ValueError(f"Unsupported problem parameter type: {type(problem_params)=}")


@overload
def generate_data(
    solution: ttb.ktensor | ttb.ttensor,
    problem_params: BaseProblem | ExistingSolution,
    pattern: ttb.tensor | None = None,
) -> ttb.tensor: ...  # pragma: no cover see coveragepy/issues/970


@overload
def generate_data(
    solution: ttb.ktensor | ttb.ttensor,
    problem_params: BaseProblem | ExistingSolution,
    pattern: ttb.sptensor,
) -> ttb.sptensor: ...  # pragma: no cover see coveragepy/issues/970


def generate_data(
    solution: ttb.ktensor | ttb.ttensor,
    problem_params: BaseProblem | ExistingSolution,
    pattern: ttb.tensor | ttb.sptensor | None = None,
) -> ttb.tensor | ttb.sptensor:
    """Generate problem data."""
    shape = solution.shape
    Rdm: ttb.tensor | ttb.sptensor
    if pattern is not None:
        if isinstance(pattern, ttb.sptensor):
            Rdm = ttb.sptensor(pattern.subs, randn((pattern.nnz, 1)), pattern.shape)
            Z = pattern * solution
        elif isinstance(pattern, ttb.tensor):
            Rdm = pattern * ttb.tensor(randn(shape))
            Z = pattern * solution.full()
        else:
            raise ValueError(f"Unsupported sparsity pattern of type {type(pattern)}")
    else:
        # TODO don't we already have a randn tensor method?
        Rdm = ttb.tensor(randn(shape))
        Z = solution.full()
        if problem_params.symmetric is not None:
            # TODO Note in MATLAB code to follow up
            Rdm = Rdm.symmetrize(np.array(problem_params.symmetric))

    D = Z + problem_params.noise * Z.norm() * Rdm / Rdm.norm()
    # Make sure the final result is definitely symmetric
    if problem_params.symmetric is not None:
        D = D.symmetrize(np.array(problem_params.symmetric))
    return D


def prosample(nsamples: int, prob: np.ndarray) -> np.ndarray:
    """Proportional Sampling."""
    bins = np.minimum(np.cumsum(np.array([0, *prob])), 1)
    bins[-1] = 1
    indices = np.digitize(np.random.random(nsamples), bins=bins)
    return indices - 1


def generate_data_sparse(
    solution: ttb.ktensor,
    problem_params: CPProblem | ExistingCPSolution,
) -> tuple[ttb.ktensor, ttb.sptensor]:
    """Generate sparse CP data from a given solution."""
    # Error check on solution
    if np.any(solution.weights < 0):
        raise ValueError("All weights must be nonnegative.")
    if any(np.any(factor < 0) for factor in solution.factor_matrices):
        raise ValueError("All factor matrices must be nonnegative.")
    if problem_params.symmetric is not None:
        logging.warning("Symmetric constraints have been ignored.")
    if problem_params.sparse_generation is None:
        raise ValueError("Cannot generate sparse data without sparse_generation set.")

    # Convert solution to probability tensor
    # NOTE: Make copy since normalize modifies in place
    P = solution.copy().normalize(normtype=1)
    eta = np.sum(P.weights)
    P.weights /= eta

    # Determine how many samples per component
    nedges = problem_params.sparse_generation
    if nedges < 1:
        nedges = np.round(nedges * math.prod(P.shape)).astype(int)
    nedges = int(nedges)
    nd = P.ndims
    nc = P.ncomponents
    csample = prosample(nedges, P.weights)
    # TODO check this
    csums = accumarray(csample, 1, size=nc)

    # Determine the subscripts for each randomly sampled entry
    shape = solution.shape
    subs: list[np.ndarray] = []
    for c in range(nc):
        nsample = csums[c]
        if nsample == 0:
            continue
        subs.append(np.zeros((nsample, nd), dtype=int))
        for d in range(nd):
            subs[-1][:, d] = prosample(nsample, P.factor_matrices[d][:, c])
    # TODO could sum csums and allocate in place with slicing
    allsubs = np.vstack(subs)
    # Assemble final tensor. Note that duplicates are summed.
    # TODO should we have sptenones for purposes like this?
    Z = ttb.sptensor.from_aggregator(
        allsubs,
        np.ones(
            (len(allsubs), 1),
        ),
        shape=shape,
    )

    # Rescale S so that it is proportional to the number of edges inserted
    solution = P
    # raise ValueError(
    #    f"{nedges=}"
    #    f"{solution.weights=}"
    # )
    solution.weights *= nedges

    # TODO no noise introduced in this special case in MATLAB

    return solution, Z
