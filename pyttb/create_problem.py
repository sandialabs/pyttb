"""Create test problems for  tensor factorizations."""

import logging
import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union, cast, overload

import numpy as np
from numpy_groupies import aggregate as accumarray

import pyttb as ttb
from pyttb.pyttb_utils import Shape

solution_generator = Callable[[Tuple[int, ...]], np.ndarray]


def randn(shape: Tuple[int, ...]) -> np.ndarray:
    """Stub for MATLAB randn.

    TODO move somewhere shareable.
    """
    return np.random.normal(0, 1, size=shape)


@dataclass
class BaseProblem:
    """Parameters general to all solutions."""

    shape: Shape
    factor_generator: solution_generator = randn
    symmetric: Optional[list[Tuple[int, int]]] = None
    num_factors: Union[int, list[int], None] = None

    def __post_init__(self):
        self.shape = ttb.pyttb_utils.parse_shape(self.shape)


@dataclass
class CPProblem(BaseProblem):
    """Parameters specifying CP Solutions."""

    num_factors: int = 2
    weight_generator: solution_generator = np.random.random
    # TODO: This is in DataParams in MATLAB, but only works for CP problems
    sparse_generation: Optional[float] = None


@dataclass
class TuckerProblem(BaseProblem):
    """Parameters specifying Tucker Solutions."""

    # TODO post_init set to [2, 2, 2]
    num_factors: Optional[list[int]] = None
    core_generator: solution_generator = randn

    def __post_init__(self):
        super().__post_init__()
        self.num_factors = self.num_factors or [2, 2, 2]


@dataclass
class DataParams:
    """Parameters to control data quality."""

    noise: float = 0.10

    def __post_init__(
        self,
    ):
        if not 0.0 <= self.noise <= 1.0:
            raise ValueError(f"Noise must be in [0,1] but got {self.noise}")


@dataclass
class MissingData:
    """Parameters to control missing data."""

    missing_ratio: float = 0.0
    sparse_model: bool = False
    # TODO add spare pattern tensor

    def __post_init__(self):
        if not 0.0 <= self.missing_ratio <= 1.0:
            raise ValueError(
                f"Missing ratio must be in [0,1] but got {self.missing_ratio}"
            )

        if self.sparse_model and self.missing_ratio > 0.0:
            raise ValueError("Can't combine missing data and sparse generation.")

    def has_missing(self) -> bool:
        """Check if any form of missing data is requested."""
        return self.sparse_model or self.missing_ratio > 0.0

    def raise_symmetric(self):
        """Raise for unsupported symmetry request."""
        if self.missing_ratio:
            raise ValueError("Can't generate a symmetric problem with missing data.")
        if self.sparse_model:
            raise ValueError("Can't generate sparse symmetric problem.")


@overload
def create_problem(
    problem_params: CPProblem, missing_params: MissingData, data_params: DataParams
) -> Tuple[
    ttb.ktensor, Union[ttb.tensor, ttb.sptensor]
]: ...  # pragma: no cover see coveragepy/issues/970


@overload
def create_problem(
    problem_params: TuckerProblem, missing_params: MissingData, data_params: DataParams
) -> Tuple[ttb.ttensor, ttb.tensor]: ...  # pragma: no cover see coveragepy/issues/970


def create_problem(
    problem_params: Union[CPProblem, TuckerProblem],
    missing_params: MissingData,
    data_params: DataParams,
) -> Tuple[Union[ttb.ktensor, ttb.ttensor], Union[ttb.tensor, ttb.sptensor]]:
    """Generate a problem and solution."""
    if problem_params.symmetric is not None:
        missing_params.raise_symmetric()

    solution = generate_solution(problem_params)

    if missing_params.sparse_model:
        raise NotImplementedError("Sparse generation not yet supported")

    data: Union[ttb.tensor, ttb.sptensor]
    if (
        isinstance(problem_params, CPProblem)
        and problem_params.sparse_generation is not None
    ):
        solution = cast(ttb.ktensor, solution)
        solution, data = generate_data_sparse(solution, problem_params, data_params)
    else:
        data = generate_data(solution, problem_params, data_params)
    return solution, data


def generate_solution_factors(base_params: BaseProblem) -> list[np.ndarray]:
    """Generate the factor matrices for either type of solution."""
    # Get shape of final tensor
    shape = cast(Tuple[int, ...], base_params.shape)

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
    for shape_i, nfactors_i in zip(shape, nfactors):
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


def generate_solution(
    problem_params: Union[CPProblem, TuckerProblem],
) -> Union[ttb.ktensor, ttb.ttensor]:
    """Generate problem solution."""
    factor_matrices = generate_solution_factors(problem_params)
    # Create final model
    if isinstance(problem_params, TuckerProblem):
        nfactors = cast(list[int], problem_params.num_factors)
        core = ttb.tensor(problem_params.core_generator(tuple(nfactors)))
        return ttb.ttensor(core, factor_matrices)
    elif isinstance(problem_params, CPProblem):
        weights = problem_params.weight_generator((problem_params.num_factors,))
        return ttb.ktensor(factor_matrices, weights)
    raise ValueError(f"Unsupported problem parameter type: {type(problem_params)=}")


def generate_data(
    solution: Union[ttb.ktensor, ttb.ttensor],
    problem_params: BaseProblem,
    data_params: DataParams,
) -> ttb.tensor:
    """Generate problem data."""
    shape = solution.shape
    # TODO handle the sparsity pattern
    # TODO don't we already have a randn tensor method?
    Rdm = ttb.tensor(randn(shape))
    Z = solution.full()
    if problem_params.symmetric is not None:
        # TODO Note in MATLAB code to follow up
        Rdm = Rdm.symmetrize(np.array(problem_params.symmetric))

    D = Z + data_params.noise * Z.norm() * Rdm / Rdm.norm()
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
    solution: ttb.ktensor, problem_params: CPProblem, data_params: DataParams
) -> Tuple[ttb.ktensor, ttb.sptensor]:
    """Generate sparse CP data from a given solution."""
    # Error check on solution
    if np.any(solution.weights < 0):
        raise ValueError("All weights must be nonnegative.")
    if any(np.any(factor < 0) for factor in solution.factor_matrices):
        raise ValueError("All factor matrices must be nonnegative.")
    if problem_params.symmetric is not None:
        logging.warning("Summetric constraints have been ignored.")
    if problem_params.sparse_generation is None:
        raise ValueError("Cannot generate sparse data without sparse_generation set.")

    # Convert solution to probability tensor
    P = solution.normalize(mode=0)
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
    Z = ttb.sptensor(
        allsubs,
        np.ones(
            len(allsubs),
        ),
        shape=shape,
    )

    # Rescale S so that it is proportional to the number of edges inserted
    solution = P
    solution.weights *= nedges

    # TODO no noise introduced in this special case in MATLAB

    return solution, Z
