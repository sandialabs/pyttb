# Copyright 2022 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

import numpy as np
import pytest
import scipy.sparse as sparse

import pyttb as ttb


@pytest.fixture()
def sample_ttensor():
    """Simple TTENSOR to verify by hand"""
    core = ttb.tensor.from_data(np.ones((2, 2, 2)))
    factors = [np.ones((1, 2))] * len(core.shape)
    ttensorInstance = ttb.ttensor().from_data(core, factors)
    return ttensorInstance


@pytest.fixture()
def random_ttensor():
    """Arbitrary TTENSOR to verify consistency between alternative operations"""
    core = ttb.tensor.from_data(np.random.random((2, 3, 4)))
    factors = [
        np.random.random((5, 2)),
        np.random.random((2, 3)),
        np.random.random((4, 4)),
    ]
    ttensorInstance = ttb.ttensor().from_data(core, factors)
    return ttensorInstance


@pytest.mark.indevelopment
def test_ttensor_initialization_empty():
    empty_tensor = ttb.tensor()

    # No args
    ttensorInstance = ttb.ttensor()
    assert ttensorInstance.core == empty_tensor
    assert ttensorInstance.u == []


@pytest.mark.indevelopment
def test_ttensor_initialization_from_data(sample_ttensor):
    ttensorInstance = sample_ttensor
    assert isinstance(ttensorInstance.core, ttb.tensor)
    assert all([isinstance(a_factor, np.ndarray) for a_factor in ttensorInstance.u])

    # Negative Tests
    non_array_factor = ttensorInstance.u + [1]
    with pytest.raises(ValueError):
        ttb.ttensor.from_data(ttensorInstance.core, non_array_factor[1:])

    non_matrix_factor = ttensorInstance.u + [np.array([1])]
    with pytest.raises(ValueError):
        ttb.ttensor.from_data(ttensorInstance.core, non_matrix_factor[1:])

    too_few_factors = ttensorInstance.u.copy()
    too_few_factors.pop()
    with pytest.raises(ValueError):
        ttb.ttensor.from_data(ttensorInstance.core, too_few_factors)

    wrong_shape_factor = ttensorInstance.u.copy()
    row, col = wrong_shape_factor[0].shape
    wrong_shape_factor[0] = np.random.random((row + 1, col + 1))
    with pytest.raises(ValueError):
        ttb.ttensor.from_data(ttensorInstance.core, wrong_shape_factor)

    # Enforce error until sptensor core/other cores supported
    with pytest.raises(ValueError):
        ttb.ttensor.from_data(
            ttb.sptensor.from_tensor_type(ttensorInstance.core), ttensorInstance.u
        )


@pytest.mark.indevelopment
def test_ttensor_initialization_from_tensor_type(sample_ttensor):
    # Copy constructor
    ttensorInstance = sample_ttensor
    ttensorCopy = ttb.ttensor.from_tensor_type(ttensorInstance)
    assert ttensorCopy.core == ttensorInstance.core
    assert ttensorCopy.u == ttensorInstance.u
    assert ttensorCopy.shape == ttensorInstance.shape


@pytest.mark.indevelopment
def test_ttensor_full(sample_ttensor):
    ttensorInstance = sample_ttensor
    tensor = ttensorInstance.full()
    # This sanity check only works for all 1's
    assert tensor.double() == np.prod(ttensorInstance.core.shape)

    # Negative tests
    sparse_core = ttb.sptensor()
    sparse_core.shape = ttensorInstance.core.shape
    sparse_u = [
        sparse.coo_matrix(np.zeros(factor.shape)) for factor in ttensorInstance.u
    ]
    # We could probably make these properties to avoid this edge case but expect to eventually cover these alternate
    # cores
    ttensorInstance.core = sparse_core
    ttensorInstance.u = sparse_u
    with pytest.raises(ValueError):
        ttensorInstance.full()


@pytest.mark.indevelopment
def test_ttensor_double(sample_ttensor):
    ttensorInstance = sample_ttensor
    # This sanity check only works for all 1's
    assert ttensorInstance.double() == np.prod(ttensorInstance.core.shape)


@pytest.mark.indevelopment
def test_ttensor_ndims(sample_ttensor):
    ttensorInstance = sample_ttensor

    assert ttensorInstance.ndims == 3


@pytest.mark.indevelopment
def test_ttensor__pos__(sample_ttensor):
    ttensorInstance = sample_ttensor
    ttensorInstance2 = +ttensorInstance

    assert ttensorInstance.isequal(ttensorInstance2)


@pytest.mark.indevelopment
def test_sptensor__neg__(sample_ttensor):
    ttensorInstance = sample_ttensor
    ttensorInstance2 = -ttensorInstance
    ttensorInstance3 = -ttensorInstance2

    assert not ttensorInstance.isequal(ttensorInstance2)
    assert ttensorInstance.isequal(ttensorInstance3)


@pytest.mark.indevelopment
def test_ttensor_innerproduct(sample_ttensor, random_ttensor):
    ttensorInstance = sample_ttensor

    # TODO these are an overly simplistic edge case for ttensors that are a single float

    # ttensor innerprod ttensor
    assert ttensorInstance.innerprod(ttensorInstance) == ttensorInstance.double() ** 2
    core_dim = ttensorInstance.core.shape[0] + 1
    ndim = ttensorInstance.ndims
    large_core_ttensor = ttb.ttensor.from_data(
        ttb.tensor.from_data(np.ones((core_dim,) * ndim)),
        [np.ones((1, core_dim))] * ndim,
    )
    assert large_core_ttensor.innerprod(
        ttensorInstance
    ) == ttensorInstance.full().innerprod(large_core_ttensor.full())

    # ttensor innerprod tensor
    assert (
        ttensorInstance.innerprod(ttensorInstance.full())
        == ttensorInstance.double() ** 2
    )

    # ttensr innerprod ktensor
    ktensorInstance = ttb.ktensor.from_data(np.array([8.0]), [np.array([[1.0]])] * 3)
    assert ttensorInstance.innerprod(ktensorInstance) == ttensorInstance.double() ** 2

    # ttensor innerprod tensor (shape larger than core)
    random_ttensor.innerprod(random_ttensor.full())

    # Negative Tests
    ttensor_extra_factors = ttb.ttensor.from_tensor_type(ttensorInstance)
    ttensor_extra_factors.u.extend(ttensorInstance.u)
    with pytest.raises(ValueError):
        ttensorInstance.innerprod(ttensor_extra_factors)

    tensor_extra_dim = ttb.tensor.from_data(np.ones(ttensorInstance.shape + (1,)))
    with pytest.raises(ValueError):
        ttensorInstance.innerprod(tensor_extra_dim)

    invalid_option = []
    with pytest.raises(ValueError):
        ttensorInstance.innerprod(invalid_option)


@pytest.mark.indevelopment
def test_ttensor__mul__(sample_ttensor):
    ttensorInstance = sample_ttensor
    mul_factor = 2

    # This sanity check only works for all 1's
    assert (ttensorInstance * mul_factor).double() == np.prod(
        ttensorInstance.core.shape
    ) * mul_factor
    assert (ttensorInstance * float(2)).double() == np.prod(
        ttensorInstance.core.shape
    ) * float(mul_factor)

    # Negative tests
    with pytest.raises(ValueError):
        _ = ttensorInstance * "some_string"


@pytest.mark.indevelopment
def test_ttensor__rmul__(sample_ttensor):
    ttensorInstance = sample_ttensor
    mul_factor = 2

    # This sanity check only works for all 1's
    assert (mul_factor * ttensorInstance).double() == np.prod(
        ttensorInstance.core.shape
    ) * mul_factor
    assert (float(2) * ttensorInstance).double() == np.prod(
        ttensorInstance.core.shape
    ) * float(mul_factor)

    # Negative tests
    with pytest.raises(ValueError):
        _ = "some_string" * ttensorInstance


@pytest.mark.indevelopment
def test_ttensor_ttv(sample_ttensor):
    ttensorInstance = sample_ttensor
    mul_factor = 1
    trivial_vectors = [np.array([mul_factor])] * len(ttensorInstance.shape)
    final_value = sample_ttensor.ttv(trivial_vectors)
    assert final_value == np.prod(ttensorInstance.core.shape)

    assert np.allclose(
        ttensorInstance.ttv(trivial_vectors[0], 0).double(),
        ttensorInstance.full().ttv(trivial_vectors[0], 0).double(),
    )

    # Negative tests
    wrong_shape_vector = trivial_vectors.copy()
    wrong_shape_vector[0] = np.array([mul_factor, mul_factor])
    with pytest.raises(ValueError):
        sample_ttensor.ttv(wrong_shape_vector)


@pytest.mark.indevelopment
def test_ttensor_mttkrp(random_ttensor):
    ttensorInstance = random_ttensor
    column_length = 6
    vectors = [np.random.random((u.shape[0], column_length)) for u in ttensorInstance.u]
    final_value = ttensorInstance.mttkrp(vectors, 2)
    full_value = ttensorInstance.full().mttkrp(vectors, 2)
    assert np.allclose(final_value, full_value), (
        f"TTensor value is: \n{final_value}\n\n" f"Full value is: \n{full_value}"
    )


@pytest.mark.indevelopment
def test_ttensor_norm(sample_ttensor, random_ttensor):
    ttensorInstance = random_ttensor
    assert np.isclose(ttensorInstance.norm(), ttensorInstance.full().norm())

    # Core larger than full tensor
    ttensorInstance = sample_ttensor
    assert np.isclose(ttensorInstance.norm(), ttensorInstance.full().norm())


@pytest.mark.indevelopment
def test_ttensor_permute(random_ttensor):
    ttensorInstance = random_ttensor
    original_order = np.arange(0, len(ttensorInstance.core.shape))
    permuted_tensor = ttensorInstance.permute(original_order)
    assert ttensorInstance.isequal(permuted_tensor)

    # Negative Tests
    with pytest.raises(ValueError):
        bad_permutation_order = np.arange(0, len(ttensorInstance.core.shape) + 1)
        ttensorInstance.permute(bad_permutation_order)


@pytest.mark.indevelopment
def test_ttensor_ttm(random_ttensor):
    ttensorInstance = random_ttensor
    row_length = 9
    matrices = [np.random.random((row_length, u.shape[0])) for u in ttensorInstance.u]
    final_value = ttensorInstance.ttm(matrices, np.arange(len(matrices)))
    reverse_value = ttensorInstance.ttm(
        list(reversed(matrices)), np.arange(len(matrices) - 1, -1, -1)
    )
    assert final_value.isequal(reverse_value), (
        f"TTensor value is: \n{final_value}\n\n" f"Full value is: \n{reverse_value}"
    )
    final_value = ttensorInstance.ttm(matrices)  # No dims
    assert final_value.isequal(reverse_value)
    final_value = ttensorInstance.ttm(
        matrices, list(range(len(matrices)))
    )  # Dims as list
    assert final_value.isequal(reverse_value)

    single_tensor_result = ttensorInstance.ttm(matrices[0], 0)
    single_tensor_full_result = ttensorInstance.full().ttm(matrices[0], 0)
    assert np.allclose(
        single_tensor_result.double(), single_tensor_full_result.double()
    ), (
        f"TTensor value is: \n{single_tensor_result.full()}\n\n"
        f"Full value is: \n{single_tensor_full_result}"
    )

    transposed_matrices = [matrix.transpose() for matrix in matrices]
    transpose_value = ttensorInstance.ttm(
        transposed_matrices, np.arange(len(matrices)), transpose=True
    )
    assert final_value.isequal(transpose_value)

    # Negative Tests
    big_wrong_size = 123
    bad_matrices = matrices.copy()
    bad_matrices[0] = np.random.random((big_wrong_size, big_wrong_size))
    with pytest.raises(ValueError):
        _ = ttensorInstance.ttm(bad_matrices, np.arange(len(bad_matrices)))

    with pytest.raises(ValueError):
        # Negative dims currently broken, ensure we catch early and
        # remove once resolved
        ttensorInstance.ttm(matrices, -1)


@pytest.mark.indevelopment
def test_ttensor_reconstruct(random_ttensor):
    ttensorInstance = random_ttensor
    # TODO: This slice drops the singleton dimension, should it? If so should ttensor squeeze during reconstruct?
    full_slice = ttensorInstance.full()[:, 1, :]
    ttensor_slice = ttensorInstance.reconstruct(1, 1)
    assert np.allclose(full_slice.double(), ttensor_slice.squeeze().double())
    assert ttensorInstance.reconstruct().isequal(ttensorInstance.full())
    sample_all_modes = [np.array([0])] * len(ttensorInstance.shape)
    sample_all_modes[-1] = 0  # Make raw scalar
    reconstruct_scalar = ttensorInstance.reconstruct(sample_all_modes).full().double()
    full_scalar = ttensorInstance.full()[tuple(sample_all_modes)]
    assert np.isclose(reconstruct_scalar, full_scalar)

    scale = np.random.random(ttensorInstance.u[1].shape).transpose()
    _ = ttensorInstance.reconstruct(scale, 1)
    # FIXME from the MATLAB docs wasn't totally clear how to validate this

    # Negative Tests
    with pytest.raises(ValueError):
        _ = ttensorInstance.reconstruct(1, [0, 1])


@pytest.mark.indevelopment
def test_ttensor_nvecs(random_ttensor):
    ttensorInstance = random_ttensor
    n = 0
    r = 2
    ttensor_eigvals = ttensorInstance.nvecs(n, r)
    full_eigvals = ttensorInstance.full().nvecs(n, r)
    assert np.allclose(ttensor_eigvals, full_eigvals)

    # Test for eig vals larger than shape-1
    n = 1
    r = 2
    full_eigvals = ttensorInstance.full().nvecs(n, r)
    with pytest.warns(Warning) as record:
        ttensor_eigvals = ttensorInstance.nvecs(n, r)
    assert (
        "Greater than or equal to tensor.shape[n] - 1 eigenvectors requires cast to dense to solve"
        in str(record[0].message)
    )
    assert np.allclose(ttensor_eigvals, full_eigvals)

    # Negative Tests
    sparse_core = ttb.sptensor()
    sparse_core.shape = ttensorInstance.core.shape
    ttensorInstance.core = sparse_core

    # Sparse core
    with pytest.raises(NotImplementedError):
        ttensorInstance.nvecs(0, 1)

    # Sparse factors
    sparse_u = [
        sparse.coo_matrix(np.zeros(factor.shape)) for factor in ttensorInstance.u
    ]
    ttensorInstance.u = sparse_u
    with pytest.raises(NotImplementedError):
        ttensorInstance.nvecs(0, 1)


@pytest.mark.indevelopment
def test_sptensor_isequal(sample_ttensor):
    ttensorInstance = sample_ttensor
    # Negative Tests
    assert not ttensorInstance.isequal(ttensorInstance.full())
    ttensor_extra_factors = ttb.ttensor.from_tensor_type(ttensorInstance)
    ttensor_extra_factors.u.extend(ttensorInstance.u)
    assert not ttensorInstance.isequal(ttensor_extra_factors)
