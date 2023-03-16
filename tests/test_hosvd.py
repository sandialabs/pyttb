import numpy as np
import pytest

import pyttb as ttb


@pytest.fixture()
def sample_tensor():
    data = np.array([[29, 39.0], [63.0, 85.0]])
    shape = (2, 2)
    params = {"data": data, "shape": shape}
    tensorInstance = ttb.tensor().from_data(data, shape)
    return params, tensorInstance


@pytest.mark.indevelopment
def test_hosvd_simple_convergence(capsys, sample_tensor):
    (data, T) = sample_tensor
    tol = 1e-4
    result = ttb.hosvd(T, tol)
    assert (result.full() - T).norm() / T.norm() < tol, f"Failed to converge"

    tol = 1e-4
    result = ttb.hosvd(T, tol, sequential=False)
    assert (
        result.full() - T
    ).norm() / T.norm() < tol, f"Failed to converge for non-sequential option"

    impossible_tol = 1e-20
    with pytest.warns(UserWarning):
        result = ttb.hosvd(T, impossible_tol)
    assert (
        result.full() - T
    ).norm() / T.norm() > impossible_tol, f"Converged beyond provided precision"


@pytest.mark.indevelopment
def test_hosvd_default_init(capsys, sample_tensor):
    (data, T) = sample_tensor
    _ = ttb.hosvd(T, 1)


@pytest.mark.indevelopment
def test_hosvd_smoke_test_verbosity(capsys, sample_tensor):
    """For now just make sure verbosity calcs don't crash"""
    (data, T) = sample_tensor
    ttb.hosvd(T, 1, verbosity=10)


@pytest.mark.indevelopment
def test_hosvd_incorrect_ranks(capsys, sample_tensor):
    (data, T) = sample_tensor
    ranks = list(range(T.ndims - 1))
    with pytest.raises(ValueError):
        _ = ttb.hosvd(T, 1, ranks=ranks)


@pytest.mark.indevelopment
def test_hosvd_incorrect_dimorder(capsys, sample_tensor):
    (data, T) = sample_tensor
    dimorder = list(range(T.ndims - 1))
    with pytest.raises(ValueError):
        _ = ttb.hosvd(T, 1, dimorder=dimorder)

    dimorder = 1
    with pytest.raises(ValueError):
        _ = ttb.hosvd(T, 1, dimorder=dimorder)
