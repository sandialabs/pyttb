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


@pytest.fixture()
def sample_tensor_3way():
    shape = (3, 3, 3)
    data = np.array(range(1, 28)).reshape(shape, order="F")
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


@pytest.mark.indevelopment
def test_hosvd_3way(capsys, sample_tensor_3way):
    (data, T) = sample_tensor_3way
    M = ttb.hosvd(T, 1e-4, verbosity=0)
    capsys.readouterr()
    print(f"M=\n{M}")
    core = np.array(
        [
            [
                [-8.301598119750199e01, -5.005881796972034e-03],
                [-1.268039597172832e-02, 5.842630378620833e00],
            ],
            [
                [3.709974006281391e-02, -1.915213813096568e00],
                [-5.157111619887230e-01, 5.243776123493664e-01],
            ],
        ]
    )
    fm0 = np.array(
        [
            [-5.452132631706279e-01, -7.321719955012304e-01],
            [-5.767748638548937e-01, -2.576993904719336e-02],
            [-6.083364645391598e-01, 6.806321174064961e-01],
        ]
    )
    fm1 = np.array(
        [
            [-4.756392343758577e-01, 7.791666394653051e-01],
            [-5.719678320081717e-01, 7.865197061237804e-02],
            [-6.682964296404851e-01, -6.218626982406427e-01],
        ]
    )
    fm2 = np.array(
        [
            [-1.922305666539489e-01, 8.924016710972924e-01],
            [-5.140779746206554e-01, 2.627873081852611e-01],
            [-8.359253825873615e-01, -3.668270547267537e-01],
        ]
    )
    expected = ttb.ttensor.from_data(ttb.tensor.from_data(core), [fm0, fm1, fm2])
    assert np.allclose(M.double(), expected.double())
    assert np.allclose(np.abs(M.core.data), np.abs(core))
    assert np.allclose(np.abs(M.u[0]), np.abs(fm0))
    assert np.allclose(np.abs(M.u[1]), np.abs(fm1))
    assert np.allclose(np.abs(M.u[2]), np.abs(fm2))
