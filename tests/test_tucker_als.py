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
def test_tucker_als_tensor_default_init(capsys, sample_tensor):
    (data, T) = sample_tensor
    (Solution, Uinit, output) = ttb.tucker_als(T, 2)
    capsys.readouterr()
    assert pytest.approx(output["fit"], 1) == 0
    assert np.all(np.isclose(Solution.double(), T.double()))

    (Solution, Uinit, output) = ttb.tucker_als(T, 2, init=Uinit)
    capsys.readouterr()
    assert pytest.approx(output["fit"], 1) == 0

    (Solution, Uinit, output) = ttb.tucker_als(T, 2, init="nvecs")
    capsys.readouterr()
    assert pytest.approx(output["fit"], 1) == 0


@pytest.mark.indevelopment
def test_tucker_als_tensor_incorrect_init(capsys, sample_tensor):
    (data, T) = sample_tensor

    non_list = np.array([1])  # TODO: Consider generalizing to iterable
    with pytest.raises(ValueError):
        _ = ttb.tucker_als(T, 2, init=non_list)

    bad_string = "foo_bar"
    with pytest.raises(ValueError):
        _ = ttb.tucker_als(T, 2, init=bad_string)

    wrong_length = [np.ones(T.shape)] * T.ndims
    wrong_length.pop()
    with pytest.raises(ValueError):
        _ = ttb.tucker_als(T, 2, init=wrong_length)

    wrong_shape = [np.ones(5)] * T.ndims
    with pytest.raises(ValueError):
        _ = ttb.tucker_als(T, 2, init=wrong_shape)


@pytest.mark.indevelopment
def test_tucker_als_tensor_incorrect_steptol(capsys, sample_tensor):
    (data, T) = sample_tensor

    non_scalar = np.array([1])
    with pytest.raises(ValueError):
        _ = ttb.tucker_als(T, 2, stoptol=non_scalar)


@pytest.mark.indevelopment
def test_tucker_als_tensor_incorrect_maxiters(capsys, sample_tensor):
    (data, T) = sample_tensor

    negative_value = -1
    with pytest.raises(ValueError):
        _ = ttb.tucker_als(T, 2, maxiters=negative_value)

    non_scalar = np.array([1])
    with pytest.raises(ValueError):
        _ = ttb.tucker_als(T, 2, maxiters=non_scalar)


@pytest.mark.indevelopment
def test_tucker_als_tensor_incorrect_printitn(capsys, sample_tensor):
    (data, T) = sample_tensor

    non_scalar = np.array([1])
    with pytest.raises(ValueError):
        _ = ttb.tucker_als(T, 2, printitn=non_scalar)


@pytest.mark.indevelopment
def test_tucker_als_tensor_incorrect_dimorder(capsys, sample_tensor):
    (data, T) = sample_tensor

    non_list = np.array([1])  # TODO: Consider generalizing to iterable
    with pytest.raises(ValueError):
        _ = ttb.tucker_als(T, 2, dimorder=non_list)

    too_few_dims = list(range(len(T.shape)))
    too_few_dims.pop()
    with pytest.raises(ValueError):
        _ = ttb.tucker_als(T, 2, dimorder=too_few_dims)
