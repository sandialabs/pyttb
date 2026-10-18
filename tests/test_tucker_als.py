# Copyright 2024 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.
from __future__ import annotations

import numpy as np
import pytest

import pyttb as ttb


@pytest.fixture()
def sample_tensor():
    data = np.array([[29, 39.0], [63.0, 85.0]])
    shape = (2, 2)
    params = {"data": data, "shape": shape}
    tensorInstance = ttb.tensor(data, shape)
    return params, tensorInstance


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

    (Solution, Uinit, output) = ttb.tucker_als(T, 2, dimorder=[1, 0])
    capsys.readouterr()
    assert pytest.approx(output["fit"], 1) == 0


def test_tucker_als_tensor_ttensor_init(capsys, sample_tensor):
    (data, T) = sample_tensor
    (Solution, Uinit, output) = ttb.tucker_als(T, 2, printitn=0)
    (Solution2, _, output2) = ttb.tucker_als(T, 2, init=Solution, printitn=0)
    capsys.readouterr()
    assert pytest.approx(output2["fit"], 1) == 0
    assert np.all(np.isclose(Solution2.double(), T.double()))


def test_tucker_als_tensor_pass_params(capsys, sample_tensor):
    (data, T) = sample_tensor
    (_, _, output) = ttb.tucker_als(T, 2, printitn=0)
    assert set(output["params"]) == {"stoptol", "maxiters", "printitn", "dimorder"}
    (_, _, output2) = ttb.tucker_als(T, 2, **output["params"])
    capsys.readouterr()
    assert pytest.approx(output2["fit"], 1) == 0


def test_tucker_als_nvecs_silent_when_printitn_zero(capsys, sample_tensor):
    (data, T) = sample_tensor
    ttb.tucker_als(T, 2, init="nvecs", printitn=0)
    assert capsys.readouterr().out == ""
    ttb.tucker_als(T, 2, init="nvecs", printitn=1)
    assert "leading e-vector" in capsys.readouterr().out


def test_tucker_als_tensor_incorrect_init(sample_tensor):
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


def test_tucker_als_tensor_incorrect_steptol(sample_tensor):
    (data, T) = sample_tensor

    non_scalar = np.array([1])
    with pytest.raises(ValueError):
        _ = ttb.tucker_als(T, 2, stoptol=non_scalar)


def test_tucker_als_tensor_incorrect_maxiters(sample_tensor):
    (data, T) = sample_tensor

    negative_value = -1
    with pytest.raises(ValueError):
        _ = ttb.tucker_als(T, 2, maxiters=negative_value)

    non_scalar = np.array([1])
    with pytest.raises(ValueError):
        _ = ttb.tucker_als(T, 2, maxiters=non_scalar)


def test_tucker_als_tensor_incorrect_printitn(sample_tensor):
    (data, T) = sample_tensor

    non_scalar = np.array([1])
    with pytest.raises(ValueError):
        _ = ttb.tucker_als(T, 2, printitn=non_scalar)


def test_tucker_als_tensor_incorrect_dimorder(sample_tensor):
    (data, T) = sample_tensor

    non_list = np.array([1])  # TODO: Consider generalizing to iterable
    with pytest.raises(ValueError):
        _ = ttb.tucker_als(T, 2, dimorder=non_list)

    too_few_dims = list(range(len(T.shape)))
    too_few_dims.pop()
    with pytest.raises(ValueError):
        _ = ttb.tucker_als(T, 2, dimorder=too_few_dims)
