# Copyright 2022 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

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
def sample_sptensor():
    subs = np.array([[0, 0], [1, 0], [1, 1]])
    vals = np.array([[0.5], [0.5], [0.5]])
    shape = (2, 2)
    data = {"subs": subs, "vals": vals, "shape": shape}
    sptensorInstance = ttb.sptensor.from_data(subs, vals, shape)
    return data, sptensorInstance


@pytest.mark.indevelopment
def test_cp_als_tensor_default_init(capsys, sample_tensor):
    (data, T) = sample_tensor
    (M, Minit, output) = ttb.cp_als(T, 2)
    capsys.readouterr()
    assert pytest.approx(output["fit"], 1) == 0


@pytest.mark.indevelopment
def test_cp_als_tensor_nvecs_init(capsys, sample_tensor):
    (data, T) = sample_tensor
    with pytest.warns(Warning) as record:
        (M, Minit, output) = ttb.cp_als(T, 1, init="nvecs")
    assert (
        "Greater than or equal to tensor.shape[n] - 1 eigenvectors requires cast to dense to solve"
        in str(record[0].message)
    )
    capsys.readouterr()
    assert pytest.approx(output["fit"], 1) == 0


@pytest.mark.indevelopment
def test_cp_als_tensor_ktensor_init(capsys, sample_tensor):
    (data, T) = sample_tensor
    KInit = ttb.ktensor.from_function(np.random.random_sample, T.shape, 2)
    (M, Minit, output) = ttb.cp_als(T, 2, init=KInit)
    capsys.readouterr()
    assert pytest.approx(output["fit"], 1) == 0


@pytest.mark.indevelopment
def test_cp_als_incorrect_init(capsys, sample_tensor):
    (data, T) = sample_tensor

    # unsupported init type
    with pytest.raises(AssertionError) as excinfo:
        (M, Minit, output) = ttb.cp_als(T, 2, init="init")
    assert "The selected initialization method is not supported" in str(excinfo)

    # incorrect size of intial ktensor
    Tshape_incorrect = list(T.shape)
    Tshape_incorrect[0] = Tshape_incorrect[0] + 1
    Tshape_incorrect = tuple(Tshape_incorrect)
    KInit = ttb.ktensor.from_function(np.random.random_sample, Tshape_incorrect, 2)
    with pytest.raises(AssertionError) as excinfo:
        (M, Minit, output) = ttb.cp_als(T, 2, init=KInit)
    assert "Mode 0 of the initial guess is the wrong size" in str(excinfo)


@pytest.mark.indevelopment
def test_cp_als_sptensor_default_init(capsys, sample_sptensor):
    (data, T) = sample_sptensor
    (M, Minit, output) = ttb.cp_als(T, 2)
    capsys.readouterr()
    assert pytest.approx(output["fit"], 1) == 0


@pytest.mark.indevelopment
def test_cp_als_sptensor_nvecs_init(capsys, sample_sptensor):
    (data, T) = sample_sptensor
    with pytest.warns(Warning) as record:
        (M, Minit, output) = ttb.cp_als(T, 1, init="nvecs")
    assert (
        "Greater than or equal to sptensor.shape[n] - 1 eigenvectors requires cast to dense to solve"
        in str(record[0].message)
    )
    capsys.readouterr()
    assert pytest.approx(output["fit"], 1) == 0


@pytest.mark.indevelopment
def test_cp_als_sptensor_ktensor_init(capsys, sample_sptensor):
    (data, T) = sample_sptensor
    KInit = ttb.ktensor.from_function(np.random.random_sample, T.shape, 2)
    (M, Minit, output) = ttb.cp_als(T, 2, init=KInit)
    capsys.readouterr()
    assert pytest.approx(output["fit"], 1) == 0


@pytest.mark.indevelopment
def test_cp_als_tensor_dimorder(capsys, sample_tensor):
    (data, T) = sample_tensor

    # default dimorder
    dimorder = [i for i in range(T.ndims)]
    print(dimorder)
    print(dimorder.__class__)
    (M, Minit, output) = ttb.cp_als(T, 2, dimorder=dimorder)
    capsys.readouterr()
    assert pytest.approx(output["fit"], 1) == 0

    # reverse should work
    dimorder = [T.ndims - i - 1 for i in range(T.ndims)]
    print(dimorder)
    print(dimorder.__class__)
    (M, Minit, output) = ttb.cp_als(T, 2, dimorder=dimorder)
    capsys.readouterr()
    assert pytest.approx(output["fit"], 1) == 0

    # dimorder not a list
    with pytest.raises(AssertionError) as excinfo:
        (M, Minit, output) = ttb.cp_als(T, 2, dimorder=2)
    assert "Dimorder must be a list" in str(excinfo)

    # dimorder not a permutation of [range(tensor.ndims)]
    dimorder = [i for i in range(T.ndims)]
    dimorder[-1] = dimorder[-1] + 1
    with pytest.raises(AssertionError) as excinfo:
        (M, Minit, output) = ttb.cp_als(T, 2, dimorder=dimorder)
    assert "Dimorder must be a list or permutation of range(tensor.ndims)" in str(
        excinfo
    )


@pytest.mark.indevelopment
def test_cp_als_tensor_zeros(capsys, sample_tensor):
    # 2-way tensor
    T2 = ttb.tensor.from_function(np.zeros, (2, 2))
    (M2, Minit2, output2) = ttb.cp_als(T2, 2)
    capsys.readouterr()
    assert pytest.approx(output2["fit"], 1) == 0
    assert output2["normresidual"] == 0

    # 3-way tensor
    T3 = ttb.tensor.from_function(np.zeros, (3, 4, 5))
    (M3, Minit3, output3) = ttb.cp_als(T3, 2)
    capsys.readouterr()
    assert pytest.approx(output3["fit"], 1) == 0
    assert output3["normresidual"] == 0


@pytest.mark.indevelopment
def test_cp_als_sptensor_zeros(capsys):
    # 2-way tensor
    shape2 = (2, 2)
    T2 = ttb.sptensor.from_function(np.zeros, shape2, np.ceil(np.prod(shape2) / 2.0))
    print(T2)
    (M2, Minit2, output2) = ttb.cp_als(T2, 2)
    capsys.readouterr()
    assert pytest.approx(output2["fit"], 1) == 0
    assert output2["normresidual"] == 0

    # 3-way tensor
    shape3 = (2, 2)
    T3 = ttb.sptensor.from_function(np.zeros, shape3, np.ceil(np.prod(shape3) / 2.0))
    (M3, Minit3, output3) = ttb.cp_als(T3, 2)
    capsys.readouterr()
    assert pytest.approx(output3["fit"], 1) == 0
    assert output3["normresidual"] == 0
