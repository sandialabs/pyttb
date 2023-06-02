# Copyright 2022 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

import os

import numpy as np
import pytest

import pyttb as ttb


@pytest.fixture()
def sample_tensor():
    # truth data
    T = ttb.tensor.from_data(np.ones((3, 3, 3)), (3, 3, 3))
    return T


@pytest.fixture()
def sample_sptensor():
    # truth data
    subs = np.array(
        [
            [0, 0, 0],
            [0, 2, 2],
            [1, 1, 1],
            [1, 2, 0],
            [1, 2, 1],
            [1, 2, 2],
            [1, 3, 1],
            [2, 0, 0],
            [2, 0, 1],
            [2, 2, 0],
            [2, 2, 1],
            [2, 3, 0],
            [2, 3, 2],
            [3, 0, 0],
            [3, 0, 1],
            [3, 2, 0],
            [4, 0, 2],
            [4, 3, 2],
        ]
    )
    vals = np.reshape(np.array(range(1, 19)), (18, 1))
    shape = (5, 4, 3)
    S = ttb.sptensor().from_data(subs, vals, shape)
    return S


@pytest.fixture()
def sample_ktensor():
    # truth data
    weights = np.array([3, 2])
    fm0 = np.array([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0], [4.0, 8.0]])
    fm1 = np.array([[2.0, 7.0], [3.0, 8.0], [4.0, 9.0], [5.0, 10.0], [6.0, 11.0]])
    fm2 = np.array([[3.0, 6.0], [4.0, 7.0], [5.0, 8.0]])
    factor_matrices = [fm0, fm1, fm2]
    K = ttb.ktensor.from_data(weights, factor_matrices)
    return K


@pytest.fixture()
def sample_array():
    # truth data
    M = np.array([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0], [4.0, 8.0]])
    return M


@pytest.mark.indevelopment
def test_import_data_tensor(sample_tensor):
    # truth data
    T = sample_tensor

    # imported data
    data_filename = os.path.join(os.path.dirname(__file__), "data", "tensor.tns")
    X = ttb.import_data(data_filename)

    assert T.isequal(X)


@pytest.mark.indevelopment
def test_import_data_sptensor(sample_sptensor):
    # truth data
    S = sample_sptensor

    # imported data
    data_filename = os.path.join(os.path.dirname(__file__), "data", "sptensor.tns")
    X = ttb.import_data(data_filename)

    assert S.isequal(X)


@pytest.mark.indevelopment
def test_import_data_ktensor(sample_ktensor):
    # truth data
    K = sample_ktensor

    # imported data
    data_filename = os.path.join(os.path.dirname(__file__), "data", "ktensor.tns")
    X = ttb.import_data(data_filename)

    assert K.isequal(X)


@pytest.mark.indevelopment
def test_import_data_array(sample_array):
    # truth data
    M = sample_array

    # imported data
    data_filename = os.path.join(os.path.dirname(__file__), "data", "matrix.tns")
    X = ttb.import_data(data_filename)

    assert (M == X).all()


@pytest.mark.indevelopment
def test_export_data_tensor(sample_tensor):
    # truth data
    T = sample_tensor

    data_filename = os.path.join(os.path.dirname(__file__), "data", "tensor.out")
    ttb.export_data(T, data_filename)

    X = ttb.import_data(data_filename)
    assert T.isequal(X)
    os.unlink(data_filename)

    data_filename = os.path.join(os.path.dirname(__file__), "data", "tensor_int.out")
    ttb.export_data(T, data_filename, fmt_data="%d")

    X = ttb.import_data(data_filename)
    assert T.isequal(X)
    os.unlink(data_filename)


@pytest.mark.indevelopment
def test_export_data_sptensor(sample_sptensor):
    # truth data
    S = sample_sptensor

    # imported data
    data_filename = os.path.join(os.path.dirname(__file__), "data", "sptensor.out")
    ttb.export_data(S, data_filename)

    X = ttb.import_data(data_filename)
    assert S.isequal(X)
    os.unlink(data_filename)

    data_filename = os.path.join(os.path.dirname(__file__), "data", "sptensor_int.out")
    ttb.export_data(S, data_filename, fmt_data="%d")

    X = ttb.import_data(data_filename)
    assert S.isequal(X)
    os.unlink(data_filename)


@pytest.mark.indevelopment
def test_export_data_ktensor(sample_ktensor):
    # truth data
    K = sample_ktensor

    # imported data
    data_filename = os.path.join(os.path.dirname(__file__), "data", "ktensor.out")
    ttb.export_data(K, data_filename)

    X = ttb.import_data(data_filename)
    assert K.isequal(X)
    os.unlink(data_filename)

    data_filename = os.path.join(os.path.dirname(__file__), "data", "ktensor_int.out")
    ttb.export_data(K, data_filename, fmt_data="%d", fmt_weights="%d")

    X = ttb.import_data(data_filename)
    assert K.isequal(X)
    os.unlink(data_filename)


@pytest.mark.indevelopment
def test_export_data_array(sample_array):
    # truth data
    M = sample_array

    # imported data
    data_filename = os.path.join(os.path.dirname(__file__), "data", "matrix.out")
    ttb.export_data(M, data_filename)

    X = ttb.import_data(data_filename)
    assert (M == X).all()
    os.unlink(data_filename)

    data_filename = os.path.join(os.path.dirname(__file__), "data", "matrix_int.out")
    ttb.export_data(M, data_filename, fmt_data="%d")

    X = ttb.import_data(data_filename)
    assert (M == X).all()
    os.unlink(data_filename)
