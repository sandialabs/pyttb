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
    _ = ttb.hosvd(T, 1)


@pytest.mark.indevelopment
def test_tucker_als_tensor_incorrect_ranks(capsys, sample_tensor):
    (data, T) = sample_tensor
    ranks = list(range(T.ndims - 1))
    with pytest.raises(ValueError):
        _ = ttb.hosvd(T, 1, ranks=ranks)


@pytest.mark.indevelopment
def test_tucker_als_tensor_incorrect_dimorder(capsys, sample_tensor):
    (data, T) = sample_tensor
    dimorder = list(range(T.ndims - 1))
    with pytest.raises(ValueError):
        _ = ttb.hosvd(T, 1, dimorder=dimorder)

    dimorder = 1
    with pytest.raises(ValueError):
        _ = ttb.hosvd(T, 1, dimorder=dimorder)
