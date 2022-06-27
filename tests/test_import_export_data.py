# Copyright 2022 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

import numpy as np
import pytest
import os
import TensorToolbox as ttb

@pytest.fixture()
def sample_tensor_2way():
    data = np.array([[1., 2., 3.], [4., 5., 6.]])
    shape = (2, 3)
    params = {'data':data, 'shape': shape}
    tensorInstance = ttb.tensor().from_data(data, shape)
    return params, tensorInstance

@pytest.fixture()
def sample_tensor_3way():
    data = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.])
    shape = (2, 3, 2)
    params = {'data':np.reshape(data, np.array(shape), order='F'), 'shape': shape}
    tensorInstance = ttb.tensor().from_data(data, shape)
    return params, tensorInstance

@pytest.fixture()
def sample_tensor_4way():
    data = np.arange(1, 82)
    shape = (3, 3, 3, 3)
    params = {'data':np.reshape(data, np.array(shape), order='F'), 'shape': shape}
    tensorInstance = ttb.tensor().from_data(data, shape)
    return params, tensorInstance

@pytest.mark.indevelopment
def test_import_data_tensor():
    # truth data
    T = ttb.tensor.from_data(np.ones((3,3,3)), (3,3,3))

    # imported data
    data_filename = os.path.join(os.path.dirname(__file__),'data','tensor.tns')
    X = ttb.import_data(data_filename)

    assert X.shape == (3, 3, 3)
    assert T.isequal(X)
 
@pytest.mark.indevelopment
def test_import_data_sptensor():
    # truth data
    subs = np.array([[0, 0, 0],[0, 2, 2],[1, 1, 1],[1, 2, 0],[1, 2, 1],[1, 2, 2],
                     [1, 3, 1],[2, 0, 0],[2, 0, 1],[2, 2, 0],[2, 2, 1],[2, 3, 0],
                     [2, 3, 2],[3, 0, 0],[3, 0, 1],[3, 2, 0],[4, 0, 2],[4, 3, 2]])
    vals = np.reshape(np.array(range(1,19)),(18,1))
    shape = (5, 4, 3)
    S = ttb.sptensor().from_data(subs, vals, shape)

    # imported data
    data_filename = os.path.join(os.path.dirname(__file__),'data','sptensor.tns')
    X = ttb.import_data(data_filename)
    
    assert S.isequal(X)

@pytest.mark.indevelopment
def test_import_data_ktensor():
    # truth data
    weights = np.array([3, 2])
    fm0 = np.array([[1., 5.], [2., 6.], [3., 7.], [4., 8.]])
    fm1 = np.array([[ 2.,  7.], [ 3.,  8.], [ 4.,  9.], [ 5., 10.], [ 6., 11.]])
    fm2 = np.array([[3., 6.], [4., 7.], [5., 8.]])
    factor_matrices = [fm0, fm1, fm2]
    K = ttb.ktensor.from_data(weights, factor_matrices)
    
    # imported data
    data_filename = os.path.join(os.path.dirname(__file__),'data','ktensor.tns')
    X = ttb.import_data(data_filename)
    
    assert K.isequal(X)

@pytest.mark.indevelopment
def test_import_data_array():
    # truth data
    M = np.array([[1., 5.], [2., 6.], [3., 7.], [4., 8.]])
    print('\nM')
    print(M)

    # imported data
    data_filename = os.path.join(os.path.dirname(__file__),'data','matrix.tns')
    X = ttb.import_data(data_filename)
    print('\nX')
    print(X)
    
    assert (M == X).all()

@pytest.mark.indevelopment
def test_export_data_tensor():
    pass

@pytest.mark.indevelopment
def test_export_data_sptensor():
    pass

@pytest.mark.indevelopment
def test_export_data_ktensor():
    pass

@pytest.mark.indevelopment
def test_export_data_array():
    pass

