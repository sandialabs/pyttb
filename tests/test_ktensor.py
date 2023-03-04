# Copyright 2022 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

import numpy as np
import pytest

import pyttb as ttb

np.set_printoptions(precision=16)


@pytest.fixture()
def sample_ktensor_2way():
    weights = np.array([1.0, 2.0])
    fm0 = np.array([[1.0, 2.0], [3.0, 4.0]])
    fm1 = np.array([[5.0, 6.0], [7.0, 8.0]])
    factor_matrices = [fm0, fm1]
    data = {"weights": weights, "factor_matrices": factor_matrices}
    ktensorInstance = ttb.ktensor.from_data(weights, factor_matrices)
    return data, ktensorInstance


@pytest.fixture()
def sample_ktensor_3way():
    rank = 2
    shape = np.array([2, 3, 4])
    vector = np.arange(1, rank * sum(shape) + 1).astype(float)
    weights = 2 * np.ones(rank).astype(float)
    vector_with_weights = np.concatenate((weights, vector), axis=0)
    # vector_with_weights = vector_with_weights.reshape((len(vector_with_weights), 1))
    # ground truth
    fm0 = np.array([[1.0, 3.0], [2.0, 4.0]])
    fm1 = np.array([[5.0, 8.0], [6.0, 9.0], [7.0, 10.0]])
    fm2 = np.array([[11.0, 15.0], [12.0, 16.0], [13.0, 17.0], [14.0, 18.0]])
    factor_matrices = [fm0, fm1, fm2]
    data = {
        "weights": weights,
        "factor_matrices": factor_matrices,
        "vector": vector,
        "vector_with_weights": vector_with_weights,
        "shape": shape,
    }
    ktensorInstance = ttb.ktensor.from_data(weights, factor_matrices)
    return data, ktensorInstance


@pytest.fixture()
def sample_ktensor_symmetric():
    weights = np.array([1.0, 1.0])
    fm0 = np.array(
        [[2.340431417384394, 4.951967353890655], [4.596069112758807, 8.012451489774961]]
    )
    fm1 = np.array(
        [[2.340431417384394, 4.951967353890655], [4.596069112758807, 8.012451489774961]]
    )
    factor_matrices = [fm0, fm1]
    data = {"weights": weights, "factor_matrices": factor_matrices}
    ktensorInstance = ttb.ktensor.from_data(weights, factor_matrices)
    return data, ktensorInstance


@pytest.mark.indevelopment
def test_ktensor_init():
    empty = np.array([])

    # No args
    K0 = ttb.ktensor()
    assert (K0.weights == empty).all()
    assert K0.factor_matrices == []


@pytest.mark.indevelopment
def test_ktensor_from_tensor_type(sample_ktensor_2way):
    (data, K0) = sample_ktensor_2way
    K1 = ttb.ktensor.from_tensor_type(K0)
    assert (K0.weights == K1.weights).all()
    assert (K0.factor_matrices[0] == K1.factor_matrices[0]).all()
    assert (K0.factor_matrices[1] == K1.factor_matrices[1]).all()
    # won't work with instances other than ktensors
    with pytest.raises(AssertionError) as excinfo:
        K2 = ttb.ktensor.from_tensor_type(np.ones((2, 2)))
    assert "Cannot convert from <class 'numpy.ndarray'> to ktensor" in str(excinfo)


@pytest.mark.indevelopment
def test_ktensor_from_factor_matrices(sample_ktensor_2way):
    (data, K0) = sample_ktensor_2way
    K0 = ttb.ktensor.from_factor_matrices(data["factor_matrices"])
    assert (K0.weights == np.ones(2)).all()
    assert (K0.factor_matrices[0] == data["factor_matrices"][0]).all()
    assert (K0.factor_matrices[1] == data["factor_matrices"][1]).all()

    # Create ktensor with weights and multiple factor matrices as arguments
    K1 = ttb.ktensor.from_factor_matrices(
        data["factor_matrices"][0], data["factor_matrices"][1]
    )
    assert (K1.weights == np.ones(2)).all()
    assert (K1.factor_matrices[0] == data["factor_matrices"][0]).all()
    assert (K1.factor_matrices[1] == data["factor_matrices"][1]).all()


@pytest.mark.indevelopment
def test_ktensor_from_data(sample_ktensor_2way, capsys):
    (data, K0) = sample_ktensor_2way
    K0 = ttb.ktensor.from_data(data["weights"], data["factor_matrices"])
    assert (K0.weights == data["weights"]).all()
    assert (K0.factor_matrices[0] == data["factor_matrices"][0]).all()
    assert (K0.factor_matrices[1] == data["factor_matrices"][1]).all()

    # Create ktensor with weights and multiple factor matrices as arguments
    K1 = ttb.ktensor.from_data(
        data["weights"], data["factor_matrices"][0], data["factor_matrices"][1]
    )
    assert (K1.weights == data["weights"]).all()
    assert (K1.factor_matrices[0] == data["factor_matrices"][0]).all()
    assert (K1.factor_matrices[1] == data["factor_matrices"][1]).all()

    # Weights that are int should be converted
    weights_int = np.array([1, 2])
    K2 = ttb.ktensor.from_data(weights_int, data["factor_matrices"])
    out, err = capsys.readouterr()
    assert (
        "converting weights from int64 to float" in out
        or "converting weights from int32 to float" in out
    )

    # Weights that are int should be converted
    fm0 = np.array([[1, 2], [3, 4]])
    fm1 = np.array([[5, 6], [7, 8]])
    factor_matrices = [fm0, fm1]
    K3 = ttb.ktensor.from_data(data["weights"], factor_matrices)
    out, err = capsys.readouterr()
    assert (
        "converting factor_matrices[0] from int64 to float" in out
        or "converting factor_matrices[0] from int32 to float" in out
    )


@pytest.mark.indevelopment
def test_ktensor_from_function():
    K0 = ttb.ktensor.from_function(np.ones, (2, 3, 4), 2)
    assert (K0.weights == np.array([1.0, 1.0])).all()
    assert (K0.factor_matrices[0] == np.ones((2, 2))).all()
    assert (K0.factor_matrices[1] == np.ones((3, 2))).all()
    assert (K0.factor_matrices[2] == np.ones((4, 2))).all()

    np.random.seed(1)
    K1 = ttb.ktensor.from_function(np.random.random_sample, (2, 3, 4), 2)
    assert (K1.weights == np.array([1.0, 1.0])).all()
    fm0 = np.array([[4.17022005e-01, 7.20324493e-01], [1.14374817e-04, 3.02332573e-01]])
    fm1 = np.array(
        [[0.14675589, 0.09233859], [0.18626021, 0.34556073], [0.39676747, 0.53881673]]
    )
    fm2 = np.array(
        [
            [0.41919451, 0.6852195],
            [0.20445225, 0.87811744],
            [0.02738759, 0.67046751],
            [0.4173048, 0.55868983],
        ]
    )
    assert np.linalg.norm(K1.factor_matrices[0] - fm0) < 1e-8
    assert np.linalg.norm(K1.factor_matrices[1] - fm1) < 1e-8
    assert np.linalg.norm(K1.factor_matrices[2] - fm2) < 1e-8


@pytest.mark.indevelopment
def test_ktensor_from_vector(sample_ktensor_3way):
    (data, K0) = sample_ktensor_3way

    # without explicit weights in x
    K0 = ttb.ktensor.from_vector(data["vector"], data["shape"], False)
    assert (K0.weights == np.ones((3, 1))).all()
    assert (K0.factor_matrices[0] == data["factor_matrices"][0]).all()
    assert (K0.factor_matrices[1] == data["factor_matrices"][1]).all()
    assert (K0.factor_matrices[2] == data["factor_matrices"][2]).all()

    # with explicit weights in x
    K1 = ttb.ktensor.from_vector(data["vector_with_weights"], data["shape"], True)
    assert (K1.weights == data["weights"]).all()
    assert (K1.factor_matrices[0] == data["factor_matrices"][0]).all()
    assert (K1.factor_matrices[1] == data["factor_matrices"][1]).all()
    assert (K1.factor_matrices[2] == data["factor_matrices"][2]).all()

    # data as a row vector will work, but will be transposed
    transposed_data = data["vector"].copy().reshape((1, len(data["vector"])))
    K2 = ttb.ktensor.from_vector(transposed_data, data["shape"], False)
    assert (K2.weights == np.ones((3, 1))).all()
    assert (K2.factor_matrices[0] == data["factor_matrices"][0]).all()
    assert (K2.factor_matrices[1] == data["factor_matrices"][1]).all()
    assert (K2.factor_matrices[2] == data["factor_matrices"][2]).all()

    # error if the shape does not match the the number of data elements
    with pytest.raises(AssertionError) as excinfo:
        K3 = ttb.ktensor.from_vector(data["vector"].T, data["shape"] + 7, False)
    assert "Input parameter 'data' is not the right length." in str(excinfo)


@pytest.mark.indevelopment
def test_ktensor_arrange(sample_ktensor_2way):
    (data, K) = sample_ktensor_2way

    # permutation only
    K0 = ttb.ktensor.from_tensor_type(K)
    p = [1, 0]
    K0.arrange(permutation=p)
    assert (K0.weights == data["weights"][p]).all()
    assert (K0.factor_matrices[0] == data["factor_matrices"][0][:, p]).all()
    assert (K0.factor_matrices[1] == data["factor_matrices"][1][:, p]).all()

    # normalize and arrange by sorting weights (default)
    K1 = ttb.ktensor.from_tensor_type(K)
    K1.arrange()
    weights = np.array([89.4427191, 27.20294102])
    fm0 = np.array([[0.4472136, 0.31622777], [0.89442719, 0.9486833]])
    fm1 = np.array([[0.6, 0.58123819], [0.8, 0.81373347]])
    assert np.linalg.norm(K1.weights - weights) < 1e-8
    assert np.linalg.norm(K1.factor_matrices[0] - fm0) < 1e-8
    assert np.linalg.norm(K1.factor_matrices[1] - fm1) < 1e-8

    # error, cannot shoft weight and permute simultaneously
    with pytest.raises(AssertionError) as excinfo:
        K1.arrange(weight_factor=0, permutation=p)
    assert (
        "Weighting and permuting the ktensor at the same time is not allowed."
        in str(excinfo)
    )

    # error, length of permutation must equal number of components in ktensor
    with pytest.raises(AssertionError) as excinfo:
        K1.arrange(permutation=[0, 1, 2])
    assert (
        "Number of elements in permutation does not match number of components in ktensor."
        in str(excinfo)
    )


@pytest.mark.indevelopment
def test_ktensor_copy(sample_ktensor_2way):
    (data, K0) = sample_ktensor_2way
    K1 = K0.copy()
    assert (K0.weights == K1.weights).all()
    assert (K0.factor_matrices[0] == K1.factor_matrices[0]).all()
    assert (K0.factor_matrices[1] == K1.factor_matrices[1]).all()

    # make sure it is a deep copy
    K1.weights[0] = 0
    assert not (K0.weights[0] == K1.weights[0])


@pytest.mark.indevelopment
def test_ktensor_double(sample_ktensor_2way, sample_ktensor_3way):
    (data2, K2) = sample_ktensor_2way
    assert (K2.double() == np.array([[29.0, 39.0], [63.0, 85.0]])).all()
    (data3, K3) = sample_ktensor_3way
    A = np.array(
        [
            830.0,
            888.0,
            946.0,
            1004.0,
            942.0,
            1008.0,
            1074.0,
            1140.0,
            1054.0,
            1128.0,
            1202.0,
            1276.0,
            1180.0,
            1264.0,
            1348.0,
            1432.0,
            1344.0,
            1440.0,
            1536.0,
            1632.0,
            1508.0,
            1616.0,
            1724.0,
            1832.0,
        ]
    ).reshape((2, 3, 4))
    assert (K3.double() == A).all()


@pytest.mark.indevelopment
def test_ktensor_end(sample_ktensor_3way):
    (data, K) = sample_ktensor_3way
    assert K.end() == 23
    assert K.end(k=0) == 1
    assert K.end(k=1) == 2
    assert K.end(k=2) == 3


@pytest.mark.indevelopment
def test_ktensor_extract(sample_ktensor_3way):
    (data, K) = sample_ktensor_3way
    weights = data["weights"][[1]]
    fm0 = data["factor_matrices"][0][:, [1]]
    fm1 = data["factor_matrices"][1][:, [1]]
    fm2 = data["factor_matrices"][2][:, [1]]
    factor_matrices = [fm0, fm1, fm2]
    K_new = ttb.ktensor.from_data(weights, factor_matrices)

    # int
    K_extracted = K.extract(1)
    assert K_new.isequal(K_extracted)
    # tuple
    K_extracted = K.extract((1))
    assert K_new.isequal(K_extracted)
    # list
    K_extracted = K.extract([1])
    assert K_new.isequal(K_extracted)
    # np.ndarray
    K_extracted = K.extract(np.array([1]))
    assert K_new.isequal(K_extracted)

    # should return copy of ktensor
    K_extracted = K.extract()
    assert K.isequal(K_extracted)

    # wrong component index type
    with pytest.raises(AssertionError) as excinfo:
        K.extract(1.0)
    assert "Input parameter must be an int, tuple, list or numpy.ndarray" in str(
        excinfo
    )

    # too many components
    with pytest.raises(AssertionError) as excinfo:
        K.extract([0, 1, 2, 3])
    assert (
        "Number of components requested is not valid: 4 (should be in [1,...,2])."
        in str(excinfo)
    )

    # component index out of range
    with pytest.raises(AssertionError) as excinfo:
        K.extract((5))
    assert "Invalid component indices to be extracted: [5] not in range(2)" in str(
        excinfo
    )


@pytest.mark.indevelopment
def test_ktensor_fixsigns(sample_ktensor_2way):
    (data, K) = sample_ktensor_2way
    K1 = ttb.ktensor.from_tensor_type(K)
    K2 = ttb.ktensor.from_tensor_type(K)

    # use same ktensor
    K1.factor_matrices[0][1, 1] = -K1.factor_matrices[0][1, 1]
    K1.factor_matrices[1][1, 1] = -K1.factor_matrices[1][1, 1]
    K2.factor_matrices[0][0, 1] = -K2.factor_matrices[0][0, 1]
    K2.factor_matrices[1][0, 1] = -K2.factor_matrices[1][0, 1]
    assert K1.fixsigns().isequal(K2)

    # use different ktensor for fixing the signs
    K3 = K.copy()
    K3.factor_matrices[0][1, 1] = -K3.factor_matrices[0][1, 1]
    K3.factor_matrices[1][1, 1] = -K3.factor_matrices[1][1, 1]
    K = K.fixsigns(K3)
    weights1 = np.array([27.202941017470888, 89.44271909999159])
    factor_matrix10 = np.array(
        [
            [0.3162277660168379, -0.4472135954999579],
            [0.9486832980505138, -0.8944271909999159],
        ]
    )
    factor_matrix11 = np.array([[0.5812381937190965, -0.6], [0.813733471206735, -0.8]])
    assert np.linalg.norm(K.weights - weights1) < 1e-8
    assert np.linalg.norm(K.factor_matrices[0] - factor_matrix10) < 1e-8
    assert np.linalg.norm(K.factor_matrices[1] - factor_matrix11) < 1e-8


@pytest.mark.indevelopment
def test_ktensor_full(sample_ktensor_2way, sample_ktensor_3way):
    (data, K2) = sample_ktensor_2way
    assert K2.full().isequal(
        ttb.tensor.from_data(np.array([[29.0, 39.0], [63.0, 85.0]]), (2, 2))
    )
    (data, K3) = sample_ktensor_3way
    print(K3.full())


@pytest.mark.indevelopment
def test_ktensor_innerprod(sample_ktensor_2way):
    (data, K) = sample_ktensor_2way
    assert K.innerprod(K) == 13556

    # test with tensor
    Tdata = np.array([[1, 2], [3, 4]])
    Tshape = (2, 2)
    T = ttb.tensor().from_data(Tdata, Tshape)
    assert K.innerprod(T) == 636

    # test with sptensor
    Ssubs = np.array([[0, 0], [0, 1], [1, 1]])
    Svals = np.array([[0.5], [1.0], [1.5]])
    Sshape = (2, 2)
    S = ttb.sptensor().from_data(Ssubs, Svals, Sshape)
    assert K.innerprod(S) == 181

    # Wrong shape
    K1 = ttb.ktensor.from_function(np.ones, (2, 3), 2)
    with pytest.raises(AssertionError) as excinfo:
        K.innerprod(K1)
    assert "Innerprod can only be computed for tensors of the same size" in str(excinfo)


@pytest.mark.indevelopment
def test_ktensor_isequal(sample_ktensor_2way):
    (data, K0) = sample_ktensor_2way
    # should be equal
    K1 = ttb.ktensor.from_tensor_type(K0)
    assert K0.isequal(K1)
    # ncomponents don't match
    K2 = ttb.ktensor.from_function(np.ones, (2, 2), 3)
    assert ~(K0.isequal(K2))
    # weights don't match
    K3 = ttb.ktensor.from_tensor_type(K0)
    K3.weights[0] = 10
    assert ~(K0.isequal(K3))
    # types don't match
    assert ~(K0.isequal(np.array([])))
    # factor_matrices don't match
    K4 = ttb.ktensor.from_tensor_type(K0)
    K4.factor_matrices[0] = np.zeros((2, 2))
    assert ~(K0.isequal(K4))


@pytest.mark.indevelopment
def test_ktensor_issymetric(sample_ktensor_2way, sample_ktensor_symmetric):
    # should not be symmetric
    (data, K) = sample_ktensor_2way
    assert ~(K.issymmetric())
    issym, diffs = K.issymmetric(return_diffs=True)
    assert (diffs == np.array([[0.0, 8.0], [0.0, 0]])).all()

    # should be symmetric
    (datas, K1) = sample_ktensor_symmetric
    assert K1.issymmetric()
    issym1, diffs1 = K1.issymmetric(return_diffs=True)
    assert (diffs1 == np.array([[0.0, 0.0], [0.0, 0]])).all()

    # Wrong shape
    K2 = ttb.ktensor.from_function(np.ones, (2, 3), 2)
    assert ~(K2.issymmetric())
    issym2, diffs2 = K2.issymmetric(return_diffs=True)
    assert (diffs2 == np.array([[0.0, np.inf], [0.0, 0]])).all()


@pytest.mark.indevelopment
def test_ktensor_mask(sample_ktensor_2way):
    (data, K) = sample_ktensor_2way
    W = ttb.tensor.from_data(np.array([[0, 1], [1, 0]]))
    assert (K.mask(W) == np.array([[63], [39]])).all()

    # Mask too large
    with pytest.raises(AssertionError) as excinfo:
        K.mask(ttb.tensor.from_function(np.ones, (2, 3, 4)))
    assert "Mask cannot be bigger than the data tensor" in str(excinfo)


@pytest.mark.indevelopment
def test_ktensor_mttkrp(sample_ktensor_3way):
    (data, K) = sample_ktensor_3way
    K1 = ttb.ktensor.from_function(np.ones, (2, 3, 4), 4)
    output0 = np.array(
        [[12492.0, 12492.0, 12492.0, 12492.0], [17856.0, 17856.0, 17856.0, 17856.0]]
    )
    assert (K.mttkrp(K1.factor_matrices, 0) == output0).all()
    output1 = np.array(
        [
            [8892.0, 8892.0, 8892.0, 8892],
            [10116.0, 10116.0, 10116.0, 10116],
            [11340.0, 11340.0, 11340.0, 11340],
        ]
    )
    assert (K.mttkrp(K1.factor_matrices, 1) == output1).all()
    output2 = np.array(
        [
            [6858.0, 6858.0, 6858.0, 6858],
            [7344.0, 7344.0, 7344.0, 7344],
            [7830.0, 7830.0, 7830.0, 7830],
            [8316.0, 8316.0, 8316.0, 8316],
        ]
    )
    assert (K.mttkrp(K1.factor_matrices, 2) == output2).all()

    # Wrong number of factor matrices
    fm_wrong_size = [
        K1.factor_matrices[0],
        K1.factor_matrices[0],
        K1.factor_matrices[0],
        np.ones((5, 4)),
    ]
    with pytest.raises(AssertionError) as excinfo:
        K.mttkrp(fm_wrong_size, 0)
    assert "List of factor matrices is the wrong length" in str(excinfo)
    # Wrong input type
    fm_wrong_type = (
        K1.factor_matrices[0],
        K1.factor_matrices[0],
        K1.factor_matrices[0],
    )
    with pytest.raises(AssertionError) as excinfo:
        K.mttkrp(fm_wrong_type, 0)
    assert "Second argument must be list of numpy.ndarray's" in str(excinfo)


@pytest.mark.indevelopment
def test_ktensor_ncomponents(sample_ktensor_2way):
    (data, K0) = sample_ktensor_2way
    assert K0.ncomponents == 2


@pytest.mark.indevelopment
def test_ktensor_ndims(sample_ktensor_2way, sample_ktensor_3way):
    (data, K0) = sample_ktensor_2way
    assert K0.ndims == 2
    data, K1 = sample_ktensor_3way
    assert K1.ndims == 3


@pytest.mark.indevelopment
def test_ktensor_norm():
    K0 = ttb.ktensor.from_function(np.zeros, (2, 3, 4), 2)
    assert pytest.approx(K0.norm(), 1e-8) == 0

    K1 = ttb.ktensor.from_function(np.ones, (2, 3, 4), 2)
    assert pytest.approx(K1.norm(), 1e-8) == 9.797958971132712

    rank = 2
    shape = np.array([2, 3, 4])
    data = np.arange(1, rank * sum(shape) + 1)
    weights = 2 * np.ones(rank)
    weights_and_data = np.concatenate((weights, data), axis=0)
    K2 = ttb.ktensor.from_vector(weights_and_data[:], shape, True)
    assert pytest.approx(K2.norm(), 1e-8) == 6.337788257744180e03


@pytest.mark.indevelopment
def test_ktensor_normalize(sample_ktensor_2way, sample_ktensor_3way):
    # get data and make several copies, so that all ktensor tests use the same data, as normalize computes in place
    data0, K0 = sample_ktensor_3way
    K1 = ttb.ktensor.from_tensor_type(K0)
    K2 = ttb.ktensor.from_tensor_type(K0)
    K3 = ttb.ktensor.from_tensor_type(K0)
    K4 = ttb.ktensor.from_tensor_type(K0)

    # normalize a single mode
    mode = 1
    # print("\nK0\n",K0)
    K0.normalize(mode=mode)
    # print("\nK0\n",K0)
    weights0 = np.array([20.97617696340303, 31.304951684997057])
    factor_matrix0 = np.array(
        [
            [0.4767312946227962, 0.5111012519999519],
            [0.5720775535473555, 0.5749889084999459],
            [0.6674238124719146, 0.6388765649999399],
        ]
    )
    assert np.linalg.norm(K0.weights - weights0) < 1e-8
    assert np.linalg.norm(K0.factor_matrices[mode] - factor_matrix0) < 1e-8

    # normalize using the defaults
    # print("\nK1\n",K1)
    K1.normalize()
    # print("\nK1\n",K1)
    weights1 = np.array([1177.285012220915, 5177.161384388167])
    factor_matrix10 = np.array([[0.4472135954999579, 0.6], [0.8944271909999159, 0.8]])
    factor_matrix11 = np.array(
        [
            [0.4767312946227962, 0.5111012519999519],
            [0.5720775535473555, 0.5749889084999459],
            [0.6674238124719146, 0.6388765649999399],
        ]
    )
    factor_matrix12 = np.array(
        [
            [0.4382504900892777, 0.4535055413676754],
            [0.4780914437337575, 0.4837392441255204],
            [0.5179323973782373, 0.5139729468833655],
            [0.5577733510227171, 0.5442066496412105],
        ]
    )
    assert np.linalg.norm(K1.weights - weights1) < 1e-8
    assert np.linalg.norm(K1.factor_matrices[0] - factor_matrix10) < 1e-8
    assert np.linalg.norm(K1.factor_matrices[1] - factor_matrix11) < 1e-8
    assert np.linalg.norm(K1.factor_matrices[2] - factor_matrix12) < 1e-8

    # normalize using vector 1-norm
    normtype = 1
    # print("\nK2\n",K2)
    K2.normalize(normtype=normtype)
    # print("\nK2\n",K2)
    weights2 = np.array([5400.0, 24948.0])
    factor_matrix20 = np.array(
        [
            [0.3333333333333333, 0.4285714285714285],
            [0.6666666666666666, 0.5714285714285714],
        ]
    )
    factor_matrix21 = np.array(
        [
            [0.2777777777777778, 0.2962962962962963],
            [0.3333333333333333, 0.3333333333333333],
            [0.3888888888888888, 0.3703703703703703],
        ]
    )
    factor_matrix22 = np.array(
        [
            [0.22, 0.2272727272727273],
            [0.24, 0.2424242424242424],
            [0.26, 0.2575757575757576],
            [0.28, 0.2727272727272727],
        ]
    )
    assert np.linalg.norm(K2.weights - weights2) < 1e-8
    assert np.linalg.norm(K2.factor_matrices[0] - factor_matrix20) < 1e-8
    assert np.linalg.norm(K2.factor_matrices[1] - factor_matrix21) < 1e-8
    assert np.linalg.norm(K2.factor_matrices[2] - factor_matrix22) < 1e-8

    # normalize and shift all weight to factor 1
    weight_factor = 1
    # print("\nK3\n",K3)
    K3.normalize(weight_factor=weight_factor)
    # print("\nK3\n",K3)
    weights3 = np.array([1.0, 1.0])
    factor_matrix30 = np.array(
        [[0.4472135954999579, 0.6000000000000001], [0.8944271909999159, 0.8]]
    )
    factor_matrix31 = np.array(
        [
            [561.2486080160912, 2646.0536653665963],
            [673.4983296193095, 2976.8103735374207],
            [785.7480512225277, 3307.567081708246],
        ]
    )
    factor_matrix32 = np.array(
        [
            [0.4382504900892776, 0.4535055413676753],
            [0.4780914437337574, 0.4837392441255204],
            [0.5179323973782373, 0.5139729468833654],
            [0.557773351022717, 0.5442066496412105],
        ]
    )
    assert np.linalg.norm(K3.weights - weights3) < 1e-8
    assert np.linalg.norm(K3.factor_matrices[0] - factor_matrix30) < 1e-8
    assert np.linalg.norm(K3.factor_matrices[1] - factor_matrix31) < 1e-8
    assert np.linalg.norm(K3.factor_matrices[2] - factor_matrix32) < 1e-8

    # error if the mode is not in the range of number of dimensions
    with pytest.raises(AssertionError) as excinfo:
        K0.normalize(mode=4)
    assert (
        "Parameter single_factor is invalid; index must be an int in range of number of dimensions"
        in str(excinfo)
    )

    # normalize and sort
    K4.normalize(sort=True)
    weights = np.array([5177.161384388167, 1177.285012220915])
    fm0 = np.array(
        [[0.6000000000000001, 0.4472135954999579], [0.8, 0.8944271909999159]]
    )
    fm1 = np.array(
        [
            [0.5111012519999519, 0.4767312946227962],
            [0.5749889084999459, 0.5720775535473555],
            [0.6388765649999399, 0.6674238124719146],
        ]
    )
    fm2 = np.array(
        [
            [0.4535055413676753, 0.4382504900892776],
            [0.4837392441255204, 0.4780914437337574],
            [0.5139729468833654, 0.5179323973782373],
            [0.5442066496412105, 0.557773351022717],
        ]
    )
    assert np.linalg.norm(K4.weights - weights) < 1e-8
    assert np.linalg.norm(K4.factor_matrices[0] - fm0) < 1e-8
    assert np.linalg.norm(K4.factor_matrices[1] - fm1) < 1e-8
    assert np.linalg.norm(K4.factor_matrices[2] - fm2) < 1e-8

    # distribute weight across all factors
    (data5, K5) = sample_ktensor_2way
    K5.normalize(weight_factor="all")
    weights = np.array([1.0, 1.0])
    fm0 = np.array(
        [
            [1.6493314105258194, 4.229485053762256],
            [4.947994231577458, 8.458970107524513],
        ]
    )
    fm1 = np.array(
        [[3.031531424242968, 5.674449654019056], [4.244143993940155, 7.565932872025407]]
    )
    assert np.allclose(K5.weights, weights)
    assert np.allclose(K5.factor_matrices[0], fm0)
    assert np.allclose(K5.factor_matrices[1], fm1)


@pytest.mark.indevelopment
def test_ktensor_nvecs(sample_ktensor_3way):
    (data, K) = sample_ktensor_3way

    with pytest.warns(Warning) as record:
        assert np.allclose(
            K.nvecs(0, 1), np.array([[0.5731077440321353], [0.8194800264377384]])
        )
    assert (
        "Greater than or equal to ktensor.shape[n] - 1 eigenvectors requires cast to dense to solve"
        in str(record[0].message)
    )
    with pytest.warns(Warning) as record:
        assert np.allclose(
            K.nvecs(0, 2),
            np.array(
                [
                    [0.5731077440321353, 0.8194800264377384],
                    [0.8194800264377384, -0.5731077440321353],
                ]
            ),
        )
    assert (
        "Greater than or equal to ktensor.shape[n] - 1 eigenvectors requires cast to dense to solve"
        in str(record[0].message)
    )

    assert np.allclose(
        K.nvecs(1, 1),
        np.array([[0.5048631426517823], [0.5745404391632514], [0.6442177356747206]]),
    )
    with pytest.warns(Warning) as record:
        assert np.allclose(
            K.nvecs(1, 2),
            np.array(
                [
                    [0.5048631426517821, 0.7605567306550753],
                    [0.5745404391632517, 0.0568912743440822],
                    [0.6442177356747206, -0.6467741818894517],
                ]
            ),
        )
    assert (
        "Greater than or equal to ktensor.shape[n] - 1 eigenvectors requires cast to dense to solve"
        in str(record[0].message)
    )

    assert np.allclose(
        K.nvecs(2, 1),
        np.array(
            [
                [0.4507198734531968],
                [0.4827189140450413],
                [0.5147179546368857],
                [0.5467169952287301],
            ]
        ),
    )
    assert np.allclose(
        K.nvecs(2, 2),
        np.array(
            [
                [0.4507198734531969, 0.7048770074600103],
                [0.4827189140450412, 0.2588096791802433],
                [0.5147179546368857, -0.1872576491687805],
                [0.5467169952287302, -0.6333249775151949],
            ]
        ),
    )

    # Test for r >= N-1, requires cast to dense
    with pytest.warns(Warning) as record:
        K.nvecs(1, 3)
    assert (
        "Greater than or equal to ktensor.shape[n] - 1 eigenvectors requires cast to dense to solve"
        in str(record[0].message)
    )


@pytest.mark.indevelopment
def test_ktensor_permute(sample_ktensor_3way):
    (data, K) = sample_ktensor_3way
    order = np.array([2, 0, 1])
    fm = [data["factor_matrices"][i] for i in order]
    K0 = ttb.ktensor.from_data(data["weights"], fm)
    assert K0.isequal(K.permute(order))

    # invalid permutation
    order_invalid = np.array([3, 0, 1])
    with pytest.raises(AssertionError) as excinfo:
        K.permute(order_invalid)
    assert "Invalid permutation" in str(excinfo)


@pytest.mark.indevelopment
def test_ktensor_redistribute(sample_ktensor_2way):
    (data, K) = sample_ktensor_2way
    K.redistribute(0)
    assert (np.array([[1, 4], [3, 8]]) == K[0]).all()
    assert (np.array([[5, 6], [7, 8]]) == K[1]).all()
    assert (np.array([1, 1]) == K.weights).all()


pytest.mark.indevelopment


def test_ktensor_score():
    A = ttb.ktensor.from_data(
        np.array([2, 1, 3]), np.ones((3, 3)), np.ones((4, 3)), np.ones((5, 3))
    )
    B = ttb.ktensor.from_data(
        np.array([2, 4]), np.ones((3, 2)), np.ones((4, 2)), np.ones((5, 2))
    )

    # defaults
    score, Aperm, flag, best_perm = A.score(B)
    assert score == 0.875
    assert np.allclose(Aperm.weights, np.array([15.49193338, 23.23790008, 7.74596669]))
    assert flag == 1
    assert (best_perm == np.array([0, 2, 1])).all()

    # compare just factor matrices (i.e., do not use weights)
    score, Aperm, flag, best_perm = A.score(B, weight_penalty=False)
    assert score == 1.0
    assert np.allclose(Aperm.weights, np.array([15.49193338, 7.74596669, 23.23790008]))
    assert flag == 1
    assert (best_perm == np.array([0, 1, 2])).all()

    # compute score using exhaustive search
    with pytest.raises(AssertionError) as excinfo:
        score, Aperm, flag, best_perm = A.score(B, greedy=False)
    assert "Not yet implemented. Only greedy method is implemented currently." in str(
        excinfo
    )

    # try to compute score with tensor type other than ktensor
    with pytest.raises(AssertionError) as excinfo:
        score, Aperm, flag, best_perm = A.score(ttb.tensor.from_tensor_type(B))
    assert "The first input should be a ktensor" in str(excinfo)

    # try to compute score when ktensor dimensions do not match
    with pytest.raises(AssertionError) as excinfo:
        # A is 3x4x5; B is 3x4x4
        B = ttb.ktensor.from_data(
            np.array([2, 4]), np.ones((3, 2)), np.ones((4, 2)), np.ones((4, 2))
        )
        score, Aperm, flag, best_perm = A.score(B)
    assert "Size mismatch" in str(excinfo)


pytest.mark.indevelopment


def test_ktensor_shape(sample_ktensor_2way, sample_ktensor_3way):
    (data, K0) = sample_ktensor_2way
    assert K0.shape == (2, 2)
    (data, K1) = sample_ktensor_3way
    assert K1.shape == (2, 3, 4)
    assert K1.shape[0] == 2
    assert K1.shape[1] == 3
    assert K1.shape[2] == 4
    K2 = ttb.ktensor()
    assert K2.shape == ()

    # error expected when parameter is not in range of number of dimensions
    with pytest.raises(IndexError) as excinfo:
        K1.shape[3]
    assert "tuple index out of range" in str(excinfo)


@pytest.mark.indevelopment
def test_ktensor_symmetrize(sample_ktensor_2way):
    (data, K) = sample_ktensor_2way
    K1 = K.symmetrize()
    weights = np.array([1.0, 1.0])
    fm = np.array(
        [[2.340431417384394, 4.951967353890656], [4.596069112758807, 8.01245148977496]]
    )
    assert np.allclose(K1.weights, weights)
    assert np.allclose(K1.factor_matrices[0], fm)
    assert np.allclose(K1.factor_matrices[1], fm)
    assert K1.issymmetric()

    # odd-ordered ktensor with negative weight
    weights = np.array([1.0, 2.0, -3.0])
    fm0 = np.ones((4, 3))
    fm1 = np.ones((4, 3))
    fm2 = -np.ones((4, 3))
    K2 = ttb.ktensor.from_data(weights, [fm0, fm1, fm2])
    K3 = K2.symmetrize()
    out_fm = np.array(
        [
            [-1.0, -1.2599210498948732, 1.442249570307408],
            [-1.0, -1.2599210498948732, 1.442249570307408],
            [-1.0, -1.2599210498948732, 1.442249570307408],
            [-1.0, -1.2599210498948732, 1.442249570307408],
        ]
    )
    assert np.allclose(K1.weights, np.ones((3, 1)))
    assert np.allclose(K3.factor_matrices[0], out_fm)
    assert np.allclose(K3.factor_matrices[1], out_fm)
    assert np.allclose(K3.factor_matrices[2], out_fm)
    assert K3.issymmetric()


@pytest.mark.indevelopment
def test_ktensor_tolist(sample_ktensor_3way):
    (data, K) = sample_ktensor_3way

    # weights spread equally to all factor matrices
    fm0 = K.tolist()
    m0 = np.array(
        [
            [1.2599210498948732, 3.7797631496846193],
            [2.5198420997897464, 5.039684199579493],
        ]
    )
    m1 = np.array(
        [
            [6.299605249474366, 10.079368399158986],
            [7.559526299369239, 11.339289449053858],
            [8.819447349264113, 12.599210498948732],
        ]
    )
    m2 = np.array(
        [
            [13.859131548843605, 18.898815748423097],
            [15.119052598738477, 20.15873679831797],
            [16.37897364863335, 21.418657848212845],
            [17.638894698528226, 22.678578898107716],
        ]
    )
    assert np.allclose(fm0[0], m0)
    assert np.allclose(fm0[1], m1)
    assert np.allclose(fm0[2], m2)

    # weights are all 1
    K1 = ttb.ktensor.from_factor_matrices(fm0)
    fm0_tolist = K1.tolist()
    for i in range(len(fm0)):
        assert np.allclose(fm0[i], fm0_tolist[i])

    # weight spread to a single factor matrix
    fm1 = K.tolist(0)
    m0 = np.array(
        [
            [526.4978632435273, 3106.2968306329008],
            [1052.9957264870545, 4141.729107510534],
        ]
    )
    m1 = np.array(
        [
            [0.4767312946227962, 0.5111012519999519],
            [0.5720775535473555, 0.5749889084999459],
            [0.6674238124719146, 0.6388765649999399],
        ]
    )
    m2 = np.array(
        [
            [0.4382504900892776, 0.4535055413676753],
            [0.4780914437337574, 0.4837392441255204],
            [0.5179323973782373, 0.5139729468833654],
            [0.557773351022717, 0.5442066496412105],
        ]
    )
    assert np.allclose(fm1[0], m0)
    assert np.allclose(fm1[1], m1)
    assert np.allclose(fm1[2], m2)

    # mode not in range of self.ndims
    with pytest.raises(AssertionError) as excinfo:
        K.tolist(4)
    assert "Input parameter'mode' must be in the range of self.ndims" in str(excinfo)


@pytest.mark.indevelopment
def test_ktensor_tovec(sample_ktensor_3way):
    (data, K0) = sample_ktensor_3way
    assert (data["vector_with_weights"] == K0.tovec()).all()
    assert (data["vector"] == K0.tovec(include_weights=False)).all()


@pytest.mark.indevelopment
def test_ktensor_ttv(sample_ktensor_3way):
    (data, K) = sample_ktensor_3way
    K0 = K.ttv(np.array([1, 1, 1]), dims=1)
    weights = np.array([36.0, 54.0])
    fm0 = np.array([[1.0, 3.0], [2.0, 4.0]])
    fm1 = np.array([[11.0, 15.0], [12.0, 16.0], [13.0, 17.0], [14.0, 18.0]])
    factor_matrices = [fm0, fm1]
    K1 = ttb.ktensor.from_data(weights, factor_matrices)
    assert K0.isequal(K1)

    # Empty dims requires that # vectors == # dimensions
    vec2 = np.array([1, 1])
    vec3 = np.array([1, 1, 1])
    vec4 = np.array([1, 1, 1, 1])
    assert K.ttv([vec2, vec3, vec4]) == 30348

    # Wrong shape
    with pytest.raises(AssertionError) as excinfo:
        K.ttv([vec2, vec3, np.array([1, 2])])
    assert "Multiplicand is wrong size" in str(excinfo)

    # Multiple dimensions, but fewer than all dimensions, not in same order as ktensor dimensions
    K2 = K.ttv([vec4, vec3], dims=np.array([2, 1]))
    weights = np.array([1800.0, 3564.0])
    fm0 = np.array([[1.0, 3.0], [2.0, 4.0]])
    assert K2.isequal(ttb.ktensor.from_data(weights, fm0))


@pytest.mark.indevelopment
def test_ktensor_update(sample_ktensor_3way):
    (data, K) = sample_ktensor_3way

    # single factor matrix updates
    K1 = ttb.ktensor.from_tensor_type(K)
    vec0 = np.random.randn(K.shape[0] * K.ncomponents)
    vec1 = np.random.randn(K.shape[1] * K.ncomponents)
    vec2 = np.random.randn(K.shape[2] * K.ncomponents)
    K1.update(0, vec0)
    assert (K1.factor_matrices[0] == vec0.reshape((K1.shape[0], K1.ncomponents))).all()
    K1.update(1, vec1)
    assert (K1.factor_matrices[1] == vec1.reshape((K1.shape[1], K1.ncomponents))).all()
    K1.update(2, vec2)
    assert (K1.factor_matrices[2] == vec2.reshape((K1.shape[2], K1.ncomponents))).all()

    # all factor matrix updates
    K2 = ttb.ktensor.from_tensor_type(K)
    vec_all = np.concatenate((vec0, vec1, vec2))
    K2.update([0, 1, 2], vec_all)
    assert (K2.factor_matrices[0] == vec0.reshape((K2.shape[0], K2.ncomponents))).all()
    assert (K2.factor_matrices[1] == vec1.reshape((K2.shape[1], K2.ncomponents))).all()
    assert (K2.factor_matrices[2] == vec2.reshape((K2.shape[2], K2.ncomponents))).all()

    # multiple but not all factor matrix updates
    K3 = ttb.ktensor.from_tensor_type(K)
    vec_some = np.concatenate((vec0, vec2))
    K3.update([0, 2], vec_some)
    assert (K3.factor_matrices[0] == vec0.reshape((K3.shape[0], K3.ncomponents))).all()
    assert (K3.factor_matrices[2] == vec2.reshape((K3.shape[2], K3.ncomponents))).all()

    # weights update
    weights = np.array([100, 200])
    K1.update(-1, weights)
    assert (K1.weights == weights).all()

    # not enough weights
    with pytest.raises(AssertionError) as excinfo:
        K1.update(-1, np.ones(1))
    assert "Data is too short" in str(excinfo)

    # not enough factor matrix values
    with pytest.raises(AssertionError) as excinfo:
        K1.update(0, np.ones(3))
    assert "Data is too short" in str(excinfo)

    # modes not sorted
    with pytest.raises(AssertionError) as excinfo:
        K1.update([0, 2, 1], np.ones(np.array(K.shape).prod()))
    assert "Modes must be sorted in ascending order" in str(excinfo)

    # invalid mode
    with pytest.raises(AssertionError) as excinfo:
        K1.update(4, np.array([]))
    assert "Invalid mode: 4" in str(excinfo)

    # too much data
    vec_all_plus = np.concatenate((vec_all, np.array([0.5])))
    with pytest.warns(Warning) as record:
        K2.update([0, 1, 2], vec_all_plus)
    assert "Failed to consume all of the input data" in str(record[0].message)


@pytest.mark.indevelopment
def test_ktensor__add__(sample_ktensor_2way, sample_ktensor_3way):
    (data0, K0) = sample_ktensor_2way
    (data1, K1) = sample_ktensor_3way
    # adding ktensor to itself
    K2 = K0 + K0
    assert (np.concatenate((data0["weights"], data0["weights"])) == K2.weights).all()
    for k in range(K2.ndims):
        assert (
            np.concatenate(
                (data0["factor_matrices"][k], data0["factor_matrices"][k]), axis=1
            )
            == K2.factor_matrices[k]
        ).all()

    # shapes do not match, should raise error
    with pytest.raises(AssertionError) as excinfo:
        K3 = K0 + K1
    assert "Must be two ktensors of the same shape" in str(excinfo)

    # types do not match, should raise error
    with pytest.raises(AssertionError) as excinfo:
        K0 + np.ones((2, 2))
    assert "Cannot add instance of this type to a ktensor" in str(excinfo)
    with pytest.raises(AssertionError) as excinfo:
        K0 + ttb.tensor()
    assert "Cannot add instance of this type to a ktensor" in str(excinfo)
    with pytest.raises(AssertionError) as excinfo:
        K0 + ttb.sptensor()
    assert "Cannot add instance of this type to a ktensor" in str(excinfo)


@pytest.mark.indevelopment
def test_ktensor__getitem__(sample_ktensor_2way):
    (data, K) = sample_ktensor_2way
    # adding ktensor to itself
    assert K[0, 0] == 29
    assert K[0, 1] == 39
    assert K[1, 0] == 63
    assert K[1, 1] == 85
    assert (data["factor_matrices"][0] == K[0]).all()
    assert (data["factor_matrices"][1] == K[1]).all()
    # to return a 2D ndarray, the columns must be defined by a slice
    assert (data["factor_matrices"][0][:, [0]] == K[0][:, [0]]).all()
    with pytest.raises(AssertionError) as excinfo:
        K[0, 0, 0]
    assert "ktensor.__getitem__ requires tuples with 2 elements" in str(excinfo)
    with pytest.raises(AssertionError) as excinfo:
        K[5]
    assert (
        "ktensor.__getitem__() can only extract single elements (tuple of indices) or factor matrices (single index)"
        in str(excinfo)
    )


@pytest.mark.indevelopment
def test_ktensor__neg__(sample_ktensor_2way):
    (data0, K0) = sample_ktensor_2way
    # adding ktensor to itself
    K1 = -K0
    K2 = -K1
    assert (-data0["weights"] == K1.weights).all()
    for k in range(K1.ndims):
        assert (data0["factor_matrices"][k] == K1.factor_matrices[k]).all()
    assert K2.isequal(K0)


@pytest.mark.indevelopment
def test_ktensor__pos__(sample_ktensor_2way):
    (data0, K0) = sample_ktensor_2way
    # adding ktensor to itself
    K1 = +K0
    assert (data0["weights"] == K1.weights).all()
    for k in range(K1.ndims):
        assert (data0["factor_matrices"][k] == K1.factor_matrices[k]).all()
    assert K1.isequal(K0)


@pytest.mark.indevelopment
def test_ktensor__setitem__(sample_ktensor_2way):
    (data, K) = sample_ktensor_2way
    # adding ktensor to itself
    with pytest.raises(AssertionError) as excinfo:
        K[0, 0] = 1
    assert (
        "Subscripted assignment cannot be used to update individual elements of a ktensor."
        in str(excinfo)
    )


@pytest.mark.indevelopment
def test_ktensor__sub__(sample_ktensor_2way, sample_ktensor_3way):
    (data0, K0) = sample_ktensor_2way
    (data1, K1) = sample_ktensor_3way
    # adding ktensor to itself
    K2 = K0 - K0
    assert (np.concatenate((data0["weights"], -data0["weights"])) == K2.weights).all()
    for k in range(K2.ndims):
        assert (
            np.concatenate(
                (data0["factor_matrices"][k], data0["factor_matrices"][k]), axis=1
            )
            == K2.factor_matrices[k]
        ).all()
    # shapes do not match, should raise error
    with pytest.raises(AssertionError) as excinfo:
        K3 = K0 - K1
    assert "Must be two ktensors of the same shape" in str(excinfo)

    # types do not match, should raise error
    with pytest.raises(AssertionError) as excinfo:
        K0 - np.ones((2, 2))
    assert "Cannot subtract instance of this type from a ktensor" in str(excinfo)
    with pytest.raises(AssertionError) as excinfo:
        K0 - ttb.tensor()
    assert "Cannot subtract instance of this type from a ktensor" in str(excinfo)
    with pytest.raises(AssertionError) as excinfo:
        K0 - ttb.sptensor()
    assert "Cannot subtract instance of this type from a ktensor" in str(excinfo)


@pytest.mark.indevelopment
def test_ktensor__mul__(sample_ktensor_2way, sample_ktensor_3way):
    (data0, K0) = sample_ktensor_2way
    K1 = 2 * K0
    assert (2 * data0["weights"] == K1.weights).all()
    for k in range(K1.ndims):
        assert (data0["factor_matrices"][k] == K1.factor_matrices[k]).all()
    K2 = K0 * 2
    assert (2 * data0["weights"] == K2.weights).all()
    for k in range(K1.ndims):
        assert (data0["factor_matrices"][k] == K2.factor_matrices[k]).all()
    with pytest.raises(AssertionError) as excinfo:
        K3 = K0 * K0
    assert (
        "Multiplication by ktensors only allowed for scalars, tensors, or sptensors"
        in str(excinfo)
    )

    # test with tensor
    Tdata = np.array([[1, 2], [3, 4]])
    Tshape = (2, 2)
    T = ttb.tensor().from_data(Tdata, Tshape)
    K0T = K0 * T
    assert (K0T.double() == np.array([[29.0, 78.0], [189.0, 340.0]])).all()

    # test with sptensor
    Ssubs = np.array([[0, 0], [0, 1], [1, 1]])
    Svals = np.array([[0.5], [1.0], [1.5]])
    Sshape = (2, 2)
    S = ttb.sptensor().from_data(Ssubs, Svals, Sshape)
    K0S = S * K0
    assert (K0S.double() == np.array([[14.5, 39.0], [0.0, 127.5]])).all()


@pytest.mark.indevelopment
def test_ktensor__str__(sample_ktensor_2way):
    (data0, K0) = sample_ktensor_2way
    s = """ktensor of shape 2 x 2\nweights=[1. 2.]\nfactor_matrices[0] =\n[[1. 2.]\n [3. 4.]]\nfactor_matrices[1] =\n[[5. 6.]\n [7. 8.]]"""
    assert K0.__str__() == s
