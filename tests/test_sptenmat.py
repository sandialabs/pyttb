# Copyright 2022 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

import numpy as np
import pytest

import pyttb as ttb

DEBUG_tests = False


@pytest.fixture()
def sample_sptensor_3way():
    subs = np.array([[0, 0, 0], [0, 0, 2], [1, 1, 1], [3, 3, 3]])
    vals = np.array([[10.5], [1.5], [2.5], [3.5]])
    rdims = np.array([0, 1])
    cdims = np.array([2])
    tshape = (4, 4, 4)
    data = {
        "subs": subs,
        "vals": vals,
        "rdims": rdims,
        "cdims": cdims,
        "tshape": tshape,
    }
    sptensorInstance = ttb.sptensor(subs, vals, tshape)
    return data, sptensorInstance


@pytest.fixture()
def sample_sptenmat():
    subs = np.array([[11, 1], [2, 2], [3, 2], [3, 3]])
    vals = np.array([[0.5], [1.5], [2.5], [3.5]])
    rdims = np.array([0, 1])
    cdims = np.array([2])
    tshape = (4, 4, 4)
    data = {
        "subs": subs,
        "vals": vals,
        "rdims": rdims,
        "cdims": cdims,
        "tshape": tshape,
    }
    sptenmatInstance = ttb.sptenmat.from_data(subs, vals, rdims, cdims, tshape)
    return data, sptenmatInstance


def test_sptenmat_initialization_empty():
    empty = np.array([])

    # No args
    sptenmatInstance = ttb.sptenmat()
    assert sptenmatInstance.shape == ()
    assert sptenmatInstance.tshape == ()
    assert (sptenmatInstance.rdims == empty).all()
    assert (sptenmatInstance.cdims == empty).all()
    assert (sptenmatInstance.subs == empty).all()
    assert (sptenmatInstance.vals == empty).all()


def test_sptenmat_initialization_from_data(sample_sptenmat):
    (params, sptenmatInstance) = sample_sptenmat

    # subs and vals should be sorted from output of np.unique
    subs = np.array([[2, 2], [3, 2], [3, 3], [11, 1]])
    vals = np.array([[1.5], [2.5], [3.5], [0.5]])
    rdims = np.array([0, 1])
    cdims = np.array([2])
    tshape = (4, 4, 4)
    shape = (np.prod(np.array(tshape)[rdims]), np.prod(np.array(tshape)[cdims]))

    # Constructor from data: subs, vals, rdims, cdims, and tshape
    S = ttb.sptenmat.from_data(subs, vals, rdims, cdims, tshape)
    assert (S.subs == subs).all()
    assert (S.vals == vals).all()
    assert (S.rdims == rdims).all()
    assert (S.cdims == cdims).all()
    assert S.tshape == tshape
    assert S.shape == shape


def test_sptenmat__str__(sample_sptensor_3way):
    (params3, sptensorInstance3) = sample_sptensor_3way
    tshape = params3["tshape"]
    rdims = params3["rdims"]
    cdims = params3["cdims"]

    # Empty
    sptenmatInstance = ttb.sptenmat()
    s = ""
    s += "sptenmat corresponding to a sptensor of shape () with 0 nonzeros\n"
    s += "rdims = [  ] (modes of sptensor corresponding to rows)\n"
    s += "cdims = [  ] (modes of sptensor corresponding to columns)\n"
    assert s == sptenmatInstance.__str__()

    # Test 3D
    sptenmatInstance3 = ttb.sptenmat.from_tensor_type(
        sptensorInstance3, rdims, cdims, tshape
    )
    s = ""
    s += "sptenmat corresponding to a sptensor of shape "
    s += (" x ").join([str(int(d)) for d in sptenmatInstance3.tshape])
    s += " with " + str(sptenmatInstance3.vals.size) + " nonzeros"
    s += "\n"
    s += "rdims = "
    s += "[ " + (", ").join([str(int(d)) for d in sptenmatInstance3.rdims]) + " ] "
    s += "(modes of sptensor corresponding to rows)\n"
    s += "cdims = "
    s += "[ " + (", ").join([str(int(d)) for d in sptenmatInstance3.cdims]) + " ] "
    s += "(modes of sptensor corresponding to columns)\n"

    for i in range(0, sptenmatInstance3.subs.shape[0]):
        s += "\t"
        s += "["
        idx = sptenmatInstance3.subs[i, :]
        s += str(idx.tolist())[1:]
        s += " = "
        s += str(sptenmatInstance3.vals[i][0])
        if i < sptenmatInstance3.subs.shape[0] - 1:
            s += "\n"
    assert s == sptenmatInstance3.__str__()
