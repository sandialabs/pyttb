# Copyright 2022 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

import os

import numpy as np

import pyttb as ttb

from .pyttb_utils import *


def import_data(filename):
    # Check if file exists
    if not os.path.isfile(filename):
        assert False, f"File path {filename} does not exist."

    # import
    fp = open(filename, "r")

    # tensor type should be on the first line
    # valid: tensor, sptensor, matrix, ktensor
    data_type = import_type(fp)

    if data_type not in ["tensor", "sptensor", "matrix", "ktensor"]:
        assert False, f"Invalid data type found: {data_type}"

    if data_type == "tensor":
        shape = import_shape(fp)
        data = import_array(fp, np.prod(shape))
        return ttb.tensor().from_data(data, shape)

    elif data_type == "sptensor":
        shape = import_shape(fp)
        nz = import_nnz(fp)
        subs, vals = import_sparse_array(fp, len(shape), nz)
        return ttb.sptensor().from_data(subs, vals, shape)

    elif data_type == "matrix":
        shape = import_shape(fp)
        mat = import_array(fp, np.prod(shape))
        mat = np.reshape(mat, np.array(shape))
        return mat

    elif data_type == "ktensor":
        shape = import_shape(fp)
        r = import_rank(fp)
        weights = import_array(fp, r)
        factor_matrices = []
        for n in range(len(shape)):
            fac_type = fp.readline().strip()
            fac_shape = import_shape(fp)
            fac = import_array(fp, np.prod(fac_shape))
            fac = np.reshape(fac, np.array(fac_shape))
            factor_matrices.append(fac)
        return ttb.ktensor().from_data(weights, factor_matrices)

    # Close file
    fp.close()


def import_type(fp):
    # Import IO data type
    return fp.readline().strip().split(" ")[0]


def import_shape(fp):
    # Import the shape of something from a file
    n = int(fp.readline().strip().split(" ")[0])
    shape = [int(d) for d in fp.readline().strip().split(" ")]
    if len(shape) != n:
        assert False, "Imported dimensions are not of expected size"
    return tuple(shape)


def import_nnz(fp):
    # Import the size of something from a file
    return int(fp.readline().strip().split(" ")[0])


def import_rank(fp):
    # Import the rank of something from a file
    return int(fp.readline().strip().split(" ")[0])


def import_sparse_array(fp, n, nz):
    # Import sparse data subs and vals from coordinate format data
    subs = np.zeros((nz, n), dtype="int64")
    vals = np.zeros((nz, 1))
    for k in range(nz):
        line = fp.readline().strip().split(" ")
        # 1-based indexing in file, 0-based indexing in package
        subs[k, :] = [np.int64(i) - 1 for i in line[:-1]]
        vals[k, 0] = line[-1]
    return subs, vals


def import_array(fp, n):
    return np.fromfile(fp, count=n, sep=" ")
