"""Utilities for importing tensor data."""

# Copyright 2024 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from __future__ import annotations

import os
from typing import TextIO, Tuple, Union

import numpy as np

import pyttb as ttb


def import_data(
    filename: str, index_base: int = 1
) -> Union[ttb.sptensor, ttb.ktensor, ttb.tensor, np.ndarray]:
    """Import tensor data.

    Parameters
    ----------
    filename:
        File to import.
    index_base:
        Index basing allows interoperability (Primarily between python and MATLAB).
    """
    # Check if file exists
    if not os.path.isfile(filename):
        assert False, f"File path {filename} does not exist."

    # import
    with open(filename, "r") as fp:
        # tensor type should be on the first line
        # valid: tensor, sptensor, matrix, ktensor
        data_type = import_type(fp)

        if data_type not in ["tensor", "sptensor", "matrix", "ktensor"]:
            assert False, f"Invalid data type found: {data_type}"

        if data_type == "tensor":
            shape = import_shape(fp)
            data = import_array(fp, np.prod(shape))
            return ttb.tensor(data, shape, copy=False)

        if data_type == "sptensor":
            shape = import_shape(fp)
            nz = import_nnz(fp)
            subs, vals = import_sparse_array(fp, len(shape), nz, index_base)
            return ttb.sptensor(subs, vals, shape)

        if data_type == "matrix":
            shape = import_shape(fp)
            mat = import_array(fp, np.prod(shape))
            mat = np.reshape(mat, np.array(shape))
            return mat

        if data_type == "ktensor":
            shape = import_shape(fp)
            r = import_rank(fp)
            weights = import_array(fp, r)
            factor_matrices = []
            for _ in range(len(shape)):
                fp.readline().strip()  # Skip factor type
                fac_shape = import_shape(fp)
                fac = import_array(fp, np.prod(fac_shape))
                fac = np.reshape(fac, np.array(fac_shape))
                factor_matrices.append(fac)
            return ttb.ktensor(factor_matrices, weights, copy=False)
    raise ValueError("Failed to load tensor data")  # pragma: no cover


def import_type(fp: TextIO) -> str:
    """Extract IO data type."""
    return fp.readline().strip().split(" ")[0]


def import_shape(fp: TextIO) -> Tuple[int, ...]:
    """Extract the shape of something from a file."""
    n = int(fp.readline().strip().split(" ")[0])
    shape = [int(d) for d in fp.readline().strip().split(" ")]
    if len(shape) != n:
        assert False, "Imported dimensions are not of expected size"
    return tuple(shape)


def import_nnz(fp: TextIO) -> int:
    """Extract the number of non-zeros of something from a file."""
    return int(fp.readline().strip().split(" ")[0])


def import_rank(fp: TextIO) -> int:
    """Extract the rank of something from a file."""
    return int(fp.readline().strip().split(" ")[0])


def import_sparse_array(
    fp: TextIO, n: int, nz: int, index_base: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract sparse data subs and vals from coordinate format data."""
    subs = np.zeros((nz, n), dtype="int64")
    vals = np.zeros((nz, 1))
    for k in range(nz):
        line = fp.readline().strip().split(" ")
        subs[k, :] = [np.int64(i) - index_base for i in line[:-1]]
        vals[k, 0] = line[-1]
    return subs, vals


def import_array(fp: TextIO, n: Union[int, np.integer]) -> np.ndarray:
    """Extract numpy array from file."""
    return np.fromfile(fp, count=n, sep=" ")
