"""Utilities for importing tensor data."""

# Copyright 2025 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from __future__ import annotations

import os
from typing import TextIO

import numpy as np

import pyttb as ttb
from pyttb.pyttb_utils import to_memory_order


def import_data(
    filename: str, index_base: int = 1
) -> ttb.sptensor | ttb.ktensor | ttb.tensor | np.ndarray:
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
    with open(filename) as fp:
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
                fac = to_memory_order(np.reshape(fac, np.array(fac_shape)), order="F")
                factor_matrices.append(fac)
            return ttb.ktensor(factor_matrices, weights, copy=False)
    raise ValueError("Failed to load tensor data")  # pragma: no cover


def import_type(fp: TextIO) -> str:
    """Extract IO data type."""
    return fp.readline().strip().split(" ")[0]


def import_shape(fp: TextIO) -> tuple[int, ...]:
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
) -> tuple[np.ndarray, np.ndarray]:
    """Extract sparse data subs and vals from coordinate format data."""
    data = np.loadtxt(fp)
    subs = data[:, :-1].astype("int64") - index_base
    vals = data[:, -1].reshape(-1, 1)
    if subs.shape[0] != nz:
        raise ValueError("Imported nonzeros are not of expected size")
    if subs.shape[1] != n:
        raise ValueError("Imported tensor is not of expected shape")
    return subs, vals


def import_array(fp: TextIO, n: int | np.integer) -> np.ndarray:
    """Extract numpy array from file."""
    return np.fromfile(fp, count=n, sep=" ")
