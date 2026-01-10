"""Utilities for importing tensor data."""

# Copyright 2025 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from __future__ import annotations

import os
from typing import TextIO

import numpy as np
from scipy.io import loadmat

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


def import_data_bin(
    filename: str,
    index_base: int = 1,
) -> ttb.sptensor | ttb.ktensor | ttb.tensor | np.ndarray:
    """Import tensor-related data from a binary file."""

    def load_bin_data(filename: str):
        npzfile = np.load(filename, allow_pickle=False)
        return {
            "header": npzfile["header"][0],
            "data": npzfile.get("data"),
            "shape": tuple(npzfile["shape"]) if "shape" in npzfile else None,
            "subs": npzfile.get("subs"),
            "vals": npzfile.get("vals"),
            "num_factor_matrices": int(npzfile["num_factor_matrices"])
            if "num_factor_matrices" in npzfile
            else None,
            "factor_matrices": [
                npzfile[f"factor_matrix_{i}"]
                for i in range(int(npzfile["num_factor_matrices"]))
            ]
            if "num_factor_matrices" in npzfile
            else None,
            "weights": npzfile.get("weights"),
        }

    return _import_tensor_data(filename, index_base, load_bin_data)


def import_data_mat(
    filename: str,
    index_base: int = 1,
) -> ttb.sptensor | ttb.ktensor | ttb.tensor | np.ndarray:
    """Import tensor-related data from a MATLAB file."""

    def load_mat_data(filename: str):
        mat_data = loadmat(filename)
        header = mat_data["header"][0]
        return {
            "header": header.split()[0],
            "data": mat_data.get("data"),
            "shape": tuple(mat_data["shape"][0]) if "shape" in mat_data else None,
            "subs": mat_data.get("subs"),
            "vals": mat_data.get("vals"),
            "num_factor_matrices": int(mat_data["num_factor_matrices"])
            if "num_factor_matrices" in mat_data
            else None,
            "factor_matrices": [
                mat_data[f"factor_matrix_{i}"]
                for i in range(int(mat_data["num_factor_matrices"]))
            ]
            if "num_factor_matrices" in mat_data
            else None,
            "weights": mat_data.get("weights").flatten()
            if "weights" in mat_data
            else None,
        }

    return _import_tensor_data(filename, index_base, load_mat_data)


def _import_tensor_data(
    filename: str,
    index_base: int,
    data_loader,
) -> ttb.sptensor | ttb.ktensor | ttb.tensor | np.ndarray:
    """Generalized function to import tensor data from different file formats.

    Parameters
    ----------
    filename:
        File to import.
    index_base:
        Index basing allows interoperability (Primarily between python and MATLAB).
    data_loader:
        Function that loads and structures the data from the file.
    """
    # Check if file exists
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"File path {filename} does not exist.")

    loaded_data = data_loader(filename)
    data_type = loaded_data["header"]

    if data_type not in ["tensor", "sptensor", "matrix", "ktensor"]:
        raise ValueError(f"Invalid data type found: '{data_type}'")

    if data_type == "tensor":
        data = loaded_data["data"]
        return ttb.tensor(data)
    elif data_type == "sptensor":
        shape = loaded_data["shape"]
        subs = loaded_data["subs"] - index_base
        vals = loaded_data["vals"]
        return ttb.sptensor(subs, vals, shape)
    elif data_type == "matrix":
        data = loaded_data["data"]
        return data
    elif data_type == "ktensor":
        factor_matrices = loaded_data["factor_matrices"]
        weights = loaded_data["weights"]
        return ttb.ktensor(factor_matrices, weights)

    raise ValueError(f"Invalid data type found: {data_type}")


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
