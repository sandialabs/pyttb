"""Utilities for saving tensor data."""

# Copyright 2025 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from __future__ import annotations

import warnings
from enum import Enum
from typing import Any, TextIO

import numpy as np
from scipy.io import savemat

import pyttb as ttb
from pyttb.pyttb_utils import Shape, parse_shape


class ExportFormat(Enum):
    """Export format enumeration."""

    NUMPY = "numpy"
    MATLAB = "matlab"


def export_data(
    data: ttb.tensor | ttb.ktensor | ttb.sptensor | np.ndarray,
    filename: str,
    fmt_data: str | None = None,
    fmt_weights: str | None = None,
    index_base: int = 1,
):
    """Export tensor-related data to a file."""
    if not isinstance(data, (ttb.tensor, ttb.sptensor, ttb.ktensor, np.ndarray)):
        assert False, f"Invalid data type for export: {type(data)}"

    # open file
    with open(filename, "w") as fp:
        if isinstance(data, ttb.tensor):
            print("tensor", file=fp)
            export_size(fp, data.shape)
            # numpy always writes the array with 'C' ordering, regardless
            # of the ordering of the array.  So we must transpose it first
            # to preserve the convention of 'F' ordering of the file.
            export_array(fp, data.data.transpose(), fmt_data)

        elif isinstance(data, ttb.sptensor):
            print("sptensor", file=fp)
            export_sparse_size(fp, data)
            export_sparse_array(fp, data, fmt_data, index_base)

        elif isinstance(data, ttb.ktensor):
            print("ktensor", file=fp)
            export_size(fp, data.shape)
            export_rank(fp, data)
            export_weights(fp, data, fmt_weights)
            for n in range(data.ndims):
                print("matrix", file=fp)
                export_size(fp, data.factor_matrices[n].shape)
                export_factor(fp, data.factor_matrices[n], fmt_data)

        elif isinstance(data, np.ndarray):
            print("matrix", file=fp)
            export_size(fp, data.shape)
            export_array(fp, data, fmt_data)


def export_data_bin(
    data: ttb.tensor | ttb.ktensor | ttb.sptensor | np.ndarray,
    filename: str,
    index_base: int = 1,
):
    """Export tensor-related data to a binary file."""
    _export_data_binary(data, filename, ExportFormat.NUMPY, index_base)


def export_data_mat(
    data: ttb.tensor | ttb.ktensor | ttb.sptensor | np.ndarray,
    filename: str,
    index_base: int = 1,
):
    """Export tensor-related data to a matlab compatible binary file."""
    _export_data_binary(data, filename, ExportFormat.MATLAB, index_base)


def _export_data_binary(
    data: ttb.tensor | ttb.ktensor | ttb.sptensor | np.ndarray,
    filename: str,
    export_format: ExportFormat,
    index_base: int = 1,
):
    """Export tensor-related data to a binary file using specified format."""
    if not isinstance(data, (ttb.tensor, ttb.sptensor, ttb.ktensor, np.ndarray)):
        raise NotImplementedError(f"Invalid data type for export: {type(data)}")

    # Prepare data for export based on type
    if isinstance(data, ttb.tensor):
        export_data_dict = _prepare_tensor_data(data)
    elif isinstance(data, ttb.sptensor):
        export_data_dict = _prepare_sptensor_data(data, index_base)
    elif isinstance(data, ttb.ktensor):
        export_data_dict = _prepare_ktensor_data(data)
    elif isinstance(data, np.ndarray):
        export_data_dict = _prepare_matrix_data(data)
    else:
        raise NotImplementedError(f"Unsupported data type: {type(data)}")

    # Save using appropriate format
    if export_format == ExportFormat.NUMPY:
        with open(filename, "wb") as fp:
            np.savez(fp, allow_pickle=False, **export_data_dict)
    elif export_format == ExportFormat.MATLAB:
        savemat(filename, export_data_dict)
    else:
        raise ValueError(f"Unsupported export format: {export_format}")


def _create_header(data_type: str) -> np.ndarray:
    """Create consistent header for tensor data."""
    # TODO encode version information
    return np.array([data_type, "F"])


def _prepare_sptensor_data(data: ttb.sptensor, index_base: int = 1) -> dict[str, Any]:
    """Prepare sparse tensor data for export."""
    return {
        "header": _create_header("sptensor"),
        "shape": np.array(data.shape),
        "nnz_array": np.array([data.nnz]),
        "subs": data.subs + index_base,
        "vals": data.vals,
    }


def _prepare_tensor_data(data: ttb.tensor) -> dict[str, Any]:
    """Prepare dense tensor data for export."""
    return {
        "header": _create_header("tensor"),
        "data": data.data,
    }


def _prepare_matrix_data(data: np.ndarray) -> dict[str, Any]:
    """Prepare matrix data for export."""
    if not np.isfortran(data):
        warnings.warn(
            "Exporting a non-Fortran ordered array. "
            "For now we only support Fortran order so reshaping."
        )
    return {
        "header": _create_header("matrix"),
        "data": np.asfortranarray(data),
    }


def _prepare_ktensor_data(data: ttb.ktensor) -> dict[str, Any]:
    """Prepare ktensor data for export."""
    factor_matrices = data.factor_matrices
    num_factor_matrices = len(factor_matrices)

    export_dict = {
        "header": _create_header("ktensor"),
        "weights": data.weights,
        "num_factor_matrices": num_factor_matrices,
    }

    # Add individual factor matrices for NumPy compatibility
    for i in range(num_factor_matrices):
        export_dict[f"factor_matrix_{i}"] = factor_matrices[i]

    return export_dict


def export_size(fp: TextIO, shape: Shape):
    """Export the size of something to a file."""
    shape = parse_shape(shape)
    print(f"{len(shape)}", file=fp)  # # of dimensions on one line
    shape_str = " ".join([str(d) for d in shape])
    print(f"{shape_str}", file=fp)  # size of each dimensions on the next line


def export_rank(fp: TextIO, data: ttb.ktensor):
    """Export the rank of a ktensor to a file."""
    print(f"{len(data.weights)}", file=fp)  # ktensor rank on one line


def export_weights(fp: TextIO, data: ttb.ktensor, fmt_weights: str | None):
    """Export KTensor weights."""
    if not fmt_weights:
        fmt_weights = "%.16e"
    data.weights.tofile(fp, sep=" ", format=fmt_weights)
    print(file=fp)


def export_array(fp: TextIO, data: np.ndarray, fmt_data: str | None):
    """Export dense data."""
    if not fmt_data:
        fmt_data = "%.16e"
    data.tofile(fp, sep="\n", format=fmt_data)
    print(file=fp)


def export_factor(fp: TextIO, data: np.ndarray, fmt_data: str | None):
    """Export KTensor factor."""
    if not fmt_data:
        fmt_data = "%.16e"
    for i in range(data.shape[0]):
        row = data[i, :]
        row.tofile(fp, sep=" ", format=fmt_data)
        print(file=fp)


def export_sparse_size(fp: TextIO, A: ttb.sptensor):
    """Export the size of something to a file."""
    print(f"{len(A.shape)}", file=fp)  # # of dimensions on one line
    shape_str = " ".join([str(d) for d in A.shape])
    print(f"{shape_str}", file=fp)  # size of each dimensions on the next line
    print(f"{A.nnz}", file=fp)  # number of nonzeros


def export_sparse_array(
    fp: TextIO, A: ttb.sptensor, fmt_data: str | None, index_base: int = 1
):
    """Export sparse array data in coordinate format."""
    if not fmt_data:
        fmt_data = "%.16e"
    # 0-based indexing in package, 1-based indexing in file
    subs = A.subs + index_base
    vals = A.vals[:, 0].reshape(-1, 1)
    np.savetxt(fp, np.hstack((subs, vals)), fmt="%d " * subs.shape[1] + fmt_data)
