"""Utilities for saving tensor data."""

# Copyright 2025 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from __future__ import annotations

from typing import TYPE_CHECKING, TextIO

import numpy as np
from scipy.io import savemat

import pyttb as ttb
from pyttb.pyttb_utils import Shape, parse_shape

if TYPE_CHECKING:
    from io import BufferedWriter


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
    if not isinstance(data, (ttb.tensor, ttb.sptensor, ttb.ktensor, np.ndarray)):
        raise NotImplementedError(f"Invalid data type for export: {type(data)}")

    with open(filename, "wb") as fp:
        if isinstance(data, ttb.tensor):
            _export_tensor_bin(fp, data)
        elif isinstance(data, ttb.sptensor):
            _export_sptensor_bin(fp, data, index_base)
        elif isinstance(data, ttb.ktensor):
            _export_ktensor_bin(fp, data)
        elif isinstance(data, np.ndarray):
            _export_matrix_bin(fp, data)


def export_data_mat(
    data: ttb.tensor | ttb.ktensor | ttb.sptensor | np.ndarray,
    filename: str,
    index_base: int = 1,
):
    """Export tensor-related data to a matlab compatible binary file."""
    if not isinstance(data, (ttb.tensor, ttb.sptensor, ttb.ktensor, np.ndarray)):
        raise NotImplementedError(f"Invalid data type for export: {type(data)}")

    if isinstance(data, ttb.tensor):
        _export_tensor_mat(filename, data)
    elif isinstance(data, ttb.sptensor):
        _export_sptensor_mat(filename, data, index_base)
    elif isinstance(data, ttb.ktensor):
        _export_ktensor_mat(filename, data)
    elif isinstance(data, np.ndarray):
        _export_matrix_mat(filename, data)


def _export_sptensor_bin(fp: BufferedWriter, data: ttb.sptensor, index_base: int = 1):
    """Export sparse array data in coordinate format using NumPy."""
    # TODO add utility for consistent header creation
    header = np.array(["sptensor", "F"])
    shape = np.array(data.shape)
    nnz = np.array([data.nnz])
    subs = data.subs + index_base
    vals = data.vals
    np.savez(
        fp,
        allow_pickle=False,
        header=header,
        shape=shape,
        nnz=nnz,
        subs=subs,
        vals=vals,
    )


def _export_tensor_bin(fp: BufferedWriter, data: ttb.tensor):
    """Export dense tensor using NumPy."""
    # TODO add utility for consistent header creation
    header = np.array(["tensor", "F"])
    internal_data = data.data
    np.savez(
        fp,
        allow_pickle=False,
        header=header,
        data=internal_data,
    )


def _export_matrix_bin(fp: BufferedWriter, data: np.ndarray):
    """Export dense matrix using NumPy."""
    # TODO add utility for consistent header creation
    header = np.array(["matrix", "F"])
    internal_data = data
    np.savez(
        fp,
        allow_pickle=False,
        header=header,
        data=internal_data,
    )


def _export_ktensor_bin(fp: BufferedWriter, data: ttb.ktensor):
    """Export ktensor using NumPy."""
    # TODO add utility for consistent header creation
    header = np.array(["ktensor", "F"])
    factor_matrices = data.factor_matrices
    num_factor_matrices = len(factor_matrices)
    all_factor_matrices = {
        f"factor_matrix_{i}": factor_matrices[i] for i in range(num_factor_matrices)
    }
    weights = data.weights
    np.savez(
        fp,
        allow_pickle=False,
        header=header,
        num_factor_matrices=num_factor_matrices,
        weights=weights,
        **all_factor_matrices,
    )


def _export_sptensor_mat(filename: str, data: ttb.sptensor, index_base: int = 1):
    """Export sparse array data in coordinate format using savemat."""
    header = np.array(["sptensor", "F"])
    shape = np.array(data.shape)
    nnz = np.array([data.nnz])
    subs = data.subs + index_base
    vals = data.vals
    savemat(filename, dict(header=header, shape=shape, nnz=nnz, subs=subs, vals=vals))


def _export_tensor_mat(filename: str, data: ttb.tensor):
    """Export dense tensor data using savemat."""
    header = np.array(["tensor", "F"])
    internal_data = data.data
    savemat(filename, dict(header=header, data=internal_data))


def _export_matrix_mat(filename: str, data: np.ndarray):
    """Export dense matrix data using savemat."""
    header = np.array(["matrix", "F"])
    internal_data = data
    savemat(filename, dict(header=header, data=internal_data))


def _export_ktensor_mat(filename: str, data: ttb.ktensor):
    """Export ktensor data using savemat."""
    header = np.array(["ktensor", "F"])
    factor_matrices = data.factor_matrices
    weights = data.weights
    savemat(
        filename, dict(header=header, factor_matrices=factor_matrices, weights=weights)
    )


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
    # TODO: looping through all values may take a long time, can this be more efficient?
    for i in range(A.nnz):
        # 0-based indexing in package, 1-based indexing in file
        subs = A.subs[i, :] + index_base
        subs.tofile(fp, sep=" ", format="%d")
        print(end=" ", file=fp)
        val = A.vals[i][0]
        val.tofile(fp, sep=" ", format=fmt_data)
        print(file=fp)
