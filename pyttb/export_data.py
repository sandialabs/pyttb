"""Utilities for saving tensor data."""

# Copyright 2025 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from __future__ import annotations

from typing import TextIO

import numpy as np

import pyttb as ttb
from pyttb.pyttb_utils import Shape, parse_shape


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
