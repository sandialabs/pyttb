"""Internal tools to aid in building MATLAB support.

Tensor classes can use these common tools, where matlab_support uses tensors.
matlab_support can depend on this, but tensors and this shouldn't depend on it.
Probably best for everything here to be private functions.
"""

# Copyright 2025 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.
from __future__ import annotations

import textwrap

import numpy as np


def _matlab_array_str(
    array: np.ndarray,
    format: str | None = None,
    name: str | None = None,
    skip_name: bool = False,
) -> str:
    """Convert numpy array to string more similar to MATLAB."""
    if name is None:
        name = type(array).__name__
    header_str = ""
    body_str = ""
    if len(array.shape) > 2:
        matlab_str = ""
        # Iterate over all possible slices (in Fortran order)
        for index in np.ndindex(
            array.shape[2:][::-1]
        ):  # Skip the first two dimensions and reverse the order
            original_index = index[::-1]  # Reverse the order back to the original
            # Construct the slice indices
            slice_indices: tuple[int | slice, ...] = (
                slice(None),
                slice(None),
                *original_index,
            )
            slice_data = array[slice_indices]
            matlab_str += f"{name}(:,:, {', '.join(map(str, original_index))}) ="
            matlab_str += "\n"
            array_str = _matlab_array_str(slice_data, format, name, skip_name=True)
            matlab_str += textwrap.indent(array_str, "\t")
            matlab_str += "\n"
        return matlab_str[:-1]  # Trim extra newline
    elif len(array.shape) == 2:
        header_str += f"{name}(:,:) ="
        for row in array:
            if format is None:
                body_str += " ".join(f"{val}" for val in row)
            else:
                body_str += " ".join(f"{val:{format}}" for val in row)
            body_str += "\n"
    else:
        header_str += f"{name}(:) ="
        for val in array:
            if format is None:
                body_str += f"{val}"
            else:
                body_str += f"{val:{format}}"
            body_str += "\n"

    if skip_name:
        return body_str
    return header_str + "\n" + textwrap.indent(body_str[:-1], "\t")
