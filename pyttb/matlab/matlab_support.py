"""A limited number of utilities to support users coming from MATLAB."""

# Copyright 2025 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from typing import Optional, Union

import numpy as np

from pyttb.tensor import tensor

from .matlab_utilities import _matlab_array_str

PRINT_CLASSES = Union[tensor, np.ndarray]


def matlab_print(
    data: Union[tensor, np.ndarray],
    format: Optional[str] = None,
    name: Optional[str] = None,
):
    """Print data in a format more similar to MATLAB.

    Arguments
    ---------
    data: Object to print
    format: Numerical formatting
    """
    if not isinstance(data, (tensor, np.ndarray)):
        raise ValueError(
            f"matlab_print only supports inputs of type {PRINT_CLASSES} but got"
            f" {type(data)}."
        )
    if isinstance(data, np.ndarray):
        print(_matlab_array_str(data, format, name))
        return
    print(data._matlab_str(format, name))
