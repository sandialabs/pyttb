"""pyttb: Python Tensor Toolbox."""

# Copyright 2025 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from __future__ import annotations

__version__ = "1.8.4-dev"


import warnings

from pyttb.decompositions.cp import cp_als, cp_apr, gcp_opt
from pyttb.decompositions.tucker import hosvd, tucker_als
from pyttb.export_data import export_data
from pyttb.import_data import import_data
from pyttb.khatrirao import khatrirao
from pyttb.matlab import matlab_support
from pyttb.tensors import (
    ktensor,
    sptenmat,
    sptensor,
    sumtensor,
    tenmat,
    tensor,
    ttensor,
)
from pyttb.tensors.dense import tendiag, teneye, tenones, tenrand, tenzeros
from pyttb.tensors.sparse import sptendiag, sptenrand


def ignore_warnings(ignore=True):
    """Disable warnings."""
    if ignore:
        warnings.simplefilter("ignore")
    else:
        warnings.simplefilter("default")


# Ruff inspection rules are too strict here
__all__ = [  # noqa: PLE0604
    cp_als.__name__,
    cp_apr.__name__,
    export_data.__name__,
    gcp_opt.__name__,
    hosvd.__name__,
    import_data.__name__,
    khatrirao.__name__,
    ktensor.__name__,
    matlab_support.__name__,
    sptenmat.__name__,
    sptendiag.__name__,
    sptenrand.__name__,
    sptensor.__name__,
    sumtensor.__name__,
    teneye.__name__,
    tenmat.__name__,
    tendiag.__name__,
    tenones.__name__,
    tenrand.__name__,
    tensor.__name__,
    tenzeros.__name__,
    ttensor.__name__,
    tucker_als.__name__,
]
