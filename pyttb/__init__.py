"""pyttb: Python Tensor Toolbox."""

# Copyright 2024 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from __future__ import annotations

__version__ = "1.8.2"


import warnings

from pyttb.cp_als import cp_als
from pyttb.cp_apr import cp_apr
from pyttb.export_data import export_data
from pyttb.gcp_opt import gcp_opt
from pyttb.hosvd import hosvd
from pyttb.import_data import import_data
from pyttb.khatrirao import khatrirao
from pyttb.ktensor import ktensor
from pyttb.sptenmat import sptenmat
from pyttb.sptensor import sptendiag, sptenrand, sptensor
from pyttb.sptensor3 import sptensor3
from pyttb.sumtensor import sumtensor
from pyttb.symktensor import symktensor
from pyttb.symtensor import symtensor
from pyttb.tenmat import tenmat
from pyttb.tensor import tendiag, teneye, tenones, tenrand, tensor, tenzeros
from pyttb.ttensor import ttensor
from pyttb.tucker_als import tucker_als


def ignore_warnings(ignore=True):
    """Disable warnings."""
    if ignore:
        warnings.simplefilter("ignore")
    else:
        warnings.simplefilter("default")


ignore_warnings(True)

# Ruff inspection rules are too strict heres
__all__ = [  # noqa: PLE0604
    cp_als.__name__,
    cp_apr.__name__,
    export_data.__name__,
    gcp_opt.__name__,
    hosvd.__name__,
    import_data.__name__,
    khatrirao.__name__,
    ktensor.__name__,
    sptenmat.__name__,
    sptendiag.__name__,
    sptenrand.__name__,
    sptensor.__name__,
    sptensor3.__name__,
    sumtensor.__name__,
    symktensor.__name__,
    symtensor.__name__,
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
