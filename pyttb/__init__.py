# Copyright 2022 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

__version__ = "1.5.1"

import warnings

from pyttb.cp_als import cp_als
from pyttb.cp_apr import *
from pyttb.export_data import export_data
from pyttb.hosvd import hosvd
from pyttb.import_data import import_data
from pyttb.khatrirao import khatrirao
from pyttb.ktensor import ktensor
from pyttb.pyttb_utils import *
from pyttb.sptenmat import sptenmat
from pyttb.sptensor import sptensor
from pyttb.sptensor3 import sptensor3
from pyttb.sumtensor import sumtensor
from pyttb.symktensor import symktensor
from pyttb.symtensor import symtensor
from pyttb.tenmat import tenmat
from pyttb.tensor import tensor
from pyttb.ttensor import ttensor
from pyttb.tucker_als import tucker_als


def ignore_warnings(ignore=True):
    if ignore:
        warnings.simplefilter("ignore")
    else:
        warnings.simplefilter("default")


ignore_warnings(True)
