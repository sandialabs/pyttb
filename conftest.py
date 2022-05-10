# Copyright 2022 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

# content of conftest.py
import pytest
import numpy
import TensorToolbox
@pytest.fixture(autouse=True)
def add_packages(doctest_namespace):
    doctest_namespace['np'] = numpy
    doctest_namespace['ttb'] = TensorToolbox
