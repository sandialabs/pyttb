"""Pyttb pytest configuration."""
# Copyright 2024 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

import numpy

# content of conftest.py
import pytest

import pyttb


@pytest.fixture(autouse=True)
def add_packages(doctest_namespace): #noqa: D103
    doctest_namespace["np"] = numpy
    doctest_namespace["ttb"] = pyttb


def pytest_addoption(parser): #noqa: D103
    parser.addoption(
        "--packaging",
        action="store_true",
        dest="packaging",
        default=False,
        help="enable slow packaging tests",
    )


def pytest_configure(config): #noqa: D103
    if not config.option.packaging:
        config.option.markexpr = "not packaging"
