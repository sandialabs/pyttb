"""Testing of general package properties such as linting and formatting"""
# Copyright 2022 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

import os
import subprocess

import pytest

import pyttb as ttb


def test_package_smoke():
    """A few sanity checks to make sure things don't explode"""
    assert len(ttb.__version__) > 0
    # Make sure warnings filter doesn't crash
    ttb.ignore_warnings(False)
    ttb.ignore_warnings(True)


def test_ruff_linting():
    """Confirm linting of the project is enforce"""
    root_dir = os.path.dirname(os.path.dirname(__file__))
    toml_file = os.path.join(root_dir, "pyproject.toml")
    subprocess.run(
        f"ruff check {os.path.join(root_dir, 'pyttb')} --config {toml_file}",
        check=True,
        shell=True,
    )


@pytest.mark.packaging
def test_formatting():
    """Confirm formatting of the project is consistent"""

    source_dir = os.path.dirname(ttb.__file__)
    root_dir = os.path.dirname(source_dir)
    subprocess.run(
        f"isort {root_dir} --check --settings-path {root_dir}", check=True, shell=True
    )
    subprocess.run(f"black --check {root_dir}", check=True, shell=True)


@pytest.mark.packaging
def test_linting():
    """Confirm linting of the project is enforce"""
    root_dir = os.path.dirname(os.path.dirname(__file__))
    toml_file = os.path.join(root_dir, "pyproject.toml")
    subprocess.run(
        f"pylint {os.path.join(root_dir, 'pyttb')} --rcfile {toml_file} -j0",
        check=True,
        shell=True,
    )


@pytest.mark.packaging
def test_typing():
    """Run type checker on package"""
    root_dir = os.path.dirname(os.path.dirname(__file__))
    toml_file = os.path.join(root_dir, "pyproject.toml")
    subprocess.run(f"mypy -p pyttb  --config-file {toml_file}", check=True, shell=True)
