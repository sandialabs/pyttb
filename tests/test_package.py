# Copyright 2024 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.
from __future__ import annotations

import os
import subprocess

import pyttb as ttb


def test_package_smoke():
    """A few sanity checks to make sure things don't explode"""
    assert len(ttb.__version__) > 0
    # Make sure warnings filter doesn't crash
    ttb.ignore_warnings(False)
    ttb.ignore_warnings(True)


def test_linting():
    """Confirm linting of the project is enforced"""
    root_dir = os.path.dirname(os.path.dirname(__file__))
    toml_file = os.path.join(root_dir, "pyproject.toml")
    subprocess.run(
        f'ruff check "{root_dir}" --config "{toml_file}"',
        check=True,
        shell=True,
    )


def test_formatting():
    """Confirm file format of the project is enforced"""
    root_dir = os.path.dirname(os.path.dirname(__file__))
    toml_file = os.path.join(root_dir, "pyproject.toml")
    subprocess.run(
        f'ruff format --check "{root_dir}" --config "{toml_file}"',
        check=True,
        shell=True,
    )


def test_typing():
    """Run type checker on package"""
    import matplotlib
    from packaging.version import Version

    skip_untyped = ""
    # Hack to support backwards compatibility testing
    if Version(matplotlib.__version__) < Version("3.8.0"):
        skip_untyped = "--disable-error-code=import-untyped"
    root_dir = os.path.dirname(os.path.dirname(__file__))
    toml_file = os.path.join(root_dir, "pyproject.toml")
    subprocess.run(
        f'mypy -p pyttb  --config-file "{toml_file}" {skip_untyped}',
        check=True,
        shell=True,
    )


def test_spelling():
    """Confirm spelling is enforced"""
    root_dir = os.path.dirname(os.path.dirname(__file__))
    toml_file = os.path.join(root_dir, "pyproject.toml")
    subprocess.run(
        f'codespell --toml "{toml_file}"',
        check=True,
        shell=True,
    )
