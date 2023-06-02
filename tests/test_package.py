"""Testing of general package properties such as linting and formatting"""
import os
import subprocess

import pytest

import pyttb as ttb


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

    enforced_files = [
        os.path.join(os.path.dirname(ttb.__file__), f"{ttb.tensor.__name__}.py"),
        os.path.join(os.path.dirname(ttb.__file__), f"{ttb.sptensor.__name__}.py"),
        ttb.pyttb_utils.__file__,
    ]
    # TODO pylint fails to import pyttb in tests
    # add mypy check
    root_dir = os.path.dirname(os.path.dirname(__file__))
    toml_file = os.path.join(root_dir, "pyproject.toml")
    subprocess.run(
        f"pylint {' '.join(enforced_files)} --rcfile {toml_file} -j0",
        check=True,
        shell=True,
    )


@pytest.mark.packaging
def test_typing():
    """Run type checker on package"""
    root_dir = os.path.dirname(os.path.dirname(__file__))
    toml_file = os.path.join(root_dir, "pyproject.toml")
    subprocess.run(f"mypy -p pyttb  --config-file {toml_file}", check=True, shell=True)
