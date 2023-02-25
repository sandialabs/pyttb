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
    subprocess.run(f"isort {root_dir} --check --settings-path {root_dir}", check=True)
    subprocess.run(f"black --check {root_dir}", check=True)


@pytest.mark.packaging
def test_linting():
    """Confirm linting of the project is enforce"""

    enforced_files = [
        os.path.join(os.path.dirname(ttb.__file__), f"{ttb.tensor.__name__}.py"),
    ]
    # TODO pylint fails to import pyttb in tests
    # add mypy check
    root_dir = os.path.dirname(os.path.dirname(__file__))
    toml_file = os.path.join(root_dir, "pyproject.toml")
    for a_file in enforced_files:
        subprocess.run(f"pylint {a_file} --rcfile {toml_file}", check=True)


@pytest.mark.packaging
def test_typing():
    """Run type checker on package"""
    root_dir = os.path.dirname(os.path.dirname(__file__))
    toml_file = os.path.join(root_dir, "pyproject.toml")
    subprocess.run(f"mypy -p pyttb  --config-file {toml_file}", check=True)
