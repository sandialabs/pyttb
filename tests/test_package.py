"""Testing of general package properties such as linting and formatting"""
import os
import subprocess

import pyttb as ttb


def test_formatting():
    """Confirm formatting of the project is consistent"""

    enforced_files = [
        __file__,
        os.path.join(os.path.dirname(ttb.__file__), f"{ttb.tensor.__name__}.py"),
    ]
    root_dir = os.path.dirname(os.path.dirname(__file__))
    toml_file = os.path.join(root_dir, "pyproject.toml")
    for a_file in enforced_files:
        subprocess.run(f"black {a_file} --check", check=True)
        subprocess.run(f"isort {a_file} --check --settings-path {root_dir}", check=True)
        subprocess.run(f"pylint {a_file} --rcfile {toml_file}", check=True)
