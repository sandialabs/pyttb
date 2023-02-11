import black
import pyttb as ttb
import pytest
import os
import subprocess


def test_formatting():
    """Confirm formatting of the project is consistent"""
    from pyttb.tensor import tensor

    enforced_files = [
        __file__,
        os.path.join(os.path.dirname(ttb.__file__), f"{tensor.__name__}.py"),
    ]
    for a_file in enforced_files:
        subprocess.run(f"black {a_file} --check", check=True)
