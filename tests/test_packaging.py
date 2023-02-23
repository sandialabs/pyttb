import pytest

import pyttb as ttb


def test_package_smoke():
    """A few sanity checks to make sure things don't explode"""
    assert len(ttb.__version__) > 0
    # Make sure warnings filter doesn't crash
    ttb.ignore_warnings(False)
    ttb.ignore_warnings(True)
