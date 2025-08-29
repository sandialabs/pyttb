"""Helper methods for tests."""

# Copyright 2024 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.
from __future__ import annotations


def assert_consistent_order(tensor, array):
    assert tensor._matches_order(
        array
    ), f"Expected array of order {tensor.order} but got {array.flags}."
