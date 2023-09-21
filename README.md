```
Copyright 2022 National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
U.S. Government retains certain rights in this software.
```
[![Regression tests](https://github.com/sandialabs/pyttb/actions/workflows/regression-tests.yml/badge.svg)](https://github.com/sandialabs/pyttb/actions/workflows/regression-tests.yml)
[![Coverage Status](https://coveralls.io/repos/github/sandialabs/pyttb/badge.svg?branch=main)](https://coveralls.io/github/sandialabs/pyttb?branch=main)
[![pypi package](https://img.shields.io/pypi/v/pyttb?label=pypi%20package)](https://pypi.org/project/pyttb/)
[![image](https://img.shields.io/pypi/pyversions/pyttb.svg)](https://pypi.python.org/pypi/pyttb)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# pyttb: Python Tensor Toolbox

Welcome to `pyttb`, a set of Python classes and methods functions for 
manipulating dense, sparse, and structured tensors, along with algorithms 
for computing low-rank tensor models.

**Tensor Classes:**
* `tensor`: dense tensors
* `sptensor`: sparse tensors
* `ktensor`: Kruskal tensors
* `tenmat`: matricized tensors
* `ttensor`: Tucker tensors

**Tensor Algorithms:**
* `cp_als`, `cp_apr`: Canonical Polyadic (CP) decompositions
* `tucker_als`: Tucker decompostions

# Getting Started
For full details see our [documentation](https://pyttb.readthedocs.io).
## Quick Start
We are on pypi
```commandline
pip install pyttb
```
or install from source
```commandline
pip install .
```

# Contributing
Check out our [contributing guide](CONTRIBUTING.md).
