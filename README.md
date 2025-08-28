```
Copyright 2025 National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
U.S. Government retains certain rights in this software.
```
[![Regression tests](https://github.com/sandialabs/pyttb/actions/workflows/regression-tests.yml/badge.svg)](https://github.com/sandialabs/pyttb/actions/workflows/regression-tests.yml)
[![Coverage Status](https://coveralls.io/repos/github/sandialabs/pyttb/badge.svg)](https://coveralls.io/github/sandialabs/pyttb)
[![pypi package](https://img.shields.io/pypi/v/pyttb?label=pypi%20package)](https://pypi.org/project/pyttb/)
[![image](https://img.shields.io/pypi/pyversions/pyttb.svg)](https://pypi.python.org/pypi/pyttb)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# pyttb: Python Tensor Toolbox

Welcome to `pyttb`, a refactor of the 
[Tensor Toolbox for MATLAB](https://www.tensortoolbox.org) in Python.

This package contains data classes and methods for manipulating dense, 
sparse, and structured tensors, along with algorithms for computing 
low-rank tensor decompositions:

- Data Classes: 
[`tensor`](https://pyttb.readthedocs.io/en/stable/tensor.html "dense tensors"), 
[`sptensor`](https://pyttb.readthedocs.io/en/stable/sptensor.html "sparse tensors"), 
[`ktensor`](https://pyttb.readthedocs.io/en/stable/ktensor.html "Kruskal tensors"), 
[`ttensor`](https://pyttb.readthedocs.io/en/stable/ttensor.html "Tucker tensors"), 
[`tenmat`](https://pyttb.readthedocs.io/en/stable/tenmat.html "matricized dense tensors"), 
[`sptenmat`](https://pyttb.readthedocs.io/en/stable/sptenmat.html "matricized sparse tensors"), 
[`sumtensor`](https://pyttb.readthedocs.io/en/stable/sumtensor.html "implicit sum of tensors")
- Algorithms:
[`cp_als`](https://pyttb.readthedocs.io/en/stable/cpals.html "CP decomposition via Alternating Least Squares"),
[`cp_apr`](https://pyttb.readthedocs.io/en/stable/cpapr.html "CP decomposition via Alternating Poisson Regression"), 
[`gcp_opt`](https://pyttb.readthedocs.io/en/stable/gcpopt.html "Generalized CP decomposition"), 
[`hosvd`](https://pyttb.readthedocs.io/en/stable/hosvd.html "Tucker decomposition via Higher Order Singular Value Decomposition"),
[`tucker_als`](https://pyttb.readthedocs.io/en/stable/tuckerals.html "Tucker decomposition via Alternating Least Squares")

## Quick Start

### Installation
```commandline
python3 -m pip install pyttb
```

### Example
```python
>>> import pyttb as ttb
>>> X = ttb.tenrand((2,2,2))
>>> type(X)
<class 'pyttb.tensor.tensor'>
>>> M = ttb.cp_als(X, rank=1)
CP_ALS:
 Iter 0: f = 7.367245e-01 f-delta = 7.4e-01
 Iter 1: f = 7.503069e-01 f-delta = 1.4e-02
 Iter 2: f = 7.508240e-01 f-delta = 5.2e-04
 Iter 3: f = 7.508253e-01 f-delta = 1.3e-06
 Final f = 7.508253e-01
 ```

### Memory layout
For historical reasons we use Fortran memory layouts, where numpy by default uses C.
This is relevant for indexing. In the future we hope to extend support for both.
```python
>>> import numpy as np
>>> c_order = np.arange(8).reshape((2,2,2))
>>> f_order = np.arange(8).reshape((2,2,2), order="F")
>>> print(c_order[0,1,1])
3
>>> print(f_order[0,1,1])
6
```

<!-- markdown-link-check-disable -->
### Getting Help
- [Documentation](https://pyttb.readthedocs.io)
- [Tutorials](https://pyttb.readthedocs.io/en/stable/tutorials.html)
- [Info for users coming from MATLAB](https://pyttb.readthedocs.io/en/stable/for_matlab_users.html)
- Learn about tensor decompositions: 
[tensor paper](https://doi.org/10.1137/07070111X "Tensor Decompositions and Applications by Tamara G. Kolda, Brett W. Bader"), 
[tensor book](https://www.mathsci.ai/post/tensor-textbook/ "Tensor Decompositions for Data Science by Grey Balard and Tamara G. Kolda") 
<!-- markdown-link-check-enable -->

### Contributing
- [Report a bug](https://github.com/sandialabs/pyttb/issues/new)
- [Guide for contributors](CONTRIBUTING.md)
- [List of contributors](CONTRIBUTORS.md)

### Citing pyttb in your work 
If you use pyttb in your work, please cite it using the citation info [here](CITATION.bib).
