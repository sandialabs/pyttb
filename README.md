```
Copyright 2022 National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
U.S. Government retains certain rights in this software.
```

# pyttb: Python Tensor Toolbox

Welcome to `pyttb`, a set of Python classes and methods functions for 
manipulating dense, sparse, and structured tensors, along with algorithms 
for computing low-rank tensor models.

**Tensor Classes:**
* `tensor`: dense tensors
* `sptensor`: sparse tensors
* `ktensor`: Kruskal tensors
* `tenmat`: matricized tensors

**Tensor Algorithms:**
* `cp_als`, `cp_apr`: Canonical Polyadic (CP) decompositions

Check out the [Documentation](https://pyttb.readthedocs.io) to get started.

---
[![Regression tests](https://github.com/sandialabs/pyttb/actions/workflows/regression-tests.yml/badge.svg)](https://github.com/sandialabs/pyttb/actions/workflows/regression-tests.yml) [![Coverage Status](https://coveralls.io/repos/github/sandialabs/pyttb/badge.svg?branch=main)](https://coveralls.io/github/sandialabs/pyttb?branch=main)
