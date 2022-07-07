```
Copyright 2022 National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
U.S. Government retains certain rights in this software.
```

[![Coverage Status](https://coveralls.io/repos/github/sandialabs/pyttb/badge.svg?branch=main)](https://coveralls.io/github/sandialabs/pyttb?branch=main)

# pyttb: Python Tensor Toolbox

## Contributors
* Danny Dunlavy, Nick Johnson, Derek Tucker

## Quick start

### Install
* User: ```python setup.py install```
* Developer: ```python setup.py develop```

### Testing
```
python -m pytest
```

### Coverage Testing
```
pytest --cov=pyttb  tests/ --cov-report=html
# output can be accessed via htmlcov/index.html
```

### Documentation
```
# requires `sphinx`
sphinx-build ./docs/source ./docs/build/html
# output can be accessed via docs/build/html/index.html
```

