# Python Tensor Toolbox Contributor Guide

## Issues
If you are looking to get started or want to propose a change please start by checking
current or filing a new [issue](https://github.com/sandialabs/pyttb/issues).

## Working on PYTTB locally
1. clone your fork and enter the directory
    ```
    $ git clone git@github.com:<your username>/pyttb.git
    $ cd pyttb
    ```
    1. setup your desired python environment as appropriate

1. install dependencies
    ```
    $ pip install -e ".[dev]"
    $ make install_dev # shorthand for above
    ```

1. Checkout a branch and make your changes
    ```
    git checkout -b my-new-feature-branch
    ```
1. Formatters and linting
   1. Run autoformatters from root of project (they will change your code)
       ```commandline
       $ isort .
       $ black .
       ```
      1. [We](./.pre-commit-config.yaml) optionally support [pre-commit hooks](https://pre-commit.com/) for this
   1. Pylint and mypy coverage is work in progress (these only raise errors)
      ```commandline
      mypy pyttb/
      pylint pyttb/file_name.py  //Today only tensor is compliant
      ```

1. Run tests (at desired fidelity)
    1. Just doctests (enabled by default)
        ```commandline
        pytest
        ```
   1. Functional tests
        ```commandline
        pytest .
        ```
   1. All tests (linting and formatting checks)
        ```commandline
        pytest . --packaging
        ```
   1. With coverage
        ```commandline
        pytest . --cov=pyttb --cov-report=term-missing
        ```