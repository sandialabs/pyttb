```
Copyright 2024 National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
U.S. Government retains certain rights in this software.
```

# Python Tensor Toolbox Contributor Guide

## Issues
If you are looking to get started or want to propose a change please start by checking
current or filing a new [issue](https://github.com/sandialabs/pyttb/issues).

## Working on PYTTB locally
1. Clone your fork and enter the directory
    ```
    git clone git@github.com:<your username>/pyttb.git
    cd pyttb
    ```
    1. Setup your desired python environment as appropriate

1. Install dependencies
   
   Most changes only require dev options
    ```commandline
    pip install -e ".[dev]"
    ```

   But if you are making larger interface changes or updating tutorials/documentation
   you can also add the required packages for documentation.
   ```commandline
   pip install -e ".[dev,doc]"
   ```

1. Checkout a branch and make your changes
    ```
    git checkout -b my-new-feature-branch
    ```
1. Formatters and linting
   1. Run autoformatters and linting from root of project (they will change your code)
       ```commandline
       ruff check . --fix
       ruff format
       ```
      1. Ruff's `--fix` won't necessarily address everything and may point out issues that need manual attention
      1. [We](./.pre-commit-config.yaml) optionally support [pre-commit hooks](https://pre-commit.com/) for this
         1. Alternatively, you can run `pre-commit run --all-files` from the command line if you don't want to install the hooks.
   1. Check typing
      ```commandline
      mypy pyttb/
      ```

1. Run tests (at desired fidelity)
   1. Just doctests (enabled by default)
        ```commandline
        pytest .
        ```
   1. Functional tests
        ```commandline
        pytest tests
        ```
   1. With coverage
        ```commandline
        pytest tests --cov=pyttb --cov-report=term-missing
        ```

1. (Optionally) Building documentation and tutorials
   1. From project root
   ```commandline
   sphinx-build ./docs/source ./docs/build
   ```
   2. Clear notebook outputs if run locally see `nbstripout` in our [pre-commit configuration](.pre-commit-config.yaml)

## GitHub Workflow

### Proposing Changes

If you want to propose a change to Python Tensor Toolbox, follow these steps:

1. **Create an Issue**
    - Use the [Issues tab](https://github.com/sandialabs/pyttb/issues) of the Python Tensor Toolbox GitHub repository and click on the "New issue" button.

1. **Fork the Repository and Create a New Branch**
    - Navigate to the main page of the Python Tensor Toolbox GitHub repository and click on the "Fork" button to create a copy of the repository under your own account.
    - Clone the forked repository to your local machine.
    - In your local repository, create a new branch for your proposed changes.

1. **Update the Code**
    - Make the necessary updates in your local environment.
    - After making your changes, stage and commit them with an informative message.

1. **Push your Changes and Create a Pull Request**
    - Push your changes to the new branch in your fork of the repository.
    - Navigate to the main page of the Python Tensor Toolbox repository and click on the "New pull request" button.
    - In the "base repository" dropdown, select the original Python Tensor Toolbox repository. In the "base" dropdown, select the branch where you want your changes to be merged.
    - In the "head repository" dropdown, select your forked repository. In the "compare" dropdown, select the branch with your changes.
    - Write a title and description for your pull request.

1. **Review Process**
    - After creating a pull request, wait for the Github CI tests to pass.
    - If any test fails, review your code, make necessary changes, and push your code again to the same branch in your forked repository.
    - If there are any comments or requested changes from the Python Tensor Toolbox team, address them and push any additional changes to the same branch.
    - Once your changes are approved, a repository admin will merge your pull request.
