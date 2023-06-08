# v1.6.2 (2023-06-08)
- New: 
  - Tensor generator helpers: 
    - `tenones`, `tenzeros`, `tendiag`, `sptendiag` (PR https://github.com/sandialabs/pyttb/pull/93)
    - `tenrand`, `sptenrand` (PR https://github.com/sandialabs/pyttb/pull/100)
  - Moved to using `logging` instead of `warnings` (PR https://github.com/sandialabs/pyttb/pull/99)
- Documentation:
  - Updated coverage testing (PRs https://github.com/sandialabs/pyttb/pull/128, https://github.com/sandialabs/pyttb/pull/131, https://github.com/sandialabs/pyttb/pull/132, https://github.com/sandialabs/pyttb/pull/133)
  - Updated dev docs for contributors (PRs https://github.com/sandialabs/pyttb/pull/106, https://github.com/sandialabs/pyttb/pull/123)
  - Clarifications in `sptensor` (PR https://github.com/sandialabs/pyttb/pull/137)
  - Minor fixes for clarification (PR https://github.com/sandialabs/pyttb/pull/117)
- Fixes/Completed:
  - Fixing indexing/slicing in `tensor` (PRs https://github.com/sandialabs/pyttb/pull/109, https://github.com/sandialabs/pyttb/pull/116)
  - Fixing `ktensor` methods: `arrange`, `normalize` (PR https://github.com/sandialabs/pyttb/pull/103)
  - Streamling `khatrirao` code (PR https://github.com/sandialabs/pyttb/pull/127)
  - Avoiding class names for variables (PRs https://github.com/sandialabs/pyttb/pull/118)

# v1.6.1 (2023-04-27)
- New: 
  - Tensor generator helpers: 
    - `tenones`, `tenzeros`, `tendiag`, `sptendiag` (PR https://github.com/sandialabs/pyttb/pull/93)
    - `tenrand`, `sptenrand` (PR https://github.com/sandialabs/pyttb/pull/100)
  - Moved to using `logging` instead of `warnings` (PR https://github.com/sandialabs/pyttb/pull/99)
- Documentation:
  - Completed: `ktensor` (PR https://github.com/sandialabs/pyttb/pull/101)
  - Fixed linking for new classes (PR https://github.com/sandialabs/pyttb/pull/98)
# v1.6.0 (2023-04-16)
- API Change (PR https://github.com/sandialabs/pyttb/pull/91)
  - *Not backwards compatible*
  - `pyttb_utils.tt_dimscheck`
    - Addresses ambiguity of -0 by using `exclude_dims` (`numpy.ndarray`) parameter
  - `ktensor.ttv`, `sptensor.ttv`, `tensor.ttv`, `ttensor.ttv`
    - Use `exlude_dims` parameter instead of `-dims`
    - Explicit nameing of dimensions to exclude
  - `tensor.ttsv`
    - Use `skip_dim` (`int`) parameter instead of `-dims`
    - Exclude all dimensions up to and including `skip_dim`
- Fixes/Completed:
  - Code cleaning: minor changes associated with replacing `-dims` with `exclude_dims`/`skip_dim`
  - Authorship: PyPI only allows one author, changing to current POC
  
# v1.5.1 (2023-04-14)
- New:
  - Dev Support: 
    - Linting: support for `pyttb_utils` and `sptensor` (PR https://github.com/sandialabs/pyttb/pull/77)
    - Pre-commit: support @ntjohnson1 in (PR https://github.com/sandialabs/pyttb/pull/83)
- Fixed/Completed:
  - `hosvd`: Negative signs can be permuted for equivalent decomposition (PR https://github.com/sandialabs/pyttb/pull/82)
  - Versioning: using dynamic version in pyproject.toml (PR https://github.com/sandialabs/pyttb/pull/86)
  - Package Testing: fixed problem with subprocesses (PR https://github.com/sandialabs/pyttb/pull/87)

# v1.5.0 (2023-03-19)
- New: 
  - Added `hosvd` Tuecker decomposition (Issue #56, PR #67)
  - Added `tucker_als` Tuecker decomposition (PR #66)
  - Autoformatting using `black` and `isort` (Issue #59, PR #60)
- Updated/Ongoing:
  - Included more testing for improved coverage (Issue #78, PR #79)

# v1.4.0 (2023-02-21)
- New: 
  - Added `ttensor` class and associated tests (Issue #10, PR #51)
- Fixed/Completed:
  -  Tensor slicing now passes through to `numpy` array slicing (Issue #41, PR #50)
- Updated/Ongoing:
  - Included more testing for improved coverage (Issue #14, PR #52)

# v1.3.9 (2023-02-20)
- Remove deprecated `numpy` code associated with aliases to built-in types and ragged arrays (Issue #48, PR #49)

# v1.3.8 (2022-10-12)
- Fixed `pyttb_utils.tt_ind2sub` (Issue #45, PR #47)
- Implemented `ktensor.score` (Issue #46, PR #47)

# v1.3.7 (2022-07-17)
- Fixed `tenmat` to accept empty arrays for `rdims` or `cdims` (Issue #42, PR #43)
- Implemented `tensor.ttt` (Issue #28, PR #44)
- Adding GitHub action to publish releases to PyPi

# v1.3.6 (2022-07-15)
- Implemented `tensor.ttm` (Issue #27, PR #40)

# v1.3.5 (2022-07-12)
- Fixing `np.reshape` in `tensor.ttv` (Issue #37, PR #38)
- Fixing `np.reshape` in remainder of `tensor` (Issue #30, PR #39)

# v1.3.4 (2022-07-12)
- Fixing issues with PyPi uploads

# v1.3.3 (2022-07-11)
- Fixed indexing bug in `tensor.mttkrp` (Issue #35, PR #36)
- Updated LICENSE to compliant format (Issue #33 , PR #34)
- Now using [coveralls.io](https://coveralls.io/github/sandialabs/pyttb) for coverage reporting
- Now using [readthedocs.io](https://pyttb.readthedocs.io/en/latest/) for documentation

# v1.3.2 (2022-07-06)
- Update `tensor.nvecs` to use `tenmat` (Issue #25, PR #31)
- Full implementation of `tensor.collapse` (Issue #2, PR #32)
- Added `CHANGELOG.md`

# v1.3.1 (2022-07-01)
- Using `pyttb.__version__` for specifying package version in code and docs
- Implemented `tenmat.__setitem__` and tests (#23)
- Fix warnings in `cp_apr` associated with divide by zero (#13)
- Several documentation fixes.

# v1.3.0 (2022-07-01)
- Changed package name to `pyttb` (#24)

# v1.2.0 (2022-07-01)
- Added `tenmat` class and associated tests (#8)
- Added `tensor.__rmul__` for preadding scalars (#18)
- Fixed error in `sptensor.__lt__` that led to creation of large boolean tensors when comparing with 0 (#15)
- Matched output of `cp_als` to Matlab (#17)

# v1.1.1 (2022-06-29)
- Fixed `tensor/mttkrp` use of `np.reshape` (#16)
- Now updating version numbers in `setup.py`
 
# v1.1.0 (2022-06-27)
- Fixed `import_data` method
- New `export_data` method
- More testing

# v1.0.0 (2022-06-27)
- Initial release of Python Tensor Toolbox
