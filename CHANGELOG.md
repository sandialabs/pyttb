# v1.8.4 (2026-01-05)
- Version Support:
  - Added support for Python 3.14; dropped support for Python 3.9 (https://github.com/sandialabs/pyttb/pull/459)
  - Updated support for recent versions of `numpy` and `spicy` (https://github.com/sandialabs/pyttb/pull/471)
- Added:
  - Added ability to choose zero-truncated Poisson distribution in `gcp_opt` (https://github.com/sandialabs/pyttb/pull/450)
  - Added `__radd__` support in `sptensor` (https://github.com/sandialabs/pyttb/pull/461)
  - Added `index_base` parameter in `export_data` (https://github.com/sandialabs/pyttb/pull/449)
- Fixed:
  - Fixed `copy` methods in data classes to use correct layout or underlying data (https://github.com/sandialabs/pyttb/pull/463)
  - Fixed `tensor` data layout in `import_data` (https://github.com/sandialabs/pyttb/pull/464)
  - Fixed edge case in testing `gcp_opt` (https://github.com/sandialabs/pyttb/pull/452)
  - Fixed type handling in `tensor.scale` (https://github.com/sandialabs/pyttb/pull/462)
- Improved:
  - Now using `numpy` method for improved performance in `export_data` and `import_data` (https://github.com/sandialabs/pyttb/pull/465)
- Dev:
  - Update ruff version used in CI testing (https://github.com/sandialabs/pyttb/pull/446)

# v1.8.3 (2025-08-29)
- Added:
  - Added `create_problem` data and solution generator to match TTB for MATLAB (https://github.com/sandialabs/pyttb/pull/442)
  - Added `immutable` flag to `double` methods for converting to `numpy.ndarray` data (https://github.com/sandialabs/pyttb/pull/432) 
  - Added `order` checks and inclusion in `__str__ output of data classes (https://github.com/sandialabs/pyttb/pull/373)
  - Added methods for TTB for MATLAB alignment [e.g., `tensor` printing] (https://github.com/sandialabs/pyttb/pull/360) 
- Fixed:
  - Fixed `tenmat` to enforce `order` properly (https://github.com/sandialabs/pyttb/pull/385)
  - Fixed typing change propagated from `numpy` (https://github.com/sandialabs/pyttb/pull/407)
  - Fixed divide-by-zero errors in algorithm iteration printing (https://github.com/sandialabs/pyttb/pull/431)
  - Aligned `ktensor.viz` method name to TTB for MATLAB (https://github.com/sandialabs/pyttb/pull/372)
  - Fixed `ktensor.viz` problem of changing weights of input when plotting (https://github.com/sandialabs/pyttb/pull/447)
- Improved:
  - Changed `tensor.reshape` to avoid making deep copies of data (https://github.com/sandialabs/pyttb/pull/386)
  - Improved efficiency in `tenrand` to avoid memory layout reordering (https://github.com/sandialabs/pyttb/pull/428)
  - Improved support on Windows when testing (https://github.com/sandialabs/pyttb/pull/388)
  - Many documentation improvements
- Deprecated:
  - Removed unsupported data classes that consisted solely of stubs (https://github.com/sandialabs/pyttb/pull/432)
- Dev:
  - Added link checks, code spelling to pre-commit hooks (https://github.com/sandialabs/pyttb/pull/359)
  - Removed publishing of coverage results to coveralls for all but earliest supported Python version (https://github.com/sandialabs/pyttb/pull/432)

# v1.8.2 (2025-01-06)
- Fixed:
    - Fixed layout and printing issues (https://github.com/sandialabs/pyttb/pull/354)
    - Fixed tutorial hierarchy (https://github.com/sandialabs/pyttb/pull/343)
- Improved:
    - Improved `pyttb_utils` (https://github.com/sandialabs/pyttb/pull/353)
    - Improved docs for coming from MATLAB (https://github.com/sandialabs/pyttb/pull/352)
    - Improved shape support in data classes (https://github.com/sandialabs/pyttb/pull/348)

# v1.8.1 (2024-11-11)
- Fixed: 
  - Aligning comparison operator output for data classes (https://github.com/sandialabs/pyttb/pull/331)
- Improved:
  - Getting starting documentation (https://github.com/sandialabs/pyttb/pull/324)
  - Development environment (https://github.com/sandialabs/pyttb/pull/329, https://github.com/sandialabs/pyttb/pull/330)
  - Documentation (https://github.com/sandialabs/pyttb/pull/328, https://github.com/sandialabs/pyttb/pull/334)

# v1.8.0 (2024-10-23)
- Added:
  - Added `ktensor.vis` method for visualizing CP decompositions (https://github.com/sandialabs/pyttb/pull/301)
  - Added support in `cp_als` to optimize only specific modes (https://github.com/sandialabs/pyttb/pull/302)
  - Added dependency on `matplotlib` for visualization support (https://github.com/sandialabs/pyttb/pull/301)
- Fixed:
  - Fixed timings and output formatting in `gcp_opt` (https://github.com/sandialabs/pyttb/pull/314)
- Improved:
  - Improved performance of `ktensor.full` (https://github.com/sandialabs/pyttb/pull/300)
- Deprecated:
  - Replaced `tt_to_dense_matrix` and `tt_from_dense_matrix` with `tenmat` data class and methods (https://github.com/sandialabs/pyttb/pull/294)
  - Removed support for Python 3.8 (end-of-life) (https://github.com/sandialabs/pyttb/pull/319)

# v1.7.0 (2024-10-23)
- **Breaking Changes:**
  - _API Change:_ Constructors (`__init__`) and helper functions have been combined for all data classes, leading to breaking changes; if you use `from_*` methods in your existing `pyttb` usage to create instances of data classes this will require changes. see the updated [documentation](https://pyttb.readthedocs.io) and [tutorials](https://pyttb.readthedocs.io/en/latest/tutorials.html) for examples of using the update APIs. (https://github.com/sandialabs/pyttb/pull/213, https://github.com/sandialabs/pyttb/pull/293)
  - _API Change:_ Changed constructors of main data classes to allow use by external packages that wrap existing data in memory. This allows for use of `pyttb` data classes by external packages without making copies of the data. (https://github.com/sandialabs/pyttb/pull/182)
  - API Change: `params` output of `cp_als` changed from `tuple` to `dict` (https://github.com/sandialabs/pyttb/pull/238)
  - _Deprecation:_ Removed unused `end` methods from data classes (https://github.com/sandialabs/pyttb/pull/195)
- New:
  - Changed support of `numpy` to < version 2 for backwards compatibility; will update in future release (https://github.com/sandialabs/pyttb/pull/307)
  - Added `gcp_opt` algorithm for Generalized CP decompositions (https://github.com/sandialabs/pyttb/pull/206)
  - Added `sptenmat` data class (https://github.com/sandialabs/pyttb/pull/290)
  - Added `sumtensor` data class (https://github.com/sandialabs/pyttb/pull/282)
  - Added `sptensor.squash` method (https://github.com/sandialabs/pyttb/pull/175)
  - Improved in `cp_apr` performance for `tensor`s (https://github.com/sandialabs/pyttb/pull/176)
  - Added `tensor.scale`, providing support for `ttensor` input in `cp_als` (https://github.com/sandialabs/pyttb/pull/221)
  - Added `teneye` (https://github.com/sandialabs/pyttb/pull/222)
  - Added support for different index bases in `import_data` (https://github.com/sandialabs/pyttb/pull/144)
- Documentation:
  - Added tutorials that mirror those in the Tensor Toolbox for MATLAB
  - Added documentatin for mapping between `pyttb` and Tensor Toolbox for MATLAB usage (https://github.com/sandialabs/pyttb/pull/291)
  - Completed documentation for all methods and algorithms
  - Improved RTD (readthedocs.io) support (https://github.com/sandialabs/pyttb/pull/178)
  - Added citation information for `pyttb` (https://github.com/sandialabs/pyttb/pull/268)
- Fixes/Completed:
  - Fixed indexing/slicing in `tensor` (https://github.com/sandialabs/pyttb/pull/150)
  - Fixed `sptensor.innerproduct` output (https://github.com/sandialabs/pyttb/pull/217)
  - Fixed `export_data` to write `tensor`s using the correct ordering (https://github.com/sandialabs/pyttb/pull/143)
  - Fixed ZeroDivisionError in `cp_als` (https://github.com/sandialabs/pyttb/pull/242)
  - Fixed how initial guesses are generated in `tucker_als` (https://github.com/sandialabs/pyttb/pull/283)
  - Fixed output formatting in `tucker_als` (https://github.com/sandialabs/pyttb/pull/265)
  - Fixed `sptensor.mask` problem with invalid indices (https://github.com/sandialabs/pyttb/pull/259)
  - Fixed `sptensor.logical_*` methods to generate correct output types (https://github.com/sandialabs/pyttb/pull/269)
- Development: 
  - Completed typing of all data classes and algorithms
  - Adding pre-commit hooks 
  - Added ruff usage to replace isort, pylint usage
  - Updated GitHub Actiob versions, pypi.org upload action

# v1.6.2 (2023-06-08)
- Documentation:
  - Updated coverage testing (https://github.com/sandialabs/pyttb/pull/128, https://github.com/sandialabs/pyttb/pull/131, https://github.com/sandialabs/pyttb/pull/132, https://github.com/sandialabs/pyttb/pull/133)
  - Updated dev docs for contributors (https://github.com/sandialabs/pyttb/pull/106, https://github.com/sandialabs/pyttb/pull/123)
  - Clarifications in `sptensor` (https://github.com/sandialabs/pyttb/pull/137)
  - Minor fixes for clarification (https://github.com/sandialabs/pyttb/pull/117)
- Fixes/Completed:
  - Fixing indexing/slicing in `tensor` (https://github.com/sandialabs/pyttb/pull/109, https://github.com/sandialabs/pyttb/pull/116)
  - Fixing `ktensor` methods: `arrange`, `normalize` (https://github.com/sandialabs/pyttb/pull/103)
  - Streamling `khatrirao` code (https://github.com/sandialabs/pyttb/pull/127)
  - Avoiding class names for variables (https://github.com/sandialabs/pyttb/pull/118)

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
    - Explicit naming of dimensions to exclude
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
