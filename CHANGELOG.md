# v1.3.2 (2022-06-27)
- Update nvecs to use tenmat (Issue #25, PR #31)
- Full implementation of collapse (Issue #2, PR #32)
- Added CHANGELOG.md

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
