# Changelog

All notable changes to `perf-libs-sparse` are documented in this file.

## [Unreleased]

### Added

### Changed

### Fixed

- Fixed SVE dispatch being compiled out when `SPARSE_ENABLE_SVE=ON` with
  baseline compiler flags. Previously, this required adding
  `-march=armv8-a+sve` to `DEFAULT_SPARSE_COMMON_FLAGS`; `spec.cpp` now
  uses `PERFLIBS_HAVE_SVE_KERNELS` while remaining compiled for the
  baseline armv8.0.

## [26.07] - 2026-07-09

### Added

- Added optimized `SpSM` execution paths for `CSR` and `CSC`.
- Added `SpSM` test coverage for all datatypes.

### Changed

- Updated `SpSM` dense-matrix handling to directly work with row-major
  and column-major `X`/`Y` layouts.
- Improved `SpSM` execution when optimize and exec transpose hints
  differ, with safer fallback handling for non-supernodal matrices.

### Fixed

- Fixed datatype validation in sparse matrix kernels so mismatched
  matrix datatypes are rejected correctly.
- Fixed null-matrix transpose handling so rectangular null matrices
  update shape correctly.
- Matrix Market benchmark readers improved with stricter input
  validation and allocation overflow checks.

## [26.05] - 2026-05-29

### Added

- Added a reference implementation for `SpSM`.
- Added `SpSM` benchmark and example code.

### Changed

- Updated standalone benchmark timers to use a portable monotonic clock.
- Updated Windows OpenMP workaround comments.
- Replaced internal references with upstream LLVM/OpenMP issue links.
