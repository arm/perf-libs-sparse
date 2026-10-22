# Changelog

All notable changes to `perf-libs-sparse` are documented in this file.

## [Unreleased]

### Added

- Added public C and Fortran reproducibility APIs to set and query the
  process-global sparse reproducibility mode.
- Added a reproducible `COO` `SpMV` path that uses serial `COO`
  accumulation when reproducibility mode is enabled.
- Added a fixed row-order calculation of the `mgmd` matrix statistic when
  reproducibility mode is enabled, preserving the existing parallel reduction
  when disabled.
- Added reproducible `CSR` and `CSC` `SpMV` execution paths. In the internal
  CSR view, transpose accumulation uses a fixed stored-entry order and
  non-transpose execution uses row-owned kernels. CSC operations use this
  view with transpose semantics reversed.
- Made `SpMV` optimization reproducibility-aware. When reproducibility mode is
  enabled, optimization bypasses timing-based auditioning.

### Changed

- Deprecated the `PERFLIBS_SPARSE_HINT_STRUCTURE` hint type and all its
  possible values. To be removed in version 27.01. This hint is not
  actually used during the optimization phase and consequently there will
  be no change in library behaviour when user applications are updated to
  avoid setting this hint.

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
