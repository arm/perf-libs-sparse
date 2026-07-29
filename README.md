# perf-libs-sparse

A library of high-performance sparse linear algebra functions targeting AArch64.

`perf-libs-sparse` provides sparse linear algebra kernels and interfaces for AArch64 systems, with C and Fortran support, configurable LP64/ILP64 interfaces, and optional OpenMP parallelism. The project may be built on Linux, Windows or macOS using CMake.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Quick Start

```bash
git clone https://github.com/arm/perf-libs-sparse.git
cd perf-libs-sparse
python3 -m pip install --user mako
CC=gcc CXX=g++ FC=gfortran cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## Installation

### Prerequisites

- CMake 3.20 or newer
- A C compiler and C++ compiler with C++17 support
- A Fortran compiler with Fortran 2003 support
- Note: GNU and LLVM are the recommended compiler suites
- Python 3 with the `mako` package available to CMake
- Arm Performance Libraries with `$ARMPL_DIR` set, or a system OpenBLAS installation if `SPARSE_USE_ARMPL=OFF` (`perf-libs-sparse` has a dependency on some CBLAS functions)

### Configure, Build, and Install

```bash
python3 -m pip install --user mako
CC=clang CXX=clang++ FC=flang cmake -S . -B build
cmake --build build --parallel
cmake --install build --prefix ./_install
```

Useful build options:

- `-DSPARSE_BUILD_SHARED=OFF` to disable the shared library build
- `-DSPARSE_BUILD_STATIC=OFF` to disable the static library build
- `-DSPARSE_ENABLE_FORTRAN=OFF` to disable the Fortran interface and module build
- `-DSPARSE_USE_ARMPL=OFF` to use a system OpenBLAS backend instead of ArmPL
- `-DSPARSE_ENABLE_OPENMP=ON` to build OpenMP variants
- `-DSPARSE_ENABLE_ASAN=ON` to enable AddressSanitizer
- `-DSPARSE_ENABLE_COVERAGE=ON` to add compiler coverage instrumentation for supported GNU/Clang toolchains
- `-DSPARSE_ENABLE_INT64=ON` to build with ILP64 interfaces
- `-DSPARSE_ENABLE_ARM64EC=ON` to target Arm64EC on Windows
- `-DSPARSE_ENABLE_TESTS=OFF` to disable building the test executables
- `-DSPARSE_ENABLE_SVE=ON` to build with SVE kernels

## Usage

Run the full test suite:

```bash
ctest --test-dir build --output-on-failure
```

Run an individual test executable:

```bash
./build/bin/spmv_basic_d_test
```

Consume the installed package from another CMake project:

```cmake
find_package(perflibs_sparse_lp64 REQUIRED CONFIG)
target_link_libraries(my_target PRIVATE perflibs_sparse_lp64::sparse_shared)
```

This is the default package for most consumers.

If you specifically need the ILP64 variant, use `perflibs_sparse_ilp64` instead.
For `find_package(... CONFIG)`, CMake can also use the standard
`<PackageName>_DIR` hint variable, so `perflibs_sparse_lp64_DIR` or
`perflibs_sparse_ilp64_DIR` may be set to the directory containing the
corresponding `*Config.cmake` file.

## Contributing

Contributions are welcome. Please follow the project guidelines in [CONTRIBUTING.md](./CONTRIBUTING.md).

## License

This project is licensed under the terms described in [LICENSE](./LICENSE).
