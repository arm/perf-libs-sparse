# Contributing to perf-libs-sparse

## Licensing

By contributing, you confirm that you have the right to submit the change under the repository license terms. Keep SPDX headers intact and add them to new source files where appropriate.

## Building

See [README.md](./README.md) for instructions on building the library. The unit tests are built by default.

## Code Structure

The public API is defined in [include/perflibs_sparse.h](./include/perflibs_sparse.h) and [include/perflibs_sparse.f90](./include/perflibs_sparse.f90).

Keep the C and Fortran interfaces aligned, and update [api/public_symbols.json](./api/public_symbols.json) if the API changes.

Internally, the library is built around sparse matrix and sparse vector objects that can be created in one format, carry structure and execution hints, and then be converted into the internal representation that best fits a given operation.

The implementation deliberately favors a low-abstraction style. C++17 is used mainly for templates and type-sharing, but the code stays close to C in structure: explicit data layouts, explicit control flow, and operation-specific kernels that remain easy to inspect. The loops, memory access patterns, and format-dependent branches that matter for optimization are meant to stay visible in the source rather than being hidden behind a large abstraction framework.

That design is reflected in [src](./src). The `*_api.cpp` files provide the public C and Fortran entry points. The hidden top-level matrix/vector types live in `types.hpp`, with narrower internal declaration headers for `object_helpers`, `matrix_state`, `vector_state`, `matvec`, `matmul`, `add`, `solve`, and `norm`. On the implementation side, `object_helpers.cpp` owns matrix/vector construction, copying, and typed top-level object creation. `matrix_state.cpp` owns shared matrix state, lifecycle, hints, and matrix-wide utilities, while `vector_state.cpp`, `vector_ops.cpp`, and `vector_storage.cpp` play the same object-support, algorithm, and storage roles on the sparse-vector side.

The main matrix optimization and execution families are split by concern: `matvec.cpp` for SpMV, `matmul.cpp` for SpMM/SpELMM/SDDMM, `add.cpp` for SpADD, `solve.cpp` for SpSV/SpSM, and `norm.cpp` for matrix norms. Matrix storage is intentionally distributed across the concrete representation files such as `compressed_sparse_rows.cpp`, `compressed_sparse_columns.cpp`, `coordinate_list.cpp`, `dense.cpp`, `sell_c_sigma.cpp`, `block_sparse_rows.cpp`, and `supernodal.cpp`. That keeps representation-specific behavior concrete and keeps conversion and execution choices explicit.

Some local low-level support files are also used. Files such as `pod_vector.hpp` and `reallocator.hpp` exist to keep allocation and Plain Old Datatype (POD) storage under direct project control without introducing unnecessary abstraction. API-level testing lives in [test](./test) and [fortran_tests](./fortran_tests).

## Coding style

The coding style is maintained by clang-format 21.1.0. You can install
clang-format 21.1.0 with pip:

```bash
pip3 install clang-format==21.1.0
```
