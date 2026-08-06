/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include "solve_parallel.hpp"
#include "util.hpp"

#include <cstdint>
#include <utility>
#include <vector>

// Forward declaration
template <typename T> struct perflibs_spmat_impl_t;

namespace perflibs::sparse {

// An internal type for Compressed Sparse Columns (CSC) structures
template <typename T> struct perflibs_csc {
  int64_t m, n;

  std::vector<T> vals;
  std::vector<perflibs_int_t> row_indx;
  std::vector<perflibs_int_t> col_ptr;

  // Pointers to either user data or the data under the vector copies
  const T *vals_ptr;
  const perflibs_int_t *row_indx_ptr;
  const perflibs_int_t *col_ptr_ptr;

  int nthreads = perflibs::sparse::omp::get_max_threads();

  /// A type containing details of the sparse triangular solve parallel setup
  par_sv_t par_sv;

  // Set m, n negative so that we know when to copy data in the copy operator
  perflibs_csc()
      : m(-1), n(-1), vals_ptr(nullptr), row_indx_ptr(nullptr),
        col_ptr_ptr(nullptr), par_sv() {};
  perflibs_csc(int64_t rows, int64_t cols, int64_t nnz, const T *vals,
               const perflibs_int_t *row_indx, const perflibs_int_t *col_ptr)
      : m(rows), n(cols), vals(vals, vals + nnz),
        row_indx(row_indx, row_indx + nnz),
        col_ptr(col_ptr, col_ptr + cols + 1), vals_ptr(this->vals.data()),
        row_indx_ptr(this->row_indx.data()), col_ptr_ptr(this->col_ptr.data()),
        par_sv() {}
  perflibs_csc(int64_t rows, int64_t cols, const T *vals,
               const perflibs_int_t *row_indx, const perflibs_int_t *col_ptr)
      : m(rows), n(cols), vals_ptr(vals), row_indx_ptr(row_indx),
        col_ptr_ptr(col_ptr), par_sv() {}

  perflibs_csc &operator=(const perflibs_csc &other);
  perflibs_csc(const perflibs_csc &other) { *this = other; }
  perflibs_csc &operator=(perflibs_csc &&other) = default;
  perflibs_csc(perflibs_csc &&other) = default;

  void scale_matrix(enum perflibs_sparse_hint_value trans, T alpha);

  /**
   * Generates the structure required for a SpTRSV parallel decomposition of the
   * matrix. Template parameter is the integer type to use in tracking row
   * dependencies. We optimistically try uint8, then fall back to uint16, etc.
   * if the dependency count is too large. We use the CSC format in parallel if
   * we're solving the transpose of the matrix. CSR structure arrays are passed
   * in because they contain row-wise dependencies (needed in the transpose
   * case).
   * @param [in] shape    Whether the matrix is upper or lower triangular
   * @param [in] row_ptr  The row pointers, taken from the CSR representation of
   * the matrix
   * @param [in] col_indx The column indices, taken from the CSR representation
   * of the matrix
   */
  template <typename T2>
  void gen_parallel_decomp_sv(perflibs_sparse_matrix_shape_t shape,
                              const perflibs_int_t *row_ptr,
                              const perflibs_int_t *col_indx);

  /// decide whether the diagonal elements are in the expected positions for all
  /// columns
  bool diag_in_place(bool lower) const noexcept {
    const perflibs_int_t base = col_ptr_ptr[0];

    for (int64_t j = 0; j < n; ++j) {
      const auto start = col_ptr_ptr[j] - base;
      const auto end = col_ptr_ptr[j + 1] - base;
      const perflibs_int_t diag = j + base;

      if (lower) {
        if (row_indx_ptr[start] != diag)
          return false;
      } else {
        if (row_indx_ptr[end - 1] != diag)
          return false;
      }
    }
    return true;
  }
};

template <typename T>
perflibs_status_t fill_initial_data_csc(perflibs_spmat_t A, perflibs_int_t m,
                                        perflibs_int_t n,
                                        const perflibs_int_t *row_indx,
                                        const perflibs_int_t *col_ptr,
                                        const T *vals, bool no_copy);

/// Returns pair containing col_ptr and row_indx only
std::pair<std::vector<perflibs_int_t>, std::vector<perflibs_int_t>>
csr2csc_struct(int64_t rows, int64_t cols, const perflibs_int_t *col_indx,
               const perflibs_int_t *row_ptr);

template <typename T>
perflibs_csc<T> csr2csc(enum sparse_hint_value_internal trans, int64_t rows,
                        int64_t cols, const T *vals,
                        const perflibs_int_t *col_indx,
                        const perflibs_int_t *row_ptr);

template <typename T>
perflibs_csc<T> supernodal2csc(perflibs_int_t m, perflibs_int_t n,
                               perflibs_int_t nnz, perflibs_int_t nsuper,
                               const perflibs_int_t *super_row_ptr,
                               const perflibs_int_t *super_col_indx,
                               const perflibs_int_t *row_indx,
                               const perflibs_int_t *col_ptr, const T *vals);
template <typename T>
void spmv_csc(perflibs_csc<T> &csc, perflibs_sparse_hint_value trans,
              const T *x, T *y, T alpha, T beta);

template <typename T>
void spsv_csc(const perflibs_csc<T> &csc, sparse_hint_value_internal trans,
              sparse_hint_value_internal uplo, sparse_hint_value_internal diag,
              T *x, const T *y, T alpha);

template <typename T>
void spnorm_inf_csc(const perflibs_csc<T> &csc,
                    perflibs::sparse::remove_complex_t<T> *result);

template <typename T>
perflibs_status_t
spmat_update_csc(perflibs_spmat_impl_t<T> *impl, perflibs_int_t n_updates,
                 const perflibs_int_t *row_indx, const perflibs_int_t *col_indx,
                 const T *vals);

template <typename T>
perflibs_status_t
spelmm_csc(perflibs_sparse_hint_value transA, perflibs_spmat_impl_t<T> *impl_A,
           perflibs_sparse_hint_value transB, perflibs_spmat_impl_t<T> *impl_B,
           perflibs_spmat_t AB);
} // namespace perflibs::sparse
