/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "compressed_sparse_columns.hpp"
#include "compressed_sparse_rows.hpp"
#include "convert.hpp"
#include "norm.hpp"
#include "object_helpers.hpp"
#include "pod_vector.hpp"
#include "types.hpp"
#include "util.hpp"

namespace perflibs::sparse {

template <typename T>
perflibs_csc<T> make_csc(perflibs_int_t rows, perflibs_int_t cols,
                         perflibs_int_t nnz, const T *vals,
                         const perflibs_int_t *row_indx,
                         const perflibs_int_t *col_ptr, bool no_copy) {
  if (no_copy) {
    return perflibs_csc<T>(rows, cols, vals, row_indx, col_ptr);
  } else {
    return perflibs_csc<T>(rows, cols, nnz, vals, row_indx, col_ptr);
  }
}

perflibs_sparse_matrix_shape_t get_shape_csc(perflibs_int_t m, perflibs_int_t n,
                                             const perflibs_int_t *col_ptr,
                                             const perflibs_int_t *row_indx) {

  // We currently only care about shape for spsv, which requires a square
  // matrix, so get out early if the matrix is rectangular.
  if (m != n) {
    return PERFLIBS_SPARSE_SHAPE_RECTANGULAR;
  }

  perflibs_sparse_matrix_shape_t current =
      PERFLIBS_SPARSE_SHAPE_DIAGONAL; // not strictly upper or lower

  auto index_base = col_ptr[0];
  for (perflibs_int_t i = 0; i < n; i++) {
    // offset for start of col
    const auto col_start = &row_indx[col_ptr[i] - index_base];
    // offset for end of col
    const auto col_end = &row_indx[col_ptr[i + 1] - index_base];

    if (col_end != col_start) {
      // Just check the min and max elements
      auto [minp, maxp] = std::minmax_element(col_start, col_end);
      auto ri_min = *minp - index_base;
      auto ri_max = *maxp - index_base;

      // If the col only has a diagonal element, skip any more checks
      if (!(ri_min == i && ri_max == i)) {
        if (ri_min >= i && ri_max >= i) {
          // we're in lower triangular territory, get out if we've previously
          // seen evidence of upper
          if (current == PERFLIBS_SPARSE_SHAPE_UPPER_TRIANGULAR) {
            return PERFLIBS_SPARSE_SHAPE_RECTANGULAR;
          } else {
            current = PERFLIBS_SPARSE_SHAPE_LOWER_TRIANGULAR;
          }
        } else if (ri_min <= i && ri_max <= i) {
          // we're in upper triangular territory, get out if we've previously
          // seen evidence of lower
          if (current == PERFLIBS_SPARSE_SHAPE_LOWER_TRIANGULAR) {
            return PERFLIBS_SPARSE_SHAPE_RECTANGULAR;
          } else {
            current = PERFLIBS_SPARSE_SHAPE_UPPER_TRIANGULAR;
          }
        }
        // not diagonal, and not upper or lower: rectangular
        else {
          return PERFLIBS_SPARSE_SHAPE_RECTANGULAR;
        }
      }
    }
  }

  // TODO handle diagonal specifically
  // For now, we just handle as a upper triangular matrix for CSC
  if (current == PERFLIBS_SPARSE_SHAPE_DIAGONAL) {
    return PERFLIBS_SPARSE_SHAPE_UPPER_TRIANGULAR;
  }

  return current;
}

template <typename T>
perflibs_sparse_matrix_diag_t
get_diag_csc(perflibs_int_t m, perflibs_int_t n, const perflibs_int_t *col_ptr,
             const perflibs_int_t *row_indx, const T *vals,
             perflibs_sparse_matrix_shape_t shape) {

  // We currently only care about diagonals for spsv, which requires a square
  // matrix, so get out early if the matrix is rectangular.
  if (m != n) {
    return PERFLIBS_SPARSE_DIAG_NON_UNIT;
  }

  auto index_base = col_ptr[0];

  perflibs_int_t unit_diag = 0;
  // Iterate over columns, returning early if we find a zero diagonal. If we
  // find a non-unit diagonal, we need to continue checking to see if there are
  // any zero diagonals in the remaining columns.
  for (perflibs_int_t i = 0; i < n; i++) {

    // If the column is empty we have a zero diagonal
    if (col_ptr[i + 1] - col_ptr[i] == 0) {
      return PERFLIBS_SPARSE_DIAG_ZERO;
    }

    // offset for start of col
    const auto col_start = &row_indx[col_ptr[i] - index_base];
    // offset for end of col
    const auto col_end = &row_indx[col_ptr[i + 1] - index_base];

    // offset for min/max element of col from the beginning
    const auto indx =
        std::distance(row_indx, shape == PERFLIBS_SPARSE_SHAPE_LOWER_TRIANGULAR
                                    ? std::min_element(col_start, col_end)
                                    : std::max_element(col_start, col_end));

    const auto ri = row_indx[indx] - index_base;

    // If this is the diagonal ...
    if (ri == i) {
      // Return if the value is zero
      if (vals[indx] == T(0)) {
        return PERFLIBS_SPARSE_DIAG_ZERO;
      }
      // count if unit diagonal
      if (vals[indx] == T(1)) {
        unit_diag++;
      }
    }
  }

  // A unit diagonal matrix has to have all diagonal entries equal to 1
  if (unit_diag == n) {
    return PERFLIBS_SPARSE_DIAG_UNIT;
  }

  return PERFLIBS_SPARSE_DIAG_NON_UNIT;
}

template <typename T>
perflibs_status_t fill_initial_data_csc(perflibs_spmat_top_t *A,
                                        perflibs_int_t m, perflibs_int_t n,
                                        const perflibs_int_t *row_indx,
                                        const perflibs_int_t *col_ptr,
                                        const T *vals, bool no_copy) {

  auto impl = reinterpret_cast<perflibs_spmat_impl_t<T> *>(A->impl);

  /* Check input values */
  if (m < 0) {
    impl->error_handle.perflibs_error_type =
        PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    impl->error_handle.perflibs_error_code = 2;
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }
  if (n < 0) {
    impl->error_handle.perflibs_error_type =
        PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    impl->error_handle.perflibs_error_code = 3;
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }
  if (col_ptr[0] != 0 && col_ptr[0] != 1) {
    impl->error_handle.perflibs_error_type =
        PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    impl->error_handle.perflibs_error_code = 4;
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }

  impl->m = m;
  impl->n = n;
  impl->index_base = col_ptr[0];
  impl->nnz = col_ptr[n] - impl->index_base;

  impl->spmat_format = perflibs_format_csc;
  impl->no_copy = no_copy;

  impl->shape = get_shape_csc(m, n, col_ptr, row_indx);
  impl->diag = get_diag_csc(m, n, col_ptr, row_indx, vals, impl->shape);

  impl->csc = make_csc<T>(impl->m, impl->n, impl->nnz, vals, row_indx, col_ptr,
                          no_copy);

  return PERFLIBS_STATUS_SUCCESS;
}

template perflibs_status_t
fill_initial_data_csc<float>(perflibs_spmat_top_t *A, perflibs_int_t m,
                             perflibs_int_t n, const perflibs_int_t *row_indx,
                             const perflibs_int_t *col_ptr, const float *vals,
                             bool no_copy);
template perflibs_status_t
fill_initial_data_csc<double>(perflibs_spmat_top_t *A, perflibs_int_t m,
                              perflibs_int_t n, const perflibs_int_t *row_indx,
                              const perflibs_int_t *col_ptr, const double *vals,
                              bool no_copy);
template perflibs_status_t fill_initial_data_csc<std::complex<float>>(
    perflibs_spmat_top_t *A, perflibs_int_t m, perflibs_int_t n,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_ptr,
    const std::complex<float> *vals, bool no_copy);
template perflibs_status_t fill_initial_data_csc<std::complex<double>>(
    perflibs_spmat_top_t *A, perflibs_int_t m, perflibs_int_t n,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_ptr,
    const std::complex<double> *vals, bool no_copy);

template <typename T>
perflibs_csc<T> &perflibs_csc<T>::operator=(const perflibs_csc &other) {
  if (&other == this) {
    return *this;
  }
  // Copy the vector variables
  m = other.m;
  n = other.n;
  nthreads = other.nthreads;
  par_sv = other.par_sv;
  if (m >= 0 && n >= 0) {
    auto nnz = other.col_ptr_ptr[n] - other.col_ptr_ptr[0];

    // If the vectors are populated make the const pointers point to them
    // otherwise construct a new vector
    copy_from_vector_or_ptr(&vals_ptr, vals, other.vals_ptr, other.vals, nnz);

    copy_from_vector_or_ptr(&row_indx_ptr, row_indx, other.row_indx_ptr,
                            other.row_indx, nnz);

    copy_from_vector_or_ptr(&col_ptr_ptr, col_ptr, other.col_ptr_ptr,
                            other.col_ptr, n + 1);
  }

  return *this;
}
template perflibs_csc<float> &
perflibs_csc<float>::operator=(const perflibs_csc &other);
template perflibs_csc<double> &
perflibs_csc<double>::operator=(const perflibs_csc &other);
template perflibs_csc<std::complex<float>> &
perflibs_csc<std::complex<float>>::operator=(const perflibs_csc &other);
template perflibs_csc<std::complex<double>> &
perflibs_csc<std::complex<double>>::operator=(const perflibs_csc &other);

// Scale the input values and write them into the current matrix object
template <typename T>
void perflibs_csc<T>::scale_matrix(enum perflibs_sparse_hint_value trans,
                                   T alpha) {
  auto nnz = col_ptr[n] - col_ptr[0];

  // Allocate our copy vector (in case we're copying from user's data)
  if (vals.data() != vals_ptr) {
    vals.resize(nnz);
  }

  if (trans == PERFLIBS_SPARSE_OPERATION_CONJTRANS) {
    for (perflibs_int_t i = 0; i < nnz; i++) {
      vals[i] = perflibs::sparse::conj(vals_ptr[i]) * alpha;
    }
  } else {
    for (perflibs_int_t i = 0; i < nnz; i++) {
      vals[i] = vals_ptr[i] * alpha;
    }
  }

  // Set the new pointer
  vals_ptr = vals.data();
}
template void
perflibs_csc<float>::scale_matrix(enum perflibs_sparse_hint_value trans,
                                  float alpha);
template void
perflibs_csc<double>::scale_matrix(enum perflibs_sparse_hint_value trans,
                                   double alpha);
template void perflibs_csc<std::complex<float>>::scale_matrix(
    enum perflibs_sparse_hint_value trans, std::complex<float> alpha);
template void perflibs_csc<std::complex<double>>::scale_matrix(
    enum perflibs_sparse_hint_value trans, std::complex<double> alpha);

std::pair<std::vector<perflibs_int_t>, std::vector<perflibs_int_t>>
csr2csc_struct(int64_t rows, int64_t cols, const perflibs_int_t *col_indx,
               const perflibs_int_t *row_ptr) {

  auto index_base = row_ptr[0];
  auto nnz = row_ptr[rows] - index_base;
  std::vector<perflibs_int_t> col_ptr(cols + 1);
  std::vector<perflibs_int_t> row_indx(nnz);

  // Figure out how many elements live in each column
  col_ptr[0] = index_base;
  for (auto i = 0; i < row_ptr[rows] - index_base; i++) {
    col_ptr[col_indx[i] + 1 - index_base]++;
  }

  // Turn this into a cumulative vector to get col_ptr
  for (auto i = 1; i <= cols; i++) {
    col_ptr[i] += col_ptr[i - 1];
  }

  auto col_ptr_cpy = col_ptr;

  for (auto i = 0; i < rows; i++) {
    for (auto k = row_ptr[i] - index_base; k < row_ptr[i + 1] - index_base;
         k++) {
      auto col = col_indx[k] - index_base;
      row_indx[col_ptr_cpy[col]++ - index_base] = i + index_base;
    }
  }

  return std::make_pair(std::move(col_ptr), std::move(row_indx));
}

template <typename T>
perflibs_csc<T> csr2csc(enum sparse_hint_value_internal trans, int64_t rows,
                        int64_t cols, const T *vals,
                        const perflibs_int_t *col_indx,
                        const perflibs_int_t *row_ptr) {

  auto [col_ptr, row_indx] = csr2csc_struct(rows, cols, col_indx, row_ptr);

  auto index_base = row_ptr[0];
  auto nnz = row_ptr[rows] - index_base;
  std::vector<T> vals_csc(nnz);

  auto conj = [=](T val) {
    return trans == PERFLIBS_OPERATION_CONJTRANS ? perflibs::sparse::conj(val)
                                                 : val;
  };

  auto col_ptr_cpy = col_ptr;

  for (auto i = 0; i < rows; i++) {
    for (auto k = row_ptr[i] - index_base; k < row_ptr[i + 1] - index_base;
         k++) {
      auto col = col_indx[k] - index_base;
      vals_csc[col_ptr_cpy[col]++ - index_base] = conj(vals[k]);
    }
  }

  return perflibs_csc<T>(rows, cols, nnz, vals_csc.data(), row_indx.data(),
                         col_ptr.data());
}
template perflibs_csc<float> csr2csc(enum sparse_hint_value_internal trans,
                                     int64_t rows, int64_t cols,
                                     const float *vals,
                                     const perflibs_int_t *col_indx,
                                     const perflibs_int_t *row_ptr);
template perflibs_csc<double> csr2csc(enum sparse_hint_value_internal trans,
                                      int64_t rows, int64_t cols,
                                      const double *vals,
                                      const perflibs_int_t *col_indx,
                                      const perflibs_int_t *row_ptr);
template perflibs_csc<std::complex<float>>
csr2csc(enum sparse_hint_value_internal trans, int64_t rows, int64_t cols,
        const std::complex<float> *vals, const perflibs_int_t *col_indx,
        const perflibs_int_t *row_ptr);
template perflibs_csc<std::complex<double>>
csr2csc(enum sparse_hint_value_internal trans, int64_t rows, int64_t cols,
        const std::complex<double> *vals, const perflibs_int_t *col_indx,
        const perflibs_int_t *row_ptr);

template <typename T>
perflibs_csc<T> supernodal2csc(perflibs_int_t m, perflibs_int_t n,
                               perflibs_int_t nnz, perflibs_int_t nsuper,
                               const perflibs_int_t *super_row_ptr,
                               const perflibs_int_t *super_col_indx,
                               const perflibs_int_t *row_indx,
                               const perflibs_int_t *col_ptr, const T *vals) {

  perflibs_int_t csc_vals_counter = 0;
  perflibs::sparse::pod_vector<T> vals_out(nnz);
  perflibs::sparse::pod_vector<perflibs_int_t> rows_out(nnz);
  perflibs_int_t index_base = super_col_indx[0];

  for (perflibs_int_t super_counter = 0; super_counter < nsuper;
       super_counter++) {
    // Diagonal blocks are symmetric across rows and cols.
    // Per supernode extract cols indx
    perflibs_int_t first_col_indx = super_col_indx[super_counter] - index_base;
    perflibs_int_t last_col_indx =
        super_col_indx[super_counter + 1] - index_base;
    // Per supernode extract rows indx
    perflibs_int_t first_row_indx = super_row_ptr[super_counter] - index_base;
    perflibs_int_t last_row_indx =
        super_row_ptr[super_counter + 1] - index_base;
    for (perflibs_int_t cols_indx = first_col_indx; cols_indx < last_col_indx;
         cols_indx++) {
      perflibs_int_t vals_start_indx = col_ptr[cols_indx] - index_base;
      for (perflibs_int_t bounded_row_indx = first_row_indx;
           bounded_row_indx < last_row_indx; bounded_row_indx++) {
        T val = vals[vals_start_indx];
        vals_out[csc_vals_counter] = val;
        rows_out[csc_vals_counter] = row_indx[bounded_row_indx];

        csc_vals_counter++;
        vals_start_indx++;
      }
    }
  }
  return perflibs_csc<T>(m, n, nnz, vals_out.data(), rows_out.data(), col_ptr);
}

template perflibs_csc<float>
supernodal2csc(perflibs_int_t m, perflibs_int_t n, perflibs_int_t nnz,
               perflibs_int_t nsuper, const perflibs_int_t *super_row_ptr,
               const perflibs_int_t *super_col_indx,
               const perflibs_int_t *row_indx, const perflibs_int_t *col_ptr,
               const float *vals);
template perflibs_csc<double>
supernodal2csc(perflibs_int_t m, perflibs_int_t n, perflibs_int_t nnz,
               perflibs_int_t nsuper, const perflibs_int_t *super_row_ptr,
               const perflibs_int_t *super_col_indx,
               const perflibs_int_t *row_indx, const perflibs_int_t *col_ptr,
               const double *vals);
template perflibs_csc<std::complex<float>>
supernodal2csc(perflibs_int_t m, perflibs_int_t n, perflibs_int_t nnz,
               perflibs_int_t nsuper, const perflibs_int_t *super_row_ptr,
               const perflibs_int_t *super_col_indx,
               const perflibs_int_t *row_indx, const perflibs_int_t *col_ptr,
               const std::complex<float> *vals);
template perflibs_csc<std::complex<double>>
supernodal2csc(perflibs_int_t m, perflibs_int_t n, perflibs_int_t nnz,
               perflibs_int_t nsuper, const perflibs_int_t *super_row_ptr,
               const perflibs_int_t *super_col_indx,
               const perflibs_int_t *row_indx, const perflibs_int_t *col_ptr,
               const std::complex<double> *vals);

template <typename T>
void spmv_csc(perflibs_csc<T> &csc, perflibs_sparse_hint_value trans,
              const T *x, T *y, T alpha, T beta) {

  sparse_hint_value_internal csr_trans = PERFLIBS_OPERATION_TRANS;
  if (trans == PERFLIBS_SPARSE_OPERATION_TRANS) {
    csr_trans = PERFLIBS_OPERATION_NOTRANS;
  } else if (trans == PERFLIBS_SPARSE_OPERATION_CONJTRANS) {
    csr_trans = PERFLIBS_OPERATION_CONJNOTRANS;
  }

  // Create a csr object (without copying arrays) and then call the csr kernel
  // with transpose appropriately flipped.
  auto csr = perflibs_csr<T>(csc.n, csc.m, csc.vals_ptr, csc.col_ptr_ptr,
                             csc.row_indx_ptr, csc.par_sv);

  spmv_csr(csr, csr_trans, x, y, alpha, beta);
}

template void spmv_csc<float>(perflibs_csc<float> &csc,
                              perflibs_sparse_hint_value trans, const float *x,
                              float *y, float alpha, float beta);
template void spmv_csc<double>(perflibs_csc<double> &csc,
                               perflibs_sparse_hint_value trans,
                               const double *x, double *y, double alpha,
                               double beta);
template void spmv_csc<std::complex<float>>(
    perflibs_csc<std::complex<float>> &csc, perflibs_sparse_hint_value trans,
    const std::complex<float> *x, std::complex<float> *y,
    std::complex<float> alpha, std::complex<float> beta);
template void spmv_csc<std::complex<double>>(
    perflibs_csc<std::complex<double>> &csc, perflibs_sparse_hint_value trans,
    const std::complex<double> *x, std::complex<double> *y,
    std::complex<double> alpha, std::complex<double> beta);

template <typename T>
void spsv_csc(const perflibs_csc<T> &csc, sparse_hint_value_internal trans,
              sparse_hint_value_internal uplo, sparse_hint_value_internal diag,
              T *x, const T *y, T alpha) {

  auto csr_trans = PERFLIBS_OPERATION_TRANS;
  if (trans == PERFLIBS_OPERATION_TRANS) {
    csr_trans = PERFLIBS_OPERATION_NOTRANS;
  } else if (trans == PERFLIBS_OPERATION_CONJTRANS) {
    csr_trans = PERFLIBS_OPERATION_CONJNOTRANS;
  }
  auto csr_uplo = uplo == PERFLIBS_SHAPE_UPPER_TRIANGULAR
                      ? PERFLIBS_SHAPE_LOWER_TRIANGULAR
                      : PERFLIBS_SHAPE_UPPER_TRIANGULAR;

  // Create a csr object (without copying arrays) and then call the csr kernel
  // with transpose appropriately flipped.
  auto csr = perflibs_csr<T>(csc.n, csc.m, csc.vals_ptr, csc.col_ptr_ptr,
                             csc.row_indx_ptr, csc.par_sv);

  spsv_csr(csr, csr_trans, csr_uplo, diag, x, y, alpha);
}

template void spsv_csc<float>(const perflibs_csc<float> &csc,
                              sparse_hint_value_internal trans,
                              sparse_hint_value_internal uplo,
                              sparse_hint_value_internal diag, float *x,
                              const float *y, float alpha);
template void spsv_csc<double>(const perflibs_csc<double> &csc,
                               sparse_hint_value_internal trans,
                               sparse_hint_value_internal uplo,
                               sparse_hint_value_internal diag, double *x,
                               const double *y, double alpha);
template void spsv_csc<std::complex<float>>(
    const perflibs_csc<std::complex<float>> &csc,
    sparse_hint_value_internal trans, sparse_hint_value_internal uplo,
    sparse_hint_value_internal diag, std::complex<float> *x,
    const std::complex<float> *y, std::complex<float> alpha);
template void spsv_csc<std::complex<double>>(
    const perflibs_csc<std::complex<double>> &csc,
    sparse_hint_value_internal trans, sparse_hint_value_internal uplo,
    sparse_hint_value_internal diag, std::complex<double> *x,
    const std::complex<double> *y, std::complex<double> alpha);

template <typename T1>
template <typename T2>
void perflibs_csc<T1>::gen_parallel_decomp_sv(
    perflibs_sparse_matrix_shape_t shape, const perflibs_int_t *row_ptr,
    const perflibs_int_t *col_indx) {
  perflibs::sparse::pod_vector<T2> ndeps(n);
  for (perflibs_int_t i = 0; i < n; i++) {
    uint64_t ndeps_col =
        col_ptr[i + 1] - col_ptr[i] - 1; // -1: discount diag, rows are not
                                         // considered dependents of themselves
    if (ndeps_col > std::numeric_limits<T2>::max()) {
      gen_parallel_decomp_sv<typename next_int<T2>::type>(shape, row_ptr,
                                                          col_indx);
      return;
    } else {
      ndeps[i] = ndeps_col;
    }
  }

  gen_parallel_decomp_sv_csx<T2>(shape, n, row_ptr, col_indx, ndeps.data(),
                                 par_sv);
}
template void perflibs_csc<float>::gen_parallel_decomp_sv<uint8_t>(
    perflibs_sparse_matrix_shape_t, const perflibs_int_t *,
    const perflibs_int_t *);
template void perflibs_csc<double>::gen_parallel_decomp_sv<uint8_t>(
    perflibs_sparse_matrix_shape_t, const perflibs_int_t *,
    const perflibs_int_t *);
template void
perflibs_csc<std::complex<float>>::gen_parallel_decomp_sv<uint8_t>(
    perflibs_sparse_matrix_shape_t, const perflibs_int_t *,
    const perflibs_int_t *);
template void
perflibs_csc<std::complex<double>>::gen_parallel_decomp_sv<uint8_t>(
    perflibs_sparse_matrix_shape_t, const perflibs_int_t *,
    const perflibs_int_t *);

template <typename T>
void spnorm_inf_csc(const perflibs_csc<T> &csc,
                    perflibs::sparse::remove_complex_t<T> *result) {
  using RT = perflibs::sparse::remove_complex_t<T>;

  std::vector<RT> rowsums(csc.m);
  auto index_base = csc.col_ptr_ptr[0];
  for (auto i = 0; i < csc.n; i++) {
    for (auto j = csc.col_ptr_ptr[i] - index_base;
         j < csc.col_ptr_ptr[i + 1] - index_base; j++) {
      rowsums[csc.row_indx_ptr[j] - index_base] += std::abs(csc.vals_ptr[j]);
    }
  }

  *result = spnorm_max(rowsums);
}
template void spnorm_inf_csc<float>(const perflibs_csc<float> &csc,
                                    float *result);
template void spnorm_inf_csc<double>(const perflibs_csc<double> &csc,
                                     double *result);
template void spnorm_inf_csc<std::complex<float>>(
    const perflibs_csc<std::complex<float>> &csc, float *result);
template void spnorm_inf_csc<std::complex<double>>(
    const perflibs_csc<std::complex<double>> &csc, double *result);

template <typename T>
perflibs_status_t
spelmm_csc(perflibs_sparse_hint_value transA, perflibs_spmat_impl_t<T> *impl_A,
           perflibs_sparse_hint_value transB, perflibs_spmat_impl_t<T> *impl_B,
           perflibs_spmat_t AB) {
  bool is_transA = transA != PERFLIBS_SPARSE_OPERATION_NOTRANS;
  bool is_transB = transB != PERFLIBS_SPARSE_OPERATION_NOTRANS;

  // If A needs to be transposed, convert it to CSR format
  if (is_transA) {
    auto ret = convert<T>(perflibs_format_csr, impl_A);
    if (ret != PERFLIBS_STATUS_SUCCESS) {
      return ret;
    }
  }

  // If B needs to be transposed, convert it to CSR format
  if (is_transB) {
    auto ret = convert<T>(perflibs_format_csr, impl_B);
    if (ret != PERFLIBS_STATUS_SUCCESS) {
      return ret;
    }
  }

  // A, B and C have the same dimensions
  auto m = is_transA ? impl_A->n : impl_A->m;
  auto n = is_transA ? impl_A->m : impl_A->n;

  // Get CSC arrays for matrix A
  const auto row_indxA =
      is_transA ? impl_A->csr.col_indx_ptr : impl_A->csc.row_indx_ptr;
  const auto col_ptrA =
      is_transA ? impl_A->csr.row_ptr_ptr : impl_A->csc.col_ptr_ptr;
  const auto valsA = is_transA ? impl_A->csr.vals_ptr : impl_A->csc.vals_ptr;

  // Get CSC arrays for matrix B
  const auto row_indxB =
      is_transB ? impl_B->csr.col_indx_ptr : impl_B->csc.row_indx_ptr;
  const auto col_ptrB =
      is_transB ? impl_B->csr.row_ptr_ptr : impl_B->csc.col_ptr_ptr;
  const auto valsB = is_transB ? impl_B->csr.vals_ptr : impl_B->csc.vals_ptr;

  // Set up CSC vectors for result matrix AB
  auto index_base = col_ptrA[0];

  perflibs::sparse::pod_vector<perflibs_int_t> col_ptrAB(n + 1);
  col_ptrAB[0] = index_base;

  auto min_nnz = std::min(col_ptrA[n], col_ptrB[n]);
  std::vector<perflibs_int_t> row_indxAB;
  row_indxAB.reserve(min_nnz);

  std::vector<T> valsAB;
  valsAB.reserve(min_nnz);

  // Do element-wise multiplication of A and B
  spelmm_csr_kernel(transA, col_ptrA, row_indxA, valsA, transB, col_ptrB,
                    row_indxB, valsB, n, index_base, col_ptrAB, row_indxAB,
                    valsAB);

  // Create CSC matrix from the result AB
  auto ret = fill_initial_data_csc(AB, m, n, row_indxAB.data(),
                                   col_ptrAB.data(), valsAB.data(), 0);
  if (ret != PERFLIBS_STATUS_SUCCESS) {
    return ret;
  }

  return PERFLIBS_STATUS_SUCCESS;
}
template perflibs_status_t
spelmm_csc<float>(perflibs_sparse_hint_value transA,
                  perflibs_spmat_impl_t<float> *impl_A,
                  perflibs_sparse_hint_value transB,
                  perflibs_spmat_impl_t<float> *impl_B, perflibs_spmat_t AB);

template perflibs_status_t
spelmm_csc<double>(perflibs_sparse_hint_value transA,
                   perflibs_spmat_impl_t<double> *impl_A,
                   perflibs_sparse_hint_value transB,
                   perflibs_spmat_impl_t<double> *impl_B, perflibs_spmat_t AB);

template perflibs_status_t spelmm_csc<std::complex<float>>(
    perflibs_sparse_hint_value transA,
    perflibs_spmat_impl_t<std::complex<float>> *impl_A,
    perflibs_sparse_hint_value transB,
    perflibs_spmat_impl_t<std::complex<float>> *impl_B, perflibs_spmat_t AB);

template perflibs_status_t spelmm_csc<std::complex<double>>(
    perflibs_sparse_hint_value transA,
    perflibs_spmat_impl_t<std::complex<double>> *impl_A,
    perflibs_sparse_hint_value transB,
    perflibs_spmat_impl_t<std::complex<double>> *impl_B, perflibs_spmat_t AB);

template <typename T>
perflibs_status_t
spmat_update_csc(perflibs_spmat_impl_t<T> *impl, perflibs_int_t n_updates,
                 const perflibs_int_t *row_indx, const perflibs_int_t *col_indx,
                 const T *vals) {
  auto nrows = impl->m;
  auto ncols = impl->n;
  auto index_base = impl->index_base;

  auto col_ptr_orig = impl->csc.col_ptr_ptr;
  auto row_indx_orig = impl->csc.row_indx_ptr;
  auto vals_orig = const_cast<T *>(impl->csc.vals_ptr);

  auto is_unit = impl->diag == PERFLIBS_SPARSE_DIAG_UNIT;

  for (auto i = 0; i < n_updates; i++) {
    auto row_num = row_indx[i] - index_base;
    auto col_num = col_indx[i] - index_base;
    if (row_num < 0 || row_num > nrows - 1) {
      return update_matrix_error(impl->error_handle, i + index_base);
    } else if (col_num < 0 || col_num > ncols - 1) {
      return update_matrix_error(impl->error_handle, i + index_base);
    } else {
      bool success = false;
      for (auto j = col_ptr_orig[col_num] - index_base;
           j < col_ptr_orig[col_num + 1] - index_base; j++) {
        if (row_indx_orig[j] == row_indx[i]) {
          vals_orig[j] = vals[i];
          if (row_num == col_num && is_unit && vals[i] != T(1)) {
            impl->diag = PERFLIBS_SPARSE_DIAG_NON_UNIT;
          }
          success = true;
          break;
        }
      }
      if (!success) {
        return update_matrix_error(impl->error_handle, i + index_base);
      }
    }
  }
  return PERFLIBS_STATUS_SUCCESS;
}
template perflibs_status_t
spmat_update_csc<float>(perflibs_spmat_impl_t<float> *impl,
                        perflibs_int_t n_updates,
                        const perflibs_int_t *row_indx,
                        const perflibs_int_t *col_indx, const float *vals);
template perflibs_status_t
spmat_update_csc<double>(perflibs_spmat_impl_t<double> *impl,
                         perflibs_int_t n_updates,
                         const perflibs_int_t *row_indx,
                         const perflibs_int_t *col_indx, const double *vals);
template perflibs_status_t spmat_update_csc<std::complex<float>>(
    perflibs_spmat_impl_t<std::complex<float>> *impl, perflibs_int_t n_updates,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_indx,
    const std::complex<float> *vals);
template perflibs_status_t spmat_update_csc<std::complex<double>>(
    perflibs_spmat_impl_t<std::complex<double>> *impl, perflibs_int_t n_updates,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_indx,
    const std::complex<double> *vals);

} // end namespace perflibs::sparse
