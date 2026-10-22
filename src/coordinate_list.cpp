/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "coordinate_list.hpp"
#include "norm.hpp"
#include "object_helpers.hpp"
#include "pod_vector.hpp"
#include "types.hpp"
#include <algorithm>

#include <perflibs_sparse_reproducibility.h>

// Implementations of functions required to support CSR

namespace perflibs::sparse {

template <typename T>
perflibs_coo<T> make_coo(perflibs_int_t m, perflibs_int_t n, perflibs_int_t nnz,
                         perflibs_int_t index_base, const T *vals,
                         const perflibs_int_t *row_indx,
                         const perflibs_int_t *col_indx, bool no_copy) {
  if (no_copy) {
    return perflibs_coo<T>(m, n, index_base, vals, row_indx, col_indx, nnz);
  } else {
    return perflibs_coo<T>(m, n, nnz, index_base, vals, row_indx, col_indx);
  }
}

perflibs_sparse_matrix_shape_t get_shape_coo(perflibs_int_t m, perflibs_int_t n,
                                             perflibs_int_t nnz,
                                             const perflibs_int_t *row_indx,
                                             const perflibs_int_t *col_indx) {

  // We currently only care about shape for spsv, which requires a square
  // matrix, so get out early if the matrix is rectangular.
  if (m != n) {
    return PERFLIBS_SPARSE_SHAPE_RECTANGULAR;
  }

  perflibs_sparse_matrix_shape_t current =
      PERFLIBS_SPARSE_SHAPE_DIAGONAL; // not strictly upper or lower

  for (perflibs_int_t i = 0; i < nnz; i++) {
    const auto ri = row_indx[i];
    const auto ci = col_indx[i];
    if (ci > ri) { // we're in upper triangular territory
      if (current == PERFLIBS_SPARSE_SHAPE_LOWER_TRIANGULAR) {
        // if we've previously seen evidence of lower, get out: matrix is
        // rectangular
        return PERFLIBS_SPARSE_SHAPE_RECTANGULAR;
      } else {
        // otherwise, current assumption is it's upper triangular
        current = PERFLIBS_SPARSE_SHAPE_UPPER_TRIANGULAR;
      }
    } else if (ci < ri) { // reverse of above!
      if (current == PERFLIBS_SPARSE_SHAPE_UPPER_TRIANGULAR) {
        return PERFLIBS_SPARSE_SHAPE_RECTANGULAR;
      } else {
        current = PERFLIBS_SPARSE_SHAPE_LOWER_TRIANGULAR;
      }
    }
  }

  // TODO handle diagonal specifically
  // For now, we just handle as a lower triangular matrix for COO
  if (current == PERFLIBS_SPARSE_SHAPE_DIAGONAL) {
    return PERFLIBS_SPARSE_SHAPE_LOWER_TRIANGULAR;
  }

  return current;
}

template <typename T>
perflibs_sparse_matrix_diag_t
get_diag_coo(perflibs_int_t m, perflibs_int_t n, perflibs_int_t index_base,
             perflibs_int_t nnz, const perflibs_int_t *row_indx,
             const perflibs_int_t *col_indx, const T *vals) {

  // We currently only care about diagonals for spsv, which requires a square
  // matrix, so get out early if the matrix is rectangular.
  if (m != n) {
    return PERFLIBS_SPARSE_DIAG_NON_UNIT;
  }

  std::vector<char> diag_seen(m);

  perflibs_int_t unit_diag = 0;
  // Iterate over rows, returning early if we find a zero diagonal. If we find a
  // non-unit diagonal, we need to continue checking to see if there are any
  // zero diagonals in the remaining rows.
  for (perflibs_int_t i = 0; i < nnz; i++) {

    const auto ri = row_indx[i];
    const auto ci = col_indx[i];
    // If this is the diagonal ...
    if (ci == ri) {
      // count if unit diagonal
      if (vals[i] == T(1)) {
        unit_diag++;
      }
      // and set the flag to say that we've seen a non-zero diag on this row
      diag_seen[ci - index_base] = vals[i] != T(0);
    }
  }

  // Check that all diagonal elements have been seen
  perflibs_int_t diags = 0;
  for (auto d : diag_seen) {
    diags += d;
  }
  if (diags != n) {
    return PERFLIBS_SPARSE_DIAG_ZERO;
  }

  // A unit diagonal matrix has to have all diagonal entries equal to 1
  if (unit_diag == n) {
    return PERFLIBS_SPARSE_DIAG_UNIT;
  }

  return PERFLIBS_SPARSE_DIAG_NON_UNIT;
}

/**
 * *          `nnz` - Number of non-zero values
 * * `index_base` - 1 for fortran, 0 for C (since fortran arrays are 1 indexed)
 */
template <typename T>
perflibs_status_t
fill_initial_data_coo(perflibs_spmat_top_t *A, perflibs_int_t m,
                      perflibs_int_t n, perflibs_int_t nnz,
                      perflibs_int_t index_base, const perflibs_int_t *row_indx,
                      const perflibs_int_t *col_indx, const T *vals,
                      bool no_copy) {

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

  impl->spmat_format = perflibs_format_coo;
  impl->no_copy = no_copy;

  impl->m = m;
  impl->n = n;
  impl->index_base = index_base;
  impl->nnz = nnz;

  impl->shape = get_shape_coo(m, n, nnz, row_indx, col_indx);
  impl->diag = get_diag_coo(m, n, index_base, nnz, row_indx, col_indx, vals);

  impl->coo = make_coo<T>(impl->m, impl->n, impl->nnz, impl->index_base, vals,
                          row_indx, col_indx, no_copy);

  return PERFLIBS_STATUS_SUCCESS;
}

template perflibs_status_t fill_initial_data_coo<float>(
    perflibs_spmat_top_t *A, perflibs_int_t m, perflibs_int_t n,
    perflibs_int_t nnz, perflibs_int_t index_base,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_indx,
    const float *vals, bool no_copy);
template perflibs_status_t fill_initial_data_coo<double>(
    perflibs_spmat_top_t *A, perflibs_int_t m, perflibs_int_t n,
    perflibs_int_t nnz, perflibs_int_t index_base,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_indx,
    const double *vals, bool no_copy);
template perflibs_status_t fill_initial_data_coo<std::complex<float>>(
    perflibs_spmat_top_t *A, perflibs_int_t m, perflibs_int_t n,
    perflibs_int_t nnz, perflibs_int_t index_base,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_indx,
    const std::complex<float> *vals, bool no_copy);
template perflibs_status_t fill_initial_data_coo<std::complex<double>>(
    perflibs_spmat_top_t *A, perflibs_int_t m, perflibs_int_t n,
    perflibs_int_t nnz, perflibs_int_t index_base,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_indx,
    const std::complex<double> *vals, bool no_copy);

template <typename T>
perflibs_coo<T> &perflibs_coo<T>::operator=(const perflibs_coo &other) {
  if (&other == this) {
    return *this;
  }
  // Copy the vector variables
  nnz = other.nnz;
  m = other.m;
  n = other.n;
  index_base = other.index_base;
  nthreads = other.nthreads;
  if (m >= 0 && n >= 0 && nnz >= 0) {
    // If the vectors are populated make the const pointers point to them
    // otherwise construct a new vector
    copy_from_vector_or_ptr(&vals_ptr, vals, other.vals_ptr, other.vals, nnz);

    copy_from_vector_or_ptr(&col_indx_ptr, col_indx, other.col_indx_ptr,
                            other.col_indx, nnz);

    copy_from_vector_or_ptr(&row_indx_ptr, row_indx, other.row_indx_ptr,
                            other.row_indx, nnz);
  }

  return *this;
}
template perflibs_coo<float> &
perflibs_coo<float>::operator=(const perflibs_coo &other);
template perflibs_coo<double> &
perflibs_coo<double>::operator=(const perflibs_coo &other);
template perflibs_coo<std::complex<float>> &
perflibs_coo<std::complex<float>>::operator=(const perflibs_coo &other);
template perflibs_coo<std::complex<double>> &
perflibs_coo<std::complex<double>>::operator=(const perflibs_coo &other);

// Scale the input values and write them into the current matrix object
template <typename T>
void perflibs_coo<T>::scale_matrix(enum perflibs_sparse_hint_value trans,
                                   T alpha) {
  // Allocate our copy vector (in case we're copying from user's data)
  if (vals.data() != vals_ptr) {
    vals.resize(nnz);
  }

  // Get conjugate of val if required
  auto conj = [&](T val, perflibs_sparse_hint_value trans) {
    return trans == PERFLIBS_SPARSE_OPERATION_CONJTRANS
               ? perflibs::sparse::conj(val)
               : val;
  };

  // Do conjugation and scaling
  for (perflibs_int_t i = 0; i < nnz; i++) {
    vals[i] = conj(vals_ptr[i], trans) * alpha;
  }

  // Set the new pointer
  vals_ptr = vals.data();

  // Swap the row and column index vectors if transposing
  if (trans != PERFLIBS_SPARSE_OPERATION_NOTRANS) {
    row_indx.swap(col_indx);
    row_indx_ptr = row_indx.data();
    col_indx_ptr = col_indx.data();
    std::swap(m, n);
  }
}
template void
perflibs_coo<float>::scale_matrix(perflibs_sparse_hint_value trans,
                                  float alpha);
template void
perflibs_coo<double>::scale_matrix(perflibs_sparse_hint_value trans,
                                   double alpha);
template void perflibs_coo<std::complex<float>>::scale_matrix(
    perflibs_sparse_hint_value trans, std::complex<float> alpha);
template void perflibs_coo<std::complex<double>>::scale_matrix(
    perflibs_sparse_hint_value trans, std::complex<double> alpha);

template <typename T>
perflibs_coo<T> csr2coo(perflibs_int_t m, perflibs_int_t n, const T *vals,
                        const perflibs_int_t *col_indx,
                        const perflibs_int_t *row_ptr) {
  auto base = row_ptr[0];
  auto nnz = row_ptr[m] - base;
  std::vector<T> vals_coo(vals, vals + nnz);
  std::vector<perflibs_int_t> col_indx_coo(col_indx, col_indx + nnz);
  std::vector<perflibs_int_t> row_indx_coo(nnz);

  for (perflibs_int_t i = 0; i < m; i++) {
    for (perflibs_int_t j = row_ptr[i] - base; j < row_ptr[i + 1] - base; j++) {
      row_indx_coo[j] = i + base;
    }
  }

  return perflibs_coo<T>(m, n, nnz, base, vals_coo.data(), row_indx_coo.data(),
                         col_indx_coo.data());
}
template perflibs_coo<float> csr2coo(perflibs_int_t m, perflibs_int_t n,
                                     const float *vals,
                                     const perflibs_int_t *col_indx,
                                     const perflibs_int_t *row_ptr);
template perflibs_coo<double> csr2coo(perflibs_int_t m, perflibs_int_t n,
                                      const double *vals,
                                      const perflibs_int_t *col_indx,
                                      const perflibs_int_t *row_ptr);
template perflibs_coo<std::complex<float>>
csr2coo(perflibs_int_t m, perflibs_int_t n, const std::complex<float> *vals,
        const perflibs_int_t *col_indx, const perflibs_int_t *row_ptr);
template perflibs_coo<std::complex<double>>
csr2coo(perflibs_int_t m, perflibs_int_t n, const std::complex<double> *vals,
        const perflibs_int_t *col_indx, const perflibs_int_t *row_ptr);

template <typename T> inline T no_conj(T value) { return value; }

template <typename T> inline T conj(T value) {
  return perflibs::sparse::conj(value);
}

// OpenMP atomic does not work properly on Windows with clang.
// Use OpenMP critical instead until the upstream LLVM/OpenMP issue is resolved:
// https://github.com/llvm/llvm-project/issues/64694
inline void update_y(float &y, float value) {
#if defined(_WIN32)
#pragma omp critical(coo_update_y_s)
  {
    y += value;
  }
#else
#pragma omp atomic
  y += value;
#endif
}

inline void update_y(double &y, double value) {
#if defined(_WIN32)
#pragma omp critical(coo_update_y_d)
  {
    y += value;
  }
#else
#pragma omp atomic
  y += value;
#endif
}

inline void update_y(std::complex<float> &y, std::complex<float> value) {
  float *cplx_y = reinterpret_cast<float *>(&y);
#if defined(_WIN32)
#pragma omp critical(coo_update_y_c)
  {
    cplx_y[0] += value.real();
    cplx_y[1] += value.imag();
  }
#else
#pragma omp atomic
  cplx_y[0] += value.real();
#pragma omp atomic
  cplx_y[1] += value.imag();
#endif
}

inline void update_y(std::complex<double> &y, std::complex<double> value) {
  double *cplx_y = reinterpret_cast<double *>(&y);
#if defined(_WIN32)
#pragma omp critical(coo_update_y_z)
  {
    cplx_y[0] += value.real();
    cplx_y[1] += value.imag();
  }
#else
#pragma omp atomic
  cplx_y[0] += value.real();
#pragma omp atomic
  cplx_y[1] += value.imag();
#endif
}

/*
   Via the magic of templates we will get 2 copies for each of the
   spmv_coo_*trans functions, one in which conj is perflibs::sparse::conj and
   one in which conj is no_conj - where the compiler ought to recognize that it
   is a noop.
     */

#define NUM_COO_UNROLL 4

template <typename T, decltype(&no_conj<T>) conj>
void spmv_coo_notrans(const perflibs_int_t *const row_indx,
                      const perflibs_int_t *const col_indx,
                      const T *const vals_ptr, const perflibs_int_t nnz,
                      const T *const x, T *const y, const T alpha, const T beta,
                      perflibs_int_t index_base) {
  int64_t ii = 0;
  for (; ii < (nnz / NUM_COO_UNROLL) * NUM_COO_UNROLL; ii += NUM_COO_UNROLL) {
    update_y(y[row_indx[ii + 0] - index_base],
             alpha * x[col_indx[ii + 0] - index_base] * conj(vals_ptr[ii + 0]));
    update_y(y[row_indx[ii + 1] - index_base],
             alpha * x[col_indx[ii + 1] - index_base] * conj(vals_ptr[ii + 1]));
    update_y(y[row_indx[ii + 2] - index_base],
             alpha * x[col_indx[ii + 2] - index_base] * conj(vals_ptr[ii + 2]));
    update_y(y[row_indx[ii + 3] - index_base],
             alpha * x[col_indx[ii + 3] - index_base] * conj(vals_ptr[ii + 3]));
  }
  for (; ii < nnz; ++ii) {
    update_y(y[row_indx[ii] - index_base],
             alpha * x[col_indx[ii] - index_base] * conj(vals_ptr[ii]));
  }
}

template <typename T, decltype(&no_conj<T>) conj>
void spmv_coo_trans(const perflibs_int_t *const row_indx,
                    const perflibs_int_t *const col_indx,
                    const T *const vals_ptr, const perflibs_int_t nnz,
                    const T *const x, T *const y, const T alpha, const T beta,
                    perflibs_int_t index_base) {
  int64_t ii = 0;
  for (; ii < (nnz / NUM_COO_UNROLL) * NUM_COO_UNROLL; ii += NUM_COO_UNROLL) {
    update_y(y[col_indx[ii + 0] - index_base],
             alpha * x[row_indx[ii + 0] - index_base] * conj(vals_ptr[ii + 0]));
    update_y(y[col_indx[ii + 1] - index_base],
             alpha * x[row_indx[ii + 1] - index_base] * conj(vals_ptr[ii + 1]));
    update_y(y[col_indx[ii + 2] - index_base],
             alpha * x[row_indx[ii + 2] - index_base] * conj(vals_ptr[ii + 2]));
    update_y(y[col_indx[ii + 3] - index_base],
             alpha * x[row_indx[ii + 3] - index_base] * conj(vals_ptr[ii + 3]));
  }
  for (; ii < nnz; ++ii) {
    update_y(y[col_indx[ii] - index_base],
             alpha * x[row_indx[ii] - index_base] * conj(vals_ptr[ii]));
  }
}

template <typename T>
void spmv_coo(perflibs_coo<T> &coo, const sparse_hint_value_internal trans,
              const T *const x, T *const y, const T alpha, const T beta,
              perflibs_int_t index_base) {

  // Pre-multiply y by beta
  auto premul_limit = (trans == PERFLIBS_OPERATION_TRANS ||
                       trans == PERFLIBS_OPERATION_CONJTRANS)
                          ? coo.n
                          : coo.m;
  if (std::real(beta) == 0 && std::imag(beta) == 0) {
    for (perflibs_int_t i = 0; i < premul_limit; i++) {
      y[i] = 0;
    }
  } else {
    for (perflibs_int_t i = 0; i < premul_limit; i++) {
      y[i] *= beta;
    }
  }

  // Find kernel to run based on trans, big ternary so we can use auto
  const auto kernel =
      trans == PERFLIBS_OPERATION_NOTRANS ? &spmv_coo_notrans<T, no_conj<T>>
      : trans == PERFLIBS_OPERATION_CONJNOTRANS ? &spmv_coo_notrans<T, conj<T>>
      : trans == PERFLIBS_OPERATION_TRANS       ? &spmv_coo_trans<T, no_conj<T>>
      : trans == PERFLIBS_OPERATION_CONJTRANS   ? &spmv_coo_trans<T, conj<T>>
                                                : nullptr;
  assert(kernel != nullptr && "Bad value of trans, no kernel found.");

  if (perflibs_sparse_get_reproducibility() ==
      PERFLIBS_SPARSE_REPRODUCIBILITY_ENABLED) {
    kernel(coo.row_indx_ptr, coo.col_indx_ptr, coo.vals_ptr, coo.nnz, x, y,
           alpha, beta, index_base);
    return;
  }

#if defined(_OPENMP)

  // Find nthreads
  int64_t nthreads = coo.nthreads;
  const int64_t inc = iround_div(coo.nnz, nthreads);

// Run the kernel
#pragma omp parallel for default(none)                                         \
    firstprivate(kernel, inc, x, y, alpha, beta, index_base) shared(coo)       \
    num_threads(nthreads)
  for (perflibs_int_t start = 0; start < coo.nnz; start += inc) {
    perflibs_int_t work = std::min(coo.nnz - start, inc);
    kernel(coo.row_indx_ptr + start, coo.col_indx_ptr + start,
           coo.vals_ptr + start, work, x, y, alpha, beta, index_base);
  }
#else
  kernel(coo.row_indx_ptr, coo.col_indx_ptr, coo.vals_ptr, coo.nnz, x, y, alpha,
         beta, index_base);
#endif
}

template void spmv_coo<float>(perflibs_coo<float> &coo,
                              const sparse_hint_value_internal trans,
                              const float *const x, float *const y,
                              const float alpha, const float beta,
                              perflibs_int_t index_base);
template void spmv_coo<double>(perflibs_coo<double> &coo,
                               const sparse_hint_value_internal trans,
                               const double *const x, double *const y,
                               const double alpha, const double beta,
                               perflibs_int_t index_base);
template void spmv_coo<std::complex<float>>(
    perflibs_coo<std::complex<float>> &coo,
    const sparse_hint_value_internal trans, const std::complex<float> *const x,
    std::complex<float> *const y, const std::complex<float> alpha,
    const std::complex<float> beta, perflibs_int_t index_base);
template void spmv_coo<std::complex<double>>(
    perflibs_coo<std::complex<double>> &coo,
    const sparse_hint_value_internal trans, const std::complex<double> *const x,
    std::complex<double> *const y, const std::complex<double> alpha,
    const std::complex<double> beta, perflibs_int_t index_base);

template <typename T>
void spnorm_inf_coo(const perflibs_coo<T> &coo,
                    perflibs::sparse::remove_complex_t<T> *result) {
  using RT = perflibs::sparse::remove_complex_t<T>;

  std::vector<RT> rowsums(coo.m);
  auto index_base = coo.index_base;
  for (auto i = 0; i < coo.nnz; i++) {
    rowsums[coo.row_indx_ptr[i] - index_base] += std::abs(coo.vals_ptr[i]);
  }

  *result = spnorm_max(rowsums);
}
template void spnorm_inf_coo<float>(const perflibs_coo<float> &coo,
                                    float *result);
template void spnorm_inf_coo<double>(const perflibs_coo<double> &coo,
                                     double *result);
template void spnorm_inf_coo<std::complex<float>>(
    const perflibs_coo<std::complex<float>> &coo, float *result);
template void spnorm_inf_coo<std::complex<double>>(
    const perflibs_coo<std::complex<double>> &coo, double *result);

template <typename T>
perflibs_status_t spelmm_coo(perflibs_sparse_hint_value transA,
                             const perflibs_coo<T> &A,
                             perflibs_sparse_hint_value transB,
                             const perflibs_coo<T> &B, perflibs_spmat_t AB) {

  // Set up vectors to store the resulting sparse matrix (AB) in COO format
  auto min_nnz = std::min(A.nnz, B.nnz);
  perflibs::sparse::pod_vector<perflibs_int_t> row_indxAB;
  row_indxAB.reserve(min_nnz);

  perflibs::sparse::pod_vector<perflibs_int_t> col_indxAB;
  col_indxAB.reserve(min_nnz);

  perflibs::sparse::pod_vector<T> valsAB;
  valsAB.reserve(min_nnz);

  // Get conjugate of val if required
  auto conj = [&](T val, perflibs_sparse_hint_value trans) {
    return trans == PERFLIBS_SPARSE_OPERATION_CONJTRANS
               ? perflibs::sparse::conj(val)
               : val;
  };

  // Determine index arrays based on whether matrices are transposed
  auto row_indxA = transA == PERFLIBS_SPARSE_OPERATION_NOTRANS ? A.row_indx_ptr
                                                               : A.col_indx_ptr;
  auto col_indxA = transA == PERFLIBS_SPARSE_OPERATION_NOTRANS ? A.col_indx_ptr
                                                               : A.row_indx_ptr;
  auto row_indxB = transB == PERFLIBS_SPARSE_OPERATION_NOTRANS ? B.row_indx_ptr
                                                               : B.col_indx_ptr;
  auto col_indxB = transB == PERFLIBS_SPARSE_OPERATION_NOTRANS ? B.col_indx_ptr
                                                               : B.row_indx_ptr;

  // Counter for non-zero elements in the output matrix AB
  perflibs_int_t nnzAB = 0;

  // Iterate through all non-zero elements of A and B
  for (auto i = 0; i < A.nnz; i++) {
    for (auto j = 0; j < B.nnz; j++) {
      // Check if A and B have a non-zero element at the same position
      if ((row_indxA[i] == row_indxB[j]) && (col_indxA[i] == col_indxB[j])) {
        // Compute element-wise product
        valsAB.push_back(conj(A.vals_ptr[i], transA) *
                         conj(B.vals_ptr[j], transB));

        // Update row and column indices for AB
        row_indxAB.push_back(row_indxA[i]);
        col_indxAB.push_back(col_indxA[i]);

        nnzAB++;
      }
    }
  }

  // Create COO matrix from the result AB
  auto m = transA == PERFLIBS_SPARSE_OPERATION_NOTRANS ? A.m : A.n;
  auto n = transA == PERFLIBS_SPARSE_OPERATION_NOTRANS ? A.n : A.m;
  auto ret =
      fill_initial_data_coo(AB, m, n, nnzAB, A.index_base, row_indxAB.data(),
                            col_indxAB.data(), valsAB.data(), false);
  if (ret != PERFLIBS_STATUS_SUCCESS) {
    return ret;
  }

  return PERFLIBS_STATUS_SUCCESS;
}
template perflibs_status_t spelmm_coo<float>(perflibs_sparse_hint_value transA,
                                             const perflibs_coo<float> &A,
                                             perflibs_sparse_hint_value transB,
                                             const perflibs_coo<float> &B,
                                             perflibs_spmat_t AB);
template perflibs_status_t spelmm_coo<double>(perflibs_sparse_hint_value transA,
                                              const perflibs_coo<double> &A,
                                              perflibs_sparse_hint_value transB,
                                              const perflibs_coo<double> &B,
                                              perflibs_spmat_t AB);
template perflibs_status_t
spelmm_coo<std::complex<float>>(perflibs_sparse_hint_value transA,
                                const perflibs_coo<std::complex<float>> &A,
                                perflibs_sparse_hint_value transB,
                                const perflibs_coo<std::complex<float>> &B,
                                perflibs_spmat_t AB);

template perflibs_status_t
spelmm_coo<std::complex<double>>(perflibs_sparse_hint_value transA,
                                 const perflibs_coo<std::complex<double>> &A,
                                 perflibs_sparse_hint_value transB,
                                 const perflibs_coo<std::complex<double>> &B,
                                 perflibs_spmat_t AB);

template <typename T>
perflibs_status_t
spmat_update_coo(perflibs_spmat_impl_t<T> *impl, perflibs_int_t n_updates,
                 const perflibs_int_t *row_indx, const perflibs_int_t *col_indx,
                 const T *vals) {
  auto nrows = impl->m;
  auto ncols = impl->n;
  auto index_base = impl->index_base;

  auto col_indx_orig = impl->coo.col_indx_ptr;
  auto row_indx_orig = impl->coo.row_indx_ptr;
  auto vals_orig = const_cast<T *>(impl->coo.vals_ptr);

  auto is_unit = impl->diag == PERFLIBS_SPARSE_DIAG_UNIT;

  for (auto ii = 0; ii < n_updates; ii++) {
    auto row_num = row_indx[ii] - index_base;
    auto col_num = col_indx[ii] - index_base;
    if (row_num < 0 || row_num > nrows - 1) {
      return update_matrix_error(impl->error_handle, ii + index_base);
    } else if (col_num < 0 || col_num > ncols - 1) {
      return update_matrix_error(impl->error_handle, ii + index_base);
    } else {
      bool success = false;
      for (auto jj = 0; jj < impl->nnz; ++jj) {
        if (row_indx_orig[jj] == row_indx[ii] &&
            col_indx_orig[jj] == col_indx[ii]) {
          vals_orig[jj] = vals[ii];
          success = true;
          if (row_indx[ii] == col_indx[ii] && is_unit && vals[ii] != T(1)) {
            impl->diag = PERFLIBS_SPARSE_DIAG_NON_UNIT;
          }
          break;
        }
      }
      if (!success) {
        return update_matrix_error(impl->error_handle, ii + index_base);
      }
    }
  }
  return PERFLIBS_STATUS_SUCCESS;
}
template perflibs_status_t
spmat_update_coo<float>(perflibs_spmat_impl_t<float> *impl,
                        perflibs_int_t n_updates,
                        const perflibs_int_t *row_indx,
                        const perflibs_int_t *col_indx, const float *vals);
template perflibs_status_t
spmat_update_coo<double>(perflibs_spmat_impl_t<double> *impl,
                         perflibs_int_t n_updates,
                         const perflibs_int_t *row_indx,
                         const perflibs_int_t *col_indx, const double *vals);
template perflibs_status_t spmat_update_coo<std::complex<float>>(
    perflibs_spmat_impl_t<std::complex<float>> *impl, perflibs_int_t n_updates,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_indx,
    const std::complex<float> *vals);
template perflibs_status_t spmat_update_coo<std::complex<double>>(
    perflibs_spmat_impl_t<std::complex<double>> *impl, perflibs_int_t n_updates,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_indx,
    const std::complex<double> *vals);

} // namespace perflibs::sparse
