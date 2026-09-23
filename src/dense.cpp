/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "dense.hpp"
#include "cblas_wrappers.hpp"
#include "norm.hpp"
#include "object_helpers.hpp"
#include "pod_vector.hpp"
#include "types.hpp"

namespace perflibs::sparse {

template <typename T>
perflibs_dense<T> make_dense(perflibs_dense_layout layout, perflibs_int_t m,
                             perflibs_int_t n, perflibs_int_t lda,
                             const T *vals, bool no_copy) {

  if (no_copy) {
    return perflibs_dense<T>(layout, m, n, lda, vals);
  } else {
    return perflibs_dense<T>(layout, m, n, lda, vals, (int64_t)m * n);
  }
}

template <typename T>
perflibs_sparse_matrix_shape_t
get_shape_dense(perflibs_dense_layout layout, perflibs_int_t m,
                perflibs_int_t n, const T *vals, perflibs_int_t lda) {

  // We currently only care about shape for spsv, which requires a square
  // matrix, so get out early if the matrix is rectangular.
  if (m != n) {
    return PERFLIBS_SPARSE_SHAPE_RECTANGULAR;
  }

  perflibs_sparse_matrix_shape_t current =
      PERFLIBS_SPARSE_SHAPE_DIAGONAL; // not strictly upper or lower

  auto get_val = [&](perflibs_int_t i, perflibs_int_t j) {
    return layout == PERFLIBS_COL_MAJOR ? vals[lda * j + i] : vals[lda * i + j];
  };

  for (perflibs_int_t i = 0; i < m; i++) {
    for (perflibs_int_t j = 0; j < n; j++) {
      const auto val = get_val(i, j);
      if (val != T(0)) {
        if (j > i) { // we're in upper triangular territory
          if (current == PERFLIBS_SPARSE_SHAPE_LOWER_TRIANGULAR) {
            // if we've previously seen evidence of lower, get out: matrix is
            // rectangular
            return PERFLIBS_SPARSE_SHAPE_RECTANGULAR;
          } else {
            // otherwise, current assumption is it's upper triangular
            current = PERFLIBS_SPARSE_SHAPE_UPPER_TRIANGULAR;
          }
        } else if (j < i) {
          if (current == PERFLIBS_SPARSE_SHAPE_UPPER_TRIANGULAR) {
            return PERFLIBS_SPARSE_SHAPE_RECTANGULAR;
          } else {
            current = PERFLIBS_SPARSE_SHAPE_LOWER_TRIANGULAR;
          }
        }
      }
    }
  }

  return current;
}
// We call get_shape_dense from BSR, so it needs to be externally callable
template perflibs_sparse_matrix_shape_t
get_shape_dense(perflibs_dense_layout layout, perflibs_int_t m,
                perflibs_int_t n, const float *vals, perflibs_int_t lda);
template perflibs_sparse_matrix_shape_t
get_shape_dense(perflibs_dense_layout layout, perflibs_int_t m,
                perflibs_int_t n, const double *vals, perflibs_int_t lda);
template perflibs_sparse_matrix_shape_t
get_shape_dense(perflibs_dense_layout layout, perflibs_int_t m,
                perflibs_int_t n, const std::complex<float> *vals,
                perflibs_int_t lda);
template perflibs_sparse_matrix_shape_t
get_shape_dense(perflibs_dense_layout layout, perflibs_int_t m,
                perflibs_int_t n, const std::complex<double> *vals,
                perflibs_int_t lda);

template <typename T>
perflibs_sparse_matrix_diag_t
get_diag_dense(perflibs_int_t m, perflibs_int_t n, const T *vals,
               perflibs_int_t lda, perflibs_sparse_matrix_shape_t shape) {
  // We currently only care about diagonals for spsv, which requires a square
  // matrix, so get out early if the matrix is rectangular.
  if (m != n || shape == PERFLIBS_SPARSE_SHAPE_RECTANGULAR) {
    return PERFLIBS_SPARSE_DIAG_NON_UNIT;
  }

  for (perflibs_int_t i = 0; i < m; i++) {
    if (vals[lda * i + i] == T(0)) {
      return PERFLIBS_SPARSE_DIAG_ZERO;
    } else if (vals[lda * i + i] != T(1)) {
      return PERFLIBS_SPARSE_DIAG_NON_UNIT;
    }
  }

  return PERFLIBS_SPARSE_DIAG_UNIT;
}

template <typename T>
perflibs_status_t
fill_initial_data_dense(perflibs_spmat_t A, perflibs_dense_layout layout,
                        perflibs_int_t m, perflibs_int_t n, perflibs_int_t lda,
                        perflibs_int_t index_base, const T *vals,
                        bool no_copy) {

  auto impl = reinterpret_cast<perflibs_spmat_impl_t<T> *>(A->impl);

  if (!(layout == PERFLIBS_COL_MAJOR || layout == PERFLIBS_ROW_MAJOR)) {
    impl->error_handle.perflibs_error_type =
        PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    impl->error_handle.perflibs_error_code = 2;
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }
  if (m < 0) {
    impl->error_handle.perflibs_error_type =
        PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    impl->error_handle.perflibs_error_code = 3;
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }
  if (n < 0) {
    impl->error_handle.perflibs_error_type =
        PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    impl->error_handle.perflibs_error_code = 4;
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }
  if ((layout == PERFLIBS_COL_MAJOR && lda < m) ||
      (layout == PERFLIBS_ROW_MAJOR && lda < n)) {
    impl->error_handle.perflibs_error_type =
        PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    impl->error_handle.perflibs_error_code = 5;
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }

  impl->spmat_format = perflibs_format_dense;
  impl->no_copy = no_copy;

  impl->m = m;
  impl->n = n;
  impl->index_base = index_base;
  impl->nnz = (int64_t)m * n;

  impl->shape = get_shape_dense(layout, m, n, vals, lda);
  impl->diag = get_diag_dense(m, n, vals, lda, impl->shape);

  impl->dense = make_dense<T>(layout, impl->m, impl->n, lda, vals, no_copy);

  return PERFLIBS_STATUS_SUCCESS;
}

template perflibs_status_t
fill_initial_data_dense<float>(perflibs_spmat_t A, perflibs_dense_layout layout,
                               perflibs_int_t m, perflibs_int_t n,
                               perflibs_int_t lda, perflibs_int_t index_base,
                               const float *vals, bool no_copy);
template perflibs_status_t fill_initial_data_dense<double>(
    perflibs_spmat_t A, perflibs_dense_layout layout, perflibs_int_t m,
    perflibs_int_t n, perflibs_int_t lda, perflibs_int_t index_base,
    const double *vals, bool no_copy);
template perflibs_status_t fill_initial_data_dense<std::complex<float>>(
    perflibs_spmat_t A, perflibs_dense_layout layout, perflibs_int_t m,
    perflibs_int_t n, perflibs_int_t lda, perflibs_int_t index_base,
    const std::complex<float> *vals, bool no_copy);
template perflibs_status_t fill_initial_data_dense<std::complex<double>>(
    perflibs_spmat_t A, perflibs_dense_layout layout, perflibs_int_t m,
    perflibs_int_t n, perflibs_int_t lda, perflibs_int_t index_base,
    const std::complex<double> *vals, bool no_copy);

template <typename T>
perflibs_dense<T> &perflibs_dense<T>::operator=(const perflibs_dense &other) {
  if (&other == this) {
    return *this;
  }
  // Copy the vector variables
  layout = other.layout;
  m = other.m;
  n = other.n;
  lda = other.lda;
  if (m >= 0 && n >= 0) {
    auto nnz = layout == PERFLIBS_COL_MAJOR ? lda * n : m * lda;

    // If the vector is populated make the const pointer point to it
    // otherwise construct a new vector
    copy_from_vector_or_ptr(&vals_ptr, vals, other.vals_ptr, other.vals, nnz);
  }

  return *this;
}
template perflibs_dense<float> &
perflibs_dense<float>::operator=(const perflibs_dense &other);
template perflibs_dense<double> &
perflibs_dense<double>::operator=(const perflibs_dense &other);
template perflibs_dense<std::complex<float>> &
perflibs_dense<std::complex<float>>::operator=(const perflibs_dense &other);
template perflibs_dense<std::complex<double>> &
perflibs_dense<std::complex<double>>::operator=(const perflibs_dense &other);

// Scale the input values and write them into the current matrix object
template <typename T>
void perflibs_dense<T>::scale_matrix(enum perflibs_sparse_hint_value trans,
                                     T alpha) {

  auto nnz = layout == PERFLIBS_COL_MAJOR ? lda * n : m * lda;

  // Allocate our copy vector (in case we're copying from user's data)
  vals.resize(nnz);

  // Now scale the matrix, reading from the old pointer
  if (layout == PERFLIBS_COL_MAJOR) {
    if (trans == PERFLIBS_SPARSE_OPERATION_CONJTRANS) {
#pragma omp parallel for
      for (perflibs_int_t i = 0; i < n; i++) {
        for (perflibs_int_t j = 0; j < m; j++) {
          vals[i * lda + j] =
              perflibs::sparse::conj(vals_ptr[i * lda + j]) * alpha;
        }
      }
    } else {
#pragma omp parallel for
      for (perflibs_int_t i = 0; i < n; i++) {
        for (perflibs_int_t j = 0; j < m; j++) {
          vals[i * lda + j] = vals_ptr[i * lda + j] * alpha;
        }
      }
    }
  } else {
    if (trans == PERFLIBS_SPARSE_OPERATION_CONJTRANS) {
#pragma omp parallel for
      for (perflibs_int_t i = 0; i < m; i++) {
        for (perflibs_int_t j = 0; j < n; j++) {
          vals[i * lda + j] =
              perflibs::sparse::conj(vals_ptr[i * lda + j]) * alpha;
        }
      }
    } else {
#pragma omp parallel for
      for (perflibs_int_t i = 0; i < m; i++) {
        for (perflibs_int_t j = 0; j < n; j++) {
          vals[i * lda + j] = vals_ptr[i * lda + j] * alpha;
        }
      }
    }
  }

  // Set the new pointer
  vals_ptr = vals.data();
}
template void
perflibs_dense<float>::scale_matrix(enum perflibs_sparse_hint_value trans,
                                    float alpha);
template void
perflibs_dense<double>::scale_matrix(enum perflibs_sparse_hint_value trans,
                                     double alpha);
template void perflibs_dense<std::complex<float>>::scale_matrix(
    enum perflibs_sparse_hint_value trans, std::complex<float> alpha);
template void perflibs_dense<std::complex<double>>::scale_matrix(
    enum perflibs_sparse_hint_value trans, std::complex<double> alpha);

// Construct a dense version of the CSR input, copy in the non-zero
// values in column-major layout (originally copied from test_spmv_csr.cpp; now
// called from within there)
template <typename T>
std::vector<T> csr2dense(perflibs_int_t m, perflibs_int_t n, const T *vals,
                         const perflibs_int_t *col_indx,
                         const perflibs_int_t *row_ptr) {
  auto index_base = row_ptr[0];
  [[maybe_unused]] auto nnz = row_ptr[m] - index_base;

  std::vector<T> A(int64_t(m) * n);

  perflibs_int_t k = 0;
  for (auto i = 0; i < m; i++) {
    for (auto j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
      auto col = col_indx[k] - index_base;
      A[col * m + i] = vals[k++];
    }
  }
  assert(k == nnz);

  return A;
}
template std::vector<float> csr2dense(perflibs_int_t m, perflibs_int_t n,
                                      const float *vals,
                                      const perflibs_int_t *col_indx,
                                      const perflibs_int_t *row_ptr);
template std::vector<double> csr2dense(perflibs_int_t m, perflibs_int_t n,
                                       const double *vals,
                                       const perflibs_int_t *col_indx,
                                       const perflibs_int_t *row_ptr);
template std::vector<std::complex<float>>
csr2dense(perflibs_int_t m, perflibs_int_t n, const std::complex<float> *vals,
          const perflibs_int_t *col_indx, const perflibs_int_t *row_ptr);
template std::vector<std::complex<double>>
csr2dense(perflibs_int_t m, perflibs_int_t n, const std::complex<double> *vals,
          const perflibs_int_t *col_indx, const perflibs_int_t *row_ptr);

template <typename T>
std::vector<T> special2dense(perflibs_int_t m, perflibs_int_t n,
                             spmat_format_t type) {
  std::vector<T> A((size_t)m * n);
  if (type == perflibs_format_identity) {
    for (perflibs_int_t i = 0; i < std::min(m, n); i++) {
      A[m * i + i] = (T)1;
    }
  }
  return A;
}
template std::vector<float> special2dense(perflibs_int_t m, perflibs_int_t n,
                                          spmat_format_t type);
template std::vector<double> special2dense(perflibs_int_t m, perflibs_int_t n,
                                           spmat_format_t type);
template std::vector<std::complex<float>>
special2dense(perflibs_int_t m, perflibs_int_t n, spmat_format_t type);
template std::vector<std::complex<double>>
special2dense(perflibs_int_t m, perflibs_int_t n, spmat_format_t type);

template <typename T>
perflibs_status_t spmv_gemv(perflibs_dense_layout layout,
                            perflibs_sparse_hint_value trans, perflibs_int_t m,
                            perflibs_int_t n, const T *A, perflibs_int_t lda,
                            T alpha, const T *x, T beta, T *y) {
  auto ctrans = trans == PERFLIBS_SPARSE_OPERATION_NOTRANS ? CblasNoTrans
                : trans == PERFLIBS_SPARSE_OPERATION_TRANS ? CblasTrans
                                                           : CblasConjTrans;
  CBLAS_LAYOUT clayout =
      layout == PERFLIBS_COL_MAJOR ? CblasColMajor : CblasRowMajor;
  perflibs::sparse::cblas_gemv<T>(clayout, ctrans, m, n, alpha, A, lda, x, 1,
                                  beta, y, 1);
  return PERFLIBS_STATUS_SUCCESS;
}
template perflibs_status_t spmv_gemv(perflibs_dense_layout layout,
                                     perflibs_sparse_hint_value trans,
                                     perflibs_int_t m, perflibs_int_t n,
                                     const float *A, perflibs_int_t lda,
                                     float alpha, const float *x, float beta,
                                     float *y);
template perflibs_status_t spmv_gemv(perflibs_dense_layout layout,
                                     perflibs_sparse_hint_value trans,
                                     perflibs_int_t m, perflibs_int_t n,
                                     const double *A, perflibs_int_t lda,
                                     double alpha, const double *x, double beta,
                                     double *y);
template perflibs_status_t
spmv_gemv(perflibs_dense_layout layout, perflibs_sparse_hint_value trans,
          perflibs_int_t m, perflibs_int_t n, const std::complex<float> *A,
          perflibs_int_t lda, std::complex<float> alpha,
          const std::complex<float> *x, std::complex<float> beta,
          std::complex<float> *y);
template perflibs_status_t
spmv_gemv(perflibs_dense_layout layout, perflibs_sparse_hint_value trans,
          perflibs_int_t m, perflibs_int_t n, const std::complex<double> *A,
          perflibs_int_t lda, std::complex<double> alpha,
          const std::complex<double> *x, std::complex<double> beta,
          std::complex<double> *y);

template <typename T>
perflibs_status_t
spmm_gemm(perflibs_dense_layout layout, perflibs_sparse_hint_value transA,
          perflibs_sparse_hint_value transB, perflibs_int_t m, perflibs_int_t n,
          perflibs_int_t k, T alpha, const T *A, perflibs_int_t lda, const T *B,
          perflibs_int_t ldb, T beta, T *C, perflibs_int_t ldc) {
  auto ctransA = transA == PERFLIBS_SPARSE_OPERATION_NOTRANS ? CblasNoTrans
                 : transA == PERFLIBS_SPARSE_OPERATION_TRANS ? CblasTrans
                                                             : CblasConjTrans;
  auto ctransB = transB == PERFLIBS_SPARSE_OPERATION_NOTRANS ? CblasNoTrans
                 : transB == PERFLIBS_SPARSE_OPERATION_TRANS ? CblasTrans
                                                             : CblasConjTrans;
  CBLAS_LAYOUT clayout =
      layout == PERFLIBS_COL_MAJOR ? CblasColMajor : CblasRowMajor;
  cblas_gemm<T>(clayout, ctransA, ctransB, m, n, k, alpha, A, lda, B, ldb, beta,
                C, ldc);
  return PERFLIBS_STATUS_SUCCESS;
}
template perflibs_status_t
spmm_gemm(perflibs_dense_layout layout, perflibs_sparse_hint_value transA,
          perflibs_sparse_hint_value transB, perflibs_int_t m, perflibs_int_t n,
          perflibs_int_t k, float alpha, const float *A, perflibs_int_t lda,
          const float *B, perflibs_int_t ldb, float beta, float *C,
          perflibs_int_t ldc);
template perflibs_status_t
spmm_gemm(perflibs_dense_layout layout, perflibs_sparse_hint_value transA,
          perflibs_sparse_hint_value transB, perflibs_int_t m, perflibs_int_t n,
          perflibs_int_t k, double alpha, const double *A, perflibs_int_t lda,
          const double *B, perflibs_int_t ldb, double beta, double *C,
          perflibs_int_t ldc);
template perflibs_status_t
spmm_gemm(perflibs_dense_layout layout, perflibs_sparse_hint_value transA,
          perflibs_sparse_hint_value transB, perflibs_int_t m, perflibs_int_t n,
          perflibs_int_t k, std::complex<float> alpha,
          const std::complex<float> *A, perflibs_int_t lda,
          const std::complex<float> *B, perflibs_int_t ldb,
          std::complex<float> beta, std::complex<float> *C, perflibs_int_t ldc);
template perflibs_status_t spmm_gemm(
    perflibs_dense_layout layout, perflibs_sparse_hint_value transA,
    perflibs_sparse_hint_value transB, perflibs_int_t m, perflibs_int_t n,
    perflibs_int_t k, std::complex<double> alpha, const std::complex<double> *A,
    perflibs_int_t lda, const std::complex<double> *B, perflibs_int_t ldb,
    std::complex<double> beta, std::complex<double> *C, perflibs_int_t ldc);

template <typename T>
perflibs_status_t
spsv_trsv(perflibs_dense_layout layout, sparse_hint_value_internal trans,
          sparse_hint_value_internal uplo, sparse_hint_value_internal diag,
          perflibs_int_t n, const T *A, perflibs_int_t lda, T *x) {
  auto ctrans = trans == PERFLIBS_OPERATION_NOTRANS ? CblasNoTrans
                : trans == PERFLIBS_OPERATION_TRANS ? CblasTrans
                                                    : CblasConjTrans;
  auto cuplo =
      uplo == PERFLIBS_SHAPE_UPPER_TRIANGULAR ? CblasUpper : CblasLower;
  auto cdiag = diag == PERFLIBS_DIAG_UNIT || diag == PERFLIBS_DIAG_KNOWN_UNIT
                   ? CblasUnit
                   : CblasNonUnit;
  CBLAS_LAYOUT clayout =
      layout == PERFLIBS_COL_MAJOR ? CblasColMajor : CblasRowMajor;
  cblas_trsv<T>(clayout, cuplo, ctrans, cdiag, n, A, lda, x, 1);
  return PERFLIBS_STATUS_SUCCESS;
}
template perflibs_status_t
spsv_trsv(perflibs_dense_layout layout, sparse_hint_value_internal trans,
          sparse_hint_value_internal uplo, sparse_hint_value_internal diag,
          perflibs_int_t n, const float *A, perflibs_int_t lda, float *x);
template perflibs_status_t
spsv_trsv(perflibs_dense_layout layout, sparse_hint_value_internal trans,
          sparse_hint_value_internal uplo, sparse_hint_value_internal diag,
          perflibs_int_t n, const double *A, perflibs_int_t lda, double *x);
template perflibs_status_t
spsv_trsv(perflibs_dense_layout layout, sparse_hint_value_internal trans,
          sparse_hint_value_internal uplo, sparse_hint_value_internal diag,
          perflibs_int_t n, const std::complex<float> *A, perflibs_int_t lda,
          std::complex<float> *x);
template perflibs_status_t
spsv_trsv(perflibs_dense_layout layout, sparse_hint_value_internal trans,
          sparse_hint_value_internal uplo, sparse_hint_value_internal diag,
          perflibs_int_t n, const std::complex<double> *A, perflibs_int_t lda,
          std::complex<double> *x);

template <typename T>
void spnorm_inf_dense(const perflibs_dense<T> &dense,
                      perflibs::sparse::remove_complex_t<T> *result) {
  using RT = perflibs::sparse::remove_complex_t<T>;

  std::vector<RT> rowsums(dense.m);
  for (auto i = 0; i < dense.m; i++) {
    for (auto j = 0; j < dense.n; j++) {
      auto val = dense.layout == PERFLIBS_COL_MAJOR
                     ? dense.vals_ptr[i + (j * dense.lda)]
                     : dense.vals_ptr[j + (i * dense.lda)];
      rowsums[i] += std::abs(val);
    }
  }

  *result = spnorm_max(rowsums);
}
template void spnorm_inf_dense<float>(const perflibs_dense<float> &dense,
                                      float *result);
template void spnorm_inf_dense<double>(const perflibs_dense<double> &dense,
                                       double *result);
template void spnorm_inf_dense<std::complex<float>>(
    const perflibs_dense<std::complex<float>> &dense, float *result);
template void spnorm_inf_dense<std::complex<double>>(
    const perflibs_dense<std::complex<double>> &dense, double *result);

template <typename T>
perflibs_status_t
spelmm_dense(perflibs_sparse_hint_value transA, const perflibs_dense<T> &A,
             perflibs_sparse_hint_value transB, const perflibs_dense<T> &B,
             perflibs_dense_layout layoutC, perflibs_int_t ldaC,
             perflibs_spmat_t AB) {
  auto m = transA == PERFLIBS_SPARSE_OPERATION_NOTRANS ? A.m : A.n;
  auto n = transA == PERFLIBS_SPARSE_OPERATION_NOTRANS ? A.n : A.m;

  perflibs::sparse::pod_vector<T> valsAB(m * n);

  // Get conjugate of val if required
  auto conj = [&](T val, perflibs_sparse_hint_value trans) {
    return trans == PERFLIBS_SPARSE_OPERATION_CONJTRANS
               ? perflibs::sparse::conj(val)
               : val;
  };

  bool use_col_maj_idxA = true;
  bool use_col_maj_idxB = true;
  if ((transA == PERFLIBS_SPARSE_OPERATION_NOTRANS &&
       A.layout == PERFLIBS_ROW_MAJOR) ||
      (transA != PERFLIBS_SPARSE_OPERATION_NOTRANS &&
       A.layout == PERFLIBS_COL_MAJOR)) {
    use_col_maj_idxA = false;
  }
  if ((transB == PERFLIBS_SPARSE_OPERATION_NOTRANS &&
       B.layout == PERFLIBS_ROW_MAJOR) ||
      (transB != PERFLIBS_SPARSE_OPERATION_NOTRANS &&
       B.layout == PERFLIBS_COL_MAJOR)) {
    use_col_maj_idxB = false;
  }

  for (auto i = 0; i < m; i++) {
    for (auto j = 0; j < n; j++) {
      auto valA = use_col_maj_idxA ? A.vals_ptr[i + (j * A.lda)]
                                   : A.vals_ptr[j + (i * A.lda)];
      auto valB = use_col_maj_idxB ? B.vals_ptr[i + (j * B.lda)]
                                   : B.vals_ptr[j + (i * B.lda)];
      auto idx =
          layoutC == PERFLIBS_COL_MAJOR ? i + (j * ldaC) : j + (i * ldaC);
      valsAB[idx] = conj(valA, transA) * conj(valB, transB);
    }
  }

  // Create dense matrix from the result AB
  auto ret =
      fill_initial_data_dense(AB, layoutC, m, n, ldaC, 0, valsAB.data(), 0);
  if (ret != PERFLIBS_STATUS_SUCCESS) {
    return ret;
  }

  return PERFLIBS_STATUS_SUCCESS;
}
template perflibs_status_t spelmm_dense<float>(
    perflibs_sparse_hint_value transA, const perflibs_dense<float> &A,
    perflibs_sparse_hint_value transB, const perflibs_dense<float> &B,
    perflibs_dense_layout layoutC, perflibs_int_t ldaC, perflibs_spmat_t AB);
template perflibs_status_t spelmm_dense<double>(
    perflibs_sparse_hint_value transA, const perflibs_dense<double> &A,
    perflibs_sparse_hint_value transB, const perflibs_dense<double> &B,
    perflibs_dense_layout layoutC, perflibs_int_t ldaC, perflibs_spmat_t AB);
template perflibs_status_t
spelmm_dense<std::complex<float>>(perflibs_sparse_hint_value transA,
                                  const perflibs_dense<std::complex<float>> &A,
                                  perflibs_sparse_hint_value transB,
                                  const perflibs_dense<std::complex<float>> &B,
                                  perflibs_dense_layout layoutC,
                                  perflibs_int_t ldaC, perflibs_spmat_t AB);
template perflibs_status_t spelmm_dense<std::complex<double>>(
    perflibs_sparse_hint_value transA,
    const perflibs_dense<std::complex<double>> &A,
    perflibs_sparse_hint_value transB,
    const perflibs_dense<std::complex<double>> &B,
    perflibs_dense_layout layoutC, perflibs_int_t ldaC, perflibs_spmat_t AB);

template <typename T>
perflibs_status_t
spmat_update_dense(perflibs_spmat_impl_t<T> *impl, perflibs_int_t n_updates,
                   const perflibs_int_t *row_indx,
                   const perflibs_int_t *col_indx, const T *vals) {
  auto A = const_cast<T *>(impl->dense.vals_ptr);

  auto is_unit = impl->diag == PERFLIBS_SPARSE_DIAG_UNIT;
  auto index_base = impl->index_base;

  for (perflibs_int_t i = 0; i < n_updates; i++) {
    auto index = impl->dense.layout == PERFLIBS_COL_MAJOR
                     ? impl->dense.lda * (col_indx[i] - index_base) +
                           row_indx[i] - index_base
                     : impl->dense.lda * (row_indx[i] - index_base) +
                           col_indx[i] - index_base;
    A[index] = vals[i];
    if (row_indx[i] == col_indx[i] && is_unit && vals[i] != T(1)) {
      impl->diag = PERFLIBS_SPARSE_DIAG_NON_UNIT;
    }
  }

  return PERFLIBS_STATUS_SUCCESS;
}
template perflibs_status_t
spmat_update_dense<float>(perflibs_spmat_impl_t<float> *impl,
                          perflibs_int_t n_updates,
                          const perflibs_int_t *row_indx,
                          const perflibs_int_t *col_indx, const float *vals);
template perflibs_status_t
spmat_update_dense<double>(perflibs_spmat_impl_t<double> *impl,
                           perflibs_int_t n_updates,
                           const perflibs_int_t *row_indx,
                           const perflibs_int_t *col_indx, const double *vals);
template perflibs_status_t spmat_update_dense<std::complex<float>>(
    perflibs_spmat_impl_t<std::complex<float>> *impl, perflibs_int_t n_updates,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_indx,
    const std::complex<float> *vals);
template perflibs_status_t spmat_update_dense<std::complex<double>>(
    perflibs_spmat_impl_t<std::complex<double>> *impl, perflibs_int_t n_updates,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_indx,
    const std::complex<double> *vals);
} // namespace perflibs::sparse
