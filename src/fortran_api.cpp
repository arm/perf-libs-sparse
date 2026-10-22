/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include <cstdint>

#include <perflibs_sparse.h>
#include <perflibs_sparse_reproducibility.h>

#include "add.hpp"
#include "export.hpp"
#include "matmul.hpp"
#include "matrix_state.hpp"
#include "matvec.hpp"
#include "norm.hpp"
#include "object_helpers.hpp"
#include "solve.hpp"
#include "vector_state.hpp"

#define FTN_SYMB(FN_NAME) FN_NAME##_

// Fortran compatibility functions

extern "C" {

void FTN_SYMB(perflibs_sparse_set_reproducibility)(const int32_t *mode,
                                                   perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs_sparse_set_reproducibility(
      (perflibs_sparse_reproducibility_value)*mode);
}

int32_t FTN_SYMB(perflibs_sparse_get_reproducibility)(void) {
  return (int32_t)perflibs_sparse_get_reproducibility();
}

void FTN_SYMB(perflibs_spmat_create_csr_s)(
    perflibs_spmat_t *A, const perflibs_int_t *m, const perflibs_int_t *n,
    const perflibs_int_t *row_ptr, const perflibs_int_t *col_indx,
    const float *vals, perflibs_int_t *flags, perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::create_spmat_top_csr(
      A, *m, *n, row_ptr, col_indx, vals, *flags);
}
void FTN_SYMB(perflibs_spmat_create_csr_d)(
    perflibs_spmat_t *A, const perflibs_int_t *m, const perflibs_int_t *n,
    const perflibs_int_t *row_ptr, const perflibs_int_t *col_indx,
    const double *vals, perflibs_int_t *flags, perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::create_spmat_top_csr(
      A, *m, *n, row_ptr, col_indx, vals, *flags);
}
void FTN_SYMB(perflibs_spmat_create_csr_c)(
    perflibs_spmat_t *A, const perflibs_int_t *m, const perflibs_int_t *n,
    const perflibs_int_t *row_ptr, const perflibs_int_t *col_indx,
    const std::complex<float> *vals, perflibs_int_t *flags,
    perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::create_spmat_top_csr(
      A, *m, *n, row_ptr, col_indx, vals, *flags);
}
void FTN_SYMB(perflibs_spmat_create_csr_z)(
    perflibs_spmat_t *A, const perflibs_int_t *m, const perflibs_int_t *n,
    const perflibs_int_t *row_ptr, const perflibs_int_t *col_indx,
    const std::complex<double> *vals, perflibs_int_t *flags,
    perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::create_spmat_top_csr(
      A, *m, *n, row_ptr, col_indx, vals, *flags);
}

void FTN_SYMB(perflibs_spmat_create_csc_s)(
    perflibs_spmat_t *A, const perflibs_int_t *m, const perflibs_int_t *n,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_ptr,
    const float *vals, perflibs_int_t *flags, perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::create_spmat_top_csc(
      A, *m, *n, row_indx, col_ptr, vals, *flags);
}
void FTN_SYMB(perflibs_spmat_create_csc_d)(
    perflibs_spmat_t *A, const perflibs_int_t *m, const perflibs_int_t *n,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_ptr,
    const double *vals, perflibs_int_t *flags, perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::create_spmat_top_csc(
      A, *m, *n, row_indx, col_ptr, vals, *flags);
}
void FTN_SYMB(perflibs_spmat_create_csc_c)(
    perflibs_spmat_t *A, const perflibs_int_t *m, const perflibs_int_t *n,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_ptr,
    const std::complex<float> *vals, perflibs_int_t *flags,
    perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::create_spmat_top_csc(
      A, *m, *n, row_indx, col_ptr, vals, *flags);
}
void FTN_SYMB(perflibs_spmat_create_csc_z)(
    perflibs_spmat_t *A, const perflibs_int_t *m, const perflibs_int_t *n,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_ptr,
    const std::complex<double> *vals, perflibs_int_t *flags,
    perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::create_spmat_top_csc(
      A, *m, *n, row_indx, col_ptr, vals, *flags);
}

void FTN_SYMB(perflibs_spmat_create_coo_s)(
    perflibs_spmat_t *A, const perflibs_int_t *m, const perflibs_int_t *n,
    perflibs_int_t *nnz, const perflibs_int_t *row_indx,
    const perflibs_int_t *col_indx, const float *vals, perflibs_int_t *flags,
    perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::create_spmat_top_coo(
      A, *m, *n, *nnz, 1, row_indx, col_indx, vals, *flags);
}
void FTN_SYMB(perflibs_spmat_create_coo_d)(
    perflibs_spmat_t *A, const perflibs_int_t *m, const perflibs_int_t *n,
    perflibs_int_t *nnz, const perflibs_int_t *row_indx,
    const perflibs_int_t *col_indx, const double *vals, perflibs_int_t *flags,
    perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::create_spmat_top_coo(
      A, *m, *n, *nnz, 1, row_indx, col_indx, vals, *flags);
}
void FTN_SYMB(perflibs_spmat_create_coo_c)(
    perflibs_spmat_t *A, const perflibs_int_t *m, const perflibs_int_t *n,
    perflibs_int_t *nnz, const perflibs_int_t *row_indx,
    const perflibs_int_t *col_indx, const std::complex<float> *vals,
    perflibs_int_t *flags, perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::create_spmat_top_coo(
      A, *m, *n, *nnz, 1, row_indx, col_indx, vals, *flags);
}
void FTN_SYMB(perflibs_spmat_create_coo_z)(
    perflibs_spmat_t *A, const perflibs_int_t *m, const perflibs_int_t *n,
    perflibs_int_t *nnz, const perflibs_int_t *row_indx,
    const perflibs_int_t *col_indx, const std::complex<double> *vals,
    perflibs_int_t *flags, perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::create_spmat_top_coo(
      A, *m, *n, *nnz, 1, row_indx, col_indx, vals, *flags);
}

void FTN_SYMB(perflibs_spmat_create_dense_s)(
    perflibs_spmat_t *A, const perflibs_dense_layout *layout,
    const perflibs_int_t *m, const perflibs_int_t *n, const perflibs_int_t *lda,
    const float *vals, perflibs_int_t *flags, perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::create_spmat_top_dense(
      A, *layout, *m, *n, *lda, 1, vals, *flags);
}
void FTN_SYMB(perflibs_spmat_create_dense_d)(
    perflibs_spmat_t *A, const perflibs_dense_layout *layout,
    const perflibs_int_t *m, const perflibs_int_t *n, const perflibs_int_t *lda,
    const double *vals, perflibs_int_t *flags, perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::create_spmat_top_dense(
      A, *layout, *m, *n, *lda, 1, vals, *flags);
}
void FTN_SYMB(perflibs_spmat_create_dense_c)(
    perflibs_spmat_t *A, const perflibs_dense_layout *layout,
    const perflibs_int_t *m, const perflibs_int_t *n, const perflibs_int_t *lda,
    const std::complex<float> *vals, perflibs_int_t *flags,
    perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::create_spmat_top_dense(
      A, *layout, *m, *n, *lda, 1, vals, *flags);
}
void FTN_SYMB(perflibs_spmat_create_dense_z)(
    perflibs_spmat_t *A, const perflibs_dense_layout *layout,
    const perflibs_int_t *m, const perflibs_int_t *n, const perflibs_int_t *lda,
    const std::complex<double> *vals, perflibs_int_t *flags,
    perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::create_spmat_top_dense(
      A, *layout, *m, *n, *lda, 1, vals, *flags);
}

void FTN_SYMB(perflibs_spmat_create_bsr_s)(
    perflibs_spmat_t *A, const perflibs_dense_layout *block_layout,
    const perflibs_int_t *m, const perflibs_int_t *n,
    const perflibs_int_t *block_size, const perflibs_int_t *row_ptr,
    const perflibs_int_t *col_indx, const float *vals, perflibs_int_t *flags,
    perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::create_spmat_top_bsr(
      A, *block_layout, *m, *n, *block_size, row_ptr, col_indx, vals, *flags);
}
void FTN_SYMB(perflibs_spmat_create_bsr_d)(
    perflibs_spmat_t *A, const perflibs_dense_layout *block_layout,
    const perflibs_int_t *m, const perflibs_int_t *n,
    const perflibs_int_t *block_size, const perflibs_int_t *row_ptr,
    const perflibs_int_t *col_indx, const double *vals, perflibs_int_t *flags,
    perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::create_spmat_top_bsr(
      A, *block_layout, *m, *n, *block_size, row_ptr, col_indx, vals, *flags);
}
void FTN_SYMB(perflibs_spmat_create_bsr_c)(
    perflibs_spmat_t *A, const perflibs_dense_layout *block_layout,
    const perflibs_int_t *m, const perflibs_int_t *n,
    const perflibs_int_t *block_size, const perflibs_int_t *row_ptr,
    const perflibs_int_t *col_indx, const std::complex<float> *vals,
    perflibs_int_t *flags, perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::create_spmat_top_bsr(
      A, *block_layout, *m, *n, *block_size, row_ptr, col_indx, vals, *flags);
}
void FTN_SYMB(perflibs_spmat_create_bsr_z)(
    perflibs_spmat_t *A, const perflibs_dense_layout *block_layout,
    const perflibs_int_t *m, const perflibs_int_t *n,
    const perflibs_int_t *block_size, const perflibs_int_t *row_ptr,
    const perflibs_int_t *col_indx, const std::complex<double> *vals,
    perflibs_int_t *flags, perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::create_spmat_top_bsr(
      A, *block_layout, *m, *n, *block_size, row_ptr, col_indx, vals, *flags);
}

perflibs_spmat_t FTN_SYMB(perflibs_spmat_create_null)(const perflibs_int_t *m,
                                                      const perflibs_int_t *n) {
  return perflibs_spmat_create_null(*m, *n);
}

perflibs_spmat_t
FTN_SYMB(perflibs_spmat_create_identity)(const perflibs_int_t *n) {
  return perflibs_spmat_create_identity(*n);
}

void FTN_SYMB(perflibs_spmat_query)(const perflibs_spmat_t *A,
                                    perflibs_int_t *index_base,
                                    perflibs_int_t *m, perflibs_int_t *n,
                                    perflibs_int_t *nnz, perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs_spmat_query(*A, index_base, m, n, nnz);
}

void FTN_SYMB(perflibs_spmat_destroy)(perflibs_spmat_t *A,
                                      perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs_spmat_destroy(*A);
}

void FTN_SYMB(perflibs_spmat_hint)(perflibs_spmat_t *A,
                                   const perflibs_sparse_hint_type *type,
                                   const perflibs_sparse_hint_value *value,
                                   perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs_spmat_hint(*A, *type, *value);
}

void FTN_SYMB(perflibs_spvec_create_s)(perflibs_spvec_t *x,
                                       const perflibs_int_t *index_base,
                                       const perflibs_int_t *n,
                                       const perflibs_int_t *nnz,
                                       const perflibs_int_t *indx,
                                       const float *vals, perflibs_int_t *flags,
                                       perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::create_spvec_top(
      x, *index_base, *n, *nnz, indx, vals, *flags);
}
void FTN_SYMB(perflibs_spvec_create_d)(
    perflibs_spvec_t *x, const perflibs_int_t *index_base,
    const perflibs_int_t *n, const perflibs_int_t *nnz,
    const perflibs_int_t *indx, const double *vals, perflibs_int_t *flags,
    perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::create_spvec_top(
      x, *index_base, *n, *nnz, indx, vals, *flags);
}
void FTN_SYMB(perflibs_spvec_create_c)(
    perflibs_spvec_t *x, const perflibs_int_t *index_base,
    const perflibs_int_t *n, const perflibs_int_t *nnz,
    const perflibs_int_t *indx, const std::complex<float> *vals,
    perflibs_int_t *flags, perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::create_spvec_top(
      x, *index_base, *n, *nnz, indx, vals, *flags);
}
void FTN_SYMB(perflibs_spvec_create_z)(
    perflibs_spvec_t *x, const perflibs_int_t *index_base,
    const perflibs_int_t *n, const perflibs_int_t *nnz,
    const perflibs_int_t *indx, const std::complex<double> *vals,
    perflibs_int_t *flags, perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::create_spvec_top(
      x, *index_base, *n, *nnz, indx, vals, *flags);
}

void FTN_SYMB(perflibs_spvec_query)(const perflibs_spvec_t *x,
                                    perflibs_int_t *index_base,
                                    perflibs_int_t *n, perflibs_int_t *nnz,
                                    perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs_spvec_query(*x, index_base, n, nnz);
}

void FTN_SYMB(perflibs_spvec_destroy)(perflibs_spvec_t *x,
                                      perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs_spvec_destroy(*x);
}

void FTN_SYMB(perflibs_spvec_export_s)(perflibs_spvec_t *x,
                                       perflibs_int_t *index_base,
                                       perflibs_int_t *n, perflibs_int_t *nnz,
                                       perflibs_int_t *indx, float *vals,
                                       perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::export_spvec(*x, index_base, n, nnz,
                                                         indx, vals);
}
void FTN_SYMB(perflibs_spvec_export_d)(perflibs_spvec_t *x,
                                       perflibs_int_t *index_base,
                                       perflibs_int_t *n, perflibs_int_t *nnz,
                                       perflibs_int_t *indx, double *vals,
                                       perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::export_spvec(*x, index_base, n, nnz,
                                                         indx, vals);
}
void FTN_SYMB(perflibs_spvec_export_c)(perflibs_spvec_t *x,
                                       perflibs_int_t *index_base,
                                       perflibs_int_t *n, perflibs_int_t *nnz,
                                       perflibs_int_t *indx,
                                       std::complex<float> *vals,
                                       perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::export_spvec(*x, index_base, n, nnz,
                                                         indx, vals);
}
void FTN_SYMB(perflibs_spvec_export_z)(perflibs_spvec_t *x,
                                       perflibs_int_t *index_base,
                                       perflibs_int_t *n, perflibs_int_t *nnz,
                                       perflibs_int_t *indx,
                                       std::complex<double> *vals,
                                       perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::export_spvec(*x, index_base, n, nnz,
                                                         indx, vals);
}

void FTN_SYMB(perflibs_spvec_gather_s)(const float *x_d,
                                       perflibs_int_t *index_base,
                                       perflibs_int_t *n, perflibs_spvec_t *x_s,
                                       perflibs_int_t *flags,
                                       perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::gather_spvec_top(x_d, *index_base,
                                                             *n, x_s, *flags);
}
void FTN_SYMB(perflibs_spvec_gather_d)(const double *x_d,
                                       perflibs_int_t *index_base,
                                       perflibs_int_t *n, perflibs_spvec_t *x_s,
                                       perflibs_int_t *flags,
                                       perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::gather_spvec_top(x_d, *index_base,
                                                             *n, x_s, *flags);
}
void FTN_SYMB(perflibs_spvec_gather_c)(const std::complex<float> *x_d,
                                       perflibs_int_t *index_base,
                                       perflibs_int_t *n, perflibs_spvec_t *x_s,
                                       perflibs_int_t *flags,
                                       perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::gather_spvec_top(x_d, *index_base,
                                                             *n, x_s, *flags);
}
void FTN_SYMB(perflibs_spvec_gather_z)(const std::complex<double> *x_d,
                                       perflibs_int_t *index_base,
                                       perflibs_int_t *n, perflibs_spvec_t *x_s,
                                       perflibs_int_t *flags,
                                       perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::gather_spvec_top(x_d, *index_base,
                                                             *n, x_s, *flags);
}

void FTN_SYMB(perflibs_spvec_scatter_s)(perflibs_spvec_t *x_s, float *x_d,
                                        perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::scatter_spvec_top(*x_s, x_d);
}
void FTN_SYMB(perflibs_spvec_scatter_d)(perflibs_spvec_t *x_s, double *x_d,
                                        perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::scatter_spvec_top(*x_s, x_d);
}
void FTN_SYMB(perflibs_spvec_scatter_c)(perflibs_spvec_t *x_s,
                                        std::complex<float> *x_d,
                                        perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::scatter_spvec_top(*x_s, x_d);
}
void FTN_SYMB(perflibs_spvec_scatter_z)(perflibs_spvec_t *x_s,
                                        std::complex<double> *x_d,
                                        perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::scatter_spvec_top(*x_s, x_d);
}

void FTN_SYMB(perflibs_spvec_update_s)(perflibs_spvec_t *x,
                                       perflibs_int_t *n_updates,
                                       const perflibs_int_t *indx,
                                       const float *vals,
                                       perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::update_spvec(*x, *n_updates, indx,
                                                         vals);
}
void FTN_SYMB(perflibs_spvec_update_d)(perflibs_spvec_t *x,
                                       perflibs_int_t *n_updates,
                                       const perflibs_int_t *indx,
                                       const double *vals,
                                       perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::update_spvec(*x, *n_updates, indx,
                                                         vals);
}
void FTN_SYMB(perflibs_spvec_update_c)(perflibs_spvec_t *x,
                                       perflibs_int_t *n_updates,
                                       const perflibs_int_t *indx,
                                       const std::complex<float> *vals,
                                       perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::update_spvec(*x, *n_updates, indx,
                                                         vals);
}
void FTN_SYMB(perflibs_spvec_update_z)(perflibs_spvec_t *x,
                                       perflibs_int_t *n_updates,
                                       const perflibs_int_t *indx,
                                       const std::complex<double> *vals,
                                       perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::update_spvec(*x, *n_updates, indx,
                                                         vals);
}

void FTN_SYMB(perflibs_spdot_exec_s)(perflibs_spvec_t *x, const float *y,
                                     float *result, perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::dot_exec_top<false>(*x, y, result);
}
void FTN_SYMB(perflibs_spdot_exec_d)(perflibs_spvec_t *x, const double *y,
                                     double *result, perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::dot_exec_top<false>(*x, y, result);
}
void FTN_SYMB(perflibs_spdotu_exec_c)(perflibs_spvec_t *x,
                                      const std::complex<float> *y,
                                      std::complex<float> *result,
                                      perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::dot_exec_top<false>(*x, y, result);
}
void FTN_SYMB(perflibs_spdotu_exec_z)(perflibs_spvec_t *x,
                                      const std::complex<double> *y,
                                      std::complex<double> *result,
                                      perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::dot_exec_top<false>(*x, y, result);
}
void FTN_SYMB(perflibs_spdotc_exec_c)(perflibs_spvec_t *x,
                                      const std::complex<float> *y,
                                      std::complex<float> *result,
                                      perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::dot_exec_top<true>(*x, y, result);
}
void FTN_SYMB(perflibs_spdotc_exec_z)(perflibs_spvec_t *x,
                                      const std::complex<double> *y,
                                      std::complex<double> *result,
                                      perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::dot_exec_top<true>(*x, y, result);
}

void FTN_SYMB(perflibs_spaxpby_exec_s)(const float *alpha, perflibs_spvec_t *x,
                                       const float *beta, float *y,
                                       perflibs_int_t *info) {
  *info =
      (perflibs_int_t)perflibs::sparse::axpby_exec_top(*alpha, *x, *beta, y);
}
void FTN_SYMB(perflibs_spaxpby_exec_d)(const double *alpha, perflibs_spvec_t *x,
                                       const double *beta, double *y,
                                       perflibs_int_t *info) {
  *info =
      (perflibs_int_t)perflibs::sparse::axpby_exec_top(*alpha, *x, *beta, y);
}
void FTN_SYMB(perflibs_spaxpby_exec_c)(const std::complex<float> *alpha,
                                       perflibs_spvec_t *x,
                                       const std::complex<float> *beta,
                                       std::complex<float> *y,
                                       perflibs_int_t *info) {
  *info =
      (perflibs_int_t)perflibs::sparse::axpby_exec_top(*alpha, *x, *beta, y);
}
void FTN_SYMB(perflibs_spaxpby_exec_z)(const std::complex<double> *alpha,
                                       perflibs_spvec_t *x,
                                       const std::complex<double> *beta,
                                       std::complex<double> *y,
                                       perflibs_int_t *info) {
  *info =
      (perflibs_int_t)perflibs::sparse::axpby_exec_top(*alpha, *x, *beta, y);
}

void FTN_SYMB(perflibs_spwaxpby_exec_s)(const float *alpha, perflibs_spvec_t *x,
                                        const float *beta, const float *y,
                                        float *w, perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::waxpby_exec_top(*alpha, *x, *beta,
                                                            y, w);
}
void FTN_SYMB(perflibs_spwaxpby_exec_d)(const double *alpha,
                                        perflibs_spvec_t *x, const double *beta,
                                        const double *y, double *w,
                                        perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::waxpby_exec_top(*alpha, *x, *beta,
                                                            y, w);
}
void FTN_SYMB(perflibs_spwaxpby_exec_c)(const std::complex<float> *alpha,
                                        perflibs_spvec_t *x,
                                        const std::complex<float> *beta,
                                        const std::complex<float> *y,
                                        std::complex<float> *w,
                                        perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::waxpby_exec_top(*alpha, *x, *beta,
                                                            y, w);
}
void FTN_SYMB(perflibs_spwaxpby_exec_z)(const std::complex<double> *alpha,
                                        perflibs_spvec_t *x,
                                        const std::complex<double> *beta,
                                        const std::complex<double> *y,
                                        std::complex<double> *w,
                                        perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::waxpby_exec_top(*alpha, *x, *beta,
                                                            y, w);
}

void FTN_SYMB(perflibs_spmv_optimize)(perflibs_spmat_t *A,
                                      perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs_spmv_optimize(*A);
}

void FTN_SYMB(perflibs_spmm_optimize)(const perflibs_sparse_hint_value *transA,
                                      const perflibs_sparse_hint_value *transB,
                                      const perflibs_sparse_hint_value *alpha,
                                      const perflibs_spmat_t *A,
                                      const perflibs_spmat_t *B,
                                      const perflibs_sparse_hint_value *beta,
                                      perflibs_spmat_t *C,
                                      perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs_spmm_optimize(*transA, *transB, *alpha, *A,
                                                 *B, *beta, *C);
}

void FTN_SYMB(perflibs_spadd_optimize)(const perflibs_sparse_hint_value *transA,
                                       const perflibs_sparse_hint_value *transB,
                                       const perflibs_sparse_hint_value *alpha,
                                       const perflibs_spmat_t *A,
                                       const perflibs_sparse_hint_value *beta,
                                       const perflibs_spmat_t *B,
                                       perflibs_spmat_t *C,
                                       perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs_spadd_optimize(*transA, *transB, *alpha, *A,
                                                  *beta, *B, *C);
}

void FTN_SYMB(perflibs_spsm_optimize)(const perflibs_sparse_hint_value *transA,
                                      perflibs_spmat_t *A, perflibs_spmat_t *X,
                                      const perflibs_sparse_hint_value *alpha,
                                      perflibs_spmat_t *Y,
                                      perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs_spsm_optimize(*transA, *A, *X, *alpha, *Y);
}

void FTN_SYMB(perflibs_spsv_optimize)(perflibs_spmat_t *A,
                                      perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs_spsv_optimize(*A);
}

void FTN_SYMB(perflibs_spelmm_optimize)(
    const perflibs_sparse_hint_value *transA,
    const perflibs_sparse_hint_value *transB,
    const perflibs_sparse_hint_value *alpha, const perflibs_spmat_t *A,
    const perflibs_spmat_t *B, const perflibs_sparse_hint_value *beta,
    perflibs_spmat_t *C, perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs_spelmm_optimize(*transA, *transB, *alpha, *A,
                                                   *B, *beta, *C);
}

void FTN_SYMB(perflibs_sddmm_optimize)(const perflibs_sparse_hint_value *transA,
                                       const perflibs_sparse_hint_value *transB,
                                       const perflibs_sparse_hint_value *alpha,
                                       const perflibs_spmat_t *A,
                                       const perflibs_spmat_t *B,
                                       const perflibs_sparse_hint_value *beta,
                                       perflibs_spmat_t *C,
                                       perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs_sddmm_optimize(*transA, *transB, *alpha, *A,
                                                  *B, *beta, *C);
}

void FTN_SYMB(perflibs_spmat_update_s)(perflibs_spmat_t *A,
                                       const perflibs_int_t *n_updates,
                                       const perflibs_int_t *row_indx,
                                       const perflibs_int_t *col_indx,
                                       const float *vals,
                                       perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::spmat_update(
      *A, *n_updates, row_indx, col_indx, vals);
}
void FTN_SYMB(perflibs_spmat_update_d)(perflibs_spmat_t *A,
                                       const perflibs_int_t *n_updates,
                                       const perflibs_int_t *row_indx,
                                       const perflibs_int_t *col_indx,
                                       const double *vals,
                                       perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::spmat_update(
      *A, *n_updates, row_indx, col_indx, vals);
}
void FTN_SYMB(perflibs_spmat_update_c)(perflibs_spmat_t *A,
                                       const perflibs_int_t *n_updates,
                                       const perflibs_int_t *row_indx,
                                       const perflibs_int_t *col_indx,
                                       const std::complex<float> *vals,
                                       perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::spmat_update(
      *A, *n_updates, row_indx, col_indx, vals);
}
void FTN_SYMB(perflibs_spmat_update_z)(perflibs_spmat_t *A,
                                       const perflibs_int_t *n_updates,
                                       const perflibs_int_t *row_indx,
                                       const perflibs_int_t *col_indx,
                                       const std::complex<double> *vals,
                                       perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::spmat_update(
      *A, *n_updates, row_indx, col_indx, vals);
}

void FTN_SYMB(perflibs_spmv_exec_s)(const perflibs_sparse_hint_value *trans,
                                    const float *alpha, perflibs_spmat_t *A,
                                    const float *x, const float *beta, float *y,
                                    perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::spmv_exec(*trans, *alpha, *A, x,
                                                      *beta, y);
}
void FTN_SYMB(perflibs_spmv_exec_d)(const perflibs_sparse_hint_value *trans,
                                    const double *alpha, perflibs_spmat_t *A,
                                    const double *x, const double *beta,
                                    double *y, perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::spmv_exec(*trans, *alpha, *A, x,
                                                      *beta, y);
}
void FTN_SYMB(perflibs_spmv_exec_c)(const perflibs_sparse_hint_value *trans,
                                    const std::complex<float> *alpha,
                                    perflibs_spmat_t *A,
                                    const std::complex<float> *x,
                                    const std::complex<float> *beta,
                                    std::complex<float> *y,
                                    perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::spmv_exec(*trans, *alpha, *A, x,
                                                      *beta, y);
}
void FTN_SYMB(perflibs_spmv_exec_z)(const perflibs_sparse_hint_value *trans,
                                    const std::complex<double> *alpha,
                                    perflibs_spmat_t *A,
                                    const std::complex<double> *x,
                                    const std::complex<double> *beta,
                                    std::complex<double> *y,
                                    perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::spmv_exec(*trans, *alpha, *A, x,
                                                      *beta, y);
}

void FTN_SYMB(perflibs_spmm_exec_s)(const perflibs_sparse_hint_value *transA,
                                    const perflibs_sparse_hint_value *transB,
                                    const float *alpha,
                                    const perflibs_spmat_t *A,
                                    const perflibs_spmat_t *B,
                                    const float *beta, perflibs_spmat_t *C,
                                    perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::spmm_exec(*transA, *transB, *alpha,
                                                      *A, *B, *beta, *C);
}

void FTN_SYMB(perflibs_spmm_exec_d)(const perflibs_sparse_hint_value *transA,
                                    const perflibs_sparse_hint_value *transB,
                                    const double *alpha,
                                    const perflibs_spmat_t *A,
                                    const perflibs_spmat_t *B,
                                    const double *beta, perflibs_spmat_t *C,
                                    perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::spmm_exec(*transA, *transB, *alpha,
                                                      *A, *B, *beta, *C);
}

void FTN_SYMB(perflibs_spmm_exec_c)(const perflibs_sparse_hint_value *transA,
                                    const perflibs_sparse_hint_value *transB,
                                    const std::complex<float> *alpha,
                                    const perflibs_spmat_t *A,
                                    const perflibs_spmat_t *B,
                                    const std::complex<float> *beta,
                                    perflibs_spmat_t *C, perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::spmm_exec(*transA, *transB, *alpha,
                                                      *A, *B, *beta, *C);
}

void FTN_SYMB(perflibs_spmm_exec_z)(const perflibs_sparse_hint_value *transA,
                                    const perflibs_sparse_hint_value *transB,
                                    const std::complex<double> *alpha,
                                    const perflibs_spmat_t *A,
                                    const perflibs_spmat_t *B,
                                    const std::complex<double> *beta,
                                    perflibs_spmat_t *C, perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::spmm_exec(*transA, *transB, *alpha,
                                                      *A, *B, *beta, *C);
}

void FTN_SYMB(perflibs_spsm_exec_s)(const perflibs_sparse_hint_value *transA,
                                    perflibs_spmat_t *A, perflibs_spmat_t *X,
                                    const float *alpha, perflibs_spmat_t *Y,
                                    perflibs_int_t *info) {
  *info =
      (perflibs_int_t)perflibs::sparse::spsm_exec(*transA, *A, *X, *alpha, *Y);
}

void FTN_SYMB(perflibs_spsm_exec_d)(const perflibs_sparse_hint_value *transA,
                                    perflibs_spmat_t *A, perflibs_spmat_t *X,
                                    const double *alpha, perflibs_spmat_t *Y,
                                    perflibs_int_t *info) {
  *info =
      (perflibs_int_t)perflibs::sparse::spsm_exec(*transA, *A, *X, *alpha, *Y);
}

void FTN_SYMB(perflibs_spsm_exec_c)(const perflibs_sparse_hint_value *transA,
                                    perflibs_spmat_t *A, perflibs_spmat_t *X,
                                    const std::complex<float> *alpha,
                                    perflibs_spmat_t *Y, perflibs_int_t *info) {
  *info =
      (perflibs_int_t)perflibs::sparse::spsm_exec(*transA, *A, *X, *alpha, *Y);
}

void FTN_SYMB(perflibs_spsm_exec_z)(const perflibs_sparse_hint_value *transA,
                                    perflibs_spmat_t *A, perflibs_spmat_t *X,
                                    const std::complex<double> *alpha,
                                    perflibs_spmat_t *Y, perflibs_int_t *info) {
  *info =
      (perflibs_int_t)perflibs::sparse::spsm_exec(*transA, *A, *X, *alpha, *Y);
}

void FTN_SYMB(perflibs_spsv_exec_s)(const perflibs_sparse_hint_value *trans,
                                    perflibs_spmat_t *A, float *x,
                                    const float *alpha, const float *y,
                                    perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::spsv_exec(*trans, *A, x, *alpha, y);
}

void FTN_SYMB(perflibs_spsv_exec_d)(const perflibs_sparse_hint_value *trans,
                                    perflibs_spmat_t *A, double *x,
                                    const double *alpha, const double *y,
                                    perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::spsv_exec(*trans, *A, x, *alpha, y);
}

void FTN_SYMB(perflibs_spsv_exec_c)(const perflibs_sparse_hint_value *trans,
                                    perflibs_spmat_t *A, std::complex<float> *x,
                                    const std::complex<float> *alpha,
                                    const std::complex<float> *y,
                                    perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::spsv_exec(*trans, *A, x, *alpha, y);
}

void FTN_SYMB(perflibs_spsv_exec_z)(const perflibs_sparse_hint_value *trans,
                                    perflibs_spmat_t *A,
                                    std::complex<double> *x,
                                    const std::complex<double> *alpha,
                                    const std::complex<double> *y,
                                    perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::spsv_exec(*trans, *A, x, *alpha, y);
}

void FTN_SYMB(perflibs_spadd_exec_s)(
    const perflibs_sparse_hint_value *transA,
    const perflibs_sparse_hint_value *transB, const float *alpha,
    const perflibs_spmat_t *A, const float *beta, const perflibs_spmat_t *B,
    perflibs_spmat_t *C, perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::spadd_exec(*transA, *transB, *alpha,
                                                       *A, *beta, *B, *C);
}

void FTN_SYMB(perflibs_spadd_exec_d)(
    const perflibs_sparse_hint_value *transA,
    const perflibs_sparse_hint_value *transB, const double *alpha,
    const perflibs_spmat_t *A, const double *beta, const perflibs_spmat_t *B,
    perflibs_spmat_t *C, perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::spadd_exec(*transA, *transB, *alpha,
                                                       *A, *beta, *B, *C);
}

void FTN_SYMB(perflibs_spadd_exec_c)(
    const perflibs_sparse_hint_value *transA,
    const perflibs_sparse_hint_value *transB, const std::complex<float> *alpha,
    const perflibs_spmat_t *A, const std::complex<float> *beta,
    const perflibs_spmat_t *B, perflibs_spmat_t *C, perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::spadd_exec(*transA, *transB, *alpha,
                                                       *A, *beta, *B, *C);
}

void FTN_SYMB(perflibs_spadd_exec_z)(
    const perflibs_sparse_hint_value *transA,
    const perflibs_sparse_hint_value *transB, const std::complex<double> *alpha,
    const perflibs_spmat_t *A, const std::complex<double> *beta,
    const perflibs_spmat_t *B, perflibs_spmat_t *C, perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::spadd_exec(*transA, *transB, *alpha,
                                                       *A, *beta, *B, *C);
}

void FTN_SYMB(perflibs_sprot_exec_s)(perflibs_spvec_t *x, float *y, float *c,
                                     float *s, perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::rot_exec_top(*x, y, *c, *s);
}

void FTN_SYMB(perflibs_sprot_exec_d)(perflibs_spvec_t *x, double *y, double *c,
                                     double *s, perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::rot_exec_top(*x, y, *c, *s);
}

void FTN_SYMB(perflibs_sprot_exec_c)(perflibs_spvec_t *x,
                                     std::complex<float> *y, float *c,
                                     std::complex<float> *s,
                                     perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::rot_exec_top(*x, y, *c, *s);
}

void FTN_SYMB(perflibs_sprot_exec_z)(perflibs_spvec_t *x,
                                     std::complex<double> *y, double *c,
                                     std::complex<double> *s,
                                     perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::rot_exec_top(*x, y, *c, *s);
}

void FTN_SYMB(perflibs_sprot_exec_cs)(perflibs_spvec_t *x,
                                      std::complex<float> *y, float *c,
                                      float *s, perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::rot_exec_top(*x, y, *c, *s);
}

void FTN_SYMB(perflibs_sprot_exec_zd)(perflibs_spvec_t *x,
                                      std::complex<double> *y, double *c,
                                      double *s, perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::rot_exec_top(*x, y, *c, *s);
}

// Export functions have their own Fortran interface blocks and implementations

void FTN_SYMB(perflibs_spmat_print_err)(perflibs_spmat_t *A) {
  perflibs::sparse::spmat_print_err(*A);
}

void FTN_SYMB(perflibs_spvec_print_err)(perflibs_spvec_t *x) {
  perflibs::sparse::spvec_print_err(*x);
}

void FTN_SYMB(perflibs_spnorm_exec_cs)(perflibs_spmat_t *A,
                                       const perflibs_sparse_norm *nrm,
                                       float *result, perflibs_int_t *info) {
  if ((*A)->datatype == PERFLIBS_DATATYPE_SINGLE) {
    *info =
        (perflibs_int_t)perflibs::sparse::spnorm_exec<float>(*A, *nrm, result);
  } else {
    *info = (perflibs_int_t)perflibs::sparse::spnorm_exec<std::complex<float>>(
        *A, *nrm, result);
  }
}

void FTN_SYMB(perflibs_spnorm_exec_zd)(perflibs_spmat_t *A,
                                       const perflibs_sparse_norm *nrm,
                                       double *result, perflibs_int_t *info) {
  if ((*A)->datatype == PERFLIBS_DATATYPE_DOUBLE) {
    *info =
        (perflibs_int_t)perflibs::sparse::spnorm_exec<double>(*A, *nrm, result);
  } else {
    *info = (perflibs_int_t)perflibs::sparse::spnorm_exec<std::complex<double>>(
        *A, *nrm, result);
  }
}

void FTN_SYMB(perflibs_spelmm_exec_s)(const perflibs_sparse_hint_value *transA,
                                      const perflibs_sparse_hint_value *transB,
                                      const float *alpha,
                                      const perflibs_spmat_t *A,
                                      const perflibs_spmat_t *B,
                                      const float *beta, perflibs_spmat_t *C,
                                      perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::spelmm_exec(
      *transA, *transB, *alpha, *A, *B, *beta, *C);
}

void FTN_SYMB(perflibs_spelmm_exec_d)(const perflibs_sparse_hint_value *transA,
                                      const perflibs_sparse_hint_value *transB,
                                      const double *alpha,
                                      const perflibs_spmat_t *A,
                                      const perflibs_spmat_t *B,
                                      const double *beta, perflibs_spmat_t *C,
                                      perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::spelmm_exec(
      *transA, *transB, *alpha, *A, *B, *beta, *C);
}

void FTN_SYMB(perflibs_spelmm_exec_c)(const perflibs_sparse_hint_value *transA,
                                      const perflibs_sparse_hint_value *transB,
                                      const std::complex<float> *alpha,
                                      const perflibs_spmat_t *A,
                                      const perflibs_spmat_t *B,
                                      const std::complex<float> *beta,
                                      perflibs_spmat_t *C,
                                      perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::spelmm_exec(
      *transA, *transB, *alpha, *A, *B, *beta, *C);
}

void FTN_SYMB(perflibs_spelmm_exec_z)(const perflibs_sparse_hint_value *transA,
                                      const perflibs_sparse_hint_value *transB,
                                      const std::complex<double> *alpha,
                                      const perflibs_spmat_t *A,
                                      const perflibs_spmat_t *B,
                                      const std::complex<double> *beta,
                                      perflibs_spmat_t *C,
                                      perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::spelmm_exec(
      *transA, *transB, *alpha, *A, *B, *beta, *C);
}

void FTN_SYMB(perflibs_sddmm_exec_s)(const perflibs_sparse_hint_value *transA,
                                     const perflibs_sparse_hint_value *transB,
                                     const float *alpha,
                                     const perflibs_spmat_t *A,
                                     const perflibs_spmat_t *B,
                                     const float *beta, perflibs_spmat_t *C,
                                     perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::sddmm_exec<float, false>(
      *transA, *transB, *alpha, *A, *B, *beta, *C);
}

void FTN_SYMB(perflibs_sddmm_exec_d)(const perflibs_sparse_hint_value *transA,
                                     const perflibs_sparse_hint_value *transB,
                                     const double *alpha,
                                     const perflibs_spmat_t *A,
                                     const perflibs_spmat_t *B,
                                     const double *beta, perflibs_spmat_t *C,
                                     perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::sddmm_exec<double, false>(
      *transA, *transB, *alpha, *A, *B, *beta, *C);
}

void FTN_SYMB(perflibs_sddmm_exec_c)(const perflibs_sparse_hint_value *transA,
                                     const perflibs_sparse_hint_value *transB,
                                     const std::complex<float> *alpha,
                                     const perflibs_spmat_t *A,
                                     const perflibs_spmat_t *B,
                                     const std::complex<float> *beta,
                                     perflibs_spmat_t *C,
                                     perflibs_int_t *info) {
  *info =
      (perflibs_int_t)perflibs::sparse::sddmm_exec<std::complex<float>, false>(
          *transA, *transB, *alpha, *A, *B, *beta, *C);
}

void FTN_SYMB(perflibs_sddmm_exec_z)(const perflibs_sparse_hint_value *transA,
                                     const perflibs_sparse_hint_value *transB,
                                     const std::complex<double> *alpha,
                                     const perflibs_spmat_t *A,
                                     const perflibs_spmat_t *B,
                                     const std::complex<double> *beta,
                                     perflibs_spmat_t *C,
                                     perflibs_int_t *info) {
  *info =
      (perflibs_int_t)perflibs::sparse::sddmm_exec<std::complex<double>, false>(
          *transA, *transB, *alpha, *A, *B, *beta, *C);
}

void FTN_SYMB(perflibs_spscale_exec_s)(const float *alpha, perflibs_spmat_t *A,
                                       perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::scale_matrix(
      PERFLIBS_SPARSE_OPERATION_NOTRANS, *alpha, *A);
}

void FTN_SYMB(perflibs_spscale_exec_d)(const double *alpha, perflibs_spmat_t *A,
                                       perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::scale_matrix(
      PERFLIBS_SPARSE_OPERATION_NOTRANS, *alpha, *A);
}

void FTN_SYMB(perflibs_spscale_exec_c)(const std::complex<float> *alpha,
                                       perflibs_spmat_t *A,
                                       perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::scale_matrix(
      PERFLIBS_SPARSE_OPERATION_NOTRANS, *alpha, *A);
}

void FTN_SYMB(perflibs_spscale_exec_z)(const std::complex<double> *alpha,
                                       perflibs_spmat_t *A,
                                       perflibs_int_t *info) {
  *info = (perflibs_int_t)perflibs::sparse::scale_matrix(
      PERFLIBS_SPARSE_OPERATION_NOTRANS, *alpha, *A);
}

void FTN_SYMB(perflibs_sptranspose_exec)(
    const perflibs_sparse_hint_value *transA, perflibs_spmat_t *A,
    perflibs_int_t *info) {

  if ((*A)->datatype == PERFLIBS_DATATYPE_SINGLE) {
    *info =
        (perflibs_int_t)perflibs::sparse::scale_matrix<float>(*transA, 1.0, *A);
  } else if ((*A)->datatype == PERFLIBS_DATATYPE_DOUBLE) {
    *info = (perflibs_int_t)perflibs::sparse::scale_matrix<double>(*transA, 1.0,
                                                                   *A);
  } else if ((*A)->datatype == PERFLIBS_DATATYPE_CPLXSINGLE) {
    *info = (perflibs_int_t)perflibs::sparse::scale_matrix<std::complex<float>>(
        *transA, (std::complex<float>)1.0, *A);
  } else {
    *info =
        (perflibs_int_t)perflibs::sparse::scale_matrix<std::complex<double>>(
            *transA, (std::complex<double>)1.0, *A);
  }
}
}
