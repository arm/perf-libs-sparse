/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#ifndef PERFLIBS_SPARSE_H
#define PERFLIBS_SPARSE_H

// Always use the Windows complex type when
// this is explicitly requested
#ifdef _CRT_USE_C_COMPLEX_H

#ifndef perflibs_singlecomplex_t
#include <complex.h>
typedef _Fcomplex perflibs_singlecomplex_t;
#define perflibs_singlecomplex_t perflibs_singlecomplex_t
#endif

#ifndef perflibs_doublecomplex_t
#include <complex.h>
typedef _Dcomplex perflibs_doublecomplex_t;
#define perflibs_doublecomplex_t perflibs_doublecomplex_t
#endif

#endif

#ifdef PERFLIBS_BUILD

#include "perflibs_complex.h"
#include "perflibs_int.h"
#include "perflibs_status.h"

#else // not PERFLIBS_BUILD

#include <stdint.h>
#ifdef INTEGER64
typedef int64_t perflibs_int_t;
#else
typedef int32_t perflibs_int_t;
#endif

#ifndef perflibs_singlecomplex_t
#include <complex.h>
#if defined(_WIN32)
typedef _Fcomplex perflibs_singlecomplex_t;
#else
typedef float _Complex perflibs_singlecomplex_t;
#endif
#define perflibs_singlecomplex_t perflibs_singlecomplex_t
#endif

#ifndef perflibs_doublecomplex_t
#include <complex.h>
#if defined(_WIN32)
typedef _Dcomplex perflibs_doublecomplex_t;
#else
typedef double _Complex perflibs_doublecomplex_t;
#endif
#define perflibs_doublecomplex_t perflibs_doublecomplex_t
#endif

#ifndef USE_PERFLIBS
// This is defined by perflibs_status.h, so don't use it if we're building
// against that lib
typedef enum perflibs_status {
  PERFLIBS_STATUS_SUCCESS = 0,
  PERFLIBS_STATUS_INPUT_PARAMETER_ERROR = 1,
  PERFLIBS_STATUS_EXECUTION_FAILURE = 2,
} perflibs_status_t;
#else
#include "perflibs_status.h"
#endif

#endif // end not PERFLIBS_BUILD

/* Structures */

typedef struct perflibs_spmat_top_t *perflibs_spmat_t;
typedef const struct perflibs_spmat_top_t *perflibs_const_spmat_t;
typedef struct perflibs_spvec_top_t *perflibs_spvec_t;
typedef const struct perflibs_spvec_top_t *perflibs_const_spvec_t;

/* ENUMs */

enum perflibs_sparse_hint_type {
  PERFLIBS_SPARSE_HINT_STRUCTURE = 50,
  PERFLIBS_SPARSE_HINT_SPMV_OPERATION = 60,
  PERFLIBS_SPARSE_HINT_SPMM_OPERATION = 61,
  PERFLIBS_SPARSE_HINT_SPADD_OPERATION = 62,
  PERFLIBS_SPARSE_HINT_SPSV_OPERATION = 63,
  PERFLIBS_SPARSE_HINT_SPMM_STRATEGY = 64,
  PERFLIBS_SPARSE_HINT_SPSM_OPERATION = 65,
  PERFLIBS_SPARSE_HINT_SPSV_STRATEGY = 66,
  PERFLIBS_SPARSE_HINT_MEMORY = 70,
  PERFLIBS_SPARSE_HINT_SPMV_INVOCATIONS = 80,
  PERFLIBS_SPARSE_HINT_SPMM_INVOCATIONS = 81,
  PERFLIBS_SPARSE_HINT_SPADD_INVOCATIONS = 82,
  PERFLIBS_SPARSE_HINT_SPSV_INVOCATIONS = 83
};

enum perflibs_dense_layout { PERFLIBS_COL_MAJOR = 90, PERFLIBS_ROW_MAJOR = 91 };

enum perflibs_sparse_hint_value {
  /* Structure hints */
  PERFLIBS_SPARSE_STRUCTURE_DENSE = 100,
  PERFLIBS_SPARSE_STRUCTURE_UNSTRUCTURED = 101,
  PERFLIBS_SPARSE_STRUCTURE_SYMMETRIC = 110,
  PERFLIBS_SPARSE_STRUCTURE_DIAGONAL = 120,
  PERFLIBS_SPARSE_STRUCTURE_BLOCKDIAGONAL = 130,
  PERFLIBS_SPARSE_STRUCTURE_BANDED = 140,
  PERFLIBS_SPARSE_STRUCTURE_TRIANGULAR = 150,
  PERFLIBS_SPARSE_STRUCTURE_BLOCKTRIANGULAR = 160,
  PERFLIBS_SPARSE_STRUCTURE_HERMITIAN = 170,
  PERFLIBS_SPARSE_STRUCTURE_HPCG = 180,
  /* Memory allocation allowed? */
  PERFLIBS_SPARSE_MEMORY_NOALLOCS = 200,
  PERFLIBS_SPARSE_MEMORY_ALLOCS = 201,
  /* SPMV transpose operation */
  PERFLIBS_SPARSE_OPERATION_NOTRANS = 300,
  PERFLIBS_SPARSE_OPERATION_TRANS = 310,
  PERFLIBS_SPARSE_OPERATION_CONJTRANS = 320,
  /* SPMV execution count estimate */
  PERFLIBS_SPARSE_INVOCATIONS_SINGLE = 400,
  PERFLIBS_SPARSE_INVOCATIONS_FEW = 410,
  PERFLIBS_SPARSE_INVOCATIONS_MANY = 420,
  /* Scalar hints */
  PERFLIBS_SPARSE_SCALAR_ONE = 500,
  PERFLIBS_SPARSE_SCALAR_ZERO = 501,
  PERFLIBS_SPARSE_SCALAR_ANY = 502,
  /* SpMM Strategies */
  PERFLIBS_SPARSE_SPMM_STRAT_UNSET = 600,
  PERFLIBS_SPARSE_SPMM_STRAT_OPT_NO_STRUCT = 601,
  PERFLIBS_SPARSE_SPMM_STRAT_OPT_PART_STRUCT = 602,
  PERFLIBS_SPARSE_SPMM_STRAT_OPT_FULL_STRUCT = 603,
  /* SpSV Strategies */
  PERFLIBS_SPARSE_SPSV_STRAT_UNSET = 700,
  PERFLIBS_SPARSE_SPSV_STRAT_SEPARATOR_SPARSE = 701,
  PERFLIBS_SPARSE_SPSV_STRAT_SEPARATOR_DENSE = 702
};

enum perflibs_sparse_norm {
  PERFLIBS_SPARSE_NORM_INF = 1001, // Infinity norm
  PERFLIBS_SPARSE_NORM_FRB = 1002  // Frobenius norm
};

/* Flags for matrix creation */
#define PERFLIBS_SPARSE_CREATE_NOCOPY 1

/* C function prototypes */

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

perflibs_status_t
perflibs_spmat_create_csr_s(perflibs_spmat_t *A, perflibs_int_t m,
                            perflibs_int_t n, const perflibs_int_t *row_ptr,
                            const perflibs_int_t *col_indx, const float *vals,
                            perflibs_int_t flags);
perflibs_status_t
perflibs_spmat_create_csr_d(perflibs_spmat_t *A, perflibs_int_t m,
                            perflibs_int_t n, const perflibs_int_t *row_ptr,
                            const perflibs_int_t *col_indx, const double *vals,
                            perflibs_int_t flags);
perflibs_status_t perflibs_spmat_create_csr_c(
    perflibs_spmat_t *A, perflibs_int_t m, perflibs_int_t n,
    const perflibs_int_t *row_ptr, const perflibs_int_t *col_indx,
    const perflibs_singlecomplex_t *vals, perflibs_int_t flags);
perflibs_status_t perflibs_spmat_create_csr_z(
    perflibs_spmat_t *A, perflibs_int_t m, perflibs_int_t n,
    const perflibs_int_t *row_ptr, const perflibs_int_t *col_indx,
    const perflibs_doublecomplex_t *vals, perflibs_int_t flags);
perflibs_status_t
perflibs_spmat_create_csc_s(perflibs_spmat_t *A, perflibs_int_t m,
                            perflibs_int_t n, const perflibs_int_t *row_indx,
                            const perflibs_int_t *col_ptr, const float *vals,
                            perflibs_int_t flags);
perflibs_status_t
perflibs_spmat_create_csc_d(perflibs_spmat_t *A, perflibs_int_t m,
                            perflibs_int_t n, const perflibs_int_t *row_indx,
                            const perflibs_int_t *col_ptr, const double *vals,
                            perflibs_int_t flags);
perflibs_status_t perflibs_spmat_create_csc_c(
    perflibs_spmat_t *A, perflibs_int_t m, perflibs_int_t n,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_ptr,
    const perflibs_singlecomplex_t *vals, perflibs_int_t flags);
perflibs_status_t perflibs_spmat_create_csc_z(
    perflibs_spmat_t *A, perflibs_int_t m, perflibs_int_t n,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_ptr,
    const perflibs_doublecomplex_t *vals, perflibs_int_t flags);
perflibs_status_t perflibs_spmat_create_coo_s(
    perflibs_spmat_t *A, perflibs_int_t m, perflibs_int_t n, perflibs_int_t nnz,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_indx,
    const float *vals, perflibs_int_t flags);
perflibs_status_t perflibs_spmat_create_coo_d(
    perflibs_spmat_t *A, perflibs_int_t m, perflibs_int_t n, perflibs_int_t nnz,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_indx,
    const double *vals, perflibs_int_t flags);
perflibs_status_t perflibs_spmat_create_coo_c(
    perflibs_spmat_t *A, perflibs_int_t m, perflibs_int_t n, perflibs_int_t nnz,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_indx,
    const perflibs_singlecomplex_t *vals, perflibs_int_t flags);
perflibs_status_t perflibs_spmat_create_coo_z(
    perflibs_spmat_t *A, perflibs_int_t m, perflibs_int_t n, perflibs_int_t nnz,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_indx,
    const perflibs_doublecomplex_t *vals, perflibs_int_t flags);
perflibs_status_t perflibs_spmat_create_dense_s(
    perflibs_spmat_t *A, enum perflibs_dense_layout layout, perflibs_int_t m,
    perflibs_int_t n, perflibs_int_t lda, const float *vals,
    perflibs_int_t flags);
perflibs_status_t perflibs_spmat_create_dense_d(
    perflibs_spmat_t *A, enum perflibs_dense_layout layout, perflibs_int_t m,
    perflibs_int_t n, perflibs_int_t lda, const double *vals,
    perflibs_int_t flags);
perflibs_status_t perflibs_spmat_create_dense_c(
    perflibs_spmat_t *A, enum perflibs_dense_layout layout, perflibs_int_t m,
    perflibs_int_t n, perflibs_int_t lda, const perflibs_singlecomplex_t *vals,
    perflibs_int_t flags);
perflibs_status_t perflibs_spmat_create_dense_z(
    perflibs_spmat_t *A, enum perflibs_dense_layout layout, perflibs_int_t m,
    perflibs_int_t n, perflibs_int_t lda, const perflibs_doublecomplex_t *vals,
    perflibs_int_t flags);
perflibs_status_t perflibs_spmat_create_bsr_s(
    perflibs_spmat_t *A, enum perflibs_dense_layout block_layout,
    perflibs_int_t m, perflibs_int_t n, perflibs_int_t block_size,
    const perflibs_int_t *row_ptr, const perflibs_int_t *col_indx,
    const float *vals, perflibs_int_t flags);
perflibs_status_t perflibs_spmat_create_bsr_d(
    perflibs_spmat_t *A, enum perflibs_dense_layout block_layout,
    perflibs_int_t m, perflibs_int_t n, perflibs_int_t block_size,
    const perflibs_int_t *row_ptr, const perflibs_int_t *col_indx,
    const double *vals, perflibs_int_t flags);
perflibs_status_t perflibs_spmat_create_bsr_c(
    perflibs_spmat_t *A, enum perflibs_dense_layout block_layout,
    perflibs_int_t m, perflibs_int_t n, perflibs_int_t block_size,
    const perflibs_int_t *row_ptr, const perflibs_int_t *col_indx,
    const perflibs_singlecomplex_t *vals, perflibs_int_t flags);
perflibs_status_t perflibs_spmat_create_bsr_z(
    perflibs_spmat_t *A, enum perflibs_dense_layout block_layout,
    perflibs_int_t m, perflibs_int_t n, perflibs_int_t block_size,
    const perflibs_int_t *row_ptr, const perflibs_int_t *col_indx,
    const perflibs_doublecomplex_t *vals, perflibs_int_t flags);
perflibs_status_t perflibs_spmat_create_supernodal_s(
    perflibs_spmat_t *A, perflibs_int_t m, perflibs_int_t n,
    perflibs_int_t nsuper, perflibs_int_t nparts,
    const perflibs_int_t *super_row_ptr, const perflibs_int_t *super_col_indx,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_ptr,
    const float *vals, const perflibs_int_t *part_indx, perflibs_int_t flags);
perflibs_status_t perflibs_spmat_create_supernodal_d(
    perflibs_spmat_t *A, perflibs_int_t m, perflibs_int_t n,
    perflibs_int_t nsuper, perflibs_int_t nparts,
    const perflibs_int_t *super_row_ptr, const perflibs_int_t *super_col_indx,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_ptr,
    const double *vals, const perflibs_int_t *part_indx, perflibs_int_t flags);
perflibs_status_t perflibs_spmat_create_supernodal_c(
    perflibs_spmat_t *A, perflibs_int_t m, perflibs_int_t n,
    perflibs_int_t nsuper, perflibs_int_t nparts,
    const perflibs_int_t *super_row_ptr, const perflibs_int_t *super_col_indx,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_ptr,
    const perflibs_singlecomplex_t *vals, const perflibs_int_t *part_indx,
    perflibs_int_t flags);
perflibs_status_t perflibs_spmat_create_supernodal_z(
    perflibs_spmat_t *A, perflibs_int_t m, perflibs_int_t n,
    perflibs_int_t nsuper, perflibs_int_t nparts,
    const perflibs_int_t *super_row_ptr, const perflibs_int_t *super_col_indx,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_ptr,
    const perflibs_doublecomplex_t *vals, const perflibs_int_t *part_indx,
    perflibs_int_t flags);

perflibs_spmat_t perflibs_spmat_create_null(perflibs_int_t m, perflibs_int_t n);
perflibs_spmat_t perflibs_spmat_create_identity(perflibs_int_t n);

perflibs_status_t perflibs_spmat_destroy(perflibs_spmat_t A);

perflibs_status_t perflibs_spmat_query(perflibs_spmat_t A,
                                       perflibs_int_t *index_base,
                                       perflibs_int_t *m, perflibs_int_t *n,
                                       perflibs_int_t *nnz);

perflibs_status_t perflibs_spmat_hint(perflibs_spmat_t A,
                                      enum perflibs_sparse_hint_type hint,
                                      enum perflibs_sparse_hint_value value);

perflibs_status_t perflibs_spvec_create_s(perflibs_spvec_t *x,
                                          perflibs_int_t index_base,
                                          perflibs_int_t n, perflibs_int_t nnz,
                                          const perflibs_int_t *indx,
                                          const float *vals,
                                          perflibs_int_t flags);
perflibs_status_t perflibs_spvec_create_d(perflibs_spvec_t *x,
                                          perflibs_int_t index_base,
                                          perflibs_int_t n, perflibs_int_t nnz,
                                          const perflibs_int_t *indx,
                                          const double *vals,
                                          perflibs_int_t flags);
perflibs_status_t perflibs_spvec_create_c(perflibs_spvec_t *x,
                                          perflibs_int_t index_base,
                                          perflibs_int_t n, perflibs_int_t nnz,
                                          const perflibs_int_t *indx,
                                          const perflibs_singlecomplex_t *vals,
                                          perflibs_int_t flags);
perflibs_status_t perflibs_spvec_create_z(perflibs_spvec_t *x,
                                          perflibs_int_t index_base,
                                          perflibs_int_t n, perflibs_int_t nnz,
                                          const perflibs_int_t *indx,
                                          const perflibs_doublecomplex_t *vals,
                                          perflibs_int_t flags);

perflibs_status_t perflibs_spvec_query(perflibs_spvec_t x,
                                       perflibs_int_t *index_base,
                                       perflibs_int_t *n, perflibs_int_t *nnz);

perflibs_status_t perflibs_spvec_destroy(perflibs_spvec_t x);

perflibs_status_t perflibs_spvec_export_s(perflibs_spvec_t x,
                                          perflibs_int_t *index_base,
                                          perflibs_int_t *n,
                                          perflibs_int_t *nnz,
                                          perflibs_int_t *indx, float *vals);
perflibs_status_t perflibs_spvec_export_d(perflibs_spvec_t x,
                                          perflibs_int_t *index_base,
                                          perflibs_int_t *n,
                                          perflibs_int_t *nnz,
                                          perflibs_int_t *indx, double *vals);
perflibs_status_t
perflibs_spvec_export_c(perflibs_spvec_t x, perflibs_int_t *index_base,
                        perflibs_int_t *n, perflibs_int_t *nnz,
                        perflibs_int_t *indx, perflibs_singlecomplex_t *vals);
perflibs_status_t
perflibs_spvec_export_z(perflibs_spvec_t x, perflibs_int_t *index_base,
                        perflibs_int_t *n, perflibs_int_t *nnz,
                        perflibs_int_t *indx, perflibs_doublecomplex_t *vals);

perflibs_status_t perflibs_spvec_gather_s(const float *x_d,
                                          perflibs_int_t index_base,
                                          perflibs_int_t n,
                                          perflibs_spvec_t *x_s,
                                          perflibs_int_t flags);
perflibs_status_t perflibs_spvec_gather_d(const double *x_d,
                                          perflibs_int_t index_base,
                                          perflibs_int_t n,
                                          perflibs_spvec_t *x_s,
                                          perflibs_int_t flags);
perflibs_status_t perflibs_spvec_gather_c(const perflibs_singlecomplex_t *x_d,
                                          perflibs_int_t index_base,
                                          perflibs_int_t n,
                                          perflibs_spvec_t *x_s,
                                          perflibs_int_t flags);
perflibs_status_t perflibs_spvec_gather_z(const perflibs_doublecomplex_t *x_d,
                                          perflibs_int_t index_base,
                                          perflibs_int_t n,
                                          perflibs_spvec_t *x_s,
                                          perflibs_int_t flags);

perflibs_status_t perflibs_spvec_scatter_s(perflibs_spvec_t x_s, float *x_d);
perflibs_status_t perflibs_spvec_scatter_d(perflibs_spvec_t x_s, double *x_d);
perflibs_status_t perflibs_spvec_scatter_c(perflibs_spvec_t x_s,
                                           perflibs_singlecomplex_t *x_d);
perflibs_status_t perflibs_spvec_scatter_z(perflibs_spvec_t x_s,
                                           perflibs_doublecomplex_t *x_d);

perflibs_status_t perflibs_spvec_update_s(perflibs_spvec_t x,
                                          perflibs_int_t n_updates,
                                          const perflibs_int_t *indx,
                                          const float *vals);
perflibs_status_t perflibs_spvec_update_d(perflibs_spvec_t x,
                                          perflibs_int_t n_updates,
                                          const perflibs_int_t *indx,
                                          const double *vals);
perflibs_status_t perflibs_spvec_update_c(perflibs_spvec_t x,
                                          perflibs_int_t n_updates,
                                          const perflibs_int_t *indx,
                                          const perflibs_singlecomplex_t *vals);
perflibs_status_t perflibs_spvec_update_z(perflibs_spvec_t x,
                                          perflibs_int_t n_updates,
                                          const perflibs_int_t *indx,
                                          const perflibs_doublecomplex_t *vals);

perflibs_status_t perflibs_spdot_exec_s(perflibs_spvec_t x, const float *y,
                                        float *result);
perflibs_status_t perflibs_spdot_exec_d(perflibs_spvec_t x, const double *y,
                                        double *result);
perflibs_status_t perflibs_spdotu_exec_c(perflibs_spvec_t x,
                                         const perflibs_singlecomplex_t *y,
                                         perflibs_singlecomplex_t *result);
perflibs_status_t perflibs_spdotu_exec_z(perflibs_spvec_t x,
                                         const perflibs_doublecomplex_t *y,
                                         perflibs_doublecomplex_t *result);
perflibs_status_t perflibs_spdotc_exec_c(perflibs_spvec_t x,
                                         const perflibs_singlecomplex_t *y,
                                         perflibs_singlecomplex_t *result);
perflibs_status_t perflibs_spdotc_exec_z(perflibs_spvec_t x,
                                         const perflibs_doublecomplex_t *y,
                                         perflibs_doublecomplex_t *result);

perflibs_status_t perflibs_spaxpby_exec_s(const float alpha, perflibs_spvec_t x,
                                          const float beta, float *y);
perflibs_status_t perflibs_spaxpby_exec_d(const double alpha,
                                          perflibs_spvec_t x, const double beta,
                                          double *y);
perflibs_status_t perflibs_spaxpby_exec_c(const perflibs_singlecomplex_t alpha,
                                          perflibs_spvec_t x,
                                          const perflibs_singlecomplex_t beta,
                                          perflibs_singlecomplex_t *y);
perflibs_status_t perflibs_spaxpby_exec_z(const perflibs_doublecomplex_t alpha,
                                          perflibs_spvec_t x,
                                          const perflibs_doublecomplex_t beta,
                                          perflibs_doublecomplex_t *y);

perflibs_status_t perflibs_spwaxpby_exec_s(const float alpha,
                                           perflibs_spvec_t x, const float beta,
                                           const float *y, float *w);
perflibs_status_t perflibs_spwaxpby_exec_d(const double alpha,
                                           perflibs_spvec_t x,
                                           const double beta, const double *y,
                                           double *w);
perflibs_status_t perflibs_spwaxpby_exec_c(const perflibs_singlecomplex_t alpha,
                                           perflibs_spvec_t x,
                                           const perflibs_singlecomplex_t beta,
                                           const perflibs_singlecomplex_t *y,
                                           perflibs_singlecomplex_t *w);
perflibs_status_t perflibs_spwaxpby_exec_z(const perflibs_doublecomplex_t alpha,
                                           perflibs_spvec_t x,
                                           const perflibs_doublecomplex_t beta,
                                           const perflibs_doublecomplex_t *y,
                                           perflibs_doublecomplex_t *w);

perflibs_status_t perflibs_spmv_optimize(perflibs_spmat_t A);
perflibs_status_t perflibs_spmm_optimize(enum perflibs_sparse_hint_value transA,
                                         enum perflibs_sparse_hint_value transB,
                                         enum perflibs_sparse_hint_value alpha,
                                         perflibs_spmat_t A, perflibs_spmat_t B,
                                         enum perflibs_sparse_hint_value beta,
                                         perflibs_spmat_t C);
perflibs_status_t
perflibs_spadd_optimize(enum perflibs_sparse_hint_value transA,
                        enum perflibs_sparse_hint_value transB,
                        enum perflibs_sparse_hint_value alpha,
                        perflibs_spmat_t A,
                        enum perflibs_sparse_hint_value beta,
                        perflibs_spmat_t B, perflibs_spmat_t C);
perflibs_status_t perflibs_spsm_optimize(enum perflibs_sparse_hint_value transA,
                                         perflibs_spmat_t A, perflibs_spmat_t X,
                                         enum perflibs_sparse_hint_value alpha,
                                         perflibs_spmat_t Y);
perflibs_status_t perflibs_spsv_optimize(perflibs_spmat_t A);
perflibs_status_t
perflibs_spelmm_optimize(enum perflibs_sparse_hint_value transA,
                         enum perflibs_sparse_hint_value transB,
                         enum perflibs_sparse_hint_value alpha,
                         perflibs_spmat_t A, perflibs_spmat_t B,
                         enum perflibs_sparse_hint_value beta,
                         perflibs_spmat_t C);
perflibs_status_t
perflibs_sddmm_optimize(enum perflibs_sparse_hint_value transA,
                        enum perflibs_sparse_hint_value transB,
                        enum perflibs_sparse_hint_value alpha,
                        perflibs_spmat_t A, perflibs_spmat_t B,
                        enum perflibs_sparse_hint_value beta,
                        perflibs_spmat_t C);

perflibs_status_t perflibs_spmv_exec_s(enum perflibs_sparse_hint_value trans,
                                       float alpha, perflibs_spmat_t A,
                                       const float *x, float beta, float *y);
perflibs_status_t perflibs_spmv_exec_d(enum perflibs_sparse_hint_value trans,
                                       double alpha, perflibs_spmat_t A,
                                       const double *x, double beta, double *y);
perflibs_status_t perflibs_spmv_exec_c(enum perflibs_sparse_hint_value trans,
                                       perflibs_singlecomplex_t alpha,
                                       perflibs_spmat_t A,
                                       const perflibs_singlecomplex_t *x,
                                       perflibs_singlecomplex_t beta,
                                       perflibs_singlecomplex_t *y);
perflibs_status_t perflibs_spmv_exec_z(enum perflibs_sparse_hint_value trans,
                                       perflibs_doublecomplex_t alpha,
                                       perflibs_spmat_t A,
                                       const perflibs_doublecomplex_t *x,
                                       perflibs_doublecomplex_t beta,
                                       perflibs_doublecomplex_t *y);

perflibs_status_t perflibs_spmm_exec_s(enum perflibs_sparse_hint_value transA,
                                       enum perflibs_sparse_hint_value transB,
                                       float alpha, perflibs_spmat_t A,
                                       perflibs_spmat_t B, float beta,
                                       perflibs_spmat_t C);
perflibs_status_t perflibs_spmm_exec_d(enum perflibs_sparse_hint_value transA,
                                       enum perflibs_sparse_hint_value transB,
                                       double alpha, perflibs_spmat_t A,
                                       perflibs_spmat_t B, double beta,
                                       perflibs_spmat_t C);
perflibs_status_t perflibs_spmm_exec_c(enum perflibs_sparse_hint_value transA,
                                       enum perflibs_sparse_hint_value transB,
                                       perflibs_singlecomplex_t alpha,
                                       perflibs_spmat_t A, perflibs_spmat_t B,
                                       perflibs_singlecomplex_t beta,
                                       perflibs_spmat_t C);
perflibs_status_t perflibs_spmm_exec_z(enum perflibs_sparse_hint_value transA,
                                       enum perflibs_sparse_hint_value transB,
                                       perflibs_doublecomplex_t alpha,
                                       perflibs_spmat_t A, perflibs_spmat_t B,
                                       perflibs_doublecomplex_t beta,
                                       perflibs_spmat_t C);

perflibs_status_t perflibs_spadd_exec_s(enum perflibs_sparse_hint_value transA,
                                        enum perflibs_sparse_hint_value transB,
                                        float alpha, perflibs_spmat_t A,
                                        float beta, perflibs_spmat_t B,
                                        perflibs_spmat_t C);
perflibs_status_t perflibs_spadd_exec_d(enum perflibs_sparse_hint_value transA,
                                        enum perflibs_sparse_hint_value transB,
                                        double alpha, perflibs_spmat_t A,
                                        double beta, perflibs_spmat_t B,
                                        perflibs_spmat_t C);
perflibs_status_t perflibs_spadd_exec_c(enum perflibs_sparse_hint_value transA,
                                        enum perflibs_sparse_hint_value transB,
                                        perflibs_singlecomplex_t alpha,
                                        perflibs_spmat_t A,
                                        perflibs_singlecomplex_t beta,
                                        perflibs_spmat_t B, perflibs_spmat_t C);
perflibs_status_t perflibs_spadd_exec_z(enum perflibs_sparse_hint_value transA,
                                        enum perflibs_sparse_hint_value transB,
                                        perflibs_doublecomplex_t alpha,
                                        perflibs_spmat_t A,
                                        perflibs_doublecomplex_t beta,
                                        perflibs_spmat_t B, perflibs_spmat_t C);

perflibs_status_t perflibs_spsm_exec_s(enum perflibs_sparse_hint_value transA,
                                       perflibs_spmat_t A, perflibs_spmat_t X,
                                       float alpha, perflibs_spmat_t Y);
perflibs_status_t perflibs_spsm_exec_d(enum perflibs_sparse_hint_value transA,
                                       perflibs_spmat_t A, perflibs_spmat_t X,
                                       double alpha, perflibs_spmat_t Y);
perflibs_status_t perflibs_spsm_exec_c(enum perflibs_sparse_hint_value transA,
                                       perflibs_spmat_t A, perflibs_spmat_t X,
                                       perflibs_singlecomplex_t alpha,
                                       perflibs_spmat_t Y);
perflibs_status_t perflibs_spsm_exec_z(enum perflibs_sparse_hint_value transA,
                                       perflibs_spmat_t A, perflibs_spmat_t X,
                                       perflibs_doublecomplex_t alpha,
                                       perflibs_spmat_t Y);

perflibs_status_t perflibs_spsv_exec_s(enum perflibs_sparse_hint_value transA,
                                       perflibs_spmat_t A, float *x,
                                       float alpha, const float *y);
perflibs_status_t perflibs_spsv_exec_d(enum perflibs_sparse_hint_value transA,
                                       perflibs_spmat_t A, double *x,
                                       double alpha, const double *y);
perflibs_status_t perflibs_spsv_exec_c(enum perflibs_sparse_hint_value transA,
                                       perflibs_spmat_t A,
                                       perflibs_singlecomplex_t *x,
                                       perflibs_singlecomplex_t alpha,
                                       const perflibs_singlecomplex_t *y);
perflibs_status_t perflibs_spsv_exec_z(enum perflibs_sparse_hint_value transA,
                                       perflibs_spmat_t A,
                                       perflibs_doublecomplex_t *x,
                                       perflibs_doublecomplex_t alpha,
                                       const perflibs_doublecomplex_t *y);

perflibs_status_t perflibs_spmat_update_s(perflibs_spmat_t A,
                                          perflibs_int_t n_updates,
                                          const perflibs_int_t *row_indx,
                                          const perflibs_int_t *col_indx,
                                          const float *vals);
perflibs_status_t perflibs_spmat_update_d(perflibs_spmat_t A,
                                          perflibs_int_t n_updates,
                                          const perflibs_int_t *row_indx,
                                          const perflibs_int_t *col_indx,
                                          const double *vals);
perflibs_status_t perflibs_spmat_update_c(perflibs_spmat_t A,
                                          perflibs_int_t n_updates,
                                          const perflibs_int_t *row_indx,
                                          const perflibs_int_t *col_indx,
                                          const perflibs_singlecomplex_t *vals);
perflibs_status_t perflibs_spmat_update_z(perflibs_spmat_t A,
                                          perflibs_int_t n_updates,
                                          const perflibs_int_t *row_indx,
                                          const perflibs_int_t *col_indx,
                                          const perflibs_doublecomplex_t *vals);

perflibs_status_t
perflibs_spmat_export_csr_s(perflibs_const_spmat_t A, perflibs_int_t index_base,
                            perflibs_int_t *m, perflibs_int_t *n,
                            perflibs_int_t **row_ptr, perflibs_int_t **col_indx,
                            float **vals);
perflibs_status_t
perflibs_spmat_export_csr_d(perflibs_const_spmat_t A, perflibs_int_t index_base,
                            perflibs_int_t *m, perflibs_int_t *n,
                            perflibs_int_t **row_ptr, perflibs_int_t **col_indx,
                            double **vals);
perflibs_status_t
perflibs_spmat_export_csr_c(perflibs_const_spmat_t A, perflibs_int_t index_base,
                            perflibs_int_t *m, perflibs_int_t *n,
                            perflibs_int_t **row_ptr, perflibs_int_t **col_indx,
                            perflibs_singlecomplex_t **vals);
perflibs_status_t
perflibs_spmat_export_csr_z(perflibs_const_spmat_t A, perflibs_int_t index_base,
                            perflibs_int_t *m, perflibs_int_t *n,
                            perflibs_int_t **row_ptr, perflibs_int_t **col_indx,
                            perflibs_doublecomplex_t **vals);
perflibs_status_t
perflibs_spmat_export_csc_s(perflibs_const_spmat_t A, perflibs_int_t index_base,
                            perflibs_int_t *m, perflibs_int_t *n,
                            perflibs_int_t **row_indx, perflibs_int_t **col_ptr,
                            float **vals);
perflibs_status_t
perflibs_spmat_export_csc_d(perflibs_const_spmat_t A, perflibs_int_t index_base,
                            perflibs_int_t *m, perflibs_int_t *n,
                            perflibs_int_t **row_indx, perflibs_int_t **col_ptr,
                            double **vals);
perflibs_status_t
perflibs_spmat_export_csc_c(perflibs_const_spmat_t A, perflibs_int_t index_base,
                            perflibs_int_t *m, perflibs_int_t *n,
                            perflibs_int_t **row_indx, perflibs_int_t **col_ptr,
                            perflibs_singlecomplex_t **vals);
perflibs_status_t
perflibs_spmat_export_csc_z(perflibs_const_spmat_t A, perflibs_int_t index_base,
                            perflibs_int_t *m, perflibs_int_t *n,
                            perflibs_int_t **row_indx, perflibs_int_t **col_ptr,
                            perflibs_doublecomplex_t **vals);
perflibs_status_t
perflibs_spmat_export_coo_s(perflibs_const_spmat_t A, perflibs_int_t *m,
                            perflibs_int_t *n, perflibs_int_t *nnz,
                            perflibs_int_t **row_indx,
                            perflibs_int_t **col_indx, float **vals);
perflibs_status_t
perflibs_spmat_export_coo_d(perflibs_const_spmat_t A, perflibs_int_t *m,
                            perflibs_int_t *n, perflibs_int_t *nnz,
                            perflibs_int_t **row_indx,
                            perflibs_int_t **col_indx, double **vals);
perflibs_status_t perflibs_spmat_export_coo_c(
    perflibs_const_spmat_t A, perflibs_int_t *m, perflibs_int_t *n,
    perflibs_int_t *nnz, perflibs_int_t **row_indx, perflibs_int_t **col_indx,
    perflibs_singlecomplex_t **vals);
perflibs_status_t perflibs_spmat_export_coo_z(
    perflibs_const_spmat_t A, perflibs_int_t *m, perflibs_int_t *n,
    perflibs_int_t *nnz, perflibs_int_t **row_indx, perflibs_int_t **col_indx,
    perflibs_doublecomplex_t **vals);
perflibs_status_t perflibs_spmat_export_dense_s(
    perflibs_const_spmat_t A, enum perflibs_dense_layout layout,
    perflibs_int_t *m, perflibs_int_t *n, float **vals);
perflibs_status_t perflibs_spmat_export_dense_d(
    perflibs_const_spmat_t A, enum perflibs_dense_layout layout,
    perflibs_int_t *m, perflibs_int_t *n, double **vals);
perflibs_status_t perflibs_spmat_export_dense_c(
    perflibs_const_spmat_t A, enum perflibs_dense_layout layout,
    perflibs_int_t *m, perflibs_int_t *n, perflibs_singlecomplex_t **vals);
perflibs_status_t perflibs_spmat_export_dense_z(
    perflibs_const_spmat_t A, enum perflibs_dense_layout layout,
    perflibs_int_t *m, perflibs_int_t *n, perflibs_doublecomplex_t **vals);
perflibs_status_t perflibs_spmat_export_bsr_s(
    perflibs_const_spmat_t A, enum perflibs_dense_layout block_layout,
    perflibs_int_t index_base, perflibs_int_t *m, perflibs_int_t *n,
    perflibs_int_t *block_size, perflibs_int_t **row_ptr,
    perflibs_int_t **col_indx, float **vals);
perflibs_status_t perflibs_spmat_export_bsr_d(
    perflibs_const_spmat_t A, enum perflibs_dense_layout block_layout,
    perflibs_int_t index_base, perflibs_int_t *m, perflibs_int_t *n,
    perflibs_int_t *block_size, perflibs_int_t **row_ptr,
    perflibs_int_t **col_indx, double **vals);
perflibs_status_t perflibs_spmat_export_bsr_c(
    perflibs_const_spmat_t A, enum perflibs_dense_layout block_layout,
    perflibs_int_t index_base, perflibs_int_t *m, perflibs_int_t *n,
    perflibs_int_t *block_size, perflibs_int_t **row_ptr,
    perflibs_int_t **col_indx, perflibs_singlecomplex_t **vals);
perflibs_status_t perflibs_spmat_export_bsr_z(
    perflibs_const_spmat_t A, enum perflibs_dense_layout block_layout,
    perflibs_int_t index_base, perflibs_int_t *m, perflibs_int_t *n,
    perflibs_int_t *block_size, perflibs_int_t **row_ptr,
    perflibs_int_t **col_indx, perflibs_doublecomplex_t **vals);

perflibs_status_t perflibs_sprot_exec_s(perflibs_spvec_t x, float *y, float c,
                                        float s);
perflibs_status_t perflibs_sprot_exec_d(perflibs_spvec_t x, double *y, double c,
                                        double s);
perflibs_status_t perflibs_sprot_exec_c(perflibs_spvec_t x,
                                        perflibs_singlecomplex_t *y, float c,
                                        perflibs_singlecomplex_t s);
perflibs_status_t perflibs_sprot_exec_cs(perflibs_spvec_t x,
                                         perflibs_singlecomplex_t *y, float c,
                                         float s);
perflibs_status_t perflibs_sprot_exec_z(perflibs_spvec_t x,
                                        perflibs_doublecomplex_t *y, double c,
                                        perflibs_doublecomplex_t s);
perflibs_status_t perflibs_sprot_exec_zd(perflibs_spvec_t x,
                                         perflibs_doublecomplex_t *y, double c,
                                         double s);

perflibs_status_t perflibs_spnorm_exec_s(perflibs_spmat_t A,
                                         enum perflibs_sparse_norm nrm,
                                         float *result);
perflibs_status_t perflibs_spnorm_exec_d(perflibs_spmat_t A,
                                         enum perflibs_sparse_norm nrm,
                                         double *result);
perflibs_status_t perflibs_spnorm_exec_c(perflibs_spmat_t A,
                                         enum perflibs_sparse_norm nrm,
                                         float *result);
perflibs_status_t perflibs_spnorm_exec_z(perflibs_spmat_t A,
                                         enum perflibs_sparse_norm nrm,
                                         double *result);

perflibs_status_t perflibs_spelmm_exec_s(enum perflibs_sparse_hint_value transA,
                                         enum perflibs_sparse_hint_value transB,
                                         float alpha, perflibs_spmat_t A,
                                         perflibs_spmat_t B, float beta,
                                         perflibs_spmat_t C);
perflibs_status_t perflibs_spelmm_exec_d(enum perflibs_sparse_hint_value transA,
                                         enum perflibs_sparse_hint_value transB,
                                         double alpha, perflibs_spmat_t A,
                                         perflibs_spmat_t B, double beta,
                                         perflibs_spmat_t C);
perflibs_status_t perflibs_spelmm_exec_c(enum perflibs_sparse_hint_value transA,
                                         enum perflibs_sparse_hint_value transB,
                                         perflibs_singlecomplex_t alpha,
                                         perflibs_spmat_t A, perflibs_spmat_t B,
                                         perflibs_singlecomplex_t beta,
                                         perflibs_spmat_t C);
perflibs_status_t perflibs_spelmm_exec_z(enum perflibs_sparse_hint_value transA,
                                         enum perflibs_sparse_hint_value transB,
                                         perflibs_doublecomplex_t alpha,
                                         perflibs_spmat_t A, perflibs_spmat_t B,
                                         perflibs_doublecomplex_t beta,
                                         perflibs_spmat_t C);

perflibs_status_t perflibs_sddmm_exec_s(enum perflibs_sparse_hint_value transA,
                                        enum perflibs_sparse_hint_value transB,
                                        float alpha, perflibs_spmat_t A,
                                        perflibs_spmat_t B, float beta,
                                        perflibs_spmat_t C);
perflibs_status_t perflibs_sddmm_exec_d(enum perflibs_sparse_hint_value transA,
                                        enum perflibs_sparse_hint_value transB,
                                        double alpha, perflibs_spmat_t A,
                                        perflibs_spmat_t B, double beta,
                                        perflibs_spmat_t C);
perflibs_status_t perflibs_sddmm_exec_c(enum perflibs_sparse_hint_value transA,
                                        enum perflibs_sparse_hint_value transB,
                                        perflibs_singlecomplex_t alpha,
                                        perflibs_spmat_t A, perflibs_spmat_t B,
                                        perflibs_singlecomplex_t beta,
                                        perflibs_spmat_t C);
perflibs_status_t perflibs_sddmm_exec_z(enum perflibs_sparse_hint_value transA,
                                        enum perflibs_sparse_hint_value transB,
                                        perflibs_doublecomplex_t alpha,
                                        perflibs_spmat_t A, perflibs_spmat_t B,
                                        perflibs_doublecomplex_t beta,
                                        perflibs_spmat_t C);

perflibs_status_t perflibs_spscale_exec_s(float alpha, perflibs_spmat_t A);
perflibs_status_t perflibs_spscale_exec_d(double alpha, perflibs_spmat_t A);
perflibs_status_t perflibs_spscale_exec_c(perflibs_singlecomplex_t alpha,
                                          perflibs_spmat_t A);
perflibs_status_t perflibs_spscale_exec_z(perflibs_doublecomplex_t alpha,
                                          perflibs_spmat_t A);

perflibs_status_t
perflibs_sptranspose_exec_s(enum perflibs_sparse_hint_value transA,
                            perflibs_spmat_t A);
perflibs_status_t
perflibs_sptranspose_exec_d(enum perflibs_sparse_hint_value transA,
                            perflibs_spmat_t A);
perflibs_status_t
perflibs_sptranspose_exec_c(enum perflibs_sparse_hint_value transA,
                            perflibs_spmat_t A);
perflibs_status_t
perflibs_sptranspose_exec_z(enum perflibs_sparse_hint_value transA,
                            perflibs_spmat_t A);

void perflibs_spmat_print_err(perflibs_spmat_t A);
void perflibs_spvec_print_err(perflibs_spvec_t x);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
