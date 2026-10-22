/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "perflibs_sparse.h"
#include "perflibs_sparse_reproducibility.h"
#include "test_utils.h"

#include <cblas.h>
#include <stdlib.h>
#include <string.h>

#define STRATEGY_PATHS_D_M 24
#define STRATEGY_PATHS_D_K 32
#define STRATEGY_PATHS_D_N 128
#define STRATEGY_PATHS_D_NNZA (STRATEGY_PATHS_D_M * 4)
#define STRATEGY_PATHS_D_NNZB (STRATEGY_PATHS_D_K * 4)

static int test_large_regular_spmv_optimize_and_update_double() {
  const perflibs_int_t m = 4096;
  const perflibs_int_t width = 12;
  const perflibs_int_t n = 16 * m;
  const perflibs_int_t nnz = m * width;
  const perflibs_int_t updated_row = 1234;
  const perflibs_int_t updated_col = 16 * updated_row;

  perflibs_int_t *row_ptr =
      (perflibs_int_t *)malloc(sizeof(perflibs_int_t) * (m + 1));
  perflibs_int_t *col_indx =
      (perflibs_int_t *)malloc(sizeof(perflibs_int_t) * nnz);
  double *vals = (double *)malloc(sizeof(double) * nnz);
  double *x = (double *)malloc(sizeof(double) * n);
  double *y = (double *)malloc(sizeof(double) * m);
  perflibs_spmat_t A = NULL;
  perflibs_int_t m_out = -1, n_out = -1;
  perflibs_int_t *row_ptr_out = NULL;
  perflibs_int_t *col_indx_out = NULL;
  double *vals_out = NULL;

  CHECK_TRUE(row_ptr && col_indx && vals && x && y,
             "allocation failure in large spmv double test");

  row_ptr[0] = 0;
  for (perflibs_int_t i = 0; i < m; ++i) {
    row_ptr[i + 1] = row_ptr[i] + width;
    for (perflibs_int_t j = 0; j < width; ++j) {
      col_indx[row_ptr[i] + j] = 16 * i + j;
      vals[row_ptr[i] + j] = 1.0;
    }
    y[i] = 0.0;
  }
  for (perflibs_int_t i = 0; i < n; ++i) {
    x[i] = 1.0;
  }

  CHECK_STATUS(
      perflibs_spmat_create_csr_d(&A, m, n, row_ptr, col_indx, vals, 0));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_SPMV_OPERATION,
                                   PERFLIBS_SPARSE_OPERATION_NOTRANS));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_SPMV_INVOCATIONS,
                                   PERFLIBS_SPARSE_INVOCATIONS_MANY));
  CHECK_STATUS(perflibs_spmv_optimize(A));

  CHECK_STATUS(perflibs_spmat_export_csr_d(A, 0, &m_out, &n_out, &row_ptr_out,
                                           &col_indx_out, &vals_out));
  CHECK_INT_EQ(m_out, m);
  CHECK_INT_EQ(n_out, n);
  CHECK_INT_EQ(row_ptr_out[m], nnz);
  CHECK_INT_EQ(col_indx_out[0], 0);
  CHECK_INT_EQ(col_indx_out[nnz - 1], n - 5);
  free(row_ptr_out);
  free(col_indx_out);
  free(vals_out);

  CHECK_STATUS(perflibs_spmv_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS, 1.0, A,
                                    x, 0.0, y));
  for (perflibs_int_t i = 0; i < m; ++i) {
    CHECK_DOUBLE_NEAR(y[i], (double)width, 1e-7);
  }

  const perflibs_int_t row_update = updated_row;
  const perflibs_int_t col_update = updated_col;
  const double val_update = 2.0;
  CHECK_STATUS(
      perflibs_spmat_update_d(A, 1, &row_update, &col_update, &val_update));
  memset(y, 0, sizeof(double) * m);
  CHECK_STATUS(perflibs_spmv_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS, 1.0, A,
                                    x, 0.0, y));
  for (perflibs_int_t i = 0; i < m; ++i) {
    double expected = i == updated_row ? (double)width + 1.0 : (double)width;
    CHECK_DOUBLE_NEAR(y[i], expected, 1e-7);
  }

  for (perflibs_int_t i = 0; i < m; ++i) {
    y[i] = 3.0;
  }
  CHECK_STATUS(perflibs_spmv_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS, 2.0, A,
                                    x, 1.0, y));
  for (perflibs_int_t i = 0; i < m; ++i) {
    double expected = i == updated_row ? 2.0 * ((double)width + 1.0) + 3.0
                                       : 2.0 * (double)width + 3.0;
    CHECK_DOUBLE_NEAR(y[i], expected, 1e-7);
  }

  CHECK_STATUS(perflibs_spmat_destroy(A));
  free(row_ptr);
  free(col_indx);
  free(vals);
  free(x);
  free(y);
  return EXIT_SUCCESS;
}

static void build_spmm_inputs_double(perflibs_int_t *row_ptrA,
                                     perflibs_int_t *col_indxA, double *valsA,
                                     perflibs_int_t *row_ptrB,
                                     perflibs_int_t *col_indxB, double *valsB,
                                     double *denseA, double *denseB) {
  const perflibs_int_t m = 24;
  const perflibs_int_t k = 32;
  const perflibs_int_t n = 128;

  memset(denseA, 0, sizeof(double) * m * k);
  memset(denseB, 0, sizeof(double) * k * n);

  row_ptrA[0] = 0;
  for (perflibs_int_t i = 0; i < m; ++i) {
    row_ptrA[i + 1] = row_ptrA[i] + 4;
    perflibs_int_t start = (i * 2) % (k - 3);
    for (perflibs_int_t j = 0; j < 4; ++j) {
      perflibs_int_t idx = row_ptrA[i] + j;
      perflibs_int_t col = start + j;
      double val = 1.0 + i * 0.1 + j * 0.01;
      col_indxA[idx] = col;
      valsA[idx] = val;
      denseA[i * k + col] = val;
    }
  }

  row_ptrB[0] = 0;
  for (perflibs_int_t i = 0; i < k; ++i) {
    row_ptrB[i + 1] = row_ptrB[i] + 4;
    for (perflibs_int_t j = 0; j < 4; ++j) {
      perflibs_int_t idx = row_ptrB[i] + j;
      perflibs_int_t col = 4 * i + j;
      double val = (double)(i + 1) * (double)(j + 1) * 0.5;
      col_indxB[idx] = col;
      valsB[idx] = val;
      denseB[i * n + col] = val;
    }
  }
}

static int test_spmm_optimized_symbolic_strategy_double(
    enum perflibs_sparse_hint_value strategy_hint) {
  perflibs_int_t row_ptrA[STRATEGY_PATHS_D_M + 1];
  perflibs_int_t col_indxA[STRATEGY_PATHS_D_NNZA];
  double valsA[STRATEGY_PATHS_D_NNZA];
  perflibs_int_t row_ptrB[STRATEGY_PATHS_D_K + 1];
  perflibs_int_t col_indxB[STRATEGY_PATHS_D_NNZB];
  double valsB[STRATEGY_PATHS_D_NNZB];
  double denseA[STRATEGY_PATHS_D_M * STRATEGY_PATHS_D_K];
  double denseB[STRATEGY_PATHS_D_K * STRATEGY_PATHS_D_N];
  double expected[STRATEGY_PATHS_D_M * STRATEGY_PATHS_D_N];

  build_spmm_inputs_double(row_ptrA, col_indxA, valsA, row_ptrB, col_indxB,
                           valsB, denseA, denseB);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, STRATEGY_PATHS_D_M,
              STRATEGY_PATHS_D_N, STRATEGY_PATHS_D_K, 1.0, denseA,
              STRATEGY_PATHS_D_K, denseB, STRATEGY_PATHS_D_N, 0.0, expected,
              STRATEGY_PATHS_D_N);

  perflibs_spmat_t A = NULL;
  perflibs_spmat_t B = NULL;
  perflibs_spmat_t C =
      perflibs_spmat_create_null(STRATEGY_PATHS_D_M, STRATEGY_PATHS_D_N);
  perflibs_int_t m_out = -1, n_out = -1;
  perflibs_int_t *row_ptrC = NULL;
  perflibs_int_t *col_indxC = NULL;
  double *valsC = NULL;
  double *denseC = NULL;

  CHECK_TRUE(C != NULL, "spmm output creation failed");
  CHECK_STATUS(perflibs_spmat_create_csr_d(&A, STRATEGY_PATHS_D_M,
                                           STRATEGY_PATHS_D_K, row_ptrA,
                                           col_indxA, valsA, 0));
  CHECK_STATUS(perflibs_spmat_create_csr_d(&B, STRATEGY_PATHS_D_K,
                                           STRATEGY_PATHS_D_N, row_ptrB,
                                           col_indxB, valsB, 0));
  CHECK_STATUS(perflibs_spmat_hint(C, PERFLIBS_SPARSE_HINT_SPMM_STRATEGY,
                                   strategy_hint));
  CHECK_STATUS(perflibs_spmm_optimize(
      PERFLIBS_SPARSE_OPERATION_NOTRANS, PERFLIBS_SPARSE_OPERATION_NOTRANS,
      PERFLIBS_SPARSE_SCALAR_ANY, A, B, PERFLIBS_SPARSE_SCALAR_ZERO, C));

  CHECK_STATUS(perflibs_spmat_export_csr_d(C, 0, &m_out, &n_out, &row_ptrC,
                                           &col_indxC, &valsC));
  CHECK_INT_EQ(m_out, STRATEGY_PATHS_D_M);
  CHECK_INT_EQ(n_out, STRATEGY_PATHS_D_N);
  CHECK_TRUE(row_ptrC[STRATEGY_PATHS_D_M] > 0,
             "optimized spmm produced an empty structure");
  free(row_ptrC);
  free(col_indxC);
  free(valsC);

  CHECK_STATUS(perflibs_spmm_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                    PERFLIBS_SPARSE_OPERATION_NOTRANS, 1.0, A,
                                    B, 0.0, C));
  CHECK_STATUS(perflibs_spmat_export_dense_d(C, PERFLIBS_ROW_MAJOR, &m_out,
                                             &n_out, &denseC));
  CHECK_DOUBLE_ARRAY(denseC, expected, STRATEGY_PATHS_D_M * STRATEGY_PATHS_D_N,
                     1e-12);
  free(denseC);

  CHECK_STATUS(perflibs_spmat_destroy(A));
  CHECK_STATUS(perflibs_spmat_destroy(B));
  CHECK_STATUS(perflibs_spmat_destroy(C));
  return EXIT_SUCCESS;
}

int main() {
  if (test_large_regular_spmv_optimize_and_update_double() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (test_spmm_optimized_symbolic_strategy_double(
          PERFLIBS_SPARSE_SPMM_STRAT_OPT_PART_STRUCT) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (test_spmm_optimized_symbolic_strategy_double(
          PERFLIBS_SPARSE_SPMM_STRAT_OPT_FULL_STRUCT) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }

  // Repeat the SpMM checks with reproducible matrix statistics, used by SVE.
  const enum perflibs_sparse_reproducibility_value initial_mode =
      perflibs_sparse_get_reproducibility();
  CHECK_STATUS(perflibs_sparse_set_reproducibility(
      PERFLIBS_SPARSE_REPRODUCIBILITY_ENABLED));
  int result = test_spmm_optimized_symbolic_strategy_double(
      PERFLIBS_SPARSE_SPMM_STRAT_OPT_PART_STRUCT);
  if (result == EXIT_SUCCESS) {
    result = test_spmm_optimized_symbolic_strategy_double(
        PERFLIBS_SPARSE_SPMM_STRAT_OPT_FULL_STRUCT);
  }
  CHECK_STATUS(perflibs_sparse_set_reproducibility(initial_mode));
  return result;
}
