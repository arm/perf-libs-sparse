/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "perflibs_sparse.h"
#include "test_utils.h"

#include <cblas.h>
#include <stdlib.h>

static void compute_expected_mm(enum perflibs_sparse_hint_value transA,
                                enum perflibs_sparse_hint_value transB,
                                double alpha, double beta, const double *denseA,
                                perflibs_int_t A_rows, perflibs_int_t A_cols,
                                const double *denseB, perflibs_int_t B_rows,
                                perflibs_int_t B_cols, const double *c_init,
                                double *out) {
  const perflibs_int_t m =
      transA == PERFLIBS_SPARSE_OPERATION_NOTRANS ? A_rows : A_cols;
  const perflibs_int_t n =
      transB == PERFLIBS_SPARSE_OPERATION_NOTRANS ? B_cols : B_rows;
  const perflibs_int_t k =
      transA == PERFLIBS_SPARSE_OPERATION_NOTRANS ? A_cols : A_rows;
  const CBLAS_TRANSPOSE blas_transA =
      transA == PERFLIBS_SPARSE_OPERATION_NOTRANS ? CblasNoTrans : CblasTrans;
  const CBLAS_TRANSPOSE blas_transB =
      transB == PERFLIBS_SPARSE_OPERATION_NOTRANS ? CblasNoTrans : CblasTrans;
  if (c_init != NULL) {
    memcpy(out, c_init, (size_t)(m * n) * sizeof(*out));
  }
  cblas_dgemm(CblasRowMajor, blas_transA, blas_transB, (int)m, (int)n, (int)k,
              alpha, denseA, (int)A_cols, denseB, (int)B_cols, beta, out,
              (int)n);
}

static void compute_expected_add(enum perflibs_sparse_hint_value transA,
                                 enum perflibs_sparse_hint_value transB,
                                 double alpha, double beta,
                                 perflibs_int_t A_rows, perflibs_int_t A_cols,
                                 const double *denseA, perflibs_int_t B_rows,
                                 perflibs_int_t B_cols, const double *denseB,
                                 double *out) {
  perflibs_int_t m =
      transA == PERFLIBS_SPARSE_OPERATION_NOTRANS ? A_rows : A_cols;
  perflibs_int_t n =
      transA == PERFLIBS_SPARSE_OPERATION_NOTRANS ? A_cols : A_rows;
  for (perflibs_int_t i = 0; i < m; ++i) {
    for (perflibs_int_t j = 0; j < n; ++j) {
      double aval = test_access_dense_d(denseA, A_cols, transA, i, j);
      double bval = test_access_dense_d(denseB, B_cols, transB, i, j);
      out[i * n + j] = alpha * aval + beta * bval;
    }
  }
}

static int test_spmm_transpose_dense_sparse() {
  const perflibs_int_t row_ptrA[] = {0, 2, 4, 6, 8};
  const perflibs_int_t col_indxA[] = {0, 2, 0, 1, 1, 2, 2, 3};
  const double valsA[] = {1.0, -1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
  const double denseA[] = {1.0, 0.0, -1.0, 0.0, 2.0, 3.0, 0.0, 0.0,
                           0.0, 4.0, 5.0,  0.0, 0.0, 0.0, 6.0, 7.0};
  const double denseB[] = {1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                           9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};

  perflibs_spmat_t A = NULL;
  perflibs_spmat_t B = NULL;

  CHECK_STATUS(
      perflibs_spmat_create_csr_d(&A, 4, 4, row_ptrA, col_indxA, valsA, 0));
  CHECK_STATUS(perflibs_spmat_create_dense_d(&B, PERFLIBS_ROW_MAJOR, 4, 4, 4,
                                             denseB, 0));
  perflibs_spmat_t optimize_null = perflibs_spmat_create_null(4, 4);
  CHECK_TRUE(optimize_null != NULL, "spmm null output creation failed");
  CHECK_STATUS(perflibs_spmm_optimize(
      PERFLIBS_SPARSE_OPERATION_NOTRANS, PERFLIBS_SPARSE_OPERATION_NOTRANS,
      PERFLIBS_SPARSE_SCALAR_ANY, A, B, PERFLIBS_SPARSE_SCALAR_ANY,
      optimize_null));
  perflibs_spmat_destroy(optimize_null);

  const enum perflibs_sparse_hint_value trans_modes[] = {
      PERFLIBS_SPARSE_OPERATION_NOTRANS, PERFLIBS_SPARSE_OPERATION_TRANS};
  double expected[16];
  for (size_t ia = 0; ia < sizeof(trans_modes) / sizeof(trans_modes[0]); ++ia) {
    for (size_t ib = 0; ib < sizeof(trans_modes) / sizeof(trans_modes[0]);
         ++ib) {
      enum perflibs_sparse_hint_value transA = trans_modes[ia];
      enum perflibs_sparse_hint_value transB = trans_modes[ib];
      perflibs_int_t rows = 4;
      perflibs_int_t cols = 4;
      perflibs_spmat_t C = perflibs_spmat_create_null(rows, cols);
      CHECK_TRUE(C != NULL, "spmm result creation failed");
      double alpha = transA == PERFLIBS_SPARSE_OPERATION_NOTRANS ? 1.0 : 1.5;
      double beta = 0.0;
      CHECK_STATUS(perflibs_spmm_exec_d(transA, transB, alpha, A, B, beta, C));
      perflibs_int_t m = -1, n = -1;
      double *dense = NULL;
      CHECK_STATUS(
          perflibs_spmat_export_dense_d(C, PERFLIBS_ROW_MAJOR, &m, &n, &dense));
      compute_expected_mm(transA, transB, alpha, beta, denseA, 4, 4, denseB, 4,
                          4, NULL, expected);
      CHECK_DOUBLE_ARRAY(dense, expected, m * n, 1e-12);
      free(dense);
      perflibs_spmat_destroy(C);
    }
  }

  const double c_init[] = {1.0, -1.0, 0.5,  2.0, 0.0, 0.25, -0.5, 0.5,
                           1.0, 1.5,  -1.5, 2.5, 3.0, -2.0, 4.0,  5.0};
  perflibs_spmat_t c_dense = NULL;
  CHECK_STATUS(perflibs_spmat_create_dense_d(&c_dense, PERFLIBS_ROW_MAJOR, 4, 4,
                                             4, c_init, 0));
  const double alpha_beta = 0.7;
  const double beta = 1.0;
  CHECK_STATUS(perflibs_spmm_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                    PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                    alpha_beta, A, B, beta, c_dense));
  perflibs_int_t m = -1, n = -1;
  double *dense_result = NULL;
  CHECK_STATUS(perflibs_spmat_export_dense_d(c_dense, PERFLIBS_ROW_MAJOR, &m,
                                             &n, &dense_result));
  compute_expected_mm(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                      PERFLIBS_SPARSE_OPERATION_NOTRANS, alpha_beta, beta,
                      denseA, 4, 4, denseB, 4, 4, c_init, expected);
  CHECK_DOUBLE_ARRAY(dense_result, expected, m * n, 1e-12);
  free(dense_result);
  perflibs_spmat_destroy(c_dense);

  double denseB_col[16];
  double c_init_col[16];
  for (perflibs_int_t row = 0; row < 4; ++row) {
    for (perflibs_int_t col = 0; col < 4; ++col) {
      denseB_col[col * 4 + row] = denseB[row * 4 + col];
      c_init_col[col * 4 + row] = c_init[row * 4 + col];
    }
  }
  perflibs_spmat_t B_col = NULL;
  perflibs_spmat_t C_col = NULL;
  CHECK_STATUS(perflibs_spmat_create_dense_d(&B_col, PERFLIBS_COL_MAJOR, 4, 4,
                                             4, denseB_col, 0));
  CHECK_STATUS(perflibs_spmat_create_dense_d(&C_col, PERFLIBS_COL_MAJOR, 4, 4,
                                             4, c_init_col, 0));
  CHECK_STATUS(perflibs_spmm_optimize(
      PERFLIBS_SPARSE_OPERATION_NOTRANS, PERFLIBS_SPARSE_OPERATION_NOTRANS,
      PERFLIBS_SPARSE_SCALAR_ANY, A, B_col, PERFLIBS_SPARSE_SCALAR_ANY, C_col));
  CHECK_STATUS(perflibs_spmm_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                    PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                    alpha_beta, A, B_col, beta, C_col));
  dense_result = NULL;
  CHECK_STATUS(perflibs_spmat_export_dense_d(C_col, PERFLIBS_ROW_MAJOR, &m, &n,
                                             &dense_result));
  compute_expected_mm(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                      PERFLIBS_SPARSE_OPERATION_NOTRANS, alpha_beta, beta,
                      denseA, 4, 4, denseB, 4, 4, c_init, expected);
  CHECK_DOUBLE_ARRAY(dense_result, expected, m * n, 1e-12);
  free(dense_result);
  perflibs_spmat_destroy(C_col);
  perflibs_spmat_destroy(B_col);

  perflibs_spmat_destroy(A);
  perflibs_spmat_destroy(B);

  return EXIT_SUCCESS;
}

static int test_spadd_transpose_index_variants() {
  const perflibs_int_t row_ptrA[] = {1, 3, 5, 7, 9};
  const perflibs_int_t col_indxA[] = {1, 3, 1, 4, 2, 3, 1, 4};
  const double valsA[] = {1.0, -1.0, 2.5, 3.0, -0.5, 4.0, 1.5, -2.0};
  const perflibs_int_t row_ptrB[] = {0, 2, 4, 6, 8};
  const perflibs_int_t col_indxB[] = {1, 3, 0, 2, 1, 3, 0, 2};
  const double valsB[] = {4.0, -2.0, 5.0, -1.5, 2.0, 3.5, -3.0, 1.0};
  double denseA[16];
  double denseB[16];

  test_csr_to_dense_d(4, 4, row_ptrA, col_indxA, valsA, row_ptrA[0], denseA);
  test_csr_to_dense_d(4, 4, row_ptrB, col_indxB, valsB, row_ptrB[0], denseB);

  perflibs_spmat_t A = NULL;
  perflibs_spmat_t B = NULL;
  CHECK_STATUS(
      perflibs_spmat_create_csr_d(&A, 4, 4, row_ptrA, col_indxA, valsA, 0));
  CHECK_STATUS(
      perflibs_spmat_create_csr_d(&B, 4, 4, row_ptrB, col_indxB, valsB, 0));

  perflibs_spmat_t optimize_null = perflibs_spmat_create_null(4, 4);
  CHECK_TRUE(optimize_null != NULL, "spadd null output creation failed");
  CHECK_STATUS(perflibs_spadd_optimize(
      PERFLIBS_SPARSE_OPERATION_NOTRANS, PERFLIBS_SPARSE_OPERATION_TRANS,
      PERFLIBS_SPARSE_SCALAR_ANY, A, PERFLIBS_SPARSE_SCALAR_ANY, B,
      optimize_null));
  perflibs_spmat_destroy(optimize_null);

  perflibs_spmat_t C = perflibs_spmat_create_null(4, 4);
  CHECK_TRUE(C != NULL, "spadd result creation failed");
  CHECK_STATUS(perflibs_spadd_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                     PERFLIBS_SPARSE_OPERATION_TRANS, 1.0, A,
                                     1.0, B, C));
  perflibs_int_t m = -1, n = -1;
  double *dense_result = NULL;
  CHECK_STATUS(perflibs_spmat_export_dense_d(C, PERFLIBS_ROW_MAJOR, &m, &n,
                                             &dense_result));
  double expected_sum[16];
  compute_expected_add(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                       PERFLIBS_SPARSE_OPERATION_TRANS, 1.0, 1.0, 4, 4, denseA,
                       4, 4, denseB, expected_sum);
  CHECK_DOUBLE_ARRAY(dense_result, expected_sum, m * n, 1e-12);
  free(dense_result);
  perflibs_spmat_destroy(C);

  perflibs_spmat_t C_bonly = perflibs_spmat_create_null(4, 4);
  CHECK_TRUE(C_bonly != NULL, "spadd result creation failed");
  CHECK_STATUS(perflibs_spadd_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                     PERFLIBS_SPARSE_OPERATION_TRANS, 0.0, A,
                                     1.0, B, C_bonly));
  dense_result = NULL;
  CHECK_STATUS(perflibs_spmat_export_dense_d(C_bonly, PERFLIBS_ROW_MAJOR, &m,
                                             &n, &dense_result));
  double expected_beta_only[16];
  compute_expected_add(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                       PERFLIBS_SPARSE_OPERATION_TRANS, 0.0, 1.0, 4, 4, denseA,
                       4, 4, denseB, expected_beta_only);
  CHECK_DOUBLE_ARRAY(dense_result, expected_beta_only, m * n, 1e-12);
  free(dense_result);
  perflibs_spmat_destroy(C_bonly);
  perflibs_spmat_destroy(A);
  perflibs_spmat_destroy(B);

  return EXIT_SUCCESS;
}

int main() {
  if (test_spmm_transpose_dense_sparse() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (test_spadd_transpose_index_variants() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
