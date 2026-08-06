/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "perflibs_sparse.h"
#include "test_utils.h"

#include <stdlib.h>

static void fill_dense_storage(double *storage,
                               enum perflibs_dense_layout layout,
                               perflibs_int_t m, perflibs_int_t n,
                               perflibs_int_t lda, const double *vals_col_major,
                               double pad_value) {
  const perflibs_int_t size = layout == PERFLIBS_COL_MAJOR ? lda * n : m * lda;
  for (perflibs_int_t i = 0; i < size; ++i) {
    storage[i] = pad_value;
  }

  for (perflibs_int_t col = 0; col < n; ++col) {
    for (perflibs_int_t row = 0; row < m; ++row) {
      const double value = vals_col_major[col * m + row];
      if (layout == PERFLIBS_COL_MAJOR) {
        storage[col * lda + row] = value;
      } else {
        storage[row * lda + col] = value;
      }
    }
  }
}

static int test_dense_spsm_layouts(enum perflibs_dense_layout layout_x,
                                   enum perflibs_dense_layout layout_y) {
  const perflibs_int_t n = 5;
  const perflibs_int_t nrhs = 2;
  const double alpha = 1.0;
  const perflibs_int_t lda_x =
      layout_x == PERFLIBS_COL_MAJOR ? n + 1 : nrhs + 2;
  const perflibs_int_t lda_y =
      layout_y == PERFLIBS_COL_MAJOR ? n + 2 : nrhs + 1;

  /*
   * Lower-triangular A (column-major):
   * [1 0 0 0 0
   *  2 1 0 0 0
   *  0 3 1 0 0
   *  0 0 4 1 0
   *  0 0 0 5 1]
   */
  const double A_vals[] = {1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.0,
                           0.0, 0.0, 0.0, 1.0, 4.0, 0.0, 0.0, 0.0, 0.0,
                           1.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0};
  const double Y_vals[] = {1.0, 4.0,  9.0,  16.0, 25.0,
                           5.0, 14.0, 15.0, 14.0, 11.0};
  const double expected_X[] = {1.0, 2.0, 3.0, 4.0, 5.0,
                               5.0, 4.0, 3.0, 2.0, 1.0};
  double X_init[25];
  double Y_storage[21];

  perflibs_spmat_t A = NULL;
  perflibs_spmat_t X = NULL;
  perflibs_spmat_t Y = NULL;
  double *X_out = NULL;
  perflibs_int_t m = -1;
  perflibs_int_t out_n = -1;

  fill_dense_storage(X_init, layout_x, n, nrhs, lda_x, (const double[10]){0.0},
                     -777.0);
  fill_dense_storage(Y_storage, layout_y, n, nrhs, lda_y, Y_vals, -555.0);

  CHECK_STATUS(perflibs_spmat_create_dense_d(&A, PERFLIBS_COL_MAJOR, n, n, n,
                                             A_vals, 0));
  CHECK_STATUS(perflibs_spmat_create_dense_d(
      &X, layout_x, n, nrhs, lda_x, X_init, PERFLIBS_SPARSE_CREATE_NOCOPY));
  CHECK_STATUS(perflibs_spmat_create_dense_d(
      &Y, layout_y, n, nrhs, lda_y, Y_storage, PERFLIBS_SPARSE_CREATE_NOCOPY));

  CHECK_STATUS(perflibs_spsm_optimize(PERFLIBS_SPARSE_OPERATION_NOTRANS, A, X,
                                      PERFLIBS_SPARSE_SCALAR_ANY, Y));

  CHECK_STATUS(
      perflibs_spsm_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS, A, X, alpha, Y));

  CHECK_STATUS(
      perflibs_spmat_export_dense_d(X, PERFLIBS_COL_MAJOR, &m, &out_n, &X_out));
  CHECK_TRUE(m == n && out_n == nrhs,
             "unexpected exported dense-layout SpSM solution shape %lld x %lld",
             test_i64(m), test_i64(out_n));
  CHECK_DOUBLE_ARRAY(X_out, expected_X, n * nrhs, 1e-12);

  free(X_out);
  CHECK_STATUS(perflibs_spmat_destroy(A));
  CHECK_STATUS(perflibs_spmat_destroy(X));
  CHECK_STATUS(perflibs_spmat_destroy(Y));
  return EXIT_SUCCESS;
}

static int test_basic_spsm_exec() {
  CHECK_STATUS(test_dense_spsm_layouts(PERFLIBS_COL_MAJOR, PERFLIBS_COL_MAJOR));
  CHECK_STATUS(test_dense_spsm_layouts(PERFLIBS_ROW_MAJOR, PERFLIBS_COL_MAJOR));
  CHECK_STATUS(test_dense_spsm_layouts(PERFLIBS_COL_MAJOR, PERFLIBS_ROW_MAJOR));
  CHECK_STATUS(test_dense_spsm_layouts(PERFLIBS_ROW_MAJOR, PERFLIBS_ROW_MAJOR));
  return EXIT_SUCCESS;
}

static int test_basic_csr_spsm_exec(enum perflibs_dense_layout layout_x,
                                    enum perflibs_dense_layout layout_y) {
  const perflibs_int_t n = 5;
  const perflibs_int_t nrhs = 3;
  const double alpha = 1.0;
  const perflibs_int_t lda_x =
      layout_x == PERFLIBS_COL_MAJOR ? n + 1 : nrhs + 2;
  const perflibs_int_t lda_y =
      layout_y == PERFLIBS_COL_MAJOR ? n + 2 : nrhs + 1;
  const perflibs_int_t row_ptr[] = {0, 1, 3, 5, 7, 9};
  const perflibs_int_t col_indx[] = {0, 0, 1, 1, 2, 2, 3, 3, 4};
  const double vals[] = {1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0};
  const double Y_vals[] = {1.0,  4.0,  9.0, 16.0, 25.0, 5.0, 14.0, 15.0,
                           14.0, 11.0, 2.0, 7.0,  8.0,  3.0, 19.0};
  const double expected_X[] = {1.0, 2.0, 3.0, 4.0, 5.0,  5.0, 4.0,  3.0,
                               2.0, 1.0, 2.0, 3.0, -1.0, 7.0, -16.0};
  double X_init[25];
  double Y_storage[21];

  perflibs_spmat_t A = NULL;
  perflibs_spmat_t X = NULL;
  perflibs_spmat_t Y = NULL;
  double *X_out = NULL;
  perflibs_int_t m = -1;
  perflibs_int_t out_n = -1;

  fill_dense_storage(X_init, layout_x, n, nrhs, lda_x, (const double[15]){0.0},
                     -333.0);
  fill_dense_storage(Y_storage, layout_y, n, nrhs, lda_y, Y_vals, -222.0);

  CHECK_STATUS(
      perflibs_spmat_create_csr_d(&A, n, n, row_ptr, col_indx, vals, 0));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_STRUCTURE,
                                   PERFLIBS_SPARSE_STRUCTURE_TRIANGULAR));
  CHECK_STATUS(perflibs_spmat_create_dense_d(
      &X, layout_x, n, nrhs, lda_x, X_init, PERFLIBS_SPARSE_CREATE_NOCOPY));
  CHECK_STATUS(perflibs_spmat_create_dense_d(
      &Y, layout_y, n, nrhs, lda_y, Y_storage, PERFLIBS_SPARSE_CREATE_NOCOPY));

  CHECK_STATUS(perflibs_spsm_optimize(PERFLIBS_SPARSE_OPERATION_NOTRANS, A, X,
                                      PERFLIBS_SPARSE_SCALAR_ANY, Y));

  CHECK_STATUS(
      perflibs_spsm_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS, A, X, alpha, Y));

  CHECK_STATUS(
      perflibs_spmat_export_dense_d(X, PERFLIBS_COL_MAJOR, &m, &out_n, &X_out));
  CHECK_TRUE(m == n && out_n == nrhs,
             "unexpected exported CSR-layout SpSM solution shape %lld x %lld",
             test_i64(m), test_i64(out_n));
  CHECK_DOUBLE_ARRAY(X_out, expected_X, n * nrhs, 1e-12);

  free(X_out);
  CHECK_STATUS(perflibs_spmat_destroy(A));
  CHECK_STATUS(perflibs_spmat_destroy(X));
  CHECK_STATUS(perflibs_spmat_destroy(Y));
  return EXIT_SUCCESS;
}

static int test_zero_alpha_csr_spsm_exec() {
  const perflibs_int_t n = 5;
  const perflibs_int_t nrhs = 3;
  const double alpha = 0.0;
  const enum perflibs_dense_layout layout_x = PERFLIBS_ROW_MAJOR;
  const enum perflibs_dense_layout layout_y = PERFLIBS_COL_MAJOR;
  const perflibs_int_t lda_x = nrhs + 2;
  const perflibs_int_t lda_y = n + 1;
  const perflibs_int_t row_ptr[] = {0, 1, 3, 5, 7, 9};
  const perflibs_int_t col_indx[] = {0, 0, 1, 1, 2, 2, 3, 3, 4};
  const double vals[] = {1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0};
  const double Y_vals[] = {1.0,  4.0,  9.0, 16.0, 25.0, 5.0, 14.0, 15.0,
                           14.0, 11.0, 2.0, 7.0,  8.0,  3.0, 19.0};
  const double expected_X[15] = {0.0};
  double X_init[25];
  double Y_storage[18];

  perflibs_spmat_t A = NULL;
  perflibs_spmat_t X = NULL;
  perflibs_spmat_t Y = NULL;
  double *X_out = NULL;
  perflibs_int_t m = -1;
  perflibs_int_t out_n = -1;

  fill_dense_storage(X_init, layout_x, n, nrhs, lda_x,
                     (const double[15]){1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                                        9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                                        15.0},
                     -444.0);
  fill_dense_storage(Y_storage, layout_y, n, nrhs, lda_y, Y_vals, -222.0);

  CHECK_STATUS(
      perflibs_spmat_create_csr_d(&A, n, n, row_ptr, col_indx, vals, 0));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_STRUCTURE,
                                   PERFLIBS_SPARSE_STRUCTURE_TRIANGULAR));
  CHECK_STATUS(perflibs_spmat_create_dense_d(
      &X, layout_x, n, nrhs, lda_x, X_init, PERFLIBS_SPARSE_CREATE_NOCOPY));
  CHECK_STATUS(perflibs_spmat_create_dense_d(
      &Y, layout_y, n, nrhs, lda_y, Y_storage, PERFLIBS_SPARSE_CREATE_NOCOPY));

  CHECK_STATUS(perflibs_spsm_optimize(PERFLIBS_SPARSE_OPERATION_NOTRANS, A, X,
                                      PERFLIBS_SPARSE_SCALAR_ANY, Y));

  CHECK_STATUS(
      perflibs_spsm_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS, A, X, alpha, Y));

  CHECK_STATUS(
      perflibs_spmat_export_dense_d(X, PERFLIBS_COL_MAJOR, &m, &out_n, &X_out));
  CHECK_TRUE(
      m == n && out_n == nrhs,
      "unexpected exported zero-alpha CSR-layout SpSM solution shape %lld "
      "x %lld",
      test_i64(m), test_i64(out_n));
  CHECK_DOUBLE_ARRAY(X_out, expected_X, n * nrhs, 1e-12);

  free(X_out);
  CHECK_STATUS(perflibs_spmat_destroy(A));
  CHECK_STATUS(perflibs_spmat_destroy(X));
  CHECK_STATUS(perflibs_spmat_destroy(Y));
  return EXIT_SUCCESS;
}

static int
test_basic_csc_spsm_notrans_exec(enum perflibs_dense_layout layout_x,
                                 enum perflibs_dense_layout layout_y) {
  const perflibs_int_t n = 5;
  const perflibs_int_t nrhs = 3;
  const double alpha = 1.0;
  const perflibs_int_t lda_x =
      layout_x == PERFLIBS_COL_MAJOR ? n + 1 : nrhs + 2;
  const perflibs_int_t lda_y =
      layout_y == PERFLIBS_COL_MAJOR ? n + 2 : nrhs + 1;
  const perflibs_int_t col_ptr[] = {0, 2, 4, 6, 8, 9};
  const perflibs_int_t row_indx[] = {0, 1, 1, 2, 2, 3, 3, 4, 4};
  const double vals[] = {1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0};
  const double Y_vals[] = {1.0,  4.0,  9.0, 16.0, 25.0, 5.0, 14.0, 15.0,
                           14.0, 11.0, 2.0, 7.0,  8.0,  3.0, 19.0};
  const double expected_X[] = {1.0, 2.0, 3.0, 4.0, 5.0,  5.0, 4.0,  3.0,
                               2.0, 1.0, 2.0, 3.0, -1.0, 7.0, -16.0};
  double X_init[25];
  double Y_storage[21];

  perflibs_spmat_t A = NULL;
  perflibs_spmat_t X = NULL;
  perflibs_spmat_t Y = NULL;
  double *X_out = NULL;
  perflibs_int_t m = -1;
  perflibs_int_t out_n = -1;

  fill_dense_storage(X_init, layout_x, n, nrhs, lda_x, (const double[15]){0.0},
                     -333.0);
  fill_dense_storage(Y_storage, layout_y, n, nrhs, lda_y, Y_vals, -222.0);

  CHECK_STATUS(
      perflibs_spmat_create_csc_d(&A, n, n, row_indx, col_ptr, vals, 0));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_STRUCTURE,
                                   PERFLIBS_SPARSE_STRUCTURE_TRIANGULAR));
  CHECK_STATUS(perflibs_spmat_create_dense_d(
      &X, layout_x, n, nrhs, lda_x, X_init, PERFLIBS_SPARSE_CREATE_NOCOPY));
  CHECK_STATUS(perflibs_spmat_create_dense_d(
      &Y, layout_y, n, nrhs, lda_y, Y_storage, PERFLIBS_SPARSE_CREATE_NOCOPY));

  CHECK_STATUS(perflibs_spsm_optimize(PERFLIBS_SPARSE_OPERATION_NOTRANS, A, X,
                                      PERFLIBS_SPARSE_SCALAR_ANY, Y));

  CHECK_STATUS(
      perflibs_spsm_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS, A, X, alpha, Y));

  CHECK_STATUS(
      perflibs_spmat_export_dense_d(X, PERFLIBS_COL_MAJOR, &m, &out_n, &X_out));
  CHECK_TRUE(m == n && out_n == nrhs,
             "unexpected exported CSC-notrans SpSM solution shape %lld x %lld",
             test_i64(m), test_i64(out_n));
  CHECK_DOUBLE_ARRAY(X_out, expected_X, n * nrhs, 1e-12);

  free(X_out);
  CHECK_STATUS(perflibs_spmat_destroy(A));
  CHECK_STATUS(perflibs_spmat_destroy(X));
  CHECK_STATUS(perflibs_spmat_destroy(Y));
  return EXIT_SUCCESS;
}

static int test_transpose_csr_spsm_exec(enum perflibs_dense_layout layout_x,
                                        enum perflibs_dense_layout layout_y) {
  const perflibs_int_t n = 5;
  const perflibs_int_t nrhs = 2;
  const double alpha = 1.0;
  const perflibs_int_t lda_x =
      layout_x == PERFLIBS_COL_MAJOR ? n + 1 : nrhs + 2;
  const perflibs_int_t lda_y =
      layout_y == PERFLIBS_COL_MAJOR ? n + 2 : nrhs + 1;
  const perflibs_int_t row_ptr[] = {0, 1, 3, 5, 7, 9};
  const perflibs_int_t col_indx[] = {0, 0, 1, 1, 2, 2, 3, 3, 4};
  const double vals[] = {1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0};
  const double Y_vals[] = {5.0,  11.0, 19.0, 29.0, 5.0,
                           13.0, 13.0, 11.0, 7.0,  1.0};
  const double expected_X[] = {1.0, 2.0, 3.0, 4.0, 5.0,
                               5.0, 4.0, 3.0, 2.0, 1.0};
  double X_init[25];
  double Y_storage[21];

  perflibs_spmat_t A = NULL;
  perflibs_spmat_t X = NULL;
  perflibs_spmat_t Y = NULL;
  double *X_out = NULL;
  perflibs_int_t m = -1;
  perflibs_int_t out_n = -1;

  fill_dense_storage(X_init, layout_x, n, nrhs, lda_x, (const double[10]){0.0},
                     -111.0);
  fill_dense_storage(Y_storage, layout_y, n, nrhs, lda_y, Y_vals, -999.0);

  CHECK_STATUS(
      perflibs_spmat_create_csr_d(&A, n, n, row_ptr, col_indx, vals, 0));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_STRUCTURE,
                                   PERFLIBS_SPARSE_STRUCTURE_TRIANGULAR));
  CHECK_STATUS(perflibs_spmat_create_dense_d(
      &X, layout_x, n, nrhs, lda_x, X_init, PERFLIBS_SPARSE_CREATE_NOCOPY));
  CHECK_STATUS(perflibs_spmat_create_dense_d(
      &Y, layout_y, n, nrhs, lda_y, Y_storage, PERFLIBS_SPARSE_CREATE_NOCOPY));

  CHECK_STATUS(perflibs_spsm_optimize(PERFLIBS_SPARSE_OPERATION_TRANS, A, X,
                                      PERFLIBS_SPARSE_SCALAR_ANY, Y));

  CHECK_STATUS(
      perflibs_spsm_exec_d(PERFLIBS_SPARSE_OPERATION_TRANS, A, X, alpha, Y));

  CHECK_STATUS(
      perflibs_spmat_export_dense_d(X, PERFLIBS_COL_MAJOR, &m, &out_n, &X_out));
  CHECK_TRUE(
      m == n && out_n == nrhs,
      "unexpected exported transpose CSR-layout SpSM solution shape %lld "
      "x %lld",
      test_i64(m), test_i64(out_n));
  CHECK_DOUBLE_ARRAY(X_out, expected_X, n * nrhs, 1e-12);

  free(X_out);
  CHECK_STATUS(perflibs_spmat_destroy(A));
  CHECK_STATUS(perflibs_spmat_destroy(X));
  CHECK_STATUS(perflibs_spmat_destroy(Y));
  return EXIT_SUCCESS;
}

static int test_chunked_csr_spsm_exec(enum perflibs_dense_layout layout_x,
                                      enum perflibs_dense_layout layout_y) {
  const perflibs_int_t n = 3;
  const perflibs_int_t nrhs = 17;
  const double alpha = 1.2;
  const perflibs_int_t lda_x =
      layout_x == PERFLIBS_COL_MAJOR ? n + 1 : nrhs + 2;
  const perflibs_int_t lda_y =
      layout_y == PERFLIBS_COL_MAJOR ? n + 2 : nrhs + 3;
  const perflibs_int_t row_ptr[] = {0, 1, 3, 6};
  const perflibs_int_t col_indx[] = {0, 0, 1, 0, 1, 2};
  const double vals[] = {2.0, 1.0, 3.0, -1.0, 2.0, 4.0};
  double X_init[85];
  double Y_storage[85];
  double Y_vals[51];
  double expected_X[51];

  for (perflibs_int_t col = 0; col < nrhs; ++col) {
    const double x0 = (double)(col + 1);
    const double x1 = x0 + 0.25;
    const double x2 = x0 + 0.5;
    expected_X[col * n] = x0;
    expected_X[col * n + 1] = x1;
    expected_X[col * n + 2] = x2;
    Y_vals[col * n] = 2.0 * x0 / alpha;
    Y_vals[col * n + 1] = (x0 + 3.0 * x1) / alpha;
    Y_vals[col * n + 2] = (-x0 + 2.0 * x1 + 4.0 * x2) / alpha;
  }

  fill_dense_storage(X_init, layout_x, n, nrhs, lda_x, (const double[51]){0.0},
                     -333.0);
  fill_dense_storage(Y_storage, layout_y, n, nrhs, lda_y, Y_vals, -222.0);

  perflibs_spmat_t A = NULL;
  perflibs_spmat_t X = NULL;
  perflibs_spmat_t Y = NULL;
  double *X_out = NULL;
  perflibs_int_t m = -1;
  perflibs_int_t out_n = -1;

  CHECK_STATUS(
      perflibs_spmat_create_csr_d(&A, n, n, row_ptr, col_indx, vals, 0));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_STRUCTURE,
                                   PERFLIBS_SPARSE_STRUCTURE_TRIANGULAR));
  CHECK_STATUS(perflibs_spmat_create_dense_d(
      &X, layout_x, n, nrhs, lda_x, X_init, PERFLIBS_SPARSE_CREATE_NOCOPY));
  CHECK_STATUS(perflibs_spmat_create_dense_d(
      &Y, layout_y, n, nrhs, lda_y, Y_storage, PERFLIBS_SPARSE_CREATE_NOCOPY));

  CHECK_STATUS(perflibs_spsm_optimize(PERFLIBS_SPARSE_OPERATION_NOTRANS, A, X,
                                      PERFLIBS_SPARSE_SCALAR_ANY, Y));
  CHECK_STATUS(
      perflibs_spsm_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS, A, X, alpha, Y));
  CHECK_STATUS(
      perflibs_spmat_export_dense_d(X, PERFLIBS_COL_MAJOR, &m, &out_n, &X_out));
  CHECK_TRUE(m == n && out_n == nrhs,
             "unexpected exported chunked SpSM solution shape %lld x %lld",
             test_i64(m), test_i64(out_n));
  CHECK_DOUBLE_ARRAY(X_out, expected_X, n * nrhs, 1e-12);

  free(X_out);
  CHECK_STATUS(perflibs_spmat_destroy(A));
  CHECK_STATUS(perflibs_spmat_destroy(X));
  CHECK_STATUS(perflibs_spmat_destroy(Y));
  return EXIT_SUCCESS;
}

int main() {
  CHECK_STATUS(test_basic_spsm_exec());
  CHECK_STATUS(
      test_basic_csr_spsm_exec(PERFLIBS_COL_MAJOR, PERFLIBS_COL_MAJOR));
  CHECK_STATUS(
      test_basic_csr_spsm_exec(PERFLIBS_ROW_MAJOR, PERFLIBS_COL_MAJOR));
  CHECK_STATUS(
      test_basic_csr_spsm_exec(PERFLIBS_COL_MAJOR, PERFLIBS_ROW_MAJOR));
  CHECK_STATUS(
      test_basic_csr_spsm_exec(PERFLIBS_ROW_MAJOR, PERFLIBS_ROW_MAJOR));
  CHECK_STATUS(test_zero_alpha_csr_spsm_exec());
  CHECK_STATUS(
      test_basic_csc_spsm_notrans_exec(PERFLIBS_COL_MAJOR, PERFLIBS_COL_MAJOR));
  CHECK_STATUS(
      test_basic_csc_spsm_notrans_exec(PERFLIBS_ROW_MAJOR, PERFLIBS_COL_MAJOR));
  CHECK_STATUS(
      test_basic_csc_spsm_notrans_exec(PERFLIBS_COL_MAJOR, PERFLIBS_ROW_MAJOR));
  CHECK_STATUS(
      test_basic_csc_spsm_notrans_exec(PERFLIBS_ROW_MAJOR, PERFLIBS_ROW_MAJOR));
  CHECK_STATUS(
      test_transpose_csr_spsm_exec(PERFLIBS_COL_MAJOR, PERFLIBS_COL_MAJOR));
  CHECK_STATUS(
      test_transpose_csr_spsm_exec(PERFLIBS_ROW_MAJOR, PERFLIBS_COL_MAJOR));
  CHECK_STATUS(
      test_transpose_csr_spsm_exec(PERFLIBS_COL_MAJOR, PERFLIBS_ROW_MAJOR));
  CHECK_STATUS(
      test_transpose_csr_spsm_exec(PERFLIBS_ROW_MAJOR, PERFLIBS_ROW_MAJOR));
  CHECK_STATUS(
      test_chunked_csr_spsm_exec(PERFLIBS_COL_MAJOR, PERFLIBS_COL_MAJOR));
  CHECK_STATUS(
      test_chunked_csr_spsm_exec(PERFLIBS_ROW_MAJOR, PERFLIBS_COL_MAJOR));
  CHECK_STATUS(
      test_chunked_csr_spsm_exec(PERFLIBS_COL_MAJOR, PERFLIBS_ROW_MAJOR));
  CHECK_STATUS(
      test_chunked_csr_spsm_exec(PERFLIBS_ROW_MAJOR, PERFLIBS_ROW_MAJOR));
  return EXIT_SUCCESS;
}
