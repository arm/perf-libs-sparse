/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "perflibs_sparse.h"
#include "test_utils.h"

#include <stdlib.h>

static void fill_dense_storage(float *storage,
                               enum perflibs_dense_layout layout,
                               perflibs_int_t m, perflibs_int_t n,
                               perflibs_int_t lda, const float *vals_col_major,
                               float pad_value) {
  const perflibs_int_t size = layout == PERFLIBS_COL_MAJOR ? lda * n : m * lda;
  for (perflibs_int_t i = 0; i < size; ++i) {
    storage[i] = pad_value;
  }

  for (perflibs_int_t col = 0; col < n; ++col) {
    for (perflibs_int_t row = 0; row < m; ++row) {
      const float value = vals_col_major[col * m + row];
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
  const perflibs_int_t n = 4;
  const perflibs_int_t nrhs = 3;
  const float alpha = 1.0f;
  const perflibs_int_t lda_x =
      layout_x == PERFLIBS_COL_MAJOR ? n + 1 : nrhs + 2;
  const perflibs_int_t lda_y =
      layout_y == PERFLIBS_COL_MAJOR ? n + 2 : nrhs + 1;
  const float A_vals[] = {1.0f, 2.0f, 0.0f, 0.0f, 0.0f, 1.0f, -1.0f, 0.0f,
                          0.0f, 0.0f, 1.0f, 3.0f, 0.0f, 0.0f, 0.0f,  1.0f};
  const float Y_vals[] = {1.0f, 4.0f, 1.0f, 13.0f, -1.0f, -2.0f,
                          2.0f, 7.0f, 2.0f, 1.0f,  4.0f,  3.0f};
  const float expected_X[] = {1.0f, 2.0f, 3.0f, 4.0f,  -1.0f, 0.0f,
                              2.0f, 1.0f, 2.0f, -3.0f, 1.0f,  0.0f};
  const float X_zero[12] = {0.0f};
  float X_init[20];
  float Y_storage[20];

  perflibs_spmat_t A = NULL;
  perflibs_spmat_t X = NULL;
  perflibs_spmat_t Y = NULL;
  float *X_out = NULL;
  perflibs_int_t m = -1;
  perflibs_int_t out_n = -1;

  fill_dense_storage(X_init, layout_x, n, nrhs, lda_x, X_zero, -777.0f);
  fill_dense_storage(Y_storage, layout_y, n, nrhs, lda_y, Y_vals, -555.0f);

  CHECK_STATUS(perflibs_spmat_create_dense_s(&A, PERFLIBS_COL_MAJOR, n, n, n,
                                             A_vals, 0));
  CHECK_STATUS(perflibs_spmat_create_dense_s(
      &X, layout_x, n, nrhs, lda_x, X_init, PERFLIBS_SPARSE_CREATE_NOCOPY));
  CHECK_STATUS(perflibs_spmat_create_dense_s(
      &Y, layout_y, n, nrhs, lda_y, Y_storage, PERFLIBS_SPARSE_CREATE_NOCOPY));

  CHECK_STATUS(perflibs_spsm_optimize(PERFLIBS_SPARSE_OPERATION_NOTRANS, A, X,
                                      PERFLIBS_SPARSE_SCALAR_ANY, Y));
  CHECK_STATUS(
      perflibs_spsm_exec_s(PERFLIBS_SPARSE_OPERATION_NOTRANS, A, X, alpha, Y));

  CHECK_STATUS(
      perflibs_spmat_export_dense_s(X, PERFLIBS_COL_MAJOR, &m, &out_n, &X_out));
  CHECK_TRUE(m == n && out_n == nrhs,
             "unexpected exported dense-layout SpSM solution shape %lld x %lld",
             test_i64(m), test_i64(out_n));
  CHECK_FLOAT_ARRAY(X_out, expected_X, n * nrhs, 5e-5f);

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
  const perflibs_int_t n = 4;
  const perflibs_int_t nrhs = 3;
  const float alpha = 1.0f;
  const perflibs_int_t lda_x =
      layout_x == PERFLIBS_COL_MAJOR ? n + 1 : nrhs + 2;
  const perflibs_int_t lda_y =
      layout_y == PERFLIBS_COL_MAJOR ? n + 2 : nrhs + 1;
  const perflibs_int_t row_ptr[] = {0, 1, 3, 5, 7};
  const perflibs_int_t col_indx[] = {0, 0, 1, 1, 2, 2, 3};
  const float vals[] = {1.0f, 2.0f, 1.0f, -1.0f, 1.0f, 3.0f, 1.0f};
  const float Y_vals[] = {1.0f, 4.0f, 1.0f, 13.0f, -1.0f, -2.0f,
                          2.0f, 7.0f, 2.0f, 1.0f,  4.0f,  3.0f};
  const float expected_X[] = {1.0f, 2.0f, 3.0f, 4.0f,  -1.0f, 0.0f,
                              2.0f, 1.0f, 2.0f, -3.0f, 1.0f,  0.0f};
  const float X_zero[12] = {0.0f};
  float X_init[20];
  float Y_storage[20];

  perflibs_spmat_t A = NULL;
  perflibs_spmat_t X = NULL;
  perflibs_spmat_t Y = NULL;
  float *X_out = NULL;
  perflibs_int_t m = -1;
  perflibs_int_t out_n = -1;

  fill_dense_storage(X_init, layout_x, n, nrhs, lda_x, X_zero, -333.0f);
  fill_dense_storage(Y_storage, layout_y, n, nrhs, lda_y, Y_vals, -222.0f);

  CHECK_STATUS(
      perflibs_spmat_create_csr_s(&A, n, n, row_ptr, col_indx, vals, 0));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_STRUCTURE,
                                   PERFLIBS_SPARSE_STRUCTURE_TRIANGULAR));
  CHECK_STATUS(perflibs_spmat_create_dense_s(
      &X, layout_x, n, nrhs, lda_x, X_init, PERFLIBS_SPARSE_CREATE_NOCOPY));
  CHECK_STATUS(perflibs_spmat_create_dense_s(
      &Y, layout_y, n, nrhs, lda_y, Y_storage, PERFLIBS_SPARSE_CREATE_NOCOPY));

  CHECK_STATUS(perflibs_spsm_optimize(PERFLIBS_SPARSE_OPERATION_NOTRANS, A, X,
                                      PERFLIBS_SPARSE_SCALAR_ANY, Y));
  CHECK_STATUS(
      perflibs_spsm_exec_s(PERFLIBS_SPARSE_OPERATION_NOTRANS, A, X, alpha, Y));

  CHECK_STATUS(
      perflibs_spmat_export_dense_s(X, PERFLIBS_COL_MAJOR, &m, &out_n, &X_out));
  CHECK_TRUE(m == n && out_n == nrhs,
             "unexpected exported CSR-layout SpSM solution shape %lld x %lld",
             test_i64(m), test_i64(out_n));
  CHECK_FLOAT_ARRAY(X_out, expected_X, n * nrhs, 5e-5f);

  free(X_out);
  CHECK_STATUS(perflibs_spmat_destroy(A));
  CHECK_STATUS(perflibs_spmat_destroy(X));
  CHECK_STATUS(perflibs_spmat_destroy(Y));
  return EXIT_SUCCESS;
}

static int
test_basic_csc_spsm_notrans_exec(enum perflibs_dense_layout layout_x,
                                 enum perflibs_dense_layout layout_y) {
  const perflibs_int_t n = 4;
  const perflibs_int_t nrhs = 3;
  const float alpha = 1.0f;
  const perflibs_int_t lda_x =
      layout_x == PERFLIBS_COL_MAJOR ? n + 1 : nrhs + 2;
  const perflibs_int_t lda_y =
      layout_y == PERFLIBS_COL_MAJOR ? n + 2 : nrhs + 1;
  const perflibs_int_t col_ptr[] = {0, 2, 4, 6, 7};
  const perflibs_int_t row_indx[] = {0, 1, 1, 2, 2, 3, 3};
  const float vals[] = {1.0f, 2.0f, 1.0f, -1.0f, 1.0f, 3.0f, 1.0f};
  const float Y_vals[] = {1.0f, 4.0f, 1.0f, 13.0f, -1.0f, -2.0f,
                          2.0f, 7.0f, 2.0f, 1.0f,  4.0f,  3.0f};
  const float expected_X[] = {1.0f, 2.0f, 3.0f, 4.0f,  -1.0f, 0.0f,
                              2.0f, 1.0f, 2.0f, -3.0f, 1.0f,  0.0f};
  const float X_zero[12] = {0.0f};
  float X_init[20];
  float Y_storage[20];

  perflibs_spmat_t A = NULL;
  perflibs_spmat_t X = NULL;
  perflibs_spmat_t Y = NULL;
  float *X_out = NULL;
  perflibs_int_t m = -1;
  perflibs_int_t out_n = -1;

  fill_dense_storage(X_init, layout_x, n, nrhs, lda_x, X_zero, -333.0f);
  fill_dense_storage(Y_storage, layout_y, n, nrhs, lda_y, Y_vals, -222.0f);

  CHECK_STATUS(
      perflibs_spmat_create_csc_s(&A, n, n, row_indx, col_ptr, vals, 0));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_STRUCTURE,
                                   PERFLIBS_SPARSE_STRUCTURE_TRIANGULAR));
  CHECK_STATUS(perflibs_spmat_create_dense_s(
      &X, layout_x, n, nrhs, lda_x, X_init, PERFLIBS_SPARSE_CREATE_NOCOPY));
  CHECK_STATUS(perflibs_spmat_create_dense_s(
      &Y, layout_y, n, nrhs, lda_y, Y_storage, PERFLIBS_SPARSE_CREATE_NOCOPY));

  CHECK_STATUS(perflibs_spsm_optimize(PERFLIBS_SPARSE_OPERATION_NOTRANS, A, X,
                                      PERFLIBS_SPARSE_SCALAR_ANY, Y));
  CHECK_STATUS(
      perflibs_spsm_exec_s(PERFLIBS_SPARSE_OPERATION_NOTRANS, A, X, alpha, Y));

  CHECK_STATUS(
      perflibs_spmat_export_dense_s(X, PERFLIBS_COL_MAJOR, &m, &out_n, &X_out));
  CHECK_TRUE(m == n && out_n == nrhs,
             "unexpected exported CSC-notrans SpSM solution shape %lld x %lld",
             test_i64(m), test_i64(out_n));
  CHECK_FLOAT_ARRAY(X_out, expected_X, n * nrhs, 5e-5f);

  free(X_out);
  CHECK_STATUS(perflibs_spmat_destroy(A));
  CHECK_STATUS(perflibs_spmat_destroy(X));
  CHECK_STATUS(perflibs_spmat_destroy(Y));
  return EXIT_SUCCESS;
}

static int test_transpose_csr_spsm_exec(enum perflibs_dense_layout layout_x,
                                        enum perflibs_dense_layout layout_y) {
  const perflibs_int_t n = 4;
  const perflibs_int_t nrhs = 2;
  const float alpha = 1.0f;
  const perflibs_int_t lda_x =
      layout_x == PERFLIBS_COL_MAJOR ? n + 1 : nrhs + 2;
  const perflibs_int_t lda_y =
      layout_y == PERFLIBS_COL_MAJOR ? n + 2 : nrhs + 1;
  const perflibs_int_t row_ptr[] = {0, 1, 3, 5, 7};
  const perflibs_int_t col_indx[] = {0, 0, 1, 1, 2, 2, 3};
  const float vals[] = {1.0f, 2.0f, 1.0f, -1.0f, 1.0f, 3.0f, 1.0f};
  const float Y_vals[] = {5.0f, -1.0f, 15.0f, 4.0f, -1.0f, -2.0f, 5.0f, 1.0f};
  const float expected_X[] = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, 0.0f, 2.0f, 1.0f};
  const float X_zero[8] = {0.0f};
  float X_init[20];
  float Y_storage[20];

  perflibs_spmat_t A = NULL;
  perflibs_spmat_t X = NULL;
  perflibs_spmat_t Y = NULL;
  float *X_out = NULL;
  perflibs_int_t m = -1;
  perflibs_int_t out_n = -1;

  fill_dense_storage(X_init, layout_x, n, nrhs, lda_x, X_zero, -111.0f);
  fill_dense_storage(Y_storage, layout_y, n, nrhs, lda_y, Y_vals, -999.0f);

  CHECK_STATUS(
      perflibs_spmat_create_csr_s(&A, n, n, row_ptr, col_indx, vals, 0));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_STRUCTURE,
                                   PERFLIBS_SPARSE_STRUCTURE_TRIANGULAR));
  CHECK_STATUS(perflibs_spmat_create_dense_s(
      &X, layout_x, n, nrhs, lda_x, X_init, PERFLIBS_SPARSE_CREATE_NOCOPY));
  CHECK_STATUS(perflibs_spmat_create_dense_s(
      &Y, layout_y, n, nrhs, lda_y, Y_storage, PERFLIBS_SPARSE_CREATE_NOCOPY));

  CHECK_STATUS(perflibs_spsm_optimize(PERFLIBS_SPARSE_OPERATION_TRANS, A, X,
                                      PERFLIBS_SPARSE_SCALAR_ANY, Y));
  CHECK_STATUS(
      perflibs_spsm_exec_s(PERFLIBS_SPARSE_OPERATION_TRANS, A, X, alpha, Y));

  CHECK_STATUS(
      perflibs_spmat_export_dense_s(X, PERFLIBS_COL_MAJOR, &m, &out_n, &X_out));
  CHECK_TRUE(
      m == n && out_n == nrhs,
      "unexpected exported transpose CSR-layout SpSM solution shape %lld "
      "x %lld",
      test_i64(m), test_i64(out_n));
  CHECK_FLOAT_ARRAY(X_out, expected_X, n * nrhs, 5e-5f);

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
  return EXIT_SUCCESS;
}
