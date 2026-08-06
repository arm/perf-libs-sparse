/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "perflibs_sparse.h"
#include "test_utils.h"

#include <stdlib.h>

static void fill_dense_storage(perflibs_singlecomplex_t *storage,
                               enum perflibs_dense_layout layout,
                               perflibs_int_t m, perflibs_int_t n,
                               perflibs_int_t lda,
                               const perflibs_singlecomplex_t *vals_col_major,
                               perflibs_singlecomplex_t pad_value) {
  const perflibs_int_t size = layout == PERFLIBS_COL_MAJOR ? lda * n : m * lda;
  for (perflibs_int_t i = 0; i < size; ++i) {
    storage[i] = pad_value;
  }

  for (perflibs_int_t col = 0; col < n; ++col) {
    for (perflibs_int_t row = 0; row < m; ++row) {
      const perflibs_singlecomplex_t value = vals_col_major[col * m + row];
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
  const perflibs_int_t n = 6;
  const perflibs_int_t nrhs = 2;
  const perflibs_singlecomplex_t alpha = TEST_C(1.0, 0.0);
  const perflibs_int_t lda_x =
      layout_x == PERFLIBS_COL_MAJOR ? n + 1 : nrhs + 2;
  const perflibs_int_t lda_y =
      layout_y == PERFLIBS_COL_MAJOR ? n + 2 : nrhs + 1;
  const perflibs_singlecomplex_t A_vals[] = {
      TEST_C(1.0, 0.0),  TEST_C(1.0, 1.0),  TEST_C(0.0, 0.0), TEST_C(0.0, 0.0),
      TEST_C(0.0, 0.0),  TEST_C(0.0, 0.0),  TEST_C(0.0, 0.0), TEST_C(1.0, 0.0),
      TEST_C(-2.0, 1.0), TEST_C(0.0, 0.0),  TEST_C(0.0, 0.0), TEST_C(0.0, 0.0),
      TEST_C(0.0, 0.0),  TEST_C(0.0, 0.0),  TEST_C(1.0, 0.0), TEST_C(1.0, -1.0),
      TEST_C(0.0, 0.0),  TEST_C(0.0, 0.0),  TEST_C(0.0, 0.0), TEST_C(0.0, 0.0),
      TEST_C(0.0, 0.0),  TEST_C(1.0, 0.0),  TEST_C(2.0, 0.0), TEST_C(0.0, 0.0),
      TEST_C(0.0, 0.0),  TEST_C(0.0, 0.0),  TEST_C(0.0, 0.0), TEST_C(0.0, 0.0),
      TEST_C(1.0, 0.0),  TEST_C(-1.0, 2.0), TEST_C(0.0, 0.0), TEST_C(0.0, 0.0),
      TEST_C(0.0, 0.0),  TEST_C(0.0, 0.0),  TEST_C(0.0, 0.0), TEST_C(1.0, 0.0)};
  const perflibs_singlecomplex_t Y_vals[] = {
      TEST_C(1.0, 0.0), TEST_C(2.0, 1.0), TEST_C(-1.0, 1.0), TEST_C(2.0, -1.0),
      TEST_C(3.0, 0.0), TEST_C(0.0, 2.0), TEST_C(1.0, 0.0),  TEST_C(0.0, 1.0),
      TEST_C(2.0, 0.0), TEST_C(1.0, 0.0), TEST_C(2.0, -2.0), TEST_C(-4.0, 4.0)};
  const perflibs_singlecomplex_t expected_X[] = {
      TEST_C(1.0, 0.0), TEST_C(1.0, 0.0),  TEST_C(1.0, 0.0), TEST_C(1.0, 0.0),
      TEST_C(1.0, 0.0), TEST_C(1.0, 0.0),  TEST_C(1.0, 0.0), TEST_C(-1.0, 0.0),
      TEST_C(0.0, 1.0), TEST_C(0.0, -1.0), TEST_C(2.0, 0.0), TEST_C(-2.0, 0.0)};
  const perflibs_singlecomplex_t X_zero[12] = {0};
  perflibs_singlecomplex_t X_init[36];
  perflibs_singlecomplex_t Y_storage[36];

  perflibs_spmat_t A = NULL;
  perflibs_spmat_t X = NULL;
  perflibs_spmat_t Y = NULL;
  perflibs_singlecomplex_t *X_out = NULL;
  perflibs_int_t m = -1;
  perflibs_int_t out_n = -1;

  fill_dense_storage(X_init, layout_x, n, nrhs, lda_x, X_zero,
                     TEST_C(-7.0, 3.0));
  fill_dense_storage(Y_storage, layout_y, n, nrhs, lda_y, Y_vals,
                     TEST_C(-5.0, 2.0));

  CHECK_STATUS(perflibs_spmat_create_dense_c(&A, PERFLIBS_COL_MAJOR, n, n, n,
                                             A_vals, 0));
  CHECK_STATUS(perflibs_spmat_create_dense_c(
      &X, layout_x, n, nrhs, lda_x, X_init, PERFLIBS_SPARSE_CREATE_NOCOPY));
  CHECK_STATUS(perflibs_spmat_create_dense_c(
      &Y, layout_y, n, nrhs, lda_y, Y_storage, PERFLIBS_SPARSE_CREATE_NOCOPY));

  CHECK_STATUS(perflibs_spsm_optimize(PERFLIBS_SPARSE_OPERATION_NOTRANS, A, X,
                                      PERFLIBS_SPARSE_SCALAR_ANY, Y));
  CHECK_STATUS(
      perflibs_spsm_exec_c(PERFLIBS_SPARSE_OPERATION_NOTRANS, A, X, alpha, Y));

  CHECK_STATUS(
      perflibs_spmat_export_dense_c(X, PERFLIBS_COL_MAJOR, &m, &out_n, &X_out));
  CHECK_TRUE(m == n && out_n == nrhs,
             "unexpected exported dense-layout SpSM solution shape %lld x %lld",
             test_i64(m), test_i64(out_n));
  CHECK_SINGLECOMPLEX_ARRAY(X_out, expected_X, n * nrhs, 2e-5f);

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
  const perflibs_int_t n = 6;
  const perflibs_int_t nrhs = 3;
  const perflibs_singlecomplex_t alpha = TEST_C(1.0, 0.0);
  const perflibs_int_t lda_x =
      layout_x == PERFLIBS_COL_MAJOR ? n + 1 : nrhs + 2;
  const perflibs_int_t lda_y =
      layout_y == PERFLIBS_COL_MAJOR ? n + 2 : nrhs + 1;
  const perflibs_int_t row_ptr[] = {0, 1, 3, 5, 7, 9, 11};
  const perflibs_int_t col_indx[] = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5};
  const perflibs_singlecomplex_t vals[] = {
      TEST_C(1.0, 0.0), TEST_C(1.0, 1.0),  TEST_C(1.0, 0.0), TEST_C(-2.0, 1.0),
      TEST_C(1.0, 0.0), TEST_C(1.0, -1.0), TEST_C(1.0, 0.0), TEST_C(2.0, 0.0),
      TEST_C(1.0, 0.0), TEST_C(-1.0, 2.0), TEST_C(1.0, 0.0)};
  const perflibs_singlecomplex_t Y_vals[] = {
      TEST_C(1.0, 0.0), TEST_C(2.0, 1.0), TEST_C(-1.0, 1.0), TEST_C(2.0, -1.0),
      TEST_C(3.0, 0.0), TEST_C(0.0, 2.0), TEST_C(1.0, 0.0),  TEST_C(0.0, 1.0),
      TEST_C(2.0, 0.0), TEST_C(1.0, 0.0), TEST_C(2.0, -2.0), TEST_C(-4.0, 4.0),
      TEST_C(0.0, 0.0), TEST_C(2.0, 0.0), TEST_C(-3.0, 1.0), TEST_C(3.0, -2.0),
      TEST_C(5.0, 0.0), TEST_C(1.0, -1.0)};
  const perflibs_singlecomplex_t expected_X[] = {
      TEST_C(1.0, 0.0),  TEST_C(1.0, 0.0),  TEST_C(1.0, 0.0),
      TEST_C(1.0, 0.0),  TEST_C(1.0, 0.0),  TEST_C(1.0, 0.0),
      TEST_C(1.0, 0.0),  TEST_C(-1.0, 0.0), TEST_C(0.0, 1.0),
      TEST_C(0.0, -1.0), TEST_C(2.0, 0.0),  TEST_C(-2.0, 0.0),
      TEST_C(0.0, 0.0),  TEST_C(2.0, 0.0),  TEST_C(1.0, -1.0),
      TEST_C(3.0, 0.0),  TEST_C(-1.0, 0.0), TEST_C(0.0, 1.0)};
  const perflibs_singlecomplex_t X_zero[18] = {0};
  perflibs_singlecomplex_t X_init[36];
  perflibs_singlecomplex_t Y_storage[36];

  perflibs_spmat_t A = NULL;
  perflibs_spmat_t X = NULL;
  perflibs_spmat_t Y = NULL;
  perflibs_singlecomplex_t *X_out = NULL;
  perflibs_int_t m = -1;
  perflibs_int_t out_n = -1;

  fill_dense_storage(X_init, layout_x, n, nrhs, lda_x, X_zero,
                     TEST_C(-3.0, -3.0));
  fill_dense_storage(Y_storage, layout_y, n, nrhs, lda_y, Y_vals,
                     TEST_C(-2.0, 4.0));

  CHECK_STATUS(
      perflibs_spmat_create_csr_c(&A, n, n, row_ptr, col_indx, vals, 0));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_STRUCTURE,
                                   PERFLIBS_SPARSE_STRUCTURE_TRIANGULAR));
  CHECK_STATUS(perflibs_spmat_create_dense_c(
      &X, layout_x, n, nrhs, lda_x, X_init, PERFLIBS_SPARSE_CREATE_NOCOPY));
  CHECK_STATUS(perflibs_spmat_create_dense_c(
      &Y, layout_y, n, nrhs, lda_y, Y_storage, PERFLIBS_SPARSE_CREATE_NOCOPY));

  CHECK_STATUS(perflibs_spsm_optimize(PERFLIBS_SPARSE_OPERATION_NOTRANS, A, X,
                                      PERFLIBS_SPARSE_SCALAR_ANY, Y));
  CHECK_STATUS(
      perflibs_spsm_exec_c(PERFLIBS_SPARSE_OPERATION_NOTRANS, A, X, alpha, Y));

  CHECK_STATUS(
      perflibs_spmat_export_dense_c(X, PERFLIBS_COL_MAJOR, &m, &out_n, &X_out));
  CHECK_TRUE(m == n && out_n == nrhs,
             "unexpected exported CSR-layout SpSM solution shape %lld x %lld",
             test_i64(m), test_i64(out_n));
  CHECK_SINGLECOMPLEX_ARRAY(X_out, expected_X, n * nrhs, 2e-5f);

  free(X_out);
  CHECK_STATUS(perflibs_spmat_destroy(A));
  CHECK_STATUS(perflibs_spmat_destroy(X));
  CHECK_STATUS(perflibs_spmat_destroy(Y));
  return EXIT_SUCCESS;
}

static int
test_basic_csc_spsm_notrans_exec(enum perflibs_dense_layout layout_x,
                                 enum perflibs_dense_layout layout_y) {
  const perflibs_int_t n = 6;
  const perflibs_int_t nrhs = 3;
  const perflibs_singlecomplex_t alpha = TEST_C(1.0, 0.0);
  const perflibs_int_t lda_x =
      layout_x == PERFLIBS_COL_MAJOR ? n + 1 : nrhs + 2;
  const perflibs_int_t lda_y =
      layout_y == PERFLIBS_COL_MAJOR ? n + 2 : nrhs + 1;
  const perflibs_int_t col_ptr[] = {0, 2, 4, 6, 8, 10, 11};
  const perflibs_int_t row_indx[] = {0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
  const perflibs_singlecomplex_t vals[] = {
      TEST_C(1.0, 0.0), TEST_C(1.0, 1.0),  TEST_C(1.0, 0.0), TEST_C(-2.0, 1.0),
      TEST_C(1.0, 0.0), TEST_C(1.0, -1.0), TEST_C(1.0, 0.0), TEST_C(2.0, 0.0),
      TEST_C(1.0, 0.0), TEST_C(-1.0, 2.0), TEST_C(1.0, 0.0)};
  const perflibs_singlecomplex_t Y_vals[] = {
      TEST_C(1.0, 0.0), TEST_C(2.0, 1.0), TEST_C(-1.0, 1.0), TEST_C(2.0, -1.0),
      TEST_C(3.0, 0.0), TEST_C(0.0, 2.0), TEST_C(1.0, 0.0),  TEST_C(0.0, 1.0),
      TEST_C(2.0, 0.0), TEST_C(1.0, 0.0), TEST_C(2.0, -2.0), TEST_C(-4.0, 4.0),
      TEST_C(0.0, 0.0), TEST_C(2.0, 0.0), TEST_C(-3.0, 1.0), TEST_C(3.0, -2.0),
      TEST_C(5.0, 0.0), TEST_C(1.0, -1.0)};
  const perflibs_singlecomplex_t expected_X[] = {
      TEST_C(1.0, 0.0),  TEST_C(1.0, 0.0),  TEST_C(1.0, 0.0),
      TEST_C(1.0, 0.0),  TEST_C(1.0, 0.0),  TEST_C(1.0, 0.0),
      TEST_C(1.0, 0.0),  TEST_C(-1.0, 0.0), TEST_C(0.0, 1.0),
      TEST_C(0.0, -1.0), TEST_C(2.0, 0.0),  TEST_C(-2.0, 0.0),
      TEST_C(0.0, 0.0),  TEST_C(2.0, 0.0),  TEST_C(1.0, -1.0),
      TEST_C(3.0, 0.0),  TEST_C(-1.0, 0.0), TEST_C(0.0, 1.0)};
  const perflibs_singlecomplex_t X_zero[18] = {0};
  perflibs_singlecomplex_t X_init[36];
  perflibs_singlecomplex_t Y_storage[36];

  perflibs_spmat_t A = NULL;
  perflibs_spmat_t X = NULL;
  perflibs_spmat_t Y = NULL;
  perflibs_singlecomplex_t *X_out = NULL;
  perflibs_int_t m = -1;
  perflibs_int_t out_n = -1;

  fill_dense_storage(X_init, layout_x, n, nrhs, lda_x, X_zero,
                     TEST_C(-3.0, -3.0));
  fill_dense_storage(Y_storage, layout_y, n, nrhs, lda_y, Y_vals,
                     TEST_C(-2.0, 4.0));

  CHECK_STATUS(
      perflibs_spmat_create_csc_c(&A, n, n, row_indx, col_ptr, vals, 0));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_STRUCTURE,
                                   PERFLIBS_SPARSE_STRUCTURE_TRIANGULAR));
  CHECK_STATUS(perflibs_spmat_create_dense_c(
      &X, layout_x, n, nrhs, lda_x, X_init, PERFLIBS_SPARSE_CREATE_NOCOPY));
  CHECK_STATUS(perflibs_spmat_create_dense_c(
      &Y, layout_y, n, nrhs, lda_y, Y_storage, PERFLIBS_SPARSE_CREATE_NOCOPY));

  CHECK_STATUS(perflibs_spsm_optimize(PERFLIBS_SPARSE_OPERATION_NOTRANS, A, X,
                                      PERFLIBS_SPARSE_SCALAR_ANY, Y));
  CHECK_STATUS(
      perflibs_spsm_exec_c(PERFLIBS_SPARSE_OPERATION_NOTRANS, A, X, alpha, Y));

  CHECK_STATUS(
      perflibs_spmat_export_dense_c(X, PERFLIBS_COL_MAJOR, &m, &out_n, &X_out));
  CHECK_TRUE(m == n && out_n == nrhs,
             "unexpected exported CSC-notrans SpSM solution shape %lld x %lld",
             test_i64(m), test_i64(out_n));
  CHECK_SINGLECOMPLEX_ARRAY(X_out, expected_X, n * nrhs, 2e-5f);

  free(X_out);
  CHECK_STATUS(perflibs_spmat_destroy(A));
  CHECK_STATUS(perflibs_spmat_destroy(X));
  CHECK_STATUS(perflibs_spmat_destroy(Y));
  return EXIT_SUCCESS;
}

static int test_conjtrans_csr_spsm_exec(enum perflibs_dense_layout layout_x,
                                        enum perflibs_dense_layout layout_y) {
  const perflibs_int_t n = 6;
  const perflibs_int_t nrhs = 2;
  const perflibs_singlecomplex_t alpha = TEST_C(1.0, 0.0);
  const perflibs_int_t lda_x =
      layout_x == PERFLIBS_COL_MAJOR ? n + 1 : nrhs + 2;
  const perflibs_int_t lda_y =
      layout_y == PERFLIBS_COL_MAJOR ? n + 2 : nrhs + 1;
  const perflibs_int_t row_ptr[] = {0, 1, 3, 5, 7, 9, 11};
  const perflibs_int_t col_indx[] = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5};
  const perflibs_singlecomplex_t vals[] = {
      TEST_C(1.0, 0.0), TEST_C(1.0, 1.0),  TEST_C(1.0, 0.0), TEST_C(-2.0, 1.0),
      TEST_C(1.0, 0.0), TEST_C(1.0, -1.0), TEST_C(1.0, 0.0), TEST_C(2.0, 0.0),
      TEST_C(1.0, 0.0), TEST_C(-1.0, 2.0), TEST_C(1.0, 0.0)};
  const perflibs_singlecomplex_t Y_vals[] = {
      TEST_C(2.0, -1.0), TEST_C(-1.0, -1.0), TEST_C(2.0, 1.0),
      TEST_C(3.0, 0.0),  TEST_C(0.0, -2.0),  TEST_C(1.0, 0.0),
      TEST_C(0.0, 1.0),  TEST_C(0.0, -2.0),  TEST_C(1.0, 0.0),
      TEST_C(4.0, -1.0), TEST_C(4.0, 4.0),   TEST_C(-2.0, 0.0)};
  const perflibs_singlecomplex_t expected_X[] = {
      TEST_C(1.0, 0.0), TEST_C(1.0, 0.0),  TEST_C(1.0, 0.0), TEST_C(1.0, 0.0),
      TEST_C(1.0, 0.0), TEST_C(1.0, 0.0),  TEST_C(1.0, 0.0), TEST_C(-1.0, 0.0),
      TEST_C(0.0, 1.0), TEST_C(0.0, -1.0), TEST_C(2.0, 0.0), TEST_C(-2.0, 0.0)};
  const perflibs_singlecomplex_t X_zero[12] = {0};
  perflibs_singlecomplex_t X_init[36];
  perflibs_singlecomplex_t Y_storage[36];

  perflibs_spmat_t A = NULL;
  perflibs_spmat_t X = NULL;
  perflibs_spmat_t Y = NULL;
  perflibs_singlecomplex_t *X_out = NULL;
  perflibs_int_t m = -1;
  perflibs_int_t out_n = -1;

  fill_dense_storage(X_init, layout_x, n, nrhs, lda_x, X_zero,
                     TEST_C(-1.0, -4.0));
  fill_dense_storage(Y_storage, layout_y, n, nrhs, lda_y, Y_vals,
                     TEST_C(-9.0, 1.0));

  CHECK_STATUS(
      perflibs_spmat_create_csr_c(&A, n, n, row_ptr, col_indx, vals, 0));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_STRUCTURE,
                                   PERFLIBS_SPARSE_STRUCTURE_TRIANGULAR));
  CHECK_STATUS(perflibs_spmat_create_dense_c(
      &X, layout_x, n, nrhs, lda_x, X_init, PERFLIBS_SPARSE_CREATE_NOCOPY));
  CHECK_STATUS(perflibs_spmat_create_dense_c(
      &Y, layout_y, n, nrhs, lda_y, Y_storage, PERFLIBS_SPARSE_CREATE_NOCOPY));

  CHECK_STATUS(perflibs_spsm_optimize(PERFLIBS_SPARSE_OPERATION_CONJTRANS, A, X,
                                      PERFLIBS_SPARSE_SCALAR_ANY, Y));
  CHECK_STATUS(perflibs_spsm_exec_c(PERFLIBS_SPARSE_OPERATION_CONJTRANS, A, X,
                                    alpha, Y));

  CHECK_STATUS(
      perflibs_spmat_export_dense_c(X, PERFLIBS_COL_MAJOR, &m, &out_n, &X_out));
  CHECK_TRUE(
      m == n && out_n == nrhs,
      "unexpected exported conj-transpose CSR-layout SpSM solution shape "
      "%lld x %lld",
      test_i64(m), test_i64(out_n));
  CHECK_SINGLECOMPLEX_ARRAY(X_out, expected_X, n * nrhs, 2e-5f);

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
      test_conjtrans_csr_spsm_exec(PERFLIBS_COL_MAJOR, PERFLIBS_COL_MAJOR));
  CHECK_STATUS(
      test_conjtrans_csr_spsm_exec(PERFLIBS_ROW_MAJOR, PERFLIBS_COL_MAJOR));
  CHECK_STATUS(
      test_conjtrans_csr_spsm_exec(PERFLIBS_COL_MAJOR, PERFLIBS_ROW_MAJOR));
  CHECK_STATUS(
      test_conjtrans_csr_spsm_exec(PERFLIBS_ROW_MAJOR, PERFLIBS_ROW_MAJOR));
  return EXIT_SUCCESS;
}
