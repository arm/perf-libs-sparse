/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "perflibs_sparse.h"
#include "perflibs_sparse_reproducibility.h"
#include "test_utils.h"

#include <stdlib.h>
#include <string.h>

/* These public-API cases test repeated execution of the same CSC object with
 * identical inputs and reset output values. Numerical expectations check the
 * arithmetic, while memcmp checks bitwise repeatability. Fresh-object
 * optimization is covered separately in spmv_optimize_reproducibility.c.
 * Comparisons here use a fixed build, hardware and floating-point environment.
 *
 * The cancellation cases can expose an unordered accumulation path, but finite
 * repetitions cannot guarantee that every such regression will be detected.
 * Disabled mode is allowed to repeat; variation is not a test requirement. */

#define REPRO_COLS 4096
#define REPRO_REPEATS 16

/* CSC NOTRANS maps to transpose on a CSR view. Every column here updates y[0],
 * so the reproducible serial traversal fixes the order of all contributions.
 * In FP64 round-to-nearest, the four equal groups lose the units while the
 * partial sum is large, cancel the large values, then retain REPRO_COLS/4
 * quarters. The expected REPRO_COLS/16 identifies this accumulation order,
 * rather than the exact-real sum. */
static int test_csc_reproducible_notrans_real(void) {
  perflibs_int_t *col_ptr = malloc(sizeof(*col_ptr) * (REPRO_COLS + 1));
  perflibs_int_t *row_indx = malloc(sizeof(*row_indx) * REPRO_COLS);
  double *vals = malloc(sizeof(*vals) * REPRO_COLS);
  double *x = malloc(sizeof(*x) * REPRO_COLS);
  perflibs_spmat_t A = NULL;
  CHECK_TRUE(col_ptr && row_indx && vals && x,
             "allocation failure for reproducible CSC SpMV");

  for (perflibs_int_t col = 0; col < REPRO_COLS; ++col) {
    col_ptr[col] = col;
    row_indx[col] = 0;
    x[col] = 1.0;
    switch (col / (REPRO_COLS / 4)) {
    case 0:
      vals[col] = 0x1p60;
      break;
    case 1:
      vals[col] = 1.0;
      break;
    case 2:
      vals[col] = -0x1p60;
      break;
    default:
      vals[col] = 0.25;
      break;
    }
  }
  col_ptr[REPRO_COLS] = REPRO_COLS;

  CHECK_STATUS(perflibs_spmat_create_csc_d(&A, 1, REPRO_COLS, row_indx, col_ptr,
                                           vals, 0));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_SPMV_OPERATION,
                                   PERFLIBS_SPARSE_OPERATION_NOTRANS));
  CHECK_STATUS(perflibs_spmv_optimize(A));

  double first = 0.0;
  for (int repeat = 0; repeat < REPRO_REPEATS; ++repeat) {
    double y[] = {0.0};
    CHECK_STATUS(perflibs_spmv_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS, 1.0, A,
                                      x, 0.0, y));
    CHECK_TRUE(y[0] == REPRO_COLS / 16,
               "CSC non-transpose accumulation order changed: %.17g", y[0]);
    if (repeat > 0) {
      CHECK_TRUE(memcmp(&first, y, sizeof(first)) == 0,
                 "CSC non-transpose result changed on repeat %d", repeat);
    }
    first = y[0];
  }

  CHECK_STATUS(perflibs_spmat_destroy(A));
  free(col_ptr);
  free(row_indx);
  free(vals);
  free(x);
  return EXIT_SUCCESS;
}

/* Apply the same shared-output witness to complex FP64. The real part retains
 * only the final quarters, giving REPRO_COLS/16. The imaginary contributions
 * sum exactly to (0.5 - 0.25) * REPRO_COLS/4, also REPRO_COLS/16. Their
 * positive sign additionally checks that NOTRANS does not conjugate the matrix
 * values. */
static int test_csc_reproducible_notrans_complex(void) {
  perflibs_int_t *col_ptr = malloc(sizeof(*col_ptr) * (REPRO_COLS + 1));
  perflibs_int_t *row_indx = malloc(sizeof(*row_indx) * REPRO_COLS);
  perflibs_doublecomplex_t *vals = malloc(sizeof(*vals) * REPRO_COLS);
  perflibs_doublecomplex_t *x = malloc(sizeof(*x) * REPRO_COLS);
  perflibs_spmat_t A = NULL;
  CHECK_TRUE(col_ptr && row_indx && vals && x,
             "allocation failure for reproducible complex CSC SpMV");

  for (perflibs_int_t col = 0; col < REPRO_COLS; ++col) {
    col_ptr[col] = col;
    row_indx[col] = 0;
    x[col] = TEST_Z(1.0, 0.0);
    switch (col / (REPRO_COLS / 4)) {
    case 0:
      vals[col] = TEST_Z(0x1p60, 0.0);
      break;
    case 1:
      vals[col] = TEST_Z(1.0, 0.5);
      break;
    case 2:
      vals[col] = TEST_Z(-0x1p60, 0.0);
      break;
    default:
      vals[col] = TEST_Z(0.25, -0.25);
      break;
    }
  }
  col_ptr[REPRO_COLS] = REPRO_COLS;

  CHECK_STATUS(perflibs_spmat_create_csc_z(&A, 1, REPRO_COLS, row_indx, col_ptr,
                                           vals, 0));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_SPMV_OPERATION,
                                   PERFLIBS_SPARSE_OPERATION_NOTRANS));
  CHECK_STATUS(perflibs_spmv_optimize(A));

  perflibs_doublecomplex_t first = TEST_Z(0.0, 0.0);
  for (int repeat = 0; repeat < REPRO_REPEATS; ++repeat) {
    perflibs_doublecomplex_t y[] = {TEST_Z(0.0, 0.0)};
    CHECK_STATUS(perflibs_spmv_exec_z(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                      TEST_Z(1.0, 0.0), A, x, TEST_Z(0.0, 0.0),
                                      y));
    CHECK_DOUBLECOMPLEX_NEAR(y[0], TEST_Z(REPRO_COLS / 16, REPRO_COLS / 16),
                             0.0);
    if (repeat > 0) {
      CHECK_TRUE(memcmp(&first, y, sizeof(first)) == 0,
                 "complex CSC non-transpose result changed on repeat %d",
                 repeat);
    }
    first = y[0];
  }

  CHECK_STATUS(perflibs_spmat_destroy(A));
  free(col_ptr);
  free(row_indx);
  free(vals);
  free(x);
  return EXIT_SUCCESS;
}

/* CSC TRANS and CONJTRANS map to non-transposed CSR operations. Each original
 * column owns one output, so a fixed local accumulation order is independent
 * of scheduling between columns. Cover real transpose and complex conjugate
 * transpose with nontrivial alpha and beta. These small binary-fraction inputs
 * check correctness and repeatability, but may give the same bits with ordinary
 * kernels too; they are not numerical witnesses for the mode decision. */
static int test_csc_reproducible_row_owned(void) {
  const perflibs_int_t col_ptr[] = {0, 3, 6, 9};
  const perflibs_int_t row_indx[] = {0, 1, 3, 0, 2, 3, 1, 2, 3};
  const float vals_s[] = {0.5f,  1.0f,  1.5f,  2.0f, 0.25f,
                          -1.0f, 1.25f, 0.75f, 0.5f};
  const float x_s[] = {1.0f, 2.0f, -1.0f, 0.5f};
  const float initial_s[] = {2.0f, -4.0f, 1.0f};
  float expected_s[3];
  const perflibs_singlecomplex_t vals_c[] = {
      TEST_C(0.5, 0.25),  TEST_C(1.0, -0.5),  TEST_C(1.5, 0.0),
      TEST_C(2.0, -0.25), TEST_C(0.25, 0.5),  TEST_C(-1.0, 0.25),
      TEST_C(1.25, 0.25), TEST_C(0.75, -0.5), TEST_C(0.5, 1.0)};
  const perflibs_singlecomplex_t x_c[] = {TEST_C(1.0, 1.0), TEST_C(2.0, -1.0),
                                          TEST_C(-1.0, 0.5), TEST_C(0.5, 1.0)};
  const perflibs_singlecomplex_t initial_c[] = {
      TEST_C(2.0, -1.0), TEST_C(-4.0, 0.5), TEST_C(1.0, 2.0)};
  const perflibs_singlecomplex_t alpha_c = TEST_C(1.5, 0.5);
  const perflibs_singlecomplex_t beta_c = TEST_C(-0.5, 0.25);
  perflibs_singlecomplex_t expected_c[3];
  perflibs_spmat_t A_s = NULL, A_c = NULL;

  for (perflibs_int_t col = 0; col < 3; ++col) {
    float sum_s = 0.0f;
    perflibs_singlecomplex_t sum_c = TEST_C(0.0, 0.0);
    for (perflibs_int_t j = col_ptr[col]; j < col_ptr[col + 1]; ++j) {
      sum_s += vals_s[j] * x_s[row_indx[j]];
      sum_c = test_caddf(sum_c,
                         test_cmulf(test_conjf(vals_c[j]), x_c[row_indx[j]]));
    }
    expected_s[col] = -0.5f * initial_s[col] + 1.5f * sum_s;
    expected_c[col] = test_caddf(test_cmulf(beta_c, initial_c[col]),
                                 test_cmulf(alpha_c, sum_c));
  }

  CHECK_STATUS(
      perflibs_spmat_create_csc_s(&A_s, 4, 3, row_indx, col_ptr, vals_s, 0));
  CHECK_STATUS(
      perflibs_spmat_create_csc_c(&A_c, 4, 3, row_indx, col_ptr, vals_c, 0));
  CHECK_STATUS(perflibs_spmat_hint(A_s, PERFLIBS_SPARSE_HINT_SPMV_OPERATION,
                                   PERFLIBS_SPARSE_OPERATION_TRANS));
  CHECK_STATUS(perflibs_spmat_hint(A_c, PERFLIBS_SPARSE_HINT_SPMV_OPERATION,
                                   PERFLIBS_SPARSE_OPERATION_CONJTRANS));
  CHECK_STATUS(perflibs_spmv_optimize(A_s));
  CHECK_STATUS(perflibs_spmv_optimize(A_c));

  float first_s[3];
  perflibs_singlecomplex_t first_c[3];
  /* Restore initial y before each execution so beta scales the same input.
   * Reference values check arithmetic, including conjugation; the separate
   * memcmp checks below require exact repeatability of both output arrays. */
  for (int repeat = 0; repeat < REPRO_REPEATS; ++repeat) {
    float y_s[3];
    perflibs_singlecomplex_t y_c[3];
    memcpy(y_s, initial_s, sizeof(y_s));
    memcpy(y_c, initial_c, sizeof(y_c));
    CHECK_STATUS(perflibs_spmv_exec_s(PERFLIBS_SPARSE_OPERATION_TRANS, 1.5f,
                                      A_s, x_s, -0.5f, y_s));
    CHECK_STATUS(perflibs_spmv_exec_c(PERFLIBS_SPARSE_OPERATION_CONJTRANS,
                                      alpha_c, A_c, x_c, beta_c, y_c));
    for (int col = 0; col < 3; ++col) {
      CHECK_DOUBLE_NEAR(y_s[col], expected_s[col], 1e-6);
    }
    CHECK_SINGLECOMPLEX_ARRAY(y_c, expected_c, 3, 1e-5f);
    if (repeat > 0) {
      CHECK_TRUE(memcmp(y_s, first_s, sizeof(y_s)) == 0 &&
                     memcmp(y_c, first_c, sizeof(y_c)) == 0,
                 "CSC row-owned result changed on repeat %d", repeat);
    }
    memcpy(first_s, y_s, sizeof(y_s));
    memcpy(first_c, y_c, sizeof(y_c));
  }

  CHECK_STATUS(perflibs_spmat_destroy(A_s));
  CHECK_STATUS(perflibs_spmat_destroy(A_c));
  return EXIT_SUCCESS;
}

static int test_csc_spmv_real_and_updates() {
  const perflibs_int_t row_indx[] = {0, 1, 1, 2, 0, 2};
  const perflibs_int_t col_ptr[] = {0, 2, 4, 6, 6};
  const double vals[] = {2.0, 1.0, 3.0, 4.0, 1.0, 5.0};
  const double x[] = {1.0, 2.0, 3.0, 0.0};
  double y[4] = {0.0, 0.0, 0.0, 0.0};
  double dense_expected[16] = {2.0, 0.0, 1.0, 0.0, 1.0, 3.0, 0.0, 0.0,
                               0.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  perflibs_spmat_t csc = NULL;
  perflibs_int_t m = -1;
  perflibs_int_t n = -1;
  double *dense = NULL;

  CHECK_STATUS(
      perflibs_spmat_create_csc_d(&csc, 4, 4, row_indx, col_ptr, vals, 0));
  CHECK_STATUS(perflibs_spmv_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS, 1.0, csc,
                                    x, 0.0, y));
  CHECK_DOUBLE_NEAR(y[0], 5.0, 1e-12);
  CHECK_DOUBLE_NEAR(y[1], 7.0, 1e-12);
  CHECK_DOUBLE_NEAR(y[2], 23.0, 1e-12);
  CHECK_DOUBLE_NEAR(y[3], 0.0, 1e-12);

  y[0] = y[1] = y[2] = y[3] = 0.0;
  CHECK_STATUS(perflibs_spmv_exec_d(PERFLIBS_SPARSE_OPERATION_TRANS, 1.0, csc,
                                    x, 0.0, y));
  CHECK_DOUBLE_NEAR(y[0], 4.0, 1e-12);
  CHECK_DOUBLE_NEAR(y[1], 18.0, 1e-12);
  CHECK_DOUBLE_NEAR(y[2], 16.0, 1e-12);
  CHECK_DOUBLE_NEAR(y[3], 0.0, 1e-12);

  const perflibs_int_t upd_row = 0;
  const perflibs_int_t upd_col = 2;
  const double upd_val = 2.0;
  CHECK_STATUS(perflibs_spmat_update_d(csc, 1, &upd_row, &upd_col, &upd_val));
  CHECK_STATUS(perflibs_spscale_exec_d(0.5, csc));
  dense_expected[2] = 1.0;
  dense_expected[0] = 1.0;
  dense_expected[4] = 0.5;
  dense_expected[5] = 1.5;
  dense_expected[9] = 2.0;
  dense_expected[10] = 2.5;
  CHECK_STATUS(
      perflibs_spmat_export_dense_d(csc, PERFLIBS_ROW_MAJOR, &m, &n, &dense));
  CHECK_DOUBLE_ARRAY(dense, dense_expected, 16, 1e-12);
  free(dense);

  CHECK_STATUS(perflibs_spmat_destroy(csc));
  return EXIT_SUCCESS;
}

static int test_csc_spsv_real_notrans_and_trans() {
  const perflibs_int_t row_indx[] = {0, 1, 1, 2, 2, 3};
  const perflibs_int_t col_ptr[] = {0, 2, 4, 5, 6};
  const double vals[] = {2.0, 1.0, 3.0, 4.0, 5.0, 1.0};
  const double rhs_notrans[] = {2.0, 7.0, 23.0, 4.0};
  const double rhs_trans[] = {4.0, 18.0, 15.0, 4.0};
  perflibs_spmat_t csc = NULL;
  double x[4] = {0.0, 0.0, 0.0, 0.0};

  CHECK_STATUS(
      perflibs_spmat_create_csc_d(&csc, 4, 4, row_indx, col_ptr, vals, 0));
  CHECK_STATUS(perflibs_spmat_hint(csc, PERFLIBS_SPARSE_HINT_STRUCTURE,
                                   PERFLIBS_SPARSE_STRUCTURE_TRIANGULAR));
  CHECK_STATUS(perflibs_spmat_hint(csc, PERFLIBS_SPARSE_HINT_SPSV_OPERATION,
                                   PERFLIBS_SPARSE_OPERATION_NOTRANS));
  CHECK_STATUS(perflibs_spsv_optimize(csc));

  CHECK_STATUS(perflibs_spsv_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS, csc, x,
                                    1.0, rhs_notrans));
  CHECK_DOUBLE_NEAR(x[0], 1.0, 1e-12);
  CHECK_DOUBLE_NEAR(x[1], 2.0, 1e-12);
  CHECK_DOUBLE_NEAR(x[2], 3.0, 1e-12);
  CHECK_DOUBLE_NEAR(x[3], 4.0, 1e-12);

  x[0] = x[1] = x[2] = x[3] = 0.0;
  CHECK_STATUS(perflibs_spsv_exec_d(PERFLIBS_SPARSE_OPERATION_TRANS, csc, x,
                                    1.0, rhs_trans));
  CHECK_DOUBLE_NEAR(x[0], 1.0, 1e-12);
  CHECK_DOUBLE_NEAR(x[1], 2.0, 1e-12);
  CHECK_DOUBLE_NEAR(x[2], 3.0, 1e-12);
  CHECK_DOUBLE_NEAR(x[3], 4.0, 1e-12);

  CHECK_STATUS(perflibs_spmat_destroy(csc));
  return EXIT_SUCCESS;
}

static int test_csc_spmv_complex_conj_paths() {
  const perflibs_int_t row_indx[] = {0, 0, 1};
  const perflibs_int_t col_ptr[] = {0, 1, 3, 3, 3};
  const perflibs_doublecomplex_t vals[] = {TEST_Z(1.0, 1.0), TEST_Z(2.0, -1.0),
                                           TEST_Z(3.0, 2.0)};
  const perflibs_doublecomplex_t x[] = {TEST_Z(1.0, -1.0), TEST_Z(2.0, 1.0),
                                        TEST_Z(0.0, 0.0), TEST_Z(0.0, 0.0)};
  perflibs_doublecomplex_t y[4] = {TEST_Z(0.0, 0.0), TEST_Z(0.0, 0.0),
                                   TEST_Z(0.0, 0.0), TEST_Z(0.0, 0.0)};
  perflibs_spmat_t csc = NULL;

  CHECK_STATUS(
      perflibs_spmat_create_csc_z(&csc, 4, 4, row_indx, col_ptr, vals, 0));
  CHECK_STATUS(perflibs_spmv_exec_z(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                    TEST_Z(1.0, 0.0), csc, x, TEST_Z(0.0, 0.0),
                                    y));
  CHECK_DOUBLECOMPLEX_NEAR(y[0], TEST_Z(7.0, 0.0), 1e-12);
  CHECK_DOUBLECOMPLEX_NEAR(y[1], TEST_Z(4.0, 7.0), 1e-12);
  CHECK_DOUBLECOMPLEX_NEAR(y[2], TEST_Z(0.0, 0.0), 1e-12);
  CHECK_DOUBLECOMPLEX_NEAR(y[3], TEST_Z(0.0, 0.0), 1e-12);

  y[0] = y[1] = y[2] = y[3] = TEST_Z(0.0, 0.0);
  CHECK_STATUS(perflibs_spmv_exec_z(PERFLIBS_SPARSE_OPERATION_CONJTRANS,
                                    TEST_Z(1.0, 0.0), csc, x, TEST_Z(0.0, 0.0),
                                    y));
  CHECK_DOUBLECOMPLEX_NEAR(y[0], TEST_Z(0.0, -2.0), 1e-12);
  CHECK_DOUBLECOMPLEX_NEAR(y[1], TEST_Z(11.0, -2.0), 1e-12);
  CHECK_DOUBLECOMPLEX_NEAR(y[2], TEST_Z(0.0, 0.0), 1e-12);
  CHECK_DOUBLECOMPLEX_NEAR(y[3], TEST_Z(0.0, 0.0), 1e-12);

  CHECK_STATUS(perflibs_spmat_destroy(csc));
  return EXIT_SUCCESS;
}

int main() {
  const enum perflibs_sparse_reproducibility_value initial_mode =
      perflibs_sparse_get_reproducibility();

  int result = EXIT_SUCCESS;
  /* Keep the existing kernel, update and solve coverage in disabled mode. */
  if (perflibs_sparse_set_reproducibility(
          PERFLIBS_SPARSE_REPRODUCIBILITY_DISABLED) !=
      PERFLIBS_STATUS_SUCCESS) {
    result = EXIT_FAILURE;
  }
  if (result == EXIT_SUCCESS) {
    result = test_csc_spmv_real_and_updates();
  }
  if (result == EXIT_SUCCESS) {
    result = test_csc_spsv_real_notrans_and_trans();
  }
  if (result == EXIT_SUCCESS) {
    result = test_csc_spmv_complex_conj_paths();
  }
  if (perflibs_sparse_set_reproducibility(
          PERFLIBS_SPARSE_REPRODUCIBILITY_ENABLED) != PERFLIBS_STATUS_SUCCESS) {
    result = EXIT_FAILURE;
  }

  /* CI supplies serial and OpenMP configurations. Repeated executions and
   * bitwise comparisons use the runtime configuration selected for this run. */
  if (result == EXIT_SUCCESS) {
    result = test_csc_reproducible_notrans_real();
  }
  if (result == EXIT_SUCCESS) {
    result = test_csc_reproducible_notrans_complex();
  }
  if (result == EXIT_SUCCESS) {
    result = test_csc_reproducible_row_owned();
  }

  if (perflibs_sparse_set_reproducibility(initial_mode) !=
      PERFLIBS_STATUS_SUCCESS) {
    return EXIT_FAILURE;
  }
  return result;
}
