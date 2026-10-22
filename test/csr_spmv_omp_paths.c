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

/* These public-API cases test repeated execution of the same CSR object with
 * identical inputs and reset output values. Numerical expectations check the
 * arithmetic, while memcmp checks bitwise repeatability. Fresh-object
 * optimization is covered separately in spmv_optimize_reproducibility.c.
 * Comparisons here use a fixed build, hardware and floating-point environment.
 *
 * The cancellation cases can expose an unordered accumulation path, but finite
 * repetitions cannot guarantee that every such regression will be detected.
 * Disabled mode is allowed to repeat; variation is not a test requirement. */

#define REPRO_ROWS 4096
#define REPRO_REPEATS 16

/* CSR TRANS sends every row's contribution to y[0]. Ordinary parallel updates
 * may reach that output in different orders; reproducible execution visits
 * stored rows serially. In FP64 round-to-nearest, the four equal groups below
 * accumulate large positive values, lose the following units, cancel the large
 * values, then retain REPRO_ROWS/4 quarters, giving REPRO_ROWS/16. This is a
 * witness for the stored accumulation order, not the exact-real sum. */
static int test_double_reproducible_transpose(void) {
  perflibs_int_t *row_ptr = malloc(sizeof(*row_ptr) * (REPRO_ROWS + 1));
  perflibs_int_t *col_indx = malloc(sizeof(*col_indx) * REPRO_ROWS);
  double *vals = malloc(sizeof(*vals) * REPRO_ROWS);
  double *x = malloc(sizeof(*x) * REPRO_ROWS);
  perflibs_spmat_t A = NULL;
  CHECK_TRUE(row_ptr && col_indx && vals && x,
             "allocation failure for reproducible CSR transpose");

  for (perflibs_int_t row = 0; row < REPRO_ROWS; ++row) {
    row_ptr[row] = row;
    col_indx[row] = 0;
    x[row] = 1.0;
    switch (row / (REPRO_ROWS / 4)) {
    case 0:
      vals[row] = 0x1p60;
      break;
    case 1:
      vals[row] = 1.0;
      break;
    case 2:
      vals[row] = -0x1p60;
      break;
    default:
      vals[row] = 0.25;
      break;
    }
  }
  row_ptr[REPRO_ROWS] = REPRO_ROWS;

  CHECK_STATUS(perflibs_spmat_create_csr_d(&A, REPRO_ROWS, 1, row_ptr, col_indx,
                                           vals, 0));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_SPMV_OPERATION,
                                   PERFLIBS_SPARSE_OPERATION_TRANS));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_SPMV_INVOCATIONS,
                                   PERFLIBS_SPARSE_INVOCATIONS_MANY));
  CHECK_STATUS(perflibs_spmv_optimize(A));

  double first = 0.0;
  for (int repeat = 0; repeat < REPRO_REPEATS; ++repeat) {
    double y[] = {0.0};
    CHECK_STATUS(perflibs_spmv_exec_d(PERFLIBS_SPARSE_OPERATION_TRANS, 1.0, A,
                                      x, 0.0, y));
    CHECK_TRUE(y[0] == REPRO_ROWS / 16,
               "CSR transpose accumulation order changed: %.17g", y[0]);
    if (repeat > 0) {
      CHECK_TRUE(memcmp(&first, y, sizeof(first)) == 0,
                 "CSR transpose result changed on repeat %d", repeat);
    }
    first = y[0];
  }

  CHECK_STATUS(perflibs_spmat_destroy(A));
  free(row_ptr);
  free(col_indx);
  free(vals);
  free(x);
  return EXIT_SUCCESS;
}

/* The same shared-output witness in complex FP32 uses 2^30 to lose the real
 * unit contributions before cancellation. Conjugation changes the imaginary
 * groups to -0.5 and +0.25, whose exact sum is -REPRO_ROWS/16. Thus the real
 * component checks accumulation order and the imaginary component also checks
 * that CONJTRANS applies conjugation. */
static int test_singlecomplex_reproducible_conjtranspose(void) {
  perflibs_int_t *row_ptr = malloc(sizeof(*row_ptr) * (REPRO_ROWS + 1));
  perflibs_int_t *col_indx = malloc(sizeof(*col_indx) * REPRO_ROWS);
  perflibs_singlecomplex_t *vals = malloc(sizeof(*vals) * REPRO_ROWS);
  perflibs_singlecomplex_t *x = malloc(sizeof(*x) * REPRO_ROWS);
  perflibs_spmat_t A = NULL;
  CHECK_TRUE(row_ptr && col_indx && vals && x,
             "allocation failure for reproducible CSR conjugate transpose");

  for (perflibs_int_t row = 0; row < REPRO_ROWS; ++row) {
    row_ptr[row] = row;
    col_indx[row] = 0;
    x[row] = TEST_C(1.0, 0.0);
    switch (row / (REPRO_ROWS / 4)) {
    case 0:
      vals[row] = TEST_C(0x1p30, 0.0);
      break;
    case 1:
      vals[row] = TEST_C(1.0, 0.5);
      break;
    case 2:
      vals[row] = TEST_C(-0x1p30, 0.0);
      break;
    default:
      vals[row] = TEST_C(0.25, -0.25);
      break;
    }
  }
  row_ptr[REPRO_ROWS] = REPRO_ROWS;

  CHECK_STATUS(perflibs_spmat_create_csr_c(&A, REPRO_ROWS, 1, row_ptr, col_indx,
                                           vals, 0));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_SPMV_OPERATION,
                                   PERFLIBS_SPARSE_OPERATION_CONJTRANS));
  CHECK_STATUS(perflibs_spmv_optimize(A));

  perflibs_singlecomplex_t first = TEST_C(0.0, 0.0);
  for (int repeat = 0; repeat < REPRO_REPEATS; ++repeat) {
    perflibs_singlecomplex_t y[] = {TEST_C(0.0, 0.0)};
    CHECK_STATUS(perflibs_spmv_exec_c(PERFLIBS_SPARSE_OPERATION_CONJTRANS,
                                      TEST_C(1.0, 0.0), A, x, TEST_C(0.0, 0.0),
                                      y));
    CHECK_SINGLECOMPLEX_NEAR(y[0], TEST_C(REPRO_ROWS / 16, -REPRO_ROWS / 16),
                             0.0f);
    if (repeat > 0) {
      CHECK_TRUE(memcmp(&first, y, sizeof(first)) == 0,
                 "CSR conjugate transpose result changed on repeat %d", repeat);
    }
    first = y[0];
  }

  CHECK_STATUS(perflibs_spmat_destroy(A));
  free(row_ptr);
  free(col_indx);
  free(vals);
  free(x);
  return EXIT_SUCCESS;
}

/* For NOTRANS, each output row belongs to one loop iteration with a fixed local
 * accumulation order; scheduling other rows cannot change its arithmetic.
 * Exercise both optimization with the mode enabled and enabling the mode on
 * existing, unoptimized objects. These small binary-fraction inputs check
 * scaling, complex arithmetic and repeatability, but may also give identical
 * results with the ordinary kernel, so they do not prove mode selection. */
static int test_row_owned_reproducible(int optimize) {
  const perflibs_int_t row_ptr[] = {0, 4, 6, 9};
  const perflibs_int_t col_indx[] = {0, 1, 2, 3, 0, 2, 0, 1, 3};
  const float vals_s[] = {0.5f, 1.0f, 0.25f, 1.5f, 2.0f,
                          3.0f, 1.0f, 1.25f, -0.5f};
  const float x_s[] = {1.0f, 1.0f, 1.0f, 1.0f};
  const float initial_s[] = {2.0f, -4.0f, 1.0f};
  const float expected_s[] = {5.875f, 5.5f, 3.125f};
  const perflibs_doublecomplex_t vals_z[] = {
      TEST_Z(0.5, 0.25), TEST_Z(1.0, -0.5),  TEST_Z(0.25, 1.0),
      TEST_Z(1.5, 0.0),  TEST_Z(2.0, -0.25), TEST_Z(3.0, 0.5),
      TEST_Z(1.0, 0.25), TEST_Z(1.25, -0.5), TEST_Z(-0.5, 1.0)};
  const perflibs_doublecomplex_t x_z[] = {TEST_Z(1.0, 1.0), TEST_Z(2.0, -1.0),
                                          TEST_Z(-1.0, 0.5), TEST_Z(0.5, 1.0)};
  const perflibs_doublecomplex_t initial_z[] = {
      TEST_Z(2.0, -1.0), TEST_Z(-4.0, 0.5), TEST_Z(1.0, 2.0)};
  const perflibs_doublecomplex_t alpha_z = TEST_Z(1.5, 0.5);
  const perflibs_doublecomplex_t beta_z = TEST_Z(-0.5, 0.25);
  perflibs_doublecomplex_t expected_z[3];
  perflibs_spmat_t A_s = NULL, A_z = NULL;

  CHECK_STATUS(perflibs_sparse_set_reproducibility(
      optimize ? PERFLIBS_SPARSE_REPRODUCIBILITY_ENABLED
               : PERFLIBS_SPARSE_REPRODUCIBILITY_DISABLED));
  for (perflibs_int_t row = 0; row < 3; ++row) {
    perflibs_doublecomplex_t sum = TEST_Z(0.0, 0.0);
    for (perflibs_int_t j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {
      sum = test_cadd(sum, test_cmul(vals_z[j], x_z[col_indx[j]]));
    }
    expected_z[row] =
        test_cadd(test_cmul(beta_z, initial_z[row]), test_cmul(alpha_z, sum));
  }

  CHECK_STATUS(
      perflibs_spmat_create_csr_s(&A_s, 3, 4, row_ptr, col_indx, vals_s, 0));
  CHECK_STATUS(
      perflibs_spmat_create_csr_z(&A_z, 3, 4, row_ptr, col_indx, vals_z, 0));
  if (optimize) {
    CHECK_STATUS(perflibs_spmv_optimize(A_s));
    CHECK_STATUS(perflibs_spmv_optimize(A_z));
  } else {
    /* The execution call must honor the current mode even though creation
     * happened in disabled mode and there is no subsequent optimization. */
    CHECK_STATUS(perflibs_sparse_set_reproducibility(
        PERFLIBS_SPARSE_REPRODUCIBILITY_ENABLED));
  }

  float first_s[3];
  perflibs_doublecomplex_t first_z[3];
  /* Nonzero beta makes restoring the original y essential on every repeat.
   * Tolerance checks cover correctness; memcmp requires identical output bits.
   */
  for (int repeat = 0; repeat < REPRO_REPEATS; ++repeat) {
    float y_s[3];
    perflibs_doublecomplex_t y_z[3];
    memcpy(y_s, initial_s, sizeof(y_s));
    memcpy(y_z, initial_z, sizeof(y_z));
    CHECK_STATUS(perflibs_spmv_exec_s(PERFLIBS_SPARSE_OPERATION_NOTRANS, 1.5f,
                                      A_s, x_s, 0.5f, y_s));
    CHECK_STATUS(perflibs_spmv_exec_z(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                      alpha_z, A_z, x_z, beta_z, y_z));
    for (int row = 0; row < 3; ++row) {
      CHECK_DOUBLE_NEAR(y_s[row], expected_s[row], 1e-6);
    }
    CHECK_DOUBLECOMPLEX_ARRAY(y_z, expected_z, 3, 1e-12);
    if (repeat > 0) {
      CHECK_TRUE(memcmp(y_s, first_s, sizeof(y_s)) == 0 &&
                     memcmp(y_z, first_z, sizeof(y_z)) == 0,
                 "CSR row-owned result changed on repeat %d", repeat);
    }
    memcpy(first_s, y_s, sizeof(y_s));
    memcpy(first_z, y_z, sizeof(y_z));
  }

  CHECK_STATUS(perflibs_spmat_destroy(A_s));
  CHECK_STATUS(perflibs_spmat_destroy(A_z));
  return EXIT_SUCCESS;
}

static void build_irregular_csr_structure(perflibs_int_t m, perflibs_int_t n,
                                          perflibs_int_t *row_ptr,
                                          perflibs_int_t *col_indx) {
  row_ptr[0] = 0;
  for (perflibs_int_t i = 0; i < m; ++i) {
    perflibs_int_t len;
    switch (i % 8) {
    case 0:
      len = 7;
      break;
    case 1:
      len = 113;
      break;
    case 2:
      len = 19;
      break;
    case 3:
      len = 181;
      break;
    case 4:
      len = 5;
      break;
    case 5:
      len = 149;
      break;
    case 6:
      len = 37;
      break;
    default:
      len = 223;
      break;
    }
    row_ptr[i + 1] = row_ptr[i] + len;

    if (col_indx != NULL) {
      const perflibs_int_t start = (29 * i + 17) % (n - len);
      for (perflibs_int_t j = 0; j < len; ++j) {
        col_indx[row_ptr[i] + j] = start + j;
      }
    }
  }
}

static int test_float_irregular_spmv_optimized() {
  const perflibs_int_t m = 1536;
  const perflibs_int_t n = 12288;
  const float alpha = 1.75f;
  const float beta = -0.5f;

  perflibs_int_t *row_ptr =
      (perflibs_int_t *)malloc(sizeof(perflibs_int_t) * (m + 1));
  perflibs_int_t *col_indx = NULL;
  float *vals = NULL;
  float *x = NULL;
  float *y = NULL;
  float *y_expected = NULL;
  perflibs_spmat_t A = NULL;

  CHECK_TRUE(row_ptr, "allocation failure for CSR SpMV OpenMP float test");

  build_irregular_csr_structure(m, n, row_ptr, NULL);
  col_indx = (perflibs_int_t *)malloc(sizeof(perflibs_int_t) * row_ptr[m]);
  vals = (float *)malloc(sizeof(float) * row_ptr[m]);
  x = (float *)malloc(sizeof(float) * n);
  y = (float *)malloc(sizeof(float) * m);
  y_expected = (float *)malloc(sizeof(float) * m);
  CHECK_TRUE(col_indx && vals && x && y && y_expected,
             "allocation failure for CSR SpMV OpenMP float buffers");

  build_irregular_csr_structure(m, n, row_ptr, col_indx);

  for (perflibs_int_t i = 0; i < row_ptr[m]; ++i) {
    vals[i] = 0.125f * (float)((i % 13) + 1);
  }
  for (perflibs_int_t i = 0; i < n; ++i) {
    x[i] = ((float)((i % 11) - 5)) * 0.5f;
  }
  for (perflibs_int_t i = 0; i < m; ++i) {
    y[i] = (float)((i % 9) - 4);
    y_expected[i] = y[i];
  }

  for (perflibs_int_t i = 0; i < m; ++i) {
    double sum = 0.0;
    for (perflibs_int_t p = row_ptr[i]; p < row_ptr[i + 1]; ++p) {
      sum += (double)vals[p] * (double)x[col_indx[p]];
    }
    y_expected[i] = beta * y_expected[i] + alpha * (float)sum;
  }

  CHECK_STATUS(
      perflibs_spmat_create_csr_s(&A, m, n, row_ptr, col_indx, vals, 0));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_SPMV_OPERATION,
                                   PERFLIBS_SPARSE_OPERATION_NOTRANS));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_SPMV_INVOCATIONS,
                                   PERFLIBS_SPARSE_INVOCATIONS_MANY));
  CHECK_STATUS(perflibs_spmv_optimize(A));

  CHECK_STATUS(perflibs_spmv_exec_s(PERFLIBS_SPARSE_OPERATION_NOTRANS, alpha, A,
                                    x, beta, y));
  for (perflibs_int_t i = 0; i < m; ++i) {
    CHECK_DOUBLE_NEAR(y[i], y_expected[i], 2e-4);
  }

  CHECK_STATUS(perflibs_spmat_destroy(A));
  free(row_ptr);
  free(col_indx);
  free(vals);
  free(x);
  free(y);
  free(y_expected);
  return EXIT_SUCCESS;
}

static int test_complex_irregular_spmv_conjnotrans_optimized() {
  const perflibs_int_t m = 1024;
  const perflibs_int_t n = 8192;
  const perflibs_singlecomplex_t alpha = TEST_C(0.75f, -0.5f);
  const perflibs_singlecomplex_t beta = TEST_C(-0.25f, 0.125f);

  perflibs_int_t *row_ptr =
      (perflibs_int_t *)malloc(sizeof(perflibs_int_t) * (m + 1));
  perflibs_int_t *col_indx = NULL;
  perflibs_singlecomplex_t *vals = NULL;
  perflibs_singlecomplex_t *x = NULL;
  perflibs_singlecomplex_t *y = NULL;
  perflibs_singlecomplex_t *y_expected = NULL;
  perflibs_spmat_t A = NULL;

  CHECK_TRUE(row_ptr, "allocation failure for CSR SpMV OpenMP complex test");

  build_irregular_csr_structure(m, n, row_ptr, NULL);
  col_indx = (perflibs_int_t *)malloc(sizeof(perflibs_int_t) * row_ptr[m]);
  vals = (perflibs_singlecomplex_t *)malloc(sizeof(perflibs_singlecomplex_t) *
                                            row_ptr[m]);
  x = (perflibs_singlecomplex_t *)malloc(sizeof(perflibs_singlecomplex_t) * n);
  y = (perflibs_singlecomplex_t *)malloc(sizeof(perflibs_singlecomplex_t) * n);
  y_expected =
      (perflibs_singlecomplex_t *)malloc(sizeof(perflibs_singlecomplex_t) * n);
  CHECK_TRUE(col_indx && vals && x && y && y_expected,
             "allocation failure for CSR SpMV OpenMP complex buffers");

  build_irregular_csr_structure(m, n, row_ptr, col_indx);

  for (perflibs_int_t i = 0; i < row_ptr[m]; ++i) {
    vals[i] = TEST_C(0.1f * (float)((i % 7) + 1), 0.05f * (float)((i % 5) - 2));
  }
  for (perflibs_int_t i = 0; i < n; ++i) {
    x[i] = TEST_C(0.25f * (float)((i % 9) - 4), 0.125f * (float)((i % 7) - 3));
  }
  for (perflibs_int_t i = 0; i < n; ++i) {
    y[i] = TEST_C(0.2f * (float)((i % 6) - 2), -0.1f * (float)((i % 5) - 2));
    y_expected[i] = test_cmulf(beta, y[i]);
  }

  for (perflibs_int_t i = 0; i < m; ++i) {
    for (perflibs_int_t p = row_ptr[i]; p < row_ptr[i + 1]; ++p) {
      perflibs_singlecomplex_t update =
          test_cmulf(alpha, test_cmulf(test_conjf(vals[p]), x[i]));
      y_expected[col_indx[p]] = test_caddf(y_expected[col_indx[p]], update);
    }
  }

  CHECK_STATUS(
      perflibs_spmat_create_csr_c(&A, m, n, row_ptr, col_indx, vals, 0));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_SPMV_OPERATION,
                                   PERFLIBS_SPARSE_OPERATION_CONJTRANS));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_SPMV_INVOCATIONS,
                                   PERFLIBS_SPARSE_INVOCATIONS_MANY));
  CHECK_STATUS(perflibs_spmv_optimize(A));

  CHECK_STATUS(perflibs_spmv_exec_c(PERFLIBS_SPARSE_OPERATION_CONJTRANS, alpha,
                                    A, x, beta, y));
  CHECK_SINGLECOMPLEX_ARRAY(y, y_expected, n, 3e-4f);

  CHECK_STATUS(perflibs_spmat_destroy(A));
  free(row_ptr);
  free(col_indx);
  free(vals);
  free(x);
  free(y);
  free(y_expected);
  return EXIT_SUCCESS;
}

int main() {
  const enum perflibs_sparse_reproducibility_value initial_mode =
      perflibs_sparse_get_reproducibility();

  int result = EXIT_SUCCESS;
  /* Keep the existing optimized-kernel accuracy coverage in disabled mode. */
  if (perflibs_sparse_set_reproducibility(
          PERFLIBS_SPARSE_REPRODUCIBILITY_DISABLED) !=
      PERFLIBS_STATUS_SUCCESS) {
    result = EXIT_FAILURE;
  }
  if (result == EXIT_SUCCESS) {
    result = test_float_irregular_spmv_optimized();
  }
  if (result == EXIT_SUCCESS) {
    result = test_complex_irregular_spmv_conjnotrans_optimized();
  }
  if (perflibs_sparse_set_reproducibility(
          PERFLIBS_SPARSE_REPRODUCIBILITY_ENABLED) != PERFLIBS_STATUS_SUCCESS) {
    result = EXIT_FAILURE;
  }

  /* CI supplies serial and OpenMP configurations. Repeated executions and
   * bitwise comparisons use the runtime configuration selected for this run. */
  if (result == EXIT_SUCCESS) {
    result = test_double_reproducible_transpose();
  }
  if (result == EXIT_SUCCESS) {
    result = test_singlecomplex_reproducible_conjtranspose();
  }
  if (result == EXIT_SUCCESS) {
    result = test_row_owned_reproducible(1);
  }
  if (result == EXIT_SUCCESS) {
    result = test_row_owned_reproducible(0);
  }

  if (perflibs_sparse_set_reproducibility(initial_mode) !=
      PERFLIBS_STATUS_SUCCESS) {
    return EXIT_FAILURE;
  }
  return result;
}
