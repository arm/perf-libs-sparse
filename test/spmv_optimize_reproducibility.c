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

/* These tests use numerical witnesses and bitwise repeatability of fresh
 * create/optimize/execute lifecycles through the public C API.
 * Each repetition restores identical inputs and initial y values. Comparisons
 * are within one process with a fixed build, hardware, floating-point
 * environment and threading configuration; fresh-process checks are separate.
 *
 * Disabled mode is allowed to repeat. The BSR mode difference below identifies
 * two particular accumulation orders without relying on scheduling variation.
 * Exact witness values describe the current implementations, rather than an
 * accuracy reference or a promise of identical bits from future kernels.
 *
 * These numerical checks do not prove that timed auditioning was bypassed:
 * ordinary optimization could repeatedly select the same candidate. */

#define REPEAT_COUNT 16
#define MATRIX_ROWS 4096
#define MATRIX_COLS 4096
#define NNZ_PER_ROW 16
#define MATRIX_NNZ (MATRIX_ROWS * NNZ_PER_ROW)
#define BSR_BLOCK_ROWS 512

static int run_bsr_mode(enum perflibs_sparse_reproducibility_value mode,
                        double *result) {
  /* With L = 2^60 and FP64 round-to-nearest, L + 1 rounds back to L.
   * Vanilla finishes each block: ((L + 1) - L) + 1 gives 1. The native
   * block-size-3 kernel accumulates matching positions across the blocks,
   * then combines them: ((L + 1) + 1) - L gives 0. This distinguishes the
   * selected arithmetic even when both implementations repeat perfectly. */
  const perflibs_int_t row_ptr[] = {0, 2};
  const perflibs_int_t col_indx[] = {0, 1};
  const double vals[] = {0x1p60, 1.0, -0x1p60, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         1.0,    0.0, 0.0,     0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  const double x[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  double y[] = {0.0, 0.0, 0.0};
  perflibs_spmat_t A = NULL;

  CHECK_STATUS(perflibs_sparse_set_reproducibility(mode));
  CHECK_STATUS(perflibs_spmat_create_bsr_d(&A, PERFLIBS_ROW_MAJOR, 3, 6, 3,
                                           row_ptr, col_indx, vals, 0));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_SPMV_OPERATION,
                                   PERFLIBS_SPARSE_OPERATION_NOTRANS));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_SPMV_INVOCATIONS,
                                   PERFLIBS_SPARSE_INVOCATIONS_MANY));
  CHECK_STATUS(perflibs_spmv_optimize(A));
  CHECK_STATUS(perflibs_spmv_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS, 1.0, A,
                                    x, 0.0, y));

  *result = y[0];
  CHECK_STATUS(perflibs_spmat_destroy(A));
  return EXIT_SUCCESS;
}

static int test_bsr_mode_discriminator(void) {
  /* Only 18 stored entries: ordinary optimization takes the nnz < 100 exit.
   * The derived 0/1 results detect ignoring the mode when choosing the kernel;
   * this case does not exercise the timing-based choice on larger problems. */
  double default_result, reproducible_result, repeated_result;

  CHECK_STATUS(
      run_bsr_mode(PERFLIBS_SPARSE_REPRODUCIBILITY_DISABLED, &default_result));
  CHECK_STATUS(run_bsr_mode(PERFLIBS_SPARSE_REPRODUCIBILITY_ENABLED,
                            &reproducible_result));
  CHECK_TRUE(default_result == 0.0 && reproducible_result == 1.0,
             "BSR optimize did not select the expected accumulation orders: "
             "disabled=%.17g enabled=%.17g",
             default_result, reproducible_result);

  for (int repeat = 0; repeat < REPEAT_COUNT; ++repeat) {
    CHECK_STATUS(run_bsr_mode(PERFLIBS_SPARSE_REPRODUCIBILITY_ENABLED,
                              &repeated_result));
    CHECK_TRUE(
        memcmp(&repeated_result, &reproducible_result, sizeof(double)) == 0,
        "BSR reproducible optimized result changed on repeat %d", repeat);
  }
  return EXIT_SUCCESS;
}

/* BSR records its kernel choice on optimization. Disabling reproducibility
 * after optimization must preserve the selected arithmetic on this object;
 * optimizing again in disabled mode must replace that choice. The same
 * numerical witness checks both transitions without inspecting a mode flag. */
static int test_bsr_mode_change_after_optimize(void) {
  const perflibs_int_t row_ptr[] = {0, 2};
  const perflibs_int_t col_indx[] = {0, 1};
  const double vals[] = {0x1p60, 1.0, -0x1p60, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         1.0,    0.0, 0.0,     0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  const double x[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  double y[] = {0.0, 0.0, 0.0};
  perflibs_spmat_t A = NULL;

  CHECK_STATUS(perflibs_sparse_set_reproducibility(
      PERFLIBS_SPARSE_REPRODUCIBILITY_ENABLED));
  CHECK_STATUS(perflibs_spmat_create_bsr_d(&A, PERFLIBS_ROW_MAJOR, 3, 6, 3,
                                           row_ptr, col_indx, vals, 0));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_SPMV_OPERATION,
                                   PERFLIBS_SPARSE_OPERATION_NOTRANS));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_SPMV_INVOCATIONS,
                                   PERFLIBS_SPARSE_INVOCATIONS_MANY));
  CHECK_STATUS(perflibs_spmv_optimize(A));

  CHECK_STATUS(perflibs_sparse_set_reproducibility(
      PERFLIBS_SPARSE_REPRODUCIBILITY_DISABLED));
  CHECK_STATUS(perflibs_spmv_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS, 1.0, A,
                                    x, 0.0, y));
  CHECK_TRUE(y[0] == 1.0, "changing mode altered the optimized BSR kernel");

  CHECK_STATUS(perflibs_spmv_optimize(A));
  y[0] = 0.0;
  CHECK_STATUS(perflibs_spmv_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS, 1.0, A,
                                    x, 0.0, y));
  CHECK_TRUE(y[0] == 0.0, "reoptimizing in disabled mode retained BSR vanilla");

  CHECK_STATUS(perflibs_spmat_destroy(A));
  return EXIT_SUCCESS;
}

static int test_large_bsr_reproducible_optimize(void) {
  const double block_vals[] = {0x1p60, 1.0, -0x1p60, 0.0, 0.0, 0.0,
                               0.0,    0.0, 0.0,     1.0, 0.0, 0.0,
                               0.0,    0.0, 0.0,     0.0, 0.0, 0.0};
  const double x[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  perflibs_int_t row_ptr[BSR_BLOCK_ROWS + 1];
  perflibs_int_t col_indx[2 * BSR_BLOCK_ROWS];
  double vals[18 * BSR_BLOCK_ROWS];
  double y[3 * BSR_BLOCK_ROWS];

  /* 9216 stored entries: above both the tiny and small audition thresholds.
   * Repeating the small witness checks the selected arithmetic at this size.
   * Each kernel iteration owns a whole block row and visits its blocks in
   * order, so scheduling different block rows cannot change their arithmetic.
   * Matching this result alone does not prove that auditioning was skipped. */
  for (perflibs_int_t row = 0; row < BSR_BLOCK_ROWS; ++row) {
    row_ptr[row] = 2 * row;
    col_indx[2 * row] = 0;
    col_indx[2 * row + 1] = 1;
    memcpy(vals + 18 * row, block_vals, sizeof(block_vals));
  }
  row_ptr[BSR_BLOCK_ROWS] = 2 * BSR_BLOCK_ROWS;

  CHECK_STATUS(perflibs_sparse_set_reproducibility(
      PERFLIBS_SPARSE_REPRODUCIBILITY_ENABLED));
  for (int repeat = 0; repeat < REPEAT_COUNT; ++repeat) {
    perflibs_spmat_t A = NULL;
    CHECK_STATUS(perflibs_spmat_create_bsr_d(&A, PERFLIBS_ROW_MAJOR,
                                             3 * BSR_BLOCK_ROWS, 6, 3, row_ptr,
                                             col_indx, vals, 0));
    CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_SPMV_INVOCATIONS,
                                     PERFLIBS_SPARSE_INVOCATIONS_MANY));
    CHECK_STATUS(perflibs_spmv_optimize(A));
    CHECK_STATUS(perflibs_spmv_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS, 1.0, A,
                                      x, 0.0, y));
    const double expected[] = {1.0, 0.0, 0.0};
    for (perflibs_int_t row = 0; row < BSR_BLOCK_ROWS; ++row) {
      CHECK_TRUE(memcmp(y + 3 * row, expected, sizeof(expected)) == 0,
                 "large BSR reproducible result changed at block row %lld "
                 "on repeat %d",
                 (long long)row, repeat);
    }
    CHECK_STATUS(perflibs_spmat_destroy(A));
  }
  return EXIT_SUCCESS;
}

static double test_value(perflibs_int_t p) {
  /* Mixed magnitudes make regrouping observable even with x filled with ones:
   * a stored-order sum loses the unit next to 2^60 before cancellation. */
  switch (p % 4) {
  case 0:
    return 0x1p60;
  case 1:
    return 1.0;
  case 2:
    return -0x1p60;
  default:
    return 1.0 + (double)((p / NNZ_PER_ROW) % 7) * 0.125;
  }
}

static void build_csr(perflibs_int_t *row_ptr, perflibs_int_t *col_indx,
                      double *vals) {
  for (perflibs_int_t row = 0; row <= MATRIX_ROWS; ++row) {
    row_ptr[row] = row * NNZ_PER_ROW;
  }

  for (perflibs_int_t row = 0; row < MATRIX_ROWS; ++row) {
    for (perflibs_int_t k = 0; k < NNZ_PER_ROW; ++k) {
      const perflibs_int_t p = row * NNZ_PER_ROW + k;
      col_indx[p] = (row * 17 + k * 257) % MATRIX_COLS;
      vals[p] = test_value(p);
    }
  }
}

static void build_csc(perflibs_int_t *col_ptr, perflibs_int_t *row_indx,
                      double *vals) {
  for (perflibs_int_t col = 0; col <= MATRIX_COLS; ++col) {
    col_ptr[col] = col * NNZ_PER_ROW;
  }

  for (perflibs_int_t col = 0; col < MATRIX_COLS; ++col) {
    for (perflibs_int_t k = 0; k < NNZ_PER_ROW; ++k) {
      const perflibs_int_t p = col * NNZ_PER_ROW + k;
      row_indx[p] = (col * 17 + k * 257) % MATRIX_ROWS;
      /* Each column repeats [L, L, s, s, -L, -L, s, s] twice, with L = 2^60.
       * In FP64 round-to-nearest, stored-order and two-part sums give 2*s;
       * four-part sums give 8*s, and reverse traversal gives zero. The native
       * CSR view and vanilla kernel therefore agree on these inputs while
       * other orders can differ. This exposes cancellation that short sums
       * of small, exactly representable values alone would miss. */
      switch (k % 8) {
      case 0:
      case 1:
        vals[p] = 0x1p60;
        break;
      case 4:
      case 5:
        vals[p] = -0x1p60;
        break;
      default:
        vals[p] = 1.0 + (double)(col % 7) * 0.125;
      }
    }
  }
}

static void fill_x(double *x, perflibs_int_t n) {
  for (perflibs_int_t i = 0; i < n; ++i) {
    x[i] = 1.0;
  }
}

static void fill_y(double *y, perflibs_int_t n) {
  for (perflibs_int_t i = 0; i < n; ++i) {
    y[i] = (double)((i % 7) - 3) * 0.5;
  }
}

/* A stored-order oracle for this witness, including beta and the initial y.
 * Its purpose is to identify the expected arithmetic, not an exact-real sum. */
static void expected_csc_trans(const perflibs_int_t *col_ptr,
                               const perflibs_int_t *row_indx,
                               const double *vals, const double *x, double beta,
                               double *y) {
  for (perflibs_int_t col = 0; col < MATRIX_COLS; ++col) {
    double sum = 0.0;
    for (perflibs_int_t p = col_ptr[col]; p < col_ptr[col + 1]; ++p) {
      sum += vals[p] * x[row_indx[p]];
    }
    y[col] = beta * y[col] + sum;
  }
}

static int run_csr_reproducible_optimize(double *y) {
  perflibs_int_t *row_ptr =
      (perflibs_int_t *)malloc(sizeof(perflibs_int_t) * (MATRIX_ROWS + 1));
  perflibs_int_t *col_indx =
      (perflibs_int_t *)malloc(sizeof(perflibs_int_t) * MATRIX_NNZ);
  double *vals = (double *)malloc(sizeof(double) * MATRIX_NNZ);
  double *x = (double *)malloc(sizeof(double) * MATRIX_COLS);
  perflibs_spmat_t A = NULL;

  CHECK_TRUE(row_ptr != NULL && col_indx != NULL && vals != NULL && x != NULL,
             "failed to allocate CSR test data");
  build_csr(row_ptr, col_indx, vals);
  fill_x(x, MATRIX_COLS);
  fill_y(y, MATRIX_ROWS);

  CHECK_STATUS(perflibs_sparse_set_reproducibility(
      PERFLIBS_SPARSE_REPRODUCIBILITY_ENABLED));
  CHECK_STATUS(perflibs_spmat_create_csr_d(&A, MATRIX_ROWS, MATRIX_COLS,
                                           row_ptr, col_indx, vals, 0));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_SPMV_OPERATION,
                                   PERFLIBS_SPARSE_OPERATION_NOTRANS));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_SPMV_INVOCATIONS,
                                   PERFLIBS_SPARSE_INVOCATIONS_MANY));
  CHECK_STATUS(perflibs_spmv_optimize(A));

  CHECK_STATUS(perflibs_spmv_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS, 1.0, A,
                                    x, -0.25, y));

  CHECK_STATUS(perflibs_spmat_destroy(A));
  free(row_ptr);
  free(col_indx);
  free(vals);
  free(x);
  return EXIT_SUCCESS;
}

/* Every run creates and optimizes a fresh CSR object, covering initialization
 * and optimization as well as execution. The selected vanilla kernel visits
 * each row in stored order with one iteration owning its output, so worker
 * scheduling cannot reorder a row's additions. memcmp checks output bits
 * against the first enabled run. */
static int test_csr_repeated_reproducible_optimize_executes(void) {
  double *reference = (double *)malloc(sizeof(double) * MATRIX_ROWS);
  double *y = (double *)malloc(sizeof(double) * MATRIX_ROWS);

  CHECK_TRUE(reference != NULL && y != NULL,
             "failed to allocate CSR test data");

  CHECK_STATUS(run_csr_reproducible_optimize(reference));

  for (int repeat = 0; repeat < REPEAT_COUNT; ++repeat) {
    CHECK_STATUS(run_csr_reproducible_optimize(y));
    CHECK_TRUE(memcmp(y, reference, sizeof(double) * MATRIX_ROWS) == 0,
               "CSR reproducible optimize result changed on repeat %d", repeat);
  }

  free(reference);
  free(y);
  return EXIT_SUCCESS;
}

static int run_csc_reproducible_optimize(double *y) {
  perflibs_int_t *col_ptr =
      (perflibs_int_t *)malloc(sizeof(perflibs_int_t) * (MATRIX_COLS + 1));
  perflibs_int_t *row_indx =
      (perflibs_int_t *)malloc(sizeof(perflibs_int_t) * MATRIX_NNZ);
  double *vals = (double *)malloc(sizeof(double) * MATRIX_NNZ);
  double *x = (double *)malloc(sizeof(double) * MATRIX_ROWS);
  perflibs_spmat_t A = NULL;

  CHECK_TRUE(col_ptr != NULL && row_indx != NULL && vals != NULL && x != NULL,
             "failed to allocate CSC test data");
  build_csc(col_ptr, row_indx, vals);
  fill_x(x, MATRIX_ROWS);
  fill_y(y, MATRIX_COLS);

  CHECK_STATUS(perflibs_sparse_set_reproducibility(
      PERFLIBS_SPARSE_REPRODUCIBILITY_ENABLED));
  CHECK_STATUS(perflibs_spmat_create_csc_d(&A, MATRIX_ROWS, MATRIX_COLS,
                                           row_indx, col_ptr, vals, 0));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_SPMV_OPERATION,
                                   PERFLIBS_SPARSE_OPERATION_TRANS));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_SPMV_INVOCATIONS,
                                   PERFLIBS_SPARSE_INVOCATIONS_MANY));
  CHECK_STATUS(perflibs_spmv_optimize(A));

  CHECK_STATUS(
      perflibs_spmv_exec_d(PERFLIBS_SPARSE_OPERATION_TRANS, 1.0, A, x, 0.5, y));

  CHECK_STATUS(perflibs_spmat_destroy(A));
  free(col_ptr);
  free(row_indx);
  free(vals);
  free(x);
  return EXIT_SUCCESS;
}

/* CSC TRANS uses a non-transposed CSR view: each column contributes to one
 * independently owned output. Its fixed local accumulation order explains
 * repeatability. Check both the derived witness result and bitwise agreement
 * across fresh-object repeats. */
static int test_csc_repeated_reproducible_optimize_executes(void) {
  perflibs_int_t *col_ptr =
      (perflibs_int_t *)malloc(sizeof(perflibs_int_t) * (MATRIX_COLS + 1));
  perflibs_int_t *row_indx =
      (perflibs_int_t *)malloc(sizeof(perflibs_int_t) * MATRIX_NNZ);
  double *vals = (double *)malloc(sizeof(double) * MATRIX_NNZ);
  double *x = (double *)malloc(sizeof(double) * MATRIX_ROWS);
  double *expected = (double *)malloc(sizeof(double) * MATRIX_COLS);
  double *reference = (double *)malloc(sizeof(double) * MATRIX_COLS);
  double *y = (double *)malloc(sizeof(double) * MATRIX_COLS);

  CHECK_TRUE(col_ptr != NULL && row_indx != NULL && vals != NULL && x != NULL &&
                 expected != NULL && reference != NULL && y != NULL,
             "failed to allocate CSC test data");

  build_csc(col_ptr, row_indx, vals);
  fill_x(x, MATRIX_ROWS);
  fill_y(expected, MATRIX_COLS);
  expected_csc_trans(col_ptr, row_indx, vals, x, 0.5, expected);

  /* Check the witness itself: changing traversal order must alter the answer.
   * This prevents a later fixture edit from making all additions exact and
   * leaving the bitwise oracle insensitive to arithmetic-order changes. */
  for (perflibs_int_t col = 0; col < MATRIX_COLS; ++col) {
    double reversed_sum = 0.0;
    for (perflibs_int_t p = col_ptr[col + 1]; p > col_ptr[col]; --p) {
      reversed_sum += vals[p - 1] * x[row_indx[p - 1]];
    }
    const double reversed_y =
        0.5 * (double)((col % 7) - 3) * 0.5 + reversed_sum;
    CHECK_TRUE(expected[col] != reversed_y,
               "CSC test data do not distinguish accumulation order");
  }

  CHECK_STATUS(run_csc_reproducible_optimize(reference));
  CHECK_TRUE(memcmp(reference, expected, sizeof(double) * MATRIX_COLS) == 0,
             "CSC reproducible optimize changed the accumulation order");

  for (int repeat = 0; repeat < REPEAT_COUNT; ++repeat) {
    CHECK_STATUS(run_csc_reproducible_optimize(y));
    CHECK_TRUE(memcmp(y, reference, sizeof(double) * MATRIX_COLS) == 0,
               "CSC reproducible optimize result changed on repeat %d", repeat);
  }

  free(col_ptr);
  free(row_indx);
  free(vals);
  free(x);
  free(expected);
  free(reference);
  free(y);
  return EXIT_SUCCESS;
}

int main(void) {
  const enum perflibs_sparse_reproducibility_value initial_mode =
      perflibs_sparse_get_reproducibility();

  int result = test_bsr_mode_discriminator();
  if (result == EXIT_SUCCESS) {
    result = test_bsr_mode_change_after_optimize();
  }
  if (result == EXIT_SUCCESS) {
    result = test_large_bsr_reproducible_optimize();
  }
  if (result == EXIT_SUCCESS) {
    result = test_csr_repeated_reproducible_optimize_executes();
  }
  if (result == EXIT_SUCCESS) {
    result = test_csc_repeated_reproducible_optimize_executes();
  }

  if (perflibs_sparse_set_reproducibility(initial_mode) !=
      PERFLIBS_STATUS_SUCCESS) {
    return EXIT_FAILURE;
  }

  return result;
}
