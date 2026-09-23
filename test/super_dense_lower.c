/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "perflibs_sparse.h"
#include "test_utils.h"

#include <math.h>
#include <stdlib.h>

#define SUPER_LOWER_N 17
#define SUPER_LOWER_NNZ 90
#define SUPER_LOWER_NSUPER 6
#define SUPER_LOWER_NPARTS 2

int main() {
  /*
      Lower-triangular supernodal example from super_lower.c, solved with the
      separator converted to dense (cblas_trsv).
      The right-hand side is chosen so the solution vector is all ones.
  */
  const perflibs_int_t n = SUPER_LOWER_N;
  const perflibs_int_t nsuper = SUPER_LOWER_NSUPER;
  const perflibs_int_t nparts = SUPER_LOWER_NPARTS;
  const float alpha = 1.1f;

  const perflibs_int_t super_col_indx[SUPER_LOWER_NSUPER + 1] = {0,  2,  4, 7,
                                                                 10, 13, 17};
  const perflibs_int_t super_row_ptr[SUPER_LOWER_NSUPER + 1] = {0,  6,  13, 17,
                                                                24, 29, 33};
  const perflibs_int_t row_indx[33] = {
      0, 1, 2, 3,  13, 15, 2,  3,  4,  5,  6,  13, 15, 4,  5,  6, 13,
      7, 8, 9, 10, 12, 13, 15, 10, 11, 12, 13, 14, 13, 14, 15, 16};
  const perflibs_int_t col_ptr[SUPER_LOWER_N + 1] = {
      0, 6, 12, 19, 26, 30, 34, 38, 45, 52, 59, 64, 69, 74, 78, 82, 86, 90};
  const float vals[SUPER_LOWER_NNZ] = {
      1, 1, 1, 1, 1, 9, 0, 2, 2, 2, 2, 8, 3, 3, 3, 3, 3, 3, 7, 0, 4, 4, 4,
      4, 4, 6, 3, 4, 5, 5, 0, 2, 6, 6, 0, 0, 1, 7, 1, 1, 1, 1, 1, 8, 6, 0,
      1, 1, 1, 1, 9, 5, 0, 0, 1, 1, 1, 1, 4, 2, 2, 2, 2, 1, 0, 3, 3, 3, 1,
      0, 0, 4, 4, 1, 5, 1, 3, 4, 0, 1, 2, 3, 0, 0, 1, 2, 0, 0, 0, 1};
  const perflibs_int_t part_indx[2 * SUPER_LOWER_NPARTS + 2] = {0, 2, 3,
                                                                4, 5, 5};
  const float rhs[SUPER_LOWER_N] = {1, 3, 6, 10, 10, 13, 19, 1, 2,
                                    3, 5, 5, 12, 60, 5,  51, 10};

  perflibs_spmat_t mat = NULL;
  float *x = NULL;

  CHECK_STATUS(perflibs_spmat_create_supernodal_s(
      &mat, n, n, nsuper, nparts, super_row_ptr, super_col_indx, row_indx,
      col_ptr, vals, part_indx, 0));

  // Standard execution flow: choose the dense separator before the optimize and
  // then solve.
  CHECK_STATUS_EQ(perflibs_spmat_hint(mat, PERFLIBS_SPARSE_HINT_SPSV_STRATEGY,
                                      PERFLIBS_SPARSE_OPERATION_NOTRANS),
                  PERFLIBS_STATUS_INPUT_PARAMETER_ERROR);

  CHECK_STATUS(perflibs_spmat_hint(mat, PERFLIBS_SPARSE_HINT_SPSV_STRATEGY,
                                   PERFLIBS_SPARSE_SPSV_STRAT_SEPARATOR_DENSE));

  CHECK_STATUS(perflibs_spsv_optimize(mat));

  x = (float *)malloc(sizeof(float) * n);
  CHECK_TRUE(x != NULL, "malloc failed");

  CHECK_STATUS(perflibs_spsv_exec_s(PERFLIBS_SPARSE_OPERATION_NOTRANS, mat, x,
                                    alpha, rhs));

  for (perflibs_int_t i = 0; i < n; ++i) {
    CHECK_TRUE(isfinite(x[i]), "solution[%lld] is not finite", test_i64(i));
    CHECK_TRUE(fabsf(x[i] - alpha) <= 1.0e-5f,
               "dense separator solution[%lld] got %.9g expected %.9g",
               test_i64(i), (double)x[i], (double)alpha);
  }
  CHECK_STATUS(perflibs_spmat_destroy(mat));

  // Optimize twice flow: the first optimize marks the separator's diagonal as
  // KNOWN_NON_UNIT, the second converts it to dense, and the dense solve
  // must still honour the flag.
  CHECK_STATUS(perflibs_spmat_create_supernodal_s(
      &mat, n, n, nsuper, nparts, super_row_ptr, super_col_indx, row_indx,
      col_ptr, vals, part_indx, 0));

  CHECK_STATUS(
      perflibs_spmat_hint(mat, PERFLIBS_SPARSE_HINT_SPSV_STRATEGY,
                          PERFLIBS_SPARSE_SPSV_STRAT_SEPARATOR_SPARSE));

  CHECK_STATUS(perflibs_spsv_optimize(mat));

  memset(x, 0, sizeof(*x) * n);

  CHECK_STATUS(perflibs_spsv_exec_s(PERFLIBS_SPARSE_OPERATION_NOTRANS, mat, x,
                                    alpha, rhs));

  for (perflibs_int_t i = 0; i < n; ++i) {
    CHECK_TRUE(isfinite(x[i]), "solution[%lld] is not finite", test_i64(i));
    CHECK_TRUE(fabsf(x[i] - alpha) <= 1.0e-5f,
               "sparse separator solution[%lld] got %.9g expected %.9g",
               test_i64(i), (double)x[i], (double)alpha);
  }

  CHECK_STATUS(perflibs_spmat_hint(mat, PERFLIBS_SPARSE_HINT_SPSV_STRATEGY,
                                   PERFLIBS_SPARSE_SPSV_STRAT_SEPARATOR_DENSE));

  CHECK_STATUS(perflibs_spsv_optimize(mat));

  memset(x, 0, sizeof(*x) * n);

  CHECK_STATUS(perflibs_spsv_exec_s(PERFLIBS_SPARSE_OPERATION_NOTRANS, mat, x,
                                    alpha, rhs));

  for (perflibs_int_t i = 0; i < n; ++i) {
    CHECK_TRUE(isfinite(x[i]), "solution[%lld] is not finite", test_i64(i));
    CHECK_TRUE(fabsf(x[i] - alpha) <= 1.0e-5f,
               "re-optimized dense separator solution[%lld] got %.9g "
               "expected %.9g",
               test_i64(i), (double)x[i], (double)alpha);
  }

  free(x);
  CHECK_STATUS(perflibs_spmat_destroy(mat));
  return EXIT_SUCCESS;
}
