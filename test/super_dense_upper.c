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

#define SUPER_UPPER_M 15
#define SUPER_UPPER_N 15
#define SUPER_UPPER_NNZ 73
#define SUPER_UPPER_NSUPER 6
#define SUPER_UPPER_NPARTS 2

int main() {
  /*
      Upper-triangular supernodal example from super_upper.c, solved with the
      separator converted to dense (cblas_trsv).
      The right-hand side is chosen so the solution vector is all ones.
  */
  const perflibs_int_t m = SUPER_UPPER_M;
  const perflibs_int_t n = SUPER_UPPER_N;
  const perflibs_int_t nsuper = SUPER_UPPER_NSUPER;
  const perflibs_int_t nparts = SUPER_UPPER_NPARTS;
  const float alpha = 1.1f;

  const perflibs_int_t super_col_indx[SUPER_UPPER_NSUPER + 1] = {0,  3,  6, 9,
                                                                 12, 13, 15};
  const perflibs_int_t super_row_ptr[SUPER_UPPER_NSUPER + 1] = {0,  3,  7, 14,
                                                                19, 25, 30};
  const perflibs_int_t row_indx[30] = {0, 1, 2,  0,  3,  4, 5, 0,  2,  4,
                                       5, 6, 7,  8,  0,  1, 9, 10, 11, 0,
                                       2, 9, 10, 11, 12, 0, 2, 12, 13, 14};
  const perflibs_int_t col_ptr[SUPER_UPPER_N + 1] = {
      0, 3, 6, 9, 13, 17, 21, 28, 35, 42, 47, 52, 57, 63, 68, 73};
  const float vals[SUPER_UPPER_NNZ] = {
      3, 0, 0, 1, 2, 0, 2, 1, 1, 3, 4, 0, 0, 4, 3, 3, 0, 5, 2, 2, 2, 6, 1, 1, 1,
      1, 0, 0, 7, 1, 1, 1, 1, 1, 0, 8, 1, 1, 1, 1, 1, 1, 9, 1, 1, 0, 0, 1, 2, 6,
      2, 0, 2, 3, 5, 4, 3, 3, 1, 4, 4, 4, 3, 4, 1, 2, 2, 0, 5, 1, 1, 1, 1};
  const perflibs_int_t part_indx[2 * SUPER_UPPER_NPARTS + 2] = {0, 0, 1,
                                                                2, 3, 5};
  const float rhs[SUPER_UPPER_N] = {63, 9,  7,  9, 8, 5, 3, 2,
                                    1,  16, 10, 7, 6, 3, 1};

  perflibs_spmat_t mat = NULL;
  float *x = NULL;

  CHECK_STATUS(perflibs_spmat_create_supernodal_s(
      &mat, m, n, nsuper, nparts, super_row_ptr, super_col_indx, row_indx,
      col_ptr, vals, part_indx, 0));

  // Standard execution flow: choose the dense separator before the optimize and
  // then solve.
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
      &mat, m, n, nsuper, nparts, super_row_ptr, super_col_indx, row_indx,
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
