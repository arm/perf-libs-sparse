/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "perflibs_sparse.h"
#include "perflibs_sparse_reproducibility.h"
#include "test_utils.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
static int test_coo_spmv_modes_and_update() {
  const perflibs_int_t row_indx[] = {0, 0, 1, 2, 2};
  const perflibs_int_t col_indx[] = {0, 2, 1, 0, 2};
  const perflibs_doublecomplex_t vals[] = {TEST_Z(1.0, 1.0), TEST_Z(2.0, -1.0),
                                           TEST_Z(-1.0, 2.0), TEST_Z(3.0, 0.0),
                                           TEST_Z(4.0, -2.0)};
  const perflibs_doublecomplex_t x[] = {TEST_Z(1.0, 0.0), TEST_Z(2.0, 0.0),
                                        TEST_Z(3.0, 0.0), TEST_Z(0.0, 0.0)};
  perflibs_doublecomplex_t y[4] = {TEST_Z(0.0, 0.0), TEST_Z(0.0, 0.0),
                                   TEST_Z(0.0, 0.0), TEST_Z(0.0, 0.0)};
  perflibs_spmat_t coo = NULL;
  perflibs_int_t index_base = -1;
  perflibs_int_t m = -1;
  perflibs_int_t n = -1;
  perflibs_int_t nnz = -1;

  CHECK_STATUS(
      perflibs_spmat_create_coo_z(&coo, 4, 4, 5, row_indx, col_indx, vals, 0));
  CHECK_STATUS(perflibs_spmat_query(coo, &index_base, &m, &n, &nnz));
  CHECK_INT_EQ(index_base, 0);
  CHECK_INT_EQ(m, 4);
  CHECK_INT_EQ(n, 4);
  CHECK_INT_EQ(nnz, 5);

  CHECK_STATUS(perflibs_spmv_exec_z(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                    TEST_Z(1.0, 0.0), coo, x, TEST_Z(0.0, 0.0),
                                    y));
  const perflibs_doublecomplex_t expected_notrans[] = {
      TEST_Z(7.0, -2.0), TEST_Z(-2.0, 4.0), TEST_Z(15.0, -6.0),
      TEST_Z(0.0, 0.0)};
  CHECK_DOUBLECOMPLEX_ARRAY(y, expected_notrans, 4, 1e-12);

  y[0] = y[1] = y[2] = y[3] = TEST_Z(0.0, 0.0);
  CHECK_STATUS(perflibs_spmv_exec_z(PERFLIBS_SPARSE_OPERATION_TRANS,
                                    TEST_Z(1.0, 0.0), coo, x, TEST_Z(0.0, 0.0),
                                    y));
  const perflibs_doublecomplex_t expected_trans[] = {
      TEST_Z(10.0, 1.0), TEST_Z(-2.0, 4.0), TEST_Z(14.0, -7.0),
      TEST_Z(0.0, 0.0)};
  CHECK_DOUBLECOMPLEX_ARRAY(y, expected_trans, 4, 1e-12);

  y[0] = y[1] = y[2] = y[3] = TEST_Z(0.0, 0.0);
  CHECK_STATUS(perflibs_spmv_exec_z(PERFLIBS_SPARSE_OPERATION_CONJTRANS,
                                    TEST_Z(1.0, 0.0), coo, x, TEST_Z(0.0, 0.0),
                                    y));
  const perflibs_doublecomplex_t expected_conj_trans[] = {
      TEST_Z(10.0, -1.0), TEST_Z(-2.0, -4.0), TEST_Z(14.0, 7.0),
      TEST_Z(0.0, 0.0)};
  CHECK_DOUBLECOMPLEX_ARRAY(y, expected_conj_trans, 4, 1e-12);

  const perflibs_int_t upd_row = 1;
  const perflibs_int_t upd_col = 1;
  const perflibs_doublecomplex_t upd_val = TEST_Z(5.0, -1.0);
  CHECK_STATUS(perflibs_spmat_update_z(coo, 1, &upd_row, &upd_col, &upd_val));
  y[0] = y[1] = y[2] = y[3] = TEST_Z(0.0, 0.0);
  CHECK_STATUS(perflibs_spmv_exec_z(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                    TEST_Z(1.0, 0.0), coo, x, TEST_Z(0.0, 0.0),
                                    y));
  const perflibs_doublecomplex_t expected_updated[] = {
      TEST_Z(7.0, -2.0), TEST_Z(10.0, -2.0), TEST_Z(15.0, -6.0),
      TEST_Z(0.0, 0.0)};
  CHECK_DOUBLECOMPLEX_ARRAY(y, expected_updated, 4, 1e-12);

  const perflibs_int_t bad_row = 5;
  const perflibs_int_t bad_col = 1;
  const perflibs_doublecomplex_t bad_val = TEST_Z(1.0, 0.0);
  CHECK_STATUS_EQ(perflibs_spmat_update_z(coo, 1, &bad_row, &bad_col, &bad_val),
                  PERFLIBS_STATUS_INPUT_PARAMETER_ERROR);

  CHECK_STATUS(perflibs_spmat_destroy(coo));
  return EXIT_SUCCESS;
}

static int test_coo_norm_and_conjtranspose_export() {
  const perflibs_int_t row_indx[] = {0, 0, 1, 2, 2};
  const perflibs_int_t col_indx[] = {0, 2, 1, 0, 2};
  const perflibs_doublecomplex_t vals[] = {TEST_Z(1.0, 1.0), TEST_Z(2.0, -1.0),
                                           TEST_Z(-1.0, 2.0), TEST_Z(3.0, 0.0),
                                           TEST_Z(4.0, -2.0)};
  const perflibs_doublecomplex_t expected_dense[] = {
      TEST_Z(1.0, -1.0), TEST_Z(0.0, 0.0), TEST_Z(3.0, 0.0),
      TEST_Z(0.0, 0.0),  TEST_Z(0.0, 0.0), TEST_Z(-1.0, -2.0),
      TEST_Z(0.0, 0.0),  TEST_Z(0.0, 0.0), TEST_Z(2.0, 1.0),
      TEST_Z(0.0, 0.0),  TEST_Z(4.0, 2.0), TEST_Z(0.0, 0.0),
      TEST_Z(0.0, 0.0),  TEST_Z(0.0, 0.0), TEST_Z(0.0, 0.0),
      TEST_Z(0.0, 0.0)};
  perflibs_spmat_t coo = NULL;
  perflibs_doublecomplex_t *dense = NULL;
  perflibs_int_t m = -1;
  perflibs_int_t n = -1;
  double inf_norm = -1.0;

  CHECK_STATUS(
      perflibs_spmat_create_coo_z(&coo, 4, 4, 5, row_indx, col_indx, vals, 0));

  CHECK_STATUS(
      perflibs_spnorm_exec_z(coo, PERFLIBS_SPARSE_NORM_INF, &inf_norm));
  CHECK_DOUBLE_NEAR(inf_norm, 3.0 + sqrt(20.0), 1e-12);

  CHECK_STATUS(
      perflibs_sptranspose_exec_z(PERFLIBS_SPARSE_OPERATION_CONJTRANS, coo));
  CHECK_STATUS(
      perflibs_spmat_export_dense_z(coo, PERFLIBS_ROW_MAJOR, &m, &n, &dense));
  CHECK_INT_EQ(m, 4);
  CHECK_INT_EQ(n, 4);
  CHECK_DOUBLECOMPLEX_ARRAY(dense, expected_dense, 16, 1e-12);

  free(dense);
  CHECK_STATUS(perflibs_spmat_destroy(coo));
  return EXIT_SUCCESS;
}

static int test_coo_spmv_reproducibility_enabled_impl() {
  enum {
    m = 64,
    n = 64,
    groups_per_row = 65,
    entries_per_group = 4,
    nnz = m * groups_per_row * entries_per_group
  };
  perflibs_int_t row_indx[nnz];
  perflibs_int_t col_indx[nnz];
  double vals[nnz];
  double x[n];
  double y[m];
  double expected[m];
  perflibs_spmat_t coo = NULL;

  for (perflibs_int_t i = 0; i < n; ++i) {
    x[i] = 1.0;
  }
  for (perflibs_int_t row = 0; row < m; ++row) {
    expected[row] = 0.0;
  }
  for (perflibs_int_t group = 0; group < groups_per_row; ++group) {
    for (perflibs_int_t row = 0; row < m; ++row) {
      const perflibs_int_t base = (group * m + row) * entries_per_group;
      const double small = (double)(row + 1);
      row_indx[base + 0] = row;
      row_indx[base + 1] = row;
      row_indx[base + 2] = row;
      row_indx[base + 3] = row;
      col_indx[base + 0] = 0;
      col_indx[base + 1] = 1;
      col_indx[base + 2] = 2;
      col_indx[base + 3] = 3;
      vals[base + 0] = 1.0e20;
      vals[base + 1] = 1.0;
      vals[base + 2] = -1.0e20;
      vals[base + 3] = small;
    }
  }
  // Deliberately reproduce the serial COO accumulation order, so exact
  // comparison is meaningful for this cancellation-sensitive test.
  for (perflibs_int_t i = 0; i < nnz; ++i) {
    expected[row_indx[i]] += vals[i] * x[col_indx[i]];
  }

  CHECK_STATUS(perflibs_spmat_create_coo_d(&coo, m, n, nnz, row_indx, col_indx,
                                           vals, 0));
  CHECK_STATUS(perflibs_spmat_hint(coo, PERFLIBS_SPARSE_HINT_SPMV_INVOCATIONS,
                                   PERFLIBS_SPARSE_INVOCATIONS_MANY));
  CHECK_STATUS(perflibs_spmv_optimize(coo));

  for (int i = 0; i < 10; ++i) {
    memset(y, 0, sizeof(double) * m);
    CHECK_STATUS(perflibs_spmv_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS, 1.0,
                                      coo, x, 0.0, y));
    CHECK_DOUBLE_ARRAY(y, expected, m, 0.0);
  }

  CHECK_STATUS(perflibs_spmat_destroy(coo));
  return EXIT_SUCCESS;
}

static int test_coo_spmv_reproducibility_enabled() {
  const enum perflibs_sparse_reproducibility_value initial_mode =
      perflibs_sparse_get_reproducibility();

  CHECK_STATUS(perflibs_sparse_set_reproducibility(
      PERFLIBS_SPARSE_REPRODUCIBILITY_ENABLED));
  const int result = test_coo_spmv_reproducibility_enabled_impl();
  CHECK_STATUS(perflibs_sparse_set_reproducibility(initial_mode));
  return result;
}

int main() {
  if (test_coo_spmv_modes_and_update() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (test_coo_norm_and_conjtranspose_export() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (test_coo_spmv_reproducibility_enabled() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
