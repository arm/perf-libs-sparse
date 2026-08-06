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

#define SUPER_UPPER_LARGE_N 64
#define SUPER_UPPER_LARGE_NSUPER 8
#define SUPER_UPPER_LARGE_NPARTS 3
#define SUPER_UPPER_LARGE_ROW_INDX 168
#define SUPER_UPPER_LARGE_NNZ 1344

static int test_supernodal_spsm(perflibs_spmat_t mat, const float *rhs,
                                enum perflibs_dense_layout layout) {
  const perflibs_int_t n = SUPER_UPPER_N;
  const perflibs_int_t nrhs = 17;
  const float alpha = 1.1f;
  const perflibs_int_t lda = layout == PERFLIBS_COL_MAJOR ? n : nrhs;
  perflibs_spmat_t X = NULL;
  perflibs_spmat_t Y = NULL;
  float *X_out = NULL;
  perflibs_int_t m_out = 0;
  perflibs_int_t n_out = 0;
  float X_vals[SUPER_UPPER_N * 17] = {0};
  float Y_vals[SUPER_UPPER_N * 17] = {0};
  float expected[SUPER_UPPER_N * 17] = {0};

  for (perflibs_int_t col = 0; col < nrhs; ++col) {
    const float scale = (float)(col % 2 + 1);
    for (perflibs_int_t i = 0; i < n; ++i) {
      if (layout == PERFLIBS_COL_MAJOR) {
        Y_vals[col * n + i] = scale * rhs[i];
      } else {
        Y_vals[i * nrhs + col] = scale * rhs[i];
      }
      expected[col * n + i] = scale * alpha;
    }
  }

  CHECK_STATUS(
      perflibs_spmat_create_dense_s(&X, layout, n, nrhs, lda, X_vals, 0));
  CHECK_STATUS(
      perflibs_spmat_create_dense_s(&Y, layout, n, nrhs, lda, Y_vals, 0));
  CHECK_STATUS(perflibs_spsm_optimize(PERFLIBS_SPARSE_OPERATION_NOTRANS, mat, X,
                                      PERFLIBS_SPARSE_SCALAR_ANY, Y));
  CHECK_STATUS(perflibs_spsm_exec_s(PERFLIBS_SPARSE_OPERATION_NOTRANS, mat, X,
                                    alpha, Y));

  CHECK_STATUS(perflibs_spmat_export_dense_s(X, PERFLIBS_COL_MAJOR, &m_out,
                                             &n_out, &X_out));
  CHECK_TRUE(m_out == n && n_out == nrhs,
             "unexpected exported SpSM solution shape %lld x %lld",
             test_i64(m_out), test_i64(n_out));
  for (perflibs_int_t i = 0; i < n * nrhs; ++i) {
    const float tol = 1.0e-5f * fmaxf(1.0f, fabsf(expected[i]));
    CHECK_TRUE(isfinite(X_out[i]), "spsm solution[%lld] is not finite",
               test_i64(i));
    CHECK_TRUE(fabsf(X_out[i] - expected[i]) <= tol,
               "spsm solution[%lld] got %.9g expected %.9g", test_i64(i),
               (double)X_out[i], (double)expected[i]);
  }

  free(X_out);
  CHECK_STATUS(perflibs_spmat_destroy(X));
  CHECK_STATUS(perflibs_spmat_destroy(Y));
  return EXIT_SUCCESS;
}

static int test_supernodal_spsm_matches_repeated_spsv(
    perflibs_spmat_t mat, const float *rhs, enum perflibs_dense_layout layout,
    enum perflibs_sparse_hint_value trans, int optimize) {
  const perflibs_int_t n = SUPER_UPPER_N;
  const perflibs_int_t nrhs = 3;
  const float alpha = 1.1f;
  const perflibs_int_t lda = layout == PERFLIBS_COL_MAJOR ? n : nrhs;
  perflibs_spmat_t X = NULL;
  perflibs_spmat_t Y = NULL;
  float *X_out = NULL;
  perflibs_int_t m_out = 0;
  perflibs_int_t n_out = 0;
  float X_vals[SUPER_UPPER_N * 3] = {0};
  float Y_vals[SUPER_UPPER_N * 3] = {0};

  for (perflibs_int_t i = 0; i < n; ++i) {
    const float y0 = rhs[i];
    const float y1 = 3.0f * rhs[i];
    const float y2 = -0.25f * rhs[i];
    if (layout == PERFLIBS_COL_MAJOR) {
      Y_vals[i] = y0;
      Y_vals[n + i] = y1;
      Y_vals[2 * n + i] = y2;
    } else {
      Y_vals[i * nrhs] = y0;
      Y_vals[i * nrhs + 1] = y1;
      Y_vals[i * nrhs + 2] = y2;
    }
  }

  CHECK_STATUS(
      perflibs_spmat_create_dense_s(&X, layout, n, nrhs, lda, X_vals, 0));
  CHECK_STATUS(
      perflibs_spmat_create_dense_s(&Y, layout, n, nrhs, lda, Y_vals, 0));
  if (optimize) {
    CHECK_STATUS(
        perflibs_spsm_optimize(trans, mat, X, PERFLIBS_SPARSE_SCALAR_ANY, Y));
  }
  CHECK_STATUS(perflibs_spsm_exec_s(trans, mat, X, alpha, Y));

  CHECK_STATUS(perflibs_spmat_export_dense_s(X, PERFLIBS_COL_MAJOR, &m_out,
                                             &n_out, &X_out));
  CHECK_TRUE(m_out == n && n_out == nrhs,
             "unexpected exported SpSM solution shape %lld x %lld",
             test_i64(m_out), test_i64(n_out));

  for (perflibs_int_t col = 0; col < nrhs; ++col) {
    float y_col[SUPER_UPPER_N] = {0};
    float x_col[SUPER_UPPER_N] = {0};
    for (perflibs_int_t row = 0; row < n; ++row) {
      y_col[row] = layout == PERFLIBS_COL_MAJOR ? Y_vals[col * n + row]
                                                : Y_vals[row * nrhs + col];
    }

    CHECK_STATUS(perflibs_spsv_exec_s(trans, mat, x_col, alpha, y_col));
    for (perflibs_int_t row = 0; row < n; ++row) {
      const perflibs_int_t idx = col * n + row;
      const float tol = 1.0e-4f * fmaxf(1.0f, fabsf(x_col[row]));
      CHECK_TRUE(isfinite(X_out[idx]), "spsm solution[%lld,%lld] is not finite",
                 test_i64(row), test_i64(col));
      CHECK_TRUE(
          fabsf(X_out[idx] - x_col[row]) <= tol,
          "spsm solution[%lld,%lld] got %.9g expected repeated spsv %.9g",
          test_i64(row), test_i64(col), (double)X_out[idx], (double)x_col[row]);
    }
  }

  free(X_out);
  CHECK_STATUS(perflibs_spmat_destroy(X));
  CHECK_STATUS(perflibs_spmat_destroy(Y));
  return EXIT_SUCCESS;
}

static void populate_large_upper_supernodal(perflibs_int_t *super_row_ptr,
                                            perflibs_int_t *super_col_indx,
                                            perflibs_int_t *row_indx,
                                            perflibs_int_t *col_ptr,
                                            float *vals,
                                            perflibs_int_t *part_indx) {
  const perflibs_int_t width = 8;
  const perflibs_int_t sep_end = 16;
  perflibs_int_t row_count = 0;
  perflibs_int_t val_count = 0;

  for (perflibs_int_t sn = 0; sn <= SUPER_UPPER_LARGE_NSUPER; ++sn) {
    super_col_indx[sn] = sn * width;
  }

  super_row_ptr[0] = 0;
  for (perflibs_int_t sn = 0; sn < SUPER_UPPER_LARGE_NSUPER; ++sn) {
    const perflibs_int_t col_start = super_col_indx[sn];
    if (sn < 2) {
      for (perflibs_int_t row = 0; row < col_start + width; ++row) {
        row_indx[row_count++] = row;
      }
    } else {
      for (perflibs_int_t row = 0; row < sep_end; ++row) {
        row_indx[row_count++] = row;
      }
      for (perflibs_int_t row = col_start; row < col_start + width; ++row) {
        row_indx[row_count++] = row;
      }
    }
    super_row_ptr[sn + 1] = row_count;
  }

  col_ptr[0] = 0;
  for (perflibs_int_t sn = 0; sn < SUPER_UPPER_LARGE_NSUPER; ++sn) {
    const perflibs_int_t col_start = super_col_indx[sn];
    for (perflibs_int_t col = col_start; col < col_start + width; ++col) {
      const perflibs_int_t first_row = super_row_ptr[sn];
      const perflibs_int_t last_row = super_row_ptr[sn + 1];
      for (perflibs_int_t row_i = first_row; row_i < last_row; ++row_i) {
        const perflibs_int_t row = row_indx[row_i];
        float val = 0.0f;
        if (row <= col) {
          if (row == col) {
            val = 5.0f + (float)(col % 7);
          } else if (row < sep_end && col >= sep_end) {
            val = 0.025f * (float)(1 + ((row + 2 * col) % 5));
          } else {
            val = 0.015f * (float)(1 + ((col - row) % 7));
          }
        }
        vals[val_count++] = val;
      }
      col_ptr[col + 1] = val_count;
    }
  }

  part_indx[0] = 0;
  part_indx[1] = 1;
  part_indx[2] = 2;
  part_indx[3] = 3;
  part_indx[4] = 4;
  part_indx[5] = 5;
  part_indx[6] = 6;
  part_indx[7] = 7;
}

static int test_large_supernodal_spsm_matches_repeated_spsv(
    enum perflibs_dense_layout layout, enum perflibs_sparse_hint_value trans,
    int optimize) {
  const perflibs_int_t n = SUPER_UPPER_LARGE_N;
  const perflibs_int_t nrhs = 5;
  const float alpha = 1.1f;
  const perflibs_int_t lda = layout == PERFLIBS_COL_MAJOR ? n : nrhs;
  perflibs_int_t super_col_indx[SUPER_UPPER_LARGE_NSUPER + 1] = {0};
  perflibs_int_t super_row_ptr[SUPER_UPPER_LARGE_NSUPER + 1] = {0};
  perflibs_int_t row_indx[SUPER_UPPER_LARGE_ROW_INDX] = {0};
  perflibs_int_t col_ptr[SUPER_UPPER_LARGE_N + 1] = {0};
  perflibs_int_t part_indx[2 * SUPER_UPPER_LARGE_NPARTS + 2] = {0};
  float vals[SUPER_UPPER_LARGE_NNZ] = {0};
  float X_vals[SUPER_UPPER_LARGE_N * 5] = {0};
  float Y_vals[SUPER_UPPER_LARGE_N * 5] = {0};
  float *X_out = NULL;
  perflibs_int_t m_out = 0;
  perflibs_int_t n_out = 0;
  perflibs_spmat_t mat = NULL;
  perflibs_spmat_t X = NULL;
  perflibs_spmat_t Y = NULL;

  populate_large_upper_supernodal(super_row_ptr, super_col_indx, row_indx,
                                  col_ptr, vals, part_indx);

  for (perflibs_int_t row = 0; row < n; ++row) {
    for (perflibs_int_t col = 0; col < nrhs; ++col) {
      const float val =
          0.2f * (float)(1 + (row % 5)) + 0.35f * (float)(col + 1);
      if (layout == PERFLIBS_COL_MAJOR) {
        Y_vals[col * n + row] = val;
      } else {
        Y_vals[row * nrhs + col] = val;
      }
    }
  }

  CHECK_STATUS(perflibs_spmat_create_supernodal_s(
      &mat, n, n, SUPER_UPPER_LARGE_NSUPER, SUPER_UPPER_LARGE_NPARTS,
      super_row_ptr, super_col_indx, row_indx, col_ptr, vals, part_indx, 0));
  CHECK_STATUS(
      perflibs_spmat_create_dense_s(&X, layout, n, nrhs, lda, X_vals, 0));
  CHECK_STATUS(
      perflibs_spmat_create_dense_s(&Y, layout, n, nrhs, lda, Y_vals, 0));

  if (optimize) {
    CHECK_STATUS(
        perflibs_spsm_optimize(trans, mat, X, PERFLIBS_SPARSE_SCALAR_ANY, Y));
  }
  CHECK_STATUS(perflibs_spsm_exec_s(trans, mat, X, alpha, Y));
  CHECK_STATUS(perflibs_spmat_export_dense_s(X, PERFLIBS_COL_MAJOR, &m_out,
                                             &n_out, &X_out));
  CHECK_TRUE(m_out == n && n_out == nrhs,
             "unexpected large SpSM solution shape %lld x %lld",
             test_i64(m_out), test_i64(n_out));

  for (perflibs_int_t col = 0; col < nrhs; ++col) {
    float y_col[SUPER_UPPER_LARGE_N] = {0};
    float x_col[SUPER_UPPER_LARGE_N] = {0};
    for (perflibs_int_t row = 0; row < n; ++row) {
      y_col[row] = layout == PERFLIBS_COL_MAJOR ? Y_vals[col * n + row]
                                                : Y_vals[row * nrhs + col];
    }

    CHECK_STATUS(perflibs_spsv_exec_s(trans, mat, x_col, alpha, y_col));
    for (perflibs_int_t row = 0; row < n; ++row) {
      const perflibs_int_t idx = col * n + row;
      const float tol = 1.0e-4f * fmaxf(1.0f, fabsf(x_col[row]));
      CHECK_TRUE(isfinite(X_out[idx]),
                 "large spsm solution[%lld,%lld] is not finite", test_i64(row),
                 test_i64(col));
      CHECK_TRUE(
          fabsf(X_out[idx] - x_col[row]) <= tol,
          "large spsm solution[%lld,%lld] got %.9g expected repeated spsv %.9g",
          test_i64(row), test_i64(col), (double)X_out[idx], (double)x_col[row]);
    }
  }

  free(X_out);
  CHECK_STATUS(perflibs_spmat_destroy(X));
  CHECK_STATUS(perflibs_spmat_destroy(Y));
  CHECK_STATUS(perflibs_spmat_destroy(mat));
  return EXIT_SUCCESS;
}

int main() {
  const perflibs_int_t m = SUPER_UPPER_M;
  const perflibs_int_t n = SUPER_UPPER_N;
  const perflibs_int_t nsuper = SUPER_UPPER_NSUPER;
  const perflibs_int_t nparts = SUPER_UPPER_NPARTS;

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

  CHECK_STATUS(perflibs_spmat_create_supernodal_s(
      &mat, m, n, nsuper, nparts, super_row_ptr, super_col_indx, row_indx,
      col_ptr, vals, part_indx, 0));

  CHECK_STATUS(test_supernodal_spsm_matches_repeated_spsv(
      mat, rhs, PERFLIBS_COL_MAJOR, PERFLIBS_SPARSE_OPERATION_NOTRANS, 0));
  CHECK_STATUS(test_supernodal_spsm_matches_repeated_spsv(
      mat, rhs, PERFLIBS_ROW_MAJOR, PERFLIBS_SPARSE_OPERATION_NOTRANS, 0));
  CHECK_STATUS(test_supernodal_spsm(mat, rhs, PERFLIBS_COL_MAJOR));
  CHECK_STATUS(test_supernodal_spsm(mat, rhs, PERFLIBS_ROW_MAJOR));
  CHECK_STATUS(test_supernodal_spsm_matches_repeated_spsv(
      mat, rhs, PERFLIBS_COL_MAJOR, PERFLIBS_SPARSE_OPERATION_NOTRANS, 1));
  CHECK_STATUS(test_supernodal_spsm_matches_repeated_spsv(
      mat, rhs, PERFLIBS_ROW_MAJOR, PERFLIBS_SPARSE_OPERATION_NOTRANS, 1));
  CHECK_STATUS(test_supernodal_spsm_matches_repeated_spsv(
      mat, rhs, PERFLIBS_COL_MAJOR, PERFLIBS_SPARSE_OPERATION_TRANS, 1));
  CHECK_STATUS(test_supernodal_spsm_matches_repeated_spsv(
      mat, rhs, PERFLIBS_ROW_MAJOR, PERFLIBS_SPARSE_OPERATION_TRANS, 1));
  CHECK_STATUS(test_supernodal_spsm_matches_repeated_spsv(
      mat, rhs, PERFLIBS_COL_MAJOR, PERFLIBS_SPARSE_OPERATION_CONJTRANS, 1));
  CHECK_STATUS(test_supernodal_spsm_matches_repeated_spsv(
      mat, rhs, PERFLIBS_ROW_MAJOR, PERFLIBS_SPARSE_OPERATION_CONJTRANS, 1));
  CHECK_STATUS(test_large_supernodal_spsm_matches_repeated_spsv(
      PERFLIBS_COL_MAJOR, PERFLIBS_SPARSE_OPERATION_NOTRANS, 0));
  CHECK_STATUS(test_large_supernodal_spsm_matches_repeated_spsv(
      PERFLIBS_ROW_MAJOR, PERFLIBS_SPARSE_OPERATION_NOTRANS, 0));
  CHECK_STATUS(test_large_supernodal_spsm_matches_repeated_spsv(
      PERFLIBS_COL_MAJOR, PERFLIBS_SPARSE_OPERATION_NOTRANS, 1));
  CHECK_STATUS(test_large_supernodal_spsm_matches_repeated_spsv(
      PERFLIBS_ROW_MAJOR, PERFLIBS_SPARSE_OPERATION_NOTRANS, 1));
  CHECK_STATUS(test_large_supernodal_spsm_matches_repeated_spsv(
      PERFLIBS_COL_MAJOR, PERFLIBS_SPARSE_OPERATION_TRANS, 1));
  CHECK_STATUS(test_large_supernodal_spsm_matches_repeated_spsv(
      PERFLIBS_ROW_MAJOR, PERFLIBS_SPARSE_OPERATION_TRANS, 1));
  CHECK_STATUS(test_large_supernodal_spsm_matches_repeated_spsv(
      PERFLIBS_COL_MAJOR, PERFLIBS_SPARSE_OPERATION_CONJTRANS, 1));
  CHECK_STATUS(test_large_supernodal_spsm_matches_repeated_spsv(
      PERFLIBS_ROW_MAJOR, PERFLIBS_SPARSE_OPERATION_CONJTRANS, 1));

  CHECK_STATUS(perflibs_spmat_destroy(mat));
  return EXIT_SUCCESS;
}
