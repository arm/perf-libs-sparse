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

#define SUPER_LOWER_LARGE_N 64
#define SUPER_LOWER_LARGE_NSUPER 8
#define SUPER_LOWER_LARGE_NPARTS 3
#define SUPER_LOWER_LARGE_ROW_INDX 176
#define SUPER_LOWER_LARGE_NNZ 1408

static int test_supernodal_spsm(perflibs_spmat_t mat, const float *rhs,
                                enum perflibs_dense_layout layout) {
  const perflibs_int_t n = SUPER_LOWER_N;
  const perflibs_int_t nrhs = 17;
  const float alpha = 1.1f;
  const perflibs_int_t lda = layout == PERFLIBS_COL_MAJOR ? n : nrhs;
  perflibs_spmat_t X = NULL;
  perflibs_spmat_t Y = NULL;
  float *X_out = NULL;
  perflibs_int_t m_out = 0;
  perflibs_int_t n_out = 0;
  float X_vals[SUPER_LOWER_N * 17] = {0};
  float Y_vals[SUPER_LOWER_N * 17] = {0};
  float expected[SUPER_LOWER_N * 17] = {0};

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
  const perflibs_int_t n = SUPER_LOWER_N;
  const perflibs_int_t nrhs = 3;
  const float alpha = 1.1f;
  const perflibs_int_t lda = layout == PERFLIBS_COL_MAJOR ? n : nrhs;
  perflibs_spmat_t X = NULL;
  perflibs_spmat_t Y = NULL;
  float *X_out = NULL;
  perflibs_int_t m_out = 0;
  perflibs_int_t n_out = 0;
  float X_vals[SUPER_LOWER_N * 3] = {0};
  float Y_vals[SUPER_LOWER_N * 3] = {0};

  for (perflibs_int_t i = 0; i < n; ++i) {
    const float y0 = rhs[i];
    const float y1 = 2.0f * rhs[i];
    const float y2 = -0.5f * rhs[i];
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
    float y_col[SUPER_LOWER_N] = {0};
    float x_col[SUPER_LOWER_N] = {0};
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

static void populate_large_lower_supernodal(perflibs_int_t *super_row_ptr,
                                            perflibs_int_t *super_col_indx,
                                            perflibs_int_t *row_indx,
                                            perflibs_int_t *col_ptr,
                                            float *vals,
                                            perflibs_int_t *part_indx) {
  const perflibs_int_t width = 8;
  const perflibs_int_t sep_start = 48;
  perflibs_int_t row_count = 0;
  perflibs_int_t val_count = 0;

  for (perflibs_int_t sn = 0; sn <= SUPER_LOWER_LARGE_NSUPER; ++sn) {
    super_col_indx[sn] = sn * width;
  }

  super_row_ptr[0] = 0;
  for (perflibs_int_t sn = 0; sn < SUPER_LOWER_LARGE_NSUPER; ++sn) {
    const perflibs_int_t col_start = super_col_indx[sn];
    if (sn < 6) {
      for (perflibs_int_t row = col_start; row < col_start + width; ++row) {
        row_indx[row_count++] = row;
      }
    }
    for (perflibs_int_t row = sep_start; row < SUPER_LOWER_LARGE_N; ++row) {
      row_indx[row_count++] = row;
    }
    super_row_ptr[sn + 1] = row_count;
  }

  col_ptr[0] = 0;
  for (perflibs_int_t sn = 0; sn < SUPER_LOWER_LARGE_NSUPER; ++sn) {
    const perflibs_int_t col_start = super_col_indx[sn];
    for (perflibs_int_t col = col_start; col < col_start + width; ++col) {
      const perflibs_int_t first_row = super_row_ptr[sn];
      const perflibs_int_t last_row = super_row_ptr[sn + 1];
      for (perflibs_int_t row_i = first_row; row_i < last_row; ++row_i) {
        const perflibs_int_t row = row_indx[row_i];
        float val = 0.0f;
        if (row >= col) {
          if (row == col) {
            val = 4.0f + (float)(col % 5);
          } else if (row >= sep_start && col < sep_start) {
            val = 0.03f * (float)(1 + ((row + 3 * col) % 5));
          } else {
            val = 0.02f * (float)(1 + ((row - col) % 7));
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
  const perflibs_int_t n = SUPER_LOWER_LARGE_N;
  const perflibs_int_t nrhs = 5;
  const float alpha = 1.1f;
  const perflibs_int_t lda = layout == PERFLIBS_COL_MAJOR ? n : nrhs;
  perflibs_int_t super_col_indx[SUPER_LOWER_LARGE_NSUPER + 1] = {0};
  perflibs_int_t super_row_ptr[SUPER_LOWER_LARGE_NSUPER + 1] = {0};
  perflibs_int_t row_indx[SUPER_LOWER_LARGE_ROW_INDX] = {0};
  perflibs_int_t col_ptr[SUPER_LOWER_LARGE_N + 1] = {0};
  perflibs_int_t part_indx[2 * SUPER_LOWER_LARGE_NPARTS + 2] = {0};
  float vals[SUPER_LOWER_LARGE_NNZ] = {0};
  float X_vals[SUPER_LOWER_LARGE_N * 5] = {0};
  float Y_vals[SUPER_LOWER_LARGE_N * 5] = {0};
  float *X_out = NULL;
  perflibs_int_t m_out = 0;
  perflibs_int_t n_out = 0;
  perflibs_spmat_t mat = NULL;
  perflibs_spmat_t X = NULL;
  perflibs_spmat_t Y = NULL;

  populate_large_lower_supernodal(super_row_ptr, super_col_indx, row_indx,
                                  col_ptr, vals, part_indx);

  for (perflibs_int_t row = 0; row < n; ++row) {
    for (perflibs_int_t col = 0; col < nrhs; ++col) {
      const float val =
          0.25f * (float)(1 + (row % 7)) + 0.5f * (float)(col + 1);
      if (layout == PERFLIBS_COL_MAJOR) {
        Y_vals[col * n + row] = val;
      } else {
        Y_vals[row * nrhs + col] = val;
      }
    }
  }

  CHECK_STATUS(perflibs_spmat_create_supernodal_s(
      &mat, n, n, SUPER_LOWER_LARGE_NSUPER, SUPER_LOWER_LARGE_NPARTS,
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
    float y_col[SUPER_LOWER_LARGE_N] = {0};
    float x_col[SUPER_LOWER_LARGE_N] = {0};
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
  const perflibs_int_t n = SUPER_LOWER_N;
  const perflibs_int_t nsuper = SUPER_LOWER_NSUPER;
  const perflibs_int_t nparts = SUPER_LOWER_NPARTS;

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

  CHECK_STATUS(perflibs_spmat_create_supernodal_s(
      &mat, n, n, nsuper, nparts, super_row_ptr, super_col_indx, row_indx,
      col_ptr, vals, part_indx, 0));
  CHECK_STATUS(test_supernodal_spsm_matches_repeated_spsv(
      mat, rhs, PERFLIBS_COL_MAJOR, PERFLIBS_SPARSE_OPERATION_NOTRANS, 0));
  CHECK_STATUS(perflibs_spmat_destroy(mat));

  CHECK_STATUS(perflibs_spmat_create_supernodal_s(
      &mat, n, n, nsuper, nparts, super_row_ptr, super_col_indx, row_indx,
      col_ptr, vals, part_indx, 0));
  CHECK_STATUS(test_supernodal_spsm_matches_repeated_spsv(
      mat, rhs, PERFLIBS_ROW_MAJOR, PERFLIBS_SPARSE_OPERATION_NOTRANS, 0));
  CHECK_STATUS(perflibs_spmat_destroy(mat));

  CHECK_STATUS(perflibs_spmat_create_supernodal_s(
      &mat, n, n, nsuper, nparts, super_row_ptr, super_col_indx, row_indx,
      col_ptr, vals, part_indx, 0));
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
