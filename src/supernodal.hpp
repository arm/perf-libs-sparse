/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include "int.hpp"
#include "pod_vector.hpp"
#include "util.hpp"
#include <inttypes.h>
#include <memory>
#include <vector>

struct perflibs_spmat_top_t;

namespace perflibs::sparse {

template <typename T> struct perflibs_supernodal {
  int64_t m;
  int64_t n;
  int64_t nsuper;
  int64_t nparts;
  int64_t nnz;

  int64_t low_sep;
  int64_t high_sep;
  int64_t index_base;
  perflibs_sparse_matrix_shape_t shape;

  // Pointers to user data
  const perflibs_int_t *super_row_ptr_ptr;
  const perflibs_int_t *super_col_indx_ptr;
  const perflibs_int_t *row_indx_ptr;
  const perflibs_int_t *col_ptr;
  const T *vals_ptr;
  const perflibs_int_t *part_indx_ptr;

  // Triangular solve specific
  std::shared_ptr<perflibs_spmat_top_t> separator;
  std::vector<std::shared_ptr<perflibs_spmat_top_t>> mats_diag;
  std::vector<std::shared_ptr<perflibs_spmat_top_t>> mats_sep;

  perflibs_supernodal()
      : m(-1), n(-1), nsuper(-1), nparts(-1), nnz(-1), low_sep(-1),
        high_sep(-1), index_base(-1), shape(PERFLIBS_SPARSE_SHAPE_RECTANGULAR),
        super_row_ptr_ptr(nullptr), super_col_indx_ptr(nullptr),
        row_indx_ptr(nullptr), col_ptr(nullptr), vals_ptr(nullptr),
        part_indx_ptr(nullptr), separator(nullptr) {}

  perflibs_supernodal(int64_t m, int64_t n, int64_t nsuper, int64_t nparts,
                      int64_t nnz, perflibs_sparse_matrix_shape_t shape,
                      const perflibs_int_t *super_row_ptr,
                      const perflibs_int_t *super_col_indx,
                      const perflibs_int_t *row_indx,
                      const perflibs_int_t *col_ptr, const T *vals,
                      const perflibs_int_t *part_indx);

  perflibs_supernodal &operator=(const perflibs_supernodal &other);
  perflibs_supernodal(const perflibs_supernodal &other) { *this = other; }
  perflibs_supernodal &operator=(perflibs_supernodal &&other) = default;
  perflibs_supernodal(perflibs_supernodal &&other) = default;

  void printer() const {
    fprintf(stderr, "shape is ");
    if (shape == PERFLIBS_SPARSE_SHAPE_LOWER_TRIANGULAR) {
      fprintf(stderr, "LOWER\n");
    } else {
      fprintf(stderr, "UPPER\n");
    }
    fprintf(stderr, "low_sep = %" PRId64 "\n", low_sep);
    fprintf(stderr, "high_sep = %" PRId64 "\n", high_sep);
    fprintf(stderr, "nparts = %" PRId64 "\n", nparts);
    fprintf(stderr, "nsuper = %" PRId64 "\n", nsuper);
    fprintf(stderr, "index_base = %" PRId64 "\n", index_base);
    fprintf(stderr, "\nsuper_col_indx (start column indices for supernodes): ");
    for (int i = 0; i < nsuper + 1; i++)
      fprintf(stderr, "%" PRId64 " ", (int64_t)super_col_indx_ptr[i]);
    fprintf(stderr, "\n\nsuper_row_ptr (ptr into row_indx to the start of each "
                    "supernode): ");
    for (int i = 0; i < nsuper + 1; i++)
      fprintf(stderr, "%" PRId64 " ", (int64_t)super_row_ptr_ptr[i]);
    fprintf(stderr, "\n\nrow_indx (indices of non-zeros in each supernode): ");
    for (int i = 0; i < super_row_ptr_ptr[nsuper] - super_row_ptr_ptr[0]; i++)
      fprintf(stderr, "%" PRId64 " ", (int64_t)row_indx_ptr[i]);
    fprintf(stderr, "\n\ncol_ptr (ptr into vals giving the start of each "
                    "column (incl. padding)): ");
    for (int i = 0; i < n + 1; i++)
      fprintf(stderr, "%" PRId64 " ", (int64_t)row_indx_ptr[i]);
    fprintf(stderr, "\n\npart_indx (start and end supernode id, inclusive in "
                    "supernode index space): ");
    for (int i = 0; i < 2 * nparts + 2; i++)
      fprintf(stderr, "%" PRId64 " ", (int64_t)part_indx_ptr[i]);
    fprintf(stderr, "\n\n");
  }
};
template <typename T>
perflibs_status_t fill_initial_data_supernodal(
    perflibs_spmat_top_t *A, perflibs_int_t m, perflibs_int_t n,
    perflibs_int_t nsuper, perflibs_int_t nparts,
    const perflibs_int_t *super_row_ptr, const perflibs_int_t *super_col_indx,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_ptr,
    const T *vals, const perflibs_int_t *part_indx, const bool no_copy);

/*
 * Performs a triangular solve of a system of linear equations supplied as a
 * sparse matrix and a dense RHS vector. The solution is written into a separate
 * dense vector.
 * @f[ op(A) x = \alpha y @f]
 * The matrix @f$ A @f$ is supplied in supernodal format
 * @param [in]  supernodal  The matrix in supernodal format
 * @param [in]  trans  The transpose operation to apply
 * @param [out] x      The output vector @f$ x @f$ to be solved for
 * @param [in]  y      The input RHS vector.
 * @param [in]  alpha  Scalar @f$ \alpha @f$ to be multiplied to the RHS vector
 * y
 */
template <typename T>
void spsv_supernodal(perflibs_supernodal<T> &supernodal,
                     perflibs_sparse_hint_value trans, T *x, const T *y,
                     T alpha);

/*
 * Performs a triangular solve for multiple dense RHS vectors using a
 * supernodal matrix.
 * @f[ op(A) X = \alpha Y @f]
 * The matrix @f$ A @f$ is supplied in supernodal format
 * @param [in]  supernodal    The matrix in supernodal format
 * @param [in]  trans         The transpose operation to apply
 * @param [out] X             The output dense matrix @f$ X @f$ to be solved for
 * @param [in]  x_stride_row  Row stride for the output dense matrix @f$ X @f$
 * @param [in]  x_stride_col  Column stride for the output dense matrix @f$ X
 * @f$
 * @param [in]  alpha         Scalar @f$ \alpha @f$ to be multiplied to the RHS
 * matrix @f$ Y @f$
 * @param [in]  Y             The input dense RHS matrix
 * @param [in]  y_stride_row  Row stride for the input dense RHS matrix @f$ Y
 * @f$
 * @param [in]  y_stride_col  Column stride for the input dense RHS matrix
 * @f$ Y @f$
 * @param [in]  nrhs          The number of RHS vectors
 */
template <typename T>
void spsm_supernodal(perflibs_supernodal<T> &supernodal,
                     perflibs_sparse_hint_value trans, T *X,
                     perflibs_int_t x_stride_row, perflibs_int_t x_stride_col,
                     T alpha, const T *Y, perflibs_int_t y_stride_row,
                     perflibs_int_t y_stride_col, perflibs_int_t nrhs);
} // namespace perflibs::sparse
