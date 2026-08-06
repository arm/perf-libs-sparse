/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "supernodal.hpp"
#include "compressed_sparse_columns.hpp"
#include "compressed_sparse_rows.hpp"
#include "convert.hpp"
#include "int.hpp"
#include "matmul.hpp"
#include "matvec.hpp"
#include "object_helpers.hpp"
#include "solve.hpp"
#include "util.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <inttypes.h>
#include <limits>
#include <map>
#include <memory>
#include <tuple>
#include <vector>

namespace perflibs::sparse {

constexpr static bool debug = false;

template <typename T> struct supernodal_block_arrays {
  perflibs::sparse::pod_vector<T> diag_csc_vals;
  perflibs::sparse::pod_vector<perflibs_int_t> diag_csc_row_indx;
  perflibs::sparse::pod_vector<perflibs_int_t> diag_csc_col_ptr;

  perflibs::sparse::pod_vector<T> sep_csc_vals;
  perflibs::sparse::pod_vector<perflibs_int_t> sep_csc_row_indx;
  perflibs::sparse::pod_vector<perflibs_int_t> sep_csc_col_ptr;
};

template <typename T>
inline void print_sub_matrix(std::shared_ptr<perflibs_spmat_top_t> &mat) {
  fprintf(stderr, "copying matrix\n");
  auto cpy_ptr = spmat_copy<T>(mat.get());
  auto cpy = reinterpret_cast<perflibs_spmat_impl_t<T> *>(cpy_ptr->impl);
  assert(cpy->spmat_format == perflibs_format_csc);
  fprintf(stderr, "convert from CSC:\n");
  fprintf(stderr, "m = %" PRId64 "\n", cpy->csc.m);
  fprintf(stderr, "n = %" PRId64 "\n", cpy->csc.n);
  auto index_base = cpy->csc.col_ptr[0];
  for (int64_t i = 0; i < cpy->csc.m; i++) {
    fprintf(stderr, "Col %" PRId64 " (%" PRId64 "-%" PRId64 "): ", i,
            (int64_t)cpy->csc.col_ptr[i], (int64_t)cpy->csc.col_ptr[i + 1]);
    for (int j = cpy->csc.col_ptr[i] - index_base;
         j < cpy->csc.col_ptr[i + 1] - index_base; j++) {
      if constexpr (std::is_same_v<float, T> || std::is_same_v<double, T>) {
        fprintf(stderr, "%" PRId64 ":%f ", (int64_t)cpy->csc.row_indx[j],
                cpy->csc.vals[j]);
      } else {
        fprintf(stderr, "%" PRId64 ":(%f,%f) ", (int64_t)cpy->csc.row_indx[j],
                cpy->csc.vals[j].real(), cpy->csc.vals[j].imag());
      }
    }
    fprintf(stderr, "\n");
  }
  fprintf(stderr, "converting matrix\n");
  [[maybe_unused]] auto stat = convert(perflibs_format_dense, cpy);
  assert(stat == PERFLIBS_STATUS_SUCCESS);
  fprintf(stderr, "matrix:\n");
  cpy->dense.printer();
  fprintf(stderr, "\n\n");
}

template <typename T> void print_supernode(perflibs_spmat_impl_t<T> *impl) {
  // First let's print the data structure as we have transformed it
  fprintf(stderr, "Printing extracted diagonal matrices...\n");
  for (auto &mat : impl->supernodal.mats_diag) {
    print_sub_matrix<T>(mat);
  }
  fprintf(stderr, "Printing extracted pre-separator matrices...\n");
  for (auto &mat : impl->supernodal.mats_sep) {
    print_sub_matrix<T>(mat);
  }
  fprintf(stderr, "Printing extracted separator matrix...\n");
  print_sub_matrix<T>(impl->supernodal.separator);

  fprintf(stderr, "Supernodal matrix as passed from user...\n");
  fprintf(stderr, "converting matrix\n");
  [[maybe_unused]] auto stat =
      perflibs::sparse::convert(perflibs_format_dense, impl);
  fprintf(stderr, "Original matrix:\n");
  impl->dense.printer();
  assert(stat == PERFLIBS_STATUS_SUCCESS);
}
template void print_supernode<float>(perflibs_spmat_impl_t<float> *impl);
template void print_supernode<double>(perflibs_spmat_impl_t<double> *impl);
template void print_supernode<std::complex<float>>(
    perflibs_spmat_impl_t<std::complex<float>> *impl);
template void print_supernode<std::complex<double>>(
    perflibs_spmat_impl_t<std::complex<double>> *impl);

template <typename T>
perflibs_status_t
get_supernodal_shape(perflibs_int_t nsuper, const perflibs_int_t *super_row_ptr,
                     const perflibs_int_t *super_col_indx,
                     const perflibs_int_t *row_indx,
                     const perflibs_int_t *col_ptr, const T *vals,
                     perflibs_sparse_matrix_shape_t &shape) {

  // Look for evidence of the matrix being either lower or upper triangular.
  // Stop once we have an element which decides the question.
  // Work down the list of supernodes, but only look for non-zeros outside of
  // the square centered on the diagonal which is of width equal to the
  // supernode width. Elements within this square could be padding and thus
  // cannot be used to make a decision either way about the shape of the matrix
  const perflibs_int_t index_base = super_row_ptr[0];
  for (perflibs_int_t i = 0; i < nsuper; i++) {
    perflibs_int_t col_indx = super_col_indx[i];
    perflibs_int_t width = super_col_indx[i + 1] - super_col_indx[i] - 1;
    for (perflibs_int_t j = super_row_ptr[i] - index_base;
         j < super_row_ptr[i + 1] - index_base; j++) {
      if (row_indx[j] > col_indx + width) {
        shape = PERFLIBS_SPARSE_SHAPE_LOWER_TRIANGULAR;
        return PERFLIBS_STATUS_SUCCESS;
      } else if (row_indx[j] < col_indx - width) {
        shape = PERFLIBS_SPARSE_SHAPE_UPPER_TRIANGULAR;
        return PERFLIBS_STATUS_SUCCESS;
      }
    }
  }

  // Not enough information to decide - return error
  return PERFLIBS_STATUS_EXECUTION_FAILURE;
}

std::tuple<perflibs_int_t, perflibs_int_t>
get_separator_indices(perflibs_int_t m, perflibs_int_t nparts,
                      const perflibs_int_t *super_col_indx,
                      const perflibs_int_t *part_indx,
                      perflibs_sparse_matrix_shape_t shape) {
  // Discovers the highest and lowest row indx of separator, row_indx must be
  // ordered.
  perflibs_int_t low_sep;
  perflibs_int_t high_sep;
  perflibs_int_t index_base = part_indx[0];

  if (shape == PERFLIBS_SPARSE_SHAPE_UPPER_TRIANGULAR) {
    auto nrows = super_col_indx[part_indx[2] - index_base] -
                 super_col_indx[part_indx[0] -
                                index_base]; // the separator must be square, so
                                             // just look at column indices
    low_sep = index_base;
    high_sep = index_base + nrows - 1;
  } else {
    auto nrows = super_col_indx[part_indx[2 * nparts + 1] + 1 - index_base] -
                 super_col_indx[part_indx[2 * nparts] - index_base];
    high_sep = m - 1 + index_base;
    low_sep = m + index_base - nrows;
  }

  return {low_sep, high_sep};
}

template <typename T>
inline void populate_lt_arrays(
    perflibs_int_t &vals_off, perflibs_int_t &diag_nnz, perflibs_int_t &sep_nnz,
    perflibs_int_t sep_bound, perflibs_int_t col_indx, perflibs_int_t row_indx,
    perflibs_int_t super_col_indx, perflibs_int_t index_base, const T *vals,
    perflibs::sparse::pod_vector<perflibs_int_t> &diag_ridx,
    perflibs::sparse::pod_vector<T> &diag_vals,
    perflibs::sparse::pod_vector<perflibs_int_t> &sep_ridx,
    perflibs::sparse::pod_vector<T> &sep_vals) {
  // discard padding (lower triangular here) and ignore indices in the same rows
  // as the separator
  if (row_indx < sep_bound && row_indx >= col_indx &&
      vals[vals_off] != (T)0) { // Needs low sep bound
    diag_ridx[diag_nnz] = row_indx - super_col_indx + index_base;
    diag_vals[diag_nnz++] = vals[vals_off];
  } else if (row_indx >= sep_bound && row_indx >= col_indx &&
             vals[vals_off] != (T)0) {
    sep_ridx[sep_nnz] = row_indx - sep_bound + index_base;
    sep_vals[sep_nnz++] = vals[vals_off];
  }

  vals_off++; // always increment!
}

template <typename T>
inline void populate_ut_arrays(
    perflibs_int_t &vals_off, perflibs_int_t &diag_nnz, perflibs_int_t &sep_nnz,
    perflibs_int_t sep_bound, perflibs_int_t col_indx, perflibs_int_t row_indx,
    perflibs_int_t super_col_indx, perflibs_int_t index_base, const T *vals,
    perflibs::sparse::pod_vector<perflibs_int_t> &diag_ridx,
    perflibs::sparse::pod_vector<T> &diag_vals,
    perflibs::sparse::pod_vector<perflibs_int_t> &sep_ridx,
    perflibs::sparse::pod_vector<T> &sep_vals) {
  // discard padding (upper triangular here) and ignore indices in the same rows
  // as the separator
  if (row_indx > sep_bound && row_indx <= col_indx &&
      vals[vals_off] != (T)0) { // Needs high sep bound
    diag_ridx[diag_nnz] = row_indx - super_col_indx + index_base;
    diag_vals[diag_nnz++] = vals[vals_off];
  } else if (row_indx <= sep_bound && row_indx <= col_indx &&
             vals[vals_off] != (T)0) {
    sep_ridx[sep_nnz] = row_indx; // Sep low bound is 0
    sep_vals[sep_nnz++] = vals[vals_off];
  }

  vals_off++;
}

inline std::tuple<perflibs_int_t, perflibs_int_t> get_vec_sizes(
    perflibs_int_t part_id, perflibs_int_t sep_bound, perflibs_int_t index_base,
    const perflibs_int_t *part_indx, const perflibs_int_t *super_row_ptr,
    const perflibs_int_t *super_col_indx, const perflibs_int_t *row_indx,
    const perflibs_int_t *col_ptr, perflibs_sparse_matrix_shape_t shape) {

  // Helper function to determine diagonal and separator block's vector sizing,
  // row_indx must be in ascending order. Per supernode, the proportion of
  // values is correlated to the proportion of row indx
  perflibs_int_t diag_vec_size = 0;
  perflibs_int_t sep_vec_size = 0;
  for (perflibs_int_t super_id = part_indx[2 * part_id] - index_base;
       super_id < part_indx[2 * part_id + 1] + 1 - index_base; super_id++) {
    // in supernode space
    perflibs_int_t nrows =
        super_row_ptr[super_id + 1] - super_row_ptr[super_id];
    // Calculate proportion of rows within separator
    perflibs_int_t pos_indx = nrows;
    for (perflibs_int_t i = 0; i < nrows; i++) {
      if (row_indx[super_row_ptr[super_id] + i - index_base] >= sep_bound &&
          shape == PERFLIBS_SPARSE_SHAPE_LOWER_TRIANGULAR) {
        pos_indx = i;
        break;
      }

      if (row_indx[super_row_ptr[super_id] + i - index_base] > sep_bound &&
          shape == PERFLIBS_SPARSE_SHAPE_UPPER_TRIANGULAR) {
        pos_indx = i;
        break;
      }
    }
    perflibs_int_t num_vals =
        col_ptr[super_col_indx[super_id + 1] - index_base] -
        col_ptr[super_col_indx[super_id] - index_base];
    double num_vals_g1 =
        std::ceil(double(num_vals) * (double(pos_indx + 1) / nrows));
    double num_vals_g2 =
        std::ceil(double(num_vals) * (double(nrows - pos_indx + 1) / nrows));

    if (shape == PERFLIBS_SPARSE_SHAPE_LOWER_TRIANGULAR) {
      diag_vec_size += num_vals_g1;
      sep_vec_size += num_vals_g2;
    } else {
      diag_vec_size += num_vals_g2;
      sep_vec_size += num_vals_g1;
    }
  }

  return {diag_vec_size, sep_vec_size};
}

template <typename T>
inline supernodal_block_arrays<T> populate_block_arrays(
    const perflibs_int_t *part_indx, const perflibs_int_t *super_row_ptr,
    const perflibs_int_t *super_col_indx, const perflibs_int_t *row_indx,
    const perflibs_int_t *col_ptr, const T *vals, const perflibs_int_t part_id,
    perflibs_int_t &vals_off, perflibs_int_t sep_bound, perflibs_int_t ncols,
    perflibs_int_t nparts, perflibs_sparse_matrix_shape_t shape) {

  // Populate internal arrays mats_diag and mats_sep.
  // Preserve index base
  supernodal_block_arrays<T> ret;

  auto index_base = super_row_ptr[0];
  auto first_sn = part_indx[2 * part_id] - index_base;
  auto last_sn = part_indx[2 * part_id + 1] + 1 - index_base;

  // Populate for parts indx instead
  auto [diag_vec_size, sep_vec_size] =
      get_vec_sizes(part_id, sep_bound, index_base, part_indx, super_row_ptr,
                    super_col_indx, row_indx, col_ptr, shape);

  ret.diag_csc_row_indx.resize(diag_vec_size);
  ret.diag_csc_vals.resize(diag_vec_size);
  ret.diag_csc_col_ptr.resize(ncols + 1);

  ret.sep_csc_row_indx.resize(sep_vec_size);
  ret.sep_csc_vals.resize(sep_vec_size);
  ret.sep_csc_col_ptr.resize(ncols + 1);

  perflibs_int_t diag_nnz = 0; // nnz in the new csc matrix
  perflibs_int_t sep_nnz = 0;
  ret.diag_csc_col_ptr[0] = index_base;
  ret.sep_csc_col_ptr[0] = index_base;

  auto super_col = super_col_indx[part_indx[2 * part_id] -
                                  index_base]; // used for relative row indices
  for (perflibs_int_t super_id = first_sn; super_id < last_sn; super_id++) {
    auto first_col = super_col_indx[super_id];
    auto last_col = super_col_indx[super_id + 1];
    auto first_row = super_row_ptr[super_id] - index_base;
    auto last_row = super_row_ptr[super_id + 1] - index_base;

    for (perflibs_int_t j = first_col; j < last_col; j++) {
      for (perflibs_int_t jj = first_row; jj < last_row; jj++) {
        if (shape == PERFLIBS_SPARSE_SHAPE_LOWER_TRIANGULAR) {
          populate_lt_arrays(vals_off, diag_nnz, sep_nnz, sep_bound, j,
                             row_indx[jj], super_col, index_base, vals,
                             ret.diag_csc_row_indx, ret.diag_csc_vals,
                             ret.sep_csc_row_indx, ret.sep_csc_vals);
        } else {
          populate_ut_arrays(vals_off, diag_nnz, sep_nnz, sep_bound, j,
                             row_indx[jj], super_col, index_base, vals,
                             ret.diag_csc_row_indx, ret.diag_csc_vals,
                             ret.sep_csc_row_indx, ret.sep_csc_vals);
        }
      }
      assert(sep_nnz < sep_vec_size);
      assert(diag_nnz < diag_vec_size);
      // Prepare for next iteration
      ret.diag_csc_col_ptr[j - super_col + 1] = diag_nnz + index_base;
      ret.sep_csc_col_ptr[j - super_col + 1] = sep_nnz + index_base;
    }
  }
  return ret;
}

template <typename T>
perflibs_status_t create_csc_separator(
    std::shared_ptr<perflibs_spmat_top_t> &separator, perflibs_int_t nparts,
    perflibs_int_t index_base, const perflibs_int_t *part_indx,
    const perflibs_int_t *super_row_ptr, const perflibs_int_t *super_col_indx,
    const perflibs_int_t *row_indx, const T *vals, perflibs_int_t vals_off,
    perflibs_int_t sep_bound, perflibs_sparse_matrix_shape_t shape) {

  perflibs_int_t sn_part_id =
      shape == PERFLIBS_SPARSE_SHAPE_LOWER_TRIANGULAR ? nparts : 0;

  const auto first_sn = part_indx[2 * sn_part_id] - index_base;
  const auto last_sn = part_indx[2 * sn_part_id + 1] + 1 - index_base;
  if (debug) {
    fprintf(stderr, "separator first sn = %" PRId64 "\n", (int64_t)first_sn);
    fprintf(stderr, "separator last sn = %" PRId64 "\n", (int64_t)last_sn);
  }

  // Dim needs to be 64-bits here to handle large problems. See the sizing of
  // csc_row_indx below.
  const int64_t dim = super_col_indx[last_sn] - super_col_indx[first_sn];
  if (debug)
    fprintf(stderr, "dim of separator = %" PRId64 "\n", dim);

  perflibs::sparse::pod_vector<perflibs_int_t> csc_col_ptr(dim + 1);
  perflibs::sparse::pod_vector<perflibs_int_t> csc_row_indx(dim * (dim + 1) /
                                                            2);
  perflibs::sparse::pod_vector<T> csc_vals(dim * (dim + 1) / 2);

  perflibs_int_t nnz = 0;
  csc_col_ptr[0] = index_base; // Preserve index_base
  for (perflibs_int_t sn = first_sn; sn < last_sn; sn++) {
    const auto first_col = super_col_indx[sn];
    const auto last_col = super_col_indx[sn + 1];

    for (perflibs_int_t col = first_col; col < last_col; col++) {
      const auto first_row = super_row_ptr[sn] - index_base;
      const auto last_row = super_row_ptr[sn + 1] - index_base;

      if (debug)
        fprintf(stderr, "processing col: %" PRId64 " vals_off = %" PRId64 "\n",
                (int64_t)(col - first_col), (int64_t)vals_off);
      for (perflibs_int_t row_i = first_row; row_i < last_row; row_i++) {
        // discard padding
        if ((shape == PERFLIBS_SPARSE_SHAPE_LOWER_TRIANGULAR &&
             row_indx[row_i] >= col) ||
            (shape == PERFLIBS_SPARSE_SHAPE_UPPER_TRIANGULAR &&
             row_indx[row_i] <= col)) {
          csc_row_indx[nnz] = row_indx[row_i] - sep_bound + index_base;
          csc_vals[nnz++] = vals[vals_off];
          if constexpr (std::is_same_v<float, T> || std::is_same_v<double, T>) {
            if (debug)
              fprintf(stderr,
                      "saving val %" PRId64 " = %f into csc_vals %" PRId64
                      " = %f\n",
                      (int64_t)vals_off, vals[vals_off], (int64_t)(nnz - 1),
                      csc_vals[nnz - 1]);
          }
        } else {
          if (debug)
            fprintf(stderr,
                    "row_indx[%" PRId64 "] = %" PRId64 " < col = %" PRId64
                    " vals_off = %" PRId64 "\n",
                    (int64_t)row_i, (int64_t)row_indx[row_i], (int64_t)col,
                    (int64_t)vals_off);
        }
        vals_off++;
      }
      csc_col_ptr[col - super_col_indx[first_sn] + 1] =
          nnz + index_base; // + 1 since first element already filled
    }
  }

  if (debug) {
    int64_t newnnz = csc_col_ptr[dim] - csc_col_ptr[0];
    fprintf(stderr, "nnz = %" PRId64 "\n", newnnz);
    fprintf(stderr, "csc:\n");
    if constexpr (std::is_same_v<float, T> || std::is_same_v<double, T>) {
      for (int64_t i = 0; i < dim; i++) {
        fprintf(stderr, "%" PRId64 "-%" PRId64 " Col %" PRId64 ":\n",
                (int64_t)csc_col_ptr[i], (int64_t)csc_col_ptr[i + 1], i);
        for (int64_t j = csc_col_ptr[i]; j < csc_col_ptr[i + 1]; j++) {
          fprintf(stderr, "%" PRId64 ":%f, ",
                  (int64_t)csc_row_indx[j - index_base],
                  csc_vals[j - index_base]);
          assert(csc_row_indx[j - index_base] - index_base < dim);
        }
        fprintf(stderr, "\n");
      }
    }
    fprintf(stderr, "calling to create csc matrix for separator\n");
  }
  perflibs_spmat_t tmp_sep_ptr = nullptr;
  auto stat = create_spmat_top_csc(&tmp_sep_ptr, dim, dim, csc_row_indx.data(),
                                   csc_col_ptr.data(), csc_vals.data(), 0);
  separator.reset(tmp_sep_ptr);

  return stat;
}

template <typename T>
perflibs_status_t populate_supernodal_solve(
    std::vector<std::shared_ptr<perflibs_spmat_top_t>> &mats_diag,
    std::vector<std::shared_ptr<perflibs_spmat_top_t>> &mats_sep,
    std::shared_ptr<perflibs_spmat_top_t> &separator, perflibs_int_t m,
    perflibs_int_t n, perflibs_int_t nsuper, perflibs_int_t nparts,
    perflibs_int_t low_sep, perflibs_int_t high_sep,
    perflibs_sparse_matrix_shape_t shape, const perflibs_int_t *super_row_ptr,
    const perflibs_int_t *super_col_indx, const perflibs_int_t *row_indx,
    const perflibs_int_t *col_ptr, const T *vals,
    const perflibs_int_t *part_indx) {

  // Approach outline:
  // 1. Create an array of diagonal CSC matrices which can be solved in
  // parallel.
  // 2. Create another array of CSC matrices representing the rectangular blocks
  // preceding the separator
  // 3. Create a final CSC matrix for the separator.

  if (debug)
    fprintf(stderr, "In populate_supernodal_solve\n");
  auto index_base = col_ptr[0];

  auto sep_bound =
      shape == PERFLIBS_SPARSE_SHAPE_LOWER_TRIANGULAR
          ? low_sep
          : high_sep; // Used for determining which value falls within separator
  if (debug)
    fprintf(stderr, "sep_bound = %" PRId64 "\n", (int64_t)sep_bound);
  auto isUT = shape == PERFLIBS_SPARSE_SHAPE_UPPER_TRIANGULAR;

  // For each part, create a csc matrix for the diagonal and another for the
  // separator blocks and add them to the arrays.
  int gstat = 0;
#pragma omp parallel for shared(gstat)
  for (perflibs_int_t part_id = isUT; part_id < nparts + isUT;
       part_id++) { // Skip first supernode if UT

    perflibs_int_t vals_off =
        col_ptr[super_col_indx[part_indx[2 * part_id] - index_base] -
                index_base] -
        index_base;

    perflibs_int_t part_ncols =
        super_col_indx[part_indx[2 * part_id + 1] - index_base + 1] -
        super_col_indx[part_indx[2 * part_id] - index_base];
    if (debug) {
      fprintf(stderr, "part_ncols is %" PRId64 "\n", (int64_t)part_ncols);
      fprintf(stderr, "start panel is %" PRId64 "\n",
              (int64_t)part_indx[2 * part_id]);
      fprintf(stderr, "end panel is %" PRId64 "\n",
              (int64_t)part_indx[2 * part_id + 1]);
      fprintf(stderr,
              "populating matrices for part %" PRId64 " of dim %" PRId64
              "...\n",
              (int64_t)part_id, (int64_t)part_ncols);
    }
    auto block_arr = populate_block_arrays(
        part_indx, super_row_ptr, super_col_indx, row_indx, col_ptr, vals,
        part_id, vals_off, sep_bound, part_ncols, nparts, shape);
    if (debug)
      fprintf(stderr, "----------> vals_off = %" PRId64 "\n",
              (int64_t)vals_off);

    // Create the csc matrix for the diagonal block
    perflibs_spmat_t tmp_diag_mat_ptr = nullptr;
    if (debug) {
      fprintf(stderr, "creating diagonal as csc...\n");
      fprintf(stderr, "part_ncols  = %" PRId64 "\n", (int64_t)part_ncols);
      fprintf(stderr, "row_indx.size = %zu\n",
              block_arr.diag_csc_row_indx.size());
      fprintf(stderr, "col_ptr.size = %zu\n",
              block_arr.diag_csc_col_ptr.size());
      fprintf(stderr, "vals.size = %zu\n", block_arr.diag_csc_vals.size());
    }
    bool is_null = (block_arr.diag_csc_col_ptr[part_ncols] -
                    block_arr.diag_csc_col_ptr[0]) == 0;
    if (is_null) {
      if (debug)
        fprintf(stderr, "part %" PRId64 " is null!\n", (int64_t)part_id);
    }
    // we don't want a null diagonal!
    if (is_null) {
#pragma omp atomic
      gstat++;
    }

    auto stat = create_spmat_top_csc(&tmp_diag_mat_ptr, part_ncols, part_ncols,
                                     block_arr.diag_csc_row_indx.data(),
                                     block_arr.diag_csc_col_ptr.data(),
                                     block_arr.diag_csc_vals.data(), 0);
    mats_diag[part_id - isUT].reset(tmp_diag_mat_ptr);
    if (stat != PERFLIBS_STATUS_SUCCESS) {
#pragma omp atomic
      gstat += (int)stat;
    }

    auto newcsc = reinterpret_cast<perflibs_spmat_impl_t<T> *>(
                      mats_diag[part_id - isUT]->impl)
                      ->csc;
    if (debug) {
      fprintf(stderr, "new matrix:\nm = %" PRId64 "\n", newcsc.m);
      fprintf(stderr, "n = %" PRId64 "\n", newcsc.n);
      fprintf(stderr, "row_indx.size = %zu\n", newcsc.row_indx.size());
      fprintf(stderr, "row_indx[0] = %" PRId64 "\n",
              (int64_t)newcsc.row_indx[0]);
      fprintf(stderr, "col_ptr.size = %zu\n", newcsc.col_ptr.size());
      fprintf(stderr, "col_ptr[0] = %" PRId64 "\n", (int64_t)newcsc.col_ptr[0]);
      fprintf(stderr, "vals.size = %zu\n", newcsc.vals.size());
    }

    // Create the csc matrix for the tail block
    // Note: although the tail and all diagonal blocks are square,
    //       the tail blocks are not necessarily square -
    //       there may be more columns than rows. E.g. if the matrix is of size
    //       11 and we have 2 parts, if the first has 6 cols then the separator
    //       is of dim 5x5, but the tail block belonging to the first supernode
    //       is of dim 5x6.
    auto nrows_sep = high_sep - low_sep + 1;
    perflibs_spmat_t tmp_sep_mat_ptr = nullptr;
    if (debug)
      fprintf(stderr, "creating tail as csc...\n");

    bool is_null2 = (block_arr.sep_csc_col_ptr[part_ncols] -
                     block_arr.sep_csc_col_ptr[0]) == 0;
    if (!is_null2) {
      stat = create_spmat_top_csc(&tmp_sep_mat_ptr, nrows_sep, part_ncols,
                                  block_arr.sep_csc_row_indx.data(),
                                  block_arr.sep_csc_col_ptr.data(),
                                  block_arr.sep_csc_vals.data(), 0);
      if (stat != PERFLIBS_STATUS_SUCCESS) {
#pragma omp atomic
        gstat += (int)stat;
      }
    } else {
      tmp_sep_mat_ptr = perflibs_spmat_create_null(part_ncols, part_ncols);
    }
    mats_sep[part_id - isUT].reset(tmp_sep_mat_ptr);
  }
  if (gstat) {
    return PERFLIBS_STATUS_EXECUTION_FAILURE;
  }

  perflibs_int_t vals_off =
      isUT ? 0 : col_ptr[low_sep - index_base] - index_base;
  if (debug) {
    fprintf(stderr, "creating separator as csc...\n");
    fprintf(stderr, "vals_off = %" PRId64 "\n", (int64_t)vals_off);
    fprintf(stderr, "col_ptr[low_sep] = %" PRId64 "\n",
            (int64_t)col_ptr[low_sep - index_base]);
  }
  auto sep_stat = create_csc_separator(separator, nparts, index_base, part_indx,
                                       super_row_ptr, super_col_indx, row_indx,
                                       vals, vals_off, low_sep,
                                       shape); // low_sep necessary

  if (sep_stat != PERFLIBS_STATUS_SUCCESS) {
    return sep_stat;
  }

  return PERFLIBS_STATUS_SUCCESS;
}

template <typename T>
perflibs_supernodal<T>::perflibs_supernodal(
    int64_t m, int64_t n, int64_t nsuper, int64_t nparts, int64_t nnz,
    perflibs_sparse_matrix_shape_t shape, const perflibs_int_t *super_row_ptr,
    const perflibs_int_t *super_col_indx, const perflibs_int_t *row_indx,
    const perflibs_int_t *col_ptr, const T *vals,
    const perflibs_int_t *part_indx)
    : m(m), n(n), nsuper(nsuper), nparts(nparts), nnz(nnz), low_sep(-1),
      high_sep(-1), index_base(super_row_ptr[0]), shape(shape),
      super_row_ptr_ptr(super_row_ptr), super_col_indx_ptr(super_col_indx),
      row_indx_ptr(row_indx), col_ptr(col_ptr), vals_ptr(vals),
      part_indx_ptr(part_indx), separator(nullptr) {

  if (debug)
    fprintf(stderr, "In perflibs_supernodal constructor\n");
  // Null matrix
  if ((m == 0) && (n == 0) && (nsuper == 0) && (nparts == 0)) {
    this->mats_diag.clear();
    this->mats_sep.clear();
    this->separator = nullptr;
    return;
  }

  this->mats_diag.resize(nparts);
  this->mats_sep.resize(nparts);

  if (debug)
    fprintf(stderr, "Getting separator indices...\n");
  std::tie(this->low_sep, this->high_sep) =
      get_separator_indices(m, nparts, super_col_indx, part_indx_ptr, shape);
  if (debug)
    fprintf(stderr, "Separator indices are %" PRId64 " and %" PRId64 "\n",
            this->low_sep, this->high_sep);

  [[maybe_unused]] auto stat = populate_supernodal_solve(
      this->mats_diag, this->mats_sep, this->separator, m, n, nsuper, nparts,
      this->low_sep, this->high_sep, this->shape, this->super_row_ptr_ptr,
      this->super_col_indx_ptr, this->row_indx_ptr, this->col_ptr,
      this->vals_ptr, this->part_indx_ptr);
  if (debug)
    fprintf(stderr, "Populated supernodal type for sptrsv\n");

  assert(stat == PERFLIBS_STATUS_SUCCESS);
}

template <typename T>
perflibs_status_t
check_supernodal_params(perflibs_spmat_impl_t<T> *impl, perflibs_int_t m,
                        perflibs_int_t n, perflibs_int_t nparts,
                        perflibs_int_t nsuper, perflibs_int_t row0,
                        perflibs_int_t col0, perflibs_int_t super_row_ptr0,
                        perflibs_int_t super_col_indx0, bool no_copy) {
  /* Check input values */
  if (m < 0) {
    impl->error_handle.perflibs_error_type =
        PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    impl->error_handle.perflibs_error_code = 2;
    impl->error_handle.err_msg = "m < 0";
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }
  if (n < 0) {
    impl->error_handle.perflibs_error_type =
        PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    impl->error_handle.perflibs_error_code = 3;
    impl->error_handle.err_msg = "n < 0";
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }
  if (nsuper < 0) {
    impl->error_handle.perflibs_error_type =
        PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    impl->error_handle.perflibs_error_code = 4;
    impl->error_handle.err_msg = "nsuper < 0";
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }
  if (no_copy) {
    impl->error_handle.perflibs_error_type =
        PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    impl->error_handle.err_msg = "Supernodal format requires the "
                                 "PERFLIBS_SPARSE_CREATE_NO_COPY flag to be "
                                 "false.";
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }
  if (col0 != super_row_ptr0 && row0 != col0 && row0 != super_col_indx0) {
    impl->error_handle.perflibs_error_type =
        PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    impl->error_handle.err_msg = "Supernodal format requires the index base to "
                                 "be consistent across all supplied iterators.";
  }
  if (nparts > m) {
    impl->error_handle.perflibs_error_type =
        PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    impl->error_handle.err_msg =
        "Supernodal format requires the nparts to be at least 1 column wide.";
  }
  if (nparts > nsuper) {
    impl->error_handle.perflibs_error_type =
        PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    impl->error_handle.err_msg =
        "Supernodal format requires at least 1 supernode per part.";
  }
  if (col0 != 0 && col0 != 1) {
    return PERFLIBS_STATUS_EXECUTION_FAILURE;
  }

  return PERFLIBS_STATUS_SUCCESS;
}

template <typename T>
perflibs_supernodal<T>
make_supernodal(perflibs_int_t m, perflibs_int_t n, perflibs_int_t nsuper,
                perflibs_int_t nparts, perflibs_sparse_matrix_shape_t shape,
                const perflibs_int_t *super_row_ptr,
                const perflibs_int_t *super_col_indx,
                const perflibs_int_t *row_indx, const perflibs_int_t *col_ptr,
                const T *vals, const perflibs_int_t *part_indx) {
  perflibs_int_t nnz = col_ptr[n] - col_ptr[0];

  return perflibs_supernodal(m, n, nsuper, nparts, nnz, shape, super_row_ptr,
                             super_col_indx, row_indx, col_ptr, vals,
                             part_indx);
}

template <typename T>
perflibs_status_t fill_initial_data_supernodal(
    perflibs_spmat_top_t *A, perflibs_int_t m, perflibs_int_t n,
    perflibs_int_t nsuper, perflibs_int_t nparts,
    const perflibs_int_t *super_row_ptr, const perflibs_int_t *super_col_indx,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_ptr,
    const T *vals, const perflibs_int_t *part_indx, const bool no_copy) {

  if (debug)
    fprintf(stderr, "In fill_initial_data_supernodal\n");
  auto impl = reinterpret_cast<perflibs_spmat_impl_t<T> *>(A->impl);
  auto ret = check_supernodal_params(impl, m, n, nparts, nsuper, row_indx[0],
                                     col_ptr[0], super_row_ptr[0],
                                     super_col_indx[0], no_copy);
  if (ret != PERFLIBS_STATUS_SUCCESS) {
    return ret;
  }
  if (debug)
    fprintf(stderr, "done check_supernodal_params\n");

  ret = get_supernodal_shape(nsuper, super_row_ptr, super_col_indx, row_indx,
                             col_ptr, vals, impl->shape);
  if (ret != PERFLIBS_STATUS_SUCCESS) {
    impl->error_handle.perflibs_error_type = PERFLIBS_STATUS_EXECUTION_FAILURE;
    impl->error_handle.err_msg =
        "Failed to detect whether the matrix is upper or lower triangular.";
    return ret;
  }
  if (debug)
    fprintf(stderr, "got shape = ");
  if (impl->shape == PERFLIBS_SPARSE_SHAPE_LOWER_TRIANGULAR) {
    if (debug)
      fprintf(stderr, "LOWER\n");
  } else {
    if (debug)
      fprintf(stderr, "UPPER\n");
  }

  impl->m = m;
  impl->n = n;
  impl->index_base = col_ptr[0];
  impl->nnz = col_ptr[n] - col_ptr[0];

  impl->spmat_format = perflibs_format_supernodal;
  impl->no_copy = no_copy;

  impl->diag = PERFLIBS_SPARSE_DIAG_NON_UNIT;

  impl->supernodal =
      make_supernodal(m, n, nsuper, nparts, impl->shape, super_row_ptr,
                      super_col_indx, row_indx, col_ptr, vals, part_indx);

  if (debug) {
    // Take a copy because printing destructively moves to a dense matrix
    auto A2 = spmat_copy<T>(A);
    reinterpret_cast<perflibs_spmat_impl_t<T> *>(A2->impl)
        ->supernodal.printer();
    print_supernode(reinterpret_cast<perflibs_spmat_impl_t<T> *>(A2->impl));
  }
  return PERFLIBS_STATUS_SUCCESS;
}

template perflibs_status_t fill_initial_data_supernodal<float>(
    perflibs_spmat_top_t *A, perflibs_int_t m, perflibs_int_t n,
    perflibs_int_t nsuper, perflibs_int_t nparts,
    const perflibs_int_t *super_row_ptr, const perflibs_int_t *super_col_indx,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_ptr,
    const float *vals, const perflibs_int_t *part_indx, const bool no_copy);
template perflibs_status_t fill_initial_data_supernodal<double>(
    perflibs_spmat_top_t *A, perflibs_int_t m, perflibs_int_t n,
    perflibs_int_t nsuper, perflibs_int_t nparts,
    const perflibs_int_t *super_row_ptr, const perflibs_int_t *super_col_indx,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_ptr,
    const double *vals, const perflibs_int_t *part_indx, const bool no_copy);
template perflibs_status_t fill_initial_data_supernodal<std::complex<float>>(
    perflibs_spmat_top_t *A, perflibs_int_t m, perflibs_int_t n,
    perflibs_int_t nsuper, perflibs_int_t nparts,
    const perflibs_int_t *super_row_ptr, const perflibs_int_t *super_col_indx,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_ptr,
    const std::complex<float> *vals, const perflibs_int_t *part_indx,
    const bool no_copy);
template perflibs_status_t fill_initial_data_supernodal<std::complex<double>>(
    perflibs_spmat_top_t *A, perflibs_int_t m, perflibs_int_t n,
    perflibs_int_t nsuper, perflibs_int_t nparts,
    const perflibs_int_t *super_row_ptr, const perflibs_int_t *super_col_indx,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_ptr,
    const std::complex<double> *vals, const perflibs_int_t *part_indx,
    const bool no_copy);

template <typename T>
perflibs_supernodal<T> &
perflibs_supernodal<T>::operator=(const perflibs_supernodal &other) {
  if (&other == this) {
    return *this;
  }
  // Copy the member variables
  m = other.m;
  n = other.n;
  nsuper = other.nsuper;
  nparts = other.nparts;
  nnz = other.nnz;

  low_sep = other.low_sep;
  high_sep = other.high_sep;
  index_base = other.index_base;
  shape = other.shape;

  if (m >= 0 && n >= 0) {
    part_indx_ptr = other.part_indx_ptr;
    super_row_ptr_ptr = other.super_row_ptr_ptr;
    super_col_indx_ptr = other.super_col_indx_ptr;
    row_indx_ptr = other.row_indx_ptr;
    col_ptr = other.col_ptr;
    vals_ptr = other.vals_ptr;
  }

  // Copy mats_diag
  mats_diag.clear();
  mats_diag.reserve(other.mats_diag.size());
  for (const auto &mat : other.mats_diag) {
    if (mat) {
      auto cpy = spmat_copy<T>(mat.get());
      mats_diag.emplace_back(
          std::shared_ptr<perflibs_spmat_top_t>(cpy.release()));
    } else {
      mats_diag.emplace_back(nullptr);
    }
  }

  // Copy mats_sep
  mats_sep.clear();
  mats_sep.reserve(other.mats_sep.size());
  for (const auto &mat : other.mats_sep) {
    if (mat) {
      auto cpy = spmat_copy<T>(mat.get());
      mats_sep.emplace_back(
          std::shared_ptr<perflibs_spmat_top_t>(cpy.release()));
    } else {
      mats_sep.emplace_back(nullptr);
    }
  }

  // Copy separator
  if (other.separator) {
    auto cpy = spmat_copy<T>(other.separator.get());
    separator = std::shared_ptr<perflibs_spmat_top_t>(cpy.release());
  } else {
    separator = nullptr;
  }

  return *this;
}

template perflibs_supernodal<float> &
perflibs_supernodal<float>::operator=(const perflibs_supernodal<float> &other);
template perflibs_supernodal<double> &perflibs_supernodal<double>::operator=(
    const perflibs_supernodal<double> &other);
template perflibs_supernodal<std::complex<float>> &
perflibs_supernodal<std::complex<float>>::operator=(
    const perflibs_supernodal<std::complex<float>> &other);
template perflibs_supernodal<std::complex<double>> &
perflibs_supernodal<std::complex<double>>::operator=(
    const perflibs_supernodal<std::complex<double>> &other);

template <typename T>
void copy_scaled(T *dst, const T *src, perflibs_int_t count, T alpha) {
  // Fuse RHS scaling with the required copy so a later unit-alpha solve does
  // not rescale accumulated separator-block contributions.
  if (count <= 0) {
    return;
  }

  if (alpha == T(1)) {
    if (dst != src) {
      std::memcpy(dst, src, sizeof(T) * count);
    }
  } else if (dst == src) {
    for (perflibs_int_t i = 0; i < count; ++i) {
      dst[i] *= alpha;
    }
  } else {
    for (perflibs_int_t i = 0; i < count; ++i) {
      dst[i] = alpha * src[i];
    }
  }
}

template <typename T>
void spsv_supernodal_parallel_ut(
    perflibs_int_t m, T alpha,
    const std::vector<std::shared_ptr<perflibs_spmat_top_t>> &mats_diag,
    const std::vector<std::shared_ptr<perflibs_spmat_top_t>> &mats_sep,
    std::shared_ptr<perflibs_spmat_top_t> separator, T *x, const T *y) {
  // Get the number of rows in the separator
  auto sep_dim =
      reinterpret_cast<perflibs_spmat_impl_t<T> *>(separator->impl)->m;

  copy_scaled(x, y, sep_dim, alpha);

  // Allocate a vector of vectors for the parallel separator block contributions
  std::vector<std::vector<T>> acc(perflibs::sparse::omp::get_max_threads(),
                                  std::vector<T>(sep_dim));

  std::vector<size_t> offsets(mats_diag.size());
  offsets[0] = sep_dim;
  for (size_t i = 1; i < offsets.size(); i++) {
    offsets[i] =
        offsets[i - 1] +
        reinterpret_cast<perflibs_spmat_impl_t<T> *>(mats_diag[i - 1]->impl)->m;
  }

#pragma omp parallel for
  for (size_t i = 0; i < mats_diag.size(); i++) {
    auto off = offsets[i];

    // Diagonal solves
    spsv_exec(PERFLIBS_SPARSE_OPERATION_NOTRANS, mats_diag[i].get(), x + off,
              alpha, y + off);

    // Separator block contributions
    spmv_exec<T>(PERFLIBS_SPARSE_OPERATION_NOTRANS, (T)1.0, mats_sep[i].get(),
                 x + off, (T)1.0,
                 acc[perflibs::sparse::omp::get_thread_num()].data());
  }

  // Accumulate x
  for (size_t i = 0; i < acc.size(); i++) {
    for (size_t row_i = 0; row_i < acc[i].size(); row_i++) {
      x[row_i] -= acc[i][row_i];
    }
  }

  // Separator solve
  spsv_exec(PERFLIBS_SPARSE_OPERATION_NOTRANS, separator.get(), x, T(1), x);
}

template <typename T>
void spsv_trans_supernodal_parallel_ut(
    perflibs_int_t m, T alpha,
    const std::vector<std::shared_ptr<perflibs_spmat_top_t>> &mats_diag,
    const std::vector<std::shared_ptr<perflibs_spmat_top_t>> &mats_sep,
    std::shared_ptr<perflibs_spmat_top_t> separator,
    perflibs_sparse_hint_value trans, T *x, const T *y) {
  // Get the number of rows in the separator
  auto sep_dim =
      reinterpret_cast<perflibs_spmat_impl_t<T> *>(separator->impl)->m;

  copy_scaled(x + sep_dim, y + sep_dim, m - sep_dim, alpha);

  std::vector<size_t> offsets(mats_diag.size());
  offsets[0] = sep_dim;
  for (size_t i = 1; i < offsets.size(); i++) {
    offsets[i] =
        offsets[i - 1] +
        reinterpret_cast<perflibs_spmat_impl_t<T> *>(mats_diag[i - 1]->impl)->m;
  }

  // Separator solve
  spsv_exec(trans, separator.get(), x, alpha, y);

#pragma omp parallel for
  for (size_t i = 0; i < mats_diag.size(); i++) {
    auto off = offsets[i];

    // Separator block contributions
    spmv_exec<T>(trans, (T)-1.0, mats_sep[i].get(), x, (T)1.0, x + off);

    // Diagonal solves
    spsv_exec(trans, mats_diag[i].get(), x + off, T(1), x + off);
  }
}

template <typename T>
void spsv_supernodal_parallel_lt(
    perflibs_int_t m, perflibs_int_t sep_indx, T alpha,
    const std::vector<std::shared_ptr<perflibs_spmat_top_t>> &mats_diag,
    const std::vector<std::shared_ptr<perflibs_spmat_top_t>> &mats_sep,
    std::shared_ptr<perflibs_spmat_top_t> separator, T *x, const T *y) {
  // Get the number of rows in the separator
  auto sep_dim =
      reinterpret_cast<perflibs_spmat_impl_t<T> *>(separator->impl)->m;

  copy_scaled(x + sep_indx, y + sep_indx, sep_dim, alpha);

  // Allocate a vector of vectors for the parallel separator block contributions
  std::vector<std::vector<T>> acc(perflibs::sparse::omp::get_max_threads(),
                                  std::vector<T>(sep_dim));

  std::vector<size_t> offsets(mats_diag.size());
  for (size_t i = 1; i < offsets.size(); i++) {
    offsets[i] =
        offsets[i - 1] +
        reinterpret_cast<perflibs_spmat_impl_t<T> *>(mats_diag[i - 1]->impl)->m;
  }

#pragma omp parallel for
  for (size_t i = 0; i < mats_diag.size(); i++) {
    auto off = offsets[i];

    // Diagonal solves
    spsv_exec(PERFLIBS_SPARSE_OPERATION_NOTRANS, mats_diag[i].get(), x + off,
              alpha, y + off);

    // Separator block contributions
    spmv_exec<T>(PERFLIBS_SPARSE_OPERATION_NOTRANS, (T)1.0, mats_sep[i].get(),
                 x + off, (T)1.0,
                 acc[perflibs::sparse::omp::get_thread_num()].data());
  }

  // Accumulate x
  for (size_t i = 0; i < acc.size(); i++) {
    for (size_t row_i = 0; row_i < acc[i].size(); row_i++) {
      x[sep_indx + row_i] -= acc[i][row_i];
    }
  }

  // Separator solve
  spsv_exec(PERFLIBS_SPARSE_OPERATION_NOTRANS, separator.get(), x + sep_indx,
            T(1), x + sep_indx);
}

template <typename T>
void spsv_trans_supernodal_parallel_lt(
    perflibs_int_t m, perflibs_int_t sep_indx, T alpha,
    const std::vector<std::shared_ptr<perflibs_spmat_top_t>> &mats_diag,
    const std::vector<std::shared_ptr<perflibs_spmat_top_t>> &mats_sep,
    std::shared_ptr<perflibs_spmat_top_t> separator,
    perflibs_sparse_hint_value trans, T *x, const T *y) {
  // Get the number of rows in the separator
  auto sep_dim =
      reinterpret_cast<perflibs_spmat_impl_t<T> *>(separator->impl)->m;

  copy_scaled(x, y, m - sep_dim, alpha);

  std::vector<size_t> offsets(mats_diag.size());
  for (size_t i = 1; i < offsets.size(); i++) {
    offsets[i] =
        offsets[i - 1] +
        reinterpret_cast<perflibs_spmat_impl_t<T> *>(mats_diag[i - 1]->impl)->m;
  }

  // Separator solve
  spsv_exec(trans, separator.get(), x + sep_indx, alpha, y + sep_indx);

#pragma omp parallel for
  for (size_t i = 0; i < mats_diag.size(); i++) {
    auto off = offsets[i];

    // Separator block contributions
    spmv_exec<T>((perflibs_sparse_hint_value)trans, (T)-1.0, mats_sep[i].get(),
                 x + sep_indx, (T)1.0, x + off);

    // Diagonal solves
    spsv_exec(trans, mats_diag[i].get(), x + off, T(1), x + off);
  }
}

template <typename T>
void spsv_supernodal_serial_ut(
    perflibs_int_t m, T alpha,
    const std::vector<std::shared_ptr<perflibs_spmat_top_t>> &mats_diag,
    const std::vector<std::shared_ptr<perflibs_spmat_top_t>> &mats_sep,
    const std::shared_ptr<perflibs_spmat_top_t> separator, T *x, const T *y) {
  auto impl_sep = reinterpret_cast<perflibs_spmat_impl_t<T> *>(separator->impl);
  auto off = impl_sep->m;
  copy_scaled(x, y, off, alpha);

  // Diagonal solves
  for (size_t i = 0; i < mats_diag.size(); i++) {
    spsv_exec(PERFLIBS_SPARSE_OPERATION_NOTRANS, mats_diag[i].get(), x + off,
              alpha, y + off);

    off += reinterpret_cast<perflibs_spmat_impl_t<T> *>(mats_diag[i]->impl)->m;
  }

  off = impl_sep->m;
  // Separator block contributions
  for (size_t i = 0; i < mats_sep.size(); i++) {
    spmv_exec<T>(PERFLIBS_SPARSE_OPERATION_NOTRANS, (T)-1.0, mats_sep[i].get(),
                 x + off, (T)1.0, x);

    off += reinterpret_cast<perflibs_spmat_impl_t<T> *>(mats_sep[i]->impl)->n;
  }

  // Separator solve
  spsv_exec(PERFLIBS_SPARSE_OPERATION_NOTRANS, separator.get(), x, T(1), x);
}

template <typename T>
void spsv_trans_supernodal_serial_ut(
    perflibs_int_t m, T alpha,
    const std::vector<std::shared_ptr<perflibs_spmat_top_t>> &mats_diag,
    const std::vector<std::shared_ptr<perflibs_spmat_top_t>> &mats_sep,
    const std::shared_ptr<perflibs_spmat_top_t> separator,
    perflibs_sparse_hint_value trans, T *x, const T *y) {
  auto impl_sep = reinterpret_cast<perflibs_spmat_impl_t<T> *>(separator->impl);
  auto off = impl_sep->m;
  copy_scaled(x + off, y + off, m - off, alpha);

  // Separator solve
  spsv_exec(trans, separator.get(), x, alpha, y);

  // Separator block contributions
  for (size_t i = 0; i < mats_sep.size(); i++) {
    spmv_exec<T>(trans, (T)-1.0, mats_sep[i].get(), x, (T)1.0, x + off);

    off += reinterpret_cast<perflibs_spmat_impl_t<T> *>(mats_sep[i]->impl)->n;
  }

  off = impl_sep->m;
  // Diagonal solves
  for (size_t i = 0; i < mats_diag.size(); i++) {
    spsv_exec((perflibs_sparse_hint_value)trans, mats_diag[i].get(), x + off,
              T(1), x + off);

    off += reinterpret_cast<perflibs_spmat_impl_t<T> *>(mats_diag[i]->impl)->m;
  }
}

template <typename T>
void spsv_supernodal_serial_lt(
    perflibs_int_t m, perflibs_int_t sep_indx, T alpha,
    const std::vector<std::shared_ptr<perflibs_spmat_top_t>> &mats_diag,
    const std::vector<std::shared_ptr<perflibs_spmat_top_t>> &mats_sep,
    const std::shared_ptr<perflibs_spmat_top_t> separator, T *x, const T *y) {
  perflibs_int_t off = 0;
  auto impl_sep = reinterpret_cast<perflibs_spmat_impl_t<T> *>(separator->impl);
  copy_scaled(x + sep_indx, y + sep_indx, impl_sep->m, alpha);

  // Diagonal solves
  for (size_t i = 0; i < mats_diag.size(); i++) {
    spsv_exec(PERFLIBS_SPARSE_OPERATION_NOTRANS, mats_diag[i].get(), x + off,
              alpha, y + off);

    off += reinterpret_cast<perflibs_spmat_impl_t<T> *>(mats_diag[i]->impl)->m;
  }

  off = 0;
  // Separator block contributions
  for (size_t i = 0; i < mats_sep.size(); i++) {
    spmv_exec<T>(PERFLIBS_SPARSE_OPERATION_NOTRANS, (T)-1.0, mats_sep[i].get(),
                 x + off, (T)1.0, x + sep_indx);

    off += reinterpret_cast<perflibs_spmat_impl_t<T> *>(mats_sep[i]->impl)->n;
  }

  // Separator solve
  spsv_exec(PERFLIBS_SPARSE_OPERATION_NOTRANS, separator.get(), x + sep_indx,
            T(1), x + sep_indx);
}

template <typename T>
void spsv_trans_supernodal_serial_lt(
    perflibs_int_t m, perflibs_int_t sep_indx, T alpha,
    const std::vector<std::shared_ptr<perflibs_spmat_top_t>> &mats_diag,
    const std::vector<std::shared_ptr<perflibs_spmat_top_t>> &mats_sep,
    const std::shared_ptr<perflibs_spmat_top_t> separator,
    perflibs_sparse_hint_value trans, T *x, const T *y) {
  perflibs_int_t off = 0;
  auto impl_sep = reinterpret_cast<perflibs_spmat_impl_t<T> *>(separator->impl);
  copy_scaled(x, y, m - impl_sep->m, alpha);

  // Separator solve
  spsv_exec(trans, separator.get(), x + sep_indx, alpha, y + sep_indx);

  // Separator block contributions
  for (size_t i = 0; i < mats_sep.size(); i++) {
    spmv_exec<T>((perflibs_sparse_hint_value)trans, (T)-1.0, mats_sep[i].get(),
                 x + sep_indx, (T)1.0, x + off);

    off += reinterpret_cast<perflibs_spmat_impl_t<T> *>(mats_sep[i]->impl)->n;
  }

  off = 0;
  // Diagonal solves
  for (size_t i = 0; i < mats_diag.size(); i++) {
    spsv_exec(trans, mats_diag[i].get(), x + off, T(1), x + off);

    off += reinterpret_cast<perflibs_spmat_impl_t<T> *>(mats_diag[i]->impl)->m;
  }
}

template <typename T>
void copy_strided_dense_to_row_major(T *dst, const T *src, perflibs_int_t nrows,
                                     perflibs_int_t nrhs,
                                     perflibs_int_t src_stride_row,
                                     perflibs_int_t src_stride_col) {
  for (perflibs_int_t row = 0; row < nrows; ++row) {
    for (perflibs_int_t col = 0; col < nrhs; ++col) {
      dst[row * nrhs + col] = src[row * src_stride_row + col * src_stride_col];
    }
  }
}

template <typename T>
void copy_row_major_to_strided_dense(T *dst, const T *src, perflibs_int_t nrows,
                                     perflibs_int_t nrhs,
                                     perflibs_int_t dst_stride_row,
                                     perflibs_int_t dst_stride_col) {
  for (perflibs_int_t row = 0; row < nrows; ++row) {
    for (perflibs_int_t col = 0; col < nrhs; ++col) {
      dst[row * dst_stride_row + col * dst_stride_col] = src[row * nrhs + col];
    }
  }
}

template <typename T>
std::unique_ptr<perflibs_spmat_top_t>
make_row_major_dense_matrix(perflibs_int_t rows, perflibs_int_t cols,
                            const T *vals, perflibs_int_t flags) {
  perflibs_spmat_top_t *mat = nullptr;
  [[maybe_unused]] auto stat = create_spmat_top_dense<T>(
      &mat, PERFLIBS_ROW_MAJOR, rows, cols, cols, 0, vals, flags);
  assert(stat == PERFLIBS_STATUS_SUCCESS);
  return std::unique_ptr<perflibs_spmat_top_t>(mat);
}

template <typename T>
void copy_row_major_dense_matrix_values(perflibs_spmat_top_t *mat, T *dst,
                                        perflibs_int_t count) {
  auto impl = reinterpret_cast<perflibs_spmat_impl_t<T> *>(mat->impl);
  assert(impl->spmat_format == perflibs_format_dense);
  assert(impl->dense.layout == PERFLIBS_ROW_MAJOR);
  std::memcpy(dst, impl->dense.vals_ptr, sizeof(T) * count);
}

template <typename T>
void spsm_supernodal_serial_lt_notrans(perflibs_supernodal<T> &supernodal, T *X,
                                       perflibs_int_t x_stride_row,
                                       perflibs_int_t x_stride_col, T alpha,
                                       const T *Y, perflibs_int_t y_stride_row,
                                       perflibs_int_t y_stride_col,
                                       perflibs_int_t nrhs) {
  const perflibs_int_t n = supernodal.n;
  const perflibs_int_t sep_indx = supernodal.low_sep - supernodal.index_base;
  auto impl_sep =
      reinterpret_cast<perflibs_spmat_impl_t<T> *>(supernodal.separator->impl);
  const perflibs_int_t sep_dim = impl_sep->m;

  const bool x_is_row_major = x_stride_row == nrhs && x_stride_col == 1;
  const bool y_is_row_major = y_stride_row == nrhs && y_stride_col == 1;

  perflibs::sparse::pod_vector<T> y_work;
  const T *y_work_ptr = Y;
  if (!y_is_row_major) {
    y_work.resize(n * nrhs);
    copy_strided_dense_to_row_major(y_work.data(), Y, n, nrhs, y_stride_row,
                                    y_stride_col);
    y_work_ptr = y_work.data();
  }

  std::vector<std::unique_ptr<perflibs_spmat_top_t>> X_blocks;
  X_blocks.reserve(supernodal.mats_diag.size());

  perflibs_int_t off = 0;
  for (size_t i = 0; i < supernodal.mats_diag.size(); ++i) {
    auto diag_impl = reinterpret_cast<perflibs_spmat_impl_t<T> *>(
        supernodal.mats_diag[i]->impl);
    const auto block_rows = diag_impl->m;
    // The solve overwrites X_block, so seed its owned storage from initialized
    // Y values rather than reading from the caller's output buffer.
    auto X_block = make_row_major_dense_matrix<T>(block_rows, nrhs,
                                                  y_work_ptr + off * nrhs, 0);
    auto Y_block = make_row_major_dense_matrix<T>(
        block_rows, nrhs, y_work_ptr + off * nrhs,
        PERFLIBS_SPARSE_CREATE_NOCOPY);
    [[maybe_unused]] auto stat = spsm_exec<T>(
        PERFLIBS_SPARSE_OPERATION_NOTRANS, supernodal.mats_diag[i].get(),
        X_block.get(), alpha, Y_block.get());
    assert(stat == PERFLIBS_STATUS_SUCCESS);
    X_blocks.push_back(std::move(X_block));

    off += block_rows;
  }

  auto X_sep = make_row_major_dense_matrix<T>(sep_dim, nrhs,
                                              y_work_ptr + sep_indx * nrhs, 0);
  auto X_sep_impl = reinterpret_cast<perflibs_spmat_impl_t<T> *>(X_sep->impl);
  for (perflibs_int_t i = 0; i < sep_dim * nrhs; ++i) {
    X_sep_impl->dense.vals[i] *= alpha;
  }
  for (size_t i = 0; i < supernodal.mats_sep.size(); ++i) {
    [[maybe_unused]] auto stat = spmm_exec<T>(
        PERFLIBS_SPARSE_OPERATION_NOTRANS, PERFLIBS_SPARSE_OPERATION_NOTRANS,
        T(-1), supernodal.mats_sep[i].get(), X_blocks[i].get(), T(1),
        X_sep.get());
    assert(stat == PERFLIBS_STATUS_SUCCESS);
  }

  [[maybe_unused]] auto stat =
      spsm_exec<T>(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                   supernodal.separator.get(), X_sep.get(), T(1), X_sep.get());
  assert(stat == PERFLIBS_STATUS_SUCCESS);

  perflibs::sparse::pod_vector<T> x_work;
  T *x_work_ptr = X;
  if (!x_is_row_major) {
    x_work.resize(n * nrhs);
    x_work_ptr = x_work.data();
  }

  off = 0;
  for (size_t i = 0; i < X_blocks.size(); ++i) {
    auto diag_impl = reinterpret_cast<perflibs_spmat_impl_t<T> *>(
        supernodal.mats_diag[i]->impl);
    copy_row_major_dense_matrix_values<T>(
        X_blocks[i].get(), x_work_ptr + off * nrhs, diag_impl->m * nrhs);
    off += diag_impl->m;
  }
  copy_row_major_dense_matrix_values<T>(
      X_sep.get(), x_work_ptr + sep_indx * nrhs, sep_dim * nrhs);

  if (!x_is_row_major) {
    copy_row_major_to_strided_dense(X, x_work.data(), n, nrhs, x_stride_row,
                                    x_stride_col);
  }
}

template <typename T>
void spsm_supernodal_serial_lt_trans(perflibs_supernodal<T> &supernodal,
                                     perflibs_sparse_hint_value trans, T *X,
                                     perflibs_int_t x_stride_row,
                                     perflibs_int_t x_stride_col, T alpha,
                                     const T *Y, perflibs_int_t y_stride_row,
                                     perflibs_int_t y_stride_col,
                                     perflibs_int_t nrhs) {
  const perflibs_int_t n = supernodal.n;
  const perflibs_int_t sep_indx = supernodal.low_sep - supernodal.index_base;
  auto impl_sep =
      reinterpret_cast<perflibs_spmat_impl_t<T> *>(supernodal.separator->impl);
  const perflibs_int_t sep_dim = impl_sep->m;

  const bool x_is_row_major = x_stride_row == nrhs && x_stride_col == 1;
  const bool y_is_row_major = y_stride_row == nrhs && y_stride_col == 1;

  perflibs::sparse::pod_vector<T> y_work;
  const T *y_work_ptr = Y;
  if (!y_is_row_major) {
    y_work.resize(n * nrhs);
    copy_strided_dense_to_row_major(y_work.data(), Y, n, nrhs, y_stride_row,
                                    y_stride_col);
    y_work_ptr = y_work.data();
  }

  auto X_sep = make_row_major_dense_matrix<T>(sep_dim, nrhs,
                                              y_work_ptr + sep_indx * nrhs, 0);
  auto Y_sep = make_row_major_dense_matrix<T>(sep_dim, nrhs,
                                              y_work_ptr + sep_indx * nrhs,
                                              PERFLIBS_SPARSE_CREATE_NOCOPY);
  [[maybe_unused]] auto stat = spsm_exec<T>(trans, supernodal.separator.get(),
                                            X_sep.get(), alpha, Y_sep.get());
  assert(stat == PERFLIBS_STATUS_SUCCESS);

  std::vector<std::unique_ptr<perflibs_spmat_top_t>> X_blocks;
  X_blocks.reserve(supernodal.mats_diag.size());

  perflibs_int_t off = 0;
  for (size_t i = 0; i < supernodal.mats_diag.size(); ++i) {
    auto diag_impl = reinterpret_cast<perflibs_spmat_impl_t<T> *>(
        supernodal.mats_diag[i]->impl);
    const auto block_rows = diag_impl->m;

    // X_block starts as alpha * Y so the off-diagonal update is not scaled
    // again by the subsequent diagonal solve.
    auto X_block = make_row_major_dense_matrix<T>(block_rows, nrhs,
                                                  y_work_ptr + off * nrhs, 0);
    auto X_block_impl =
        reinterpret_cast<perflibs_spmat_impl_t<T> *>(X_block->impl);
    for (perflibs_int_t j = 0; j < block_rows * nrhs; ++j) {
      X_block_impl->dense.vals[j] *= alpha;
    }
    [[maybe_unused]] auto spmm_stat = spmm_exec<T>(
        trans, PERFLIBS_SPARSE_OPERATION_NOTRANS, T(-1),
        supernodal.mats_sep[i].get(), X_sep.get(), T(1), X_block.get());
    assert(spmm_stat == PERFLIBS_STATUS_SUCCESS);

    [[maybe_unused]] auto spsm_stat =
        spsm_exec<T>(trans, supernodal.mats_diag[i].get(), X_block.get(), T(1),
                     X_block.get());
    assert(spsm_stat == PERFLIBS_STATUS_SUCCESS);
    X_blocks.push_back(std::move(X_block));

    off += block_rows;
  }

  perflibs::sparse::pod_vector<T> x_work;
  T *x_work_ptr = X;
  if (!x_is_row_major) {
    x_work.resize(n * nrhs);
    x_work_ptr = x_work.data();
  }

  off = 0;
  for (size_t i = 0; i < X_blocks.size(); ++i) {
    auto diag_impl = reinterpret_cast<perflibs_spmat_impl_t<T> *>(
        supernodal.mats_diag[i]->impl);
    copy_row_major_dense_matrix_values<T>(
        X_blocks[i].get(), x_work_ptr + off * nrhs, diag_impl->m * nrhs);
    off += diag_impl->m;
  }
  copy_row_major_dense_matrix_values<T>(
      X_sep.get(), x_work_ptr + sep_indx * nrhs, sep_dim * nrhs);

  if (!x_is_row_major) {
    copy_row_major_to_strided_dense(X, x_work.data(), n, nrhs, x_stride_row,
                                    x_stride_col);
  }
}

template <typename T>
void spsm_supernodal_parallel_lt_notrans(perflibs_supernodal<T> &supernodal,
                                         T *X, perflibs_int_t x_stride_row,
                                         perflibs_int_t x_stride_col, T alpha,
                                         const T *Y,
                                         perflibs_int_t y_stride_row,
                                         perflibs_int_t y_stride_col,
                                         perflibs_int_t nrhs) {
  const perflibs_int_t n = supernodal.n;
  const perflibs_int_t sep_indx = supernodal.low_sep - supernodal.index_base;
  auto impl_sep =
      reinterpret_cast<perflibs_spmat_impl_t<T> *>(supernodal.separator->impl);
  const perflibs_int_t sep_dim = impl_sep->m;

  const bool x_is_row_major = x_stride_row == nrhs && x_stride_col == 1;
  const bool y_is_row_major = y_stride_row == nrhs && y_stride_col == 1;

  perflibs::sparse::pod_vector<T> y_work;
  const T *y_work_ptr = Y;
  if (!y_is_row_major) {
    y_work.resize(n * nrhs);
    copy_strided_dense_to_row_major(y_work.data(), Y, n, nrhs, y_stride_row,
                                    y_stride_col);
    y_work_ptr = y_work.data();
  }

  std::vector<size_t> offsets(supernodal.mats_diag.size());
  for (size_t i = 1; i < offsets.size(); ++i) {
    offsets[i] = offsets[i - 1] + reinterpret_cast<perflibs_spmat_impl_t<T> *>(
                                      supernodal.mats_diag[i - 1]->impl)
                                      ->m;
  }

  std::vector<std::unique_ptr<perflibs_spmat_top_t>> X_blocks(
      supernodal.mats_diag.size());
  // Give each thread a private separator accumulator so SpMM updates do not
  // race without allocating one dense contribution matrix per diagonal block.
  std::vector<std::unique_ptr<perflibs_spmat_top_t>> sep_contribs(
      perflibs::sparse::omp::get_max_threads());
  {
    // An initial dummy array which gets copied into each dense matrix in the
    // loop below
    std::vector<T> init_zero_sep_contrib_vals((size_t)sep_dim * (size_t)nrhs);
    for (auto &sep_contrib : sep_contribs) {
      sep_contrib = make_row_major_dense_matrix<T>(
          sep_dim, nrhs, init_zero_sep_contrib_vals.data(), 0);
    }
  }

#pragma omp parallel for
  for (size_t i = 0; i < supernodal.mats_diag.size(); ++i) {
    const auto off = offsets[i];
    auto diag_impl = reinterpret_cast<perflibs_spmat_impl_t<T> *>(
        supernodal.mats_diag[i]->impl);
    const auto block_rows = diag_impl->m;

    // The solve overwrites X_block, so seed its owned storage from initialized
    // Y values rather than reading from the caller's output buffer.
    auto X_block = make_row_major_dense_matrix<T>(block_rows, nrhs,
                                                  y_work_ptr + off * nrhs, 0);
    auto Y_block = make_row_major_dense_matrix<T>(
        block_rows, nrhs, y_work_ptr + off * nrhs,
        PERFLIBS_SPARSE_CREATE_NOCOPY);
    [[maybe_unused]] auto spsm_stat = spsm_exec<T>(
        PERFLIBS_SPARSE_OPERATION_NOTRANS, supernodal.mats_diag[i].get(),
        X_block.get(), alpha, Y_block.get());
    assert(spsm_stat == PERFLIBS_STATUS_SUCCESS);

    auto &sep_contrib = sep_contribs[perflibs::sparse::omp::get_thread_num()];
    [[maybe_unused]] auto spmm_stat = spmm_exec<T>(
        PERFLIBS_SPARSE_OPERATION_NOTRANS, PERFLIBS_SPARSE_OPERATION_NOTRANS,
        T(1), supernodal.mats_sep[i].get(), X_block.get(), T(1),
        sep_contrib.get());
    assert(spmm_stat == PERFLIBS_STATUS_SUCCESS);

    X_blocks[i] = std::move(X_block);
  }

  auto X_sep = make_row_major_dense_matrix<T>(sep_dim, nrhs,
                                              y_work_ptr + sep_indx * nrhs, 0);
  auto X_sep_impl = reinterpret_cast<perflibs_spmat_impl_t<T> *>(X_sep->impl);
  T *x_sep_vals = X_sep_impl->dense.vals.data();
  for (perflibs_int_t i = 0; i < sep_dim * nrhs; ++i) {
    x_sep_vals[i] *= alpha;
  }
  for (const auto &sep_contrib : sep_contribs) {
    auto sep_contrib_impl =
        reinterpret_cast<perflibs_spmat_impl_t<T> *>(sep_contrib->impl);
    const T *contrib_vals = sep_contrib_impl->dense.vals_ptr;
    for (perflibs_int_t i = 0; i < sep_dim * nrhs; ++i) {
      x_sep_vals[i] -= contrib_vals[i];
    }
  }

  [[maybe_unused]] auto stat =
      spsm_exec<T>(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                   supernodal.separator.get(), X_sep.get(), T(1), X_sep.get());
  assert(stat == PERFLIBS_STATUS_SUCCESS);

  perflibs::sparse::pod_vector<T> x_work;
  T *x_work_ptr = X;
  if (!x_is_row_major) {
    x_work.resize(n * nrhs);
    x_work_ptr = x_work.data();
  }

  for (size_t i = 0; i < X_blocks.size(); ++i) {
    auto diag_impl = reinterpret_cast<perflibs_spmat_impl_t<T> *>(
        supernodal.mats_diag[i]->impl);
    copy_row_major_dense_matrix_values<T>(
        X_blocks[i].get(), x_work_ptr + offsets[i] * nrhs, diag_impl->m * nrhs);
  }
  copy_row_major_dense_matrix_values<T>(
      X_sep.get(), x_work_ptr + sep_indx * nrhs, sep_dim * nrhs);

  if (!x_is_row_major) {
    copy_row_major_to_strided_dense(X, x_work.data(), n, nrhs, x_stride_row,
                                    x_stride_col);
  }
}

template <typename T>
void spsm_supernodal_parallel_lt_trans(perflibs_supernodal<T> &supernodal,
                                       perflibs_sparse_hint_value trans, T *X,
                                       perflibs_int_t x_stride_row,
                                       perflibs_int_t x_stride_col, T alpha,
                                       const T *Y, perflibs_int_t y_stride_row,
                                       perflibs_int_t y_stride_col,
                                       perflibs_int_t nrhs) {
  const perflibs_int_t n = supernodal.n;
  const perflibs_int_t sep_indx = supernodal.low_sep - supernodal.index_base;
  auto impl_sep =
      reinterpret_cast<perflibs_spmat_impl_t<T> *>(supernodal.separator->impl);
  const perflibs_int_t sep_dim = impl_sep->m;

  const bool x_is_row_major = x_stride_row == nrhs && x_stride_col == 1;
  const bool y_is_row_major = y_stride_row == nrhs && y_stride_col == 1;

  perflibs::sparse::pod_vector<T> y_work;
  const T *y_work_ptr = Y;
  if (!y_is_row_major) {
    y_work.resize(n * nrhs);
    copy_strided_dense_to_row_major(y_work.data(), Y, n, nrhs, y_stride_row,
                                    y_stride_col);
    y_work_ptr = y_work.data();
  }

  std::vector<size_t> offsets(supernodal.mats_diag.size());
  for (size_t i = 1; i < offsets.size(); ++i) {
    offsets[i] = offsets[i - 1] + reinterpret_cast<perflibs_spmat_impl_t<T> *>(
                                      supernodal.mats_diag[i - 1]->impl)
                                      ->m;
  }

  auto X_sep = make_row_major_dense_matrix<T>(sep_dim, nrhs,
                                              y_work_ptr + sep_indx * nrhs, 0);
  auto Y_sep = make_row_major_dense_matrix<T>(sep_dim, nrhs,
                                              y_work_ptr + sep_indx * nrhs,
                                              PERFLIBS_SPARSE_CREATE_NOCOPY);
  [[maybe_unused]] auto stat = spsm_exec<T>(trans, supernodal.separator.get(),
                                            X_sep.get(), alpha, Y_sep.get());
  assert(stat == PERFLIBS_STATUS_SUCCESS);

  std::vector<std::unique_ptr<perflibs_spmat_top_t>> X_blocks(
      supernodal.mats_diag.size());

#pragma omp parallel for
  for (size_t i = 0; i < supernodal.mats_diag.size(); ++i) {
    const auto off = offsets[i];
    auto diag_impl = reinterpret_cast<perflibs_spmat_impl_t<T> *>(
        supernodal.mats_diag[i]->impl);
    const auto block_rows = diag_impl->m;

    // X_block starts as alpha * Y so the off-diagonal update is not scaled
    // again by the subsequent diagonal solve.
    auto X_block = make_row_major_dense_matrix<T>(block_rows, nrhs,
                                                  y_work_ptr + off * nrhs, 0);
    auto X_block_impl =
        reinterpret_cast<perflibs_spmat_impl_t<T> *>(X_block->impl);
    for (perflibs_int_t j = 0; j < block_rows * nrhs; ++j) {
      X_block_impl->dense.vals[j] *= alpha;
    }
    [[maybe_unused]] auto spmm_stat = spmm_exec<T>(
        trans, PERFLIBS_SPARSE_OPERATION_NOTRANS, T(-1),
        supernodal.mats_sep[i].get(), X_sep.get(), T(1), X_block.get());
    assert(spmm_stat == PERFLIBS_STATUS_SUCCESS);

    [[maybe_unused]] auto spsm_stat =
        spsm_exec<T>(trans, supernodal.mats_diag[i].get(), X_block.get(), T(1),
                     X_block.get());
    assert(spsm_stat == PERFLIBS_STATUS_SUCCESS);

    X_blocks[i] = std::move(X_block);
  }

  perflibs::sparse::pod_vector<T> x_work;
  T *x_work_ptr = X;
  if (!x_is_row_major) {
    x_work.resize(n * nrhs);
    x_work_ptr = x_work.data();
  }

  for (size_t i = 0; i < X_blocks.size(); ++i) {
    auto diag_impl = reinterpret_cast<perflibs_spmat_impl_t<T> *>(
        supernodal.mats_diag[i]->impl);
    copy_row_major_dense_matrix_values<T>(
        X_blocks[i].get(), x_work_ptr + offsets[i] * nrhs, diag_impl->m * nrhs);
  }
  copy_row_major_dense_matrix_values<T>(
      X_sep.get(), x_work_ptr + sep_indx * nrhs, sep_dim * nrhs);

  if (!x_is_row_major) {
    copy_row_major_to_strided_dense(X, x_work.data(), n, nrhs, x_stride_row,
                                    x_stride_col);
  }
}

template <typename T>
void spsm_supernodal_serial_ut_notrans(perflibs_supernodal<T> &supernodal, T *X,
                                       perflibs_int_t x_stride_row,
                                       perflibs_int_t x_stride_col, T alpha,
                                       const T *Y, perflibs_int_t y_stride_row,
                                       perflibs_int_t y_stride_col,
                                       perflibs_int_t nrhs) {
  const perflibs_int_t n = supernodal.n;
  auto impl_sep =
      reinterpret_cast<perflibs_spmat_impl_t<T> *>(supernodal.separator->impl);
  const perflibs_int_t sep_dim = impl_sep->m;

  const bool x_is_row_major = x_stride_row == nrhs && x_stride_col == 1;
  const bool y_is_row_major = y_stride_row == nrhs && y_stride_col == 1;

  perflibs::sparse::pod_vector<T> y_work;
  const T *y_work_ptr = Y;
  if (!y_is_row_major) {
    y_work.resize(n * nrhs);
    copy_strided_dense_to_row_major(y_work.data(), Y, n, nrhs, y_stride_row,
                                    y_stride_col);
    y_work_ptr = y_work.data();
  }

  std::vector<std::unique_ptr<perflibs_spmat_top_t>> X_blocks;
  X_blocks.reserve(supernodal.mats_diag.size());

  perflibs_int_t off = sep_dim;
  for (size_t i = 0; i < supernodal.mats_diag.size(); ++i) {
    auto diag_impl = reinterpret_cast<perflibs_spmat_impl_t<T> *>(
        supernodal.mats_diag[i]->impl);
    const auto block_rows = diag_impl->m;

    // The solve overwrites X_block, so seed its owned storage from initialized
    // Y values rather than reading from the caller's output buffer.
    auto X_block = make_row_major_dense_matrix<T>(block_rows, nrhs,
                                                  y_work_ptr + off * nrhs, 0);
    auto Y_block = make_row_major_dense_matrix<T>(
        block_rows, nrhs, y_work_ptr + off * nrhs,
        PERFLIBS_SPARSE_CREATE_NOCOPY);
    [[maybe_unused]] auto stat = spsm_exec<T>(
        PERFLIBS_SPARSE_OPERATION_NOTRANS, supernodal.mats_diag[i].get(),
        X_block.get(), alpha, Y_block.get());
    assert(stat == PERFLIBS_STATUS_SUCCESS);
    X_blocks.push_back(std::move(X_block));

    off += block_rows;
  }

  auto X_sep = make_row_major_dense_matrix<T>(sep_dim, nrhs, y_work_ptr, 0);
  auto X_sep_impl = reinterpret_cast<perflibs_spmat_impl_t<T> *>(X_sep->impl);
  for (perflibs_int_t i = 0; i < sep_dim * nrhs; ++i) {
    X_sep_impl->dense.vals[i] *= alpha;
  }
  for (size_t i = 0; i < supernodal.mats_sep.size(); ++i) {
    [[maybe_unused]] auto stat = spmm_exec<T>(
        PERFLIBS_SPARSE_OPERATION_NOTRANS, PERFLIBS_SPARSE_OPERATION_NOTRANS,
        T(-1), supernodal.mats_sep[i].get(), X_blocks[i].get(), T(1),
        X_sep.get());
    assert(stat == PERFLIBS_STATUS_SUCCESS);
  }

  [[maybe_unused]] auto stat =
      spsm_exec<T>(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                   supernodal.separator.get(), X_sep.get(), T(1), X_sep.get());
  assert(stat == PERFLIBS_STATUS_SUCCESS);

  perflibs::sparse::pod_vector<T> x_work;
  T *x_work_ptr = X;
  if (!x_is_row_major) {
    x_work.resize(n * nrhs);
    x_work_ptr = x_work.data();
  }

  copy_row_major_dense_matrix_values<T>(X_sep.get(), x_work_ptr,
                                        sep_dim * nrhs);
  off = sep_dim;
  for (size_t i = 0; i < X_blocks.size(); ++i) {
    auto diag_impl = reinterpret_cast<perflibs_spmat_impl_t<T> *>(
        supernodal.mats_diag[i]->impl);
    copy_row_major_dense_matrix_values<T>(
        X_blocks[i].get(), x_work_ptr + off * nrhs, diag_impl->m * nrhs);
    off += diag_impl->m;
  }

  if (!x_is_row_major) {
    copy_row_major_to_strided_dense(X, x_work.data(), n, nrhs, x_stride_row,
                                    x_stride_col);
  }
}

template <typename T>
void spsm_supernodal_serial_ut_trans(perflibs_supernodal<T> &supernodal,
                                     perflibs_sparse_hint_value trans, T *X,
                                     perflibs_int_t x_stride_row,
                                     perflibs_int_t x_stride_col, T alpha,
                                     const T *Y, perflibs_int_t y_stride_row,
                                     perflibs_int_t y_stride_col,
                                     perflibs_int_t nrhs) {
  const perflibs_int_t n = supernodal.n;
  auto impl_sep =
      reinterpret_cast<perflibs_spmat_impl_t<T> *>(supernodal.separator->impl);
  const perflibs_int_t sep_dim = impl_sep->m;

  const bool x_is_row_major = x_stride_row == nrhs && x_stride_col == 1;
  const bool y_is_row_major = y_stride_row == nrhs && y_stride_col == 1;

  perflibs::sparse::pod_vector<T> y_work;
  const T *y_work_ptr = Y;
  if (!y_is_row_major) {
    y_work.resize(n * nrhs);
    copy_strided_dense_to_row_major(y_work.data(), Y, n, nrhs, y_stride_row,
                                    y_stride_col);
    y_work_ptr = y_work.data();
  }

  auto X_sep = make_row_major_dense_matrix<T>(sep_dim, nrhs, y_work_ptr, 0);
  auto Y_sep = make_row_major_dense_matrix<T>(sep_dim, nrhs, y_work_ptr,
                                              PERFLIBS_SPARSE_CREATE_NOCOPY);
  [[maybe_unused]] auto stat = spsm_exec<T>(trans, supernodal.separator.get(),
                                            X_sep.get(), alpha, Y_sep.get());
  assert(stat == PERFLIBS_STATUS_SUCCESS);

  std::vector<std::unique_ptr<perflibs_spmat_top_t>> X_blocks;
  X_blocks.reserve(supernodal.mats_diag.size());

  perflibs_int_t off = sep_dim;
  for (size_t i = 0; i < supernodal.mats_diag.size(); ++i) {
    auto diag_impl = reinterpret_cast<perflibs_spmat_impl_t<T> *>(
        supernodal.mats_diag[i]->impl);
    const auto block_rows = diag_impl->m;

    // X_block starts as alpha * Y so the off-diagonal update is not scaled
    // again by the subsequent diagonal solve.
    auto X_block = make_row_major_dense_matrix<T>(block_rows, nrhs,
                                                  y_work_ptr + off * nrhs, 0);
    auto X_block_impl =
        reinterpret_cast<perflibs_spmat_impl_t<T> *>(X_block->impl);
    for (perflibs_int_t j = 0; j < block_rows * nrhs; ++j) {
      X_block_impl->dense.vals[j] *= alpha;
    }
    [[maybe_unused]] auto spmm_stat = spmm_exec<T>(
        trans, PERFLIBS_SPARSE_OPERATION_NOTRANS, T(-1),
        supernodal.mats_sep[i].get(), X_sep.get(), T(1), X_block.get());
    assert(spmm_stat == PERFLIBS_STATUS_SUCCESS);

    [[maybe_unused]] auto spsm_stat =
        spsm_exec<T>(trans, supernodal.mats_diag[i].get(), X_block.get(), T(1),
                     X_block.get());
    assert(spsm_stat == PERFLIBS_STATUS_SUCCESS);
    X_blocks.push_back(std::move(X_block));

    off += block_rows;
  }

  perflibs::sparse::pod_vector<T> x_work;
  T *x_work_ptr = X;
  if (!x_is_row_major) {
    x_work.resize(n * nrhs);
    x_work_ptr = x_work.data();
  }

  copy_row_major_dense_matrix_values<T>(X_sep.get(), x_work_ptr,
                                        sep_dim * nrhs);
  off = sep_dim;
  for (size_t i = 0; i < X_blocks.size(); ++i) {
    auto diag_impl = reinterpret_cast<perflibs_spmat_impl_t<T> *>(
        supernodal.mats_diag[i]->impl);
    copy_row_major_dense_matrix_values<T>(
        X_blocks[i].get(), x_work_ptr + off * nrhs, diag_impl->m * nrhs);
    off += diag_impl->m;
  }

  if (!x_is_row_major) {
    copy_row_major_to_strided_dense(X, x_work.data(), n, nrhs, x_stride_row,
                                    x_stride_col);
  }
}

template <typename T>
void spsm_supernodal_parallel_ut_notrans(perflibs_supernodal<T> &supernodal,
                                         T *X, perflibs_int_t x_stride_row,
                                         perflibs_int_t x_stride_col, T alpha,
                                         const T *Y,
                                         perflibs_int_t y_stride_row,
                                         perflibs_int_t y_stride_col,
                                         perflibs_int_t nrhs) {
  const perflibs_int_t n = supernodal.n;
  auto impl_sep =
      reinterpret_cast<perflibs_spmat_impl_t<T> *>(supernodal.separator->impl);
  const perflibs_int_t sep_dim = impl_sep->m;

  const bool x_is_row_major = x_stride_row == nrhs && x_stride_col == 1;
  const bool y_is_row_major = y_stride_row == nrhs && y_stride_col == 1;

  perflibs::sparse::pod_vector<T> y_work;
  const T *y_work_ptr = Y;
  if (!y_is_row_major) {
    y_work.resize(n * nrhs);
    copy_strided_dense_to_row_major(y_work.data(), Y, n, nrhs, y_stride_row,
                                    y_stride_col);
    y_work_ptr = y_work.data();
  }

  std::vector<size_t> offsets(supernodal.mats_diag.size());
  if (!offsets.empty()) {
    offsets[0] = sep_dim;
  }
  for (size_t i = 1; i < offsets.size(); ++i) {
    offsets[i] = offsets[i - 1] + reinterpret_cast<perflibs_spmat_impl_t<T> *>(
                                      supernodal.mats_diag[i - 1]->impl)
                                      ->m;
  }

  std::vector<std::unique_ptr<perflibs_spmat_top_t>> X_blocks(
      supernodal.mats_diag.size());
  // Give each thread a private separator accumulator so SpMM updates do not
  // race without allocating one dense contribution matrix per diagonal block.
  std::vector<std::unique_ptr<perflibs_spmat_top_t>> sep_contribs(
      perflibs::sparse::omp::get_max_threads());
  {
    // An initial dummy array which gets copied into each dense matrix in the
    // loop below
    std::vector<T> init_zero_sep_contrib_vals((size_t)sep_dim * (size_t)nrhs);
    for (auto &sep_contrib : sep_contribs) {
      sep_contrib = make_row_major_dense_matrix<T>(
          sep_dim, nrhs, init_zero_sep_contrib_vals.data(), 0);
    }
  }

#pragma omp parallel for
  for (size_t i = 0; i < supernodal.mats_diag.size(); ++i) {
    const auto off = offsets[i];
    auto diag_impl = reinterpret_cast<perflibs_spmat_impl_t<T> *>(
        supernodal.mats_diag[i]->impl);
    const auto block_rows = diag_impl->m;

    // The solve overwrites X_block, so seed its owned storage from initialized
    // Y values rather than reading from the caller's output buffer.
    auto X_block = make_row_major_dense_matrix<T>(block_rows, nrhs,
                                                  y_work_ptr + off * nrhs, 0);
    auto Y_block = make_row_major_dense_matrix<T>(
        block_rows, nrhs, y_work_ptr + off * nrhs,
        PERFLIBS_SPARSE_CREATE_NOCOPY);
    [[maybe_unused]] auto spsm_stat = spsm_exec<T>(
        PERFLIBS_SPARSE_OPERATION_NOTRANS, supernodal.mats_diag[i].get(),
        X_block.get(), alpha, Y_block.get());
    assert(spsm_stat == PERFLIBS_STATUS_SUCCESS);

    auto &sep_contrib = sep_contribs[perflibs::sparse::omp::get_thread_num()];
    [[maybe_unused]] auto spmm_stat = spmm_exec<T>(
        PERFLIBS_SPARSE_OPERATION_NOTRANS, PERFLIBS_SPARSE_OPERATION_NOTRANS,
        T(1), supernodal.mats_sep[i].get(), X_block.get(), T(1),
        sep_contrib.get());
    assert(spmm_stat == PERFLIBS_STATUS_SUCCESS);

    X_blocks[i] = std::move(X_block);
  }

  auto X_sep = make_row_major_dense_matrix<T>(sep_dim, nrhs, y_work_ptr, 0);
  auto X_sep_impl = reinterpret_cast<perflibs_spmat_impl_t<T> *>(X_sep->impl);
  T *x_sep_vals = X_sep_impl->dense.vals.data();
  for (perflibs_int_t i = 0; i < sep_dim * nrhs; ++i) {
    x_sep_vals[i] *= alpha;
  }
  for (const auto &sep_contrib : sep_contribs) {
    auto sep_contrib_impl =
        reinterpret_cast<perflibs_spmat_impl_t<T> *>(sep_contrib->impl);
    const T *contrib_vals = sep_contrib_impl->dense.vals_ptr;
    for (perflibs_int_t i = 0; i < sep_dim * nrhs; ++i) {
      x_sep_vals[i] -= contrib_vals[i];
    }
  }

  [[maybe_unused]] auto stat =
      spsm_exec<T>(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                   supernodal.separator.get(), X_sep.get(), T(1), X_sep.get());
  assert(stat == PERFLIBS_STATUS_SUCCESS);

  perflibs::sparse::pod_vector<T> x_work;
  T *x_work_ptr = X;
  if (!x_is_row_major) {
    x_work.resize(n * nrhs);
    x_work_ptr = x_work.data();
  }

  copy_row_major_dense_matrix_values<T>(X_sep.get(), x_work_ptr,
                                        sep_dim * nrhs);
  for (size_t i = 0; i < X_blocks.size(); ++i) {
    auto diag_impl = reinterpret_cast<perflibs_spmat_impl_t<T> *>(
        supernodal.mats_diag[i]->impl);
    copy_row_major_dense_matrix_values<T>(
        X_blocks[i].get(), x_work_ptr + offsets[i] * nrhs, diag_impl->m * nrhs);
  }

  if (!x_is_row_major) {
    copy_row_major_to_strided_dense(X, x_work.data(), n, nrhs, x_stride_row,
                                    x_stride_col);
  }
}

template <typename T>
void spsm_supernodal_parallel_ut_trans(perflibs_supernodal<T> &supernodal,
                                       perflibs_sparse_hint_value trans, T *X,
                                       perflibs_int_t x_stride_row,
                                       perflibs_int_t x_stride_col, T alpha,
                                       const T *Y, perflibs_int_t y_stride_row,
                                       perflibs_int_t y_stride_col,
                                       perflibs_int_t nrhs) {
  const perflibs_int_t n = supernodal.n;
  auto impl_sep =
      reinterpret_cast<perflibs_spmat_impl_t<T> *>(supernodal.separator->impl);
  const perflibs_int_t sep_dim = impl_sep->m;

  const bool x_is_row_major = x_stride_row == nrhs && x_stride_col == 1;
  const bool y_is_row_major = y_stride_row == nrhs && y_stride_col == 1;

  perflibs::sparse::pod_vector<T> y_work;
  const T *y_work_ptr = Y;
  if (!y_is_row_major) {
    y_work.resize(n * nrhs);
    copy_strided_dense_to_row_major(y_work.data(), Y, n, nrhs, y_stride_row,
                                    y_stride_col);
    y_work_ptr = y_work.data();
  }

  std::vector<size_t> offsets(supernodal.mats_diag.size());
  if (!offsets.empty()) {
    offsets[0] = sep_dim;
  }
  for (size_t i = 1; i < offsets.size(); ++i) {
    offsets[i] = offsets[i - 1] + reinterpret_cast<perflibs_spmat_impl_t<T> *>(
                                      supernodal.mats_diag[i - 1]->impl)
                                      ->m;
  }

  auto X_sep = make_row_major_dense_matrix<T>(sep_dim, nrhs, y_work_ptr, 0);
  auto Y_sep = make_row_major_dense_matrix<T>(sep_dim, nrhs, y_work_ptr,
                                              PERFLIBS_SPARSE_CREATE_NOCOPY);
  [[maybe_unused]] auto stat = spsm_exec<T>(trans, supernodal.separator.get(),
                                            X_sep.get(), alpha, Y_sep.get());
  assert(stat == PERFLIBS_STATUS_SUCCESS);

  std::vector<std::unique_ptr<perflibs_spmat_top_t>> X_blocks(
      supernodal.mats_diag.size());

#pragma omp parallel for
  for (size_t i = 0; i < supernodal.mats_diag.size(); ++i) {
    const auto off = offsets[i];
    auto diag_impl = reinterpret_cast<perflibs_spmat_impl_t<T> *>(
        supernodal.mats_diag[i]->impl);
    const auto block_rows = diag_impl->m;

    // X_block starts as alpha * Y so the off-diagonal update is not scaled
    // again by the subsequent diagonal solve.
    auto X_block = make_row_major_dense_matrix<T>(block_rows, nrhs,
                                                  y_work_ptr + off * nrhs, 0);
    auto X_block_impl =
        reinterpret_cast<perflibs_spmat_impl_t<T> *>(X_block->impl);
    for (perflibs_int_t j = 0; j < block_rows * nrhs; ++j) {
      X_block_impl->dense.vals[j] *= alpha;
    }
    [[maybe_unused]] auto spmm_stat = spmm_exec<T>(
        trans, PERFLIBS_SPARSE_OPERATION_NOTRANS, T(-1),
        supernodal.mats_sep[i].get(), X_sep.get(), T(1), X_block.get());
    assert(spmm_stat == PERFLIBS_STATUS_SUCCESS);

    [[maybe_unused]] auto spsm_stat =
        spsm_exec<T>(trans, supernodal.mats_diag[i].get(), X_block.get(), T(1),
                     X_block.get());
    assert(spsm_stat == PERFLIBS_STATUS_SUCCESS);

    X_blocks[i] = std::move(X_block);
  }

  perflibs::sparse::pod_vector<T> x_work;
  T *x_work_ptr = X;
  if (!x_is_row_major) {
    x_work.resize(n * nrhs);
    x_work_ptr = x_work.data();
  }

  copy_row_major_dense_matrix_values<T>(X_sep.get(), x_work_ptr,
                                        sep_dim * nrhs);
  for (size_t i = 0; i < X_blocks.size(); ++i) {
    auto diag_impl = reinterpret_cast<perflibs_spmat_impl_t<T> *>(
        supernodal.mats_diag[i]->impl);
    copy_row_major_dense_matrix_values<T>(
        X_blocks[i].get(), x_work_ptr + offsets[i] * nrhs, diag_impl->m * nrhs);
  }

  if (!x_is_row_major) {
    copy_row_major_to_strided_dense(X, x_work.data(), n, nrhs, x_stride_row,
                                    x_stride_col);
  }
}

template <typename T>
void spsv_supernodal(perflibs_supernodal<T> &supernodal,
                     perflibs_sparse_hint_value trans, T *x, const T *y,
                     T alpha) {
  auto index_base = supernodal.index_base;

  // Do nothing if empty matrix
  if (supernodal.m == 0 || supernodal.n == 0 || supernodal.nsuper == 0) {
    return;
  }

  assert(supernodal.mats_diag.size() == supernodal.mats_sep.size());

  if (perflibs::sparse::omp::get_max_threads() > 1) {
    if (supernodal.shape == PERFLIBS_SPARSE_SHAPE_LOWER_TRIANGULAR) {
      if (trans == PERFLIBS_SPARSE_OPERATION_NOTRANS) {
        spsv_supernodal_parallel_lt(supernodal.m,
                                    (supernodal.low_sep - index_base), alpha,
                                    supernodal.mats_diag, supernodal.mats_sep,
                                    supernodal.separator, x, y);
      } else {
        spsv_trans_supernodal_parallel_lt(
            supernodal.m, (supernodal.low_sep - index_base), alpha,
            supernodal.mats_diag, supernodal.mats_sep, supernodal.separator,
            trans, x, y);
      }
    } else {
      if (trans == PERFLIBS_SPARSE_OPERATION_NOTRANS) {
        spsv_supernodal_parallel_ut(supernodal.m, alpha, supernodal.mats_diag,
                                    supernodal.mats_sep, supernodal.separator,
                                    x, y);
      } else {
        spsv_trans_supernodal_parallel_ut(
            supernodal.m, alpha, supernodal.mats_diag, supernodal.mats_sep,
            supernodal.separator, trans, x, y);
      }
    }
  } else {
    if (supernodal.shape == PERFLIBS_SPARSE_SHAPE_LOWER_TRIANGULAR) {
      if (trans == PERFLIBS_SPARSE_OPERATION_NOTRANS) {
        spsv_supernodal_serial_lt(supernodal.m,
                                  (supernodal.low_sep - index_base), alpha,
                                  supernodal.mats_diag, supernodal.mats_sep,
                                  supernodal.separator, x, y);
      } else {
        spsv_trans_supernodal_serial_lt(
            supernodal.m, (supernodal.low_sep - index_base), alpha,
            supernodal.mats_diag, supernodal.mats_sep, supernodal.separator,
            trans, x, y);
      }
    } else {
      if (trans == PERFLIBS_SPARSE_OPERATION_NOTRANS) {
        spsv_supernodal_serial_ut(supernodal.m, alpha, supernodal.mats_diag,
                                  supernodal.mats_sep, supernodal.separator, x,
                                  y);
      } else {
        spsv_trans_supernodal_serial_ut(
            supernodal.m, alpha, supernodal.mats_diag, supernodal.mats_sep,
            supernodal.separator, trans, x, y);
      }
    }
  }
}

template <typename T>
void spsm_supernodal(perflibs_supernodal<T> &supernodal,
                     perflibs_sparse_hint_value trans, T *X,
                     perflibs_int_t x_stride_row, perflibs_int_t x_stride_col,
                     T alpha, const T *Y, perflibs_int_t y_stride_row,
                     perflibs_int_t y_stride_col, perflibs_int_t nrhs) {
  if (nrhs <= 0) {
    return;
  }

  assert(supernodal.mats_diag.size() == supernodal.mats_sep.size());

  if (supernodal.m != 0 && supernodal.n != 0 &&
      (supernodal.shape == PERFLIBS_SPARSE_SHAPE_LOWER_TRIANGULAR ||
       supernodal.shape == PERFLIBS_SPARSE_SHAPE_UPPER_TRIANGULAR)) {
    if (perflibs::sparse::omp::get_max_threads() > 1) {
      if (supernodal.shape == PERFLIBS_SPARSE_SHAPE_LOWER_TRIANGULAR) {
        if (trans == PERFLIBS_SPARSE_OPERATION_NOTRANS) {
          spsm_supernodal_parallel_lt_notrans<T>(
              supernodal, X, x_stride_row, x_stride_col, alpha, Y, y_stride_row,
              y_stride_col, nrhs);
        } else {
          spsm_supernodal_parallel_lt_trans<T>(
              supernodal, trans, X, x_stride_row, x_stride_col, alpha, Y,
              y_stride_row, y_stride_col, nrhs);
        }
      } else {
        if (trans == PERFLIBS_SPARSE_OPERATION_NOTRANS) {
          spsm_supernodal_parallel_ut_notrans<T>(
              supernodal, X, x_stride_row, x_stride_col, alpha, Y, y_stride_row,
              y_stride_col, nrhs);
        } else {
          spsm_supernodal_parallel_ut_trans<T>(
              supernodal, trans, X, x_stride_row, x_stride_col, alpha, Y,
              y_stride_row, y_stride_col, nrhs);
        }
      }
    } else {
      if (supernodal.shape == PERFLIBS_SPARSE_SHAPE_LOWER_TRIANGULAR) {
        if (trans == PERFLIBS_SPARSE_OPERATION_NOTRANS) {
          spsm_supernodal_serial_lt_notrans<T>(
              supernodal, X, x_stride_row, x_stride_col, alpha, Y, y_stride_row,
              y_stride_col, nrhs);
        } else {
          spsm_supernodal_serial_lt_trans<T>(supernodal, trans, X, x_stride_row,
                                             x_stride_col, alpha, Y,
                                             y_stride_row, y_stride_col, nrhs);
        }
      } else {
        if (trans == PERFLIBS_SPARSE_OPERATION_NOTRANS) {
          spsm_supernodal_serial_ut_notrans<T>(
              supernodal, X, x_stride_row, x_stride_col, alpha, Y, y_stride_row,
              y_stride_col, nrhs);
        } else {
          spsm_supernodal_serial_ut_trans<T>(supernodal, trans, X, x_stride_row,
                                             x_stride_col, alpha, Y,
                                             y_stride_row, y_stride_col, nrhs);
        }
      }
    }
    return;
  }

  const bool unit_row_strides = x_stride_row == 1 && y_stride_row == 1;
  perflibs::sparse::pod_vector<T> xbuf(unit_row_strides ? 0 : supernodal.n);
  perflibs::sparse::pod_vector<T> ybuf(unit_row_strides ? 0 : supernodal.n);

  for (perflibs_int_t col = 0; col < nrhs; ++col) {
    T *x_col = X + col * x_stride_col;
    const T *y_col = Y + col * y_stride_col;

    // The initial implementation delegates each RHS column to the existing
    // vector solve. If the column is already contiguous, pass it through
    // directly so the behaviour is exactly the same as spsv_supernodal().
    if (unit_row_strides) {
      spsv_supernodal<T>(supernodal, trans, x_col, y_col, alpha);
      continue;
    }

    // Otherwise gather the strided dense RHS column into contiguous storage,
    // solve it, then scatter the contiguous result back to the caller's dense
    // output layout.
    for (perflibs_int_t row = 0; row < supernodal.n; ++row) {
      ybuf[row] = y_col[row * y_stride_row];
    }

    spsv_supernodal<T>(supernodal, trans, xbuf.data(), ybuf.data(), alpha);

    for (perflibs_int_t row = 0; row < supernodal.n; ++row) {
      x_col[row * x_stride_row] = xbuf[row];
    }
  }
}

template void spsv_supernodal<float>(perflibs_supernodal<float> &supernodal,
                                     perflibs_sparse_hint_value trans, float *x,
                                     const float *y, float alpha);
template void spsv_supernodal<double>(perflibs_supernodal<double> &supernodal,
                                      perflibs_sparse_hint_value trans,
                                      double *x, const double *y, double alpha);
template void spsv_supernodal<std::complex<float>>(
    perflibs_supernodal<std::complex<float>> &supernodal,
    perflibs_sparse_hint_value trans, std::complex<float> *x,
    const std::complex<float> *y, std::complex<float> alpha);
template void spsv_supernodal<std::complex<double>>(
    perflibs_supernodal<std::complex<double>> &supernodal,
    perflibs_sparse_hint_value trans, std::complex<double> *x,
    const std::complex<double> *y, std::complex<double> alpha);
template void
spsm_supernodal<float>(perflibs_supernodal<float> &supernodal,
                       perflibs_sparse_hint_value trans, float *X,
                       perflibs_int_t x_stride_row, perflibs_int_t x_stride_col,
                       float alpha, const float *Y, perflibs_int_t y_stride_row,
                       perflibs_int_t y_stride_col, perflibs_int_t nrhs);
template void spsm_supernodal<double>(
    perflibs_supernodal<double> &supernodal, perflibs_sparse_hint_value trans,
    double *X, perflibs_int_t x_stride_row, perflibs_int_t x_stride_col,
    double alpha, const double *Y, perflibs_int_t y_stride_row,
    perflibs_int_t y_stride_col, perflibs_int_t nrhs);
template void spsm_supernodal<std::complex<float>>(
    perflibs_supernodal<std::complex<float>> &supernodal,
    perflibs_sparse_hint_value trans, std::complex<float> *X,
    perflibs_int_t x_stride_row, perflibs_int_t x_stride_col,
    std::complex<float> alpha, const std::complex<float> *Y,
    perflibs_int_t y_stride_row, perflibs_int_t y_stride_col,
    perflibs_int_t nrhs);
template void spsm_supernodal<std::complex<double>>(
    perflibs_supernodal<std::complex<double>> &supernodal,
    perflibs_sparse_hint_value trans, std::complex<double> *X,
    perflibs_int_t x_stride_row, perflibs_int_t x_stride_col,
    std::complex<double> alpha, const std::complex<double> *Y,
    perflibs_int_t y_stride_row, perflibs_int_t y_stride_col,
    perflibs_int_t nrhs);

} // namespace perflibs::sparse
