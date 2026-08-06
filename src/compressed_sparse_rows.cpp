/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "compressed_sparse_rows.hpp"
#include "convert.hpp"
#include "object_helpers.hpp"
#include "types.hpp"

#include <algorithm>
#include <atomic>
#include <limits>
#include <type_traits>

// Implementations of functions required to support CSR

namespace perflibs::sparse {

template <typename T>
perflibs_csr<T> make_csr(perflibs_int_t rows, perflibs_int_t cols,
                         perflibs_int_t nnz, const T *vals,
                         const perflibs_int_t *row_ptr,
                         const perflibs_int_t *col_indx, bool no_copy) {
  if (no_copy) {
    return perflibs_csr<T>(rows, cols, vals, row_ptr, col_indx, {});
  } else {
    return perflibs_csr<T>(rows, cols, nnz, vals, row_ptr, col_indx);
  }
}

template <typename T>
perflibs_csr<T>
make_csr(perflibs_int_t rows, perflibs_int_t cols, perflibs_int_t nnz,
         perflibs::sparse::pod_vector<T> vals,
         std::vector<perflibs_int_t> row_ptr,
         perflibs::sparse::pod_vector<perflibs_int_t> col_indx) {
  return perflibs_csr<T>(rows, cols, std::move(vals), std::move(row_ptr),
                         std::move(col_indx));
}

template <typename T>
perflibs_status_t check_csr_params(perflibs_spmat_impl_t<T> *impl,
                                   perflibs_int_t m, perflibs_int_t n,
                                   const perflibs_int_t *row_ptr) {
  /* Check input values */
  if (m < 0) {
    impl->error_handle.perflibs_error_type =
        PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    impl->error_handle.perflibs_error_code = 2;
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }
  if (n < 0) {
    impl->error_handle.perflibs_error_type =
        PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    impl->error_handle.perflibs_error_code = 3;
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }
  if (row_ptr[0] != 0 && row_ptr[0] != 1) {
    impl->error_handle.perflibs_error_type =
        PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    impl->error_handle.perflibs_error_code = 4;
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }
  return PERFLIBS_STATUS_SUCCESS;
}

template <typename T>
void init_csr(perflibs_spmat_impl_t<T> *impl, perflibs_int_t m,
              perflibs_int_t n, const perflibs_int_t *row_ptr, bool no_copy) {
  impl->m = m;
  impl->n = n;
  impl->index_base = row_ptr[0];
  impl->nnz = row_ptr[m] - impl->index_base;
  impl->spmat_format = perflibs_format_csr;
  impl->no_copy = no_copy;
}

struct csr_triangular_solve_info {
  perflibs_int_t off;
  bool known_diag;
  bool unit;
  bool upper;
  perflibs_int_t diag_lo;
  perflibs_int_t diag_hi;
};

inline csr_triangular_solve_info
get_csr_triangular_solve_info(const perflibs_int_t *row_ptr,
                              sparse_hint_value_internal uplo,
                              sparse_hint_value_internal diag) {
  const bool known_diag =
      diag == PERFLIBS_DIAG_KNOWN_UNIT || diag == PERFLIBS_DIAG_KNOWN_NON_UNIT;
  const bool unit =
      diag == PERFLIBS_DIAG_KNOWN_UNIT || diag == PERFLIBS_DIAG_UNIT;
  const bool upper = uplo == PERFLIBS_SHAPE_UPPER_TRIANGULAR;
  const perflibs_int_t diag_lo =
      known_diag && upper; // add one to the first row index pos if we know
                           // where the diag is and it's upper
  const perflibs_int_t diag_hi =
      known_diag && !upper; // subtract one from the last row index pos if we
                            // know where the diag is and it's lower
  return {row_ptr[0], known_diag, unit, upper, diag_lo, diag_hi};
}

perflibs_sparse_matrix_shape_t get_shape_csr(perflibs_int_t m, perflibs_int_t n,
                                             const perflibs_int_t *row_ptr,
                                             const perflibs_int_t *col_indx) {

  // We currently only care about shape for spsv, which requires a square
  // matrix, so get out early if the matrix is rectangular.
  if (m != n) {
    return PERFLIBS_SPARSE_SHAPE_RECTANGULAR;
  }

  perflibs_sparse_matrix_shape_t current =
      PERFLIBS_SPARSE_SHAPE_DIAGONAL; // not strictly upper or lower

  auto index_base = row_ptr[0];
  for (perflibs_int_t i = 0; i < m; i++) {
    // offset for start of row
    const auto row_start = &col_indx[row_ptr[i] - index_base];
    // offset for end of row
    const auto row_end = &col_indx[row_ptr[i + 1] - index_base];

    if (row_end != row_start) {
      // Just check the min and max elements
      auto [minp, maxp] = std::minmax_element(row_start, row_end);
      auto ci_min = *minp - index_base;
      auto ci_max = *maxp - index_base;

      // If the row only has a diagonal element, skip any more checks
      if (!(ci_min == i && ci_max == i)) {
        if (ci_min <= i && ci_max <= i) {
          // we're in lower triangular territory, get out if we've previously
          // seen evidence of upper
          if (current == PERFLIBS_SPARSE_SHAPE_UPPER_TRIANGULAR) {
            return PERFLIBS_SPARSE_SHAPE_RECTANGULAR;
          } else {
            current = PERFLIBS_SPARSE_SHAPE_LOWER_TRIANGULAR;
          }
        } else if (ci_min >= i && ci_max >= i) {
          // we're in upper triangular territory, get out if we've previously
          // seen evidence of lower
          if (current == PERFLIBS_SPARSE_SHAPE_LOWER_TRIANGULAR) {
            return PERFLIBS_SPARSE_SHAPE_RECTANGULAR;
          } else {
            current = PERFLIBS_SPARSE_SHAPE_UPPER_TRIANGULAR;
          }
        }
        // not diagonal, and not upper or lower: rectangular
        else {
          return PERFLIBS_SPARSE_SHAPE_RECTANGULAR;
        }
      }
    }
  }

  // TODO handle diagonal specifically
  // For now, we just handle as a lower triangular matrix for CSR
  if (current == PERFLIBS_SPARSE_SHAPE_DIAGONAL) {
    return PERFLIBS_SPARSE_SHAPE_LOWER_TRIANGULAR;
  }

  return current;
}

template <typename T>
perflibs_sparse_matrix_diag_t
get_diag_csr(perflibs_int_t m, perflibs_int_t n, const perflibs_int_t *row_ptr,
             const perflibs_int_t *col_indx, const T *vals,
             perflibs_sparse_matrix_shape_t shape) {

  // We currently only care about diagonals for spsv, which requires a square
  // matrix, so get out early if the matrix is rectangular.
  if (m != n || shape == PERFLIBS_SPARSE_SHAPE_RECTANGULAR) {
    return PERFLIBS_SPARSE_DIAG_NON_UNIT;
  }

  auto index_base = row_ptr[0];

  perflibs_int_t unit_diag = 0;
  // Iterate over rows, returning early if we find a zero diagonal. If we find a
  // non-unit diagonal, we need to continue checking to see if there are any
  // zero diagonals in the remaining rows.
  for (perflibs_int_t i = 0; i < m; i++) {

    // If the row is empty we have a zero diagonal
    if (row_ptr[i + 1] - row_ptr[i] == 0) {
      return PERFLIBS_SPARSE_DIAG_ZERO;
    }

    // offset for start of row
    const auto row_start = &col_indx[row_ptr[i] - index_base];
    // offset for end of row
    const auto row_end = &col_indx[row_ptr[i + 1] - index_base];

    // offset for min/max element of row from the beginning
    const auto indx =
        std::distance(col_indx, shape == PERFLIBS_SPARSE_SHAPE_LOWER_TRIANGULAR
                                    ? std::max_element(row_start, row_end)
                                    : std::min_element(row_start, row_end));

    const auto ci = col_indx[indx] - index_base;

    // If this is the diagonal ...
    if (ci == i) {
      // Return if the value is zero
      if (vals[indx] == T(0)) {
        return PERFLIBS_SPARSE_DIAG_ZERO;
      }
      // count if unit diagonal
      if (vals[indx] == T(1)) {
        unit_diag++;
      }
    }
  }

  // A unit diagonal matrix has to have all diagonal entries equal to 1
  if (unit_diag == n) {
    return PERFLIBS_SPARSE_DIAG_UNIT;
  }

  return PERFLIBS_SPARSE_DIAG_NON_UNIT;
}

template <typename T>
perflibs_status_t
fill_initial_data_csr(perflibs_spmat_top_t *A, perflibs_int_t m,
                      perflibs_int_t n, std::vector<perflibs_int_t> &&row_ptr,
                      perflibs::sparse::pod_vector<perflibs_int_t> &&col_indx,
                      perflibs::sparse::pod_vector<T> &&vals) {

  auto impl = reinterpret_cast<perflibs_spmat_impl_t<T> *>(A->impl);
  auto ret = check_csr_params(impl, m, n, row_ptr.data());
  if (ret == PERFLIBS_STATUS_SUCCESS) {
    init_csr(impl, m, n, row_ptr.data(), false);
  } else {
    return ret;
  }

  impl->shape = get_shape_csr(m, n, row_ptr.data(), col_indx.data());
  impl->diag = get_diag_csr(m, n, row_ptr.data(), col_indx.data(), vals.data(),
                            impl->shape);

  impl->csr = make_csr<T>(impl->m, impl->n, impl->nnz, std::move(vals),
                          std::move(row_ptr), std::move(col_indx));
  return PERFLIBS_STATUS_SUCCESS;
}
template perflibs_status_t fill_initial_data_csr<float>(
    perflibs_spmat_top_t *A, perflibs_int_t m, perflibs_int_t n,
    std::vector<perflibs_int_t> &&row_ptr,
    perflibs::sparse::pod_vector<perflibs_int_t> &&col_indx,
    perflibs::sparse::pod_vector<float> &&vals);
template perflibs_status_t fill_initial_data_csr<double>(
    perflibs_spmat_top_t *A, perflibs_int_t m, perflibs_int_t n,
    std::vector<perflibs_int_t> &&row_ptr,
    perflibs::sparse::pod_vector<perflibs_int_t> &&col_indx,
    perflibs::sparse::pod_vector<double> &&vals);
template perflibs_status_t fill_initial_data_csr<std::complex<float>>(
    perflibs_spmat_top_t *A, perflibs_int_t m, perflibs_int_t n,
    std::vector<perflibs_int_t> &&row_ptr,
    perflibs::sparse::pod_vector<perflibs_int_t> &&col_indx,
    perflibs::sparse::pod_vector<std::complex<float>> &&vals);
template perflibs_status_t fill_initial_data_csr<std::complex<double>>(
    perflibs_spmat_top_t *A, perflibs_int_t m, perflibs_int_t n,
    std::vector<perflibs_int_t> &&row_ptr,
    perflibs::sparse::pod_vector<perflibs_int_t> &&col_indx,
    perflibs::sparse::pod_vector<std::complex<double>> &&vals);

template <typename T>
perflibs_status_t fill_initial_data_csr(perflibs_spmat_top_t *A,
                                        perflibs_int_t m, perflibs_int_t n,
                                        const perflibs_int_t *row_ptr,
                                        const perflibs_int_t *col_indx,
                                        const T *vals, bool no_copy) {
  auto impl = reinterpret_cast<perflibs_spmat_impl_t<T> *>(A->impl);
  auto ret = check_csr_params(impl, m, n, row_ptr);
  if (ret == PERFLIBS_STATUS_SUCCESS) {
    init_csr(impl, m, n, row_ptr, no_copy);
  } else {
    return ret;
  }

  impl->shape = get_shape_csr(m, n, row_ptr, col_indx);
  impl->diag = get_diag_csr(m, n, row_ptr, col_indx, vals, impl->shape);

  impl->csr = make_csr<T>(impl->m, impl->n, impl->nnz, vals, row_ptr, col_indx,
                          no_copy);

  return PERFLIBS_STATUS_SUCCESS;
}
template perflibs_status_t
fill_initial_data_csr<float>(perflibs_spmat_top_t *A, perflibs_int_t m,
                             perflibs_int_t n, const perflibs_int_t *row_ptr,
                             const perflibs_int_t *col_indx, const float *vals,
                             bool no_copy);
template perflibs_status_t
fill_initial_data_csr<double>(perflibs_spmat_top_t *A, perflibs_int_t m,
                              perflibs_int_t n, const perflibs_int_t *row_ptr,
                              const perflibs_int_t *col_indx,
                              const double *vals, bool no_copy);
template perflibs_status_t fill_initial_data_csr<std::complex<float>>(
    perflibs_spmat_top_t *A, perflibs_int_t m, perflibs_int_t n,
    const perflibs_int_t *row_ptr, const perflibs_int_t *col_indx,
    const std::complex<float> *vals, bool no_copy);
template perflibs_status_t fill_initial_data_csr<std::complex<double>>(
    perflibs_spmat_top_t *A, perflibs_int_t m, perflibs_int_t n,
    const perflibs_int_t *row_ptr, const perflibs_int_t *col_indx,
    const std::complex<double> *vals, bool no_copy);

template <typename T>
perflibs_csr<T> &perflibs_csr<T>::operator=(const perflibs_csr<T> &other) {
  if (&other == this) {
    return *this;
  }

  // Copy the vector variables
  m = other.m;
  n = other.n;
  use_vanilla = other.use_vanilla;
  par_mv = other.par_mv;

  if (m >= 0 && n >= 0) {
    auto nnz = other.row_ptr_ptr[m] - other.row_ptr_ptr[0];

    // If the vectors are populated make the const pointers point to them
    // otherwise construct a new vector
    copy_from_vector_or_ptr(&vals_ptr, vals, other.vals_ptr, other.vals, nnz);

    copy_from_vector_or_ptr(&col_indx_ptr, col_indx, other.col_indx_ptr,
                            other.col_indx, nnz);

    copy_from_vector_or_ptr(&row_ptr_ptr, row_ptr, other.row_ptr_ptr,
                            other.row_ptr, m + 1);
  }

  return *this;
}
template perflibs_csr<float> &
perflibs_csr<float>::operator=(const perflibs_csr<float> &other);
template perflibs_csr<double> &
perflibs_csr<double>::operator=(const perflibs_csr<double> &other);
template perflibs_csr<std::complex<float>> &
perflibs_csr<std::complex<float>>::operator=(
    const perflibs_csr<std::complex<float>> &other);
template perflibs_csr<std::complex<double>> &
perflibs_csr<std::complex<double>>::operator=(
    const perflibs_csr<std::complex<double>> &other);

// Scale the input values and write them into the current matrix object
template <typename T>
void perflibs_csr<T>::scale_matrix(enum perflibs_sparse_hint_value trans,
                                   T alpha) {
  auto nnz = row_ptr[m] - row_ptr[0];

  // Allocate our copy vector (in case we're copying from user's data)
  if (vals.data() != vals_ptr) {
    vals.resize(nnz);
  }

  if (trans == PERFLIBS_SPARSE_OPERATION_CONJTRANS) {
    for (perflibs_int_t i = 0; i < nnz; i++) {
      vals[i] = perflibs::sparse::conj(vals_ptr[i]) * alpha;
    }
  } else {
    for (perflibs_int_t i = 0; i < nnz; i++) {
      vals[i] = vals_ptr[i] * alpha;
    }
  }

  // Set the new pointer
  vals_ptr = vals.data();
}
template void
perflibs_csr<float>::scale_matrix(enum perflibs_sparse_hint_value trans,
                                  float alpha);
template void
perflibs_csr<double>::scale_matrix(enum perflibs_sparse_hint_value trans,
                                   double alpha);
template void perflibs_csr<std::complex<float>>::scale_matrix(
    enum perflibs_sparse_hint_value trans, std::complex<float> alpha);
template void perflibs_csr<std::complex<double>>::scale_matrix(
    enum perflibs_sparse_hint_value trans, std::complex<double> alpha);

template <typename T>
void perflibs_csr<T>::gen_parallel_decomp_mv(
    perflibs_parallel_decomp_strategy strat) {
  const auto nthreads = par_mv.nthreads;

  if (nthreads == 1) {
    return; // Nothing to do
  }

  auto off = row_ptr[0];
  auto nnz = row_ptr[m] - off;

  // Ensure that the size of the parallel decomposition vectors is correct
  par_mv.thread_rows.resize(nthreads + 1);
  auto thread_rows = par_mv.thread_rows.data();
  par_mv.thread_inds.resize(nthreads + 1);
  auto thread_inds = par_mv.thread_inds.data();

  if (strat == perflibs_parallel_decomp_strategy::static_rows) {
    // Split the rows up as evenly as possible over the number of threads. This
    // is a cheap and effective strategy if the number of non-zeros per row is
    // similar
    perflibs_int_t rows_per_thread = iround_div(m, nthreads);
    perflibs_int_t i;
    for (i = 0; i < nthreads; ++i) {
      // Apart from the last thread, assign all of the rows_per_thread to the
      // current thread
      if (i * rows_per_thread >= m) {
        // Check that we haven't gone past the end of the matrix. Can
        // occur if the number of threads is close to the number of rows
        break;
      }
      thread_rows[i] = i * rows_per_thread;
      thread_inds[i] = row_ptr[i * rows_per_thread] - off;
    }
    for (; i < nthreads + 1; ++i) {
      thread_inds[i] = row_ptr[m] - row_ptr[0];
      thread_rows[i] = m;
    }
  } else if (strat == perflibs_parallel_decomp_strategy::nnz_full_rows) {
    perflibs_int_t min_nnz = std::numeric_limits<perflibs_int_t>::max();
    perflibs_int_t max_nnz = 0;
    // Get the target for the number of non-zero elements we want per thread
    auto nnz_per_thread = iround_div(nnz, nthreads);
    thread_inds[0] = 0;
    thread_rows[0] = 0;
    perflibs_int_t curr_row = 0;
    // Loop over the number of threads
    for (perflibs_int_t t = 1; t < nthreads; ++t) {
      if (curr_row == m) {
        thread_rows[t] = m;
        thread_inds[t] = nnz;
        // Store the number of non-zeros in this row
        min_nnz = std::min(min_nnz, thread_inds[t] - thread_inds[t - 1]);
        max_nnz = std::max(max_nnz, thread_inds[t] - thread_inds[t - 1]);
        continue;
      }
      // The number of non-zero values assigned to this thread, and
      // assign at least one row
      auto thread_nnz = row_ptr[curr_row + 1] - row_ptr[curr_row];
      curr_row++;
      while (thread_nnz < nnz_per_thread && curr_row < m) {
        auto diff = row_ptr[curr_row + 1] - row_ptr[curr_row];
        // Make sure that we don't assign too much to a single thread
        if (thread_nnz + diff > 1.1 * nnz_per_thread)
          break;
        thread_nnz += diff;
        curr_row++;
      }
      thread_rows[t] = curr_row;
      thread_inds[t] = row_ptr[curr_row] - off;
      // Store the number of non-zeros in this row
      min_nnz = std::min(min_nnz, thread_inds[t] - thread_inds[t - 1]);
      max_nnz = std::max(max_nnz, thread_inds[t] - thread_inds[t - 1]);
    }
    thread_rows[nthreads] = m;
    thread_inds[nthreads] = nnz;
    min_nnz =
        std::min(min_nnz, thread_inds[nthreads] - thread_inds[nthreads - 1]);
    max_nnz =
        std::max(max_nnz, thread_inds[nthreads] - thread_inds[nthreads - 1]);
    // Store the number of non-zeros in this row

    // Check that the distribution is relatively even
    if (1.5 * min_nnz < max_nnz) {
      // In the case that we have some significant outliers, split the matrix up
      // by the number of non-zero values
      gen_parallel_decomp_mv(perflibs_parallel_decomp_strategy::nnz);
    }
  } else { // perflibs_parallel_decomp_strategy::nnz
    auto nnz_per_thread = iround_div(nnz, nthreads);
    thread_inds[0] = 0;
    thread_rows[0] = 0;
    thread_inds[nthreads] = nnz;
    thread_rows[nthreads] = m;
    perflibs_int_t curr_row = 0;

    // For each of the threads, assign the number of non-zeros, and also figure
    // out which rows are assigned
    perflibs_int_t t;
    for (t = 1; t < nthreads; ++t) {
      thread_inds[t] = thread_inds[t - 1] + nnz_per_thread;
      if (thread_inds[t] >= nnz) {
        thread_inds[t] = nnz;
        curr_row = m;
        t++;
        break;
      }
      while (row_ptr[curr_row] - off <= thread_inds[t]) {
        curr_row++;
      }
      thread_rows[t] = --curr_row;
    }
    for (; t < nthreads; ++t) {
      thread_inds[t] = nnz;
      thread_rows[t] = m;
    }
  }
}

template void perflibs_csr<float>::gen_parallel_decomp_mv(
    perflibs_parallel_decomp_strategy strat);
template void perflibs_csr<double>::gen_parallel_decomp_mv(
    perflibs_parallel_decomp_strategy strat);
template void perflibs_csr<std::complex<float>>::gen_parallel_decomp_mv(
    perflibs_parallel_decomp_strategy strat);
template void perflibs_csr<std::complex<double>>::gen_parallel_decomp_mv(
    perflibs_parallel_decomp_strategy strat);

template <typename T> void perflibs_csr<T>::gen_data_vectors_mv() {
  const auto nthreads = par_mv.nthreads;
  const auto thread_inds = par_mv.thread_inds.data();

  if (nthreads == 1 || par_mv.thread_inds.empty()) {
    return; // Nothing to do
  }

  decltype(vals) new_vals(vals.size());
  decltype(col_indx) new_col_indx(col_indx.size());
#pragma omp parallel for default(none) schedule(static)                        \
    shared(vals, col_indx, new_vals, new_col_indx)                             \
    firstprivate(nthreads, thread_inds) num_threads(nthreads)
  for (perflibs_int_t t = nthreads - 1; t >= 0; --t) {
    // For each of the threads, loop over the values and column indices,
    // and copy these into the uninitialized vector
    for (perflibs_int_t i = thread_inds[t]; i < thread_inds[t + 1]; ++i) {
      new_vals[i] = vals[i];
      new_col_indx[i] = col_indx[i];
    }
  }

  // Swap the data around
  vals = std::move(new_vals);
  col_indx = std::move(new_col_indx);

  // Remember to update the pointers as well
  vals_ptr = vals.data();
  col_indx_ptr = col_indx.data();
}
template void perflibs_csr<float>::gen_data_vectors_mv();
template void perflibs_csr<double>::gen_data_vectors_mv();
template void perflibs_csr<std::complex<float>>::gen_data_vectors_mv();
template void perflibs_csr<std::complex<double>>::gen_data_vectors_mv();

template <typename T1>
template <typename T2>
void perflibs_csr<T1>::gen_parallel_decomp_sv(
    perflibs_sparse_matrix_shape_t shape, const perflibs_int_t *col_ptr,
    const perflibs_int_t *row_indx) {
  perflibs::sparse::pod_vector<T2> ndeps(n);
  for (perflibs_int_t i = 0; i < n; i++) {
    uint64_t ndeps_row =
        row_ptr[i + 1] - row_ptr[i] - 1; // -1: discount diag, rows are not
                                         // considered dependents of themselves
    if (ndeps_row > std::numeric_limits<T2>::max()) {
      gen_parallel_decomp_sv<typename next_int<T2>::type>(shape, col_ptr,
                                                          row_indx);
      return;
    } else {
      ndeps[i] = ndeps_row;
    }
  }

  gen_parallel_decomp_sv_csx<T2>(shape, n, col_ptr, row_indx, ndeps.data(),
                                 par_sv);
}
template void perflibs_csr<float>::gen_parallel_decomp_sv<uint8_t>(
    perflibs_sparse_matrix_shape_t, const perflibs_int_t *,
    const perflibs_int_t *);
template void perflibs_csr<double>::gen_parallel_decomp_sv<uint8_t>(
    perflibs_sparse_matrix_shape_t, const perflibs_int_t *,
    const perflibs_int_t *);
template void
perflibs_csr<std::complex<float>>::gen_parallel_decomp_sv<uint8_t>(
    perflibs_sparse_matrix_shape_t, const perflibs_int_t *,
    const perflibs_int_t *);
template void
perflibs_csr<std::complex<double>>::gen_parallel_decomp_sv<uint8_t>(
    perflibs_sparse_matrix_shape_t, const perflibs_int_t *,
    const perflibs_int_t *);

/** Just converts to csr then calls into csr2scs */
template <typename T>
perflibs_csr<T> coo2csr(perflibs_int_t m, perflibs_int_t n, perflibs_int_t nnz,
                        const T *vals, const perflibs_int_t *col_indx,
                        const perflibs_int_t *row_indx,
                        perflibs_int_t index_base) {
  // Sort first, then compress. This could potentially be faster with some
  // reuse of allocated memory.
  // TODO Change to perflibs_int_t, currently this won't work for int32 SVE
  // builds because of a weird compiler bug (?)
  std::vector<std::pair<T, int64_t>> coords_1d;
  coords_1d.reserve(nnz);
  for (auto ii = 0; ii < nnz; ++ii) {
    // We promote row_indx to int64_t before multiplying in order to
    // avoid overflow bug
    coords_1d.emplace_back(
        vals[ii],
        (col_indx[ii] - index_base + ((int64_t)row_indx[ii] - index_base) * n));
  }
  // Sort by the 2nd item in the tuple, which is the index into the matrix
  std::sort(coords_1d.begin(), coords_1d.end(),
            [](auto x_, auto y_) { return x_.second < y_.second; });

  std::vector<perflibs_int_t> row_ptr;
  row_ptr.reserve(m + 1);
  row_ptr.push_back(index_base);
  perflibs_int_t nz_count = index_base;
  perflibs_int_t curr_row = 1;
  for (perflibs_int_t ii = 0; ii < nnz; ++ii) {
    // This is a loop rather than an if for long blocks of 0 rows.
    while (coords_1d[ii].second / n > curr_row - 1) {
      row_ptr.push_back(nz_count);
      curr_row += 1;
    }
    nz_count += 1;
  }
  for (auto ii = curr_row; ii < m + 1; ++ii) {
    row_ptr.push_back(nz_count);
  }

  // Get the col_indx and vals sorted
  std::vector<perflibs_int_t> new_col_indx;
  std::vector<T> new_vals;
  new_col_indx.reserve(nnz);
  new_vals.reserve(nnz);
  for (auto ii = 0; ii < nnz; ++ii) {
    new_col_indx.push_back((coords_1d[ii].second % n) + index_base);
    new_vals.push_back(coords_1d[ii].first);
  }
  return perflibs_csr<T>(m, n, nnz, new_vals.data(), row_ptr.data(),
                         new_col_indx.data());
};
template perflibs_csr<float>
coo2csr<float>(perflibs_int_t m, perflibs_int_t n, perflibs_int_t nnz,
               const float *vals, const perflibs_int_t *col_indx,
               const perflibs_int_t *row_indx, perflibs_int_t index_base);
template perflibs_csr<double>
coo2csr<double>(perflibs_int_t m, perflibs_int_t n, perflibs_int_t nnz,
                const double *vals, const perflibs_int_t *col_indx,
                const perflibs_int_t *row_indx, perflibs_int_t index_base);
template perflibs_csr<std::complex<float>> coo2csr<std::complex<float>>(
    perflibs_int_t m, perflibs_int_t n, perflibs_int_t nnz,
    const std::complex<float> *vals, const perflibs_int_t *col_indx,
    const perflibs_int_t *row_indx, perflibs_int_t index_base);
template perflibs_csr<std::complex<double>> coo2csr<std::complex<double>>(
    perflibs_int_t m, perflibs_int_t n, perflibs_int_t nnz,
    const std::complex<double> *vals, const perflibs_int_t *col_indx,
    const perflibs_int_t *row_indx, perflibs_int_t index_base);

template <typename T>
perflibs_csr<T> dense2csr(perflibs_dense_layout layout, perflibs_int_t m,
                          perflibs_int_t n, perflibs_int_t lda, const T *A,
                          perflibs_int_t index_base) {

  if (layout == PERFLIBS_COL_MAJOR) {
    assert(lda >= m);
  } else if (layout == PERFLIBS_ROW_MAJOR) {
    assert(lda >= n);
  }

  size_t nnz = 0;
  std::vector<perflibs_int_t> row_ptr(m + 1);

  // Our first job is to figure out how many non-zeros there are, and at the
  // same time find the number of non-zeros in each row
  if (layout == PERFLIBS_COL_MAJOR) {
    for (perflibs_int_t j = 0; j < n; j++) {
      for (perflibs_int_t i = 0; i < m; i++) {
        if (A[j * lda + i] != (T)0) {
          nnz++;
          row_ptr[i + 1]++;
        }
      }
    }
  } else {
    for (perflibs_int_t i = 0; i < m; i++) {
      for (perflibs_int_t j = 0; j < n; j++) {
        if (A[i * lda + j] != (T)0) {
          nnz++;
          row_ptr[i + 1]++;
        }
      }
    }
  }

  // Now construct row_ptr
  row_ptr[0] = index_base;
  for (perflibs_int_t i = 1; i <= m; i++) {
    row_ptr[i] += row_ptr[i - 1];
  }

  std::vector<T> vals(nnz);
  std::vector<perflibs_int_t> col_indx(nnz);

  // Enter the non-zero values and column indices into the array
  // Note that this involves accessing A in the dis-contiguous order, so if
  // performance is a problem with this function this loop nest is a prime
  // candidate for optimization
  size_t k = 0;
  if (layout == PERFLIBS_COL_MAJOR) {
    for (perflibs_int_t i = 0; i < m; i++) {
      for (perflibs_int_t j = 0; j < n; j++) {
        if (A[j * lda + i] != (T)0) {
          col_indx[k] = j + index_base;
          vals[k++] = A[j * lda + i];
        }
      }
    }
  } else {
    for (perflibs_int_t i = 0; i < m; i++) {
      for (perflibs_int_t j = 0; j < n; j++) {
        if (A[i * lda + j] != (T)0) {
          col_indx[k] = j + index_base;
          vals[k++] = A[i * lda + j];
        }
      }
    }
  }

  return perflibs_csr<T>(m, n, nnz, vals.data(), row_ptr.data(),
                         col_indx.data());
}

template perflibs_csr<float> dense2csr(perflibs_dense_layout layout,
                                       perflibs_int_t m, perflibs_int_t n,
                                       perflibs_int_t lda, const float *A,
                                       perflibs_int_t index_base);
template perflibs_csr<double> dense2csr(perflibs_dense_layout layout,
                                        perflibs_int_t m, perflibs_int_t n,
                                        perflibs_int_t lda, const double *A,
                                        perflibs_int_t index_base);
template perflibs_csr<std::complex<float>>
dense2csr(perflibs_dense_layout layout, perflibs_int_t m, perflibs_int_t n,
          perflibs_int_t lda, const std::complex<float> *A,
          perflibs_int_t index_base);
template perflibs_csr<std::complex<double>>
dense2csr(perflibs_dense_layout layout, perflibs_int_t m, perflibs_int_t n,
          perflibs_int_t lda, const std::complex<double> *A,
          perflibs_int_t index_base);

std::pair<std::vector<perflibs_int_t>, std::vector<perflibs_int_t>>
csc2csr_struct(int64_t m, int64_t n, const perflibs_int_t *row_indx,
               const perflibs_int_t *col_ptr) {

  auto index_base = col_ptr[0];
  auto nnz = col_ptr[n] - index_base;
  std::vector<perflibs_int_t> row_ptr(m + 1);
  std::vector<perflibs_int_t> col_indx(nnz);

  // Figure out how many elements live in each column
  row_ptr[0] = index_base;
  for (auto i = 0; i < col_ptr[n] - index_base; i++) {
    row_ptr[row_indx[i] + 1 - index_base]++;
  }

  // Turn this into a cumulative vector to get row_ptr
  for (auto i = 1; i <= m; i++) {
    row_ptr[i] += row_ptr[i - 1];
  }

  auto row_ptr_cpy = row_ptr;

  for (auto i = 0; i < n; i++) {
    for (auto k = col_ptr[i] - index_base; k < col_ptr[i + 1] - index_base;
         k++) {
      auto row = row_indx[k] - index_base;
      col_indx[row_ptr_cpy[row]++ - index_base] = i + index_base;
    }
  }

  return std::make_pair(std::move(row_ptr), std::move(col_indx));
}

template <typename T>
perflibs_csr<T> csc2csr(enum sparse_hint_value_internal trans, int64_t m,
                        int64_t n, const T *vals,
                        const perflibs_int_t *row_indx,
                        const perflibs_int_t *col_ptr) {

  auto [row_ptr, col_indx] = csc2csr_struct(m, n, row_indx, col_ptr);

  auto index_base = col_ptr[0];
  auto nnz = col_ptr[n] - index_base;
  std::vector<T> vals_csr(nnz);

  auto conj = [=](T val) {
    return trans == PERFLIBS_OPERATION_CONJTRANS ? perflibs::sparse::conj(val)
                                                 : val;
  };

  auto row_ptr_cpy = row_ptr;

  for (auto i = 0; i < n; i++) {
    for (auto k = col_ptr[i] - index_base; k < col_ptr[i + 1] - index_base;
         k++) {
      auto row = row_indx[k] - index_base;
      vals_csr[row_ptr_cpy[row]++ - index_base] = conj(vals[k]);
    }
  }

  return perflibs_csr<T>(m, n, nnz, vals_csr.data(), row_ptr.data(),
                         col_indx.data());
}
template perflibs_csr<float> csc2csr(enum sparse_hint_value_internal trans,
                                     int64_t m, int64_t n, const float *vals,
                                     const perflibs_int_t *row_indx,
                                     const perflibs_int_t *col_ptr);
template perflibs_csr<double> csc2csr(enum sparse_hint_value_internal trans,
                                      int64_t m, int64_t n, const double *vals,
                                      const perflibs_int_t *row_indx,
                                      const perflibs_int_t *col_ptr);
template perflibs_csr<std::complex<float>>
csc2csr(enum sparse_hint_value_internal trans, int64_t m, int64_t n,
        const std::complex<float> *vals, const perflibs_int_t *row_indx,
        const perflibs_int_t *col_ptr);
template perflibs_csr<std::complex<double>>
csc2csr(enum sparse_hint_value_internal trans, int64_t m, int64_t n,
        const std::complex<double> *vals, const perflibs_int_t *row_indx,
        const perflibs_int_t *col_ptr);

// Templated on T1, real type, and T2, the underlying type of the compressed
// col_indx_offsets, this function fills in the vals and col_indx arrays for
// conversion to CSR
template <typename T1, typename T2>
void scs2csr_get_nnz_vectors(
    perflibs_int_t m, perflibs_int_t C, perflibs::sparse::pod_vector<T1> &vals,
    perflibs::sparse::pod_vector<perflibs_int_t> &col_indx,
    std::vector<perflibs_int_t> row_ptr, std::vector<int64_t> &scs_row_permd2in,
    std::vector<int64_t> &scs_cl, std::vector<int64_t> &scs_cs,
    perflibs::sparse::pod_vector<T1> &scs_vals,
    perflibs::sparse::pod_vector<perflibs_int_t> &scs_col_indx_offsets,
    std::vector<int64_t> &col_indx_min) {

  auto scs_col_indx = reinterpret_cast<const T2 *>(scs_col_indx_offsets.data());

  // Now build up col_indx and vals
  perflibs_int_t ci = 0;
  for (perflibs_int_t i = 0; i < m; i++) {
    perflibs_int_t k = i % C; // The row within this chunk

    auto ii = scs_row_permd2in[i];

    // Iterate through scs.vals for the current chunk length
    for (perflibs_int_t j = 0; j < scs_cl[ci]; j++) {
      auto val = scs_vals[scs_cs[ci] + j * C + k];
      if (val != (T1)0) {
        vals[row_ptr[ii]] = val;
        col_indx[row_ptr[ii]] =
            scs_col_indx[scs_cs[ci] + j * C + k] + col_indx_min[ci];
        row_ptr[ii]++;
      }
    }

    // If the next row marks the start of a new chunk, increment the chunk index
    if ((i + 1) % C == 0) {
      ci++;
    }
  }
}

/// Convert from SCS back to CSR, mainly for export
template <typename T>
perflibs_csr<T>
scs2csr(perflibs_int_t m, perflibs_int_t n, perflibs_int_t C,
        perflibs::sparse::pod_vector<T> &scs_vals,
        perflibs::sparse::pod_vector<perflibs_int_t> &scs_col_indx_offsets,
        std::vector<int64_t> &col_indx_min, perflibs_int_t col_indx_bytes,
        std::vector<int64_t> &scs_cl, std::vector<int64_t> &scs_cs,
        std::vector<int64_t> scs_row_permd2in) {

  std::vector<perflibs_int_t> row_ptr(m + 1);

  // Keep track of which chunk we're in
  perflibs_int_t ci = 0;
  for (perflibs_int_t i = 0; i < m; i++) {
    perflibs_int_t k = i % C; // The row within this chunk

    auto ii = scs_row_permd2in[i];

    // Iterate through scs.vals for the current chunk length
    for (perflibs_int_t j = 0; j < scs_cl[ci]; j++) {
      auto val = scs_vals[scs_cs[ci] + j * C + k];
      if (val != (T)0) {
        row_ptr[ii + 1]++;
      }
    }

    // If the next row marks the start of a new chunk, increment the chunk index
    if ((i + 1) % C == 0) {
      ci++;
    }
  }

  for (perflibs_int_t i = 0; i < m; i++) {
    row_ptr[i + 1] += row_ptr[i];
  }

  auto nnz = row_ptr[m] - row_ptr[0];
  perflibs::sparse::pod_vector<perflibs_int_t> col_indx(nnz);
  perflibs::sparse::pod_vector<T> vals(nnz);

  switch (col_indx_bytes) {
  case (8):
    scs2csr_get_nnz_vectors<T, int64_t>(
        m, C, vals, col_indx, row_ptr, scs_row_permd2in, scs_cl, scs_cs,
        scs_vals, scs_col_indx_offsets, col_indx_min);
    break;
  case (4):
    scs2csr_get_nnz_vectors<T, int32_t>(
        m, C, vals, col_indx, row_ptr, scs_row_permd2in, scs_cl, scs_cs,
        scs_vals, scs_col_indx_offsets, col_indx_min);
    break;
  case (2):
    scs2csr_get_nnz_vectors<T, int16_t>(
        m, C, vals, col_indx, row_ptr, scs_row_permd2in, scs_cl, scs_cs,
        scs_vals, scs_col_indx_offsets, col_indx_min);
    break;
  case (1):
    scs2csr_get_nnz_vectors<T, int8_t>(
        m, C, vals, col_indx, row_ptr, scs_row_permd2in, scs_cl, scs_cs,
        scs_vals, scs_col_indx_offsets, col_indx_min);
    break;
  default:
    assert(false);
  }

  return perflibs_csr<T>(m, n, vals, row_ptr, col_indx);
}
template perflibs_csr<float>
scs2csr(perflibs_int_t m, perflibs_int_t n, perflibs_int_t C,
        perflibs::sparse::pod_vector<float> &scs_vals,
        perflibs::sparse::pod_vector<perflibs_int_t> &scs_col_indx_offsets,
        std::vector<int64_t> &col_indx_min, perflibs_int_t col_indx_bytes,
        std::vector<int64_t> &scs_cl, std::vector<int64_t> &scs_cs,
        std::vector<int64_t> scs_row_permd2in);
template perflibs_csr<double>
scs2csr(perflibs_int_t m, perflibs_int_t n, perflibs_int_t C,
        perflibs::sparse::pod_vector<double> &scs_vals,
        perflibs::sparse::pod_vector<perflibs_int_t> &scs_col_indx_offsets,
        std::vector<int64_t> &col_indx_min, perflibs_int_t col_indx_bytes,
        std::vector<int64_t> &scs_cl, std::vector<int64_t> &scs_cs,
        std::vector<int64_t> scs_row_permd2in);
template perflibs_csr<std::complex<float>>
scs2csr(perflibs_int_t m, perflibs_int_t n, perflibs_int_t C,
        perflibs::sparse::pod_vector<std::complex<float>> &scs_vals,
        perflibs::sparse::pod_vector<perflibs_int_t> &scs_col_indx_offsets,
        std::vector<int64_t> &col_indx_min, perflibs_int_t col_indx_bytes,
        std::vector<int64_t> &scs_cl, std::vector<int64_t> &scs_cs,
        std::vector<int64_t> scs_row_permd2in);
template perflibs_csr<std::complex<double>>
scs2csr(perflibs_int_t m, perflibs_int_t n, perflibs_int_t C,
        perflibs::sparse::pod_vector<std::complex<double>> &scs_vals,
        perflibs::sparse::pod_vector<perflibs_int_t> &scs_col_indx_offsets,
        std::vector<int64_t> &col_indx_min, perflibs_int_t col_indx_bytes,
        std::vector<int64_t> &scs_cl, std::vector<int64_t> &scs_cs,
        std::vector<int64_t> scs_row_permd2in);

template <typename T>
perflibs_csr<T> bsr2csr(perflibs_dense_layout block_layout, perflibs_int_t m,
                        perflibs_int_t n, perflibs_int_t block_size,
                        perflibs_int_t nnzb, perflibs_int_t nrowsb,
                        const T *vals, const perflibs_int_t *row_ptr,
                        const perflibs_int_t *col_indx) {

  auto index_base = row_ptr[0];
  auto bsr_nnz = nnzb * block_size * block_size;

  // Row Major representation of the BSR matrix
  std::vector<T> bsr_vals_rm(bsr_nnz);

  // First get BSR block_layout into ROW MAJOR and count csr_nnz at the same
  // time.
  perflibs_int_t csr_nnz = 0;
  for (perflibs_int_t block = 0; block < nnzb; block++) {
    auto block_offset = block_size * block_size * block;
    for (perflibs_int_t i = 0; i < block_size; i++) {
      for (perflibs_int_t j = 0; j < block_size; j++) {
        auto index = block_offset + i * block_size + j;
        auto trans_index = block_offset + i + block_size * j;
        auto reindex =
            (block_layout == PERFLIBS_COL_MAJOR) ? trans_index : index;
        bsr_vals_rm[index] = vals[reindex];

        // Check for non-zero elements
        if (bsr_vals_rm[index] != (T)0) {
          csr_nnz++;
        }
      }
    }
  }

  // kth value of diffs gives the number of non_zero blocks in the kth blocked
  // row.
  std::vector<perflibs_int_t> diffs(nrowsb);

  // Vectors to populate and return in perflibs_csr<T> constructor.
  std::vector<perflibs_int_t> csr_row_ptr;
  std::vector<perflibs_int_t> csr_col_indx;
  std::vector<T> csr_vals;

  csr_row_ptr.reserve(m + 1);
  csr_col_indx.reserve(csr_nnz);
  csr_vals.reserve(csr_nnz);

  // Restoring Base Index
  csr_row_ptr.push_back(index_base);

  // Construct all arrays here.
  // row_count stores the index of an entry in the column array offset by
  // index_base. Once the end of the CSR row is reached, the value of row_count
  // is pushed to the back of the csr_row_ptr array.
  perflibs_int_t row_count = index_base;
  for (perflibs_int_t rowb = 0; rowb < nrowsb; rowb++) {
    diffs[rowb] = row_ptr[rowb + 1] - row_ptr[rowb];
    if (diffs[rowb] == 0) {
      for (perflibs_int_t l = 0; l < block_size; l++) {
        csr_row_ptr.push_back(row_count);
      }
      continue;
    }
    perflibs_int_t row_offset =
        (row_ptr[rowb] - index_base) * block_size * block_size;
    for (perflibs_int_t i = 0; i < block_size; i++) {
      for (perflibs_int_t d = 0; d < diffs[rowb]; d++) {
        perflibs_int_t vals_offset =
            row_offset + d * block_size * block_size + i * block_size;
        perflibs_int_t col_indx_offset =
            block_size *
                (col_indx[row_ptr[rowb] + d - index_base] - index_base) +
            index_base;
        for (perflibs_int_t j = 0; j < block_size; j++) {
          if (bsr_vals_rm[vals_offset + j] == (T)0) {
            continue;
          }
          perflibs_int_t vals_index = vals_offset + j;
          perflibs_int_t col_indx_index = col_indx_offset + j;
          csr_vals.push_back(bsr_vals_rm[vals_index]);
          csr_col_indx.push_back(col_indx_index);
          row_count++;
        }
      }
      csr_row_ptr.push_back(row_count);
    }
  }

  return perflibs_csr<T>(m, n, csr_nnz, csr_vals.data(), csr_row_ptr.data(),
                         csr_col_indx.data());
}

template perflibs_csr<float>
bsr2csr(perflibs_dense_layout block_layout, perflibs_int_t m, perflibs_int_t n,
        perflibs_int_t block_size, perflibs_int_t nnzb, perflibs_int_t nrowsb,
        const float *vals, const perflibs_int_t *row_ptr,
        const perflibs_int_t *col_indx);
template perflibs_csr<double>
bsr2csr(perflibs_dense_layout block_layout, perflibs_int_t m, perflibs_int_t n,
        perflibs_int_t block_size, perflibs_int_t nnzb, perflibs_int_t nrowsb,
        const double *vals, const perflibs_int_t *row_ptr,
        const perflibs_int_t *col_indx);
template perflibs_csr<std::complex<float>>
bsr2csr(perflibs_dense_layout block_layout, perflibs_int_t m, perflibs_int_t n,
        perflibs_int_t block_size, perflibs_int_t nnzb, perflibs_int_t nrowsb,
        const std::complex<float> *vals, const perflibs_int_t *row_ptr,
        const perflibs_int_t *col_indx);
template perflibs_csr<std::complex<double>>
bsr2csr(perflibs_dense_layout block_layout, perflibs_int_t m, perflibs_int_t n,
        perflibs_int_t block_size, perflibs_int_t nnzb, perflibs_int_t nrowsb,
        const std::complex<double> *vals, const perflibs_int_t *row_ptr,
        const perflibs_int_t *col_indx);

/// Convert null into CSR format in order to complete export functionality
template <typename T>
perflibs_csr<T> null2csr(perflibs_int_t m, perflibs_int_t n) {
  std::vector<perflibs_int_t> row_ptr(m + 1);
  perflibs::sparse::pod_vector<perflibs_int_t> col_indx(0);
  perflibs::sparse::pod_vector<T> vals(0);
  return perflibs_csr<T>(m, n, vals, row_ptr, col_indx);
}
template perflibs_csr<float> null2csr(perflibs_int_t m, perflibs_int_t n);
template perflibs_csr<double> null2csr(perflibs_int_t m, perflibs_int_t n);
template perflibs_csr<std::complex<float>> null2csr(perflibs_int_t m,
                                                    perflibs_int_t n);
template perflibs_csr<std::complex<double>> null2csr(perflibs_int_t m,
                                                     perflibs_int_t n);

/// Convert identity into CSR format in order to complete export functionality
template <typename T> perflibs_csr<T> identity2csr(perflibs_int_t n) {
  std::vector<perflibs_int_t> row_ptr(n + 1);
  for (perflibs_int_t i = 0; i < n + 1; i++) {
    row_ptr[i] = i;
  }
  perflibs::sparse::pod_vector<perflibs_int_t> col_indx(n);
  for (perflibs_int_t i = 0; i < n; i++) {
    col_indx[i] = i;
  }
  perflibs::sparse::pod_vector<T> vals(n, (T)1);
  return perflibs_csr<T>(n, n, vals, row_ptr, col_indx);
}
template perflibs_csr<float> identity2csr(perflibs_int_t n);
template perflibs_csr<double> identity2csr(perflibs_int_t n);
template perflibs_csr<std::complex<float>> identity2csr(perflibs_int_t n);
template perflibs_csr<std::complex<double>> identity2csr(perflibs_int_t n);

/* CSR kernels */

template <typename T>
void spmv_csr_vanilla(const perflibs_csr<T> &csr, const T *x, T *y, T alpha,
                      T beta) {

  const auto row_ptr = csr.row_ptr_ptr;
  const auto col_indx = csr.col_indx_ptr;
  const auto vals = csr.vals_ptr;
  const auto off = row_ptr[0];

#pragma omp parallel for default(none) schedule(static)                        \
    firstprivate(row_ptr, col_indx, vals, off) shared(csr, x, y, alpha, beta)  \
    num_threads(csr.par_mv.nthreads)
  for (int i = 0; i < csr.m; i++) {
    T sum = 0;
    for (int j = row_ptr[i] - off; j < row_ptr[i + 1] - off; j++) {
      sum += x[col_indx[j] - off] * vals[j];
    }

    // Test beta not zero
    if (beta != (T)0)
      y[i] = y[i] * beta + alpha * sum;
    else
      y[i] = alpha * sum;
  }
};

template <typename T> inline T no_conj(T value) { return value; }

template <typename T> inline T conj(T value) {
  return perflibs::sparse::conj(value);
}

// OpenMP atomic does not work properly on Windows with clang.
// Use OpenMP critical instead until the upstream LLVM/OpenMP issue is resolved:
// https://github.com/llvm/llvm-project/issues/64694
inline void update_y(float &y, float value) {
#if defined(_WIN32)
#pragma omp critical(csr_update_y_s)
  {
    y += value;
  }
#else
#pragma omp atomic
  y += value;
#endif
}

inline void update_y(double &y, double value) {
#if defined(_WIN32)
#pragma omp critical(csr_update_y_d)
  {
    y += value;
  }
#else
#pragma omp atomic
  y += value;
#endif
}

inline void update_y(std::complex<float> &y, std::complex<float> value) {
  float *cplx_y = reinterpret_cast<float *>(&y);
#if defined(_WIN32)
#pragma omp critical(csr_update_y_c)
  {
    cplx_y[0] += value.real();
    cplx_y[1] += value.imag();
  }
#else
#pragma omp atomic
  cplx_y[0] += value.real();
#pragma omp atomic
  cplx_y[1] += value.imag();
#endif
}

inline void update_y(std::complex<double> &y, std::complex<double> value) {
  double *cplx_y = reinterpret_cast<double *>(&y);
#if defined(_WIN32)
#pragma omp critical(csr_update_y_z)
  {
    cplx_y[0] += value.real();
    cplx_y[1] += value.imag();
  }
#else
#pragma omp atomic
  cplx_y[0] += value.real();
#pragma omp atomic
  cplx_y[1] += value.imag();
#endif
}

/*
   Via the magic of templates we will get 2 copies for each of the
   spmv_csr_*trans functions, one in which conj is perflibs::sparse::conj and
   one in which conj is no_conj - where the compiler ought to recognize that it
   is a noop.
*/

// Perform SpMV with a default assumption of a parallel decomposition over rows
// of the matrix. This is used in the case that no parallel decomposition has
// been created in the CSR class
template <typename T, decltype(&no_conj<T>) conj>
static void spmv_csr_notrans_no_parallel_strat(const perflibs_csr<T> &csr,
                                               const T *x, T *y, T alpha,
                                               T beta) {
  const auto row_ptr = csr.row_ptr_ptr;
  const auto col_indx = csr.col_indx_ptr;
  const auto vals = csr.vals_ptr;
  const auto off = row_ptr[0];

#pragma omp parallel for default(none) schedule(static)                        \
    firstprivate(row_ptr, col_indx, vals, off) shared(csr, x, y, alpha, beta)  \
    num_threads(csr.par_mv.nthreads)
  for (perflibs_int_t i = 0; i < csr.m - 1; i += 2) {
    T sum0 = 0;
    T sum1 = 0;
    T sum2 = 0;
    T sum3 = 0;
    perflibs_int_t j, k;

    auto row_len_0 = row_ptr[i + 1] - row_ptr[i];
    auto row_len_1 = row_ptr[i + 2] - row_ptr[i + 1];

    if (row_len_0 == row_len_1) {
      for (j = row_ptr[i] - off, k = row_ptr[i + 1] - off;
           j < row_ptr[i + 1] - off - 1; j += 2, k += 2) {
        sum0 += conj(vals[j]) * x[col_indx[j] - off];
        sum1 += conj(vals[j + 1]) * x[col_indx[j + 1] - off];
        sum2 += conj(vals[k]) * x[col_indx[k] - off];
        sum3 += conj(vals[k + 1]) * x[col_indx[k + 1] - off];
      }
      for (; j < row_ptr[i + 1] - off; j++, k++) {
        sum0 += conj(vals[j]) * x[col_indx[j] - off];
        sum2 += conj(vals[k]) * x[col_indx[k] - off];
      }
    } else if (row_len_0 < row_len_1) {
      for (j = row_ptr[i] - off, k = row_ptr[i + 1] - off;
           j < row_ptr[i + 1] - off - 1; j += 2, k += 2) {
        sum0 += conj(vals[j]) * x[col_indx[j] - off];
        sum1 += conj(vals[j + 1]) * x[col_indx[j + 1] - off];
        sum2 += conj(vals[k]) * x[col_indx[k] - off];
        sum3 += conj(vals[k + 1]) * x[col_indx[k + 1] - off];
      }
      for (; j < row_ptr[i + 1] - off; j++, k++) {
        sum0 += conj(vals[j]) * x[col_indx[j] - off];
        sum2 += conj(vals[k]) * x[col_indx[k] - off];
      }
      for (; k < row_ptr[i + 2] - off - 1; k += 2) {
        sum2 += conj(vals[k]) * x[col_indx[k] - off];
        sum3 += conj(vals[k + 1]) * x[col_indx[k + 1] - off];
      }
      for (; k < row_ptr[i + 2] - off; k++) {
        sum2 += conj(vals[k]) * x[col_indx[k] - off];
      }
    } else if (row_len_0 > row_len_1) {

      for (j = row_ptr[i] - off, k = row_ptr[i + 1] - off;
           k < row_ptr[i + 2] - off - 1; j += 2, k += 2) {
        sum0 += conj(vals[j]) * x[col_indx[j] - off];
        sum1 += conj(vals[j + 1]) * x[col_indx[j + 1] - off];
        sum2 += conj(vals[k]) * x[col_indx[k] - off];
        sum3 += conj(vals[k + 1]) * x[col_indx[k + 1] - off];
      }
      for (; k < row_ptr[i + 2] - off; j++, k++) {
        sum0 += conj(vals[j]) * x[col_indx[j] - off];
        sum2 += conj(vals[k]) * x[col_indx[k] - off];
      }
      for (; j < row_ptr[i + 1] - off - 1; j += 2) {
        sum0 += conj(vals[j]) * x[col_indx[j] - off];
        sum1 += conj(vals[j + 1]) * x[col_indx[j + 1] - off];
      }
      for (; j < row_ptr[i + 1] - off; j++) {
        sum0 += conj(vals[j]) * x[col_indx[j] - off];
      }
    }
    sum0 += sum1;
    sum2 += sum3;
    sum0 *= alpha;
    sum2 *= alpha;
    if (beta != (T)0) {
      sum0 += beta * y[i];
      sum2 += beta * y[i + 1];
    }
    y[i] = sum0;
    y[i + 1] = sum2;
  }

  if (csr.m % 2 != 0) {
    T sum0 = 0;
    T sum1 = 0;
    perflibs_int_t i = csr.m - 1;
    perflibs_int_t j;

    for (j = row_ptr[i] - off; j < row_ptr[i + 1] - off - 1; j += 2) {
      sum0 += conj(vals[j]) * x[col_indx[j] - off];
      sum1 += conj(vals[j + 1]) * x[col_indx[j + 1] - off];
    }
    for (; j < row_ptr[i + 1] - off; j++) {
      sum0 += conj(vals[j]) * x[col_indx[j] - off];
    }
    sum0 += sum1;
    sum0 *= alpha;
    if (beta != (T)0) {
      sum0 += beta * y[i];
    }
    y[i] = sum0;
  }
}

/* Calculates the update of a single row (or part thereof) of an SpMV operation
 * for a CSR matrix. Iteration is performed from @p start_ind to @p end_ind, and
 * it is assumed that these lie within the same row, excepting that an entire
 * row is updated, in which case end_ind will point to the start of the next
 * row. We cannot scale the input vector y in here by value beta in the axpy
 * operation, as this update may be partial and we will then get multiple
 * additions of the input vector y.
 * @param [in] col_indx    The column indices of the entries in the CSR matrix
 * @param [in] off         The offset of the index (i.e. zero or one) in the
 * columns array
 * @param [in] vals        The values in the CSR matrix
 * @param [in] x           The x vector in the axpy operation
 * @param [in] alpha       Scalar alpha in the axpy operation
 * @param [in] start_ind   The index in the matrix from which to start iteration
 * @param [in] end_ind     The index in the matrix at which to end iteration
 * @param [in] row         Zero-based index of the row being operated on
 * @return The contribution to the SpMV operation for the given row
 */
template <typename T, decltype(&no_conj<T>) conj>
inline T spmv_csr_notrans_partial_row(const perflibs_int_t *col_indx,
                                      const perflibs_int_t off, const T *vals,
                                      const T *x, T alpha,
                                      perflibs_int_t start_ind,
                                      perflibs_int_t end_ind) {
  T sum0 = 0;
  T sum1 = 0;
  perflibs_int_t j;

  for (j = start_ind; j < end_ind - 1; j += 2) {
    sum0 += conj(vals[j]) * x[col_indx[j] - off];
    sum1 += conj(vals[j + 1]) * x[col_indx[j + 1] - off];
  }
  for (; j < end_ind; j++) {
    sum0 += conj(vals[j]) * x[col_indx[j] - off];
  }
  sum0 += sum1;
  sum0 *= alpha;
  return sum0;
}

// Perform SpMV using the parallel decomposition defined in the CSR class. This
// is populated using methods gen_parallel_decomp_mv and gen_data_vectors_mv
template <typename T, decltype(&no_conj<T>) conj>
static void spmv_csr_notrans_first_touch(const perflibs_csr<T> &csr, const T *x,
                                         T *y, T alpha, T beta) {

  const auto row_ptr = csr.row_ptr_ptr;
  const auto col_indx = csr.col_indx_ptr;
  const auto vals = csr.vals_ptr;
  const auto off = row_ptr[0];
  // Use a vector of synchronization variables to help to scale the input
  // vector by beta
  std::vector<std::atomic_bool> scaled_beta(csr.par_mv.nthreads);

  const auto thread_inds = csr.par_mv.thread_inds.data();
  const auto thread_rows = csr.par_mv.thread_rows.data();

// The parallel loop uses a synchronization parameter, which is set in a thread
// with a 'higher' index. To avoid a deadlock in the case that the number of
// threads assigned is not csr.par_mv.nthreads, loop through in reverse order,
// so that if a thread needs to wait on itself, it will find the synchronization
// variable set
#pragma omp parallel for default(none) schedule(static, 1) firstprivate(       \
        row_ptr, col_indx, vals, off, alpha, beta, thread_rows, thread_inds)   \
    shared(csr, x, y, scaled_beta) num_threads(csr.par_mv.nthreads)
  for (perflibs_int_t t = csr.par_mv.nthreads - 1; t >= 0; --t) {
    // Figure out what rows we are working on
    perflibs_int_t start_row = thread_rows[t];
    perflibs_int_t end_row = thread_rows[t + 1];
    bool single_row = start_row == end_row;
    bool start_of_row = thread_inds[t] == csr.row_ptr[start_row] - off;
    perflibs_int_t thread_to_sync = -1;
    // Helper function to find out what thread has scaled y by beta in the
    // case that a row is split across threads
    auto get_thread_to_sync =
        [&](perflibs_int_t sync_thread) -> perflibs_int_t {
      while (thread_rows[sync_thread + 1] == end_row)
        sync_thread++;
      return sync_thread;
    };
    if (single_row) {
      // Is there any work to do, or is there no work for the thread to perform
      if (thread_inds[t] == thread_inds[t + 1])
        continue;
      // If we are here, this is guaranteed to be a partial row update. We need
      // to figure out what thread we need to synchronize with for the scaling
      // of y.
      thread_to_sync = get_thread_to_sync(t + 1);

      auto sum = spmv_csr_notrans_partial_row<T, conj>(
          col_indx, off, vals, x, alpha, thread_inds[t], thread_inds[t + 1]);
      // Wait until the input vector has been scaled appropriately
      while (!scaled_beta[thread_to_sync].load(std::memory_order_acquire)) {
      }

      update_y(y[start_row], sum);

      // We've updated our one partial row, so move on to the next iteration
      continue;
    } else {
      if (!start_of_row) {
        // The first part of this thread is to update the end of a row. This
        // is the thread that is responsible for performing the scaling of y by
        // beta We don't need this operation to be atomic, as the order of the
        // store and load across threads is enforced by the memory fences and
        // the fact that the result of the atomic store must be observed
        if (beta != (T)0)
          y[start_row] *= beta;
        else
          y[start_row] = 0;
        // Don't allow reordering of memory operations before the store to the
        // synchronization variable
        scaled_beta[t].store(true, std::memory_order_release);
        auto sum = spmv_csr_notrans_partial_row<T, conj>(
            col_indx, off, vals, x, alpha, thread_inds[t],
            row_ptr[start_row + 1] - off);
        update_y(y[start_row], sum);
        start_row++;
      }

      if (thread_inds[t + 1] != csr.row_ptr[end_row] - off) {
        // Figure out what thread to sync with. Most often this is simply
        // the next thread
        thread_to_sync = get_thread_to_sync(t + 1);
      }
    }

    // Loop over full rows
    for (perflibs_int_t i = start_row; i < end_row - 1; i += 2) {
      T sum0 = 0;
      T sum1 = 0;
      T sum2 = 0;
      T sum3 = 0;
      perflibs_int_t j, k;

      auto row_len_0 = row_ptr[i + 1] - row_ptr[i];
      auto row_len_1 = row_ptr[i + 2] - row_ptr[i + 1];

      if (row_len_0 == row_len_1) {
        for (j = row_ptr[i] - off, k = row_ptr[i + 1] - off;
             j < row_ptr[i + 1] - off - 1; j += 2, k += 2) {
          sum0 += conj(vals[j]) * x[col_indx[j] - off];
          sum1 += conj(vals[j + 1]) * x[col_indx[j + 1] - off];
          sum2 += conj(vals[k]) * x[col_indx[k] - off];
          sum3 += conj(vals[k + 1]) * x[col_indx[k + 1] - off];
        }
        for (; j < row_ptr[i + 1] - off; j++, k++) {
          sum0 += conj(vals[j]) * x[col_indx[j] - off];
          sum2 += conj(vals[k]) * x[col_indx[k] - off];
        }
      } else if (row_len_0 < row_len_1) {
        for (j = row_ptr[i] - off, k = row_ptr[i + 1] - off;
             j < row_ptr[i + 1] - off - 1; j += 2, k += 2) {
          sum0 += conj(vals[j]) * x[col_indx[j] - off];
          sum1 += conj(vals[j + 1]) * x[col_indx[j + 1] - off];
          sum2 += conj(vals[k]) * x[col_indx[k] - off];
          sum3 += conj(vals[k + 1]) * x[col_indx[k + 1] - off];
        }
        for (; j < row_ptr[i + 1] - off; j++, k++) {
          sum0 += conj(vals[j]) * x[col_indx[j] - off];
          sum2 += conj(vals[k]) * x[col_indx[k] - off];
        }
        for (; k < row_ptr[i + 2] - off - 1; k += 2) {
          sum2 += conj(vals[k]) * x[col_indx[k] - off];
          sum3 += conj(vals[k + 1]) * x[col_indx[k + 1] - off];
        }
        for (; k < row_ptr[i + 2] - off; k++) {
          sum2 += conj(vals[k]) * x[col_indx[k] - off];
        }
      } else if (row_len_0 > row_len_1) {

        for (j = row_ptr[i] - off, k = row_ptr[i + 1] - off;
             k < row_ptr[i + 2] - off - 1; j += 2, k += 2) {
          sum0 += conj(vals[j]) * x[col_indx[j] - off];
          sum1 += conj(vals[j + 1]) * x[col_indx[j + 1] - off];
          sum2 += conj(vals[k]) * x[col_indx[k] - off];
          sum3 += conj(vals[k + 1]) * x[col_indx[k + 1] - off];
        }
        for (; k < row_ptr[i + 2] - off; j++, k++) {
          sum0 += conj(vals[j]) * x[col_indx[j] - off];
          sum2 += conj(vals[k]) * x[col_indx[k] - off];
        }
        for (; j < row_ptr[i + 1] - off - 1; j += 2) {
          sum0 += conj(vals[j]) * x[col_indx[j] - off];
          sum1 += conj(vals[j + 1]) * x[col_indx[j + 1] - off];
        }
        for (; j < row_ptr[i + 1] - off; j++) {
          sum0 += conj(vals[j]) * x[col_indx[j] - off];
        }
      }
      sum0 += sum1;
      sum2 += sum3;
      sum0 *= alpha;
      sum2 *= alpha;
      if (beta != (T)0) {
        sum0 += beta * y[i];
        sum2 += beta * y[i + 1];
      }
      y[i] = sum0;
      y[i + 1] = sum2;
    }

    if ((end_row - start_row) % 2 != 0) {
      auto sum = spmv_csr_notrans_partial_row<T, conj>(
          col_indx, off, vals, x, alpha, row_ptr[end_row - 1] - off,
          row_ptr[end_row] - off);
      if (beta != (T)0)
        sum += beta * y[end_row - 1];
      y[end_row - 1] = sum;
    }

    if (thread_inds[t + 1] != csr.row_ptr[end_row] - off) {
      // Last row is a partial row.
      auto sum = spmv_csr_notrans_partial_row<T, conj>(
          col_indx, off, vals, x, alpha, row_ptr[end_row] - off,
          thread_inds[t + 1]);
      // Synchronize with the thread that has scaled y before updating the value
      while (!scaled_beta[thread_to_sync].load(std::memory_order_acquire)) {
      }

      update_y(y[end_row], sum);
    }
  }
}

template <typename T, decltype(&no_conj<T>) conj>
void spmv_csr_notrans(const perflibs_csr<T> &csr, const T *x, T *y, T alpha,
                      T beta) {
  if (csr.par_mv.thread_inds.empty()) {
    spmv_csr_notrans_no_parallel_strat<T, conj>(csr, x, y, alpha, beta);
  } else {
    spmv_csr_notrans_first_touch<T, conj>(csr, x, y, alpha, beta);
  }
}

template <typename T, decltype(&no_conj<T>) conj>
void spmv_csr_trans(const perflibs_csr<T> &csr, const T *x, T *y, T alpha) {

  const auto row_ptr = csr.row_ptr_ptr;
  const auto col_indx = csr.col_indx_ptr;
  const auto vals = csr.vals_ptr;
  const auto off = row_ptr[0];

#pragma omp parallel for default(none) schedule(static)                        \
    firstprivate(row_ptr, col_indx, vals, off) shared(csr, x, y, alpha)        \
    num_threads(csr.par_mv.nthreads)
  for (perflibs_int_t i = 0; i < csr.m - 1; i += 2) {

    auto row_len_0 = row_ptr[i + 1] - row_ptr[i];
    auto row_len_1 = row_ptr[i + 2] - row_ptr[i + 1];

    if (row_len_0 == row_len_1) {
      perflibs_int_t j, k;
      for (j = row_ptr[i] - off, k = row_ptr[i + 1] - off;
           j < row_ptr[i + 1] - off - 1; j += 2, k += 2) {
        update_y(y[col_indx[j] - off], alpha * conj(vals[j]) * x[i]);
        update_y(y[col_indx[j + 1] - off], alpha * conj(vals[j + 1]) * x[i]);
        update_y(y[col_indx[k] - off], alpha * conj(vals[k]) * x[i + 1]);
        update_y(y[col_indx[k + 1] - off],
                 alpha * conj(vals[k + 1]) * x[i + 1]);
      }
      for (; j < row_ptr[i + 1] - off; j++, k++) {
        update_y(y[col_indx[j] - off], alpha * conj(vals[j]) * x[i]);
        update_y(y[col_indx[k] - off], alpha * conj(vals[k]) * x[i + 1]);
      }
    } else if (row_len_0 < row_len_1) {
      perflibs_int_t j, k;
      for (j = row_ptr[i] - off, k = row_ptr[i + 1] - off;
           j < row_ptr[i + 1] - off - 1; j += 2, k += 2) {
        update_y(y[col_indx[j] - off], alpha * conj(vals[j]) * x[i]);
        update_y(y[col_indx[j + 1] - off], alpha * conj(vals[j + 1]) * x[i]);
        update_y(y[col_indx[k] - off], alpha * conj(vals[k]) * x[i + 1]);
        update_y(y[col_indx[k + 1] - off],
                 alpha * conj(vals[k + 1]) * x[i + 1]);
      }
      for (; j < row_ptr[i + 1] - off; j++, k++) {
        update_y(y[col_indx[j] - off], alpha * conj(vals[j]) * x[i]);
        update_y(y[col_indx[k] - off], alpha * conj(vals[k]) * x[i + 1]);
      }
      for (; k < row_ptr[i + 2] - off - 1; k += 2) {
        update_y(y[col_indx[k] - off], alpha * conj(vals[k]) * x[i + 1]);
        update_y(y[col_indx[k + 1] - off],
                 alpha * conj(vals[k + 1]) * x[i + 1]);
      }
      for (; k < row_ptr[i + 2] - off; k++) {
        update_y(y[col_indx[k] - off], alpha * conj(vals[k]) * x[i + 1]);
      }
    } else if (row_len_0 > row_len_1) {
      perflibs_int_t j, k;
      for (j = row_ptr[i] - off, k = row_ptr[i + 1] - off;
           k < row_ptr[i + 2] - off - 1; j += 2, k += 2) {
        update_y(y[col_indx[j] - off], alpha * conj(vals[j]) * x[i]);
        update_y(y[col_indx[j + 1] - off], alpha * conj(vals[j + 1]) * x[i]);
        update_y(y[col_indx[k] - off], alpha * conj(vals[k]) * x[i + 1]);
        update_y(y[col_indx[k + 1] - off],
                 alpha * conj(vals[k + 1]) * x[i + 1]);
      }
      for (; k < row_ptr[i + 2] - off; j++, k++) {
        update_y(y[col_indx[j] - off], alpha * conj(vals[j]) * x[i]);
        update_y(y[col_indx[k] - off], alpha * conj(vals[k]) * x[i + 1]);
      }
      for (; j < row_ptr[i + 1] - off - 1; j += 2) {
        update_y(y[col_indx[j] - off], alpha * conj(vals[j]) * x[i]);
        update_y(y[col_indx[j + 1] - off], alpha * conj(vals[j + 1]) * x[i]);
      }
      for (; j < row_ptr[i + 1] - off; j++) {
        update_y(y[col_indx[j] - off], alpha * conj(vals[j]) * x[i]);
      }
    }
  }

  if (csr.m % 2 != 0) {
    perflibs_int_t i = csr.m - 1;
    perflibs_int_t j;
    for (j = row_ptr[i] - off; j < row_ptr[i + 1] - off - 1; j += 2) {
      y[col_indx[j] - off] += alpha * conj(vals[j]) * x[i];
      y[col_indx[j + 1] - off] += alpha * conj(vals[j + 1]) * x[i];
    }
    for (; j < row_ptr[i + 1] - off; j++) {
      y[col_indx[j] - off] += alpha * conj(vals[j]) * x[i];
    }
  }
}

template <typename T>
void spmv_csr(const perflibs_csr<T> &csr, sparse_hint_value_internal trans,
              const T *x, T *y, T alpha, T beta) {

  if (trans == PERFLIBS_OPERATION_NOTRANS) {
    if (csr.use_vanilla) {
      spmv_csr_vanilla<T>(csr, x, y, alpha, beta);
    } else {
      spmv_csr_notrans<T, no_conj<T>>(csr, x, y, alpha, beta);
    }
  } else if (trans ==
             PERFLIBS_OPERATION_CONJNOTRANS) { // used for CSC in CONJTRANS case
    spmv_csr_notrans<T, conj<T>>(csr, x, y, alpha, beta);
  } else {
    // Transpose case, set up y according to beta
    if (beta == (T)0) {
      for (perflibs_int_t i = 0; i < csr.n; i++) {
        y[i] = 0;
      }
    } else {
      for (perflibs_int_t i = 0; i < csr.n; i++) {
        y[i] *= beta;
      }
    }

    if (trans == PERFLIBS_OPERATION_TRANS) {
      spmv_csr_trans<T, no_conj<T>>(csr, x, y, alpha);
    } else if (trans == PERFLIBS_OPERATION_CONJTRANS) {
      spmv_csr_trans<T, conj<T>>(csr, x, y, alpha);
    }
  }
}

template void spmv_csr<float>(const perflibs_csr<float> &csr,
                              sparse_hint_value_internal trans, const float *x,
                              float *y, float alpha, float beta);
template void spmv_csr<double>(const perflibs_csr<double> &csr,
                               sparse_hint_value_internal trans,
                               const double *x, double *y, double alpha,
                               double beta);
template void
spmv_csr<std::complex<float>>(const perflibs_csr<std::complex<float>> &csr,
                              sparse_hint_value_internal trans,
                              const std::complex<float> *x,
                              std::complex<float> *y, std::complex<float> alpha,
                              std::complex<float> beta);
template void spmv_csr<std::complex<double>>(
    const perflibs_csr<std::complex<double>> &csr,
    sparse_hint_value_internal trans, const std::complex<double> *x,
    std::complex<double> *y, std::complex<double> alpha,
    std::complex<double> beta);

template <typename T, decltype(&no_conj<T>) conjA, decltype(&no_conj<T>) conjB>
void spmm_rowwise_csr_blocked_m_kernel(const perflibs_csr<T> &csr, const T *B,
                                       perflibs_int_t ldb, T *C,
                                       perflibs_int_t ldc, perflibs_int_t n,
                                       T alpha, T beta) {
  const auto m = csr.m;
  const auto row_ptr = csr.row_ptr_ptr;
  const auto col_indx = csr.col_indx_ptr;
  const auto vals = csr.vals_ptr;
  const auto off = row_ptr[0];

  // Row block size - this helps the compiler with vectorization even if
  // we're not doing an OpenMP build (where the same could be achieved with
  // schedule(static, 64) on a non-blocked loop. It gives information
  // about the working set, even if we iterate in the same order if
  // we remove the outer blocking loop.
  constexpr perflibs_int_t TM = 64;

  // zero/nan prop handling is done outside the kernel
  const bool use_beta = beta != (T)0 && beta != (T)1;

#pragma omp parallel for schedule(static)
  for (perflibs_int_t rb = 0; rb < m; rb += TM) {
    const perflibs_int_t rb_max = std::min<perflibs_int_t>(rb + TM, m);
    for (perflibs_int_t i = rb; i < rb_max; ++i) {
      T *C_row = &C[i * ldc];
      if (use_beta) {
#pragma omp simd
        for (perflibs_int_t j = 0; j < n; ++j) {
          C_row[j] *= beta;
        }
      }

      for (perflibs_int_t p = row_ptr[i] - off; p < row_ptr[i + 1] - off; ++p) {
        const perflibs_int_t col = col_indx[p] - off;
        T a_val = conjA(vals[p]);
        a_val *= alpha;
        const T *B_row = &B[col * ldb];
#pragma omp simd
        for (perflibs_int_t j = 0; j < n; ++j) {
          T b_val = conjB(B_row[j]);
          C_row[j] += a_val * b_val;
        }
      }
    }
  }
}

template <typename T, bool conjA, bool conjB>
void spmm_rowwise_csr_blocked_m(const perflibs_csr<T> &csr, const T *B,
                                perflibs_int_t ldb, T *C, perflibs_int_t ldc,
                                perflibs_int_t n, T alpha, T beta) {
  if constexpr (!conjA && !conjB) {
    spmm_rowwise_csr_blocked_m_kernel<T, no_conj, no_conj>(csr, B, ldb, C, ldc,
                                                           n, alpha, beta);
  } else if constexpr (conjA && conjB) {
    spmm_rowwise_csr_blocked_m_kernel<T, conj, conj>(csr, B, ldb, C, ldc, n,
                                                     alpha, beta);
  } else if constexpr (!conjA && conjB) {
    spmm_rowwise_csr_blocked_m_kernel<T, no_conj, conj>(csr, B, ldb, C, ldc, n,
                                                        alpha, beta);
  } else { // if constexpr (conjA && !conjB)
    spmm_rowwise_csr_blocked_m_kernel<T, conj, no_conj>(csr, B, ldb, C, ldc, n,
                                                        alpha, beta);
  }
}
template void spmm_rowwise_csr_blocked_m<float, false, false>(
    const perflibs_csr<float> &csr, const float *B, perflibs_int_t ldb,
    float *C, perflibs_int_t ldc, perflibs_int_t n, float alpha, float beta);
template void spmm_rowwise_csr_blocked_m<float, true, true>(
    const perflibs_csr<float> &csr, const float *B, perflibs_int_t ldb,
    float *C, perflibs_int_t ldc, perflibs_int_t n, float alpha, float beta);
template void spmm_rowwise_csr_blocked_m<float, false, true>(
    const perflibs_csr<float> &csr, const float *B, perflibs_int_t ldb,
    float *C, perflibs_int_t ldc, perflibs_int_t n, float alpha, float beta);
template void spmm_rowwise_csr_blocked_m<float, true, false>(
    const perflibs_csr<float> &csr, const float *B, perflibs_int_t ldb,
    float *C, perflibs_int_t ldc, perflibs_int_t n, float alpha, float beta);

template void spmm_rowwise_csr_blocked_m<double, false, false>(
    const perflibs_csr<double> &csr, const double *B, perflibs_int_t ldb,
    double *C, perflibs_int_t ldc, perflibs_int_t n, double alpha, double beta);
template void spmm_rowwise_csr_blocked_m<double, true, true>(
    const perflibs_csr<double> &csr, const double *B, perflibs_int_t ldb,
    double *C, perflibs_int_t ldc, perflibs_int_t n, double alpha, double beta);
template void spmm_rowwise_csr_blocked_m<double, false, true>(
    const perflibs_csr<double> &csr, const double *B, perflibs_int_t ldb,
    double *C, perflibs_int_t ldc, perflibs_int_t n, double alpha, double beta);
template void spmm_rowwise_csr_blocked_m<double, true, false>(
    const perflibs_csr<double> &csr, const double *B, perflibs_int_t ldb,
    double *C, perflibs_int_t ldc, perflibs_int_t n, double alpha, double beta);

template void spmm_rowwise_csr_blocked_m<std::complex<float>, false, false>(
    const perflibs_csr<std::complex<float>> &csr, const std::complex<float> *B,
    perflibs_int_t ldb, std::complex<float> *C, perflibs_int_t ldc,
    perflibs_int_t n, std::complex<float> alpha, std::complex<float> beta);
template void spmm_rowwise_csr_blocked_m<std::complex<float>, true, true>(
    const perflibs_csr<std::complex<float>> &csr, const std::complex<float> *B,
    perflibs_int_t ldb, std::complex<float> *C, perflibs_int_t ldc,
    perflibs_int_t n, std::complex<float> alpha, std::complex<float> beta);
template void spmm_rowwise_csr_blocked_m<std::complex<float>, false, true>(
    const perflibs_csr<std::complex<float>> &csr, const std::complex<float> *B,
    perflibs_int_t ldb, std::complex<float> *C, perflibs_int_t ldc,
    perflibs_int_t n, std::complex<float> alpha, std::complex<float> beta);
template void spmm_rowwise_csr_blocked_m<std::complex<float>, true, false>(
    const perflibs_csr<std::complex<float>> &csr, const std::complex<float> *B,
    perflibs_int_t ldb, std::complex<float> *C, perflibs_int_t ldc,
    perflibs_int_t n, std::complex<float> alpha, std::complex<float> beta);

template void spmm_rowwise_csr_blocked_m<std::complex<double>, false, false>(
    const perflibs_csr<std::complex<double>> &csr,
    const std::complex<double> *B, perflibs_int_t ldb, std::complex<double> *C,
    perflibs_int_t ldc, perflibs_int_t n, std::complex<double> alpha,
    std::complex<double> beta);
template void spmm_rowwise_csr_blocked_m<std::complex<double>, true, true>(
    const perflibs_csr<std::complex<double>> &csr,
    const std::complex<double> *B, perflibs_int_t ldb, std::complex<double> *C,
    perflibs_int_t ldc, perflibs_int_t n, std::complex<double> alpha,
    std::complex<double> beta);
template void spmm_rowwise_csr_blocked_m<std::complex<double>, false, true>(
    const perflibs_csr<std::complex<double>> &csr,
    const std::complex<double> *B, perflibs_int_t ldb, std::complex<double> *C,
    perflibs_int_t ldc, perflibs_int_t n, std::complex<double> alpha,
    std::complex<double> beta);
template void spmm_rowwise_csr_blocked_m<std::complex<double>, true, false>(
    const perflibs_csr<std::complex<double>> &csr,
    const std::complex<double> *B, perflibs_int_t ldb, std::complex<double> *C,
    perflibs_int_t ldc, perflibs_int_t n, std::complex<double> alpha,
    std::complex<double> beta);

template <typename T>
inline void spsv_dot(T &sum, const perflibs_int_t *col_indx, const T *vals,
                     const T *x, perflibs_int_t off,
                     perflibs_int_t row_start_indx,
                     perflibs_int_t row_end_indx) {
  for (perflibs_int_t j = row_start_indx; j <= row_end_indx; ++j) {
    const perflibs_int_t idx = col_indx[j] - off; // scalar load of index
    sum += x[idx] * vals[j];
  }
};

// Specialized ACLE version for optimization of fp64 use cases
template <>
inline void spsv_dot<double>(double &sum, const perflibs_int_t *col_indx,
                             const double *vals, const double *x,
                             perflibs_int_t off, perflibs_int_t row_start_indx,
                             perflibs_int_t row_end_indx) {
  float64x2_t vsum0 = vdupq_n_f64(0.0);
  float64x2_t vsum1 = vdupq_n_f64(0.0);

  perflibs_int_t j = row_start_indx;
  for (; j + 3 <= row_end_indx; j += 4) {
    const perflibs_int_t idx0 = col_indx[j] - off;
    const perflibs_int_t idx1 = col_indx[j + 1] - off;
    const perflibs_int_t idx2 = col_indx[j + 2] - off;
    const perflibs_int_t idx3 = col_indx[j + 3] - off;

    const float64x2_t v01 = vld1q_f64(&vals[j]);
    const float64x2_t v23 = vld1q_f64(&vals[j + 2]);

    float64x2_t x01v = vdupq_n_f64(0.0);
    x01v = vsetq_lane_f64(x[idx0], x01v, 0);
    x01v = vsetq_lane_f64(x[idx1], x01v, 1);

    float64x2_t x23v = vdupq_n_f64(0.0);
    x23v = vsetq_lane_f64(x[idx2], x23v, 0);
    x23v = vsetq_lane_f64(x[idx3], x23v, 1);

    vsum0 = vfmaq_f64(vsum0, v01, x01v);
    vsum1 = vfmaq_f64(vsum1, v23, x23v);
  }

  for (; j <= row_end_indx; ++j) {
    const perflibs_int_t idx = col_indx[j] - off;
    sum += x[idx] * vals[j];
  }

  const float64x2_t vtot = vaddq_f64(vsum0, vsum1);
  sum += vaddvq_f64(vtot);
};

// A wrapper so that we can use the ACLE version for SpDOT
void spsv_dot_fp64_helper(double &sum, const perflibs_int_t *col_indx,
                          const double *vals, const double *x,
                          perflibs_int_t off, perflibs_int_t row_start_indx,
                          perflibs_int_t row_end_indx) {
  return spsv_dot<double>(sum, col_indx, vals, x, off, row_start_indx,
                          row_end_indx);
}

// SpSV computation to update row i when notrans/conjnotrans is specified
template <bool IsConj, typename T>
inline __attribute__((always_inline)) void spsv_notrans_kernel(
    const bool unit, const perflibs_int_t diag_indx, const perflibs_int_t i,
    const perflibs_int_t off, const perflibs_int_t *col_indx, const T *vals,
    T *x, const T *y, const T alpha, perflibs_int_t const row_start_indx,
    const perflibs_int_t row_end_indx) {
  perflibs_int_t j = row_end_indx, jdiag = diag_indx;
  T sum = 0.0;
  if (diag_indx >= 0) {
    spsv_dot(sum, col_indx, vals, x, off, row_start_indx, row_end_indx);
  } else {
    jdiag = row_end_indx;
    for (; j >= row_start_indx; --j) {
      auto indx = col_indx[j] - off;
      if (indx != i) {
        sum += x[indx] * vals[j];
      } else {
        jdiag = j;
      }
    }
  }
  if constexpr (IsConj) {
    x[i] = perflibs::sparse::conj(alpha * y[i]) - sum;
  } else {
    x[i] = alpha * y[i] - sum;
  }
  if (!unit) {
    x[i] /= vals[jdiag];
  }
};

// SpSM computation to update row i when notrans/conjnotrans is specified
template <bool IsConj, typename T>
inline __attribute__((always_inline)) void spsm_notrans_kernel(
    const bool unit, const perflibs_int_t diag_indx, const perflibs_int_t i,
    const perflibs_int_t off, const perflibs_int_t *col_indx, const T *vals,
    T *X, const perflibs_int_t x_stride_row, const perflibs_int_t x_stride_col,
    const T *Y, const perflibs_int_t y_stride_row,
    const perflibs_int_t y_stride_col, const perflibs_int_t nrhs, const T alpha,
    const perflibs_int_t row_start_indx, const perflibs_int_t row_end_indx) {
  perflibs_int_t jdiag = diag_indx;
  T *x_row = X + i * x_stride_row;
  const T *y_row = Y + i * y_stride_row;

  for (perflibs_int_t r = 0; r < nrhs; ++r) {
    if constexpr (IsConj) {
      x_row[r * x_stride_col] =
          perflibs::sparse::conj(alpha * y_row[r * y_stride_col]);
    } else {
      x_row[r * x_stride_col] = alpha * y_row[r * y_stride_col];
    }
  }

  if (diag_indx >= 0) {
    for (perflibs_int_t j = row_start_indx; j <= row_end_indx; ++j) {
      const perflibs_int_t idx = col_indx[j] - off;
      const T a = vals[j];
      const T *x_dep = X + idx * x_stride_row;
      for (perflibs_int_t r = 0; r < nrhs; ++r) {
        x_row[r * x_stride_col] -= x_dep[r * x_stride_col] * a;
      }
    }
  } else {
    jdiag = row_end_indx;
    for (perflibs_int_t j = row_end_indx; j >= row_start_indx; --j) {
      const perflibs_int_t idx = col_indx[j] - off;
      if (idx != i) {
        const T a = vals[j];
        const T *x_dep = X + idx * x_stride_row;
        for (perflibs_int_t r = 0; r < nrhs; ++r) {
          x_row[r * x_stride_col] -= x_dep[r * x_stride_col] * a;
        }
      } else {
        jdiag = j;
      }
    }
  }

  if (!unit) {
    const T diagv = vals[jdiag];
    for (perflibs_int_t r = 0; r < nrhs; ++r) {
      x_row[r * x_stride_col] /= diagv;
    }
  }
};

// SpSM computation to update row i for small, fixed row-major RHS counts.
template <perflibs_int_t NRHS, bool IsConj, typename T>
inline __attribute__((always_inline)) void spsm_notrans_kernel_fixed_nrhs(
    const bool unit, const perflibs_int_t diag_indx, const perflibs_int_t i,
    const perflibs_int_t off, const perflibs_int_t *col_indx, const T *vals,
    T *X, const perflibs_int_t x_stride_row, const T *Y,
    const perflibs_int_t y_stride_row, const T alpha,
    const perflibs_int_t row_start_indx, const perflibs_int_t row_end_indx) {
  perflibs_int_t jdiag = diag_indx;
  T *x_row = X + i * x_stride_row;
  const T *y_row = Y + i * y_stride_row;
  T x[NRHS];

#pragma omp simd
  for (perflibs_int_t r = 0; r < NRHS; ++r) {
    if constexpr (IsConj) {
      x[r] = perflibs::sparse::conj(alpha * y_row[r]);
    } else {
      x[r] = alpha * y_row[r];
    }
  }

  if (diag_indx >= 0) {
    for (perflibs_int_t j = row_start_indx; j <= row_end_indx; ++j) {
      const perflibs_int_t idx = col_indx[j] - off;
      const T a = vals[j];
      const T *x_dep = X + idx * x_stride_row;
#pragma omp simd
      for (perflibs_int_t r = 0; r < NRHS; ++r) {
        x[r] -= x_dep[r] * a;
      }
    }
  } else {
    jdiag = row_end_indx;
    for (perflibs_int_t j = row_end_indx; j >= row_start_indx; --j) {
      const perflibs_int_t idx = col_indx[j] - off;
      if (idx != i) {
        const T a = vals[j];
        const T *x_dep = X + idx * x_stride_row;
#pragma omp simd
        for (perflibs_int_t r = 0; r < NRHS; ++r) {
          x[r] -= x_dep[r] * a;
        }
      } else {
        jdiag = j;
      }
    }
  }

  if (!unit) {
    const T diagv = vals[jdiag];
#pragma omp simd
    for (perflibs_int_t r = 0; r < NRHS; ++r) {
      x[r] /= diagv;
    }
  }

#pragma omp simd
  for (perflibs_int_t r = 0; r < NRHS; ++r) {
    x_row[r] = x[r];
  }
};

template <bool IsConj, typename T>
inline __attribute__((always_inline)) void spsm_notrans_kernel_dispatch(
    const bool unit, const perflibs_int_t diag_indx, const perflibs_int_t i,
    const perflibs_int_t off, const perflibs_int_t *col_indx, const T *vals,
    T *X, const perflibs_int_t x_stride_row, const perflibs_int_t x_stride_col,
    const T *Y, const perflibs_int_t y_stride_row,
    const perflibs_int_t y_stride_col, const perflibs_int_t nrhs, const T alpha,
    const perflibs_int_t row_start_indx, const perflibs_int_t row_end_indx) {
  if (x_stride_col == 1 && y_stride_col == 1) {
    switch (nrhs) {
    case 2:
      spsm_notrans_kernel_fixed_nrhs<2, IsConj, T>(
          unit, diag_indx, i, off, col_indx, vals, X, x_stride_row, Y,
          y_stride_row, alpha, row_start_indx, row_end_indx);
      return;
    case 3:
      spsm_notrans_kernel_fixed_nrhs<3, IsConj, T>(
          unit, diag_indx, i, off, col_indx, vals, X, x_stride_row, Y,
          y_stride_row, alpha, row_start_indx, row_end_indx);
      return;
    case 4:
      spsm_notrans_kernel_fixed_nrhs<4, IsConj, T>(
          unit, diag_indx, i, off, col_indx, vals, X, x_stride_row, Y,
          y_stride_row, alpha, row_start_indx, row_end_indx);
      return;
    case 5:
      spsm_notrans_kernel_fixed_nrhs<5, IsConj, T>(
          unit, diag_indx, i, off, col_indx, vals, X, x_stride_row, Y,
          y_stride_row, alpha, row_start_indx, row_end_indx);
      return;
    case 6:
      spsm_notrans_kernel_fixed_nrhs<6, IsConj, T>(
          unit, diag_indx, i, off, col_indx, vals, X, x_stride_row, Y,
          y_stride_row, alpha, row_start_indx, row_end_indx);
      return;
    case 7:
      spsm_notrans_kernel_fixed_nrhs<7, IsConj, T>(
          unit, diag_indx, i, off, col_indx, vals, X, x_stride_row, Y,
          y_stride_row, alpha, row_start_indx, row_end_indx);
      return;
    case 8:
      spsm_notrans_kernel_fixed_nrhs<8, IsConj, T>(
          unit, diag_indx, i, off, col_indx, vals, X, x_stride_row, Y,
          y_stride_row, alpha, row_start_indx, row_end_indx);
      return;
    }
  }

  spsm_notrans_kernel<IsConj, T>(
      unit, diag_indx, i, off, col_indx, vals, X, x_stride_row, x_stride_col, Y,
      y_stride_row, y_stride_col, nrhs, alpha, row_start_indx, row_end_indx);
}

template <bool IsConj, typename T>
void spsv_csr_parallel(sparse_hint_value_internal uplo,
                       sparse_hint_value_internal diag, perflibs_int_t nrows,
                       const perflibs_int_t *row_ptr,
                       const perflibs_int_t *col_indx, const T *vals,
                       const par_sv_t &par_sv, T *x, const T *y, T alpha) {

  const auto info = get_csr_triangular_solve_info(row_ptr, uplo, diag);

  [[maybe_unused]] const auto nthreads =
      std::min(par_sv.nthreads, perflibs::sparse::omp::get_max_threads());
#pragma omp parallel num_threads(nthreads)
  {
    const int tid = perflibs::sparse::omp::get_thread_num();

    for (perflibs_int_t level = 0; level < par_sv.n_levels;) {

      auto first = par_sv.level_ptr[level];
      auto last = par_sv.level_ptr[level + 1] - 1;

      // while there are multiple levels with only one degree of parallelism,
      // avoid separate barrier for each level: just have a single thread do the
      // work and have other threads wait at a barrier once
      if (last - first == 0) {

        while (tid == 0 && level < par_sv.n_levels &&
               par_sv.level_ptr[level + 1] - par_sv.level_ptr[level] == 1) {
          auto ii = par_sv.level_ptr[level];
          auto i = par_sv.par_rows[ii];
          if (info.upper) {
            for (perflibs_int_t k = i; k > i - par_sv.chain_len[ii]; k--) {
              const perflibs_int_t row_start_indx =
                  row_ptr[k] - info.off + info.diag_lo;
              const perflibs_int_t row_end_indx =
                  row_ptr[k + 1] - 1 - info.off - info.diag_hi;
              const perflibs_int_t diag_indx =
                  info.known_diag ? row_start_indx - 1 : -1;
              spsv_notrans_kernel<IsConj, T>(info.unit, diag_indx, k, info.off,
                                             col_indx, vals, x, y, alpha,
                                             row_start_indx, row_end_indx);
            }
          } else {
            for (perflibs_int_t k = i; k < i + par_sv.chain_len[ii]; k++) {
              const perflibs_int_t row_start_indx =
                  row_ptr[k] - info.off + info.diag_lo;
              const perflibs_int_t row_end_indx =
                  row_ptr[k + 1] - 1 - info.off - info.diag_hi;
              const perflibs_int_t diag_indx =
                  info.known_diag ? row_end_indx + 1 : -1;
              spsv_notrans_kernel<IsConj, T>(info.unit, diag_indx, k, info.off,
                                             col_indx, vals, x, y, alpha,
                                             row_start_indx, row_end_indx);
            }
          }

          level++;
        }

        while (tid > 0 && level < par_sv.n_levels &&
               par_sv.level_ptr[level + 1] - par_sv.level_ptr[level] == 1) {
          level++;
        }
#pragma omp barrier
      } else {

#pragma omp for
        for (perflibs_int_t ii = first; ii <= last; ii++) {
          auto i = par_sv.par_rows[ii];
          if (info.upper) {
            for (perflibs_int_t k = i; k > i - par_sv.chain_len[ii]; k--) {
              const perflibs_int_t row_start_indx =
                  row_ptr[k] - info.off + info.diag_lo;
              const perflibs_int_t row_end_indx =
                  row_ptr[k + 1] - 1 - info.off - info.diag_hi;
              const perflibs_int_t diag_indx =
                  info.known_diag ? row_start_indx - 1 : -1;
              spsv_notrans_kernel<IsConj, T>(info.unit, diag_indx, k, info.off,
                                             col_indx, vals, x, y, alpha,
                                             row_start_indx, row_end_indx);
            }
          } else {
            for (perflibs_int_t k = i; k < i + par_sv.chain_len[ii]; k++) {
              const perflibs_int_t row_start_indx =
                  row_ptr[k] - info.off + info.diag_lo;
              const perflibs_int_t row_end_indx =
                  row_ptr[k + 1] - 1 - info.off - info.diag_hi;
              const perflibs_int_t diag_indx =
                  info.known_diag ? row_end_indx + 1 : -1;
              spsv_notrans_kernel<IsConj, T>(info.unit, diag_indx, k, info.off,
                                             col_indx, vals, x, y, alpha,
                                             row_start_indx, row_end_indx);
            }
          }
        }
        level++;
      }

    } // levels
  } // parallel

  // Instead of solving (A^*)x = alpha*y, we solve A(x^*) = (alpha*y)^* and then
  // take x^**
  if constexpr (IsConj) {
    for (perflibs_int_t i = 0; i < nrows; ++i) {
      x[i] = perflibs::sparse::conj(x[i]);
    }
  }
};

template <bool IsConj, typename T>
void spsm_csr_parallel(sparse_hint_value_internal uplo,
                       sparse_hint_value_internal diag, perflibs_int_t nrows,
                       const perflibs_int_t *row_ptr,
                       const perflibs_int_t *col_indx, const T *vals,
                       const par_sv_t &par_sv, T *X,
                       perflibs_int_t x_stride_row, perflibs_int_t x_stride_col,
                       const T *Y, perflibs_int_t y_stride_row,
                       perflibs_int_t y_stride_col, perflibs_int_t nrhs,
                       T alpha) {

  const auto info = get_csr_triangular_solve_info(row_ptr, uplo, diag);
  [[maybe_unused]] const auto nthreads =
      std::min(par_sv.nthreads, perflibs::sparse::omp::get_max_threads());
#pragma omp parallel num_threads(nthreads)
  {
    const int tid = perflibs::sparse::omp::get_thread_num();

    for (perflibs_int_t level = 0; level < par_sv.n_levels;) {

      auto first = par_sv.level_ptr[level];
      auto last = par_sv.level_ptr[level + 1] - 1;

      // while there are multiple levels with only one degree of parallelism,
      // avoid separate barrier for each level: just have a single thread do the
      // work and have other threads wait at a barrier once
      if (last - first == 0) {

        while (tid == 0 && level < par_sv.n_levels &&
               par_sv.level_ptr[level + 1] - par_sv.level_ptr[level] == 1) {
          auto ii = par_sv.level_ptr[level];
          auto i = par_sv.par_rows[ii];
          if (info.upper) {
            for (perflibs_int_t k = i; k > i - par_sv.chain_len[ii]; k--) {
              const perflibs_int_t row_start_indx =
                  row_ptr[k] - info.off + info.diag_lo;
              const perflibs_int_t row_end_indx =
                  row_ptr[k + 1] - 1 - info.off - info.diag_hi;
              const perflibs_int_t diag_indx =
                  info.known_diag ? row_start_indx - 1 : -1;
              spsm_notrans_kernel_dispatch<IsConj, T>(
                  info.unit, diag_indx, k, info.off, col_indx, vals, X,
                  x_stride_row, x_stride_col, Y, y_stride_row, y_stride_col,
                  nrhs, alpha, row_start_indx, row_end_indx);
            }
          } else {
            for (perflibs_int_t k = i; k < i + par_sv.chain_len[ii]; k++) {
              const perflibs_int_t row_start_indx =
                  row_ptr[k] - info.off + info.diag_lo;
              const perflibs_int_t row_end_indx =
                  row_ptr[k + 1] - 1 - info.off - info.diag_hi;
              const perflibs_int_t diag_indx =
                  info.known_diag ? row_end_indx + 1 : -1;
              spsm_notrans_kernel_dispatch<IsConj, T>(
                  info.unit, diag_indx, k, info.off, col_indx, vals, X,
                  x_stride_row, x_stride_col, Y, y_stride_row, y_stride_col,
                  nrhs, alpha, row_start_indx, row_end_indx);
            }
          }

          level++;
        }

        while (tid > 0 && level < par_sv.n_levels &&
               par_sv.level_ptr[level + 1] - par_sv.level_ptr[level] == 1) {
          level++;
        }
#pragma omp barrier
      } else {

#pragma omp for
        for (perflibs_int_t ii = first; ii <= last; ii++) {
          auto i = par_sv.par_rows[ii];
          if (info.upper) {
            for (perflibs_int_t k = i; k > i - par_sv.chain_len[ii]; k--) {
              const perflibs_int_t row_start_indx =
                  row_ptr[k] - info.off + info.diag_lo;
              const perflibs_int_t row_end_indx =
                  row_ptr[k + 1] - 1 - info.off - info.diag_hi;
              const perflibs_int_t diag_indx =
                  info.known_diag ? row_start_indx - 1 : -1;
              spsm_notrans_kernel_dispatch<IsConj, T>(
                  info.unit, diag_indx, k, info.off, col_indx, vals, X,
                  x_stride_row, x_stride_col, Y, y_stride_row, y_stride_col,
                  nrhs, alpha, row_start_indx, row_end_indx);
            }
          } else {
            for (perflibs_int_t k = i; k < i + par_sv.chain_len[ii]; k++) {
              const perflibs_int_t row_start_indx =
                  row_ptr[k] - info.off + info.diag_lo;
              const perflibs_int_t row_end_indx =
                  row_ptr[k + 1] - 1 - info.off - info.diag_hi;
              const perflibs_int_t diag_indx =
                  info.known_diag ? row_end_indx + 1 : -1;
              spsm_notrans_kernel_dispatch<IsConj, T>(
                  info.unit, diag_indx, k, info.off, col_indx, vals, X,
                  x_stride_row, x_stride_col, Y, y_stride_row, y_stride_col,
                  nrhs, alpha, row_start_indx, row_end_indx);
            }
          }
        }
        level++;
      }

    } // levels
  } // parallel

  // Instead of solving (A^*)X = alpha*Y, we solve A(X^*) = (alpha*Y)^* and then
  // take X^**
  if constexpr (IsConj) {
    for (perflibs_int_t i = 0; i < nrows; ++i) {
      T *x_row = X + i * x_stride_row;
      for (perflibs_int_t r = 0; r < nrhs; ++r) {
        x_row[r * x_stride_col] =
            perflibs::sparse::conj(x_row[r * x_stride_col]);
      }
    }
  }
};

template <bool IsConj, typename T>
void spsv_csr_vanilla_notrans(const perflibs_csr<T> &csr,
                              sparse_hint_value_internal uplo,
                              sparse_hint_value_internal diag, T *x, const T *y,
                              T alpha) {

  const auto row_ptr = csr.row_ptr_ptr;
  const auto col_indx = csr.col_indx_ptr;
  const auto vals = csr.vals_ptr;
  const auto info = get_csr_triangular_solve_info(row_ptr, uplo, diag);
  const auto nrows = csr.m;

  if (info.upper) {
    // UPPER TRIANGULAR
    for (perflibs_int_t i = nrows - 1; i >= 0; --i) {
      const perflibs_int_t row_start_indx =
          row_ptr[i] - info.off + info.diag_lo;
      const perflibs_int_t row_end_indx =
          row_ptr[i + 1] - 1 - info.off - info.diag_hi;
      const perflibs_int_t diag_indx =
          info.known_diag ? row_start_indx - 1 : -1;
      spsv_notrans_kernel<IsConj, T>(info.unit, diag_indx, i, info.off,
                                     col_indx, vals, x, y, alpha,
                                     row_start_indx, row_end_indx);
    }
  } else {
    // LOWER TRIANGULAR
    for (perflibs_int_t i = 0; i < nrows; ++i) {
      const perflibs_int_t row_start_indx =
          row_ptr[i] - info.off + info.diag_lo;
      const perflibs_int_t row_end_indx =
          row_ptr[i + 1] - 1 - info.off - info.diag_hi;
      const perflibs_int_t diag_indx = info.known_diag ? row_end_indx + 1 : -1;
      spsv_notrans_kernel<IsConj, T>(info.unit, diag_indx, i, info.off,
                                     col_indx, vals, x, y, alpha,
                                     row_start_indx, row_end_indx);
    }
  }

  // Instead of solving (A^*)x = alpha*y, we solve A(x^*) = (alpha*y)^* and then
  // take x^**
  if constexpr (IsConj) {
    for (perflibs_int_t i = 0; i < nrows; ++i) {
      x[i] = perflibs::sparse::conj(x[i]);
    }
  }
};

template <bool IsConj, typename T>
void spsm_csr_vanilla_notrans(
    const perflibs_csr<T> &csr, sparse_hint_value_internal uplo,
    sparse_hint_value_internal diag, T *X, perflibs_int_t x_stride_row,
    perflibs_int_t x_stride_col, const T *Y, perflibs_int_t y_stride_row,
    perflibs_int_t y_stride_col, perflibs_int_t nrhs, T alpha) {

  const auto row_ptr = csr.row_ptr_ptr;
  const auto col_indx = csr.col_indx_ptr;
  const auto vals = csr.vals_ptr;
  const auto info = get_csr_triangular_solve_info(row_ptr, uplo, diag);
  const auto nrows = csr.m;
  if (info.upper) {
    // UPPER TRIANGULAR
    for (perflibs_int_t i = nrows - 1; i >= 0; --i) {
      const perflibs_int_t row_start_indx =
          row_ptr[i] - info.off + info.diag_lo;
      const perflibs_int_t row_end_indx =
          row_ptr[i + 1] - 1 - info.off - info.diag_hi;
      const perflibs_int_t diag_indx =
          info.known_diag ? row_start_indx - 1 : -1;
      spsm_notrans_kernel_dispatch<IsConj, T>(
          info.unit, diag_indx, i, info.off, col_indx, vals, X, x_stride_row,
          x_stride_col, Y, y_stride_row, y_stride_col, nrhs, alpha,
          row_start_indx, row_end_indx);
    }
  } else {
    // LOWER TRIANGULAR
    for (perflibs_int_t i = 0; i < nrows; ++i) {
      const perflibs_int_t row_start_indx =
          row_ptr[i] - info.off + info.diag_lo;
      const perflibs_int_t row_end_indx =
          row_ptr[i + 1] - 1 - info.off - info.diag_hi;
      const perflibs_int_t diag_indx = info.known_diag ? row_end_indx + 1 : -1;
      spsm_notrans_kernel_dispatch<IsConj, T>(
          info.unit, diag_indx, i, info.off, col_indx, vals, X, x_stride_row,
          x_stride_col, Y, y_stride_row, y_stride_col, nrhs, alpha,
          row_start_indx, row_end_indx);
    }
  }

  // Instead of solving (A^*)X = alpha*Y, we solve A(X^*) = (alpha*Y)^* and then
  // take X^**
  if constexpr (IsConj) {
    for (perflibs_int_t i = 0; i < nrows; ++i) {
      T *x_row = X + i * x_stride_row;
      for (perflibs_int_t r = 0; r < nrhs; ++r) {
        x_row[r * x_stride_col] =
            perflibs::sparse::conj(x_row[r * x_stride_col]);
      }
    }
  }
};

// SpSV computation to update row i when trans/conjtrans is specified
template <bool IsConj, typename T>
inline __attribute__((always_inline)) void spsv_trans_kernel(
    const bool unit, const perflibs_int_t diag_indx, const perflibs_int_t i,
    const perflibs_int_t off, const perflibs_int_t *col_indx, const T *vals,
    T *x, const T *y, const T alpha, T *sum,
    const perflibs_int_t row_start_indx, const perflibs_int_t row_end_indx) {
  if constexpr (IsConj) {
    x[i] = perflibs::sparse::conj(alpha * y[i]) - sum[i];
  } else {
    x[i] = alpha * y[i] - sum[i];
  }
  if (!unit) {
    if (diag_indx >= 0) {
      x[i] /= vals[diag_indx];
    } else {
      for (auto j = row_start_indx; j <= row_end_indx; ++j) {
        auto indx = col_indx[j] - off;
        if (indx == i) {
          x[i] /= vals[j];
          break;
        }
      }
    }
  }
  if (diag_indx >= 0) {
    for (auto j = row_start_indx; j <= row_end_indx; ++j) {
      auto indx = col_indx[j] - off;
      sum[indx] += x[i] * vals[j];
    }
  } else {
    for (auto j = row_start_indx; j <= row_end_indx; ++j) {
      auto indx = col_indx[j] - off;
      if (indx != i) {
        sum[indx] += x[i] * vals[j];
      }
    }
  }
};

template <bool IsConj, typename T>
void spsv_csr_vanilla_trans(const perflibs_csr<T> &csr,
                            sparse_hint_value_internal uplo,
                            sparse_hint_value_internal diag, T *x, const T *y,
                            T alpha) {

  const auto row_ptr = csr.row_ptr_ptr;
  const auto col_indx = csr.col_indx_ptr;
  const auto vals = csr.vals_ptr;
  const auto nrows = csr.m;
  const auto info = get_csr_triangular_solve_info(row_ptr, uplo, diag);

  std::vector<T> sum(nrows, T(0.0));

  if (info.upper) {
    for (perflibs_int_t i = 0; i < nrows; ++i) {
      auto row_start_indx = row_ptr[i] - info.off + info.diag_lo;
      auto row_end_indx = row_ptr[i + 1] - 1 - info.off - info.diag_hi;
      const perflibs_int_t diag_indx =
          info.known_diag ? row_start_indx - 1 : -1;
      spsv_trans_kernel<IsConj, T>(info.unit, diag_indx, i, info.off, col_indx,
                                   vals, x, y, alpha, &sum[0], row_start_indx,
                                   row_end_indx);
    }
  } else {
    for (perflibs_int_t i = nrows - 1; i >= 0; --i) {
      auto row_start_indx = row_ptr[i] - info.off + info.diag_lo;
      auto row_end_indx = row_ptr[i + 1] - 1 - info.off - info.diag_hi;
      const perflibs_int_t diag_indx = info.known_diag ? row_end_indx + 1 : -1;
      spsv_trans_kernel<IsConj, T>(info.unit, diag_indx, i, info.off, col_indx,
                                   vals, x, y, alpha, &sum[0], row_start_indx,
                                   row_end_indx);
    }
  }

  // Instead of solving (A^*)x = alpha*y, we solve A(x^*) = (alpha*y)^* and then
  // take x^**
  if constexpr (IsConj) {
    for (perflibs_int_t i = 0; i < nrows; ++i) {
      x[i] = perflibs::sparse::conj(x[i]);
    }
  }
};

template <typename T>
void spsv_csr(const perflibs_csr<T> &csr, sparse_hint_value_internal trans,
              sparse_hint_value_internal uplo, sparse_hint_value_internal diag,
              T *x, const T *y, T alpha) {

  // parallel, allocating code paths: transpose question has been handled by
  // converting to CSC at this point, conj is the only question left
  if (perflibs::sparse::omp::is_mp && csr.par_sv.nthreads > 1 &&
      !csr.par_sv.level_ptr.empty()) {
    if (trans == PERFLIBS_OPERATION_NOTRANS ||
        trans == PERFLIBS_OPERATION_TRANS) {
      spsv_csr_parallel<false>(uplo, diag, csr.m, csr.row_ptr_ptr,
                               csr.col_indx_ptr, csr.vals_ptr, csr.par_sv, x, y,
                               alpha);
    } else if (trans == PERFLIBS_OPERATION_CONJTRANS ||
               trans == PERFLIBS_OPERATION_CONJNOTRANS) {
      spsv_csr_parallel<true>(uplo, diag, csr.m, csr.row_ptr_ptr,
                              csr.col_indx_ptr, csr.vals_ptr, csr.par_sv, x, y,
                              alpha);
    }
  } else { // serial, no allocating code paths
    if (trans == PERFLIBS_OPERATION_NOTRANS ||
        (!perflibs::sparse::is_complex_v<T> &&
         trans == PERFLIBS_OPERATION_CONJNOTRANS)) {
      spsv_csr_vanilla_notrans<false>(csr, uplo, diag, x, y, alpha);
    } else if (perflibs::sparse::is_complex_v<T> &&
               trans == PERFLIBS_OPERATION_CONJNOTRANS) {
      spsv_csr_vanilla_notrans<true>(csr, uplo, diag, x, y, alpha);
    } else if (trans == PERFLIBS_OPERATION_TRANS ||
               (!perflibs::sparse::is_complex_v<T> &&
                trans == PERFLIBS_OPERATION_CONJTRANS)) {
      spsv_csr_vanilla_trans<false>(csr, uplo, diag, x, y, alpha);
    } else if (perflibs::sparse::is_complex_v<T> &&
               trans == PERFLIBS_OPERATION_CONJTRANS) {
      spsv_csr_vanilla_trans<true>(csr, uplo, diag, x, y, alpha);
    }
  }
};

template <typename T>
void spsm_csr(const perflibs_csr<T> &csr, sparse_hint_value_internal trans,
              sparse_hint_value_internal uplo, sparse_hint_value_internal diag,
              T *X, perflibs_int_t x_stride_row, perflibs_int_t x_stride_col,
              const T *Y, perflibs_int_t y_stride_row,
              perflibs_int_t y_stride_col, perflibs_int_t nrhs, T alpha) {
  const bool use_parallel_notrans = perflibs::sparse::omp::is_mp &&
                                    csr.par_sv.nthreads > 1 &&
                                    !csr.par_sv.level_ptr.empty() &&
                                    (trans == PERFLIBS_OPERATION_NOTRANS ||
                                     trans == PERFLIBS_OPERATION_CONJNOTRANS);
  if (use_parallel_notrans) {
    if (trans == PERFLIBS_OPERATION_NOTRANS ||
        (!perflibs::sparse::is_complex_v<T> &&
         trans == PERFLIBS_OPERATION_CONJNOTRANS)) {
      spsm_csr_parallel<false>(uplo, diag, csr.m, csr.row_ptr_ptr,
                               csr.col_indx_ptr, csr.vals_ptr, csr.par_sv, X,
                               x_stride_row, x_stride_col, Y, y_stride_row,
                               y_stride_col, nrhs, alpha);
    } else if (perflibs::sparse::is_complex_v<T> &&
               trans == PERFLIBS_OPERATION_CONJNOTRANS) {
      spsm_csr_parallel<true>(uplo, diag, csr.m, csr.row_ptr_ptr,
                              csr.col_indx_ptr, csr.vals_ptr, csr.par_sv, X,
                              x_stride_row, x_stride_col, Y, y_stride_row,
                              y_stride_col, nrhs, alpha);
    }
  } else {
    if (trans == PERFLIBS_OPERATION_NOTRANS ||
        (!perflibs::sparse::is_complex_v<T> &&
         trans == PERFLIBS_OPERATION_CONJNOTRANS)) {
      spsm_csr_vanilla_notrans<false>(csr, uplo, diag, X, x_stride_row,
                                      x_stride_col, Y, y_stride_row,
                                      y_stride_col, nrhs, alpha);
    } else if (perflibs::sparse::is_complex_v<T> &&
               trans == PERFLIBS_OPERATION_CONJNOTRANS) {
      spsm_csr_vanilla_notrans<true>(csr, uplo, diag, X, x_stride_row,
                                     x_stride_col, Y, y_stride_row,
                                     y_stride_col, nrhs, alpha);
    }
  }
};
template void spsm_csr<float>(
    const perflibs_csr<float> &csr, sparse_hint_value_internal trans,
    sparse_hint_value_internal uplo, sparse_hint_value_internal diag, float *X,
    perflibs_int_t x_stride_row, perflibs_int_t x_stride_col, const float *Y,
    perflibs_int_t y_stride_row, perflibs_int_t y_stride_col,
    perflibs_int_t nrhs, float alpha);
template void spsm_csr<double>(
    const perflibs_csr<double> &csr, sparse_hint_value_internal trans,
    sparse_hint_value_internal uplo, sparse_hint_value_internal diag, double *X,
    perflibs_int_t x_stride_row, perflibs_int_t x_stride_col, const double *Y,
    perflibs_int_t y_stride_row, perflibs_int_t y_stride_col,
    perflibs_int_t nrhs, double alpha);
template void spsm_csr<std::complex<float>>(
    const perflibs_csr<std::complex<float>> &csr,
    sparse_hint_value_internal trans, sparse_hint_value_internal uplo,
    sparse_hint_value_internal diag, std::complex<float> *X,
    perflibs_int_t x_stride_row, perflibs_int_t x_stride_col,
    const std::complex<float> *Y, perflibs_int_t y_stride_row,
    perflibs_int_t y_stride_col, perflibs_int_t nrhs,
    std::complex<float> alpha);
template void spsm_csr<std::complex<double>>(
    const perflibs_csr<std::complex<double>> &csr,
    sparse_hint_value_internal trans, sparse_hint_value_internal uplo,
    sparse_hint_value_internal diag, std::complex<double> *X,
    perflibs_int_t x_stride_row, perflibs_int_t x_stride_col,
    const std::complex<double> *Y, perflibs_int_t y_stride_row,
    perflibs_int_t y_stride_col, perflibs_int_t nrhs,
    std::complex<double> alpha);

template void spsv_csr<float>(const perflibs_csr<float> &csr,
                              sparse_hint_value_internal trans,
                              sparse_hint_value_internal uplo,
                              sparse_hint_value_internal diag, float *x,
                              const float *y, float alpha);
template void spsv_csr<double>(const perflibs_csr<double> &csr,
                               sparse_hint_value_internal trans,
                               sparse_hint_value_internal uplo,
                               sparse_hint_value_internal diag, double *x,
                               const double *y, double alpha);
template void spsv_csr<std::complex<float>>(
    const perflibs_csr<std::complex<float>> &csr,
    sparse_hint_value_internal trans, sparse_hint_value_internal uplo,
    sparse_hint_value_internal diag, std::complex<float> *x,
    const std::complex<float> *y, std::complex<float> alpha);
template void spsv_csr<std::complex<double>>(
    const perflibs_csr<std::complex<double>> &csr,
    sparse_hint_value_internal trans, sparse_hint_value_internal uplo,
    sparse_hint_value_internal diag, std::complex<double> *x,
    const std::complex<double> *y, std::complex<double> alpha);

template <typename T>
void spelmm_csr_kernel(perflibs_sparse_hint_value transA,
                       const perflibs_int_t *row_ptrA,
                       const perflibs_int_t *col_indxA, const T *valsA,
                       perflibs_sparse_hint_value transB,
                       const perflibs_int_t *row_ptrB,
                       const perflibs_int_t *col_indxB, const T *valsB,
                       perflibs_int_t m, perflibs_int_t index_base,
                       perflibs::sparse::pod_vector<perflibs_int_t> &row_ptrAB,
                       std::vector<perflibs_int_t> &col_indxAB,
                       std::vector<T> &valsAB) {
  // Get conjugate of val if required
  auto conj = [&](T val, perflibs_sparse_hint_value trans) {
    return trans == PERFLIBS_SPARSE_OPERATION_CONJTRANS
               ? perflibs::sparse::conj(val)
               : val;
  };

  // Compute element-wise multiplication of A and B
  for (auto i = 0; i < m; i++) {
    // Counter for nonzeros in the current row
    perflibs_int_t nnz_in_row = 0;
    // Iterate over nonzeros in row i of A
    for (auto j = row_ptrA[i] - index_base; j < row_ptrA[i + 1] - index_base;
         j++) {
      auto A_col_indx = col_indxA[j];
      // Iterate over nonzeros in row i of B
      for (auto k = row_ptrB[i] - index_base; k < row_ptrB[i + 1] - index_base;
           k++) {
        auto B_col_indx = col_indxB[k];
        // If A and B share the same column index, perform element-wise
        // multiplication
        if (A_col_indx == B_col_indx) {
          // Update column indices and vals of matrix AB
          col_indxAB.push_back(A_col_indx);
          valsAB.push_back(conj(valsA[j], transA) * conj(valsB[k], transB));
          nnz_in_row++;
        }
      }
    }
    // Update row pointer for AB
    row_ptrAB[i + 1] = row_ptrAB[i] + nnz_in_row;
  }
}
template void spelmm_csr_kernel<float>(
    perflibs_sparse_hint_value transA, const perflibs_int_t *row_ptrA,
    const perflibs_int_t *col_indxA, const float *valsA,
    perflibs_sparse_hint_value transB, const perflibs_int_t *row_ptrB,
    const perflibs_int_t *col_indxB, const float *valsB, perflibs_int_t m,
    perflibs_int_t index_base,
    perflibs::sparse::pod_vector<perflibs_int_t> &row_ptrAB,
    std::vector<perflibs_int_t> &col_indxAB, std::vector<float> &valsAB);

template void spelmm_csr_kernel<double>(
    perflibs_sparse_hint_value transA, const perflibs_int_t *row_ptrA,
    const perflibs_int_t *col_indxA, const double *valsA,
    perflibs_sparse_hint_value transB, const perflibs_int_t *row_ptrB,
    const perflibs_int_t *col_indxB, const double *valsB, perflibs_int_t m,
    perflibs_int_t index_base,
    perflibs::sparse::pod_vector<perflibs_int_t> &row_ptrAB,
    std::vector<perflibs_int_t> &col_indxAB, std::vector<double> &valsAB);

template void spelmm_csr_kernel<std::complex<float>>(
    perflibs_sparse_hint_value transA, const perflibs_int_t *row_ptrA,
    const perflibs_int_t *col_indxA, const std::complex<float> *valsA,
    perflibs_sparse_hint_value transB, const perflibs_int_t *row_ptrB,
    const perflibs_int_t *col_indxB, const std::complex<float> *valsB,
    perflibs_int_t m, perflibs_int_t index_base,
    perflibs::sparse::pod_vector<perflibs_int_t> &row_ptrAB,
    std::vector<perflibs_int_t> &col_indxAB,
    std::vector<std::complex<float>> &valsAB);

template void spelmm_csr_kernel<std::complex<double>>(
    perflibs_sparse_hint_value transA, const perflibs_int_t *row_ptrA,
    const perflibs_int_t *col_indxA, const std::complex<double> *valsA,
    perflibs_sparse_hint_value transB, const perflibs_int_t *row_ptrB,
    const perflibs_int_t *col_indxB, const std::complex<double> *valsB,
    perflibs_int_t m, perflibs_int_t index_base,
    perflibs::sparse::pod_vector<perflibs_int_t> &row_ptrAB,
    std::vector<perflibs_int_t> &col_indxAB,
    std::vector<std::complex<double>> &valsAB);

template <typename T>
perflibs_status_t
spelmm_csr(perflibs_sparse_hint_value transA, perflibs_spmat_impl_t<T> *impl_A,
           perflibs_sparse_hint_value transB, perflibs_spmat_impl_t<T> *impl_B,
           perflibs_spmat_t AB) {
  bool is_transA = transA != PERFLIBS_SPARSE_OPERATION_NOTRANS;
  bool is_transB = transB != PERFLIBS_SPARSE_OPERATION_NOTRANS;

  // If A needs to be transposed, convert it to CSC format
  if (is_transA) {
    auto ret = convert<T>(perflibs_format_csc, impl_A);
    if (ret != PERFLIBS_STATUS_SUCCESS) {
      return ret;
    }
  }

  // If B needs to be transposed, convert it to CSR format
  if (is_transB) {
    auto ret = convert<T>(perflibs_format_csc, impl_B);
    if (ret != PERFLIBS_STATUS_SUCCESS) {
      return ret;
    }
  }

  // A, B and C have the same dimensions
  auto m = is_transA ? impl_A->n : impl_A->m;
  auto n = is_transA ? impl_A->m : impl_A->n;

  // Get CSR arrays for matrix A
  const auto row_ptrA =
      is_transA ? impl_A->csc.col_ptr_ptr : impl_A->csr.row_ptr_ptr;
  const auto col_indxA =
      is_transA ? impl_A->csc.row_indx_ptr : impl_A->csr.col_indx_ptr;
  const auto valsA = is_transA ? impl_A->csc.vals_ptr : impl_A->csr.vals_ptr;

  // Get CSR arrays for matrix B
  const auto row_ptrB =
      is_transB ? impl_B->csc.col_ptr_ptr : impl_B->csr.row_ptr_ptr;
  const auto col_indxB =
      is_transB ? impl_B->csc.row_indx_ptr : impl_B->csr.col_indx_ptr;
  const auto valsB = is_transB ? impl_B->csc.vals_ptr : impl_B->csr.vals_ptr;

  // Set up CSR vectors for result matrix AB
  const auto index_base = row_ptrA[0];

  perflibs::sparse::pod_vector<perflibs_int_t> row_ptrAB(m + 1);
  row_ptrAB[0] = index_base;

  auto min_nnz = std::min(row_ptrA[m], row_ptrB[m]);
  std::vector<perflibs_int_t> col_indxAB;
  col_indxAB.reserve(min_nnz);

  std::vector<T> valsAB;
  valsAB.reserve(min_nnz);

  // Do element-wise multiplication of A and B
  spelmm_csr_kernel(transA, row_ptrA, col_indxA, valsA, transB, row_ptrB,
                    col_indxB, valsB, m, index_base, row_ptrAB, col_indxAB,
                    valsAB);

  // Create CSR matrix from the result AB
  auto ret = fill_initial_data_csr(AB, m, n, row_ptrAB.data(),
                                   col_indxAB.data(), valsAB.data(), 0);
  if (ret != PERFLIBS_STATUS_SUCCESS) {
    return ret;
  }

  return PERFLIBS_STATUS_SUCCESS;
}
template perflibs_status_t
spelmm_csr<float>(perflibs_sparse_hint_value transA,
                  perflibs_spmat_impl_t<float> *impl_A,
                  perflibs_sparse_hint_value transB,
                  perflibs_spmat_impl_t<float> *impl_B, perflibs_spmat_t AB);

template perflibs_status_t
spelmm_csr<double>(perflibs_sparse_hint_value transA,
                   perflibs_spmat_impl_t<double> *impl_A,
                   perflibs_sparse_hint_value transB,
                   perflibs_spmat_impl_t<double> *impl_B, perflibs_spmat_t AB);

template perflibs_status_t spelmm_csr<std::complex<float>>(
    perflibs_sparse_hint_value transA,
    perflibs_spmat_impl_t<std::complex<float>> *impl_A,
    perflibs_sparse_hint_value transB,
    perflibs_spmat_impl_t<std::complex<float>> *impl_B, perflibs_spmat_t AB);

template perflibs_status_t spelmm_csr<std::complex<double>>(
    perflibs_sparse_hint_value transA,
    perflibs_spmat_impl_t<std::complex<double>> *impl_A,
    perflibs_sparse_hint_value transB,
    perflibs_spmat_impl_t<std::complex<double>> *impl_B, perflibs_spmat_t AB);

template <typename T>
perflibs_status_t
sddmm_csr(perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
          T alpha, perflibs_spmat_impl_t<T> *impl_A,
          perflibs_spmat_impl_t<T> *impl_B, T beta,
          perflibs_spmat_impl_t<T> *impl_C, perflibs_spmat_t AB) {
  // Get matrix dimensions
  auto m = impl_C->m;
  auto n = impl_C->n;
  auto k = transA == PERFLIBS_SPARSE_OPERATION_NOTRANS ? impl_A->n : impl_A->m;

  // Determine memory layout for indexing
  bool use_col_maj_indxA = true;
  bool use_col_maj_indxB = true;
  if ((transA == PERFLIBS_SPARSE_OPERATION_NOTRANS &&
       impl_A->dense.layout == PERFLIBS_ROW_MAJOR) ||
      (transA != PERFLIBS_SPARSE_OPERATION_NOTRANS &&
       impl_A->dense.layout == PERFLIBS_COL_MAJOR)) {
    use_col_maj_indxA = false;
  }
  if ((transB == PERFLIBS_SPARSE_OPERATION_NOTRANS &&
       impl_B->dense.layout == PERFLIBS_ROW_MAJOR) ||
      (transB != PERFLIBS_SPARSE_OPERATION_NOTRANS &&
       impl_B->dense.layout == PERFLIBS_COL_MAJOR)) {
    use_col_maj_indxB = false;
  }

  // Get leading dimensions of A and B
  auto ldA = impl_A->dense.lda;
  auto ldB = impl_B->dense.lda;

  // Get CSR properties of C
  const auto row_ptr = impl_C->csr.row_ptr_ptr;
  const auto col_indx = impl_C->csr.col_indx_ptr;
  const auto index_base = row_ptr[0];
  const auto nnz = row_ptr[m] - index_base;

  // Set up structure for result, which will match the sparsity structure of C
  std::vector<perflibs_int_t> row_ptrAB(row_ptr, row_ptr + m + 1);
  std::vector<perflibs_int_t> col_indxAB(col_indx, col_indx + nnz);
  std::vector<T> vals_AB(nnz);

  constexpr perflibs_int_t Ti = 32;
  constexpr perflibs_int_t Tj = 32;
  constexpr perflibs_int_t Tk = 128;

  // Values of A are conjugated
  auto conj_a = [&](perflibs_int_t a_indx_start, perflibs_int_t b_indx_start,
                    perflibs_int_t a_indx_scale, perflibs_int_t b_indx_scale) {
#if __ARM_LINUX_COMPILER__ == 1
#pragma omp parallel for
#else
#pragma omp parallel for schedule(dynamic)
#endif
    for (perflibs_int_t ii = 0; ii < m; ii += Ti) {
      for (perflibs_int_t jj = 0; jj < n; jj += Tj) {
        for (perflibs_int_t kk = 0; kk < k; kk += Tk) {
          for (perflibs_int_t i = ii; i < std::min(ii + Ti, m); i++) {
            const T *a_ptr = &impl_A->dense.vals_ptr[i * a_indx_start];
            for (perflibs_int_t nonzero_ptr = row_ptr[i] - index_base;
                 nonzero_ptr < row_ptr[i + 1] - index_base; nonzero_ptr++) {
              perflibs_int_t j = col_indx[nonzero_ptr] - index_base;
              if (j >= jj && j < std::min(jj + Tj, n)) {
                const T *b_ptr = &impl_B->dense.vals_ptr[j * b_indx_start];
                T sum = 0.0;
#pragma omp simd
                for (perflibs_int_t ki = kk; ki < std::min(kk + Tk, k); ki++) {
                  sum += perflibs::sparse::conj(a_ptr[ki * a_indx_scale]) *
                         b_ptr[ki * b_indx_scale];
                }
                vals_AB[nonzero_ptr] += sum;
              }
            }
          }
        }
      }
    }
  };

  // Values of B are conjugated
  auto conj_b = [&](perflibs_int_t a_indx_start, perflibs_int_t b_indx_start,
                    perflibs_int_t a_indx_scale, perflibs_int_t b_indx_scale) {
#if __ARM_LINUX_COMPILER__ == 1
#pragma omp parallel for
#else
#pragma omp parallel for schedule(dynamic)
#endif
    for (perflibs_int_t ii = 0; ii < m; ii += Ti) {
      for (perflibs_int_t jj = 0; jj < n; jj += Tj) {
        for (perflibs_int_t kk = 0; kk < k; kk += Tk) {
          for (perflibs_int_t i = ii; i < std::min(ii + Ti, m); i++) {
            const T *a_ptr = &impl_A->dense.vals_ptr[i * a_indx_start];
            for (perflibs_int_t nonzero_ptr = row_ptr[i] - index_base;
                 nonzero_ptr < row_ptr[i + 1] - index_base; nonzero_ptr++) {
              perflibs_int_t j = col_indx[nonzero_ptr] - index_base;
              if (j >= jj && j < std::min(jj + Tj, n)) {
                const T *b_ptr = &impl_B->dense.vals_ptr[j * b_indx_start];
                T sum = 0;
#pragma omp simd
                for (perflibs_int_t ki = kk; ki < std::min(kk + Tk, k); ki++) {
                  sum += a_ptr[ki * a_indx_scale] *
                         perflibs::sparse::conj(b_ptr[ki * b_indx_scale]);
                }
                vals_AB[nonzero_ptr] += sum;
              }
            }
          }
        }
      }
    }
  };

  // Values of A and B are both conjugated, or neither are
  auto conj_neither_or_both = [&](auto conj_tag, perflibs_int_t a_indx_start,
                                  perflibs_int_t b_indx_start,
                                  perflibs_int_t a_indx_scale,
                                  perflibs_int_t b_indx_scale) {
    constexpr bool conj = decltype(conj_tag)::value;

#if __ARM_LINUX_COMPILER__ == 1
#pragma omp parallel for
#else
#pragma omp parallel for schedule(dynamic)
#endif
    for (perflibs_int_t ii = 0; ii < m; ii += Ti) {
      for (perflibs_int_t jj = 0; jj < n; jj += Tj) {
        for (perflibs_int_t kk = 0; kk < k; kk += Tk) {
          for (perflibs_int_t i = ii; i < std::min(ii + Ti, m); i++) {
            const T *a_ptr = &impl_A->dense.vals_ptr[i * a_indx_start];
            for (perflibs_int_t nonzero_ptr = row_ptr[i] - index_base;
                 nonzero_ptr < row_ptr[i + 1] - index_base; nonzero_ptr++) {
              perflibs_int_t j = col_indx[nonzero_ptr] - index_base;
              if (j >= jj && j < std::min(jj + Tj, n)) {
                const T *b_ptr = &impl_B->dense.vals_ptr[j * b_indx_start];
                T sum = 0;
#pragma omp simd
                for (perflibs_int_t ki = kk; ki < std::min(kk + Tk, k); ki++) {
                  sum += a_ptr[ki * a_indx_scale] * b_ptr[ki * b_indx_scale];
                }
                if constexpr (conj) {
                  vals_AB[nonzero_ptr] += perflibs::sparse::conj(sum);
                } else {
                  vals_AB[nonzero_ptr] += sum;
                }
              }
            }
          }
        }
      }
    }
  };

  // Determine parameters for correctly indexing into the values of A and B
  perflibs_int_t a_indx_start = use_col_maj_indxA ? 1 : ldA;
  perflibs_int_t a_indx_scale = use_col_maj_indxA ? ldA : 1;
  perflibs_int_t b_indx_start = use_col_maj_indxB ? ldB : 1;
  perflibs_int_t b_indx_scale = use_col_maj_indxB ? 1 : ldB;
  if (transA == PERFLIBS_SPARSE_OPERATION_CONJTRANS) {
    if (transB == PERFLIBS_SPARSE_OPERATION_CONJTRANS) {
      conj_neither_or_both(std::true_type{}, a_indx_start, b_indx_start,
                           a_indx_scale, b_indx_scale);
    } else {
      conj_a(a_indx_start, b_indx_start, a_indx_scale, b_indx_scale);
    }
  } else if (transB == PERFLIBS_SPARSE_OPERATION_CONJTRANS) {
    conj_b(a_indx_start, b_indx_start, a_indx_scale, b_indx_scale);
  } else {
    conj_neither_or_both(std::false_type{}, a_indx_start, b_indx_start,
                         a_indx_scale, b_indx_scale);
  }

  // Create a new CSR matrix for sampled(AB)
  auto ret = perflibs::sparse::fill_initial_data_csr(
      AB, m, n, row_ptrAB.data(), col_indxAB.data(), vals_AB.data(), 0);
  return ret;
}
template perflibs_status_t
sddmm_csr<float>(perflibs_sparse_hint_value transA,
                 perflibs_sparse_hint_value transB, float alpha,
                 perflibs_spmat_impl_t<float> *impl_A,
                 perflibs_spmat_impl_t<float> *impl_B, float beta,
                 perflibs_spmat_impl_t<float> *impl_C, perflibs_spmat_t AB);
template perflibs_status_t
sddmm_csr<double>(perflibs_sparse_hint_value transA,
                  perflibs_sparse_hint_value transB, double alpha,
                  perflibs_spmat_impl_t<double> *impl_A,
                  perflibs_spmat_impl_t<double> *impl_B, double beta,
                  perflibs_spmat_impl_t<double> *impl_C, perflibs_spmat_t AB);
template perflibs_status_t sddmm_csr<std::complex<float>>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    std::complex<float> alpha,
    perflibs_spmat_impl_t<std::complex<float>> *impl_A,
    perflibs_spmat_impl_t<std::complex<float>> *impl_B,
    std::complex<float> beta,
    perflibs_spmat_impl_t<std::complex<float>> *impl_C, perflibs_spmat_t AB);
template perflibs_status_t sddmm_csr<std::complex<double>>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    std::complex<double> alpha,
    perflibs_spmat_impl_t<std::complex<double>> *impl_A,
    perflibs_spmat_impl_t<std::complex<double>> *impl_B,
    std::complex<double> beta,
    perflibs_spmat_impl_t<std::complex<double>> *impl_C, perflibs_spmat_t AB);

/// Make sure two matrices agree on the index_base index; if not copy the
/// indices arrays of one of them and ajdust to zero-based
void make_index_base_equal(
    perflibs_int_t m, perflibs_int_t k, perflibs_int_t nnzA,
    perflibs_int_t nnzB, const perflibs_int_t **row_ptrA_ref,
    const perflibs_int_t **row_ptrB_ref, const perflibs_int_t **col_indxA_ref,
    const perflibs_int_t **col_indxB_ref,
    perflibs::sparse::pod_vector<perflibs_int_t> &index_copy,
    perflibs::sparse::pod_vector<perflibs_int_t> &ptr_copy) {

  auto row_ptrA = *row_ptrA_ref;
  auto col_indxA = *col_indxA_ref;
  auto row_ptrB = *row_ptrB_ref;
  auto col_indxB = *col_indxB_ref;

  if (row_ptrA[0] != row_ptrB[0]) {
    if (row_ptrA[0] == 1) {
      index_copy.resize(nnzA);
      for (perflibs_int_t i = 0; i < nnzA; i++) {
        index_copy[i] = col_indxA[i] - 1;
      }
      *col_indxA_ref = index_copy.data();
      ptr_copy.resize(m + 1);
      for (perflibs_int_t i = 0; i < m + 1; i++) {
        ptr_copy[i] = row_ptrA[i] - 1;
      }
      *row_ptrA_ref = ptr_copy.data();
    } else {
      index_copy.resize(nnzB);
      for (perflibs_int_t i = 0; i < nnzB; i++) {
        index_copy[i] = col_indxB[i] - 1;
      }
      *col_indxB_ref = index_copy.data();
      ptr_copy.resize(k + 1);
      for (perflibs_int_t i = 0; i < k + 1; i++) {
        ptr_copy[i] = row_ptrB[i] - 1;
      }
      *row_ptrB_ref = ptr_copy.data();
    }
  }
}

template <typename T>
perflibs_status_t
spmat_update_csr(perflibs_spmat_impl_t<T> *impl, perflibs_int_t n_updates,
                 const perflibs_int_t *row_indx, const perflibs_int_t *col_indx,
                 const T *vals) {
  auto nrows = impl->m;
  auto ncols = impl->n;
  auto index_base = impl->index_base;

  auto row_ptr_orig = impl->csr.row_ptr_ptr;
  auto col_indx_orig = impl->csr.col_indx_ptr;
  auto vals_orig = const_cast<T *>(impl->csr.vals_ptr);

  auto is_unit = impl->diag == PERFLIBS_SPARSE_DIAG_UNIT;

  for (auto i = 0; i < n_updates; i++) {
    auto row_num = row_indx[i] - index_base;
    auto col_num = col_indx[i] - index_base;
    if (row_num < 0 || row_num > nrows - 1) {
      return update_matrix_error(impl->error_handle, i + index_base);
    } else if (col_num < 0 || col_num > ncols - 1) {
      return update_matrix_error(impl->error_handle, i + index_base);
    } else {
      bool success = false;
      for (auto j = row_ptr_orig[row_num] - index_base;
           j < row_ptr_orig[row_num + 1] - index_base; j++) {
        if (col_indx_orig[j] == col_indx[i]) {
          vals_orig[j] = vals[i];
          if (row_num == col_num && is_unit && vals[i] != T(1)) {
            impl->diag = PERFLIBS_SPARSE_DIAG_NON_UNIT;
          }
          success = true;
          break;
        }
      }
      if (!success) {
        return update_matrix_error(impl->error_handle, i + index_base);
      }
    }
  }
  return PERFLIBS_STATUS_SUCCESS;
}
template perflibs_status_t
spmat_update_csr<float>(perflibs_spmat_impl_t<float> *impl,
                        perflibs_int_t n_updates,
                        const perflibs_int_t *row_indx,
                        const perflibs_int_t *col_indx, const float *vals);
template perflibs_status_t
spmat_update_csr<double>(perflibs_spmat_impl_t<double> *impl,
                         perflibs_int_t n_updates,
                         const perflibs_int_t *row_indx,
                         const perflibs_int_t *col_indx, const double *vals);
template perflibs_status_t spmat_update_csr<std::complex<float>>(
    perflibs_spmat_impl_t<std::complex<float>> *impl, perflibs_int_t n_updates,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_indx,
    const std::complex<float> *vals);
template perflibs_status_t spmat_update_csr<std::complex<double>>(
    perflibs_spmat_impl_t<std::complex<double>> *impl, perflibs_int_t n_updates,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_indx,
    const std::complex<double> *vals);

} // namespace perflibs::sparse
