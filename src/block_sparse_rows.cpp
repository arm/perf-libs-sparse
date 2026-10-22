/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "block_sparse_rows.hpp"
#include "dense.hpp"
#include "norm.hpp"
#include "object_helpers.hpp"
#include "types.hpp"

#include "util.hpp"

// Implementations of functions required to support BSR

namespace perflibs::sparse {

template <typename T>
perflibs_bsr<T> make_bsr(perflibs_dense_layout block_layout,
                         perflibs_int_t rows, perflibs_int_t cols,
                         perflibs_int_t block_size, perflibs_int_t nnzb,
                         perflibs_int_t nrowsb, perflibs_int_t nnz,
                         const T *vals, const perflibs_int_t *row_ptr,
                         const perflibs_int_t *col_indx, bool no_copy) {
  if (no_copy) {
    return perflibs_bsr<T>(block_layout, rows, cols, block_size, nnzb, nrowsb,
                           vals, row_ptr, col_indx);
  } else {
    return perflibs_bsr<T>(block_layout, rows, cols, block_size, nnzb, nrowsb,
                           nnz, vals, row_ptr, col_indx);
  }
}

template <typename T>
perflibs_sparse_matrix_shape_t
get_shape_bsr(perflibs_dense_layout layout, perflibs_int_t m, perflibs_int_t n,
              perflibs_int_t block_size, const perflibs_int_t *row_ptr,
              const perflibs_int_t *col_indx, const T *vals) {

  // We currently only care about shape for spsv, which requires a square
  // matrix, so get out early if the matrix is rectangular.
  if (m != n) {
    return PERFLIBS_SPARSE_SHAPE_RECTANGULAR;
  }

  perflibs_sparse_matrix_shape_t current =
      PERFLIBS_SPARSE_SHAPE_DIAGONAL; // not strictly upper or lower

  auto index_base = row_ptr[0];
  // We don't force the blocks themselves to be upper or lower triangular
  for (perflibs_int_t i = 0; i < m / block_size; i++) {
    // offset for start of row
    const auto row_start = &col_indx[row_ptr[i] - index_base];
    // offset for end of row
    const auto row_end = &col_indx[row_ptr[i + 1] - index_base];

    if (row_end != row_start) {
      // Just check the min and max elements
      auto [minp, maxp] = std::minmax_element(row_start, row_end);
      auto ci_min = *minp - index_base;
      auto ci_max = *maxp - index_base;

      if (ci_min < i && ci_max <= i) {
        // we're in lower triangular territory, get out if we've previously seen
        // evidence of upper
        if (current == PERFLIBS_SPARSE_SHAPE_UPPER_TRIANGULAR) {
          return PERFLIBS_SPARSE_SHAPE_RECTANGULAR;
        } else {
          current = PERFLIBS_SPARSE_SHAPE_LOWER_TRIANGULAR;
        }
      } else if (ci_min >= i && ci_max > i) {
        // we're in upper triangular territory, get out if we've previously seen
        // evidence of lower
        if (current == PERFLIBS_SPARSE_SHAPE_LOWER_TRIANGULAR) {
          return PERFLIBS_SPARSE_SHAPE_RECTANGULAR;
        } else {
          current = PERFLIBS_SPARSE_SHAPE_UPPER_TRIANGULAR;
        }
      } else if (ci_min < i && ci_max > i) {
        // if blocks are either side of the diagonal we've definitely got a
        // rectangular matrix
        return PERFLIBS_SPARSE_SHAPE_RECTANGULAR;
      }

      // Check the details of the diagonal block
      auto diag_i = ci_min == i ? ci_min : ci_max == i ? ci_max : -1;
      if (diag_i != -1) { // If we have a diagonal...

        // get the shape of the diagonal
        auto diag_vals =
            &vals[(row_ptr[diag_i] - index_base) * (block_size * block_size)];
        auto diag_block_shape = get_shape_dense(layout, block_size, block_size,
                                                diag_vals, block_size);

        // Remember that 'current' is never set to rectangular, we return
        // immediately, so it is either diag, upper or lower. If the block is
        // diagonal, we have no useful new information, so move on
        if (diag_block_shape != PERFLIBS_SPARSE_SHAPE_DIAGONAL) {
          // if we've previously only seen evidence of diagonal, update to
          // either lower/upper
          if (current == PERFLIBS_SPARSE_SHAPE_DIAGONAL) {
            current = diag_block_shape;
          }
          // otherwise diag_shape and current are either lower or upper, so if
          // they disagree return rect
          else if (current != diag_block_shape) {
            return PERFLIBS_SPARSE_SHAPE_RECTANGULAR;
          }
        }
      }
      // no diagonal in the row, and not strictly upper or strictly lower:
      // rectangular
      else {
        return PERFLIBS_SPARSE_SHAPE_RECTANGULAR;
      }
    }
  }

  // TODO handle diagonal specifically
  // For now, we just handle as a lower triangular matrix for BSR
  if (current == PERFLIBS_SPARSE_SHAPE_DIAGONAL) {
    return PERFLIBS_SPARSE_SHAPE_LOWER_TRIANGULAR;
  }

  return current;
}

template <typename T>
perflibs_sparse_matrix_diag_t
get_diag_bsr(perflibs_int_t m, perflibs_int_t n, perflibs_int_t block_size,
             const perflibs_int_t *row_ptr, const perflibs_int_t *col_indx,
             const T *vals, perflibs_sparse_matrix_shape_t shape) {

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
  for (perflibs_int_t i = 0; i < m / block_size; i++) {

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

      auto val_start = i * block_size * block_size;
      // check whether the diagonal of the block is all 1, if not return
      for (perflibs_int_t bi = 0; bi < block_size; bi++) {
        auto elem = vals[val_start + bi * block_size + bi];
        if (elem == T(0)) {
          return PERFLIBS_SPARSE_DIAG_ZERO;
        } else if (elem == T(1)) {
          unit_diag++;
        } else {
          return PERFLIBS_SPARSE_DIAG_NON_UNIT;
        }
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
perflibs_status_t fill_initial_data_bsr(perflibs_spmat_top_t *A,
                                        perflibs_dense_layout block_layout,
                                        perflibs_int_t m, perflibs_int_t n,
                                        perflibs_int_t block_size,
                                        const perflibs_int_t *row_ptr,
                                        const perflibs_int_t *col_indx,
                                        const T *vals, bool no_copy) {

  auto impl = reinterpret_cast<perflibs_spmat_impl_t<T> *>(A->impl);

  // Checking input values - error codes are inline with the argument number
  // they refer to. i.e. 1, 2, 3, ...
  if (!(block_layout == PERFLIBS_COL_MAJOR ||
        block_layout == PERFLIBS_ROW_MAJOR)) {
    impl->error_handle.perflibs_error_type =
        PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    impl->error_handle.perflibs_error_code = 2;
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }
  if (m < 0) {
    impl->error_handle.perflibs_error_type =
        PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    impl->error_handle.perflibs_error_code = 3;
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }
  if (n < 0) {
    impl->error_handle.perflibs_error_type =
        PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    impl->error_handle.perflibs_error_code = 4;
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }
  if (block_size <= 0 || m % block_size != 0 || n % block_size != 0) {
    impl->error_handle.perflibs_error_type =
        PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    impl->error_handle.perflibs_error_code = 5;
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }
  // Valid base index check
  if (row_ptr[0] != 0 && row_ptr[0] != 1) {
    impl->error_handle.perflibs_error_type =
        PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    impl->error_handle.perflibs_error_code = 6;
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }

  impl->m = m;
  impl->n = n;
  impl->index_base = row_ptr[0];

  // no. of rows in the blocked matrix
  auto nrowsb = m / block_size;

  // no. of non-zero blocks
  auto nnzb = row_ptr[nrowsb] - impl->index_base;

  // no. of non-zero elements. NOTE: Some of these may actually be zero.
  // Perhaps, a more accurate name would be 'no. of values stored'
  impl->nnz = nnzb * block_size * block_size;

  impl->spmat_format = perflibs_format_bsr;
  impl->no_copy = no_copy;

  impl->shape =
      get_shape_bsr(block_layout, m, n, block_size, row_ptr, col_indx, vals);
  impl->diag =
      get_diag_bsr(m, n, block_size, row_ptr, col_indx, vals, impl->shape);

  impl->bsr = make_bsr<T>(block_layout, impl->m, impl->n, block_size, nnzb,
                          nrowsb, impl->nnz, vals, row_ptr, col_indx, no_copy);

  return PERFLIBS_STATUS_SUCCESS;
}

template perflibs_status_t fill_initial_data_bsr<float>(
    perflibs_spmat_top_t *A, perflibs_dense_layout block_layout,
    perflibs_int_t m, perflibs_int_t n, perflibs_int_t block_size,
    const perflibs_int_t *row_ptr, const perflibs_int_t *col_indx,
    const float *vals, bool no_copy);
template perflibs_status_t fill_initial_data_bsr<double>(
    perflibs_spmat_top_t *A, perflibs_dense_layout block_layout,
    perflibs_int_t m, perflibs_int_t n, perflibs_int_t block_size,
    const perflibs_int_t *row_ptr, const perflibs_int_t *col_indx,
    const double *vals, bool no_copy);
template perflibs_status_t fill_initial_data_bsr<std::complex<float>>(
    perflibs_spmat_top_t *A, perflibs_dense_layout block_layout,
    perflibs_int_t m, perflibs_int_t n, perflibs_int_t block_size,
    const perflibs_int_t *row_ptr, const perflibs_int_t *col_indx,
    const std::complex<float> *vals, bool no_copy);
template perflibs_status_t fill_initial_data_bsr<std::complex<double>>(
    perflibs_spmat_top_t *A, perflibs_dense_layout block_layout,
    perflibs_int_t m, perflibs_int_t n, perflibs_int_t block_size,
    const perflibs_int_t *row_ptr, const perflibs_int_t *col_indx,
    const std::complex<double> *vals, bool no_copy);

template <typename T>
perflibs_bsr<T> &perflibs_bsr<T>::operator=(const perflibs_bsr<T> &other) {
  if (&other == this) {
    return *this;
  }
  // Copy variables
  block_layout = other.block_layout;
  m = other.m;
  n = other.n;
  block_size = other.block_size;
  if (m >= 0 && n >= 0 && block_size > 0) {
    nrowsb = other.nrowsb;
    nnzb = other.nnzb;
    auto nnz = nnzb * block_size * block_size;

    // If the vectors are populated, make the const pointers point to them
    // otherwise construct a new vector
    copy_from_vector_or_ptr(&vals_ptr, vals, other.vals_ptr, other.vals, nnz);

    copy_from_vector_or_ptr(&col_indx_ptr, col_indx, other.col_indx_ptr,
                            other.col_indx, nnzb);

    copy_from_vector_or_ptr(&row_ptr_ptr, row_ptr, other.row_ptr_ptr,
                            other.row_ptr, nrowsb + 1);
  }
  return *this;
}
template perflibs_bsr<float> &
perflibs_bsr<float>::operator=(const perflibs_bsr<float> &other);
template perflibs_bsr<double> &
perflibs_bsr<double>::operator=(const perflibs_bsr<double> &other);
template perflibs_bsr<std::complex<float>> &
perflibs_bsr<std::complex<float>>::operator=(
    const perflibs_bsr<std::complex<float>> &other);
template perflibs_bsr<std::complex<double>> &
perflibs_bsr<std::complex<double>>::operator=(
    const perflibs_bsr<std::complex<double>> &other);

// Scale the input values and write them into the current matrix object
template <typename T> void perflibs_bsr<T>::scale_matrix(T alpha) {
  auto nnz = nnzb * block_size * block_size;

  // Allocate our copy vector (in case we're copying from user's data)
  if (vals.data() != vals_ptr) {
    vals.resize(nnz);
  }

  for (perflibs_int_t i = 0; i < nnz; i++) {
    vals[i] = vals_ptr[i] * alpha;
  }

  // Set the new pointer
  vals_ptr = vals.data();
}
template void perflibs_bsr<float>::scale_matrix(float alpha);
template void perflibs_bsr<double>::scale_matrix(double alpha);
template void
perflibs_bsr<std::complex<float>>::scale_matrix(std::complex<float> alpha);
template void
perflibs_bsr<std::complex<double>>::scale_matrix(std::complex<double> alpha);

template <typename T>
perflibs_bsr<T> csr2bsr(perflibs_int_t m, perflibs_int_t n, const T *vals,
                        const perflibs_int_t *row_ptr,
                        const perflibs_int_t *col_indx) {

  perflibs_dense_layout block_layout = PERFLIBS_COL_MAJOR;
  perflibs_int_t block_size = (int64_t)1;

  auto nrowsb = m;
  auto nnz = row_ptr[m] - row_ptr[0];
  auto nnzb = nnz;

  // Copy csr to data into bsr structure. nrowsb/nnzb used here for clarity
  std::vector<perflibs_int_t> bsr_row_ptr(row_ptr, row_ptr + nrowsb + 1);
  std::vector<perflibs_int_t> bsr_col_indx(col_indx, col_indx + nnzb);
  std::vector<T> bsr_vals(vals, vals + nnz);

  return perflibs_bsr<T>(block_layout, m, n, block_size, nnzb, nrowsb, nnz,
                         bsr_vals.data(), bsr_row_ptr.data(),
                         bsr_col_indx.data());
}

template perflibs_bsr<float> csr2bsr(perflibs_int_t m, perflibs_int_t n,
                                     const float *vals,
                                     const perflibs_int_t *row_ptr,
                                     const perflibs_int_t *col_indx);
template perflibs_bsr<double> csr2bsr(perflibs_int_t m, perflibs_int_t n,
                                      const double *vals,
                                      const perflibs_int_t *row_ptr,
                                      const perflibs_int_t *col_indx);
template perflibs_bsr<std::complex<float>>
csr2bsr(perflibs_int_t m, perflibs_int_t n, const std::complex<float> *vals,
        const perflibs_int_t *row_ptr, const perflibs_int_t *col_indx);
template perflibs_bsr<std::complex<double>>
csr2bsr(perflibs_int_t m, perflibs_int_t n, const std::complex<double> *vals,
        const perflibs_int_t *row_ptr, const perflibs_int_t *col_indx);

/* BSR kernels */

template <typename T> void spmv_bsr_realloc_data(perflibs_bsr<T> &bsr) {
  // If we are only using a single thread, nothing to be done
  if (bsr.nthreads == 1)
    return;

  // Use a pod_vector, which only assigns memory, doesn't touch any of it
  decltype(bsr.vals) new_vals(bsr.vals.size());
  decltype(bsr.col_indx) new_col_indx(bsr.col_indx.size());
  decltype(bsr.row_ptr) new_row_ptr(bsr.row_ptr.size());

  const auto nrowsb = bsr.nrowsb;
  const auto num_block_els = bsr.block_size * bsr.block_size;
  const auto row_ptr = bsr.row_ptr_ptr;
  const auto off = row_ptr[0];
  const auto vals = bsr.vals_ptr;
  const auto col_indx = bsr.col_indx_ptr;

// Loop over the rows in the vector in the same manner as we do in SpMV
#pragma omp parallel for default(none)                                         \
    firstprivate(nrowsb, num_block_els, row_ptr, off, vals, col_indx)          \
    shared(new_vals, new_col_indx, new_row_ptr) num_threads(bsr.nthreads)      \
    schedule(static)
  // Loop over the block rows
  for (int64_t b_row = 0; b_row < nrowsb; ++b_row) {
    // Copy the values from the current vector in to the new one
    perflibs_int_t vals_start_ind = (row_ptr[b_row] - off) * num_block_els;
    perflibs_int_t vals_end_ind = (row_ptr[b_row + 1] - off) * num_block_els;
    for (auto i = vals_start_ind; i < vals_end_ind; ++i) {
      new_vals[i] = vals[i];
    }

    // Now copy the column indices
    for (auto i = row_ptr[b_row] - off; i < row_ptr[b_row + 1] - off; ++i) {
      new_col_indx[i] = col_indx[i];
    }
    new_row_ptr[b_row] = row_ptr[b_row];
    if (b_row == nrowsb - 1)
      new_row_ptr[nrowsb] = row_ptr[nrowsb];
  }

  // Swap the newly created vectors with the ones in the block sparse row matrix
  // passed in
  bsr.vals.swap(new_vals);
  bsr.col_indx.swap(new_col_indx);
  bsr.row_ptr.swap(new_row_ptr);
  // And don't forget to point to the correct data
  bsr.vals_ptr = bsr.vals.data();
  bsr.col_indx_ptr = bsr.col_indx.data();
  bsr.row_ptr_ptr = bsr.row_ptr.data();
}

template void spmv_bsr_realloc_data(perflibs_bsr<float> &bsr);
template void spmv_bsr_realloc_data(perflibs_bsr<double> &bsr);
template void spmv_bsr_realloc_data(perflibs_bsr<std::complex<float>> &bsr);
template void spmv_bsr_realloc_data(perflibs_bsr<std::complex<double>> &bsr);

/*
 * Performs SpMV multiplication of the form alpha Ax + beta y for matrix A,
 * vectors x and y, and scalars alpha and beta.
 * @param [in] bsr		The block sparse matrix with which to perform
 * the SpMV
 * @param [in] x		The vector with which to multiply the matrix
 * @param [in,out] y	Initial values in this vector are multiplied by beta.
 *                  	The result of the axpy operation is stored in here.
 * @param [in] alpha	Scalar to scale the result of the matrix-vector
 * multiplication
 * @param [in] beta		Scalar to scale the input vector y by before
 * adding the result of the matrix-vector multiplication
 *
 * This kernel is treated as reproducible. Each loop iteration owns one output
 * block row and accumulates blocks in stored order.
 */
template <typename T>
void spmv_bsr_vanilla(const perflibs_bsr<T> &bsr, const T *const x, T *y,
                      const T alpha, const T beta) {

  const auto block_row_ptr = bsr.row_ptr_ptr;
  const auto block_col_ptr = bsr.col_indx_ptr;
  const auto vals = bsr.vals_ptr;
  const auto off = block_row_ptr[0];
  const auto num_block_els = bsr.block_size * bsr.block_size;
  const auto nrowsb = bsr.nrowsb;
  const auto block_size = bsr.block_size;
  const auto block_layout = bsr.block_layout;

#pragma omp parallel for default(none)                                         \
    firstprivate(block_row_ptr, block_col_ptr, vals, off, num_block_els,       \
                     nrowsb, block_layout, block_size, x, y, alpha, beta)      \
    num_threads(bsr.nthreads)
  // Loop over the blocks
  for (int64_t b_row = 0; b_row < nrowsb; ++b_row) {
    // Set y for the current row(s) in the block
    auto curr_row_ind = b_row * block_size;
    if (beta != (T)0) {
      for (auto i = curr_row_ind; i < curr_row_ind + block_size; ++i) {
        y[i] *= beta;
      }
    } else {
      for (auto i = curr_row_ind; i < curr_row_ind + block_size; ++i) {
        y[i] = (T)0;
      }
    }

    // Now perform the multiplication for the blocks in the row
    for (auto block_num = block_row_ptr[b_row] - off;
         block_num < block_row_ptr[b_row + 1] - off; ++block_num) {
      // Get the index into the array of values in the matrix. This depends on
      // the linear index of the block
      auto val_ind = block_num * num_block_els;
      auto start_col = (block_col_ptr[block_num] - off) * block_size;
      if (block_layout == PERFLIBS_ROW_MAJOR) {
        for (int64_t i = 0; i < block_size; ++i) {
          // Sum the contribution to the MV product over each row of the block
          T sum = 0;
          for (int64_t j = 0; j < block_size; ++j) {
            sum += x[start_col + j] * vals[val_ind + i * block_size + j];
          }
          y[curr_row_ind + i] += alpha * sum;
        }
      } else {
        for (int64_t j = 0; j < block_size; ++j) {
          for (int64_t i = 0; i < block_size; ++i) {
            y[curr_row_ind + i] +=
                alpha * x[start_col + j] * vals[val_ind + j * block_size + i];
          }
        }
      }
    }
  }
}

template <typename T>
void spmv_bsr_notrans_block_size_3(const perflibs_bsr<T> &bsr, const T *const x,
                                   T *y, const T alpha, const T beta) {

  const auto block_row_ptr = bsr.row_ptr_ptr;
  const auto block_col_ptr = bsr.col_indx_ptr;
  const auto vals = bsr.vals_ptr;
  const auto off = block_row_ptr[0];
  const auto nrowsb = bsr.nrowsb;
  const auto block_layout = bsr.block_layout;
  constexpr int64_t num_block_els = 9;
  constexpr int64_t block_size = 3;

#pragma omp parallel for default(none)                                         \
    firstprivate(block_row_ptr, block_col_ptr, vals, off, nrowsb,              \
                     block_layout, x, y, alpha, beta)                          \
    num_threads(bsr.nthreads) schedule(static)
  // Loop over the blocks
  for (int64_t b_row = 0; b_row < nrowsb; ++b_row) {
    // Set variables to store partial sums
    T sum1[block_size] = {};
    T sum2[block_size] = {};
    T sum3[block_size] = {};

    // Now loop over the blocks in the row. We assume row major ordering
    // within the blocks for now. We loop over two blocks at a time, if
    // we can, so that there are an even number of multiply adds to perform
    for (perflibs_int_t block_num = block_row_ptr[b_row] - off;
         block_num < block_row_ptr[b_row + 1] - off; ++block_num) {
      // Get the index into the array of values for this block and the next one
      auto val_ind = block_num * num_block_els;
      auto col_ind = (block_col_ptr[block_num] - off) * block_size;

      if (block_layout == PERFLIBS_ROW_MAJOR) {
        // Set the partial sums for the first row first the first block
        sum1[0] += vals[val_ind] * x[col_ind];
        sum1[1] += vals[val_ind + 1] * x[col_ind + 1];
        // Set the partial sums for the second row for the first block
        sum2[0] += vals[val_ind + block_size] * x[col_ind];
        sum2[1] += vals[val_ind + block_size + 1] * x[col_ind + 1];

        sum3[0] += vals[val_ind + block_size * 2] * x[col_ind];
        sum3[1] += vals[val_ind + block_size * 2 + 1] * x[col_ind + 1];

        // Tail from the first block
        sum1[2] += vals[val_ind + 2] * x[col_ind + 2];
        sum2[2] += vals[val_ind + block_size + 2] * x[col_ind + 2];
        sum3[2] += vals[val_ind + block_size * 2 + 2] * x[col_ind + 2];
      } else { // PERFLIBS_COL_MAJOR
        sum1[0] += vals[val_ind] * x[col_ind];
        sum2[0] += vals[val_ind + 1] * x[col_ind];
        sum3[0] += vals[val_ind + 2] * x[col_ind];

        sum1[1] += vals[val_ind + block_size] * x[col_ind + 1];
        sum2[1] += vals[val_ind + block_size + 1] * x[col_ind + 1];
        sum3[1] += vals[val_ind + block_size + 2] * x[col_ind + 1];

        sum1[2] += vals[val_ind + block_size * 2] * x[col_ind + 2];
        sum2[2] += vals[val_ind + block_size * 2 + 1] * x[col_ind + 2];
        sum3[2] += vals[val_ind + block_size * 2 + 2] * x[col_ind + 2];
      }
    }

    // Update the sum of the rows
    sum1[0] += sum1[1];
    sum2[0] += sum2[1];
    sum3[0] += sum3[1];
    sum1[0] += sum1[2];
    sum2[0] += sum2[2];
    sum3[0] += sum3[2];

    // And add the result to the output vector
    const auto curr_row_ind = b_row * block_size;
    // Scale y for the current rows in the block
    if (beta != (T)0) {
      y[curr_row_ind] *= beta;
      y[curr_row_ind + 1] *= beta;
      y[curr_row_ind + 2] *= beta;
    } else {
      y[curr_row_ind] = (T)0;
      y[curr_row_ind + 1] = (T)0;
      y[curr_row_ind + 2] = (T)0;
    }

    y[curr_row_ind] += alpha * sum1[0];
    y[curr_row_ind + 1] += alpha * sum2[0];
    y[curr_row_ind + 2] += alpha * sum3[0];
  }
}

template <typename T, int N>
void spmv_bsr_notrans_generic(const perflibs_bsr<T> &bsr, const T *const x,
                              T *y, const T alpha, const T beta) {
  const auto block_row_ptr = bsr.row_ptr_ptr;
  const auto block_col_ptr = bsr.col_indx_ptr;
  const auto vals = bsr.vals_ptr;
  const auto off = block_row_ptr[0];
  const auto nrowsb = bsr.nrowsb;
  const auto block_layout = bsr.block_layout;
  constexpr perflibs_int_t num_block_els = N * N;
  constexpr perflibs_int_t block_size = N;

#pragma omp parallel for default(none)                                         \
    firstprivate(block_row_ptr, block_col_ptr, vals, off, nrowsb,              \
                     block_layout, x, y, alpha, beta)                          \
    num_threads(bsr.nthreads)
  for (int64_t b_row = 0; b_row < nrowsb; ++b_row) {

    // Set variables to store partial sums, and initialize to zero
    T sums[block_size * block_size] = {};

    // Loop over the blocks in the row
    for (perflibs_int_t block_num = block_row_ptr[b_row] - off;
         block_num < block_row_ptr[b_row + 1] - off; ++block_num) {
      const auto val_ind = block_num * num_block_els;
      const auto col_ind = (block_col_ptr[block_num] - off) * block_size;

      if (block_layout == PERFLIBS_ROW_MAJOR) {
        // Set the partial sums for each row in the block
        for (perflibs_int_t i = 0; i < block_size; ++i) {
#pragma omp simd
          for (perflibs_int_t j = 0; j < block_size; ++j) {
            sums[i * block_size + j] +=
                vals[val_ind + i * block_size + j] * x[col_ind + j];
          }
        }
      } else { // PERFLIBS_COL_MAJOR
        for (perflibs_int_t j = 0; j < block_size; ++j) {
#pragma omp simd
          for (perflibs_int_t i = 0; i < block_size; ++i) {
            sums[i + j * block_size] +=
                vals[val_ind + i + j * block_size] * x[col_ind + j];
          }
        }
      }
    }

    const auto curr_row_ind = b_row * block_size;
    if (block_layout == PERFLIBS_ROW_MAJOR) {
      for (perflibs_int_t i = 0; i < block_size; ++i) {
#pragma omp simd
        for (perflibs_int_t j = 1; j < block_size; ++j) {
          sums[i * block_size] += sums[i * block_size + j];
        }
      }
      if (beta != (T)0) {
#pragma omp simd
        for (perflibs_int_t i = 0; i < block_size; ++i) {
          y[curr_row_ind + i] *= beta;
        }
      } else {
#pragma omp simd
        for (perflibs_int_t i = 0; i < block_size; ++i) {
          y[curr_row_ind + i] = (T)0;
        }
      }
#pragma omp simd
      for (perflibs_int_t i = 0; i < block_size; ++i) {
        y[curr_row_ind + i] += alpha * sums[i * block_size];
      }
    } else { // PERFLIBS_COL_MAJOR
      for (perflibs_int_t j = 1; j < block_size; ++j) {
#pragma omp simd
        for (perflibs_int_t i = 0; i < block_size; ++i) {
          sums[i] += sums[i + j * block_size];
        }
      }
      if (beta != (T)0) {
#pragma omp simd
        for (perflibs_int_t i = 0; i < block_size; ++i) {
          y[curr_row_ind + i] *= beta;
        }
      } else {
#pragma omp simd
        for (perflibs_int_t i = 0; i < block_size; ++i) {
          y[curr_row_ind + i] = (T)0;
        }
      }
#pragma omp simd
      for (perflibs_int_t i = 0; i < block_size; ++i) {
        y[curr_row_ind + i] += alpha * sums[i];
      }
    }
  }
}

/*
 * Performs a sparse matrix vector multiplication in the case that no transpose
 * is required
 */
template <typename T>
void spmv_bsr_notrans(const perflibs_bsr<T> &bsr, const T *x, T *y, T alpha,
                      T beta) {
  switch (bsr.block_size) {
  case 2:
    spmv_bsr_notrans_generic<T, 2>(bsr, x, y, alpha, beta);
    return;
  case 3:
    spmv_bsr_notrans_block_size_3(bsr, x, y, alpha, beta);
    return;
  case 4:
    spmv_bsr_notrans_generic<T, 4>(bsr, x, y, alpha, beta);
    return;
  case 5:
    spmv_bsr_notrans_generic<T, 5>(bsr, x, y, alpha, beta);
    return;
  default:
    spmv_bsr_vanilla<T>(bsr, x, y, alpha, beta);
    return;
  }
}

template <typename T>
void spmv_bsr(const perflibs_bsr<T> &bsr, sparse_hint_value_internal trans,
              const T *x, T *y, T alpha, T beta) {
  if (trans == PERFLIBS_OPERATION_NOTRANS) {
    if (bsr.use_vanilla) {
      spmv_bsr_vanilla<T>(bsr, x, y, alpha, beta);
    } else {
      spmv_bsr_notrans<T>(bsr, x, y, alpha, beta);
    }
  } else {
    // No transpose of BSR yet implemented
    assert(false);
  }
}

template void spmv_bsr<float>(const perflibs_bsr<float> &bsr,
                              sparse_hint_value_internal trans, const float *x,
                              float *y, float alpha, float beta);
template void spmv_bsr<double>(const perflibs_bsr<double> &bsr,
                               sparse_hint_value_internal trans,
                               const double *x, double *y, double alpha,
                               double beta);
template void
spmv_bsr<std::complex<float>>(const perflibs_bsr<std::complex<float>> &bsr,
                              sparse_hint_value_internal trans,
                              const std::complex<float> *x,
                              std::complex<float> *y, std::complex<float> alpha,
                              std::complex<float> beta);
template void spmv_bsr<std::complex<double>>(
    const perflibs_bsr<std::complex<double>> &bsr,
    sparse_hint_value_internal trans, const std::complex<double> *x,
    std::complex<double> *y, std::complex<double> alpha,
    std::complex<double> beta);

template <bool isConj, typename T>
void spsv_bsr_notrans_lower(const perflibs_bsr<T> &bsr,
                            sparse_hint_value_internal uplo,
                            sparse_hint_value_internal diag, T *x, const T *y,
                            T alpha) {
  //
  // We want to solve the following for x,
  //
  //    Ax = y,
  //
  // where A is a lower triangular matrix. Splitting A and y into
  // submatrices/subvectors;
  //      ___________________________
  //     |      |      |      |      |
  //     | A_00 |      |      |      |
  //     |______|______|______|______|
  //     |      |      |      |      |
  //     | A_10 | A_11 |      |      |
  // A = |______|______|______|______|
  //     |      |      |      |      |
  //     | A_20 | A_21 | A_22 |      |
  //     |______|______|______|______|
  //     |      |      |      |      |
  //     | A_30 | A_31 | A_32 | A_33 |
  //     |______|______|______|______|
  //
  //      _____
  //     |     |
  //     | y_0 |
  //     |_____|
  //     |     |
  //     | y_1 |
  // y = |_____|
  //     |     |
  //     | y_2 |
  //     |_____|
  //     |     |
  //     | y_3 |
  //     |_____|
  //
  // It can be solved with the following iterative algorithm
  //
  //    x_0 = y_0 / A_00
  //
  //    x_i = (y_i - sum_j(A_ij * x_j)) / A_ii
  //
  // x_i requires all x_j where j<i. A_ii is a lower triangular matrix, so
  // applying it's inverse can be done with a trsv.
  //

  static_assert(!isConj, "Not supported yet");

  const auto k = bsr.block_size;
  const auto row_ptr = bsr.row_ptr_ptr;
  const auto col_indxes = bsr.col_indx_ptr;
  const auto vals = bsr.vals_ptr;
  const auto nrows = bsr.m / bsr.block_size;
  const auto off = row_ptr[0];
  const auto layout = bsr.block_layout;

  const auto trans_gemv = PERFLIBS_SPARSE_OPERATION_NOTRANS;
  const auto trans_trsv =
      PERFLIBS_OPERATION_NOTRANS; // We use different enums to pass trans around
                                  // :eye_roll:

  for (int i = 0; i < bsr.n; i++) {
    x[i] = alpha * y[i];
  }

  for (perflibs_int_t row_indx = 0; row_indx < nrows; ++row_indx) {
    auto row_start_indx = row_ptr[row_indx] - off;
    auto row_end_indx = row_ptr[row_indx + 1] - 1 - off;

    for (auto val_index = row_start_indx; val_index <= row_end_indx;
         ++val_index) {
      const auto col_indx = col_indxes[val_index] - off;
      const auto vals_sub = &vals[k * k * val_index];
      const auto y_sub = &x[k * col_indx];
      auto x_sub = &x[k * row_indx];

      if (col_indx == row_indx) {
        spsv_trsv(layout, trans_trsv, uplo, diag, k, vals_sub, k, x_sub);
      } else if (col_indx < row_indx) {
        spmv_gemv(layout, trans_gemv, k, k, vals_sub, k, T{-1}, y_sub, T{1},
                  x_sub);
      } else {
        // Currently we support sparse triangular matrices with non zero blocks
        // that are implicitly zero. The zeroes are implied by the structure
        // hint passed in by the user. In the future we plan on insisting that
        // the matrix matches the structure passed in as a hint. Until then,
        // this break is needed.
        break;
      }
    }
  }
}

template <typename T>
void spsv_bsr(const perflibs_bsr<T> &bsr, sparse_hint_value_internal trans,
              sparse_hint_value_internal uplo, sparse_hint_value_internal diag,
              T *x, const T *y, T alpha) {
  const auto is_trans = trans == PERFLIBS_OPERATION_TRANS ||
                        trans == PERFLIBS_OPERATION_CONJTRANS;
  const auto is_conj = trans == PERFLIBS_OPERATION_CONJTRANS ||
                       trans == PERFLIBS_OPERATION_CONJNOTRANS;
  if (!is_trans && !is_conj && uplo == PERFLIBS_SHAPE_LOWER_TRIANGULAR) {
    spsv_bsr_notrans_lower<false>(bsr, uplo, diag, x, y, alpha);
  } else {
    const auto csr =
        bsr2csr(bsr.block_layout, bsr.m, bsr.n, bsr.block_size, bsr.nnzb,
                bsr.nrowsb, bsr.vals_ptr, bsr.row_ptr_ptr, bsr.col_indx_ptr);
    spsv_csr(csr, trans, uplo, diag, x, y, alpha);
  }
}

template void spsv_bsr<float>(const perflibs_bsr<float> &bsr,
                              sparse_hint_value_internal trans,
                              sparse_hint_value_internal uplo,
                              sparse_hint_value_internal diag, float *x,
                              const float *y, float alpha);
template void spsv_bsr<double>(const perflibs_bsr<double> &bsr,
                               sparse_hint_value_internal trans,
                               sparse_hint_value_internal uplo,
                               sparse_hint_value_internal diag, double *x,
                               const double *y, double alpha);
template void spsv_bsr<std::complex<float>>(
    const perflibs_bsr<std::complex<float>> &bsr,
    sparse_hint_value_internal trans, sparse_hint_value_internal uplo,
    sparse_hint_value_internal diag, std::complex<float> *x,
    const std::complex<float> *y, std::complex<float> alpha);
template void spsv_bsr<std::complex<double>>(
    const perflibs_bsr<std::complex<double>> &bsr,
    sparse_hint_value_internal trans, sparse_hint_value_internal uplo,
    sparse_hint_value_internal diag, std::complex<double> *x,
    const std::complex<double> *y, std::complex<double> alpha);

template <typename T>
void spnorm_inf_bsr(const perflibs_bsr<T> &bsr,
                    perflibs::sparse::remove_complex_t<T> *result) {
  using RT = perflibs::sparse::remove_complex_t<T>;

  std::vector<RT> rowsums(bsr.m);
  auto index_base = bsr.row_ptr_ptr[0];
  auto block_size = bsr.block_size;
  for (auto rowb = 0; rowb < bsr.nrowsb; rowb++) {
    auto rowsum_index_base = rowb * block_size;
    for (auto b = bsr.row_ptr_ptr[rowb] - index_base;
         b < bsr.row_ptr_ptr[rowb + 1] - index_base; b++) {
      auto vals_index_base = b * block_size * block_size;
      for (auto i = 0; i < block_size; i++) {
        auto vals_index = vals_index_base + (i * block_size);
        for (auto j = 0; j < block_size; j++) {
          auto rowsum_offset = bsr.block_layout == PERFLIBS_ROW_MAJOR ? i : j;
          rowsums[rowsum_index_base + rowsum_offset] +=
              std::abs(bsr.vals_ptr[vals_index + j]);
        }
      }
    }
  }

  *result = spnorm_max(rowsums);
}
template void spnorm_inf_bsr<float>(const perflibs_bsr<float> &bsr,
                                    float *result);
template void spnorm_inf_bsr<double>(const perflibs_bsr<double> &bsr,
                                     double *result);
template void spnorm_inf_bsr<std::complex<float>>(
    const perflibs_bsr<std::complex<float>> &bsr, float *result);
template void spnorm_inf_bsr<std::complex<double>>(
    const perflibs_bsr<std::complex<double>> &bsr, double *result);

template <typename T>
perflibs_status_t
spmat_update_bsr(perflibs_spmat_impl_t<T> *impl, perflibs_int_t n_updates,
                 const perflibs_int_t *row_indx, const perflibs_int_t *col_indx,
                 const T *vals) {
  auto index_base = impl->index_base;
  auto block_size = impl->bsr.block_size;
  auto block_layout = impl->bsr.block_layout;
  auto nrowsb = impl->bsr.nrowsb;
  auto ncolsb = impl->n / block_size;
  auto is_unit = impl->diag == PERFLIBS_SPARSE_DIAG_UNIT;

  auto row_ptr_orig = impl->bsr.row_ptr_ptr;
  auto col_indx_orig = impl->bsr.col_indx_ptr;
  auto vals_orig = const_cast<T *>(impl->bsr.vals_ptr);

  for (auto i = 0; i < n_updates; i++) {
    perflibs_int_t sub_row = (row_indx[i] - index_base) % block_size;
    perflibs_int_t sub_col = (col_indx[i] - index_base) % block_size;

    perflibs_int_t block_row = (row_indx[i] - index_base) / block_size;
    perflibs_int_t block_col = (col_indx[i] - index_base) / block_size;

    if (block_row < 0 || block_row > nrowsb - 1 || block_col < 0 ||
        block_col > ncolsb - 1) {
      return update_matrix_error(impl->error_handle, i + index_base);
    }
    bool success = false;
    for (auto j = row_ptr_orig[block_row] - index_base;
         j < row_ptr_orig[block_row + 1] - index_base; j++) {
      if (col_indx_orig[j] - index_base < block_col) {
        continue;
      }
      if (col_indx_orig[j] - index_base > block_col) {
        break;
      }

      auto block_offset = j * block_size * block_size;
      auto sub_index = (block_layout == PERFLIBS_COL_MAJOR)
                           ? sub_col * block_size + sub_row
                           : sub_row * block_size + sub_col;
      auto update_index = block_offset + sub_index;
      vals_orig[update_index] = vals[i];

      if (row_indx[i] == col_indx[i] && is_unit && vals[i] != T(1)) {
        impl->diag = PERFLIBS_SPARSE_DIAG_NON_UNIT;
      }

      success = true;
      break;
    }
    if (!success) {
      return update_matrix_error(impl->error_handle, i + index_base);
    }
  }
  return PERFLIBS_STATUS_SUCCESS;
}
template perflibs_status_t
spmat_update_bsr<float>(perflibs_spmat_impl_t<float> *impl,
                        perflibs_int_t n_updates,
                        const perflibs_int_t *row_indx,
                        const perflibs_int_t *col_indx, const float *vals);
template perflibs_status_t
spmat_update_bsr<double>(perflibs_spmat_impl_t<double> *impl,
                         perflibs_int_t n_updates,
                         const perflibs_int_t *row_indx,
                         const perflibs_int_t *col_indx, const double *vals);
template perflibs_status_t spmat_update_bsr<std::complex<float>>(
    perflibs_spmat_impl_t<std::complex<float>> *impl, perflibs_int_t n_updates,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_indx,
    const std::complex<float> *vals);
template perflibs_status_t spmat_update_bsr<std::complex<double>>(
    perflibs_spmat_impl_t<std::complex<double>> *impl, perflibs_int_t n_updates,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_indx,
    const std::complex<double> *vals);

} // namespace perflibs::sparse
