/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "sell_c_sigma.hpp"
#include "compressed_sparse_rows.hpp"
#include "types.hpp"
#include <algorithm>
#include <inttypes.h>

extern "C" void
s_spmv_scs_kernel_neon_C4(int64_t nchunks, const float *vals,
                          const perflibs_int_t *col_indx, const float *x,
                          float *y, const int64_t *cs, const int64_t *cl,
                          const int64_t *row_permd2in, int64_t col_indx_bytes,
                          const int64_t *col_indx_min, float alpha, float beta);

// Neon kernels
extern "C" decltype(s_spmv_scs_kernel_neon_C4) s_spmv_scs_kernel_neon_C8;
extern "C" decltype(s_spmv_scs_kernel_neon_C4) s_spmv_scs_kernel_neon_C12;
extern "C" decltype(s_spmv_scs_kernel_neon_C4) s_spmv_scs_kernel_neon_C16;

extern "C" decltype(s_spmv_scs_kernel_neon_C4) s_spmv_scs_kernel_neon_C4_sig0;
extern "C" decltype(s_spmv_scs_kernel_neon_C4) s_spmv_scs_kernel_neon_C8_sig0;
extern "C" decltype(s_spmv_scs_kernel_neon_C4) s_spmv_scs_kernel_neon_C12_sig0;
extern "C" decltype(s_spmv_scs_kernel_neon_C4) s_spmv_scs_kernel_neon_C16_sig0;

extern "C" void d_spmv_scs_kernel_neon_C2(
    int64_t nchunks, const double *vals, const perflibs_int_t *col_indx,
    const double *x, double *y, const int64_t *cs, const int64_t *cl,
    const int64_t *row_permd2in, int64_t col_indx_bytes,
    const int64_t *col_indx_min, double alpha, double beta);

// Neon kernels
extern "C" decltype(d_spmv_scs_kernel_neon_C2) d_spmv_scs_kernel_neon_C4;
extern "C" decltype(d_spmv_scs_kernel_neon_C2) d_spmv_scs_kernel_neon_C6;
extern "C" decltype(d_spmv_scs_kernel_neon_C2) d_spmv_scs_kernel_neon_C8;
extern "C" decltype(d_spmv_scs_kernel_neon_C2) d_spmv_scs_kernel_neon_C10;
extern "C" decltype(d_spmv_scs_kernel_neon_C2) d_spmv_scs_kernel_neon_C12;
extern "C" decltype(d_spmv_scs_kernel_neon_C2) d_spmv_scs_kernel_neon_C14;
extern "C" decltype(d_spmv_scs_kernel_neon_C2) d_spmv_scs_kernel_neon_C16;

extern "C" decltype(d_spmv_scs_kernel_neon_C2) d_spmv_scs_kernel_neon_C2_sig0;
extern "C" decltype(d_spmv_scs_kernel_neon_C2) d_spmv_scs_kernel_neon_C4_sig0;
extern "C" decltype(d_spmv_scs_kernel_neon_C2) d_spmv_scs_kernel_neon_C6_sig0;
extern "C" decltype(d_spmv_scs_kernel_neon_C2) d_spmv_scs_kernel_neon_C8_sig0;
extern "C" decltype(d_spmv_scs_kernel_neon_C2) d_spmv_scs_kernel_neon_C10_sig0;
extern "C" decltype(d_spmv_scs_kernel_neon_C2) d_spmv_scs_kernel_neon_C12_sig0;
extern "C" decltype(d_spmv_scs_kernel_neon_C2) d_spmv_scs_kernel_neon_C14_sig0;
extern "C" decltype(d_spmv_scs_kernel_neon_C2) d_spmv_scs_kernel_neon_C16_sig0;

namespace perflibs::sparse {

static perflibs_int_t selected_bytes = 0;
void set_selected_bytes(perflibs_int_t b) { selected_bytes = b; }

// For Arm64EC there are no valid optimized kernels we can use for any datatype
#if defined(_M_ARM64EC) || defined(__arm64ec__)
template <typename T>
const std::vector<std::pair<int, scs_spmv_kernels<T>>> &
scs_get_valid_C_default(T) {
  using cval_t = std::pair<int, scs_spmv_kernels<T>>;
  static const std::vector<cval_t> cvals{
      {16, {nullptr, nullptr}}, {14, {nullptr, nullptr}},
      {12, {nullptr, nullptr}}, {10, {nullptr, nullptr}},
      {8, {nullptr, nullptr}},  {6, {nullptr, nullptr}},
      {4, {nullptr, nullptr}},  {2, {nullptr, nullptr}}};
  return cvals;
}
template const std::vector<std::pair<int, scs_spmv_kernels<float>>> &
scs_get_valid_C_default(float);
template const std::vector<std::pair<int, scs_spmv_kernels<double>>> &
scs_get_valid_C_default(double);
template const std::vector<
    std::pair<int, scs_spmv_kernels<std::complex<float>>>> &
    scs_get_valid_C_default(std::complex<float>);
template const std::vector<
    std::pair<int, scs_spmv_kernels<std::complex<double>>>> &
    scs_get_valid_C_default(std::complex<double>);

#else

template <>
const std::vector<std::pair<int, scs_spmv_kernels<float>>> &
scs_get_valid_C_default(float) {
  static std::vector<std::pair<int, scs_spmv_kernels<float>>> cvals{
      {16, {s_spmv_scs_kernel_neon_C16, s_spmv_scs_kernel_neon_C16_sig0}},
      {12, {s_spmv_scs_kernel_neon_C12, s_spmv_scs_kernel_neon_C12_sig0}},
      {8, {s_spmv_scs_kernel_neon_C8, s_spmv_scs_kernel_neon_C8_sig0}},
      {4, {s_spmv_scs_kernel_neon_C4, s_spmv_scs_kernel_neon_C4_sig0}}};
  return cvals;
}

template <>
const std::vector<std::pair<int, scs_spmv_kernels<double>>> &
scs_get_valid_C_default(double) {
  static std::vector<std::pair<int, scs_spmv_kernels<double>>> cvals{
      {16, {d_spmv_scs_kernel_neon_C16, d_spmv_scs_kernel_neon_C16_sig0}},
      {14, {d_spmv_scs_kernel_neon_C14, d_spmv_scs_kernel_neon_C14_sig0}},
      {12, {d_spmv_scs_kernel_neon_C12, d_spmv_scs_kernel_neon_C12_sig0}},
      {10, {d_spmv_scs_kernel_neon_C10, d_spmv_scs_kernel_neon_C10_sig0}},
      {8, {d_spmv_scs_kernel_neon_C8, d_spmv_scs_kernel_neon_C8_sig0}},
      {6, {d_spmv_scs_kernel_neon_C6, d_spmv_scs_kernel_neon_C6_sig0}},
      {4, {d_spmv_scs_kernel_neon_C4, d_spmv_scs_kernel_neon_C4_sig0}},
      {2, {d_spmv_scs_kernel_neon_C2, d_spmv_scs_kernel_neon_C2_sig0}},
  };
  return cvals;
}

template <>
const std::vector<std::pair<int, scs_spmv_kernels<std::complex<float>>>> &
scs_get_valid_C_default(std::complex<float>) {
  static std::vector<std::pair<int, scs_spmv_kernels<std::complex<float>>>>
      cvals{{16, {nullptr, nullptr}}, {14, {nullptr, nullptr}},
            {12, {nullptr, nullptr}}, {10, {nullptr, nullptr}},
            {8, {nullptr, nullptr}},  {6, {nullptr, nullptr}},
            {4, {nullptr, nullptr}},  {2, {nullptr, nullptr}}};
  return cvals;
}

template <>
const std::vector<std::pair<int, scs_spmv_kernels<std::complex<double>>>> &
scs_get_valid_C_default(std::complex<double>) {
  static std::vector<std::pair<int, scs_spmv_kernels<std::complex<double>>>>
      cvals{{16, {nullptr, nullptr}}, {14, {nullptr, nullptr}},
            {12, {nullptr, nullptr}}, {10, {nullptr, nullptr}},
            {8, {nullptr, nullptr}},  {6, {nullptr, nullptr}},
            {4, {nullptr, nullptr}},  {2, {nullptr, nullptr}}};
  return cvals;
}

#endif

/**
 * Given CSR input arrays create an perflibs_scs object that is populated with
 * all of the auxiliary arrays such as chunk starting points, chunk row lengths
 * and permutation arrays, but contains none of the length-nnz arrays. Copying
 * these vectors should be done after a call to csr2scs via a call to
 * perflibs_scs::gen_data_vectors and should be performed as few times as
 * possible due to the cost.
 */
template <typename T>
perflibs_scs<T>
csr2scs(enum sparse_hint_value_internal trans, perflibs_int_t m,
        perflibs_int_t n, const T *vals, const perflibs_int_t *col_indx,
        const perflibs_int_t *row_ptr, perflibs_int_t C, perflibs_int_t sigma,
        int nthreads, scs_spmv_kernels<T> kernels) {

  assert(sigma % C == 0 || sigma == 0);

  auto rows = m;
  auto cols = n;

  perflibs_csc<T> csr_trans;
  // First of all transpose the CSR input if needed - return a CSC structure
  // and just set pointers to the new data.
  if (trans == PERFLIBS_OPERATION_TRANS ||
      trans == PERFLIBS_OPERATION_CONJTRANS) {
    csr_trans = csr2csc(trans, m, n, vals, col_indx, row_ptr);
    vals = csr_trans.vals.data();
    col_indx = csr_trans.row_indx.data();
    row_ptr = csr_trans.col_ptr.data();
    rows = n;
    cols = m;
  }

  // Create the thing to return
  perflibs_scs<T> scs_ret(rows, cols, C, sigma, nthreads, kernels);
  scs_ret.optimized_op = trans;

  bool do_sort = true;
  if (sigma == 0) {
    sigma = C;
    do_sort = false;
  }

  // How many chunks in sigma rows?
  const auto sigma_chunks = sigma / C;

  // How many sigma_chunks are there in total?
  const perflibs_int_t nschunks = iround_div(rows, sigma);

  // A vector of vector of pairs. Pairs are (key,values) where keys are row
  // lengths and values are row pointers - each element of the vector outer
  // represents sigma rows
  std::vector<std::vector<std::pair<perflibs_int_t, perflibs_int_t>>> schunks(
      nschunks, std::vector<std::pair<perflibs_int_t, perflibs_int_t>>(sigma));

  for (perflibs_int_t i = 0; i < rows; i++) {
    auto rl = perflibs_int_t(row_ptr[i + 1] - row_ptr[i]);
    schunks[i / sigma][i % sigma] = std::make_pair(rl, i);
  }

  // Sort the nschunks inner rows
  if (do_sort) {
    for (perflibs_int_t i = 0; i < nschunks; i++) {
      std::sort(schunks[i].begin(), schunks[i].end(), std::greater<>{});
    }
  }

  // Store the width of each chunk
  for (perflibs_int_t i = 0; i < scs_ret.nC; i++) {
    auto schunk_id = i / sigma_chunks; // Identify the sigma chunk
    auto row_id = (i * C) % sigma; // Identify the start of the required chunk
                                   // within the sigma chunk
    if (do_sort) {
      scs_ret.cl[i] = schunks[schunk_id][row_id].first;
    } else { // If we haven't sorted then the largest row length is not
             // guaranteed to be in the first position in the chunk, so do a
             // search. If we haven't sorted then sigma_chunks = 1 and nC =
             // nschunks, so just iterate over the inner vector, which
             // represents a chunks and a sigma_chunk.
      perflibs_int_t max_rl = 0;
      for (auto it = schunks[schunk_id].cbegin();
           it != schunks[schunk_id].cend(); ++it) {
        if (it->first > max_rl) {
          max_rl = it->first;
        }
      }
      scs_ret.cl[i] = max_rl;
    }
  }

  // Store the offset of each chunk
  scs_ret.cs[0] = 0;
  for (perflibs_int_t i = 1; i < scs_ret.nC; i++) {
    scs_ret.cs[i] = scs_ret.cs[i - 1] + scs_ret.cl[i - 1] * C;
  }

  auto index_base = row_ptr[0];

  // Create the space saving datastructures
  const auto perflibs_type_range = std::numeric_limits<perflibs_int_t>::max();
  std::vector<perflibs_int_t> col_indx_min(rows, perflibs_type_range);
  std::vector<perflibs_int_t> col_indx_max(rows);
#pragma omp parallel for
  for (perflibs_int_t i = 0; i < rows; i++) {
    // Work out the column index range per row
    for (perflibs_int_t j = row_ptr[i] - index_base;
         j < row_ptr[i + 1] - index_base; j++) {
      if (col_indx[j] - index_base < col_indx_min[i]) {
        col_indx_min[i] = col_indx[j] - index_base;
      }
      if (col_indx[j] - index_base > col_indx_max[i]) {
        col_indx_max[i] = col_indx[j] - index_base;
      }
    }
  }

  // Find the minimum col_indx value per chunk and set the input to permuted row
  // mapping vectors
  perflibs_int_t range = 0;
#pragma omp parallel for reduction(max : range)
  for (auto ic = 0; ic < scs_ret.nC; ic++) {
    auto schunk_id = ic / sigma_chunks; // Identify the sigma chunk
    auto row_id = (ic * C) % sigma; // Identify the start of the required chunk
                                    // within the sigma chunk

    // Identify the current chunk
    auto chunk_nrows = std::min<perflibs_int_t>(C, rows - ic * C);
    const auto cur_chunk = &schunks[schunk_id][row_id];

    perflibs_int_t chunk_col_indx_min = perflibs_type_range;
    perflibs_int_t chunk_col_indx_max = 0;
    scs_ret.chunk_min_nnz[ic] = 0;
    // Find the minimum & maximum col_indx values in this chunk
    for (auto ir = 0; ir < chunk_nrows;
         ir++) { // Iterate over each row in the current chunk
      auto input_row_num = cur_chunk[ir].second;
      scs_ret.chunk_min_nnz[ic] += cur_chunk[ir].first;
      if (col_indx_min[input_row_num] < chunk_col_indx_min) {
        chunk_col_indx_min = col_indx_min[input_row_num];
      }
      if (col_indx_max[input_row_num] > chunk_col_indx_max) {
        chunk_col_indx_max = col_indx_max[input_row_num];
      }
    }

    // Figure out the global max range
    auto chunk_range = chunk_col_indx_max - chunk_col_indx_min;
    if (chunk_range > range) {
      range = chunk_range;
    }
    // Store the column index minimum values
    scs_ret.col_indx_min[ic] = chunk_col_indx_min;

    for (auto ir = 0; ir < chunk_nrows;
         ir++) { // Iterate over each row in the current chunk
      // Create a mapping between the input and permuted row numbers
      auto input_row_num = cur_chunk[ir].second;
      auto permd_row_num = ic * C + ir;

      // Note that we need to process empty rows as well here, because the
      // corresponding output element will still need to be scaled by beta. So
      // the mapping below is required for all rows.
      scs_ret.row_in2permd[input_row_num] = permd_row_num;
      scs_ret.row_permd2in[permd_row_num] = input_row_num;
      if (!do_sort) {
        assert(permd_row_num == input_row_num);
      }
    } // end for chunk_nrows
  }

  // We now know the largest range required to store col_indx as offsets to
  // col_indx_min values. Work out how many bytes this is.
  auto bits = 0;
  while (range) {
    range >>= 1;
    bits++;
  }
  auto bytes = (bits / 8) + 1;
  if (bytes == 3) {
    bytes = 4;
  } else if (bytes > 4) {
    bytes = 8;
  }
  scs_ret.col_indx_bytes = selected_bytes >= bytes ? selected_bytes : bytes;

  // Decide what to do with the final chunk
  scs_ret.n_full_chunks = scs_ret.row_permd2in[scs_ret.nC * C - 1] < 0
                              ? scs_ret.nC - 1
                              : scs_ret.nC;

  auto padding = 0;
  while (padding < C &&
         scs_ret.row_permd2in[scs_ret.nC * C - padding - 1] < 0) {
    padding++;
  }
  scs_ret.ntidyup = padding == 0 ? 0 : C - padding;
  assert(scs_ret.ntidyup >= 0);

  return scs_ret;
};

template perflibs_scs<float>
csr2scs<float>(enum sparse_hint_value_internal trans, perflibs_int_t m,
               perflibs_int_t n, const float *vals,
               const perflibs_int_t *col_indx, const perflibs_int_t *row_ptr,
               perflibs_int_t C, perflibs_int_t sigma, int nthreads,
               scs_spmv_kernels<float> kernels);

template perflibs_scs<double>
csr2scs<double>(enum sparse_hint_value_internal trans, perflibs_int_t m,
                perflibs_int_t n, const double *vals,
                const perflibs_int_t *col_indx, const perflibs_int_t *row_ptr,
                perflibs_int_t C, perflibs_int_t sigma, int nthreads,
                scs_spmv_kernels<double> kernels);

template perflibs_scs<std::complex<float>> csr2scs<std::complex<float>>(
    enum sparse_hint_value_internal trans, perflibs_int_t m, perflibs_int_t n,
    const std::complex<float> *vals, const perflibs_int_t *col_indx,
    const perflibs_int_t *row_ptr, perflibs_int_t C, perflibs_int_t sigma,
    int nthreads, scs_spmv_kernels<std::complex<float>> kernels);

template perflibs_scs<std::complex<double>> csr2scs<std::complex<double>>(
    enum sparse_hint_value_internal trans, perflibs_int_t m, perflibs_int_t n,
    const std::complex<double> *vals, const perflibs_int_t *col_indx,
    const perflibs_int_t *row_ptr, perflibs_int_t C, perflibs_int_t sigma,
    int nthreads, scs_spmv_kernels<std::complex<double>> kernels);

/**
 * Return beta as defined in the SCS paper.
 * @param nnz_min The number of nonzeroes in the input (i.e. row_ptr[m] -
 * row_ptr[0])
 */
template <typename T>
double perflibs::sparse::perflibs_scs<T>::get_beta(perflibs_int_t nnz_min) {
  // The number of nonzeroes we're parallelizing over (full chunks)
  int no_tidyup = ntidyup == 0;
  const perflibs_int_t nnz_scs_par = cs[nC - 1] + no_tidyup * cl[nC - 1] * C;
  // The number of nonzeroes we're processing in SCS
  const perflibs_int_t nnz_scs = nnz_scs_par + cl[nC - 1] * ntidyup;
  // The value of beta defined in the paper
  double scs_beta = (double)nnz_min / nnz_scs;
  return scs_beta;
};

template double
perflibs::sparse::perflibs_scs<float>::get_beta(perflibs_int_t nnz_min);
template double
perflibs::sparse::perflibs_scs<double>::get_beta(perflibs_int_t nnz_min);
template double perflibs::sparse::perflibs_scs<std::complex<float>>::get_beta(
    perflibs_int_t nnz_min);
template double perflibs::sparse::perflibs_scs<std::complex<double>>::get_beta(
    perflibs_int_t nnz_min);

template <typename T>
perflibs::sparse::scs_beta_threads_t
perflibs::sparse::perflibs_scs<T>::get_beta_threads(perflibs_int_t nnz_min) {
  if (nthreads == 1) {
    auto val = get_beta(nnz_min);
    return {val, val};
  }
  // If the division of chunks over threads has not been performed, do that now
  if (nchunks_thread[0] == 0) {
    gen_parallel_decomp(perflibs::sparse::perflibs_scs_strategy_t::
                            perflibs_scs_parallel_nnz_division);
  }

  double min_beta = 1.;
  double max_beta = 0.;
  // Get the minimum ratio of the number of non-zeros in a thread and the number
  // of non-zeros in the csr matrix chunk Deal with the threads (except the last
  // one) using just the chunk start inds
  perflibs_int_t t = 0;
  for (; t < nthreads - 1 && nC > 1; ++t) {
    // Get the number of non-zeros in this thread
    auto start_chunk = chk_offset[t];
    auto end_chunk = chk_offset[t + 1];
    // It may be the case that there are threads without any chunks assigned
    if (end_chunk == nC)
      break;

    assert(end_chunk >= start_chunk);
    auto t_nnz = cs[end_chunk] - cs[start_chunk];
    assert(t_nnz >= 0);
    assert(thread_min_nnz[t] <= t_nnz);
    if (thread_min_nnz[t] > 0) {
      min_beta = std::min<double>(double(thread_min_nnz[t]) / t_nnz, min_beta);
      max_beta = std::max<double>(double(thread_min_nnz[t]) / t_nnz, max_beta);
    }
  }

  // Deal with the last thread to have chunks assigned separately
  // Get the number of non-zeros in the matrix
  assert(t < nthreads);
  // If there is more than one chunk, make sure that the chk_offset for the last
  // thread with data in is equal to the last chunk
  assert(t == nthreads - 1 || nC == 1 ? true : chk_offset[t + 1] == nC);
  int no_tidyup = ntidyup == 0;
  const perflibs_int_t nnz_scs_par = cs[nC - 1] + no_tidyup * cl[nC - 1] * C;
  const perflibs_int_t nnz_scs = nnz_scs_par + cl[nC - 1] * ntidyup;
  auto last_t_nnz = nnz_scs - cs[chk_offset[t]];
  assert(last_t_nnz >= 0);
  assert(thread_min_nnz[t] <= last_t_nnz);

  if (thread_min_nnz[t] > 0) {
    min_beta =
        std::min<double>((double)thread_min_nnz[t] / last_t_nnz, min_beta);
    max_beta =
        std::max<double>((double)thread_min_nnz[t] / last_t_nnz, max_beta);
  } else {
    min_beta = 0.;
  }

  return {min_beta, max_beta};
}

template perflibs::sparse::scs_beta_threads_t
perflibs::sparse::perflibs_scs<float>::get_beta_threads(perflibs_int_t nnz_min);
template perflibs::sparse::scs_beta_threads_t
perflibs::sparse::perflibs_scs<double>::get_beta_threads(
    perflibs_int_t nnz_min);
template perflibs::sparse::scs_beta_threads_t
perflibs::sparse::perflibs_scs<std::complex<float>>::get_beta_threads(
    perflibs_int_t nnz_min);
template perflibs::sparse::scs_beta_threads_t
perflibs::sparse::perflibs_scs<std::complex<double>>::get_beta_threads(
    perflibs_int_t nnz_min);

template <typename T>
void perflibs::sparse::perflibs_scs<T>::gen_parallel_decomp(
    perflibs_scs_strategy_t strat) {

  if (strat == perflibs_scs_parallel_chunk_division) {
    // STRATEGY 1
    // Simply divide up the chunks evenly - this works well when chunks have
    // close to equal row lengths
    const int nchunks = (n_full_chunks + nthreads - 1) / nthreads;
    for (int i = 0; i < nthreads; i++) {
      nchunks_thread[i] = nchunks;
      chk_offset[i] = i * nchunks;
      if (chk_offset[i] + nchunks > n_full_chunks) {
        nchunks_thread[i] =
            std::max<perflibs_int_t>(n_full_chunks - chk_offset[i], 0);
      }
      if (i > 0 && nchunks_thread[i] > 0) {
        thread_min_nnz[i - 1] = 0;
        for (auto c = chk_offset[i - 1]; c < chk_offset[i]; ++c) {
          thread_min_nnz[i - 1] += chunk_min_nnz[c];
        }
      }
    }

    thread_min_nnz[nthreads - 1] = 0;
    for (auto c = chk_offset[nthreads - 1]; c < nC; ++c) {
      thread_min_nnz[nthreads - 1] += chunk_min_nnz[c];
    }
  } else if (strat == perflibs_scs_parallel_nnz_division) {

    // We accumulate into this vector, so clear it first since it may have been
    // populated with a different strategy beforehand.
    for (int i = 0; i < nthreads; i++) {
      nchunks_thread[i] = 0;
      thread_min_nnz[i] = 0;
    }

    // STRATEGY 2
    // Now try to see if it is fairer to distribute based on dividing up
    // nnz_scs_par - this works well when chunks don't have close to equal row
    // lengths. The number of nonzeroes we're parallelizing over (full chunks)
    int no_tidyup = ntidyup == 0;
    const perflibs_int_t nnz_scs_par = cs[nC - 1] + no_tidyup * cl[nC - 1] * C;
    const perflibs_int_t vals_per_thread =
        std::max<int64_t>((nnz_scs_par - nthreads + 1) / nthreads, 1);
    perflibs_int_t vals_next_thread = vals_per_thread;
    perflibs_int_t vals_this_thread = 0;
    perflibs_int_t this_thread = 0;
    chk_offset[0] = 0;
    for (perflibs_int_t i = 0; i < n_full_chunks; i++) {

      // If we've reached the next threshold and the next chunks is not empty
      // set the offset and move on to the next thread
      if (vals_this_thread >= vals_next_thread && cl[i] > 0) {
        if (this_thread > 0) {
          chk_offset[this_thread] =
              nchunks_thread[this_thread - 1] + chk_offset[this_thread - 1];
        }
        vals_next_thread += nchunks_thread[this_thread] * C + vals_per_thread;
        this_thread++;
      }

      assert(this_thread < nthreads);
      vals_this_thread += cl[i] * C;
      nchunks_thread[this_thread]++;
      thread_min_nnz[this_thread] += chunk_min_nnz[i];
    }
    //  Set the offset for the final threads
    chk_offset[this_thread] =
        this_thread == 0
            ? 0
            : nchunks_thread[this_thread - 1] + chk_offset[this_thread - 1];
    // nchunks_thread[this_thread] = nC - chk_offset[this_thread];
    thread_min_nnz[this_thread] = 0;
    for (auto c = chk_offset[this_thread]; c < nC; ++c) {
      thread_min_nnz[this_thread] += chunk_min_nnz[c];
    }

    assert(chk_offset[this_thread] + nchunks_thread[this_thread] ==
           n_full_chunks);

    // If there are some threads without chunks assigned
    for (++this_thread; this_thread < nthreads; ++this_thread) {
      chk_offset[this_thread] = nC;
      nchunks_thread[this_thread] = 0;
      thread_min_nnz[this_thread] = 0;
    }
  }
};
template void perflibs::sparse::perflibs_scs<float>::gen_parallel_decomp(
    perflibs_scs_strategy_t strat);
template void perflibs::sparse::perflibs_scs<double>::gen_parallel_decomp(
    perflibs_scs_strategy_t strat);
template void
perflibs::sparse::perflibs_scs<std::complex<float>>::gen_parallel_decomp(
    perflibs_scs_strategy_t strat);
template void
perflibs::sparse::perflibs_scs<std::complex<double>>::gen_parallel_decomp(
    perflibs_scs_strategy_t strat);

template <typename T>
static void write_offset(T *ptr, int width, size_t ofs, int64_t data) {
  switch (width) {
  case 1:
    ((int8_t *)ptr)[ofs] = data;
    break;
  case 2:
    ((int16_t *)ptr)[ofs] = data;
    break;
  case 4:
    ((int32_t *)ptr)[ofs] = data;
    break;
  case 8:
    ((int64_t *)ptr)[ofs] = data;
    break;
  default:
    assert(false);
  }
}

template <typename T>
void perflibs::sparse::perflibs_scs<T>::gen_data_vectors(
    enum sparse_hint_value_internal trans, const perflibs_int_t *row_ptr,
    const perflibs_int_t *col_indx, const T *vals) {

  perflibs_csc<T> csr_trans;
  // First of all transpose the CSR input if needed - return a CSC structure
  // and just set pointers to the new data.
  if (trans == PERFLIBS_OPERATION_TRANS ||
      trans == PERFLIBS_OPERATION_CONJTRANS) {
    // rows and cols have already been swapped in the prior call to csr2scs
    csr_trans = csr2csc(trans, n, m, vals, col_indx, row_ptr);
    vals = csr_trans.vals.data();
    col_indx = csr_trans.row_indx.data();
    row_ptr = csr_trans.col_ptr.data();
  }

  std::vector<T> conjvals;
  if (trans == PERFLIBS_OPERATION_CONJNOTRANS) {
    auto nnz = row_ptr[m] - row_ptr[0];
    conjvals.resize(nnz);
    for (int i = 0; i < nnz; i++) {
      conjvals[i] = perflibs::sparse::conj(vals[i]);
    }
    vals = conjvals.data();
  }

  // Create the new data vectors
  auto new_data_size = cs[nC - 1] + cl[nC - 1] * C;
  this->vals.resize(new_data_size);
  this->vals.shrink_to_fit();
  col_indx_offsets.resize(new_data_size);
  col_indx_offsets.shrink_to_fit();
  char *offsets_proxy = reinterpret_cast<char *>(col_indx_offsets.data());

  auto index_base = row_ptr[0];

  auto nt = this->nthreads;

// Copy the contents of vals, col_indx into new vectors in the sorted order
// This loop copies the same parallelization as the optimized execute functions
// to ensure that the thread which first touches an element of col_indx_offsets
// and vals members of the SCS object will be the same thread to access that
// element in the execute call, thus ensuring first-touch memory allocation will
// match what is required.
#pragma omp parallel for schedule(static) num_threads(nt)
  for (auto me = 0; me < nt; me++) {
    auto nchunks = this->nchunks_thread[me];
    auto chk_offset = this->chk_offset[me];

    assert(chk_offset + nchunks <= this->nC);
    for (auto ic = chk_offset; ic < chk_offset + nchunks; ic++) {
      auto chunk_ncols = cl[ic];
      for (auto ir = 0; ir < C;
           ir++) { // Iterate over each row in the current chunk
        auto permd_row_num = ic * C + ir;
        auto input_row_num = row_permd2in[permd_row_num];

        auto in_row_start = row_ptr[input_row_num] - index_base;
        auto in_row_end = row_ptr[input_row_num + 1] - index_base;
        auto in_row_width = in_row_end - in_row_start;

        for (perflibs_int_t j = 0; j < chunk_ncols; j++) {
          // The arrays vals, col_indx are copied-to column-major within chunks
          // (see diagram in paper)
          auto out_j = cs[ic] + j * C + ir;
          this->vals[out_j] = j < in_row_width ? vals[in_row_start + j] : 0;

          // Store the column index offsets - used in optimized functions
          perflibs_int_t offset =
              j < in_row_width
                  ? col_indx[in_row_start + j] - index_base - col_indx_min[ic]
                  : 0;
          write_offset(offsets_proxy, col_indx_bytes, out_j, offset);
        }
      } // end for chunk_nrows
    }
  }

  if (nC > n_full_chunks) {
    auto ic = nC - 1;

    auto chunk_nrows = m - ic * C;
    auto chunk_ncols = cl[ic];

    for (perflibs_int_t ir = 0; ir < chunk_nrows;
         ir++) { // Iterate over each row in the current chunk
      auto permd_row_num = ic * C + ir;
      auto input_row_num = row_permd2in[permd_row_num];

      auto in_row_start = row_ptr[input_row_num] - index_base;
      auto in_row_end = row_ptr[input_row_num + 1] - index_base;
      auto in_row_width = in_row_end - in_row_start;

      for (perflibs_int_t j = 0; j < chunk_ncols; j++) {
        // The arrays vals, col_indx are copied-to column-major within chunks
        // (see diagram in paper)
        auto out_j = cs[ic] + j * C + ir;
        // Only copy the value if we're not past the last row and we're not in
        // the padding past the end of the row
        this->vals[out_j] = (j < in_row_width) ? vals[in_row_start + j] : 0;

        // Store the column index offsets - used in optimized functions
        perflibs_int_t offset =
            (j < in_row_width)
                ? col_indx[in_row_start + j] - index_base - col_indx_min[ic]
                : 0;
        write_offset(offsets_proxy, col_indx_bytes, out_j, offset);
      }
    }
    for (perflibs_int_t ir = chunk_nrows; ir < C;
         ir++) { // Iterate over padding rows
      for (perflibs_int_t j = 0; j < chunk_ncols; j++) {
        // The arrays vals, col_indx are copied-to column-major within chunks
        // (see diagram in paper)
        auto out_j = cs[ic] + j * C + ir;
        this->vals[out_j] = 0;
        write_offset(offsets_proxy, col_indx_bytes, out_j, 0);
      }
    } // end for chunk_nrows
  }
};
template void perflibs_scs<float>::gen_data_vectors(
    enum sparse_hint_value_internal trans, const perflibs_int_t *row_ptr,
    const perflibs_int_t *col_indx, const float *vals);
template void perflibs_scs<double>::gen_data_vectors(
    enum sparse_hint_value_internal trans, const perflibs_int_t *row_ptr,
    const perflibs_int_t *col_indx, const double *vals);
template void perflibs_scs<std::complex<float>>::gen_data_vectors(
    enum sparse_hint_value_internal trans, const perflibs_int_t *row_ptr,
    const perflibs_int_t *col_indx, const std::complex<float> *vals);
template void perflibs_scs<std::complex<double>>::gen_data_vectors(
    enum sparse_hint_value_internal trans, const perflibs_int_t *row_ptr,
    const perflibs_int_t *col_indx, const std::complex<double> *vals);

template <typename T, int C>
void spmv_scs(const perflibs_scs<T> &scs, const T *x, T *y, T alpha, T beta) {
  assert(scs.C == C);

#pragma omp parallel
  {
    T work[C];
#pragma omp for
    for (perflibs_int_t ci = 0; ci < scs.nC; ci++) {

      // Zero work vector
      for (perflibs_int_t cd = 0; cd < C; cd++) {
        work[cd] = T(0);
      }

      // Perform multiplication
      auto col_indx_min = scs.col_indx_min[ci];
      if (scs.col_indx_bytes == 1) {
        auto col_indx =
            reinterpret_cast<const int8_t *>(scs.col_indx_offsets.data());
        for (perflibs_int_t ri = 0; ri < scs.cl[ci]; ri++) {
          for (perflibs_int_t cd = 0; cd < C; cd++) {
            auto dat_index = scs.cs[ci] + ri * C + cd;
            work[cd] +=
                scs.vals[dat_index] * x[col_indx[dat_index] + col_indx_min];
          }
        }
      } else if (scs.col_indx_bytes == 2) {
        auto col_indx =
            reinterpret_cast<const int16_t *>(scs.col_indx_offsets.data());
        for (perflibs_int_t ri = 0; ri < scs.cl[ci]; ri++) {
          for (perflibs_int_t cd = 0; cd < C; cd++) {
            auto dat_index = scs.cs[ci] + ri * C + cd;
            work[cd] +=
                scs.vals[dat_index] * x[col_indx[dat_index] + col_indx_min];
          }
        }
      } else if (scs.col_indx_bytes == 4) {
        auto col_indx =
            reinterpret_cast<const int32_t *>(scs.col_indx_offsets.data());
        for (perflibs_int_t ri = 0; ri < scs.cl[ci]; ri++) {
          for (perflibs_int_t cd = 0; cd < C; cd++) {
            auto dat_index = scs.cs[ci] + ri * C + cd;
            work[cd] +=
                scs.vals[dat_index] * x[col_indx[dat_index] + col_indx_min];
          }
        }
      } else if (scs.col_indx_bytes == 8) {
        auto col_indx =
            reinterpret_cast<const int64_t *>(scs.col_indx_offsets.data());
        for (perflibs_int_t ri = 0; ri < scs.cl[ci]; ri++) {
          for (perflibs_int_t cd = 0; cd < C; cd++) {
            auto dat_index = scs.cs[ci] + ri * C + cd;
            work[cd] +=
                scs.vals[dat_index] * x[col_indx[dat_index] + col_indx_min];
          }
        }
      }

      // Apply alpha and beta
      if (beta != T(0)) {
        for (perflibs_int_t cd = 0; cd < C; cd++) {
          auto row_index = ci * C + cd;
          if (scs.row_permd2in[row_index] > -1) {
            y[scs.row_permd2in[row_index]] =
                alpha * work[cd] + beta * y[scs.row_permd2in[row_index]];
          }
        }
      } else {
        for (perflibs_int_t cd = 0; cd < C; cd++) {
          auto row_index = ci * C + cd;
          if (scs.row_permd2in[row_index] > -1) {
            y[scs.row_permd2in[row_index]] = alpha * work[cd];
          }
        }
      }
    }
  }
};

// As we have no optimized kernels for Arm64EC we need to instantiate the
// non-optimized templates for float and double as well
#if defined(_M_ARM64EC) || defined(__arm64ec__)
template void spmv_scs<float, 16>(const perflibs_scs<float> &scs,
                                  const float *x, float *y, float alpha,
                                  float beta);
template void spmv_scs<double, 16>(const perflibs_scs<double> &scs,
                                   const double *x, double *y, double alpha,
                                   double beta);
template void spmv_scs<float, 14>(const perflibs_scs<float> &scs,
                                  const float *x, float *y, float alpha,
                                  float beta);
template void spmv_scs<double, 14>(const perflibs_scs<double> &scs,
                                   const double *x, double *y, double alpha,
                                   double beta);
template void spmv_scs<float, 12>(const perflibs_scs<float> &scs,
                                  const float *x, float *y, float alpha,
                                  float beta);
template void spmv_scs<double, 12>(const perflibs_scs<double> &scs,
                                   const double *x, double *y, double alpha,
                                   double beta);
template void spmv_scs<float, 10>(const perflibs_scs<float> &scs,
                                  const float *x, float *y, float alpha,
                                  float beta);
template void spmv_scs<double, 10>(const perflibs_scs<double> &scs,
                                   const double *x, double *y, double alpha,
                                   double beta);
template void spmv_scs<float, 8>(const perflibs_scs<float> &scs, const float *x,
                                 float *y, float alpha, float beta);
template void spmv_scs<double, 8>(const perflibs_scs<double> &scs,
                                  const double *x, double *y, double alpha,
                                  double beta);
template void spmv_scs<float, 6>(const perflibs_scs<float> &scs, const float *x,
                                 float *y, float alpha, float beta);
template void spmv_scs<double, 6>(const perflibs_scs<double> &scs,
                                  const double *x, double *y, double alpha,
                                  double beta);
template void spmv_scs<float, 4>(const perflibs_scs<float> &scs, const float *x,
                                 float *y, float alpha, float beta);
template void spmv_scs<double, 4>(const perflibs_scs<double> &scs,
                                  const double *x, double *y, double alpha,
                                  double beta);
template void spmv_scs<float, 2>(const perflibs_scs<float> &scs, const float *x,
                                 float *y, float alpha, float beta);
template void spmv_scs<double, 2>(const perflibs_scs<double> &scs,
                                  const double *x, double *y, double alpha,
                                  double beta);
#endif

template void spmv_scs<std::complex<float>, 16>(
    const perflibs_scs<std::complex<float>> &scs, const std::complex<float> *x,
    std::complex<float> *y, std::complex<float> alpha,
    std::complex<float> beta);
template void spmv_scs<std::complex<double>, 16>(
    const perflibs_scs<std::complex<double>> &scs,
    const std::complex<double> *x, std::complex<double> *y,
    std::complex<double> alpha, std::complex<double> beta);
template void spmv_scs<std::complex<float>, 14>(
    const perflibs_scs<std::complex<float>> &scs, const std::complex<float> *x,
    std::complex<float> *y, std::complex<float> alpha,
    std::complex<float> beta);
template void spmv_scs<std::complex<double>, 14>(
    const perflibs_scs<std::complex<double>> &scs,
    const std::complex<double> *x, std::complex<double> *y,
    std::complex<double> alpha, std::complex<double> beta);
template void spmv_scs<std::complex<float>, 12>(
    const perflibs_scs<std::complex<float>> &scs, const std::complex<float> *x,
    std::complex<float> *y, std::complex<float> alpha,
    std::complex<float> beta);
template void spmv_scs<std::complex<double>, 12>(
    const perflibs_scs<std::complex<double>> &scs,
    const std::complex<double> *x, std::complex<double> *y,
    std::complex<double> alpha, std::complex<double> beta);
template void spmv_scs<std::complex<float>, 10>(
    const perflibs_scs<std::complex<float>> &scs, const std::complex<float> *x,
    std::complex<float> *y, std::complex<float> alpha,
    std::complex<float> beta);
template void spmv_scs<std::complex<double>, 10>(
    const perflibs_scs<std::complex<double>> &scs,
    const std::complex<double> *x, std::complex<double> *y,
    std::complex<double> alpha, std::complex<double> beta);
template void spmv_scs<std::complex<float>, 8>(
    const perflibs_scs<std::complex<float>> &scs, const std::complex<float> *x,
    std::complex<float> *y, std::complex<float> alpha,
    std::complex<float> beta);
template void spmv_scs<std::complex<double>, 8>(
    const perflibs_scs<std::complex<double>> &scs,
    const std::complex<double> *x, std::complex<double> *y,
    std::complex<double> alpha, std::complex<double> beta);
template void spmv_scs<std::complex<float>, 6>(
    const perflibs_scs<std::complex<float>> &scs, const std::complex<float> *x,
    std::complex<float> *y, std::complex<float> alpha,
    std::complex<float> beta);
template void spmv_scs<std::complex<double>, 6>(
    const perflibs_scs<std::complex<double>> &scs,
    const std::complex<double> *x, std::complex<double> *y,
    std::complex<double> alpha, std::complex<double> beta);
template void spmv_scs<std::complex<float>, 4>(
    const perflibs_scs<std::complex<float>> &scs, const std::complex<float> *x,
    std::complex<float> *y, std::complex<float> alpha,
    std::complex<float> beta);
template void spmv_scs<std::complex<double>, 4>(
    const perflibs_scs<std::complex<double>> &scs,
    const std::complex<double> *x, std::complex<double> *y,
    std::complex<double> alpha, std::complex<double> beta);
template void spmv_scs<std::complex<float>, 2>(
    const perflibs_scs<std::complex<float>> &scs, const std::complex<float> *x,
    std::complex<float> *y, std::complex<float> alpha,
    std::complex<float> beta);
template void spmv_scs<std::complex<double>, 2>(
    const perflibs_scs<std::complex<double>> &scs,
    const std::complex<double> *x, std::complex<double> *y,
    std::complex<double> alpha, std::complex<double> beta);

template <typename T>
void perflibs::sparse::perflibs_scs<T>::print_structure() {
  auto read_offset = [](perflibs_int_t *ptr, perflibs_int_t width,
                        int64_t ofs) -> int64_t {
    switch (width) {
    case 1:
      return (int64_t)((int8_t *)ptr)[ofs];
    case 2:
      return (int64_t)((int16_t *)ptr)[ofs];
    case 4:
      return (int64_t)((int32_t *)ptr)[ofs];
    case 8:
      return (int64_t)((int64_t *)ptr)[ofs];
    default:
      assert(false);
      return int64_t{0};
    }
  };

  printf("Original matrix dims: %" PRId64 " by %" PRId64 "\n", (int64_t)m,
         (int64_t)n);
  printf("Chunk size %" PRId64 "\n", (int64_t)C);
  printf("Number of Chunks %" PRId64 "\n", (int64_t)nC);
  printf("Sigma %" PRId64 "\n", (int64_t)sigma);
  printf("col_indx_bytes = %" PRId64 "\n", (int64_t)col_indx_bytes);
  for (int64_t i = 0; i < nC; i++) {
    printf("----- Chunk %" PRId64 " (length = %" PRId64
           "; col_indx_min = %" PRId64 ")\n",
           i, cl[i], col_indx_min[i]);
    for (int64_t j = 0; j < C; j++) {
      printf("-- Row %" PRId64 " (input row %" PRId64 ")\n- ", j,
             row_permd2in[i * C + j]);
      if (row_permd2in[i * C + j] > -1) {
        for (int64_t k = 0; k < cl[i]; k++) {
          printf("(%" PRId64 ")%" PRId64 ", ", k,
                 read_offset(&col_indx_offsets[0], col_indx_bytes,
                             cs[i] + k * C + j) +
                     col_indx_min[i]);
        }
      } else {
        printf(" padding");
      }
      printf("\n");
    }
  }
}
template void perflibs::sparse::perflibs_scs<float>::print_structure();
template void perflibs::sparse::perflibs_scs<double>::print_structure();
template void
perflibs::sparse::perflibs_scs<std::complex<float>>::print_structure();
template void
perflibs::sparse::perflibs_scs<std::complex<double>>::print_structure();

template <typename T>
perflibs_status_t
spmat_update_scs(perflibs_spmat_impl_t<T> *impl, perflibs_int_t n_updates,
                 const perflibs_int_t *row_indx, const perflibs_int_t *col_indx,
                 const T *vals) {
  auto nrows = impl->m;
  auto ncols = impl->n;
  auto index_base = impl->index_base;
  auto is_unit = impl->diag == PERFLIBS_SPARSE_DIAG_UNIT;

  for (auto i = 0; i < n_updates; i++) {
    auto row_num = row_indx[i] - index_base;
    auto col_num = col_indx[i] - index_base;
    if (row_num < 0 || row_num > nrows - 1) {
      return update_matrix_error(impl->error_handle, i + index_base);
    } else if (col_num < 0 || col_num > ncols - 1) {
      return update_matrix_error(impl->error_handle, i + index_base);
    } else {
      auto stored_row = row_num;
      auto stored_col = col_num;
      auto stored_val = vals[i];
      if (impl->scs.optimized_op == PERFLIBS_OPERATION_TRANS ||
          impl->scs.optimized_op == PERFLIBS_OPERATION_CONJTRANS) {
        std::swap(stored_row, stored_col);
      }
      if (impl->scs.optimized_op == PERFLIBS_OPERATION_CONJNOTRANS ||
          impl->scs.optimized_op == PERFLIBS_OPERATION_CONJTRANS) {
        stored_val = perflibs::sparse::conj(stored_val);
      }

      auto C = impl->scs.C;
      auto pi = impl->scs.row_in2permd[stored_row];
      auto cid = pi / C;
      auto r_cid = pi % C;
      auto offset = impl->scs.cs[cid] + r_cid;
      bool success = false;
      for (auto j = 0; j < impl->scs.cl[cid]; j++) {
        if (impl->scs.col_indx_bytes == 1) {
          auto col_indx_off = reinterpret_cast<const int8_t *>(
              impl->scs.col_indx_offsets.data());
          if (col_indx_off[offset + j * C] + impl->scs.col_indx_min[cid] ==
              stored_col) {
            impl->scs.vals[offset + j * C] = stored_val;
            success = true;
            break;
          }
        } else if (impl->scs.col_indx_bytes == 2) {
          auto col_indx_off = reinterpret_cast<const int16_t *>(
              impl->scs.col_indx_offsets.data());
          if (col_indx_off[offset + j * C] + impl->scs.col_indx_min[cid] ==
              stored_col) {
            impl->scs.vals[offset + j * C] = stored_val;
            success = true;
            break;
          }
        } else if (impl->scs.col_indx_bytes == 4) {
          auto col_indx_off = reinterpret_cast<const int32_t *>(
              impl->scs.col_indx_offsets.data());
          if (col_indx_off[offset + j * C] + impl->scs.col_indx_min[cid] ==
              stored_col) {
            impl->scs.vals[offset + j * C] = stored_val;
            success = true;
            break;
          }
        } else if (impl->scs.col_indx_bytes == 8) {
          auto col_indx_off = reinterpret_cast<const int64_t *>(
              impl->scs.col_indx_offsets.data());
          if (col_indx_off[offset + j * C] + impl->scs.col_indx_min[cid] ==
              stored_col) {
            impl->scs.vals[offset + j * C] = stored_val;
            success = true;
            break;
          }
        }
      }
      if (!success) {
        return update_matrix_error(impl->error_handle, i + index_base);
      } else if (row_num == col_num && is_unit && vals[i] != T(1)) {
        impl->diag = PERFLIBS_SPARSE_DIAG_NON_UNIT;
      }
    }
  }
  return PERFLIBS_STATUS_SUCCESS;
}
template perflibs_status_t
spmat_update_scs<float>(perflibs_spmat_impl_t<float> *impl,
                        perflibs_int_t n_updates,
                        const perflibs_int_t *row_indx,
                        const perflibs_int_t *col_indx, const float *vals);
template perflibs_status_t
spmat_update_scs<double>(perflibs_spmat_impl_t<double> *impl,
                         perflibs_int_t n_updates,
                         const perflibs_int_t *row_indx,
                         const perflibs_int_t *col_indx, const double *vals);
template perflibs_status_t spmat_update_scs<std::complex<float>>(
    perflibs_spmat_impl_t<std::complex<float>> *impl, perflibs_int_t n_updates,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_indx,
    const std::complex<float> *vals);
template perflibs_status_t spmat_update_scs<std::complex<double>>(
    perflibs_spmat_impl_t<std::complex<double>> *impl, perflibs_int_t n_updates,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_indx,
    const std::complex<double> *vals);

} // end namespace perflibs::sparse
