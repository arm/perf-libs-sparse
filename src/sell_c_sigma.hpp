/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#pragma once

/* Functions for SELL-C-σ (SCS) */
#include "pod_vector.hpp"
#include "util.hpp"

#include <cstdint>
#include <utility>
#include <vector>

template <typename T> struct perflibs_spmat_impl_t;

namespace perflibs::sparse {

enum perflibs_scs_strategy_t {
  perflibs_scs_parallel_chunk_division,
  perflibs_scs_parallel_nnz_division
};

template <typename T>
using scs_spmv_kernel_t = void (*)(
    int64_t nchunks, const T *vals, const perflibs_int_t *col_indx, const T *x,
    T *y, const int64_t *cs, const int64_t *cl, const int64_t *row_permd2in,
    int64_t col_indx_bytes, const int64_t *col_indx_min, T alpha, T beta);

template <typename T> struct scs_spmv_kernels {
  scs_spmv_kernel_t<T> spmv_kernel = nullptr;
  scs_spmv_kernel_t<T> spmv_kernel_sig0 = nullptr;
};

struct scs_beta_threads_t {
  double min = 0.0;
  double max = 0.0;
};

template <typename T> struct perflibs_scs {

  perflibs_int_t m = 0;
  perflibs_int_t n = 0;
  // SCS stores the matrix operation applied during optimization, so its
  // execution kernel always performs an effective NOTRANS operation.
  sparse_hint_value_internal optimized_op = PERFLIBS_OPERATION_NOTRANS;
  perflibs_int_t C = 0;     // Chunk size - number of rows
  perflibs_int_t nC = 0;    // Number of chunks
  perflibs_int_t sigma = 0; // Number of consecutive rows to sort

  std::vector<int64_t> cs; // Starting offset of each chunk
  std::vector<int64_t> cl; // Width of each chunk (i.e. longest row)
  std::vector<int64_t>
      row_in2permd; // Provided an input row index, gives the permuted index (-1
                    // indicates unused row due to padding up to nC*C rows)
  std::vector<int64_t>
      row_permd2in; // Provided a permuted row index, gives the input index
  perflibs::sparse::pod_vector<T> vals; // Values of the matrix in the SCS
                                        // ordering (column major within chunks)
  perflibs::sparse::pod_vector<perflibs_int_t>
      col_indx_offsets; // Column indices of corresponding vals expressed as
                        // offsets from col_indx_min
  std::vector<int64_t> col_indx_min; // Minimum column index of the chunk
  perflibs_int_t col_indx_bytes =
      0; // The number of bytes used to store each element of col_indx_offsets
  std::vector<perflibs_int_t>
      chunk_min_nnz; // The number of non-zeros in a 'chunk' of the CSR matrix

  perflibs_int_t n_full_chunks = 0;
  perflibs_int_t ntidyup = 0;
  int nthreads = perflibs::sparse::omp::get_max_threads();

  std::vector<perflibs_int_t>
      thread_min_nnz; // The number of non-zeros per thread
  std::vector<perflibs_int_t> nchunks_thread;
  std::vector<perflibs_int_t> chk_offset;

  scs_spmv_kernels<T> kernels;

  perflibs_scs(perflibs_scs &&) = default;
  perflibs_scs(const perflibs_scs &) = default;

  perflibs_scs &operator=(perflibs_scs &&) = default;
  perflibs_scs &operator=(const perflibs_scs &) = default;

  perflibs_scs() : m(-1), n(-1) {};
  perflibs_scs(perflibs_int_t m, perflibs_int_t n, perflibs_int_t C,
               perflibs_int_t sigma, int nthreads, scs_spmv_kernels<T> kernels)
      : m(m), n(n), C(C), nC(iround_div(m, C)), sigma(sigma), cs(nC), cl(nC),
        row_in2permd(nC * C, -1), row_permd2in(nC * C, -1), col_indx_min(nC),
        chunk_min_nnz(nC), nthreads(nthreads), thread_min_nnz(nthreads),
        nchunks_thread(nthreads), chk_offset(nthreads), kernels(kernels) {}

  void gen_data_vectors(enum sparse_hint_value_internal trans,
                        const perflibs_int_t *row_ptr,
                        const perflibs_int_t *col_indx, const T *vals);
  void gen_parallel_decomp(perflibs_scs_strategy_t strat);
  double get_beta(perflibs_int_t nnz_min);
  scs_beta_threads_t get_beta_threads(perflibs_int_t nnz_min);
  void print_structure();
};

template <typename T>
perflibs_scs<T>
csr2scs(enum sparse_hint_value_internal trans, perflibs_int_t m,
        perflibs_int_t n, const T *vals, const perflibs_int_t *col_indx,
        const perflibs_int_t *row_ptr, perflibs_int_t C, perflibs_int_t sigma,
        int nthreads, scs_spmv_kernels<T> kernels);

// C++ implementation
template <typename T, int C>
void spmv_scs(const perflibs_scs<T> &scs, const T *x, T *y, T alpha, T beta);

// Execution wrapper
template <typename T>
void spmv_scs_opt(const perflibs_scs<T> &scs, const T *x, T *y, T alpha,
                  T beta);

template <typename T>
perflibs_status_t
spmat_update_scs(perflibs_spmat_impl_t<T> *impl, perflibs_int_t n_updates,
                 const perflibs_int_t *row_indx, const perflibs_int_t *col_indx,
                 const T *vals);

/**
 * This function returns the valid C for the given type in the
 * order of preference to be used during auditioning.
 */
template <typename T>
const std::vector<std::pair<int, scs_spmv_kernels<T>>> &
scs_get_valid_C_default(T vals);

} // namespace perflibs::sparse
