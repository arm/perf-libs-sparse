/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "perflibs_sparse_reproducibility.h"
#include "spec.hpp"
#include "types.hpp"

namespace {

// The caller supplies a nonempty row. Preserve the existing formula: divide
// the sum of logarithms by the number of entries, not the number of gaps.
double csr_row_mgmd(perflibs_int_t row, const perflibs_int_t *row_ptr,
                    const perflibs_int_t *col_indx, perflibs_int_t index_base) {
  auto len = row_ptr[row + 1] - row_ptr[row];
  double diff = 0.0;
  auto indx_cur = col_indx[row_ptr[row] - index_base];
  perflibs_int_t indx_last;
  for (auto j = row_ptr[row] + 1; j < row_ptr[row + 1]; j++) {
    indx_last = indx_cur;
    indx_cur = col_indx[j - index_base];
    diff += std::log(std::abs(indx_cur - indx_last));
  }
  return std::exp(diff / len);
}

double csr_mgmd(perflibs_int_t m, const perflibs_int_t *row_ptr,
                const perflibs_int_t *col_indx, perflibs_int_t index_base) {
  const bool reproducible = perflibs_sparse_get_reproducibility() ==
                            PERFLIBS_SPARSE_REPRODUCIBILITY_ENABLED;
  if (reproducible) {
    // Each row keeps its original arithmetic. Only the outer sum changes:
    // increasing row order makes it independent of OpenMP reduction order.
    double mgmd = 0.0;
    for (perflibs_int_t i = 0; i < m; ++i) {
      if (row_ptr[i + 1] != row_ptr[i]) {
        mgmd += csr_row_mgmd(i, row_ptr, col_indx, index_base);
      }
    }
    // Empty rows contribute zero but still count in the average.
    return mgmd / m;
  }

#if defined(_WIN32)
  // Windows OpenMP reduction has produced incorrect results with the
  // supported Windows toolchain. The upstream LLVM/OpenMP issue was closed as
  // not reproducible, so keep the manual reduction until this is verified
  // with the supported Windows toolchain:
  // https://github.com/llvm/llvm-project/issues/64693
  auto nthreads = perflibs::sparse::omp::get_max_threads();
  std::vector<double> mgmd(nthreads, 0.0);
#pragma omp parallel shared(mgmd)
  {
    auto tid = perflibs::sparse::omp::get_thread_num();
#pragma omp for
    for (perflibs_int_t i = 0; i < m; i++) {
      if (row_ptr[i + 1] != row_ptr[i]) {
        mgmd[tid] += csr_row_mgmd(i, row_ptr, col_indx, index_base);
      }
    }
  }
  for (perflibs_int_t i = 1; i < nthreads; ++i) {
    mgmd[0] += mgmd[i];
  }
  return mgmd[0] / m;
#else
  double mgmd = 0.0;
#pragma omp parallel for reduction(+ : mgmd)
  for (perflibs_int_t i = 0; i < m; i++) {
    if (row_ptr[i + 1] != row_ptr[i]) {
      mgmd += csr_row_mgmd(i, row_ptr, col_indx, index_base);
    }
  }
  return mgmd / m;
#endif
}

} // namespace

namespace perflibs::sparse {

// By default, don't compute anything!
template <typename T> bool get_if_mgmd_is_required_default() { return false; }

template bool get_if_mgmd_is_required_default<float>();
template bool get_if_mgmd_is_required_default<double>();
template bool get_if_mgmd_is_required_default<std::complex<float>>();
template bool get_if_mgmd_is_required_default<std::complex<double>>();

} // end namespace perflibs::sparse

template <typename T> void perflibs_spmat_impl_t<T>::fill_matrix_stats_mgmd() {
  if (!perflibs::sparse::get_if_mgmd_is_required<T>()) {
    return;
  }
  if (spmat_format == perflibs_format_csr) {
    stats.mgmd = csr_mgmd(csr.m, csr.row_ptr_ptr, csr.col_indx_ptr, index_base);
  }
}

template void perflibs_spmat_impl_t<float>::fill_matrix_stats_mgmd();
template void perflibs_spmat_impl_t<double>::fill_matrix_stats_mgmd();
template void
perflibs_spmat_impl_t<std::complex<float>>::fill_matrix_stats_mgmd();
template void
perflibs_spmat_impl_t<std::complex<double>>::fill_matrix_stats_mgmd();
