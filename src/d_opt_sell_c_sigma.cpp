/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "sell_c_sigma.hpp"
#include <algorithm>
#include <cstring>

namespace perflibs::sparse {

// Because we don't have valid kernels for Arm64EC we cannot use this
// implementation
#if !defined(_M_ARM64EC) && !defined(__arm64ec__)
// This kernel is treated as reproducible for a fixed SCS representation.
// Each chunk owns complete output rows, so thread scheduling does not
// change their accumulation order.
template <>
void spmv_scs_opt(const perflibs_scs<double> &scs, const double *x, double *y,
                  double alpha, double beta) {
  auto nt = scs.nthreads;

  int tidy = 0;

#pragma omp parallel for schedule(static) shared(tidy) num_threads(nt)
  for (int me = 0; me < nt; me++) {

    {
      perflibs_int_t nchunks = scs.nchunks_thread[me];
      perflibs_int_t chk_offset = scs.chk_offset[me];
      auto row_offset = chk_offset * scs.C;

      if (nchunks) {
        auto *kernel =
            scs.sigma ? scs.kernels.spmv_kernel : scs.kernels.spmv_kernel_sig0;
        auto y_offset = scs.sigma ? 0 : row_offset;
        (*kernel)(nchunks, &scs.vals[0], &scs.col_indx_offsets[0], &x[0],
                  &y[y_offset], &scs.cs[chk_offset], &scs.cl[chk_offset],
                  &scs.row_permd2in[row_offset], scs.col_indx_bytes,
                  &scs.col_indx_min[chk_offset], alpha, beta);
      }
    } // end task

    int do_tidy;
#pragma omp atomic capture
    {
      do_tidy = tidy;
      ++tidy;
    }

    if (!do_tidy && scs.ntidyup) {
      auto ci = scs.nC - 1;

      // Zero work vector
      perflibs::sparse::pod_vector<double> work(scs.C);
      memset(&work[0], 0, sizeof(double) * scs.C);

      // Perform multiplication
      auto col_indx_min = scs.col_indx_min[ci];

      if (scs.col_indx_bytes == 1) {

        auto col_indx_off =
            reinterpret_cast<const char *>(scs.col_indx_offsets.data());

        auto dat_index = scs.cs[ci];
        for (perflibs_int_t ri = 0; ri < scs.cl[ci]; ri++) {
          for (perflibs_int_t i = 0; i < scs.ntidyup; i++) {
            auto col0 = col_indx_off[dat_index + i] + col_indx_min;
            work[i] += scs.vals[dat_index + i] * x[col0];
          }
          dat_index += scs.C;
        }
      } else if (scs.col_indx_bytes == 2) {

        auto col_indx_off =
            reinterpret_cast<const short *>(scs.col_indx_offsets.data());

        auto dat_index = scs.cs[ci];
        for (perflibs_int_t ri = 0; ri < scs.cl[ci]; ri++) {
          for (perflibs_int_t i = 0; i < scs.ntidyup; i++) {
            auto col0 = col_indx_off[dat_index + i] + col_indx_min;
            work[i] += scs.vals[dat_index + i] * x[col0];
          }
          dat_index += scs.C;
        }
      } else if (scs.col_indx_bytes == 4) {

        auto col_indx_off =
            reinterpret_cast<const int *>(scs.col_indx_offsets.data());

        auto dat_index = scs.cs[ci];
        for (perflibs_int_t ri = 0; ri < scs.cl[ci]; ri++) {
          for (perflibs_int_t i = 0; i < scs.ntidyup; i++) {
            auto col0 = col_indx_off[dat_index + i] + col_indx_min;
            work[i] += scs.vals[dat_index + i] * x[col0];
          }
          dat_index += scs.C;
        }
      } else if (scs.col_indx_bytes == 8) {

        auto col_indx_off =
            reinterpret_cast<const long long *>(scs.col_indx_offsets.data());

        auto dat_index = scs.cs[ci];
        for (perflibs_int_t ri = 0; ri < scs.cl[ci]; ri++) {
          for (perflibs_int_t i = 0; i < scs.ntidyup; i++) {
            auto col0 = col_indx_off[dat_index + i] + col_indx_min;
            work[i] += scs.vals[dat_index + i] * x[col0];
          }
          dat_index += scs.C;
        }
      }

      // Apply alpha and beta
      bool do_alpha = alpha != 1.0;
      bool do_beta = beta != 0.0;
      auto row_index = ci * scs.C;
      if (do_alpha && do_beta) {
        for (perflibs_int_t i = 0; i < scs.ntidyup; i++) {
          auto row0 = scs.row_permd2in[row_index + i];
          y[row0] = work[i] * alpha + y[row0] * beta;
        }
      } else if (do_alpha) {
        for (perflibs_int_t i = 0; i < scs.ntidyup; i++) {
          auto row0 = scs.row_permd2in[row_index + i];
          y[row0] = work[i] * alpha;
        }
      } else if (do_beta) {
        for (perflibs_int_t i = 0; i < scs.ntidyup; i++) {
          auto row0 = scs.row_permd2in[row_index + i];
          y[row0] = work[i] + y[row0] * beta;
        }
      } else {
        for (perflibs_int_t i = 0; i < scs.ntidyup; i++) {
          auto row0 = scs.row_permd2in[row_index + i];
          y[row0] = work[i];
        }
      }
    } // end tidyup task
  } // end parallel
}
#endif

} // namespace perflibs::sparse
