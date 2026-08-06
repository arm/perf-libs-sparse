/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "matvec.hpp"
#include "block_sparse_rows.hpp"
#include "compressed_sparse_columns.hpp"
#include "compressed_sparse_rows.hpp"
#include "convert.hpp"
#include "coordinate_list.hpp"
#include "dense.hpp"
#include "object_helpers.hpp"
#include "types.hpp"

#include "cblas_wrappers.hpp"
#include "sell_c_sigma.hpp"
#include "spec.hpp"
#include "special.hpp"
#include "statistics.hpp"
#include "timer.hpp"
#include "unused.hpp"

#include <cstring>
#include <inttypes.h>

namespace perflibs::sparse {
template <typename T>
perflibs_status_t call_spmv(perflibs_sparse_hint_value trans, T alpha,
                            perflibs_spmat_impl_t<T> *impl, const T *x, T beta,
                            T *y) {

  if (impl->spmat_format == perflibs_format_csr) {
    spmv_csr<T>(impl->csr, (sparse_hint_value_internal)trans, x, y, alpha,
                beta);
  } else if (impl->spmat_format == perflibs_format_csc) {
    spmv_csc<T>(impl->csc, trans, x, y, alpha, beta);
  } else if (impl->spmat_format == perflibs_format_coo) {
    spmv_coo<T>(impl->coo, (sparse_hint_value_internal)trans, x, y, alpha, beta,
                impl->index_base);
  } else if (impl->spmat_format == perflibs_format_scs) {
    // This is an invariant, since it's checked in spmv_exec, but worth
    // checking here. Return an error in release builds
    const bool op_matches =
        impl->scs.optimized_op == (sparse_hint_value_internal)trans;
    assert(op_matches);
    if (!op_matches) {
      return PERFLIBS_STATUS_EXECUTION_FAILURE;
    }

    spmv_scs_opt(impl->scs, x, y, alpha, beta);
  } else if (impl->spmat_format == perflibs_format_bsr) {
    spmv_bsr<T>(impl->bsr, (sparse_hint_value_internal)trans, x, y, alpha,
                beta);
  }
  return PERFLIBS_STATUS_SUCCESS;
};

template <typename T>
perflibs_status_t spmv_exec_impl(perflibs_sparse_hint_value trans, T alpha,
                                 perflibs_spmat_impl_t<T> *impl, const T *x,
                                 T beta, T *y) {

  if (trans == PERFLIBS_SPARSE_OPERATION_NOTRANS ||
      trans == PERFLIBS_SPARSE_OPERATION_TRANS ||
      trans == PERFLIBS_SPARSE_OPERATION_CONJTRANS) {

    perflibs_int_t ylen =
        trans == PERFLIBS_SPARSE_OPERATION_NOTRANS ? impl->m : impl->n;
    perflibs_int_t xlen =
        trans == PERFLIBS_SPARSE_OPERATION_NOTRANS ? impl->n : impl->m;

    if (ylen != 0) {
      if (beta == T(0)) {
        std::memset(reinterpret_cast<void *>(y), 0, sizeof(T) * ylen);
        if (xlen == 0 || alpha == T(0)) {
          return PERFLIBS_STATUS_SUCCESS;
        }
      } else if (xlen == 0 || alpha == T(0)) {
        perflibs_int_t one = 1;
        cblas_scal<T>(ylen, beta, y, one);
        return PERFLIBS_STATUS_SUCCESS;
      }
    } else {
      return PERFLIBS_STATUS_SUCCESS;
    }

    if (impl->spmat_format == perflibs_format_identity) {
      return spmv_exec_identity(trans, alpha, impl, x, beta, y);
    } else if (impl->spmat_format == perflibs_format_null) {
      return spmv_exec_null(trans, impl, beta, y);
    } else if (impl->spmat_format == perflibs_format_dense) {
      return spmv_gemv(impl->dense.layout, trans, impl->m, impl->n,
                       impl->dense.vals_ptr, impl->dense.lda, alpha, x, beta,
                       y);
    }

    return perflibs::sparse::call_spmv(trans, alpha, impl, x, beta, y);
  }

  impl->error_handle.perflibs_error_type =
      PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  impl->error_handle.perflibs_error_code = 1;
  impl->error_handle.err_msg = "Incorrect transpose option.";
  return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
};

template <typename T>
perflibs_status_t spmv_exec(perflibs_sparse_hint_value trans, T alpha,
                            perflibs_spmat_top_t *A, const T *x, T beta, T *y) {

  auto impl = reinterpret_cast<perflibs_spmat_impl_t<T> *>(A->impl);

  // Fall back to CSR if the hint used for optimization disagrees with the one
  // used here for exec, and invalidate any parallel setup (if it was already
  // CSR).
  const bool scs_op_mismatch =
      impl->spmat_format == perflibs_format_scs &&
      impl->scs.optimized_op != (sparse_hint_value_internal)trans;
  if (trans != impl->userhint_spmv_op || scs_op_mismatch) {
    if (impl->spmat_format != perflibs_format_identity &&
        impl->spmat_format != perflibs_format_null) {
      convert(perflibs_format_csr, impl);
      impl->userhint_spmv_op = trans;
    }
    impl->csr.par_mv = {};
  }

  return spmv_exec_impl<T>(trans, alpha, impl, x, beta, y);
};
template perflibs_status_t
spmv_exec<float>(perflibs_sparse_hint_value trans, float alpha,
                 perflibs_spmat_top_t *A, const float *x, float beta, float *y);
template perflibs_status_t spmv_exec<double>(perflibs_sparse_hint_value trans,
                                             double alpha,
                                             perflibs_spmat_top_t *A,
                                             const double *x, double beta,
                                             double *y);
template perflibs_status_t spmv_exec<std::complex<float>>(
    perflibs_sparse_hint_value trans, std::complex<float> alpha,
    perflibs_spmat_top_t *A, const std::complex<float> *x,
    std::complex<float> beta, std::complex<float> *y);
template perflibs_status_t spmv_exec<std::complex<double>>(
    perflibs_sparse_hint_value trans, std::complex<double> alpha,
    perflibs_spmat_top_t *A, const std::complex<double> *x,
    std::complex<double> beta, std::complex<double> *y);

/*
 * Reallocates the data in the matrix to ensure a favorable parallel
 * decomposition for SpMV operations.
 * @param [in,out] impl    The matrix to update the data for
 * @param [in] format      The format of the matrix (i.e. CSR, CSC, etc)
 * @param [in] decomp      The type of decomposition to prefer
 * @return @p true when data has been reallocated, else @p false. This is not an
 * indication of success or failure.
 */
template <typename T>
bool spmv_realloc_data_first_touch(perflibs_spmat_impl_t<T> *impl,
                                   spmat_format_t format) {
  // In the serial case, return
  if (perflibs::sparse::omp::get_max_threads() == 1)
    return false;

  switch (format) {
  case perflibs_format_bsr:
    spmv_bsr_realloc_data(impl->bsr);
    return true;
  case perflibs_format_csr:
    impl->csr.gen_parallel_decomp_mv(
        perflibs_parallel_decomp_strategy::nnz_full_rows);
    impl->csr.gen_data_vectors_mv();
    return true;
  case perflibs_format_scs:
  case perflibs_format_dense:
  case perflibs_format_identity:
  case perflibs_format_null:
  case perflibs_format_coo:
  case perflibs_format_csc:
  case perflibs_format_supernodal:
    // Not yet implemented
    return false;
  }
  assert(false);
  return false;
}

template <typename T>
void audition_spmv(perflibs_spmat_impl_t<T> *impl, perflibs_int_t C_force,
                   perflibs_int_t sigma_force) {

  constexpr bool verbose = false;

  bool force_scs = C_force > 0 && sigma_force >= 0;

  // Save the input format
  auto in_format = impl->spmat_format;

  bool tiny_prob = impl->nnz < 100 && !force_scs;
  bool small_prob = sizeof(T) * impl->nnz < 32000 && !force_scs;

  if (small_prob) {
    if (in_format == perflibs_format_csr) {
      impl->csr.par_mv.nthreads = 1;
    } else if (in_format == perflibs_format_csc) {
      impl->csc.nthreads = 1;
    } else if (in_format == perflibs_format_coo) {
      impl->coo.nthreads = 1;
    } else if (in_format == perflibs_format_bsr) {
      impl->bsr.nthreads = 1;
    }

    if (verbose) {
      printf("Problem size is small, so using 1 thread\n");
    }
  }

  // Don't waste time if the problem is tiny, return early unless we're forcing
  // use of SCS
  if (tiny_prob) {
    if (verbose) {
      printf("Returning early with no auditioning since this is a tiny "
             "problem.\n");
    }
    return;
  }

  if (force_scs) {
    impl->audition_tl = 0.0;
  }

  // For now we don't audition different nthreads values
  int best_nthreads = perflibs::sparse::omp::get_max_threads();

  int64_t rows = impl->m;
  int64_t cols = impl->n;
  auto trans = (sparse_hint_value_internal)impl->userhint_spmv_op;

  // Get pointers for CSR data
  T *vals = nullptr;
  perflibs_int_t *row_ptr = nullptr;
  perflibs_int_t *col_indx = nullptr;

  perflibs_csr<T> csr;
  if (in_format == perflibs_format_csr) {
    vals = impl->csr.vals.data();
    row_ptr = impl->csr.row_ptr.data();
    col_indx = impl->csr.col_indx.data();
  } else if (in_format == perflibs_format_csc) {
    vals = impl->csc.vals.data();
    row_ptr = impl->csc.col_ptr.data();
    col_indx = impl->csc.row_indx.data();

    trans = impl->userhint_spmv_op == PERFLIBS_SPARSE_OPERATION_NOTRANS
                ? PERFLIBS_OPERATION_TRANS
            : impl->userhint_spmv_op == PERFLIBS_SPARSE_OPERATION_TRANS
                ? PERFLIBS_OPERATION_NOTRANS
                : PERFLIBS_OPERATION_CONJNOTRANS;

    rows = impl->n;
    cols = impl->m;
  } else if (in_format == perflibs_format_coo) {
    csr = perflibs::sparse::coo2csr<T>(
        int64_t(impl->m), int64_t(impl->n), impl->nnz, impl->coo.vals.data(),
        impl->coo.col_indx.data(), impl->coo.row_indx.data(), impl->index_base);
    vals = csr.vals.data();
    row_ptr = csr.row_ptr.data();
    col_indx = csr.col_indx.data();
  } else if (in_format == perflibs_format_bsr) {
    perflibs::sparse::perflibs_bsr<T> &bsr = impl->bsr;
    csr = perflibs::sparse::bsr2csr<T>(
        bsr.block_layout, bsr.m, bsr.n, bsr.block_size, bsr.nnzb, bsr.nrowsb,
        bsr.vals.data(), bsr.row_ptr.data(), bsr.col_indx.data());
    vals = csr.vals.data();
    row_ptr = csr.row_ptr.data();
    col_indx = csr.col_indx.data();
  } else {
    assert(false);
  }

  // Find the best value of beta for a range of candidate C and sigma parameters
  bool scs_candidate = false;
  auto find_best_scs = [&]() {
    if (force_scs) {
      if (verbose) {
        printf("Forcing use of SCS with C = %" PRId64 ", sigma = %" PRId64 "\n",
               (int64_t)C_force, (int64_t)sigma_force);
      }
      for (auto [C, details] : perflibs::sparse::scs_get_valid_C(vals[0])) {
        if (C == C_force) {
          scs_candidate = true;
          return perflibs::sparse::csr2scs<T>(
              trans, rows, cols, vals, col_indx, row_ptr, C_force,
              C_force * sigma_force, best_nthreads, details);
        }
      }
      assert(false);
    }

    // Iterate over parameters in order of preference. Return as soon as an
    // scs_beta is found that is acceptable
    double best_beta = 0;
    perflibs_scs<T> best_scs;
    constexpr double scs_beta_threshold = 0.96;
    const auto &cvals = perflibs::sparse::scs_get_valid_C(vals[0]);
    for (auto [C, details] : cvals) {
      for (perflibs_int_t sigma : {0, 200, 2000}) {
        auto cur_scs = perflibs::sparse::csr2scs<T>(
            trans, rows, cols, vals, col_indx, row_ptr, C, C * sigma,
            best_nthreads, details);
        auto cur_beta = cur_scs.get_beta(row_ptr[rows] - row_ptr[0]);
        auto [min_beta, max_beta] =
            cur_scs.get_beta_threads(row_ptr[rows] - row_ptr[0]);
        PERFLIBS_UNUSED(max_beta);
        if (cur_beta > best_beta) {
          constexpr double scs_thread_beta_threshold = 0.85;
          auto thread_beta = min_beta;
          best_scs = std::move(cur_scs);
          best_beta = cur_beta;
          if (thread_beta > scs_thread_beta_threshold &&
              best_beta >= scs_beta_threshold) {
            scs_candidate = true;
            return best_scs;
          }
        }
      }
    }
    return best_scs;
  };

  perflibs_scs<T> best_scs = find_best_scs();
  // The operation above may have been adjusted to interpret CSC storage as
  // CSR. Record the operation applied to the original logical matrix.
  best_scs.optimized_op = (sparse_hint_value_internal)impl->userhint_spmv_op;

  if (scs_candidate && small_prob) {
    best_scs.nthreads = 1;
    if (verbose) {
      printf("Problem size is tiny, so using 1 thread\n");
    }
  }

  bool use_scs = force_scs;
  auto best_strat = perflibs_scs_parallel_chunk_division;

  T alpha = T(1.0);
  T beta = T(0.0);
  auto m = impl->userhint_spmv_op == PERFLIBS_SPARSE_OPERATION_NOTRANS
               ? impl->m
               : impl->n;
  auto n = impl->userhint_spmv_op == PERFLIBS_SPARSE_OPERATION_NOTRANS
               ? impl->n
               : impl->m;
  std::vector<T> x(n);
  std::vector<T> y(m);
  // Scale the time limit by NNZ
  double time_scal = std::min((double)impl->nnz / 45000000., 1.0);
  double target_secs = std::min(0.50, impl->audition_tl) * time_scal;
  // Given that we're warming up cache there should be little variation in the
  // samples
  double margin = 0.3;
  perflibs::sparse::statistics::normal_distribution best_dist;

  auto time_exec =
      [&](perflibs::sparse::statistics::normal_distribution &candidate_dist) {
        // Discard the first 3 runs in order to warm up cache etc.
        auto t_start = timer_start();
        perflibs::sparse::spmv_exec_impl(impl->userhint_spmv_op, alpha, impl,
                                         &x[0], beta, &y[0]);
        perflibs::sparse::spmv_exec_impl(impl->userhint_spmv_op, alpha, impl,
                                         &x[0], beta, &y[0]);
        perflibs::sparse::spmv_exec_impl(impl->userhint_spmv_op, alpha, impl,
                                         &x[0], beta, &y[0]);
        double elapsed_secs = timer_end(std::move(t_start));

        do {
          auto t_candidate = timer_start();
          perflibs::sparse::spmv_exec_impl(impl->userhint_spmv_op, alpha, impl,
                                           &x[0], beta, &y[0]);
          double candidate_diff_secs = timer_end(std::move(t_candidate));

          candidate_dist =
              perflibs::sparse::statistics::sample_normal_incremental(
                  candidate_dist, candidate_diff_secs);
          auto prob = perflibs::sparse::statistics::welch_t_test(best_dist,
                                                                 candidate_dist)
                          .v1_p;
          if (prob < margin || prob > (1 - margin)) {
            break;
          }
          elapsed_secs += candidate_diff_secs;
        } while (elapsed_secs + candidate_dist.mean < target_secs);
      };

  bool medium_prob = sizeof(T) * impl->nnz < 256000 && !force_scs;
  bool try_vanilla =
      (in_format == perflibs_format_csr || in_format == perflibs_format_bsr) &&
      impl->userhint_spmv_op == PERFLIBS_SPARSE_OPERATION_NOTRANS;

  // Update the native data structures to be optimized for OpenMP first
  // touch. Prefer a method that splits on the number of non-zero values
  if (spmv_realloc_data_first_touch(impl, in_format)) {
    // Need to make sure that the pointers to the csr are updated correctly, if
    // the format is CSR
    if (in_format == perflibs_format_csr) {
      row_ptr = impl->csr.row_ptr.data();
      vals = impl->csr.vals.data();
      col_indx = impl->csr.col_indx.data();
    }
    if (verbose)
      printf("Reallocating data for OpenMP first touch\n");
  }

  if (scs_candidate || try_vanilla) {
    // Get the native performance
    time_exec(best_dist);
    if (verbose) {
      printf("Native mean time = %f nruns = %" PRId64 "\n", best_dist.mean,
             (int64_t)best_dist.n);
    }

    // Check the vanilla implementation if there is one
    if (try_vanilla) {
      perflibs::sparse::statistics::normal_distribution candidate_dist;
      impl->csr.use_vanilla = true;
      impl->bsr.use_vanilla = true;
      time_exec(candidate_dist);
      if (candidate_dist.mean < best_dist.mean) {
        best_dist = std::move(candidate_dist);
        if (verbose) {
          printf("Preferring vanilla kernel, mean time = %f nruns = %" PRId64
                 "\n",
                 best_dist.mean, (int64_t)best_dist.n);
        }
      } else {
        impl->csr.use_vanilla = false;
        impl->bsr.use_vanilla = false;
      }
    }
  }

  // If we have an acceptable beta value or if the problem is medium then
  // audition the two SCS parallel strategies
  if (scs_candidate || medium_prob) {

    // Make sure that the scs matrix is populated
    assert(best_scs.m != -1);
    // Check best SCS
    impl->spmat_format = perflibs_format_scs;
    impl->scs = std::move(best_scs);
    for (auto strat : {perflibs_scs_parallel_nnz_division,
                       perflibs_scs_parallel_chunk_division}) {
      impl->scs.gen_parallel_decomp(strat);

      // Generate the nnz-length arrays only once using a first-touch policy
      // that benefits nnz division
      if (strat == perflibs_scs_parallel_nnz_division) {
        impl->scs.gen_data_vectors(trans, row_ptr, col_indx, vals);
      }

      perflibs::sparse::statistics::normal_distribution candidate_dist;
      time_exec(candidate_dist);
      if (candidate_dist.mean < best_dist.mean || force_scs) {
        best_strat = strat;
        if (verbose) {
          if (strat == perflibs_scs_parallel_chunk_division) {
            printf("Preferring perflibs_scs_parallel_chunk_division, mean = %f "
                   "nruns = %" PRId64 "\n",
                   candidate_dist.mean, (int64_t)candidate_dist.n);
          } else {
            printf("Preferring perflibs_scs_parallel_nnz_division, mean = %f "
                   "nruns = %" PRId64 "\n",
                   candidate_dist.mean, (int64_t)candidate_dist.n);
          }
        }
        best_dist = std::move(candidate_dist);
        use_scs = true;
      }
    }
  }

  // Create the final SCS datastructure if required
  if (use_scs) {
    impl->scs.gen_parallel_decomp(best_strat);
    impl->spmat_format = perflibs_format_scs;
    if (verbose) {
      auto [min_beta, max_beta] =
          impl->scs.get_beta_threads(row_ptr[rows] - row_ptr[0]);
      PERFLIBS_UNUSED(max_beta);
      printf("Using SCS kernel with C = %" PRId64 " and sigma = %" PRId64
             ", beta = %f, col_indx_bytes = %" PRId64 "\n",
             (int64_t)impl->scs.C, (int64_t)impl->scs.sigma, min_beta,
             (int64_t)impl->scs.col_indx_bytes);
    }

    // Clear contents of old data structures
    if (in_format == perflibs_format_csr) {
      impl->csr = {};
    } else if (in_format == perflibs_format_csc) {
      impl->csc = {};
    } else if (in_format == perflibs_format_coo) {
      impl->coo = {};
    } else if (in_format == perflibs_format_bsr) {
      impl->bsr = {};
    } else if (in_format == perflibs_format_dense) {
      impl->dense = {};
    } else if (in_format == perflibs_format_supernodal) {
      impl->supernodal = {};
    }
  } else {
    impl->spmat_format = in_format;
    if (verbose) {
      printf("Using input format\n");
    }
    impl->scs = {};
  }
};
template void audition_spmv<float>(perflibs_spmat_impl_t<float> *A);
template void audition_spmv<double>(perflibs_spmat_impl_t<double> *A);
template void audition_spmv<std::complex<float>>(
    perflibs_spmat_impl_t<std::complex<float>> *A);
template void audition_spmv<std::complex<double>>(
    perflibs_spmat_impl_t<std::complex<double>> *A);

template <typename T> void audition_spmv(perflibs_spmat_impl_t<T> *impl) {
  audition_spmv(impl, impl->use_C, impl->use_sigma);
}

template <typename T>
perflibs_status_t spmat_update(perflibs_spmat_top_t *A,
                               perflibs_int_t n_updates,
                               const perflibs_int_t *row_indx,
                               const perflibs_int_t *col_indx, const T *vals) {
  if (n_updates == 0)
    return PERFLIBS_STATUS_SUCCESS;

  auto impl = reinterpret_cast<perflibs_spmat_impl_t<T> *>(A->impl);
  if (n_updates < 0) {
    impl->error_handle.perflibs_error_type =
        PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    impl->error_handle.perflibs_error_code = 1;
    impl->error_handle.err_msg = "The number of updates can not be negative.";
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }

  if (impl->spmat_format == perflibs_format_csr) {
    return spmat_update_csr(impl, n_updates, row_indx, col_indx, vals);
  } else if (impl->spmat_format == perflibs_format_csc) {
    return spmat_update_csc(impl, n_updates, row_indx, col_indx, vals);
  } else if (impl->spmat_format == perflibs_format_scs) {
    return spmat_update_scs(impl, n_updates, row_indx, col_indx, vals);
  } else if (impl->spmat_format == perflibs_format_coo) {
    return spmat_update_coo(impl, n_updates, row_indx, col_indx, vals);
  } else if (impl->spmat_format == perflibs_format_dense) {
    return spmat_update_dense(impl, n_updates, row_indx, col_indx, vals);
  } else if (impl->spmat_format == perflibs_format_bsr) {
    return spmat_update_bsr(impl, n_updates, row_indx, col_indx, vals);
  } else {
    return PERFLIBS_STATUS_EXECUTION_FAILURE;
  }
};

template perflibs_status_t spmat_update<float>(perflibs_spmat_top_t *impl,
                                               perflibs_int_t n_updates,
                                               const perflibs_int_t *row_indx,
                                               const perflibs_int_t *col_indx,
                                               const float *vals);
template perflibs_status_t spmat_update<double>(perflibs_spmat_top_t *impl,
                                                perflibs_int_t n_updates,
                                                const perflibs_int_t *row_indx,
                                                const perflibs_int_t *col_indx,
                                                const double *vals);
template perflibs_status_t spmat_update<std::complex<float>>(
    perflibs_spmat_top_t *impl, perflibs_int_t n_updates,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_indx,
    const std::complex<float> *vals);
template perflibs_status_t spmat_update<std::complex<double>>(
    perflibs_spmat_top_t *impl, perflibs_int_t n_updates,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_indx,
    const std::complex<double> *vals);

template <typename T>
perflibs_status_t spmv_optimize(perflibs_spmat_impl_t<T> *impl) {

  if (is_special(impl->spmat_format)) {
    return PERFLIBS_STATUS_SUCCESS;
  }

  if (impl->spmat_format == perflibs_format_csr && impl->csr.vals.empty()) {
    return PERFLIBS_STATUS_SUCCESS;
  }

  if (impl->spmat_format == perflibs_format_csc && impl->csc.vals.empty()) {
    return PERFLIBS_STATUS_SUCCESS;
  }

  if (impl->spmat_format == perflibs_format_coo && impl->coo.vals.empty()) {
    return PERFLIBS_STATUS_SUCCESS;
  }

  if (impl->spmat_format == perflibs_format_bsr && impl->bsr.vals.empty()) {
    return PERFLIBS_STATUS_SUCCESS;
  }

  if (impl->spmat_format == perflibs_format_dense) {
    return PERFLIBS_STATUS_SUCCESS;
  }

  if (impl->spmat_format == perflibs_format_scs) {
    if (impl->scs.optimized_op ==
        (sparse_hint_value_internal)impl->userhint_spmv_op) {
      return PERFLIBS_STATUS_SUCCESS;
    }
    auto status = convert(perflibs_format_csr, impl);
    if (status != PERFLIBS_STATUS_SUCCESS) {
      return status;
    }
    if (impl->csr.vals.empty()) {
      return PERFLIBS_STATUS_SUCCESS;
    }
  }

  if (impl->userhint_spmv_invocations == PERFLIBS_SPARSE_INVOCATIONS_MANY) {
    // Audition to get best C and sigma parameters
    perflibs::sparse::audition_spmv<T>(impl);
  }

  return PERFLIBS_STATUS_SUCCESS;
};

template perflibs_status_t
spmv_optimize<float>(perflibs_spmat_impl_t<float> *impl);
template perflibs_status_t
spmv_optimize<double>(perflibs_spmat_impl_t<double> *impl);
template perflibs_status_t spmv_optimize<std::complex<float>>(
    perflibs_spmat_impl_t<std::complex<float>> *impl);
template perflibs_status_t spmv_optimize<std::complex<double>>(
    perflibs_spmat_impl_t<std::complex<double>> *impl);

} // namespace perflibs::sparse
