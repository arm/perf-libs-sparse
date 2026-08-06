/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "matmul.hpp"
#include "add.hpp"
#include "compressed_sparse_columns.hpp"
#include "compressed_sparse_rows.hpp"
#include "convert.hpp"
#include "coordinate_list.hpp"
#include "dense.hpp"
#include "export.hpp"
#include "matmul_gustavson.hpp"
#include "matrix_state.hpp"
#include "matvec.hpp"
#include "object_helpers.hpp"
#include "pod_vector.hpp"
#include "symbolic_matmul_compressed.hpp"
#include "types.hpp"

namespace perflibs::sparse {

// Repeated SpMV is faster than the row-blocked SpMM kernel for very small RHS
// counts.  Keep the threshold exclusive: nrhs=5 and above uses SpMM.
constexpr perflibs_int_t spmm_spmv_nrhs_threshold = 5;

static bool is_external_spmv_op(sparse_hint_value_internal trans) {
  return trans == PERFLIBS_OPERATION_NOTRANS ||
         trans == PERFLIBS_OPERATION_TRANS ||
         trans == PERFLIBS_OPERATION_CONJTRANS;
}

static bool is_spmv_format(spmat_format_t format) {
  return format == perflibs_format_csr || format == perflibs_format_csc ||
         format == perflibs_format_coo || format == perflibs_format_scs ||
         format == perflibs_format_bsr;
}

template <typename T>
perflibs_status_t spmm_repeated_spmv_exec_dense_columns(
    sparse_hint_value_internal transA, bool conjB, T alpha, perflibs_spmat_t A,
    perflibs_int_t b_rows, perflibs_int_t c_rows, const T *B,
    perflibs_int_t b_stride_row, perflibs_int_t b_stride_col, T beta, T *C,
    perflibs_int_t c_stride_row, perflibs_int_t c_stride_col,
    perflibs_int_t n) {
  const bool pack_b = conjB || b_stride_row != 1;
  const bool pack_c = c_stride_row != 1;
  perflibs::sparse::pod_vector<T> b_work(pack_b ? b_rows : 0);
  perflibs::sparse::pod_vector<T> c_work(pack_c ? c_rows : 0);
  const auto spmv_trans = to_external_enum(transA);

  for (perflibs_int_t rhs = 0; rhs < n; ++rhs) {
    const T *b_col = B + rhs * b_stride_col;
    if (pack_b) {
      for (perflibs_int_t row = 0; row < b_rows; ++row) {
        const auto value = b_col[row * b_stride_row];
        b_work[row] = conjB ? perflibs::sparse::conj(value) : value;
      }
      b_col = b_work.data();
    }

    T *c_col = C + rhs * c_stride_col;
    if (pack_c) {
      if (beta != T(0)) {
        for (perflibs_int_t row = 0; row < c_rows; ++row) {
          c_work[row] = c_col[row * c_stride_row];
        }
      }
      c_col = c_work.data();
    }

    auto status = spmv_exec<T>(spmv_trans, alpha, A, b_col, beta, c_col);
    if (status != PERFLIBS_STATUS_SUCCESS) {
      return status;
    }

    if (pack_c) {
      T *c_dst = C + rhs * c_stride_col;
      for (perflibs_int_t row = 0; row < c_rows; ++row) {
        c_dst[row * c_stride_row] = c_col[row];
      }
    }
  }

  return PERFLIBS_STATUS_SUCCESS;
}

template <typename T>
perflibs_status_t spmm_dispatch_symbolic(perflibs_int_t m, perflibs_int_t n,
                                         perflibs_int_t k,
                                         const perflibs_int_t *row_ptrA,
                                         const perflibs_int_t *col_indxA,
                                         const perflibs_int_t *row_ptrB,
                                         const perflibs_int_t *col_indxB,
                                         perflibs_spmat_t C_out) {
  // Compress B
  auto [compression_done, compressed_B] =
      make_symbolic_matmul_compressed(k, row_ptrB, col_indxB);

  if (!compression_done) {
    return PERFLIBS_STATUS_SUCCESS;
  }

  // Compute structure of C using compressed_B
  std::vector<perflibs_int_t> row_ptrC_vec(m + 1);
  perflibs::sparse::pod_vector<perflibs_int_t> col_indxC_vec;
  perflibs::sparse::pod_vector<T> valsC_vec;

  perflibs_spmm_opt_t opt_type = perflibs_spmm_single_phase;

  auto impl_C_out = reinterpret_cast<perflibs_spmat_impl_t<T> *>(C_out->impl);

  // Strategy 1: allocate col_indx, vals, find row_ptr values
  if (impl_C_out->userhint_spmm_strat ==
      PERFLIBS_SPARSE_SPMM_STRAT_OPT_PART_STRUCT) {
    get_symbolic_mm_structure_alloc_only(m, n, row_ptrA, col_indxA,
                                         compressed_B, row_ptrB[0],
                                         row_ptrC_vec, col_indxC_vec);
    opt_type = perflibs_spmm_part_struct;
  }

  // Strategy 2: allocate col_indx, vals, find row_ptr, col_indx values
  else if (impl_C_out->userhint_spmm_strat ==
           PERFLIBS_SPARSE_SPMM_STRAT_OPT_FULL_STRUCT) {
    get_symbolic_mm_structure(m, n, row_ptrA, col_indxA, compressed_B,
                              row_ptrB[0], row_ptrC_vec, col_indxC_vec);
    opt_type = perflibs_spmm_full_struct;
  }

  else {
    // If we get in here then the strategy should have been set to
    // OPT_NO_STRUCT, even if it was initially UNSET
    assert(impl_C_out->userhint_spmm_strat ==
           PERFLIBS_SPARSE_SPMM_STRAT_OPT_NO_STRUCT);
  }

  // Assign the allocated vectors to an perflibs_spmat_t data structure
  valsC_vec.resize(row_ptrC_vec[m] - row_ptrC_vec[0]);
  auto C = create_new_matrix<T>();
  auto ret =
      fill_initial_data_csr(C.get(), m, n, std::move(row_ptrC_vec),
                            std::move(col_indxC_vec), std::move(valsC_vec));
  if (ret != PERFLIBS_STATUS_SUCCESS) {
    return ret;
  }

  // Mark this as optimized for SpMM, move it to the outside world's ptr and
  // return
  auto impl_C = reinterpret_cast<perflibs_spmat_impl_t<T> *>(C->impl);
  impl_C->optimized_mm = opt_type;
  *C_out = std::move(*C);

  return PERFLIBS_STATUS_SUCCESS;
}

/*
 * Dispatches the operation \f$C = \alpha*op(A)*op(B)\f$ for $op$
 * either the identity or transpose operation. Matrices A and B are converted to
 * CSR format with the same base indexing. For details on the SpMM operation
 * \see spmm_csr_kernel.
 */
template <typename T>
perflibs_status_t spmm_dispatch(enum perflibs_sparse_hint_value transA,
                                enum perflibs_sparse_hint_value transB, T alpha,
                                perflibs_spmat_t A, perflibs_spmat_t B, T beta,
                                perflibs_spmat_t C_out, bool exec) {

  auto impl_A = reinterpret_cast<perflibs_spmat_impl_t<T> *>(A->impl);
  auto impl_B = reinterpret_cast<perflibs_spmat_impl_t<T> *>(B->impl);
  auto impl_C_out = reinterpret_cast<perflibs_spmat_impl_t<T> *>(C_out->impl);

  // Assumption here is that we've come from spmm_exec and A, B aren't aliases
  // of the same object
  assert(impl_A != impl_B);

  // These should have been checked before we get here, so
  // an incorrect value would point to corruption of the sparse object
  if (!(impl_A->index_base == 0 || impl_A->index_base == 1)) {
    return PERFLIBS_STATUS_EXECUTION_FAILURE;
  }

  if (!(impl_B->index_base == 0 || impl_B->index_base == 1)) {
    return PERFLIBS_STATUS_EXECUTION_FAILURE;
  }

  auto target_formatA = transA == PERFLIBS_SPARSE_OPERATION_NOTRANS
                            ? perflibs_format_csr
                            : perflibs_format_csc;
  convert(target_formatA, impl_A);

  auto target_formatB = transB == PERFLIBS_SPARSE_OPERATION_NOTRANS
                            ? perflibs_format_csr
                            : perflibs_format_csc;
  convert(target_formatB, impl_B);

  perflibs_int_t m, n, k;
  const perflibs_int_t *row_ptrA, *col_indxA;
  const perflibs_int_t *row_ptrB, *col_indxB;
  const T *valsA, *valsB;

  if (transA == PERFLIBS_SPARSE_OPERATION_NOTRANS &&
      transB == PERFLIBS_SPARSE_OPERATION_NOTRANS) { // NN csr*csr
    m = impl_A->m;
    k = impl_A->n;
    n = impl_B->n;

    row_ptrA = impl_A->csr.row_ptr_ptr;
    col_indxA = impl_A->csr.col_indx_ptr;
    valsA = impl_A->csr.vals_ptr;

    row_ptrB = impl_B->csr.row_ptr_ptr;
    col_indxB = impl_B->csr.col_indx_ptr;
    valsB = impl_B->csr.vals_ptr;
  } else if (transA == PERFLIBS_SPARSE_OPERATION_NOTRANS &&
             transB != PERFLIBS_SPARSE_OPERATION_NOTRANS) { // NT csr*csc
    m = impl_A->m;
    k = impl_A->n;
    n = impl_B->m;

    row_ptrA = impl_A->csr.row_ptr_ptr;
    col_indxA = impl_A->csr.col_indx_ptr;
    valsA = impl_A->csr.vals_ptr;

    row_ptrB = impl_B->csc.col_ptr_ptr;
    col_indxB = impl_B->csc.row_indx_ptr;
    valsB = impl_B->csc.vals_ptr;
  } else if (transA != PERFLIBS_SPARSE_OPERATION_NOTRANS &&
             transB == PERFLIBS_SPARSE_OPERATION_NOTRANS) { // TN csc*csr
    m = impl_A->n;
    k = impl_A->m;
    n = impl_B->n;

    row_ptrA = impl_A->csc.col_ptr_ptr;
    col_indxA = impl_A->csc.row_indx_ptr;
    valsA = impl_A->csc.vals_ptr;

    row_ptrB = impl_B->csr.row_ptr_ptr;
    col_indxB = impl_B->csr.col_indx_ptr;
    valsB = impl_B->csr.vals_ptr;
  } else if (transA != PERFLIBS_SPARSE_OPERATION_NOTRANS &&
             transB != PERFLIBS_SPARSE_OPERATION_NOTRANS) { // TT csc*csc
    m = impl_A->n;
    k = impl_A->m;
    n = impl_B->m;

    row_ptrA = impl_A->csc.col_ptr_ptr;
    col_indxA = impl_A->csc.row_indx_ptr;
    valsA = impl_A->csc.vals_ptr;

    row_ptrB = impl_B->csc.col_ptr_ptr;
    col_indxB = impl_B->csc.row_indx_ptr;
    valsB = impl_B->csc.vals_ptr;
  } else {
    return PERFLIBS_STATUS_EXECUTION_FAILURE;
  }

  auto nthreads = perflibs::sparse::omp::get_max_threads();
  auto strat_in = impl_C_out->userhint_spmm_strat;

  // If the strategy is undefined then set appropriate defaults - Gustavson
  // for serial, or two-phase with column indices populated in symbolic phase
  // for parallel
  if (strat_in == PERFLIBS_SPARSE_SPMM_STRAT_UNSET) {
    if (nthreads == 1) {
      impl_C_out->userhint_spmm_strat =
          PERFLIBS_SPARSE_SPMM_STRAT_OPT_NO_STRUCT;
    } else {
      impl_C_out->userhint_spmm_strat =
          PERFLIBS_SPARSE_SPMM_STRAT_OPT_FULL_STRUCT;
    }
  }

  // If we're not being forced to use single shot AND beta is zero (SpMM only)
  // AND we're optimizing only then perform the symbolic phase and return
  if (impl_C_out->userhint_spmm_strat !=
          PERFLIBS_SPARSE_SPMM_STRAT_OPT_NO_STRUCT &&
      beta == (T)0 && !exec) {
    impl_B->fill_matrix_stats_mgmd();
    return spmm_dispatch_symbolic<T>(m, n, k, row_ptrA, col_indxA, row_ptrB,
                                     col_indxB, C_out);
  } else if (!exec) {
    return PERFLIBS_STATUS_SUCCESS;
  }

  // Execution phase below

  perflibs::sparse::pod_vector<perflibs_int_t> index_copy;
  perflibs::sparse::pod_vector<perflibs_int_t> ptr_copy;
  // Optionally copy one of the index arrays to make the index_base the same in
  // both
  make_index_base_equal(m, k, impl_A->nnz, impl_B->nnz, &row_ptrA, &row_ptrB,
                        &col_indxA, &col_indxB, index_copy, ptr_copy);

  // If an optimization has been applied call the matching execute function and
  // return, otherwise fall through to Gustavson, single phase
  auto row_ptrC = impl_C_out->csr.row_ptr_ptr;
  auto valsC = impl_C_out->csr.vals.data();

  // Gustavson numeric phase only, pre-allocated C and col_indx computed already
  if (impl_C_out->optimized_mm == perflibs_spmm_full_struct) {
    auto col_indxC = impl_C_out->csr.col_indx_ptr;
    spmm_csr_gustavson_noalloc_vals_only<T>(
        transA, transB, m, n, alpha, row_ptrA, col_indxA, valsA, row_ptrB,
        col_indxB, valsB, row_ptrC, col_indxC, valsC, std::move(impl_B->stats));
    return PERFLIBS_STATUS_SUCCESS;
  }
  // Gustavson numeric phase only, pre-allocated C, find vals and col_indx
  else if (impl_C_out->optimized_mm == perflibs_spmm_part_struct) {
    auto col_indxC = impl_C_out->csr.col_indx.data();
    spmm_csr_gustavson_noalloc<T>(transA, transB, m, n, alpha, row_ptrA,
                                  col_indxA, valsA, row_ptrB, col_indxB, valsB,
                                  row_ptrC, col_indxC, valsC);
    return PERFLIBS_STATUS_SUCCESS;
  }

  // Gustavson serial, allocating code path
  std::vector<perflibs_int_t> row_ptrC_vec(m + 1);
  // Use std::vectors rather than pod_vectors in the case that we are building
  // the vectors up from scratch: memory management seems far superior E.g.
  // A=hugebubbles_00000, A*A Sandia benchmark case went from around 6 secs. to
  // over 22 secs. if below are pod_vectors
  std::vector<perflibs_int_t> col_indxC_vec;
  std::vector<T> valsC_vec;

  spmm_csr_gustavson<T>(transA, transB, m, n, alpha, row_ptrA, col_indxA, valsA,
                        row_ptrB, col_indxB, valsB, row_ptrC_vec, col_indxC_vec,
                        valsC_vec);

  auto C = create_new_matrix<T>();
  auto ret = fill_initial_data_csr(C.get(), m, n, row_ptrC_vec.data(),
                                   col_indxC_vec.data(), valsC_vec.data(), 0);
  if (ret != PERFLIBS_STATUS_SUCCESS) {
    return ret;
  }
  *C_out = std::move(*C);

  return PERFLIBS_STATUS_SUCCESS;
}

template <typename T>
perflibs_status_t spmm_exec_checked(enum sparse_hint_value_internal transA,
                                    enum sparse_hint_value_internal transB,
                                    T alpha, perflibs_spmat_t A,
                                    perflibs_spmat_t B, T beta,
                                    perflibs_spmat_t C, const bool exec) {

  // Note, in here we completely ignore the PERFLIBS_SPARSE_CREATE_NOCOPY flag
  // in order to return a result. For SpMM that flag creates too many
  // constraints to consider!

  auto impl_A = reinterpret_cast<perflibs_spmat_impl_t<T> *>(A->impl);
  auto impl_B = reinterpret_cast<perflibs_spmat_impl_t<T> *>(B->impl);
  auto impl_C = reinterpret_cast<perflibs_spmat_impl_t<T> *>(C->impl);

  const bool c_is_null = impl_C->spmat_format == perflibs_format_null;

  auto m = transA == PERFLIBS_OPERATION_NOTRANS ||
                   transA == PERFLIBS_OPERATION_CONJNOTRANS
               ? impl_A->m
               : impl_A->n;
  auto n = transB == PERFLIBS_OPERATION_NOTRANS ||
                   transB == PERFLIBS_OPERATION_CONJNOTRANS
               ? impl_B->n
               : impl_B->m;
  auto k = transA == PERFLIBS_OPERATION_NOTRANS ||
                   transA == PERFLIBS_OPERATION_CONJNOTRANS
               ? impl_A->n
               : impl_A->m;

  // The result of the multiplication
  auto AB = create_new_matrix<T>();

  bool alias_c = impl_A == impl_C || impl_B == impl_C;

  /*
   * First off handle all of the special cases for multiplication
   */

  // Trivial case - no multiplication is required
  if (m == 0 || n == 0 || k == 0 || alpha == (T)0 ||
      impl_A->spmat_format == perflibs_format_null ||
      impl_B->spmat_format == perflibs_format_null) {
    AB = null_matrix(m, n);
  }

  // If A and B are the identity matrices and alpha is 1 then set AB to the
  // identity
  else if (alpha == (T)1 && impl_A->spmat_format == perflibs_format_identity &&
           impl_B->spmat_format == perflibs_format_identity) {
    AB = identity_matrix(n);
  }

  // If either A or B is the identity I, copy the other to AB, transpose and
  // scale by alpha. The scale_matrix function will handle converting from I in
  // the case where A=B=I.
  else if (impl_A->spmat_format == perflibs_format_identity ||
           impl_B->spmat_format == perflibs_format_identity) {
    auto impl_AB = reinterpret_cast<perflibs_spmat_impl_t<T> *>(AB->impl);
    *impl_AB =
        impl_A->spmat_format == perflibs_format_identity ? *impl_B : *impl_A;

    if (!(impl_AB->index_base == 0 || impl_AB->index_base == 1)) {
      return PERFLIBS_STATUS_EXECUTION_FAILURE;
    }

    auto trans =
        impl_A->spmat_format == perflibs_format_identity ? transB : transA;
    auto ret = scale_matrix(to_external_enum(trans), alpha, AB.get());
    if (ret != PERFLIBS_STATUS_SUCCESS) {
      return ret;
    }
  }

  // Handle the dense * dense case
  else if (!alias_c && impl_A->spmat_format == perflibs_format_dense &&
           impl_B->spmat_format == perflibs_format_dense &&
           impl_C->spmat_format == perflibs_format_dense &&
           impl_A->dense.layout == impl_B->dense.layout &&
           impl_A->dense.layout == impl_C->dense.layout) {

    // No need to optimize this case.
    // In future we could do something advanced via this interface on an
    // optimize call, such as create the transpose/interleave matrices in
    // advance.
    if (!exec) {
      return PERFLIBS_STATUS_SUCCESS;
    }

    // First we need to make sure that we can write into C. If it has been
    // created with the no_copy flag then we need to override that and allocate
    // the internal buffers.
    impl_C->dense.make_writable();

    return spmm_gemm<T>(impl_A->dense.layout, to_external_enum(transA),
                        to_external_enum(transB), m, n, k, alpha,
                        impl_A->dense.vals_ptr, impl_A->dense.lda,
                        impl_B->dense.vals_ptr, impl_B->dense.lda, beta,
                        impl_C->dense.vals.data(), impl_C->dense.lda);
  }

  // Handle sparse * dense cases via the row-blocked CSR kernel when layouts
  // permit
  else if (!alias_c && impl_A->spmat_format != perflibs_format_dense &&
           impl_B->spmat_format == perflibs_format_dense &&
           (impl_C->spmat_format == perflibs_format_dense ||
            impl_C->spmat_format == perflibs_format_null)) {

    const bool small_external_spmv =
        n < spmm_spmv_nrhs_threshold && is_external_spmv_op(transA);
    if (small_external_spmv && is_spmv_format(impl_A->spmat_format)) {
      // Keep the original dense layouts and pack only genuinely strided
      // columns. Column-major output lets SpMV write each column directly.
      if (c_is_null) {
        std::vector<T> null_vals(m * n);
        auto dense_holder = create_new_matrix<T>();
        auto status = fill_initial_data_dense<T>(
            dense_holder.get(), PERFLIBS_COL_MAJOR, m, n, m, 0,
            null_vals.data(), /*nocopy=*/false);
        if (status != PERFLIBS_STATUS_SUCCESS) {
          return status;
        }
        *C = std::move(*dense_holder);
        impl_C = reinterpret_cast<perflibs_spmat_impl_t<T> *>(C->impl);
        impl_C->dense.matrix_is_zero = true;
      } else {
        impl_C->dense.make_writable();
      }

      if (!exec) {
        const auto spmv_op = to_external_enum(transA);
        if (impl_A->userhint_spmv_op != spmv_op &&
            impl_A->spmat_format == perflibs_format_csr) {
          impl_A->csr.par_mv = {};
        }
        impl_A->userhint_spmv_op = spmv_op;
        impl_A->userhint_spmv_invocations = impl_A->userhint_spmm_invocations;
        auto status = spmv_optimize<T>(impl_A);
        if (status != PERFLIBS_STATUS_SUCCESS) {
          return status;
        }
        return PERFLIBS_STATUS_SUCCESS;
      }

      perflibs_int_t b_stride_row =
          impl_B->dense.layout == PERFLIBS_ROW_MAJOR ? impl_B->dense.lda : 1;
      perflibs_int_t b_stride_col =
          impl_B->dense.layout == PERFLIBS_COL_MAJOR ? impl_B->dense.lda : 1;
      if (transB == PERFLIBS_OPERATION_TRANS ||
          transB == PERFLIBS_OPERATION_CONJTRANS) {
        std::swap(b_stride_row, b_stride_col);
      }
      const perflibs_int_t c_stride_row =
          impl_C->dense.layout == PERFLIBS_ROW_MAJOR ? impl_C->dense.lda : 1;
      const perflibs_int_t c_stride_col =
          impl_C->dense.layout == PERFLIBS_COL_MAJOR ? impl_C->dense.lda : 1;
      const bool conjB = transB == PERFLIBS_OPERATION_CONJTRANS ||
                         transB == PERFLIBS_OPERATION_CONJNOTRANS;

      return spmm_repeated_spmv_exec_dense_columns<T>(
          transA, conjB, alpha, A, k, m, impl_B->dense.vals_ptr, b_stride_row,
          b_stride_col, beta, impl_C->dense.vals.data(), c_stride_row,
          c_stride_col, n);
    }
    if (!exec && small_external_spmv) {
      // Preserve an unsupported format so execution also follows the existing
      // blocked-SpMM path instead of changing dispatch after optimization.
      return PERFLIBS_STATUS_SUCCESS;
    }

    // Change layout of dense B if needed - requires new allocation and copies
    // in 2 specific cases i.e. if NO transpose is required and B is column
    // major, and if Transpose is required and B is row-major.
    perflibs_int_t mb = 0, nb = 0;
    T *new_B_vals = nullptr;
    if ((transB == PERFLIBS_OPERATION_NOTRANS ||
         transB == PERFLIBS_OPERATION_CONJNOTRANS) &&
        impl_B->dense.layout == PERFLIBS_COL_MAJOR) {
      spmat_export_dense(B, PERFLIBS_ROW_MAJOR, &mb, &nb, &new_B_vals);
      assert(new_B_vals);
      // Populate B with the row-major version of the matrix
      auto Bnew = create_new_matrix<T>();
      auto ret = fill_initial_data_dense(Bnew.get(), PERFLIBS_ROW_MAJOR, mb, nb,
                                         nb, 0, new_B_vals, false);
      if (ret != PERFLIBS_STATUS_SUCCESS) {
        return ret;
      }
      *B = std::move(*Bnew);
    } else if ((transB == PERFLIBS_OPERATION_TRANS ||
                transB == PERFLIBS_OPERATION_CONJTRANS) &&
               impl_B->dense.layout == PERFLIBS_ROW_MAJOR) {
      spmat_export_dense(B, PERFLIBS_COL_MAJOR, &mb, &nb, &new_B_vals);
      assert(new_B_vals);
      // Populate B with the column-major version of the matrix
      auto Bnew = create_new_matrix<T>();
      auto ret = fill_initial_data_dense(Bnew.get(), PERFLIBS_COL_MAJOR, mb, nb,
                                         mb, 0, new_B_vals, false);
      if (ret != PERFLIBS_STATUS_SUCCESS) {
        return ret;
      }
      *B = std::move(*Bnew);
    }
    // Refresh our view of B in case it changed above
    impl_B = reinterpret_cast<perflibs_spmat_impl_t<T> *>(B->impl);

    // Free the temporary returned by export
    if (new_B_vals) {
      free(new_B_vals);
    }

    // If C is a null_matrix type, materialize a matrix to store our result into
    if (c_is_null) {
      std::vector<T> null_vals(m * n);
      auto dense_holder = create_new_matrix<T>();
      auto status = fill_initial_data_dense<T>(
          dense_holder.get(), PERFLIBS_ROW_MAJOR, m, n, n, 0, null_vals.data(),
          /*nocopy=*/false);
      if (status != PERFLIBS_STATUS_SUCCESS) {
        return status;
      }
      *C = std::move(*dense_holder);
      impl_C = reinterpret_cast<perflibs_spmat_impl_t<T> *>(C->impl);
      impl_C->dense.matrix_is_zero = true;
    }

    // C is now guaranteed to point to a dense matrix
    if (impl_C->spmat_format != perflibs_format_dense) {
      return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    }

    // If C is not row major it must be a user-supplied dense matrix, which we
    // need to transpose.
    if (impl_C->dense.layout == PERFLIBS_COL_MAJOR && beta != (T)0) {
      perflibs_int_t mc, nc;
      T *new_C_vals = nullptr;
      spmat_export_dense(C, PERFLIBS_ROW_MAJOR, &mc, &nc, &new_C_vals);
      assert(new_C_vals);
      auto Cnew = create_new_matrix<T>();
      auto ret = fill_initial_data_dense(Cnew.get(), PERFLIBS_ROW_MAJOR, mc, nc,
                                         nc, 0, new_C_vals, false);
      if (ret != PERFLIBS_STATUS_SUCCESS) {
        return ret;
      }
      *C = std::move(*Cnew);
      impl_C = reinterpret_cast<perflibs_spmat_impl_t<T> *>(C->impl);
      free(new_C_vals);
    } else if (!c_is_null) {
      // If C is untouched to this point then make sure we can write to it
      impl_C->dense.make_writable();
    }
    assert(impl_C->dense.layout == PERFLIBS_ROW_MAJOR);
    assert(impl_C->dense.vals.data());

    // Zero C, if beta is zero and we don't already know that C is zero
    T *C_vals = impl_C->dense.vals.data();
    auto ldc = impl_C->dense.lda;
    auto nc = impl_C->dense.n;
    if (!(c_is_null || impl_C->dense.matrix_is_zero) && beta == (T)0) {
#pragma omp parallel for
      for (perflibs_int_t row = 0; row < impl_C->dense.m; ++row) {
        std::fill_n(C_vals + row * ldc, nc, (T)0);
      }
    }

    // From here we're all set with dense row-major views of B and C.
    const T *B_vals = impl_B->dense.vals_ptr;
    const auto ldb = impl_B->dense.lda;

    const bool conjA = transA == PERFLIBS_OPERATION_CONJTRANS ||
                       transA == PERFLIBS_OPERATION_CONJNOTRANS;
    const bool conjB = transB == PERFLIBS_OPERATION_CONJTRANS ||
                       transB == PERFLIBS_OPERATION_CONJNOTRANS;

    // Change layout of sparse A if needed - convert to match a row-wise sparse
    // matrix given the transpose options, and then create a CSR view of that.
    perflibs_csr<T> csr_view{};
    if (transA == PERFLIBS_OPERATION_NOTRANS ||
        transA == PERFLIBS_OPERATION_CONJNOTRANS) {
      convert(perflibs_format_csr, impl_A);
      csr_view = std::move(perflibs_csr<T>(
          impl_A->m, impl_A->n, impl_A->csr.vals_ptr, impl_A->csr.row_ptr_ptr,
          impl_A->csr.col_indx_ptr, {}));
    } else { // if (transA == PERFLIBS_OPERATION_TRANS || transA ==
             // PERFLIBS_OPERATION_CONJTRANS)
      convert(perflibs_format_csc, impl_A);
      csr_view = std::move(perflibs_csr<T>(
          impl_A->n, impl_A->m, impl_A->csc.vals_ptr, impl_A->csc.col_ptr_ptr,
          impl_A->csc.row_indx_ptr, {}));
    }

    if (!exec) {
      return PERFLIBS_STATUS_SUCCESS;
    }

    if (!conjA && !conjB) {
      spmm_rowwise_csr_blocked_m<T, false, false>(csr_view, B_vals, ldb, C_vals,
                                                  ldc, nc, alpha, beta);
    } else if (conjA && conjB) {
      spmm_rowwise_csr_blocked_m<T, true, true>(csr_view, B_vals, ldb, C_vals,
                                                ldc, nc, alpha, beta);
    } else if (!conjA && conjB) {
      spmm_rowwise_csr_blocked_m<T, false, true>(csr_view, B_vals, ldb, C_vals,
                                                 ldc, nc, alpha, beta);
    } else { // if (conjA && !conjB)
      spmm_rowwise_csr_blocked_m<T, true, false>(csr_view, B_vals, ldb, C_vals,
                                                 ldc, nc, alpha, beta);
    }

    return PERFLIBS_STATUS_SUCCESS;
  }

  // dense * sparse - recursive call
  else if (!alias_c && impl_B->spmat_format != perflibs_format_dense &&
           impl_A->spmat_format == perflibs_format_dense &&
           (impl_C->spmat_format == perflibs_format_dense ||
            impl_C->spmat_format == perflibs_format_null)) {

    // Make use of identity: (AB)ᴴ = Bᴴ Aᴴ. For the given input trans flag we
    // need to flip the flag used in the recursive call that will end up in the
    // sparse*dense path
    auto flip_flag = [](sparse_hint_value_internal t) {
      switch (t) {
      case PERFLIBS_OPERATION_NOTRANS:
        return PERFLIBS_OPERATION_CONJTRANS; // N → C
      case PERFLIBS_OPERATION_TRANS:
        return PERFLIBS_OPERATION_CONJNOTRANS; // T → CN
      case PERFLIBS_OPERATION_CONJTRANS:
        return PERFLIBS_OPERATION_NOTRANS; // C → N
      // We should never need CONJNOTRANS since it's not a user input option
      case PERFLIBS_OPERATION_CONJNOTRANS:
        assert(false);
        return PERFLIBS_OPERATION_TRANS; // CN → T
      default:
        assert(false);
        return PERFLIBS_OPERATION_NOTRANS;
      }
    };

    auto rev_transA = flip_flag(transA);
    auto rev_transB = flip_flag(transB);

    // C_H will hold the reversed product (shape n x m)
    std::unique_ptr<perflibs_spmat_top_t> C_H;

    perflibs_spmat_t C_ptr;
    bool use_C_as_C_H = c_is_null || beta == (T)0;
    if (use_C_as_C_H) {
      // If C is output only let's use it as our C_H temporary.
      // This improves performance in this common case, since we
      // can now initialize it to all zeros during optimization.
      C_ptr = C;
    } else {
      // Otherwise, we set the C_H matrix to a fresh null matrix
      // which comes with a performance penalty since we do this
      // repeat this during execution
      C_H = null_matrix(n, m);
      C_ptr = C_H.get();
    }

    // If during execution C is dense (!c_is_null implies dense here)
    // and has been marked as zero then we know it's in the logically
    // transposed state ready to be used as C_ptr, so logically
    // transpose it back
    if (exec && !c_is_null && impl_C->dense.matrix_is_zero) {
      impl_C->dense.transpose(); // This just swaps the layout
      std::swap(impl_C->m, impl_C->n);
    }

    // C_H := α · opB′(B) · opA′(A)
    auto ret = spmm_exec_checked(rev_transB, rev_transA, alpha, B, A,
                                 /*beta=*/(T)0, C_ptr, exec);
    if (ret != PERFLIBS_STATUS_SUCCESS) {
      return ret;
    }

    // If we're optimizing, we have setup C_H in C, we can avoid
    // the cost of setting C_H up again during execution.
    // However, we must transpose the matrix to pass parameter checks
    // when execute is called again. Signal this by recording that
    // the dense matrix is zero.
    if (!exec && use_C_as_C_H) {
      auto impl_Ct = reinterpret_cast<perflibs_spmat_impl_t<T> *>(C->impl);
      impl_Ct->dense.transpose(); // This just swaps the layout
      std::swap(impl_Ct->m, impl_Ct->n);
    }

    if (!exec) {
      return PERFLIBS_STATUS_SUCCESS;
    }

    // Set AB and fall through to the tail of the function to handle beta*C, as
    // for other paths
    *AB = std::move(*C_ptr);
    ret = scale_matrix(PERFLIBS_SPARSE_OPERATION_CONJTRANS, (T)1, AB.get());
    if (ret != PERFLIBS_STATUS_SUCCESS) {
      return ret;
    }
  } else {

    // Default spmm path when none of the specializations above have been
    // triggered... We may need to transform A or B into different formats, so
    // take a copy if A and B are aliases for the same thing
    std::unique_ptr<perflibs_spmat_top_t> B_copy;
    auto B_use = B;
    if (impl_A == impl_B) {
      if (!exec) { // If we're optimizing then just return - all bets are off
                   // for optimization if A==B!
        return PERFLIBS_STATUS_SUCCESS;
      }
      B_copy = spmat_copy<T>(B);
      B_use = B_copy.get();
    }

    perflibs_status_t ret;
    // If beta is zero then write the result into C directly and return; in this
    // case C may contain optimization information so it needs to be passed into
    // spmm_dispatch in order to check
    if (beta == (T)0) {
      return spmm_dispatch(to_external_enum(transA), to_external_enum(transB),
                           alpha, A, B_use, beta, C, exec);
    } else {
      ret = spmm_dispatch(to_external_enum(transA), to_external_enum(transB),
                          alpha, A, B_use, beta, AB.get(), exec);
    }
    if (ret != PERFLIBS_STATUS_SUCCESS) {
      return ret;
    }
  }

  // If beta is zero we have our result
  if (AB && beta == (T)0) {
    *C = std::move(*AB);
  }

  // If we're not executing or we have our result (above) then return
  if (!exec || beta == (T)0) {
    return PERFLIBS_STATUS_SUCCESS;
  }

  T one = (T)1;
  auto [ret2, C_out] = spadd_dispatch(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                      PERFLIBS_SPARSE_OPERATION_NOTRANS, one,
                                      AB.get(), beta, C, true);
  if (ret2 != PERFLIBS_STATUS_SUCCESS) {
    return ret2;
  }
  *C = std::move(*C_out);

  return ret2;
}

template <typename T>
perflibs_status_t spmm_exec(enum perflibs_sparse_hint_value transA,
                            enum perflibs_sparse_hint_value transB, T alpha,
                            perflibs_spmat_t A, perflibs_spmat_t B, T beta,
                            perflibs_spmat_t C) {

  auto ret = spmm_check_params<T>(transA, transB, PERFLIBS_SPARSE_SCALAR_ANY, A,
                                  B, PERFLIBS_SPARSE_SCALAR_ANY, C);
  if (ret != PERFLIBS_STATUS_SUCCESS) {
    return ret;
  }

  return spmm_exec_checked(to_internal_enum(transA), to_internal_enum(transB),
                           alpha, A, B, beta, C, true);
}
template perflibs_status_t
spmm_exec<float>(enum perflibs_sparse_hint_value transA,
                 enum perflibs_sparse_hint_value transB, float alpha,
                 perflibs_spmat_t A, perflibs_spmat_t B, float beta,
                 perflibs_spmat_t C);
template perflibs_status_t
spmm_exec<double>(enum perflibs_sparse_hint_value transA,
                  enum perflibs_sparse_hint_value transB, double alpha,
                  perflibs_spmat_t A, perflibs_spmat_t B, double beta,
                  perflibs_spmat_t C);
template perflibs_status_t
spmm_exec<std::complex<float>>(enum perflibs_sparse_hint_value transA,
                               enum perflibs_sparse_hint_value transB,
                               std::complex<float> alpha, perflibs_spmat_t A,
                               perflibs_spmat_t B, std::complex<float> beta,
                               perflibs_spmat_t C);
template perflibs_status_t
spmm_exec<std::complex<double>>(enum perflibs_sparse_hint_value transA,
                                enum perflibs_sparse_hint_value transB,
                                std::complex<double> alpha, perflibs_spmat_t A,
                                perflibs_spmat_t B, std::complex<double> beta,
                                perflibs_spmat_t C);

/// The optimize function currently calls the execute function, telling it to
/// prepare the matrices, but don't actually execute
template <typename T>
perflibs_status_t spmm_optimize(enum perflibs_sparse_hint_value transA,
                                enum perflibs_sparse_hint_value transB,
                                perflibs_sparse_hint_value alpha,
                                perflibs_spmat_t A, perflibs_spmat_t B,
                                perflibs_sparse_hint_value beta,
                                perflibs_spmat_t C) {

  auto ret = spmm_check_params<T>(transA, transB, alpha, A, B, beta, C);
  if (ret != PERFLIBS_STATUS_SUCCESS) {
    return ret;
  }

  // Create some representative values for the optimization
  T alpha_ = alpha == PERFLIBS_SPARSE_SCALAR_ZERO  ? (T)0
             : alpha == PERFLIBS_SPARSE_SCALAR_ONE ? (T)1
                                                   : (T)1.5;
  T beta_ = beta == PERFLIBS_SPARSE_SCALAR_ZERO  ? (T)0
            : beta == PERFLIBS_SPARSE_SCALAR_ONE ? (T)1
                                                 : (T)1.5;

  // Call the exec function with exec=false (i.e. "optimize only")
  return spmm_exec_checked(to_internal_enum(transA), to_internal_enum(transB),
                           alpha_, A, B, beta_, C, false);
}
template perflibs_status_t
spmm_optimize<float>(enum perflibs_sparse_hint_value transA,
                     enum perflibs_sparse_hint_value transB,
                     perflibs_sparse_hint_value alpha, perflibs_spmat_t A,
                     perflibs_spmat_t B, perflibs_sparse_hint_value beta,
                     perflibs_spmat_t C);
template perflibs_status_t
spmm_optimize<double>(enum perflibs_sparse_hint_value transA,
                      enum perflibs_sparse_hint_value transB,
                      perflibs_sparse_hint_value alpha, perflibs_spmat_t A,
                      perflibs_spmat_t B, perflibs_sparse_hint_value beta,
                      perflibs_spmat_t C);
template perflibs_status_t spmm_optimize<std::complex<float>>(
    enum perflibs_sparse_hint_value transA,
    enum perflibs_sparse_hint_value transB, perflibs_sparse_hint_value alpha,
    perflibs_spmat_t A, perflibs_spmat_t B, perflibs_sparse_hint_value beta,
    perflibs_spmat_t C);
template perflibs_status_t spmm_optimize<std::complex<double>>(
    enum perflibs_sparse_hint_value transA,
    enum perflibs_sparse_hint_value transB, perflibs_sparse_hint_value alpha,
    perflibs_spmat_t A, perflibs_spmat_t B, perflibs_sparse_hint_value beta,
    perflibs_spmat_t C);

template <typename T>
perflibs_status_t
spelmm_check_params(perflibs_sparse_hint_value transA,
                    perflibs_sparse_hint_value transB,
                    perflibs_sparse_hint_value alpha, perflibs_spmat_top_t *A,
                    perflibs_spmat_top_t *B, perflibs_sparse_hint_value beta,
                    perflibs_spmat_top_t *C) {

  if (!have_compatible_matrix_datatypes(A, B, C)) {
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }

  auto impl_A = reinterpret_cast<perflibs_spmat_impl_t<T> *>(A->impl);
  auto impl_B = reinterpret_cast<perflibs_spmat_impl_t<T> *>(B->impl);
  auto impl_C = reinterpret_cast<perflibs_spmat_impl_t<T> *>(C->impl);

  if (!(transA == PERFLIBS_SPARSE_OPERATION_NOTRANS ||
        transA == PERFLIBS_SPARSE_OPERATION_TRANS ||
        transA == PERFLIBS_SPARSE_OPERATION_CONJTRANS)) {
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }

  if (!(transB == PERFLIBS_SPARSE_OPERATION_NOTRANS ||
        transB == PERFLIBS_SPARSE_OPERATION_TRANS ||
        transB == PERFLIBS_SPARSE_OPERATION_CONJTRANS)) {
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }

  // A, B and C must have the same (non-negative) dimensions
  auto mA = transA == PERFLIBS_SPARSE_OPERATION_NOTRANS ? impl_A->m : impl_A->n;
  auto nA = transA == PERFLIBS_SPARSE_OPERATION_NOTRANS ? impl_A->n : impl_A->m;

  auto mB = transB == PERFLIBS_SPARSE_OPERATION_NOTRANS ? impl_B->m : impl_B->n;
  auto nB = transB == PERFLIBS_SPARSE_OPERATION_NOTRANS ? impl_B->n : impl_B->m;

  auto mC = impl_C->m;
  auto nC = impl_C->n;

  if ((mC < 0) || (nC < 0) || (mA != mC) || (mB != mC) || (nA != nC) ||
      (nB != nC)) {
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }

  // Ensure that A, B, and C have the same format unless they are null or
  // identity matrices
  auto formatA = impl_A->spmat_format;
  auto formatB = impl_B->spmat_format;
  auto formatC = impl_C->spmat_format;

  // If A is null, B and C must either be null or have the same format
  if (formatA == perflibs_format_null) {
    bool isBNull = (formatB == perflibs_format_null);
    bool isCNull = (formatC == perflibs_format_null);

    if (!isBNull && !isCNull && (formatC != formatB)) {
      return PERFLIBS_STATUS_EXECUTION_FAILURE;
    }
  }
  // Otherwise, B and C must match A if they are not null
  else {
    bool isBIncompatible =
        (formatB != perflibs_format_null && formatB != formatA);
    bool isCIncompatible =
        (formatC != perflibs_format_null && formatC != formatA);

    if (isBIncompatible || isCIncompatible) {
      return PERFLIBS_STATUS_EXECUTION_FAILURE;
    }
  }

  // For now, no_copy is unsupported
  if ((formatA != perflibs_format_null && formatA != perflibs_format_identity &&
       impl_A->no_copy) ||
      (formatB != perflibs_format_null && formatB != perflibs_format_identity &&
       impl_B->no_copy) ||
      (formatC != perflibs_format_null && formatC != perflibs_format_identity &&
       impl_C->no_copy)) {
    return PERFLIBS_STATUS_EXECUTION_FAILURE;
  }

  // Check that valid scalar values have been provided
  if (!(alpha == PERFLIBS_SPARSE_SCALAR_ONE ||
        alpha == PERFLIBS_SPARSE_SCALAR_ZERO ||
        alpha == PERFLIBS_SPARSE_SCALAR_ANY)) {
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }
  if (!(beta == PERFLIBS_SPARSE_SCALAR_ONE ||
        beta == PERFLIBS_SPARSE_SCALAR_ZERO ||
        beta == PERFLIBS_SPARSE_SCALAR_ANY)) {
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }

  return PERFLIBS_STATUS_SUCCESS;
}
template perflibs_status_t spelmm_check_params<float>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    perflibs_sparse_hint_value alpha, perflibs_spmat_top_t *A,
    perflibs_spmat_top_t *B, perflibs_sparse_hint_value beta,
    perflibs_spmat_top_t *C);
template perflibs_status_t spelmm_check_params<double>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    perflibs_sparse_hint_value alpha, perflibs_spmat_top_t *A,
    perflibs_spmat_top_t *B, perflibs_sparse_hint_value beta,
    perflibs_spmat_top_t *C);
template perflibs_status_t spelmm_check_params<std::complex<float>>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    perflibs_sparse_hint_value alpha, perflibs_spmat_top_t *A,
    perflibs_spmat_top_t *B, perflibs_sparse_hint_value beta,
    perflibs_spmat_top_t *C);
template perflibs_status_t spelmm_check_params<std::complex<double>>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    perflibs_sparse_hint_value alpha, perflibs_spmat_top_t *A,
    perflibs_spmat_top_t *B, perflibs_sparse_hint_value beta,
    perflibs_spmat_top_t *C);

template <typename T>
perflibs_status_t
spelmm_optimize(perflibs_sparse_hint_value transA,
                perflibs_sparse_hint_value transB,
                perflibs_sparse_hint_value alpha, perflibs_spmat_top_t *A,
                perflibs_spmat_top_t *B, perflibs_sparse_hint_value beta,
                perflibs_spmat_top_t *C) {
  auto ret = spelmm_check_params<T>(transA, transB, alpha, A, B, beta, C);
  if (ret != PERFLIBS_STATUS_SUCCESS) {
    return ret;
  }
  // No optimizations implemented yet
  return PERFLIBS_STATUS_SUCCESS;
}
template perflibs_status_t spelmm_optimize<float>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    perflibs_sparse_hint_value alpha, perflibs_spmat_top_t *A,
    perflibs_spmat_top_t *B, perflibs_sparse_hint_value beta,
    perflibs_spmat_top_t *C);
template perflibs_status_t spelmm_optimize<double>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    perflibs_sparse_hint_value alpha, perflibs_spmat_top_t *A,
    perflibs_spmat_top_t *B, perflibs_sparse_hint_value beta,
    perflibs_spmat_top_t *C);
template perflibs_status_t spelmm_optimize<std::complex<float>>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    perflibs_sparse_hint_value alpha, perflibs_spmat_top_t *A,
    perflibs_spmat_top_t *B, perflibs_sparse_hint_value beta,
    perflibs_spmat_top_t *C);
template perflibs_status_t spelmm_optimize<std::complex<double>>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    perflibs_sparse_hint_value alpha, perflibs_spmat_top_t *A,
    perflibs_spmat_top_t *B, perflibs_sparse_hint_value beta,
    perflibs_spmat_top_t *C);

template <typename T>
perflibs_status_t spelmm_exec(perflibs_sparse_hint_value transA,
                              perflibs_sparse_hint_value transB, T alpha,
                              perflibs_spmat_top_t *A, perflibs_spmat_top_t *B,
                              T beta, perflibs_spmat_top_t *C) {
  auto ret = spelmm_check_params<T>(transA, transB, PERFLIBS_SPARSE_SCALAR_ANY,
                                    A, B, PERFLIBS_SPARSE_SCALAR_ANY, C);
  if (ret != PERFLIBS_STATUS_SUCCESS) {
    return ret;
  }

  auto impl_A = reinterpret_cast<perflibs_spmat_impl_t<T> *>(A->impl);
  auto impl_B = reinterpret_cast<perflibs_spmat_impl_t<T> *>(B->impl);
  auto impl_C = reinterpret_cast<perflibs_spmat_impl_t<T> *>(C->impl);

  // If alpha = 0 don't do element-wise multiplication
  if (alpha == (T)0) {
    if (impl_C->spmat_format == perflibs_format_null) {
      // Do nothing
      return PERFLIBS_STATUS_SUCCESS;
    } else {
      // If beta = 0 make C null
      if (beta == (T)0) {
        auto null_mat = null_matrix(impl_C->m, impl_C->n);
        *C = std::move(*null_mat);
        return PERFLIBS_STATUS_SUCCESS;
      } else {
        // Scale C by beta
        ret = scale_matrix(PERFLIBS_SPARSE_OPERATION_NOTRANS, beta, C);
        return ret;
      }
    }
  }

  // Create a new sparse matrix to store intermediate results
  auto AB = create_new_matrix<T>();

  if (impl_A->spmat_format == perflibs_format_identity &&
      impl_B->spmat_format == perflibs_format_identity) {
    // When both A and B are identity matrices, their element-wise
    // multiplication (without scaling) results in another identity matrix.
    // Scaling by alpha is applied later in the spadd_dispatch call.
    *AB = std::move(*(identity_matrix(impl_C->m)));
  } else if (impl_A->spmat_format == perflibs_format_null ||
             impl_B->spmat_format == perflibs_format_null) {
    // If A or B is null, the result of element-wise multiplication will be null
    *AB = std::move(*(null_matrix(impl_C->m, impl_C->n)));
  } else {
    switch (impl_A->spmat_format) {
    case perflibs_format_csr: {
      // Do CSR element-wise multiplication and store result in AB
      auto ret = spelmm_csr<T>(transA, impl_A, transB, impl_B, AB.get());
      if (ret != PERFLIBS_STATUS_SUCCESS) {
        return ret;
      }
      break;
    }
    case perflibs_format_csc: {
      // Do CSC element-wise multiplication and store result in AB
      auto ret = spelmm_csc<T>(transA, impl_A, transB, impl_B, AB.get());
      if (ret != PERFLIBS_STATUS_SUCCESS) {
        return ret;
      }
      break;
    }
    case perflibs_format_coo: {
      // Do COO element-wise multiplication and store result in AB
      auto ret =
          spelmm_coo<T>(transA, impl_A->coo, transB, impl_B->coo, AB.get());
      if (ret != PERFLIBS_STATUS_SUCCESS) {
        return ret;
      }
      break;
    }
    case perflibs_format_dense: {
      // If C is null, default to row-major layout
      perflibs_dense_layout layoutC =
          impl_C->spmat_format == perflibs_format_null ? PERFLIBS_ROW_MAJOR
                                                       : impl_C->dense.layout;
      auto ldaC = impl_C->spmat_format == perflibs_format_null
                      ? impl_C->n
                      : impl_C->dense.lda;

      // Do dense element-wise multiplication and store result in AB
      auto ret = spelmm_dense<T>(transA, impl_A->dense, transB, impl_B->dense,
                                 layoutC, ldaC, AB.get());
      if (ret != PERFLIBS_STATUS_SUCCESS) {
        return ret;
      }
      break;
    }
    case perflibs_format_bsr: {
      // Convert A and B to CSR format
      auto ret = convert(perflibs_format_csr, impl_A);
      if (ret != PERFLIBS_STATUS_SUCCESS) {
        return ret;
      }
      ret = convert(perflibs_format_csr, impl_B);
      if (ret != PERFLIBS_STATUS_SUCCESS) {
        return ret;
      }

      // Do CSR element-wise multiplication and store result in AB
      ret = spelmm_csr<T>(transA, impl_A, transB, impl_B, AB.get());
      if (ret != PERFLIBS_STATUS_SUCCESS) {
        return ret;
      }

      break;
    }
    default:
      // Unrecognized format
      return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    }
  }

  // Don't do addition if C is null or beta = 0
  if (impl_C->spmat_format == perflibs_format_null || beta == (T)0) {
    // Update C to be the result of the element-wise matrix multiplication of A
    // and B
    *C = std::move(*AB);
    if (alpha != (T)1) {
      // Scale C by alpha
      ret = scale_matrix(PERFLIBS_SPARSE_OPERATION_NOTRANS, alpha, C);
      if (ret != PERFLIBS_STATUS_SUCCESS) {
        return ret;
      }
    }
    return PERFLIBS_STATUS_SUCCESS;
  }

  // Calculate C_acc = alpha * AB + beta * C
  auto [ret_add, C_acc] = spadd_dispatch(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                         PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                         alpha, AB.get(), beta, C, true);
  if (ret_add != PERFLIBS_STATUS_SUCCESS) {
    return ret_add;
  }

  // Update C
  *C = std::move(*C_acc);

  return PERFLIBS_STATUS_SUCCESS;
}
template perflibs_status_t
spelmm_exec<float>(perflibs_sparse_hint_value transA,
                   perflibs_sparse_hint_value transB, float alpha,
                   perflibs_spmat_top_t *A, perflibs_spmat_top_t *B, float beta,
                   perflibs_spmat_top_t *C);
template perflibs_status_t
spelmm_exec<double>(perflibs_sparse_hint_value transA,
                    perflibs_sparse_hint_value transB, double alpha,
                    perflibs_spmat_top_t *A, perflibs_spmat_top_t *B,
                    double beta, perflibs_spmat_top_t *C);
template perflibs_status_t spelmm_exec<std::complex<float>>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    std::complex<float> alpha, perflibs_spmat_top_t *A, perflibs_spmat_top_t *B,
    std::complex<float> beta, perflibs_spmat_top_t *C);
template perflibs_status_t spelmm_exec<std::complex<double>>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    std::complex<double> alpha, perflibs_spmat_top_t *A,
    perflibs_spmat_top_t *B, std::complex<double> beta,
    perflibs_spmat_top_t *C);

// Check the correctness of matrix-matrix multiplication input parameters
template <typename T>
perflibs_status_t spmm_check_params(enum perflibs_sparse_hint_value transA,
                                    enum perflibs_sparse_hint_value transB,
                                    enum perflibs_sparse_hint_value alpha,
                                    perflibs_spmat_t A, perflibs_spmat_t B,
                                    enum perflibs_sparse_hint_value beta,
                                    perflibs_spmat_t C) {

  if (!have_compatible_matrix_datatypes(A, B, C)) {
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }

  auto impl_A = reinterpret_cast<perflibs_spmat_impl_t<T> *>(A->impl);
  auto impl_B = reinterpret_cast<perflibs_spmat_impl_t<T> *>(B->impl);
  auto impl_C = reinterpret_cast<perflibs_spmat_impl_t<T> *>(C->impl);

  perflibs_int_t c_m, c_n;

  if (!(transA == PERFLIBS_SPARSE_OPERATION_NOTRANS ||
        transA == PERFLIBS_SPARSE_OPERATION_TRANS ||
        transA == PERFLIBS_SPARSE_OPERATION_CONJTRANS)) {
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }

  if (!(transB == PERFLIBS_SPARSE_OPERATION_NOTRANS ||
        transB == PERFLIBS_SPARSE_OPERATION_TRANS ||
        transB == PERFLIBS_SPARSE_OPERATION_CONJTRANS)) {
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }

  // Check that the values of m and n for each matrix are non-negative
  // Even check C since it should at least be the null matrix if beta = 0
  if (impl_A->m < 0 || impl_A->n < 0 || impl_B->m < 0 || impl_B->n < 0 ||
      impl_C->m < 0 || impl_C->n < 0) {
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }

  // Check that the dimensions of A and B are compatible, given the transpose
  // options
  if (transA == PERFLIBS_SPARSE_OPERATION_NOTRANS &&
      transB == PERFLIBS_SPARSE_OPERATION_NOTRANS) {
    if (impl_A->n != impl_B->m) {
      return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    }
    c_m = impl_A->m;
    c_n = impl_B->n;
  } else if (transA == PERFLIBS_SPARSE_OPERATION_NOTRANS &&
             transB != PERFLIBS_SPARSE_OPERATION_NOTRANS) {
    if (impl_A->n != impl_B->n) {
      return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    }
    c_m = impl_A->m;
    c_n = impl_B->m;
  } else if (transA != PERFLIBS_SPARSE_OPERATION_NOTRANS &&
             transB == PERFLIBS_SPARSE_OPERATION_NOTRANS) {
    if (impl_A->m != impl_B->m) {
      return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    }
    c_m = impl_A->n;
    c_n = impl_B->n;
  } else if (transA != PERFLIBS_SPARSE_OPERATION_NOTRANS &&
             transB != PERFLIBS_SPARSE_OPERATION_NOTRANS) {
    if (impl_A->m != impl_B->n) {
      return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    }
    c_m = impl_A->n;
    c_n = impl_B->m;
  } else { // an invalid trans option has been provided
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }

  // Check that the dimensions of C match the result of the multiplication, c_m
  // by c_n
  if (!(impl_C->m == c_m && impl_C->n == c_n)) {
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }

  // Check that valid scalar values have been provided
  if (!(alpha == PERFLIBS_SPARSE_SCALAR_ONE ||
        alpha == PERFLIBS_SPARSE_SCALAR_ZERO ||
        alpha == PERFLIBS_SPARSE_SCALAR_ANY)) {
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }
  if (!(beta == PERFLIBS_SPARSE_SCALAR_ONE ||
        beta == PERFLIBS_SPARSE_SCALAR_ZERO ||
        beta == PERFLIBS_SPARSE_SCALAR_ANY)) {
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }

  // Work out whether the result will be materialized as dense. Whenever at
  // least one operand is dense we end up in a dense code path (GEMM, sparse *
  // dense, or dense * sparse).
  const bool result_is_dense = impl_A->spmat_format == perflibs_format_dense ||
                               impl_B->spmat_format == perflibs_format_dense;

  if (beta == PERFLIBS_SPARSE_SCALAR_ZERO) {
    // With beta == 0 we either create a new matrix (null C) or write directly
    // into a dense output buffer.
    if (impl_C->spmat_format == perflibs_format_null) {
      // Always fine: we will materialize the result in the appropriate format
      // later.
    } else if (impl_C->spmat_format == perflibs_format_dense &&
               result_is_dense) {
      // Dense output is allowed when the overall result is dense (i.e. one
      // operand is dense).
    } else {
      impl_C->error_handle.perflibs_error_type =
          PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
      impl_C->error_handle.perflibs_error_code =
          7; // C is the 7th parameter in the interface
      return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    }
  }

  return PERFLIBS_STATUS_SUCCESS;
}
template perflibs_status_t
spmm_check_params<float>(enum perflibs_sparse_hint_value transA,
                         enum perflibs_sparse_hint_value transB,
                         enum perflibs_sparse_hint_value alpha,
                         perflibs_spmat_t A, perflibs_spmat_t B,
                         enum perflibs_sparse_hint_value beta,
                         perflibs_spmat_t C);
template perflibs_status_t
spmm_check_params<double>(enum perflibs_sparse_hint_value transA,
                          enum perflibs_sparse_hint_value transB,
                          enum perflibs_sparse_hint_value alpha,
                          perflibs_spmat_t A, perflibs_spmat_t B,
                          enum perflibs_sparse_hint_value beta,
                          perflibs_spmat_t C);
template perflibs_status_t
spmm_check_params<std::complex<float>>(enum perflibs_sparse_hint_value transA,
                                       enum perflibs_sparse_hint_value transB,
                                       enum perflibs_sparse_hint_value alpha,
                                       perflibs_spmat_t A, perflibs_spmat_t B,
                                       enum perflibs_sparse_hint_value beta,
                                       perflibs_spmat_t C);
template perflibs_status_t
spmm_check_params<std::complex<double>>(enum perflibs_sparse_hint_value transA,
                                        enum perflibs_sparse_hint_value transB,
                                        enum perflibs_sparse_hint_value alpha,
                                        perflibs_spmat_t A, perflibs_spmat_t B,
                                        enum perflibs_sparse_hint_value beta,
                                        perflibs_spmat_t C);

template <typename T>
perflibs_status_t sddmm_check_params(perflibs_sparse_hint_value transA,
                                     perflibs_sparse_hint_value transB,
                                     perflibs_sparse_hint_value alpha,
                                     perflibs_spmat_t A, perflibs_spmat_t B,
                                     perflibs_sparse_hint_value beta,
                                     perflibs_spmat_t C) {

  // The same checks as for spmm apply here
  auto ret = spmm_check_params<T>(transA, transB, alpha, A, B, beta, C);
  if (ret != PERFLIBS_STATUS_SUCCESS) {
    return ret;
  }

  auto impl_A = reinterpret_cast<perflibs_spmat_impl_t<T> *>(A->impl);
  auto impl_B = reinterpret_cast<perflibs_spmat_impl_t<T> *>(B->impl);
  auto impl_C = reinterpret_cast<perflibs_spmat_impl_t<T> *>(C->impl);

  auto formatA = impl_A->spmat_format;
  auto formatB = impl_B->spmat_format;
  auto formatC = impl_C->spmat_format;

  // For now, no_copy is unsupported
  if ((formatA != perflibs_format_null && formatA != perflibs_format_identity &&
       impl_A->no_copy) ||
      (formatB != perflibs_format_null && formatB != perflibs_format_identity &&
       impl_B->no_copy) ||
      (formatC != perflibs_format_null && formatC != perflibs_format_identity &&
       impl_C->no_copy)) {
    return PERFLIBS_STATUS_EXECUTION_FAILURE;
  }

  auto m = transA == PERFLIBS_SPARSE_OPERATION_NOTRANS ? impl_A->m : impl_A->n;
  auto n = transB == PERFLIBS_SPARSE_OPERATION_NOTRANS ? impl_B->n : impl_B->m;
  auto k = transA == PERFLIBS_SPARSE_OPERATION_NOTRANS ? impl_A->n : impl_A->m;

  // Don't allow dimensions of 0
  if (m == 0 || n == 0 || k == 0) {
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }

  return PERFLIBS_STATUS_SUCCESS;
}

template perflibs_status_t sddmm_check_params<float>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    perflibs_sparse_hint_value alpha, perflibs_spmat_t A, perflibs_spmat_t B,
    perflibs_sparse_hint_value beta, perflibs_spmat_t C);
template perflibs_status_t sddmm_check_params<double>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    perflibs_sparse_hint_value alpha, perflibs_spmat_t A, perflibs_spmat_t B,
    perflibs_sparse_hint_value beta, perflibs_spmat_t C);
template perflibs_status_t sddmm_check_params<std::complex<float>>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    perflibs_sparse_hint_value alpha, perflibs_spmat_t A, perflibs_spmat_t B,
    perflibs_sparse_hint_value beta, perflibs_spmat_t C);
template perflibs_status_t sddmm_check_params<std::complex<double>>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    perflibs_sparse_hint_value alpha, perflibs_spmat_t A, perflibs_spmat_t B,
    perflibs_sparse_hint_value beta, perflibs_spmat_t C);

template <typename T>
perflibs_status_t sddmm_sampled_mm(perflibs_sparse_hint_value transA,
                                   perflibs_sparse_hint_value transB, T alpha,
                                   perflibs_spmat_t A, perflibs_spmat_t B,
                                   T beta, perflibs_spmat_t C) {
  auto impl_A = reinterpret_cast<perflibs_spmat_impl_t<T> *>(A->impl);
  auto impl_B = reinterpret_cast<perflibs_spmat_impl_t<T> *>(B->impl);
  auto impl_C = reinterpret_cast<perflibs_spmat_impl_t<T> *>(C->impl);

  // Convert C to CSR format if required
  if (impl_C->spmat_format != perflibs_format_csr) {
    convert(perflibs_format_csr, impl_C);
  }

  // Convert A and B to dense format if required
  if (impl_A->spmat_format != perflibs_format_dense) {
    convert(perflibs_format_dense, impl_A);
  }
  if (impl_B->spmat_format != perflibs_format_dense) {
    convert(perflibs_format_dense, impl_B);
  }

  // Create a new CSR matrix for sampled(AB)
  auto AB = create_new_matrix<T>();

  // Calculate sampled AB
  auto info =
      sddmm_csr(transA, transB, alpha, impl_A, impl_B, beta, impl_C, AB.get());
  if (info != PERFLIBS_STATUS_SUCCESS) {
    return info;
  }

  // Do the final computation: alpha * sampled(AB) + beta * C
  auto [ret_add, C_acc] = spadd_dispatch(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                         PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                         alpha, AB.get(), beta, C, true);
  if (ret_add != PERFLIBS_STATUS_SUCCESS) {
    return ret_add;
  }

  // Update C
  *C = std::move(*C_acc);

  return PERFLIBS_STATUS_SUCCESS;
}

template <typename T>
perflibs_status_t sddmm_gemm(perflibs_sparse_hint_value transA,
                             perflibs_sparse_hint_value transB, T alpha,
                             perflibs_spmat_t A, perflibs_spmat_t B, T beta,
                             perflibs_spmat_t C) {
  auto impl_A = reinterpret_cast<perflibs_spmat_impl_t<T> *>(A->impl);
  auto impl_B = reinterpret_cast<perflibs_spmat_impl_t<T> *>(B->impl);
  auto impl_C = reinterpret_cast<perflibs_spmat_impl_t<T> *>(C->impl);

  // Convert A and B to dense format if required
  if (impl_A->spmat_format != perflibs_format_dense) {
    convert(perflibs_format_dense, impl_A);
  }
  if (impl_B->spmat_format != perflibs_format_dense) {
    convert(perflibs_format_dense, impl_B);
  }

  // Fall back to alternative implementation if layouts of A and B don't match
  if (impl_A->dense.layout != impl_B->dense.layout) {
    return sddmm_sampled_mm(transA, transB, alpha, A, B, beta, C);
  }

  // Convert C to dense format if required
  if (impl_C->spmat_format != perflibs_format_dense) {
    convert(perflibs_format_dense, impl_C);
  }

  auto m = impl_C->m;
  auto n = impl_C->n;
  auto k = transA == PERFLIBS_SPARSE_OPERATION_NOTRANS ? impl_A->n : impl_A->m;

  // Copy C to C_result
  // The layout of C_result needs to match that of A and B so transpose if
  // necessary
  perflibs::sparse::pod_vector<T> C_result(m * n);
  auto layoutC = impl_C->dense.layout;
  auto layoutC_result = impl_A->dense.layout;
  auto ldC = impl_C->dense.lda;
  auto ldC_result = layoutC_result == PERFLIBS_COL_MAJOR ? m : n;
  for (perflibs_int_t i = 0; i < m; i++) {
    for (perflibs_int_t j = 0; j < n; j++) {
      auto indxC = layoutC == PERFLIBS_COL_MAJOR ? i + j * ldC : i * ldC + j;
      auto indxC_result = layoutC_result == PERFLIBS_COL_MAJOR
                              ? i + j * ldC_result
                              : i * ldC_result + j;
      C_result[indxC_result] = impl_C->dense.vals_ptr[indxC];
    }
  }

  // Perform dense matrix multiplication: C_result = alpha * A * B + beta *
  // C_result
  auto ret = spmm_gemm<T>(impl_A->dense.layout, transA, transB, m, n, k, alpha,
                          impl_A->dense.vals_ptr, impl_A->dense.lda,
                          impl_B->dense.vals_ptr, impl_B->dense.lda, beta,
                          C_result.data(), ldC_result);
  if (ret != PERFLIBS_STATUS_SUCCESS) {
    return ret;
  }

  // Match the sparsity pattern of C_result to the original C
  for (perflibs_int_t i = 0; i < m; i++) {
    for (perflibs_int_t j = 0; j < n; j++) {
      auto indxC = layoutC == PERFLIBS_COL_MAJOR ? i + j * ldC : i * ldC + j;
      if (impl_C->dense.vals_ptr[indxC] == (T)0) {
        auto indxC_result = layoutC_result == PERFLIBS_COL_MAJOR
                                ? i + j * ldC_result
                                : i * ldC_result + j;
        C_result[indxC_result] = (T)0;
      }
    }
  }

  // Create a new matrix from the result
  auto AB = create_new_matrix<T>();
  ret = perflibs::sparse::fill_initial_data_dense(
      AB.get(), impl_A->dense.layout, m, n, ldC_result, 0, C_result.data(), 0);
  if (ret != PERFLIBS_STATUS_SUCCESS) {
    return ret;
  }

  *C = std::move(*AB);

  return PERFLIBS_STATUS_SUCCESS;
}

template <typename T>
perflibs_status_t sddmm_exec_checked(perflibs_sparse_hint_value transA,
                                     perflibs_sparse_hint_value transB, T alpha,
                                     perflibs_spmat_t A, perflibs_spmat_t B,
                                     T beta, perflibs_spmat_t C, bool use_gemm,
                                     bool exec) {
  auto impl_A = reinterpret_cast<perflibs_spmat_impl_t<T> *>(A->impl);
  auto impl_B = reinterpret_cast<perflibs_spmat_impl_t<T> *>(B->impl);
  auto impl_C = reinterpret_cast<perflibs_spmat_impl_t<T> *>(C->impl);

  if (impl_C->spmat_format == perflibs_format_null) {
    // If there are no non-zeros in the input matrix C, the result is null
    // If sddmm_optimized is set, no need to recompute C
    return PERFLIBS_STATUS_SUCCESS;
  }

  if (alpha == (T)0 || impl_A->spmat_format == perflibs_format_null ||
      impl_B->spmat_format == perflibs_format_null) {
    if (!exec) {
      return PERFLIBS_STATUS_SUCCESS;
    }
    auto ret = scale_matrix(PERFLIBS_SPARSE_OPERATION_NOTRANS, beta, C);
    return ret;
  } else if (alpha == (T)1 &&
             impl_A->spmat_format == perflibs_format_identity &&
             impl_B->spmat_format == perflibs_format_identity &&
             impl_C->spmat_format == perflibs_format_identity) {
    if (!exec) {
      return PERFLIBS_STATUS_SUCCESS;
    }
    auto ret = scale_matrix(PERFLIBS_SPARSE_OPERATION_NOTRANS, beta + (T)1, C);
    return ret;
  }

  if (use_gemm || impl_C->perflibs_sddmm_use_gemm) {
    if (!exec) {
      impl_C->perflibs_sddmm_use_gemm = true;
      return PERFLIBS_STATUS_SUCCESS;
    }
    return sddmm_gemm(transA, transB, alpha, A, B, beta, C);
  }

  if (!exec) {
    return PERFLIBS_STATUS_SUCCESS;
  }

  return sddmm_sampled_mm(transA, transB, alpha, A, B, beta, C);
}
template perflibs_status_t
sddmm_exec_checked<float>(perflibs_sparse_hint_value transA,
                          perflibs_sparse_hint_value transB, float alpha,
                          perflibs_spmat_t A, perflibs_spmat_t B, float beta,
                          perflibs_spmat_t C, bool use_gemm, bool exec);
template perflibs_status_t
sddmm_exec_checked<double>(perflibs_sparse_hint_value transA,
                           perflibs_sparse_hint_value transB, double alpha,
                           perflibs_spmat_t A, perflibs_spmat_t B, double beta,
                           perflibs_spmat_t C, bool use_gemm, bool exec);
template perflibs_status_t sddmm_exec_checked<std::complex<float>>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    std::complex<float> alpha, perflibs_spmat_t A, perflibs_spmat_t B,
    std::complex<float> beta, perflibs_spmat_t C, bool use_gemm, bool exec);
template perflibs_status_t sddmm_exec_checked<std::complex<double>>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    std::complex<double> alpha, perflibs_spmat_t A, perflibs_spmat_t B,
    std::complex<double> beta, perflibs_spmat_t C, bool use_gemm, bool exec);

template <typename T, bool use_gemm>
perflibs_status_t sddmm_exec(perflibs_sparse_hint_value transA,
                             perflibs_sparse_hint_value transB, T alpha,
                             perflibs_spmat_t A, perflibs_spmat_t B, T beta,
                             perflibs_spmat_t C) {
  auto ret = sddmm_check_params<T>(transA, transB, PERFLIBS_SPARSE_SCALAR_ANY,
                                   A, B, PERFLIBS_SPARSE_SCALAR_ANY, C);
  if (ret != PERFLIBS_STATUS_SUCCESS) {
    return ret;
  }

  return sddmm_exec_checked<T>(transA, transB, alpha, A, B, beta, C, use_gemm,
                               true);
}
template perflibs_status_t
sddmm_exec<float, true>(perflibs_sparse_hint_value transA,
                        perflibs_sparse_hint_value transB, float alpha,
                        perflibs_spmat_t A, perflibs_spmat_t B, float beta,
                        perflibs_spmat_t C);
template perflibs_status_t
sddmm_exec<float, false>(perflibs_sparse_hint_value transA,
                         perflibs_sparse_hint_value transB, float alpha,
                         perflibs_spmat_t A, perflibs_spmat_t B, float beta,
                         perflibs_spmat_t C);
template perflibs_status_t
sddmm_exec<double, true>(perflibs_sparse_hint_value transA,
                         perflibs_sparse_hint_value transB, double alpha,
                         perflibs_spmat_t A, perflibs_spmat_t B, double beta,
                         perflibs_spmat_t C);
template perflibs_status_t
sddmm_exec<double, false>(perflibs_sparse_hint_value transA,
                          perflibs_sparse_hint_value transB, double alpha,
                          perflibs_spmat_t A, perflibs_spmat_t B, double beta,
                          perflibs_spmat_t C);
template perflibs_status_t sddmm_exec<std::complex<float>, true>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    std::complex<float> alpha, perflibs_spmat_t A, perflibs_spmat_t B,
    std::complex<float> beta, perflibs_spmat_t C);
template perflibs_status_t sddmm_exec<std::complex<float>, false>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    std::complex<float> alpha, perflibs_spmat_t A, perflibs_spmat_t B,
    std::complex<float> beta, perflibs_spmat_t C);
template perflibs_status_t sddmm_exec<std::complex<double>, true>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    std::complex<double> alpha, perflibs_spmat_t A, perflibs_spmat_t B,
    std::complex<double> beta, perflibs_spmat_t C);
template perflibs_status_t sddmm_exec<std::complex<double>, false>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    std::complex<double> alpha, perflibs_spmat_t A, perflibs_spmat_t B,
    std::complex<double> beta, perflibs_spmat_t C);

template <typename T>
perflibs_status_t
sddmm_optimize(perflibs_sparse_hint_value transA,
               perflibs_sparse_hint_value transB,
               perflibs_sparse_hint_value alpha, perflibs_spmat_top_t *A,
               perflibs_spmat_top_t *B, perflibs_sparse_hint_value beta,
               perflibs_spmat_top_t *C) {
  auto ret = sddmm_check_params<T>(transA, transB, alpha, A, B, beta, C);
  if (ret != PERFLIBS_STATUS_SUCCESS) {
    return ret;
  }

  auto impl_C = reinterpret_cast<perflibs_spmat_impl_t<T> *>(C->impl);

  if (is_special(impl_C->spmat_format)) {
    return PERFLIBS_STATUS_SUCCESS;
  }

  if (impl_C->spmat_format == perflibs_format_csr && impl_C->csr.vals.empty()) {
    return PERFLIBS_STATUS_SUCCESS;
  }

  if (impl_C->spmat_format == perflibs_format_csc && impl_C->csc.vals.empty()) {
    return PERFLIBS_STATUS_SUCCESS;
  }

  if (impl_C->spmat_format == perflibs_format_coo && impl_C->coo.vals.empty()) {
    return PERFLIBS_STATUS_SUCCESS;
  }

  if (impl_C->spmat_format == perflibs_format_bsr && impl_C->bsr.vals.empty()) {
    return PERFLIBS_STATUS_SUCCESS;
  }

  perflibs_int_t m = impl_C->m;
  perflibs_int_t n = impl_C->n;
  perflibs_int_t nnz = impl_C->nnz;

  // Choose whether to perform the GEMM implementation based on the density of
  // the input matrix C.
  bool do_gemm = false;
  if (nnz > m * n * 0.25) {
    do_gemm = true;
  }

  // Create some representative values for the optimization
  T alpha_ = alpha == PERFLIBS_SPARSE_SCALAR_ZERO  ? (T)0
             : alpha == PERFLIBS_SPARSE_SCALAR_ONE ? (T)1
                                                   : (T)1.5;
  T beta_ = beta == PERFLIBS_SPARSE_SCALAR_ZERO  ? (T)0
            : beta == PERFLIBS_SPARSE_SCALAR_ONE ? (T)1
                                                 : (T)1.5;
  return sddmm_exec_checked<T>(transA, transB, alpha_, A, B, beta_, C, do_gemm,
                               false);
}
template perflibs_status_t
sddmm_optimize<float>(perflibs_sparse_hint_value transA,
                      perflibs_sparse_hint_value transB,
                      perflibs_sparse_hint_value alpha, perflibs_spmat_top_t *A,
                      perflibs_spmat_top_t *B, perflibs_sparse_hint_value beta,
                      perflibs_spmat_top_t *C);
template perflibs_status_t sddmm_optimize<double>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    perflibs_sparse_hint_value alpha, perflibs_spmat_top_t *A,
    perflibs_spmat_top_t *B, perflibs_sparse_hint_value beta,
    perflibs_spmat_top_t *C);
template perflibs_status_t sddmm_optimize<std::complex<float>>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    perflibs_sparse_hint_value alpha, perflibs_spmat_top_t *A,
    perflibs_spmat_top_t *B, perflibs_sparse_hint_value beta,
    perflibs_spmat_top_t *C);
template perflibs_status_t sddmm_optimize<std::complex<double>>(
    perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
    perflibs_sparse_hint_value alpha, perflibs_spmat_top_t *A,
    perflibs_spmat_top_t *B, perflibs_sparse_hint_value beta,
    perflibs_spmat_top_t *C);

} // namespace perflibs::sparse
