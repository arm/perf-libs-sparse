/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#pragma once

struct perflibs_spmat_top_t;
struct perflibs_spvec_top_t;

#include "block_sparse_rows.hpp"
#include "compressed_sparse_columns.hpp"
#include "compressed_sparse_rows.hpp"
#include "coordinate_list.hpp"
#include "dense.hpp"
#include "matrix_statistics.hpp"
#include "sell_c_sigma.hpp"
#include "supernodal.hpp"
#include "util.hpp"
#include "vector_storage.hpp"

#include <memory>

/** @file types.hpp
 * Hidden sparse matrix and sparse vector implementation types used behind the
 * public user-facing handles.
 */

template <typename T> struct perflibs_spmat_impl_t {

  perflibs_int_t m;
  perflibs_int_t n;
  perflibs_int_t index_base;
  perflibs_int_t nnz;
  perflibs_sparse_matrix_shape_t shape;
  perflibs_sparse_matrix_diag_t diag;

  /* Supplied hints */
  perflibs_sparse_hint_value userhint_structure =
      PERFLIBS_SPARSE_STRUCTURE_UNSTRUCTURED;
  perflibs_sparse_hint_value userhint_memory = PERFLIBS_SPARSE_MEMORY_ALLOCS;

  perflibs_sparse_hint_value userhint_spmv_op =
      PERFLIBS_SPARSE_OPERATION_NOTRANS;
  perflibs_sparse_hint_value userhint_spmv_invocations =
      PERFLIBS_SPARSE_INVOCATIONS_MANY;

  perflibs_sparse_hint_value userhint_spmm_op =
      PERFLIBS_SPARSE_OPERATION_NOTRANS;
  perflibs_sparse_hint_value userhint_spmm_invocations =
      PERFLIBS_SPARSE_INVOCATIONS_MANY;

  perflibs_sparse_hint_value userhint_spadd_op =
      PERFLIBS_SPARSE_OPERATION_NOTRANS;
  perflibs_sparse_hint_value userhint_spadd_invocations =
      PERFLIBS_SPARSE_INVOCATIONS_MANY;

  perflibs_sparse_hint_value userhint_spmm_strat =
      PERFLIBS_SPARSE_SPMM_STRAT_UNSET;

  perflibs_sparse_hint_value userhint_spsm_op =
      PERFLIBS_SPARSE_OPERATION_NOTRANS;
  perflibs_sparse_hint_value userhint_spsm_invocations =
      PERFLIBS_SPARSE_INVOCATIONS_MANY;

  perflibs_sparse_hint_value userhint_spsv_op =
      PERFLIBS_SPARSE_OPERATION_NOTRANS;
  perflibs_sparse_hint_value userhint_spsv_invocations =
      PERFLIBS_SPARSE_INVOCATIONS_MANY;

  perflibs_sparse_hint_value userhint_spsv_strat =
      PERFLIBS_SPARSE_SPSV_STRAT_UNSET;

  bool userhint_hpcg = false;
  /// A flag to say what type of optimization has actually been performed
  /// previously for spmm (as opposed to what was requested, which is the hint
  /// userhint_spmm_strat)
  perflibs_spmm_opt_t optimized_mm = perflibs_spmm_single_phase;

  // Flag to say which SDDMM implementation to use
  bool perflibs_sddmm_use_gemm = false;

  /* Error handling */
  sp_error_t error_handle = {};

  /// Time limit for auditioning
  double audition_tl = 1.00;

  /// Force C params for SCS
  int use_C = -1;
  /// Force sigma params for SCS
  int use_sigma = -1;

  /* Sparse format */
  spmat_format_t spmat_format;

  // Has the user requested NO_COPY on matrix creation?
  // If so, then the representation below will reference
  // user arrays. Note: we currently break this sometimes,
  // so keeping track explicitly should help us avoid
  // doing so.
  bool no_copy;

  /*
      Internal representations
      ------------------------
      Note that exactly one of these is valid at any given point, and which
      is given by spmat_format above. Once an optimization decision has been
      made to transform from the input representation to another then there is
     no going back. The upshot of this is that the spmat_update function only
      has to update whatever is the current valid structure.
  */
  perflibs::sparse::perflibs_coo<T> coo;
  perflibs::sparse::perflibs_csr<T> csr;
  perflibs::sparse::perflibs_csc<T> csc;
  perflibs::sparse::perflibs_dense<T> dense;
  perflibs::sparse::perflibs_scs<T> scs;
  perflibs::sparse::perflibs_bsr<T> bsr;
  perflibs::sparse::perflibs_supernodal<T> supernodal;

  // Statistics to support optimization choices
  perflibs::sparse::sparse_matrix_statistics stats = {-1.0};

  /**
   * Populate the matrix stats of the underlying format with
   * mgmd statistic. This is formed by taking, for each row,
   * the geometric mean of distances between non-zero elements,
   * and then taking the arithmetic mean of those over all rows.
   * A small value indicates that non-zeros are, on average,
   * clustered together, in theory making vectorization more
   * efficient.
   *
   * At present, this is only implemented for CSR matrices,
   * since it is currently only used for SpMM, in which we
   * always use CSR.
   */
  void fill_matrix_stats_mgmd();
};

namespace perflibs::sparse {
perflibs_status_t perflibs_spmat_destroy_impl(perflibs_spmat_top_t *A);
perflibs_status_t perflibs_spvec_destroy_impl(perflibs_spvec_top_t *x);
} // namespace perflibs::sparse

// We need to be able to identify the type of the non-templated thing
// passed down to e.g. perflibs_spmat_destroy
struct perflibs_spmat_top_t {
  perflibs_datatype datatype;
  void *impl = nullptr; // this will actually be a pointer to the templated
                        // perflibs_spmat_impl_t above

  perflibs_spmat_top_t() = default;
  perflibs_spmat_top_t(const perflibs_spmat_top_t &) = delete;
  perflibs_spmat_top_t(perflibs_spmat_top_t &&other)
      : datatype(other.datatype), impl(other.impl) {
    other.impl = nullptr;
  }

  ~perflibs_spmat_top_t() {
    perflibs::sparse::perflibs_spmat_destroy_impl(this);
  }

  // Delete the copy-operator; use perflibs::sparse::spmat_copy instead
  perflibs_spmat_top_t operator=(const perflibs_spmat_top_t &other) = delete;

  // Move-assignment operator
  perflibs_spmat_top_t &operator=(perflibs_spmat_top_t &&other) {
    if (this != &other) {
      perflibs::sparse::perflibs_spmat_destroy_impl(
          this); // delete the impl we're overwriting first
      this->datatype = other.datatype;
      this->impl = other.impl;
      other.impl = nullptr;
    }
    return *this;
  }
};

template <typename T> struct perflibs_spvec_impl_t {

  perflibs_int_t index_base;
  perflibs_int_t n;
  perflibs_int_t nnz;

  /* Error handling */
  sp_error_t error_handle;

  /*
      Internal representations
      ------------------------
  */
  perflibs::sparse::perflibs_vec<T> vec;
};

// We need to be able to identify the type of the non-templated thing
// passed down to e.g. perflibs_spvec_destroy
struct perflibs_spvec_top_t {
  perflibs_datatype datatype;
  void *impl = nullptr; // this will actually be a pointer to the templated
                        // perflibs_spvec_impl_t above

  perflibs_spvec_top_t() = default;
  perflibs_spvec_top_t(const perflibs_spvec_top_t &) = delete;
  perflibs_spvec_top_t(perflibs_spvec_top_t &&other)
      : datatype(other.datatype), impl(other.impl) {
    other.impl = nullptr;
  }

  ~perflibs_spvec_top_t() {
    perflibs::sparse::perflibs_spvec_destroy_impl(this);
  }

  // Delete the copy-operator; use perflibs::sparse::spvec_copy instead
  perflibs_spvec_top_t operator=(const perflibs_spvec_top_t &other) = delete;

  // Move-assignment operator
  perflibs_spvec_top_t &operator=(perflibs_spvec_top_t &&other) {
    if (this != &other) {
      perflibs::sparse::perflibs_spvec_destroy_impl(
          this); // delete the impl we're overwriting first
      this->datatype = other.datatype;
      this->impl = other.impl;
      other.impl = nullptr;
    }
    return *this;
  }
};
