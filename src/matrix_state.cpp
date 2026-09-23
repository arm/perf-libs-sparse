/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "matrix_state.hpp"
#include "compressed_sparse_columns.hpp"
#include "compressed_sparse_rows.hpp"
#include "convert.hpp"
#include "object_helpers.hpp"
#include "types.hpp"
#include "util.hpp"

#include <cassert>
#include <complex>
#include <cstddef>
#include <utility>

namespace perflibs::sparse {

static inline bool is_typeless_placeholder(perflibs_const_spmat_t A) {
  // Null and identity matrices are always template-typed as single precision.
  if (A->datatype != PERFLIBS_DATATYPE_SINGLE) {
    return false;
  }

  auto impl = reinterpret_cast<const perflibs_spmat_impl_t<float> *>(A->impl);
  return is_special(impl->spmat_format);
}

static inline bool have_compatible_datatypes(perflibs_const_spmat_t A,
                                             perflibs_const_spmat_t B) {
  return is_typeless_placeholder(A) || is_typeless_placeholder(B) ||
         A->datatype == B->datatype;
}

std::unique_ptr<perflibs_spmat_top_t> null_matrix(perflibs_int_t m,
                                                  perflibs_int_t n) {

  if (n < 0 || m < 0) {
    return NULL;
  }

  // The underlying datatype doesn't matter in this case, so use float
  // arbitrarily
  auto Atop = create_new_matrix<float>();
  auto impl = reinterpret_cast<perflibs_spmat_impl_t<float> *>(Atop->impl);
  impl->m = m;
  impl->n = n;
  impl->nnz = 0;
  impl->index_base = 0;
  impl->spmat_format = perflibs_format_null;
  Atop->impl = impl;

  return Atop;
}

std::unique_ptr<perflibs_spmat_top_t> identity_matrix(perflibs_int_t n) {

  if (n < 0) {
    return NULL;
  }

  // The underlying datatype doesn't matter in this case, so use float
  // arbitrarily
  auto Atop = create_new_matrix<float>();
  auto impl = reinterpret_cast<perflibs_spmat_impl_t<float> *>(Atop->impl);
  impl->m = n;
  impl->n = n;
  impl->nnz = n;
  impl->index_base = 0;
  impl->spmat_format = perflibs_format_identity;
  Atop->impl = impl;

  return Atop;
}

bool have_compatible_matrix_datatypes(perflibs_const_spmat_t A,
                                      perflibs_const_spmat_t B,
                                      perflibs_const_spmat_t C) {
  return have_compatible_datatypes(A, B) && have_compatible_datatypes(A, C) &&
         have_compatible_datatypes(B, C);
}

template <typename T>
perflibs_status_t scale_matrix(perflibs_sparse_hint_value trans, T alpha,
                               perflibs_spmat_t A) {
  auto impl_A = reinterpret_cast<perflibs_spmat_impl_t<T> *>(A->impl);
  auto m = impl_A->m;
  auto n = impl_A->n;

  switch (impl_A->spmat_format) {
  case (perflibs_format_null):
    if (trans != PERFLIBS_SPARSE_OPERATION_NOTRANS) {
      std::swap(impl_A->m, impl_A->n);
    }
    return PERFLIBS_STATUS_SUCCESS;
  case (perflibs_format_coo):
    impl_A->coo.scale_matrix(trans, alpha);
    if (trans != PERFLIBS_SPARSE_OPERATION_NOTRANS) {
      std::swap(impl_A->m, impl_A->n);
    }
    break;
  case (perflibs_format_bsr):
    if (trans == PERFLIBS_SPARSE_OPERATION_NOTRANS) {
      impl_A->bsr.scale_matrix(alpha);
      break;
    }
  case (perflibs_format_identity):
  case (perflibs_format_scs):
    convert(perflibs_format_csr,
            impl_A); // Convert to CSR and then fall through to handle as CSR in
                     // the next case
  case (perflibs_format_csr):
    // Represent as a CSC matrix if we want the transpose
    if (trans != PERFLIBS_SPARSE_OPERATION_NOTRANS) {
      auto info = fill_initial_data_csc(A, n, m, impl_A->csr.col_indx_ptr,
                                        impl_A->csr.row_ptr_ptr,
                                        impl_A->csr.vals_ptr, 0);
      if (info != PERFLIBS_STATUS_SUCCESS) {
        return PERFLIBS_STATUS_EXECUTION_FAILURE;
      }
      impl_A->csr = {};
      impl_A->spmat_format = perflibs_format_csc;
      impl_A->csc.scale_matrix(trans, alpha);
    } else {
      impl_A->csr.scale_matrix(trans, alpha);
    }
    break;
  case (perflibs_format_csc):
    // Represent as a CSR matrix if we want the transpose
    if (trans != PERFLIBS_SPARSE_OPERATION_NOTRANS) {
      auto info = fill_initial_data_csr(A, n, m, impl_A->csc.col_ptr_ptr,
                                        impl_A->csc.row_indx_ptr,
                                        impl_A->csc.vals_ptr, 0);
      if (info != PERFLIBS_STATUS_SUCCESS) {
        return PERFLIBS_STATUS_EXECUTION_FAILURE;
      }
      impl_A->csc = {};
      impl_A->spmat_format = perflibs_format_csr;
      impl_A->csr.scale_matrix(trans, alpha);
    } else {
      impl_A->csc.scale_matrix(trans, alpha);
    }
    break;
  case (perflibs_format_dense):
    if (trans != PERFLIBS_SPARSE_OPERATION_NOTRANS) {
      impl_A->dense.transpose(); // This just swaps the layout
      std::swap(impl_A->m, impl_A->n);
    }
    impl_A->dense.scale_matrix(trans, alpha);
    break;
  case (perflibs_format_supernodal):
    assert(false);
  }

  return PERFLIBS_STATUS_SUCCESS;
}
template perflibs_status_t scale_matrix(perflibs_sparse_hint_value trans,
                                        float alpha, perflibs_spmat_t A);
template perflibs_status_t scale_matrix(perflibs_sparse_hint_value trans,
                                        double alpha, perflibs_spmat_t A);
template perflibs_status_t scale_matrix(perflibs_sparse_hint_value trans,
                                        std::complex<float> alpha,
                                        perflibs_spmat_t A);
template perflibs_status_t scale_matrix(perflibs_sparse_hint_value trans,
                                        std::complex<double> alpha,
                                        perflibs_spmat_t A);

perflibs_status_t basic_query(perflibs_spmat_top_t *A,
                              perflibs_int_t *index_base, perflibs_int_t *m,
                              perflibs_int_t *n, perflibs_int_t *nnz) {
  // The underlying datatype doesn't matter in this case, so use float
  // arbitrarily
  auto impl = reinterpret_cast<perflibs_spmat_impl_t<float> *>(A->impl);
  if (index_base) {
    *index_base = impl->index_base;
  }
  if (m) {
    *m = impl->m;
  }
  if (n) {
    *n = impl->n;
  }
  if (nnz) {
    *nnz = impl->nnz;
  }
  return PERFLIBS_STATUS_SUCCESS;
}

perflibs_status_t basic_query(perflibs_spvec_top_t *x,
                              perflibs_int_t *index_base, perflibs_int_t *n,
                              perflibs_int_t *nnz) {
  // The underlying datatype doesn't matter in this case, so use float
  // arbitrarily
  auto impl = reinterpret_cast<perflibs_spvec_impl_t<float> *>(x->impl);
  if (index_base) {
    *index_base = impl->index_base;
  }
  if (n) {
    *n = impl->n;
  }
  if (nnz) {
    *nnz = impl->nnz;
  }
  return PERFLIBS_STATUS_SUCCESS;
}

perflibs_status_t perflibs_spmat_destroy_impl(perflibs_spmat_top_t *A) {
  if (A->impl != nullptr) {
    if (A->datatype == PERFLIBS_DATATYPE_SINGLE) {
      delete reinterpret_cast<perflibs_spmat_impl_t<float> *>(A->impl);
    } else if (A->datatype == PERFLIBS_DATATYPE_DOUBLE) {
      delete reinterpret_cast<perflibs_spmat_impl_t<double> *>(A->impl);
    } else if (A->datatype == PERFLIBS_DATATYPE_CPLXSINGLE) {
      delete reinterpret_cast<perflibs_spmat_impl_t<std::complex<float>> *>(
          A->impl);
    } else if (A->datatype == PERFLIBS_DATATYPE_CPLXDOUBLE) {
      delete reinterpret_cast<perflibs_spmat_impl_t<std::complex<double>> *>(
          A->impl);
    }
  }
  return PERFLIBS_STATUS_SUCCESS;
}

perflibs_status_t perflibs_spvec_destroy_impl(perflibs_spvec_top_t *x) {
  if (x->impl != nullptr) {
    if (x->datatype == PERFLIBS_DATATYPE_SINGLE) {
      delete reinterpret_cast<perflibs_spvec_impl_t<float> *>(x->impl);
    } else if (x->datatype == PERFLIBS_DATATYPE_DOUBLE) {
      delete reinterpret_cast<perflibs_spvec_impl_t<double> *>(x->impl);
    } else if (x->datatype == PERFLIBS_DATATYPE_CPLXSINGLE) {
      delete reinterpret_cast<perflibs_spvec_impl_t<std::complex<float>> *>(
          x->impl);
    } else if (x->datatype == PERFLIBS_DATATYPE_CPLXDOUBLE) {
      delete reinterpret_cast<perflibs_spvec_impl_t<std::complex<double>> *>(
          x->impl);
    }
  }
  return PERFLIBS_STATUS_SUCCESS;
}

template <typename T>
perflibs_status_t set_hint(perflibs_spmat_impl_t<T> *impl,
                           perflibs_sparse_hint_type hint,
                           perflibs_sparse_hint_value value) {

  /* Check which hint has been supplied */
  switch (hint) {

  case PERFLIBS_SPARSE_HINT_STRUCTURE:
    if (value == PERFLIBS_SPARSE_STRUCTURE_DENSE ||
        value == PERFLIBS_SPARSE_STRUCTURE_UNSTRUCTURED ||
        value == PERFLIBS_SPARSE_STRUCTURE_SYMMETRIC ||
        value == PERFLIBS_SPARSE_STRUCTURE_DIAGONAL ||
        value == PERFLIBS_SPARSE_STRUCTURE_BLOCKDIAGONAL ||
        value == PERFLIBS_SPARSE_STRUCTURE_BANDED ||
        value == PERFLIBS_SPARSE_STRUCTURE_TRIANGULAR ||
        value == PERFLIBS_SPARSE_STRUCTURE_TRIANGULAR ||
        value == PERFLIBS_SPARSE_STRUCTURE_BLOCKTRIANGULAR ||
        value == PERFLIBS_SPARSE_STRUCTURE_HERMITIAN) {
      impl->userhint_structure = value;
    } else if (PERFLIBS_SPARSE_STRUCTURE_HPCG == value) {
      impl->userhint_hpcg = value;
    } else {
      return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    }
    break;

  case PERFLIBS_SPARSE_HINT_MEMORY:
    if (value == PERFLIBS_SPARSE_MEMORY_NOALLOCS ||
        value == PERFLIBS_SPARSE_MEMORY_ALLOCS) {
      impl->userhint_memory = value;
    } else {
      return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    }
    break;

  case PERFLIBS_SPARSE_HINT_SPMV_OPERATION:
    if (value == PERFLIBS_SPARSE_OPERATION_NOTRANS ||
        value == PERFLIBS_SPARSE_OPERATION_TRANS ||
        value == PERFLIBS_SPARSE_OPERATION_CONJTRANS) {
      impl->userhint_spmv_op = value;
    } else {
      return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    }
    break;

  case PERFLIBS_SPARSE_HINT_SPMV_INVOCATIONS:
    if (value == PERFLIBS_SPARSE_INVOCATIONS_SINGLE ||
        value == PERFLIBS_SPARSE_INVOCATIONS_FEW ||
        value == PERFLIBS_SPARSE_INVOCATIONS_MANY) {
      impl->userhint_spmv_invocations = value;
    } else {
      return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    }
    break;

  case PERFLIBS_SPARSE_HINT_SPMM_OPERATION:
    if (value == PERFLIBS_SPARSE_OPERATION_NOTRANS ||
        value == PERFLIBS_SPARSE_OPERATION_TRANS ||
        value == PERFLIBS_SPARSE_OPERATION_CONJTRANS) {
      impl->userhint_spmm_op = value;
    } else {
      return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    }
    break;

  case PERFLIBS_SPARSE_HINT_SPMM_INVOCATIONS:
    if (value == PERFLIBS_SPARSE_INVOCATIONS_SINGLE ||
        value == PERFLIBS_SPARSE_INVOCATIONS_FEW ||
        value == PERFLIBS_SPARSE_INVOCATIONS_MANY) {
      impl->userhint_spmm_invocations = value;
    } else {
      return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    }
    break;

  case PERFLIBS_SPARSE_HINT_SPADD_OPERATION:
    if (value == PERFLIBS_SPARSE_OPERATION_NOTRANS ||
        value == PERFLIBS_SPARSE_OPERATION_TRANS ||
        value == PERFLIBS_SPARSE_OPERATION_CONJTRANS) {
      impl->userhint_spadd_op = value;
    } else {
      return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    }
    break;

  case PERFLIBS_SPARSE_HINT_SPADD_INVOCATIONS:
    if (value == PERFLIBS_SPARSE_INVOCATIONS_SINGLE ||
        value == PERFLIBS_SPARSE_INVOCATIONS_FEW ||
        value == PERFLIBS_SPARSE_INVOCATIONS_MANY) {
      impl->userhint_spadd_invocations = value;
    } else {
      return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    }
    break;

  case PERFLIBS_SPARSE_HINT_SPMM_STRATEGY:
    if (value == PERFLIBS_SPARSE_SPMM_STRAT_UNSET ||
        value == PERFLIBS_SPARSE_SPMM_STRAT_OPT_NO_STRUCT ||
        value == PERFLIBS_SPARSE_SPMM_STRAT_OPT_PART_STRUCT ||
        value == PERFLIBS_SPARSE_SPMM_STRAT_OPT_FULL_STRUCT) {
      impl->userhint_spmm_strat = value;
    } else {
      return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    }
    break;

  case PERFLIBS_SPARSE_HINT_SPSM_OPERATION:
    if (value == PERFLIBS_SPARSE_OPERATION_NOTRANS ||
        value == PERFLIBS_SPARSE_OPERATION_TRANS ||
        value == PERFLIBS_SPARSE_OPERATION_CONJTRANS) {
      impl->userhint_spsm_op = value;
    } else {
      return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    }
    break;

  case PERFLIBS_SPARSE_HINT_SPSV_OPERATION:
    if (value == PERFLIBS_SPARSE_OPERATION_NOTRANS ||
        value == PERFLIBS_SPARSE_OPERATION_TRANS ||
        value == PERFLIBS_SPARSE_OPERATION_CONJTRANS) {
      impl->userhint_spsv_op = value;
    } else {
      return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    }
    break;

  case PERFLIBS_SPARSE_HINT_SPSV_INVOCATIONS:
    if (value == PERFLIBS_SPARSE_INVOCATIONS_SINGLE ||
        value == PERFLIBS_SPARSE_INVOCATIONS_FEW ||
        value == PERFLIBS_SPARSE_INVOCATIONS_MANY) {
      impl->userhint_spsv_invocations = value;
    } else {
      return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    }
    break;

  case PERFLIBS_SPARSE_HINT_SPSV_STRATEGY:
    if (value == PERFLIBS_SPARSE_SPSV_STRAT_UNSET ||
        value == PERFLIBS_SPARSE_SPSV_STRAT_SEPARATOR_SPARSE ||
        value == PERFLIBS_SPARSE_SPSV_STRAT_SEPARATOR_DENSE) {
      impl->userhint_spsv_strat = value;
    } else {
      return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    }
    break;

  default:
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }

  // Propagate hints for submatrices in the supernodal format
  if (impl->spmat_format == perflibs_format_supernodal) {
    for (auto &mat : impl->supernodal.mats_diag) {
      auto diag = reinterpret_cast<perflibs_spmat_impl_t<T> *>(mat->impl);
      set_hint(diag, hint, value);
    }

    auto sep = reinterpret_cast<perflibs_spmat_impl_t<T> *>(
        impl->supernodal.separator->impl);
    set_hint(sep, hint, value);

    // Translate the hint to the SPMV operation type if needed for the separator
    // blocks, which do SpMV, not SpSV
    hint = hint == PERFLIBS_SPARSE_HINT_SPSV_OPERATION
               ? PERFLIBS_SPARSE_HINT_SPMV_OPERATION
               : hint;
    for (auto &mat : impl->supernodal.mats_sep) {
      auto sep_blk = reinterpret_cast<perflibs_spmat_impl_t<T> *>(mat->impl);
      set_hint(sep_blk, hint, value);
    }
  }

  return PERFLIBS_STATUS_SUCCESS;
};
template perflibs_status_t set_hint<float>(perflibs_spmat_impl_t<float> *impl,
                                           perflibs_sparse_hint_type hint,
                                           perflibs_sparse_hint_value value);
template perflibs_status_t set_hint<double>(perflibs_spmat_impl_t<double> *impl,
                                            perflibs_sparse_hint_type hint,
                                            perflibs_sparse_hint_value value);
template perflibs_status_t
set_hint<std::complex<float>>(perflibs_spmat_impl_t<std::complex<float>> *impl,
                              perflibs_sparse_hint_type hint,
                              perflibs_sparse_hint_value value);
template perflibs_status_t set_hint<std::complex<double>>(
    perflibs_spmat_impl_t<std::complex<double>> *impl,
    perflibs_sparse_hint_type hint, perflibs_sparse_hint_value value);

template <typename T> void set_time_limit(perflibs_spmat_top_t *A, double tl) {
  auto impl = reinterpret_cast<perflibs_spmat_impl_t<T> *>(A->impl);
  impl->audition_tl = tl;
};
template void set_time_limit<float>(perflibs_spmat_top_t *A, double tl);
template void set_time_limit<double>(perflibs_spmat_top_t *A, double tl);
template void set_time_limit<std::complex<float>>(perflibs_spmat_top_t *A,
                                                  double tl);
template void set_time_limit<std::complex<double>>(perflibs_spmat_top_t *A,
                                                   double tl);

template <typename T> void set_C(perflibs_spmat_top_t *A, int C) {
  auto impl = reinterpret_cast<perflibs_spmat_impl_t<T> *>(A->impl);
  impl->use_C = C;
};
template void set_C<float>(perflibs_spmat_top_t *A, int C);
template void set_C<double>(perflibs_spmat_top_t *A, int C);
template void set_C<std::complex<float>>(perflibs_spmat_top_t *A, int C);
template void set_C<std::complex<double>>(perflibs_spmat_top_t *A, int C);

template <typename T> void set_sigma(perflibs_spmat_top_t *A, int sigma) {
  auto impl = reinterpret_cast<perflibs_spmat_impl_t<T> *>(A->impl);
  impl->use_sigma = sigma;
};
template void set_sigma<float>(perflibs_spmat_top_t *A, int sigma);
template void set_sigma<double>(perflibs_spmat_top_t *A, int sigma);
template void set_sigma<std::complex<float>>(perflibs_spmat_top_t *A,
                                             int sigma);
template void set_sigma<std::complex<double>>(perflibs_spmat_top_t *A,
                                              int sigma);

} // namespace perflibs::sparse
