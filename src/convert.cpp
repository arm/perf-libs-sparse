/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "convert.hpp"
#include "util.hpp"

namespace perflibs::sparse {

template <typename T>
perflibs_csr<T> csc2csr(enum sparse_hint_value_internal trans,
                        perflibs_csc<T> &csc) {
  return csc2csr(trans, csc.m, csc.n, csc.vals_ptr, csc.row_indx_ptr,
                 csc.col_ptr_ptr);
}

template <typename T>
perflibs_csc<T> csr2csc(enum sparse_hint_value_internal trans,
                        perflibs_csr<T> &csr) {
  return csr2csc(trans, csr.m, csr.n, csr.vals_ptr, csr.col_indx_ptr,
                 csr.row_ptr_ptr);
}

template <typename T> perflibs_csr<T> coo2csr(perflibs_coo<T> &coo) {
  return coo2csr(coo.m, coo.n, coo.nnz, coo.vals_ptr, coo.col_indx_ptr,
                 coo.row_indx_ptr, coo.index_base);
}

// The dense type doesn't have index_base (since it just stores values) so
// index_base here is really signifying whether the matrix came from Fortran
// (index_base=1) or C (index_base=0)
template <typename T>
perflibs_csr<T> dense2csr(perflibs_dense<T> &dense, perflibs_int_t index_base) {
  return dense2csr(dense.layout, dense.m, dense.n, dense.lda, dense.vals_ptr,
                   index_base);
}

template <typename T> perflibs_coo<T> csr2coo(perflibs_csr<T> &csr) {
  return csr2coo(csr.m, csr.n, csr.vals_ptr, csr.col_indx_ptr, csr.row_ptr_ptr);
}

template <typename T> perflibs_dense<T> csr2dense(perflibs_csr<T> &csr) {
  auto A =
      csr2dense(csr.m, csr.n, csr.vals_ptr, csr.col_indx_ptr, csr.row_ptr_ptr);
  return {PERFLIBS_COL_MAJOR,    csr.m, csr.n, csr.m, A.data(),
          (int64_t)csr.m * csr.n};
}

template <typename T> perflibs_csr<T> scs2csr(perflibs_scs<T> &scs) {
  auto csr = scs2csr(scs.m, scs.n, scs.C, scs.vals, scs.col_indx_offsets,
                     scs.col_indx_min, scs.col_indx_bytes, scs.cl, scs.cs,
                     scs.row_permd2in);

  if (scs.optimized_op == PERFLIBS_OPERATION_NOTRANS) {
    return csr;
  }
  if (scs.optimized_op == PERFLIBS_OPERATION_CONJNOTRANS) {
    for (auto &val : csr.vals) {
      val = perflibs::sparse::conj(val);
    }
    return csr;
  }

  // TRANS and CONJTRANS SCS store op(A). Convert that CSR to CSC and
  // reinterpret its arrays as CSR to transpose it back to the logical A.
  const auto undo_op = scs.optimized_op == PERFLIBS_OPERATION_CONJTRANS
                           ? PERFLIBS_OPERATION_CONJTRANS
                           : PERFLIBS_OPERATION_NOTRANS;
  auto csc = csr2csc(undo_op, csr);
  const auto index_base = csc.col_ptr_ptr[0];
  const auto nnz = csc.col_ptr_ptr[csc.n] - index_base;
  return perflibs_csr<T>(csc.n, csc.m, nnz, csc.vals_ptr, csc.col_ptr_ptr,
                         csc.row_indx_ptr);
}

template <typename T> perflibs_csr<T> bsr2csr(perflibs_bsr<T> &bsr) {
  return bsr2csr(bsr.block_layout, bsr.m, bsr.n, bsr.block_size, bsr.nnzb,
                 bsr.nrowsb, bsr.vals_ptr, bsr.row_ptr_ptr, bsr.col_indx_ptr);
}

template <typename T> perflibs_bsr<T> csr2bsr(perflibs_csr<T> &csr) {
  return csr2bsr(csr.m, csr.n, csr.vals_ptr, csr.row_ptr_ptr, csr.col_indx_ptr);
}

template <typename T>
perflibs_csc<T> supernodal2csc(perflibs_supernodal<T> &sn) {
  return supernodal2csc(sn.m, sn.n, sn.nnz, sn.nsuper, sn.super_row_ptr_ptr,
                        sn.super_col_indx_ptr, sn.row_indx_ptr, sn.col_ptr,
                        sn.vals_ptr);
}

// A function for converting a matrix from one format to another
template <typename T>
perflibs_status_t convert(spmat_format_t dest_format,
                          perflibs_spmat_impl_t<T> &mat_impl) {

  auto source_format = mat_impl.spmat_format;

  if (source_format != dest_format) {

    // Switch to CSR
    switch (source_format) {
    case (perflibs_format_csr):
      break;
    case (perflibs_format_csc):
      mat_impl.csr = csc2csr(PERFLIBS_OPERATION_NOTRANS, mat_impl.csc);
      mat_impl.csc = {};
      mat_impl.spmat_format = perflibs_format_csr;
      break;
    case (perflibs_format_coo):
      mat_impl.csr = coo2csr(mat_impl.coo);
      mat_impl.coo = {};
      mat_impl.spmat_format = perflibs_format_csr;
      break;
    case (perflibs_format_dense):
      // We remove explicit zeros from the matrix during conversion, so reset
      // nnz for the matrix
      mat_impl.csr = dense2csr(mat_impl.dense, mat_impl.index_base);
      mat_impl.nnz = mat_impl.csr.row_ptr[mat_impl.m] - mat_impl.csr.row_ptr[0];
      mat_impl.dense = {};
      mat_impl.spmat_format = perflibs_format_csr;
      break;
    case (perflibs_format_null):
      mat_impl.csr = null2csr<T>(mat_impl.m, mat_impl.n);
      mat_impl.spmat_format = perflibs_format_csr;
      break;
    case (perflibs_format_identity):
      mat_impl.csr = identity2csr<T>(mat_impl.n);
      mat_impl.spmat_format = perflibs_format_csr;
      break;
    case (perflibs_format_scs):
      mat_impl.csr = scs2csr<T>(mat_impl.scs);
      mat_impl.scs = {};
      mat_impl.spmat_format = perflibs_format_csr;
      break;
    case (perflibs_format_bsr):
      mat_impl.csr = bsr2csr<T>(mat_impl.bsr);
      // We remove explicit zeros from the blocks during conversion, so reset
      // nnz for the matrix
      mat_impl.nnz = mat_impl.csr.row_ptr[mat_impl.m] - mat_impl.csr.row_ptr[0];
      mat_impl.bsr = {};
      mat_impl.spmat_format = perflibs_format_csr;
      break;
    case (perflibs_format_supernodal):
      mat_impl.csc = supernodal2csc<T>(mat_impl.supernodal);
      mat_impl.supernodal = {};
      mat_impl.csr = csc2csr(PERFLIBS_OPERATION_NOTRANS, mat_impl.csc);
      mat_impl.csc = {};
      mat_impl.spmat_format = perflibs_format_csr;
      break;
    }
    assert(mat_impl.spmat_format == perflibs_format_csr);

    // Switch source CSR into dest_format
    switch (dest_format) {
    case (perflibs_format_csr):
      break;
    case (perflibs_format_csc):
      mat_impl.csc = csr2csc(PERFLIBS_OPERATION_NOTRANS, mat_impl.csr);
      mat_impl.csr = {};
      mat_impl.spmat_format = perflibs_format_csc;
      break;
    case (perflibs_format_coo):
      mat_impl.coo = csr2coo(mat_impl.csr);
      mat_impl.csr = {};
      mat_impl.spmat_format = perflibs_format_coo;
      break;
    case (perflibs_format_dense):
      mat_impl.dense = csr2dense(mat_impl.csr);
      mat_impl.csr = {};
      mat_impl.spmat_format = perflibs_format_dense;
      break;
    case (perflibs_format_bsr):
      mat_impl.bsr = csr2bsr(mat_impl.csr);
      mat_impl.csr = {};
      mat_impl.spmat_format = perflibs_format_bsr;
      break;
    case (perflibs_format_null):
    case (perflibs_format_identity):
    case (perflibs_format_supernodal):
    case (perflibs_format_scs): // We don't want to use this function to convert
                                // to SCS - this is to be done via auditioning
      return PERFLIBS_STATUS_EXECUTION_FAILURE;
    }
  }
  return PERFLIBS_STATUS_SUCCESS;
}
template perflibs_status_t convert(spmat_format_t dest_format,
                                   perflibs_spmat_impl_t<float> &mat_impl);
template perflibs_status_t convert(spmat_format_t dest_format,
                                   perflibs_spmat_impl_t<double> &mat_impl);
template perflibs_status_t
convert(spmat_format_t dest_format,
        perflibs_spmat_impl_t<std::complex<float>> &mat_impl);
template perflibs_status_t
convert(spmat_format_t dest_format,
        perflibs_spmat_impl_t<std::complex<double>> &mat_impl);

template <typename T>
perflibs_status_t convert(spmat_format_t dest_format,
                          perflibs_spmat_impl_t<T> *mat_impl) {
  return convert(dest_format, *mat_impl);
}
template perflibs_status_t convert(spmat_format_t dest_format,
                                   perflibs_spmat_impl_t<float> *mat_impl);
template perflibs_status_t convert(spmat_format_t dest_format,
                                   perflibs_spmat_impl_t<double> *mat_impl);
template perflibs_status_t
convert(spmat_format_t dest_format,
        perflibs_spmat_impl_t<std::complex<float>> *mat_impl);
template perflibs_status_t
convert(spmat_format_t dest_format,
        perflibs_spmat_impl_t<std::complex<double>> *mat_impl);

} // end namespace perflibs::sparse
