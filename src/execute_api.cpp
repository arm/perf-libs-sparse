/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "add.hpp"
#include "c_api_complex_abi.hpp"
#include "matmul.hpp"
#include "matrix_state.hpp"
#include "matvec.hpp"
#include "norm.hpp"
#include "solve.hpp"
#include "types.hpp"

perflibs_status_t perflibs_spmv_optimize(perflibs_spmat_top_t *A) {
  if (A->datatype == PERFLIBS_DATATYPE_SINGLE) {
    return perflibs::sparse::spmv_optimize(
        reinterpret_cast<perflibs_spmat_impl_t<float> *>(A->impl));
  } else if (A->datatype == PERFLIBS_DATATYPE_DOUBLE) {
    return perflibs::sparse::spmv_optimize(
        reinterpret_cast<perflibs_spmat_impl_t<double> *>(A->impl));
  } else if (A->datatype == PERFLIBS_DATATYPE_CPLXSINGLE) {
    return perflibs::sparse::spmv_optimize(
        reinterpret_cast<perflibs_spmat_impl_t<std::complex<float>> *>(
            A->impl));
  } else if (A->datatype == PERFLIBS_DATATYPE_CPLXDOUBLE) {
    return perflibs::sparse::spmv_optimize(
        reinterpret_cast<perflibs_spmat_impl_t<std::complex<double>> *>(
            A->impl));
  }
  return PERFLIBS_STATUS_EXECUTION_FAILURE;
}

perflibs_status_t perflibs_spmm_optimize(enum perflibs_sparse_hint_value transA,
                                         enum perflibs_sparse_hint_value transB,
                                         perflibs_sparse_hint_value alpha,
                                         perflibs_spmat_t A, perflibs_spmat_t B,
                                         perflibs_sparse_hint_value beta,
                                         perflibs_spmat_t C) {
  if (A->datatype == PERFLIBS_DATATYPE_SINGLE) {
    return perflibs::sparse::spmm_optimize<float>(transA, transB, alpha, A, B,
                                                  beta, C);
  } else if (A->datatype == PERFLIBS_DATATYPE_DOUBLE) {
    return perflibs::sparse::spmm_optimize<double>(transA, transB, alpha, A, B,
                                                   beta, C);
  } else if (A->datatype == PERFLIBS_DATATYPE_CPLXSINGLE) {
    return perflibs::sparse::spmm_optimize<std::complex<float>>(
        transA, transB, alpha, A, B, beta, C);
  } else if (A->datatype == PERFLIBS_DATATYPE_CPLXDOUBLE) {
    return perflibs::sparse::spmm_optimize<std::complex<double>>(
        transA, transB, alpha, A, B, beta, C);
  }
  return PERFLIBS_STATUS_EXECUTION_FAILURE;
}

perflibs_status_t
perflibs_spadd_optimize(enum perflibs_sparse_hint_value transA,
                        enum perflibs_sparse_hint_value transB,
                        perflibs_sparse_hint_value alpha, perflibs_spmat_t A,
                        perflibs_sparse_hint_value beta, perflibs_spmat_t B,
                        perflibs_spmat_t C) {
  if (A->datatype == PERFLIBS_DATATYPE_SINGLE) {
    return perflibs::sparse::spadd_optimize<float>(transA, transB, alpha, A,
                                                   beta, B, C);
  } else if (A->datatype == PERFLIBS_DATATYPE_DOUBLE) {
    return perflibs::sparse::spadd_optimize<double>(transA, transB, alpha, A,
                                                    beta, B, C);
  } else if (A->datatype == PERFLIBS_DATATYPE_CPLXSINGLE) {
    return perflibs::sparse::spadd_optimize<std::complex<float>>(
        transA, transB, alpha, A, beta, B, C);
  } else if (A->datatype == PERFLIBS_DATATYPE_CPLXDOUBLE) {
    return perflibs::sparse::spadd_optimize<std::complex<double>>(
        transA, transB, alpha, A, beta, B, C);
  }
  return PERFLIBS_STATUS_EXECUTION_FAILURE;
}

perflibs_status_t perflibs_spsv_optimize(perflibs_spmat_top_t *A) {
  if (A->datatype == PERFLIBS_DATATYPE_SINGLE) {
    return perflibs::sparse::spsv_optimize(
        reinterpret_cast<perflibs_spmat_impl_t<float> *>(A->impl));
  } else if (A->datatype == PERFLIBS_DATATYPE_DOUBLE) {
    return perflibs::sparse::spsv_optimize(
        reinterpret_cast<perflibs_spmat_impl_t<double> *>(A->impl));
  } else if (A->datatype == PERFLIBS_DATATYPE_CPLXSINGLE) {
    return perflibs::sparse::spsv_optimize(
        reinterpret_cast<perflibs_spmat_impl_t<std::complex<float>> *>(
            A->impl));
  } else if (A->datatype == PERFLIBS_DATATYPE_CPLXDOUBLE) {
    return perflibs::sparse::spsv_optimize(
        reinterpret_cast<perflibs_spmat_impl_t<std::complex<double>> *>(
            A->impl));
  }
  return PERFLIBS_STATUS_EXECUTION_FAILURE;
}

perflibs_status_t perflibs_spsm_optimize(enum perflibs_sparse_hint_value transA,
                                         perflibs_spmat_t A, perflibs_spmat_t X,
                                         perflibs_sparse_hint_value alpha,
                                         perflibs_spmat_t Y) {
  (void)alpha;
  auto info =
      perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_SPSM_OPERATION, transA);
  if (info != PERFLIBS_STATUS_SUCCESS) {
    return info;
  }
  info = perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_SPSV_OPERATION, transA);
  if (info != PERFLIBS_STATUS_SUCCESS) {
    return info;
  }
  if (!perflibs::sparse::have_compatible_matrix_datatypes(A, X, Y)) {
    return PERFLIBS_STATUS_EXECUTION_FAILURE;
  } else if (A->datatype == PERFLIBS_DATATYPE_SINGLE) {
    return perflibs::sparse::spsm_optimize<float>(transA, A, X, Y);
  } else if (A->datatype == PERFLIBS_DATATYPE_DOUBLE) {
    return perflibs::sparse::spsm_optimize<double>(transA, A, X, Y);
  } else if (A->datatype == PERFLIBS_DATATYPE_CPLXSINGLE) {
    return perflibs::sparse::spsm_optimize<std::complex<float>>(transA, A, X,
                                                                Y);
  } else if (A->datatype == PERFLIBS_DATATYPE_CPLXDOUBLE) {
    return perflibs::sparse::spsm_optimize<std::complex<double>>(transA, A, X,
                                                                 Y);
  }
  return PERFLIBS_STATUS_EXECUTION_FAILURE;
}

perflibs_status_t
perflibs_spelmm_optimize(enum perflibs_sparse_hint_value transA,
                         enum perflibs_sparse_hint_value transB,
                         perflibs_sparse_hint_value alpha, perflibs_spmat_t A,
                         perflibs_spmat_t B, perflibs_sparse_hint_value beta,
                         perflibs_spmat_t C) {
  if (A->datatype == PERFLIBS_DATATYPE_SINGLE) {
    return perflibs::sparse::spelmm_optimize<float>(transA, transB, alpha, A, B,
                                                    beta, C);
  } else if (A->datatype == PERFLIBS_DATATYPE_DOUBLE) {
    return perflibs::sparse::spelmm_optimize<double>(transA, transB, alpha, A,
                                                     B, beta, C);
  } else if (A->datatype == PERFLIBS_DATATYPE_CPLXSINGLE) {
    return perflibs::sparse::spelmm_optimize<std::complex<float>>(
        transA, transB, alpha, A, B, beta, C);
  } else if (A->datatype == PERFLIBS_DATATYPE_CPLXDOUBLE) {
    return perflibs::sparse::spelmm_optimize<std::complex<double>>(
        transA, transB, alpha, A, B, beta, C);
  }
  return PERFLIBS_STATUS_EXECUTION_FAILURE;
}

perflibs_status_t
perflibs_sddmm_optimize(enum perflibs_sparse_hint_value transA,
                        enum perflibs_sparse_hint_value transB,
                        perflibs_sparse_hint_value alpha, perflibs_spmat_t A,
                        perflibs_spmat_t B, perflibs_sparse_hint_value beta,
                        perflibs_spmat_t C) {
  if (A->datatype == PERFLIBS_DATATYPE_SINGLE) {
    return perflibs::sparse::sddmm_optimize<float>(transA, transB, alpha, A, B,
                                                   beta, C);
  } else if (A->datatype == PERFLIBS_DATATYPE_DOUBLE) {
    return perflibs::sparse::sddmm_optimize<double>(transA, transB, alpha, A, B,
                                                    beta, C);
  } else if (A->datatype == PERFLIBS_DATATYPE_CPLXSINGLE) {
    return perflibs::sparse::sddmm_optimize<std::complex<float>>(
        transA, transB, alpha, A, B, beta, C);
  } else if (A->datatype == PERFLIBS_DATATYPE_CPLXDOUBLE) {
    return perflibs::sparse::sddmm_optimize<std::complex<double>>(
        transA, transB, alpha, A, B, beta, C);
  }
  return PERFLIBS_STATUS_EXECUTION_FAILURE;
}

perflibs_status_t perflibs_spmv_exec_s(perflibs_sparse_hint_value trans,
                                       float alpha, perflibs_spmat_top_t *A,
                                       const float *x, float beta, float *y) {
  return perflibs::sparse::spmv_exec(trans, alpha, A, x, beta, y);
}

perflibs_status_t perflibs_spmv_exec_d(perflibs_sparse_hint_value trans,
                                       double alpha, perflibs_spmat_top_t *A,
                                       const double *x, double beta,
                                       double *y) {
  return perflibs::sparse::spmv_exec(trans, alpha, A, x, beta, y);
}

perflibs_status_t perflibs_spmv_exec_c(perflibs_sparse_hint_value trans,
                                       perflibs_singlecomplex_t alpha,
                                       perflibs_spmat_top_t *A,
                                       const perflibs_singlecomplex_t *x,
                                       perflibs_singlecomplex_t beta,
                                       perflibs_singlecomplex_t *y) {
  return perflibs::sparse::spmv_exec(
      trans, perflibs::sparse::c_api::to_cpp_scalar(alpha), A,
      reinterpret_cast<const std::complex<float> *>(x),
      perflibs::sparse::c_api::to_cpp_scalar(beta),
      reinterpret_cast<std::complex<float> *>(y));
}

perflibs_status_t perflibs_spmv_exec_z(perflibs_sparse_hint_value trans,
                                       perflibs_doublecomplex_t alpha,
                                       perflibs_spmat_top_t *A,
                                       const perflibs_doublecomplex_t *x,
                                       perflibs_doublecomplex_t beta,
                                       perflibs_doublecomplex_t *y) {
  return perflibs::sparse::spmv_exec(
      trans, perflibs::sparse::c_api::to_cpp_scalar(alpha), A,
      reinterpret_cast<const std::complex<double> *>(x),
      perflibs::sparse::c_api::to_cpp_scalar(beta),
      reinterpret_cast<std::complex<double> *>(y));
}

perflibs_status_t perflibs_spmm_exec_s(enum perflibs_sparse_hint_value transA,
                                       enum perflibs_sparse_hint_value transB,
                                       float alpha, perflibs_spmat_t A,
                                       perflibs_spmat_t B, float beta,
                                       perflibs_spmat_t C) {
  return perflibs::sparse::spmm_exec(transA, transB, alpha, A, B, beta, C);
}

perflibs_status_t perflibs_spmm_exec_d(enum perflibs_sparse_hint_value transA,
                                       enum perflibs_sparse_hint_value transB,
                                       double alpha, perflibs_spmat_t A,
                                       perflibs_spmat_t B, double beta,
                                       perflibs_spmat_t C) {
  return perflibs::sparse::spmm_exec(transA, transB, alpha, A, B, beta, C);
}

perflibs_status_t perflibs_spmm_exec_c(enum perflibs_sparse_hint_value transA,
                                       enum perflibs_sparse_hint_value transB,
                                       perflibs_singlecomplex_t alpha,
                                       perflibs_spmat_t A, perflibs_spmat_t B,
                                       perflibs_singlecomplex_t beta,
                                       perflibs_spmat_t C) {
  return perflibs::sparse::spmm_exec(
      transA, transB, perflibs::sparse::c_api::to_cpp_scalar(alpha), A, B,
      perflibs::sparse::c_api::to_cpp_scalar(beta), C);
}

perflibs_status_t perflibs_spmm_exec_z(enum perflibs_sparse_hint_value transA,
                                       enum perflibs_sparse_hint_value transB,
                                       perflibs_doublecomplex_t alpha,
                                       perflibs_spmat_t A, perflibs_spmat_t B,
                                       perflibs_doublecomplex_t beta,
                                       perflibs_spmat_t C) {
  return perflibs::sparse::spmm_exec(
      transA, transB, perflibs::sparse::c_api::to_cpp_scalar(alpha), A, B,
      perflibs::sparse::c_api::to_cpp_scalar(beta), C);
}

perflibs_status_t perflibs_spadd_exec_s(enum perflibs_sparse_hint_value transA,
                                        enum perflibs_sparse_hint_value transB,
                                        float alpha, perflibs_spmat_t A,
                                        float beta, perflibs_spmat_t B,
                                        perflibs_spmat_t C) {
  return perflibs::sparse::spadd_exec(transA, transB, alpha, A, beta, B, C);
}

perflibs_status_t perflibs_spadd_exec_d(enum perflibs_sparse_hint_value transA,
                                        enum perflibs_sparse_hint_value transB,
                                        double alpha, perflibs_spmat_t A,
                                        double beta, perflibs_spmat_t B,
                                        perflibs_spmat_t C) {
  return perflibs::sparse::spadd_exec(transA, transB, alpha, A, beta, B, C);
}

perflibs_status_t perflibs_spadd_exec_c(enum perflibs_sparse_hint_value transA,
                                        enum perflibs_sparse_hint_value transB,
                                        perflibs_singlecomplex_t alpha,
                                        perflibs_spmat_t A,
                                        perflibs_singlecomplex_t beta,
                                        perflibs_spmat_t B,
                                        perflibs_spmat_t C) {
  return perflibs::sparse::spadd_exec(
      transA, transB, perflibs::sparse::c_api::to_cpp_scalar(alpha), A,
      perflibs::sparse::c_api::to_cpp_scalar(beta), B, C);
}

perflibs_status_t perflibs_spadd_exec_z(enum perflibs_sparse_hint_value transA,
                                        enum perflibs_sparse_hint_value transB,
                                        perflibs_doublecomplex_t alpha,
                                        perflibs_spmat_t A,
                                        perflibs_doublecomplex_t beta,
                                        perflibs_spmat_t B,
                                        perflibs_spmat_t C) {
  return perflibs::sparse::spadd_exec(
      transA, transB, perflibs::sparse::c_api::to_cpp_scalar(alpha), A,
      perflibs::sparse::c_api::to_cpp_scalar(beta), B, C);
}

perflibs_status_t perflibs_spsm_exec_s(enum perflibs_sparse_hint_value transA,
                                       perflibs_spmat_t A, perflibs_spmat_t X,
                                       float alpha, perflibs_spmat_t Y) {

  if (!(perflibs::sparse::have_compatible_matrix_datatypes(A, X, Y) &&
        A->datatype == PERFLIBS_DATATYPE_SINGLE)) {
    return PERFLIBS_STATUS_EXECUTION_FAILURE;
  }
  return perflibs::sparse::spsm_exec(transA, A, X, alpha, Y);
}

perflibs_status_t perflibs_spsm_exec_d(enum perflibs_sparse_hint_value transA,
                                       perflibs_spmat_t A, perflibs_spmat_t X,
                                       double alpha, perflibs_spmat_t Y) {

  if (!(perflibs::sparse::have_compatible_matrix_datatypes(A, X, Y) &&
        A->datatype == PERFLIBS_DATATYPE_DOUBLE)) {
    return PERFLIBS_STATUS_EXECUTION_FAILURE;
  }
  return perflibs::sparse::spsm_exec(transA, A, X, alpha, Y);
}

perflibs_status_t perflibs_spsm_exec_c(enum perflibs_sparse_hint_value transA,
                                       perflibs_spmat_t A, perflibs_spmat_t X,
                                       perflibs_singlecomplex_t alpha,
                                       perflibs_spmat_t Y) {

  if (!(perflibs::sparse::have_compatible_matrix_datatypes(A, X, Y) &&
        A->datatype == PERFLIBS_DATATYPE_CPLXSINGLE)) {
    return PERFLIBS_STATUS_EXECUTION_FAILURE;
  }
  return perflibs::sparse::spsm_exec(
      transA, A, X, perflibs::sparse::c_api::to_cpp_scalar(alpha), Y);
}

perflibs_status_t perflibs_spsm_exec_z(enum perflibs_sparse_hint_value transA,
                                       perflibs_spmat_t A, perflibs_spmat_t X,
                                       perflibs_doublecomplex_t alpha,
                                       perflibs_spmat_t Y) {

  if (!(perflibs::sparse::have_compatible_matrix_datatypes(A, X, Y) &&
        A->datatype == PERFLIBS_DATATYPE_CPLXDOUBLE)) {
    return PERFLIBS_STATUS_EXECUTION_FAILURE;
  }
  return perflibs::sparse::spsm_exec(
      transA, A, X, perflibs::sparse::c_api::to_cpp_scalar(alpha), Y);
}

perflibs_status_t perflibs_spsv_exec_s(enum perflibs_sparse_hint_value transA,
                                       perflibs_spmat_t A, float *x,
                                       float alpha, const float *y) {
  return perflibs::sparse::spsv_exec(transA, A, x, alpha, y);
}

perflibs_status_t perflibs_spsv_exec_d(enum perflibs_sparse_hint_value transA,
                                       perflibs_spmat_t A, double *x,
                                       double alpha, const double *y) {
  return perflibs::sparse::spsv_exec(transA, A, x, alpha, y);
}

perflibs_status_t perflibs_spsv_exec_c(enum perflibs_sparse_hint_value transA,
                                       perflibs_spmat_t A,
                                       perflibs_singlecomplex_t *x,
                                       perflibs_singlecomplex_t alpha,
                                       const perflibs_singlecomplex_t *y) {
  return perflibs::sparse::spsv_exec(
      transA, A, reinterpret_cast<std::complex<float> *>(x),
      perflibs::sparse::c_api::to_cpp_scalar(alpha),
      reinterpret_cast<const std::complex<float> *>(y));
}

perflibs_status_t perflibs_spsv_exec_z(enum perflibs_sparse_hint_value transA,
                                       perflibs_spmat_t A,
                                       perflibs_doublecomplex_t *x,
                                       perflibs_doublecomplex_t alpha,
                                       const perflibs_doublecomplex_t *y) {
  return perflibs::sparse::spsv_exec(
      transA, A, reinterpret_cast<std::complex<double> *>(x),
      perflibs::sparse::c_api::to_cpp_scalar(alpha),
      reinterpret_cast<const std::complex<double> *>(y));
}

perflibs_status_t perflibs_spnorm_exec_s(perflibs_spmat_t A,
                                         enum perflibs_sparse_norm nrm,
                                         float *result) {
  return perflibs::sparse::spnorm_exec<float>(A, nrm, result);
}

perflibs_status_t perflibs_spnorm_exec_d(perflibs_spmat_t A,
                                         enum perflibs_sparse_norm nrm,
                                         double *result) {
  return perflibs::sparse::spnorm_exec<double>(A, nrm, result);
}

perflibs_status_t perflibs_spnorm_exec_c(perflibs_spmat_t A,
                                         enum perflibs_sparse_norm nrm,
                                         float *result) {
  return perflibs::sparse::spnorm_exec<std::complex<float>>(A, nrm, result);
}

perflibs_status_t perflibs_spnorm_exec_z(perflibs_spmat_t A,
                                         enum perflibs_sparse_norm nrm,
                                         double *result) {
  return perflibs::sparse::spnorm_exec<std::complex<double>>(A, nrm, result);
}

perflibs_status_t perflibs_spelmm_exec_s(enum perflibs_sparse_hint_value transA,
                                         enum perflibs_sparse_hint_value transB,
                                         float alpha, perflibs_spmat_t A,
                                         perflibs_spmat_t B, float beta,
                                         perflibs_spmat_t C) {
  return perflibs::sparse::spelmm_exec(transA, transB, alpha, A, B, beta, C);
}

perflibs_status_t perflibs_spelmm_exec_d(enum perflibs_sparse_hint_value transA,
                                         enum perflibs_sparse_hint_value transB,
                                         double alpha, perflibs_spmat_t A,
                                         perflibs_spmat_t B, double beta,
                                         perflibs_spmat_t C) {
  return perflibs::sparse::spelmm_exec(transA, transB, alpha, A, B, beta, C);
}

perflibs_status_t perflibs_spelmm_exec_c(enum perflibs_sparse_hint_value transA,
                                         enum perflibs_sparse_hint_value transB,
                                         perflibs_singlecomplex_t alpha,
                                         perflibs_spmat_t A, perflibs_spmat_t B,
                                         perflibs_singlecomplex_t beta,
                                         perflibs_spmat_t C) {
  return perflibs::sparse::spelmm_exec(
      transA, transB, perflibs::sparse::c_api::to_cpp_scalar(alpha), A, B,
      perflibs::sparse::c_api::to_cpp_scalar(beta), C);
}

perflibs_status_t perflibs_spelmm_exec_z(enum perflibs_sparse_hint_value transA,
                                         enum perflibs_sparse_hint_value transB,
                                         perflibs_doublecomplex_t alpha,
                                         perflibs_spmat_t A, perflibs_spmat_t B,
                                         perflibs_doublecomplex_t beta,
                                         perflibs_spmat_t C) {
  return perflibs::sparse::spelmm_exec(
      transA, transB, perflibs::sparse::c_api::to_cpp_scalar(alpha), A, B,
      perflibs::sparse::c_api::to_cpp_scalar(beta), C);
}

perflibs_status_t perflibs_sddmm_exec_s(enum perflibs_sparse_hint_value transA,
                                        enum perflibs_sparse_hint_value transB,
                                        float alpha, perflibs_spmat_t A,
                                        perflibs_spmat_t B, float beta,
                                        perflibs_spmat_t C) {
  return perflibs::sparse::sddmm_exec<float, false>(transA, transB, alpha, A, B,
                                                    beta, C);
}

perflibs_status_t perflibs_sddmm_exec_d(enum perflibs_sparse_hint_value transA,
                                        enum perflibs_sparse_hint_value transB,
                                        double alpha, perflibs_spmat_t A,
                                        perflibs_spmat_t B, double beta,
                                        perflibs_spmat_t C) {
  return perflibs::sparse::sddmm_exec<double, false>(transA, transB, alpha, A,
                                                     B, beta, C);
}

perflibs_status_t perflibs_sddmm_exec_c(enum perflibs_sparse_hint_value transA,
                                        enum perflibs_sparse_hint_value transB,
                                        perflibs_singlecomplex_t alpha,
                                        perflibs_spmat_t A, perflibs_spmat_t B,
                                        perflibs_singlecomplex_t beta,
                                        perflibs_spmat_t C) {
  return perflibs::sparse::sddmm_exec<std::complex<float>, false>(
      transA, transB, perflibs::sparse::c_api::to_cpp_scalar(alpha), A, B,
      perflibs::sparse::c_api::to_cpp_scalar(beta), C);
}

perflibs_status_t perflibs_sddmm_exec_z(enum perflibs_sparse_hint_value transA,
                                        enum perflibs_sparse_hint_value transB,
                                        perflibs_doublecomplex_t alpha,
                                        perflibs_spmat_t A, perflibs_spmat_t B,
                                        perflibs_doublecomplex_t beta,
                                        perflibs_spmat_t C) {
  return perflibs::sparse::sddmm_exec<std::complex<double>, false>(
      transA, transB, perflibs::sparse::c_api::to_cpp_scalar(alpha), A, B,
      perflibs::sparse::c_api::to_cpp_scalar(beta), C);
}
