/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "matmul_gustavson.hpp"
#include "sell_c_sigma.hpp"
#include "spec_sve.hpp"
#include <complex>

namespace perflibs::sparse {

// This is a dependency external to the sparse library
bool get_sve();

// SCS kernels for SpMV
template <typename T>
const std::vector<std::pair<int, scs_spmv_kernels<T>>> &
scs_get_valid_C(T vals) {
// Avoid dependencies on SVE code altogether if SVE is not enabled
#if defined(PERFLIBS_HAVE_SVE_KERNELS)
  if (get_sve()) {
    return scs_get_valid_C_sve<T>(vals);
  } else
#endif
    return scs_get_valid_C_default<T>(vals);
}
template const std::vector<std::pair<int, scs_spmv_kernels<float>>> &
scs_get_valid_C<float>(float);

template const std::vector<std::pair<int, scs_spmv_kernels<double>>> &
scs_get_valid_C<double>(double);

template const std::vector<
    std::pair<int, scs_spmv_kernels<std::complex<float>>>> &
    scs_get_valid_C<std::complex<float>>(std::complex<float>);

template const std::vector<
    std::pair<int, scs_spmv_kernels<std::complex<double>>>> &
    scs_get_valid_C<std::complex<double>>(std::complex<double>);

// Gustavson kernels for SpMM
template <typename T>
gs_kernel_t<T> get_gustavson_kernel(perflibs_sparse_hint_value transA,
                                    perflibs_sparse_hint_value transB,
                                    perflibs_int_t m, perflibs_int_t n,
                                    T alpha) {
#if defined(PERFLIBS_HAVE_SVE_KERNELS)
  if (get_sve()) {
    return get_gustavson_kernel_sve<T>(transA, transB, m, n, alpha);
  } else
#endif
    return get_gustavson_kernel_default<T>(transA, transB, m, n, alpha);
}
template gs_kernel_t<float>
get_gustavson_kernel<float>(perflibs_sparse_hint_value,
                            perflibs_sparse_hint_value, perflibs_int_t,
                            perflibs_int_t, float);
template gs_kernel_t<double>
get_gustavson_kernel<double>(perflibs_sparse_hint_value,
                             perflibs_sparse_hint_value, perflibs_int_t,
                             perflibs_int_t, double);
template gs_kernel_t<std::complex<float>>
    get_gustavson_kernel<std::complex<float>>(perflibs_sparse_hint_value,
                                              perflibs_sparse_hint_value,
                                              perflibs_int_t, perflibs_int_t,
                                              std::complex<float>);
template gs_kernel_t<std::complex<double>>
    get_gustavson_kernel<std::complex<double>>(perflibs_sparse_hint_value,
                                               perflibs_sparse_hint_value,
                                               perflibs_int_t, perflibs_int_t,
                                               std::complex<double>);

template <typename T>
gs_na_kernel_t<T>
get_gustavson_kernel_vals_only(perflibs_sparse_hint_value transA,
                               perflibs_sparse_hint_value transB,
                               perflibs_int_t m, perflibs_int_t n, T alpha,
                               sparse_matrix_statistics stats) {
#if defined(PERFLIBS_HAVE_SVE_KERNELS)
  if (get_sve()) {
    return get_gustavson_kernel_vals_only_sve<T>(transA, transB, m, n, alpha,
                                                 stats);
  } else
#endif
    return get_gustavson_kernel_vals_only_default<T>(transA, transB, m, n,
                                                     alpha, stats);
}
template gs_na_kernel_t<float> get_gustavson_kernel_vals_only<float>(
    perflibs_sparse_hint_value, perflibs_sparse_hint_value, perflibs_int_t,
    perflibs_int_t, float, sparse_matrix_statistics);
template gs_na_kernel_t<double> get_gustavson_kernel_vals_only<double>(
    perflibs_sparse_hint_value, perflibs_sparse_hint_value, perflibs_int_t,
    perflibs_int_t, double, sparse_matrix_statistics);
template gs_na_kernel_t<std::complex<float>>
    get_gustavson_kernel_vals_only<std::complex<float>>(
        perflibs_sparse_hint_value, perflibs_sparse_hint_value, perflibs_int_t,
        perflibs_int_t, std::complex<float>, sparse_matrix_statistics);
template gs_na_kernel_t<std::complex<double>>
    get_gustavson_kernel_vals_only<std::complex<double>>(
        perflibs_sparse_hint_value, perflibs_sparse_hint_value, perflibs_int_t,
        perflibs_int_t, std::complex<double>, sparse_matrix_statistics);

template <typename T> bool get_if_mgmd_is_required() {
#if defined(PERFLIBS_HAVE_SVE_KERNELS)
  if (get_sve()) {
    return get_if_mgmd_is_required_sve<T>();
  } else
#endif
    return get_if_mgmd_is_required_default<T>();
}
template bool get_if_mgmd_is_required<float>();
template bool get_if_mgmd_is_required<double>();
template bool get_if_mgmd_is_required<std::complex<float>>();
template bool get_if_mgmd_is_required<std::complex<double>>();

} // end namespace perflibs::sparse
