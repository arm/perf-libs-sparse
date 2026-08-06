/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include "util.hpp"

struct perflibs_spmat_top_t;
template <typename T> struct perflibs_spmat_impl_t;

namespace perflibs::sparse {

template <typename T>
perflibs_status_t spsv_optimize(perflibs_spmat_impl_t<T> *impl);

template <typename T>
perflibs_status_t
spsm_optimize(perflibs_sparse_hint_value trans, perflibs_spmat_top_t *A,
              perflibs_spmat_top_t *X, perflibs_spmat_top_t *Y);

template <typename T>
perflibs_status_t spsm_exec(perflibs_sparse_hint_value trans,
                            perflibs_spmat_top_t *A, perflibs_spmat_top_t *X,
                            T alpha, perflibs_spmat_top_t *Y);

template <typename T>
perflibs_status_t spsv_exec(perflibs_sparse_hint_value trans,
                            perflibs_spmat_top_t *A, T *x, T alpha, const T *y);

template <typename T>
perflibs_status_t spsv_exec_impl(perflibs_sparse_hint_value trans,
                                 perflibs_spmat_impl_t<T> *impl, T *x, T alpha,
                                 const T *y);

} // namespace perflibs::sparse
