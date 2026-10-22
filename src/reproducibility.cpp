/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include <atomic>

#include <perflibs_sparse.h>
#include <perflibs_sparse_reproducibility.h>

namespace {
std::atomic<perflibs_sparse_reproducibility_value> reproducibility_mode{
    PERFLIBS_SPARSE_REPRODUCIBILITY_DISABLED};
}

extern "C" perflibs_status_t perflibs_sparse_set_reproducibility(
    enum perflibs_sparse_reproducibility_value mode) {
  switch (mode) {
  case PERFLIBS_SPARSE_REPRODUCIBILITY_DISABLED:
  case PERFLIBS_SPARSE_REPRODUCIBILITY_ENABLED:
    reproducibility_mode.store(mode, std::memory_order_relaxed);
    return PERFLIBS_STATUS_SUCCESS;
  }

  return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
}

extern "C" enum perflibs_sparse_reproducibility_value
perflibs_sparse_get_reproducibility(void) {
  return reproducibility_mode.load(std::memory_order_relaxed);
}
