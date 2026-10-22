/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#ifndef PERFLIBS_SPARSE_REPRODUCIBILITY_H
#define PERFLIBS_SPARSE_REPRODUCIBILITY_H

#include "perflibs_sparse.h"

enum perflibs_sparse_reproducibility_value {
  PERFLIBS_SPARSE_REPRODUCIBILITY_DISABLED = 2001,
  PERFLIBS_SPARSE_REPRODUCIBILITY_ENABLED = 2002
};

#ifdef __cplusplus
extern "C" {
#endif

/* Sets the process-global sparse reproducibility mode, which is initially
 * disabled. Passing PERFLIBS_SPARSE_REPRODUCIBILITY_DISABLED selects the
 * default mode, while PERFLIBS_SPARSE_REPRODUCIBILITY_ENABLED selects
 * reproducibility mode.
 *
 * Returns PERFLIBS_STATUS_SUCCESS when the mode is set. If mode is not a
 * valid value of enum perflibs_sparse_reproducibility_value, returns
 * PERFLIBS_STATUS_INPUT_PARAMETER_ERROR and leaves the mode unchanged.
 *
 * Reproducibility mode must be configured outside active OpenMP parallel
 * regions. The mode must not be changed concurrently with library calls or
 * another mode-setting operation. */
perflibs_status_t perflibs_sparse_set_reproducibility(
    enum perflibs_sparse_reproducibility_value mode);

/* Returns the current process-global sparse reproducibility mode. */
enum perflibs_sparse_reproducibility_value
perflibs_sparse_get_reproducibility(void);

#ifdef __cplusplus
}
#endif

#endif
