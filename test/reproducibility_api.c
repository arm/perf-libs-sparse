/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "perflibs_sparse.h"
#include "test_utils.h"

#include <stdlib.h>

static int test_reproducibility_default(void) {
  CHECK_INT_EQ(perflibs_sparse_get_reproducibility(),
               PERFLIBS_SPARSE_REPRODUCIBILITY_DISABLED);
  return EXIT_SUCCESS;
}

static int test_reproducibility_modes(void) {
  CHECK_STATUS(perflibs_sparse_set_reproducibility(
      PERFLIBS_SPARSE_REPRODUCIBILITY_ENABLED));
  CHECK_INT_EQ(perflibs_sparse_get_reproducibility(),
               PERFLIBS_SPARSE_REPRODUCIBILITY_ENABLED);

  CHECK_STATUS(perflibs_sparse_set_reproducibility(
      PERFLIBS_SPARSE_REPRODUCIBILITY_DISABLED));
  CHECK_INT_EQ(perflibs_sparse_get_reproducibility(),
               PERFLIBS_SPARSE_REPRODUCIBILITY_DISABLED);

  return EXIT_SUCCESS;
}

static int test_reproducibility_invalid_mode(void) {
  CHECK_STATUS(perflibs_sparse_set_reproducibility(
      PERFLIBS_SPARSE_REPRODUCIBILITY_DISABLED));

  CHECK_STATUS_EQ(perflibs_sparse_set_reproducibility(
                      (enum perflibs_sparse_reproducibility_value)7),
                  PERFLIBS_STATUS_INPUT_PARAMETER_ERROR);
  CHECK_INT_EQ(perflibs_sparse_get_reproducibility(),
               PERFLIBS_SPARSE_REPRODUCIBILITY_DISABLED);

  return EXIT_SUCCESS;
}

int main(void) {
  if (test_reproducibility_default() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (test_reproducibility_modes() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (test_reproducibility_invalid_mode() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
