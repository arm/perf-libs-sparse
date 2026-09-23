! SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
! affiliates <open-source-office@arm.com></text>
!
! SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

module perflibs_sparse_params
    use perflibs_kinds
    implicit none

!   Fortran versions of C enums

    integer(kind=perflibs_i4), parameter :: perflibs_status_success = 0
    integer(kind=perflibs_i4), parameter :: perflibs_status_input_parameter_error = 1
    integer(kind=perflibs_i4), parameter :: perflibs_status_execution_failure = 2
    integer(kind=perflibs_i4), parameter :: perflibs_sparse_create_nocopy = 3

    integer(kind=perflibs_i4), parameter :: perflibs_sparse_hint_structure = 50
    integer(kind=perflibs_i4), parameter :: perflibs_sparse_hint_spmv_operation = 60
    integer(kind=perflibs_i4), parameter :: perflibs_sparse_hint_spmm_operation = 61
    integer(kind=perflibs_i4), parameter :: perflibs_sparse_hint_spadd_operation = 62
    integer(kind=perflibs_i4), parameter :: perflibs_sparse_hint_spsv_operation = 63
    integer(kind=perflibs_i4), parameter :: perflibs_sparse_hint_spmm_strategy = 64
    integer(kind=perflibs_i4), parameter :: perflibs_sparse_hint_spsm_operation = 65
    integer(kind=perflibs_i4), parameter :: perflibs_sparse_hint_spsv_strategy = 66
    integer(kind=perflibs_i4), parameter :: perflibs_sparse_hint_memory = 70
    integer(kind=perflibs_i4), parameter :: perflibs_sparse_hint_spmv_invocations = 80
    integer(kind=perflibs_i4), parameter :: perflibs_sparse_hint_spmm_invocations = 81
    integer(kind=perflibs_i4), parameter :: perflibs_sparse_hint_spadd_invocations = 82
    integer(kind=perflibs_i4), parameter :: perflibs_sparse_hint_spsv_invocations = 83

    integer(kind=perflibs_i4), parameter :: perflibs_col_major = 90
    integer(kind=perflibs_i4), parameter :: perflibs_row_major = 91

    integer(kind=perflibs_i4), parameter :: perflibs_sparse_structure_dense = 100
    integer(kind=perflibs_i4), parameter :: perflibs_sparse_structure_unstructured = 101
    integer(kind=perflibs_i4), parameter :: perflibs_sparse_structure_symmetric = 110
    integer(kind=perflibs_i4), parameter :: perflibs_sparse_structure_diagonal = 120
    integer(kind=perflibs_i4), parameter :: perflibs_sparse_structure_blockdiagonal = 130
    integer(kind=perflibs_i4), parameter :: perflibs_sparse_structure_banded = 140
    integer(kind=perflibs_i4), parameter :: perflibs_sparse_structure_triangular = 150
    integer(kind=perflibs_i4), parameter :: perflibs_sparse_structure_blocktriangular = 160
    integer(kind=perflibs_i4), parameter :: perflibs_sparse_structure_hermitian = 170
    integer(kind=perflibs_i4), parameter :: perflibs_sparse_structure_hpcg = 180

    integer(kind=perflibs_i4), parameter :: perflibs_sparse_memory_noallocs = 200
    integer(kind=perflibs_i4), parameter :: perflibs_sparse_memory_allocs = 201

    integer(kind=perflibs_i4), parameter :: perflibs_sparse_operation_notrans = 300
    integer(kind=perflibs_i4), parameter :: perflibs_sparse_operation_trans = 310
    integer(kind=perflibs_i4), parameter :: perflibs_sparse_operation_conjtrans = 320

    integer(kind=perflibs_i4), parameter :: perflibs_sparse_invocations_single = 400
    integer(kind=perflibs_i4), parameter :: perflibs_sparse_invocations_few = 410
    integer(kind=perflibs_i4), parameter :: perflibs_sparse_invocations_many = 420

    integer(kind=perflibs_i4), parameter :: perflibs_sparse_scalar_one = 500
    integer(kind=perflibs_i4), parameter :: perflibs_sparse_scalar_zero = 501
    integer(kind=perflibs_i4), parameter :: perflibs_sparse_scalar_any = 502

    integer(kind=perflibs_i4), parameter :: perflibs_sparse_spmm_strat_unset = 600
    integer(kind=perflibs_i4), parameter :: perflibs_sparse_spmm_strat_opt_no_struct = 601
    integer(kind=perflibs_i4), parameter :: perflibs_sparse_spmm_strat_opt_part_struct = 602
    integer(kind=perflibs_i4), parameter :: perflibs_sparse_spmm_strat_opt_full_struct = 603

    integer(kind=perflibs_i4), parameter :: perflibs_sparse_spsv_strat_unset = 700
    integer(kind=perflibs_i4), parameter :: perflibs_sparse_spsv_strat_separator_sparse = 701
    integer(kind=perflibs_i4), parameter :: perflibs_sparse_spsv_strat_separator_dense = 702

    integer(kind=perflibs_i4), parameter :: perflibs_sparse_norm_inf = 1001
    integer(kind=perflibs_i4), parameter :: perflibs_sparse_norm_frb = 1002

end module perflibs_sparse_params
