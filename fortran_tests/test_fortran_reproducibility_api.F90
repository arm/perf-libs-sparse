! SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
! affiliates <open-source-office@arm.com></text>
!
! SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

program test_fortran_reproducibility_api
    use perflibs_sparse
    use perflibs_sparse_params
    implicit none

    integer :: info

    if (perflibs_sparse_get_reproducibility() .ne. perflibs_sparse_reproducibility_disabled) then
        write (*,*) "Expected default reproducibility mode to be disabled"
        stop 1
    endif

    call perflibs_sparse_set_reproducibility(perflibs_sparse_reproducibility_enabled, info)
    if (info .ne. perflibs_status_success) then
        write (*,*) "Expected enabled reproducibility setter to succeed, got", info
        stop 2
    endif
    if (perflibs_sparse_get_reproducibility() .ne. perflibs_sparse_reproducibility_enabled) then
        write (*,*) "Expected reproducibility mode to be enabled"
        stop 3
    endif

    call perflibs_sparse_set_reproducibility(perflibs_sparse_reproducibility_disabled, info)
    if (info .ne. perflibs_status_success) then
        write (*,*) "Expected disabled reproducibility setter to succeed, got", info
        stop 4
    endif
    if (perflibs_sparse_get_reproducibility() .ne. perflibs_sparse_reproducibility_disabled) then
        write (*,*) "Expected reproducibility mode to be disabled"
        stop 5
    endif

    call perflibs_sparse_set_reproducibility(7_perflibs_i4, info)
    if (info .ne. perflibs_status_input_parameter_error) then
        write (*,*) "Expected invalid reproducibility setter to fail, got", info
        stop 6
    endif
    if (perflibs_sparse_get_reproducibility() .ne. perflibs_sparse_reproducibility_disabled) then
        write (*,*) "Expected invalid setter to leave reproducibility mode disabled"
        stop 7
    endif

    write (*,*) "Success - test_fortran_reproducibility_api"
end program test_fortran_reproducibility_api
