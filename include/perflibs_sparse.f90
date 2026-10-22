! SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
! affiliates <open-source-office@arm.com></text>
!
! SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

module perflibs_sparse

    use perflibs_kinds
    use perflibs_sparse_params

!   Sparse reproducibility control interfaces

    interface
        subroutine perflibs_sparse_set_reproducibility(mode, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i4), intent(in) :: mode
            integer, intent(out) :: info
        end subroutine perflibs_sparse_set_reproducibility
    end interface

    interface
        integer(kind=perflibs_i4) function perflibs_sparse_get_reproducibility()
            use perflibs_kinds
            implicit none
        end function perflibs_sparse_get_reproducibility
    end interface

!   Sparse matrix utility subroutines

    interface perflibs_spmat_create_csr

        subroutine perflibs_spmat_create_csr_s(A, m, n, row_ptr, col_indx, vals, flags, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer, intent(in) :: m, n
            integer, intent(in) :: row_ptr(m+1)
            integer, intent(in) :: col_indx(*)
            real(kind=perflibs_r32), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine perflibs_spmat_create_csr_s

        subroutine perflibs_spmat_create_csr_d(A, m, n, row_ptr, col_indx, vals, flags, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer, intent(in) :: m, n
            integer, intent(in) :: row_ptr(m+1)
            integer, intent(in) :: col_indx(*)
            real(kind=perflibs_r64), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine perflibs_spmat_create_csr_d

        subroutine perflibs_spmat_create_csr_c(A, m, n, row_ptr, col_indx, vals, flags, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer, intent(in) :: m, n
            integer, intent(in) :: row_ptr(m+1)
            integer, intent(in) :: col_indx(*)
            complex(kind=perflibs_r32), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine perflibs_spmat_create_csr_c

        subroutine perflibs_spmat_create_csr_z(A, m, n, row_ptr, col_indx, vals, flags, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer, intent(in) :: m, n
            integer, intent(in) :: row_ptr(m+1)
            integer, intent(in) :: col_indx(*)
            complex(kind=perflibs_r64), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine perflibs_spmat_create_csr_z

    end interface perflibs_spmat_create_csr

    interface perflibs_spmat_create_csc

        subroutine perflibs_spmat_create_csc_s(A, m, n, row_indx, col_ptr, vals, flags, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer, intent(in) :: m, n
            integer, intent(in) :: row_indx(*)
            integer, intent(in) :: col_ptr(n+1)
            real(kind=perflibs_r32), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine perflibs_spmat_create_csc_s

        subroutine perflibs_spmat_create_csc_d(A, m, n, row_indx, col_ptr, vals, flags, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer, intent(in) :: m, n
            integer, intent(in) :: row_indx(*)
            integer, intent(in) :: col_ptr(n+1)
            real(kind=perflibs_r64), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine perflibs_spmat_create_csc_d

        subroutine perflibs_spmat_create_csc_c(A, m, n, row_indx, col_ptr, vals, flags, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer, intent(in) :: m, n
            integer, intent(in) :: row_indx(*)
            integer, intent(in) :: col_ptr(n+1)
            complex(kind=perflibs_r32), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine perflibs_spmat_create_csc_c

        subroutine perflibs_spmat_create_csc_z(A, m, n, row_indx, col_ptr, vals, flags, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer, intent(in) :: m, n
            integer, intent(in) :: row_indx(*)
            integer, intent(in) :: col_ptr(n+1)
            complex(kind=perflibs_r64), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine perflibs_spmat_create_csc_z

    end interface perflibs_spmat_create_csc

    interface perflibs_spmat_create_coo

        subroutine perflibs_spmat_create_coo_s(A, m, n, nnz, row_indx, col_indx, vals, flags, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer, intent(in) :: m, n, nnz
            integer, intent(in) :: row_indx(*)
            integer, intent(in) :: col_indx(*)
            real(kind=perflibs_r32), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine perflibs_spmat_create_coo_s

        subroutine perflibs_spmat_create_coo_d(A, m, n, nnz, row_indx, col_indx, vals, flags, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer, intent(in) :: m, n, nnz
            integer, intent(in) :: row_indx(*)
            integer, intent(in) :: col_indx(*)
            real(kind=perflibs_r64), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine perflibs_spmat_create_coo_d

        subroutine perflibs_spmat_create_coo_c(A, m, n, nnz, row_indx, col_indx, vals, flags, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer, intent(in) :: m, n, nnz
            integer, intent(in) :: row_indx(*)
            integer, intent(in) :: col_indx(*)
            complex(kind=perflibs_r32), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine perflibs_spmat_create_coo_c

        subroutine perflibs_spmat_create_coo_z(A, m, n, nnz, row_indx, col_indx, vals, flags, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer, intent(in) :: m, n, nnz
            integer, intent(in) :: row_indx(*)
            integer, intent(in) :: col_indx(*)
            complex(kind=perflibs_r64), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine perflibs_spmat_create_coo_z

    end interface perflibs_spmat_create_coo

    interface
        subroutine perflibs_spmat_destroy(A, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer, intent(out) :: info
        end subroutine perflibs_spmat_destroy
    end interface

    interface
        subroutine perflibs_spmat_hint(A, sparse_hint_type, sparse_hint_value, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i4), intent(in) :: sparse_hint_type, sparse_hint_value
            integer, intent(out) :: info
        end subroutine perflibs_spmat_hint
    end interface

    interface
        subroutine perflibs_spmv_optimize(A, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer, intent(out) :: info
        end subroutine perflibs_spmv_optimize
    end interface

    interface perflibs_spmat_update

        subroutine perflibs_spmat_update_s(A, n_updates, row_indx, col_indx, vals, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer, intent(in) :: n_updates
            integer, intent(in) :: row_indx(n_updates)
            integer, intent(in) :: col_indx(n_updates)
            real(kind=perflibs_r32), intent(in) :: vals(n_updates)
            integer, intent(out) :: info
        end subroutine perflibs_spmat_update_s

        subroutine perflibs_spmat_update_d(A, n_updates, row_indx, col_indx, vals, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer, intent(in) :: n_updates
            integer, intent(in) :: row_indx(n_updates)
            integer, intent(in) :: col_indx(n_updates)
            real(kind=perflibs_r64), intent(in) :: vals(n_updates)
            integer, intent(out) :: info
        end subroutine perflibs_spmat_update_d

        subroutine perflibs_spmat_update_c(A, n_updates, row_indx, col_indx, vals, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer, intent(in) :: n_updates
            integer, intent(in) :: row_indx(n_updates)
            integer, intent(in) :: col_indx(n_updates)
            complex(kind=perflibs_r32), intent(in) :: vals(n_updates)
            integer, intent(out) :: info
        end subroutine perflibs_spmat_update_c

        subroutine perflibs_spmat_update_z(A, n_updates, row_indx, col_indx, vals, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer, intent(in) :: n_updates
            integer, intent(in) :: row_indx(n_updates)
            integer, intent(in) :: col_indx(n_updates)
            complex(kind=perflibs_r64), intent(in) :: vals(n_updates)
            integer, intent(out) :: info
        end subroutine perflibs_spmat_update_z

    end interface perflibs_spmat_update

    interface perflibs_spmat_create_dense

        subroutine perflibs_spmat_create_dense_s(A, layout, m, n, lda, vals, flags, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i4), intent(in) :: layout
            integer, intent(in) :: m
            integer, intent(in) :: n
            integer, intent(in) :: lda
            real(kind=perflibs_r32), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine perflibs_spmat_create_dense_s

        subroutine perflibs_spmat_create_dense_d(A, layout, m, n, lda, vals, flags, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i4), intent(in) :: layout
            integer, intent(in) :: m
            integer, intent(in) :: n
            integer, intent(in) :: lda
            real(kind=perflibs_r64), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine perflibs_spmat_create_dense_d

        subroutine perflibs_spmat_create_dense_c(A, layout, m, n, lda, vals, flags, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i4), intent(in) :: layout
            integer, intent(in) :: m
            integer, intent(in) :: n
            integer, intent(in) :: lda
            complex(kind=perflibs_r32), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine perflibs_spmat_create_dense_c

        subroutine perflibs_spmat_create_dense_z(A, layout, m, n, lda, vals, flags, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i4), intent(in) :: layout
            integer, intent(in) :: m
            integer, intent(in) :: n
            integer, intent(in) :: lda
            complex(kind=perflibs_r64), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine perflibs_spmat_create_dense_z

    end interface perflibs_spmat_create_dense

    interface perflibs_spmat_create_bsr

        subroutine perflibs_spmat_create_bsr_s(A, block_layout, m, n, block_size, row_ptr, col_indx, vals, flags, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i4), intent(in) :: block_layout
            integer, intent(in) :: m
            integer, intent(in) :: n
            integer, intent(in) :: block_size
            integer, intent(in) :: row_ptr(*)
            integer, intent(in) :: col_indx(*)
            real(kind=perflibs_r32), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine perflibs_spmat_create_bsr_s

        subroutine perflibs_spmat_create_bsr_d(A, block_layout, m, n, block_size, row_ptr, col_indx, vals, flags, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i4), intent(in) :: block_layout
            integer, intent(in) :: m
            integer, intent(in) :: n
            integer, intent(in) :: block_size
            integer, intent(in) :: row_ptr(*)
            integer, intent(in) :: col_indx(*)
            real(kind=perflibs_r64), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine perflibs_spmat_create_bsr_d

        subroutine perflibs_spmat_create_bsr_c(A, block_layout, m, n, block_size, row_ptr, col_indx, vals, flags, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i4), intent(in) :: block_layout
            integer, intent(in) :: m
            integer, intent(in) :: n
            integer, intent(in) :: block_size
            integer, intent(in) :: row_ptr(*)
            integer, intent(in) :: col_indx(*)
            complex(kind=perflibs_r32), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine perflibs_spmat_create_bsr_c

        subroutine perflibs_spmat_create_bsr_z(A, block_layout, m, n, block_size, row_ptr, col_indx, vals, flags, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i4), intent(in) :: block_layout
            integer, intent(in) :: m
            integer, intent(in) :: n
            integer, intent(in) :: block_size
            integer, intent(in) :: row_ptr(*)
            integer, intent(in) :: col_indx(*)
            complex(kind=perflibs_r64), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine perflibs_spmat_create_bsr_z

    end interface perflibs_spmat_create_bsr

    interface
        function perflibs_spmat_create_null(m, n)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8) :: perflibs_spmat_create_null
            integer, intent(in) :: m
            integer, intent(in) :: n
        end function perflibs_spmat_create_null
    end interface

    interface
        function perflibs_spmat_create_identity(n)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8) :: perflibs_spmat_create_identity
            integer, intent(in) :: n
        end function perflibs_spmat_create_identity
    end interface

    interface
        subroutine perflibs_spmat_query(A, index_base, m, n, nnz, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer, intent(out) :: index_base
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, intent(out) :: nnz
            integer, intent(out) :: info
        end subroutine perflibs_spmat_query
    end interface

    interface perflibs_spvec_create
        subroutine perflibs_spvec_create_s(x, index_base, n, nnz, indx, vals, flags, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: x
            integer, intent(in) :: index_base
            integer, intent(in) :: n
            integer, intent(in) :: nnz
            integer, intent(in) :: indx(*)
            real(kind=perflibs_r32), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine perflibs_spvec_create_s

        subroutine perflibs_spvec_create_d(x, index_base, n, nnz, indx, vals, flags, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: x
            integer, intent(in) :: index_base
            integer, intent(in) :: n
            integer, intent(in) :: nnz
            integer, intent(in) :: indx(*)
            real(kind=perflibs_r64), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine perflibs_spvec_create_d

        subroutine perflibs_spvec_create_c(x, index_base, n, nnz, indx, vals, flags, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: x
            integer, intent(in) :: index_base
            integer, intent(in) :: n
            integer, intent(in) :: nnz
            integer, intent(in) :: indx(*)
            complex(kind=perflibs_r32), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine perflibs_spvec_create_c

        subroutine perflibs_spvec_create_z(x, index_base, n, nnz, indx, vals, flags, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: x
            integer, intent(in) :: index_base
            integer, intent(in) :: n
            integer, intent(in) :: nnz
            integer, intent(in) :: indx(*)
            complex(kind=perflibs_r64), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine perflibs_spvec_create_z
    end interface perflibs_spvec_create

    interface
        subroutine perflibs_spvec_query(x, index_base, n, nnz, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: x
            integer, intent(out) :: index_base
            integer, intent(out) :: n
            integer, intent(out) :: nnz
            integer, intent(out) :: info
        end subroutine perflibs_spvec_query
    end interface

    interface
        subroutine perflibs_spvec_destroy(x, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: x
            integer, intent(out) :: info
        end subroutine perflibs_spvec_destroy
    end interface

    interface perflibs_spvec_export
        subroutine perflibs_spvec_export_s(x, index_base, n, nnz, indx, vals, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: x
            integer, intent(out) :: index_base
            integer, intent(out) :: n
            integer, intent(out) :: nnz
            integer, intent(out) :: indx(*)
            real(kind=perflibs_r32), intent(out) :: vals(*)
            integer, intent(out) :: info
        end subroutine perflibs_spvec_export_s

        subroutine perflibs_spvec_export_d(x, index_base, n, nnz, indx, vals, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: x
            integer, intent(out) :: index_base
            integer, intent(out) :: n
            integer, intent(out) :: nnz
            integer, intent(out) :: indx(*)
            real(kind=perflibs_r64), intent(out) :: vals(*)
            integer, intent(out) :: info
        end subroutine perflibs_spvec_export_d

        subroutine perflibs_spvec_export_c(x, index_base, n, nnz, indx, vals, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: x
            integer, intent(out) :: index_base
            integer, intent(out) :: n
            integer, intent(out) :: nnz
            integer, intent(out) :: indx(*)
            complex(kind=perflibs_r32), intent(out) :: vals(*)
            integer, intent(out) :: info
        end subroutine perflibs_spvec_export_c

        subroutine perflibs_spvec_export_z(x, index_base, n, nnz, indx, vals, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: x
            integer, intent(out) :: index_base
            integer, intent(out) :: n
            integer, intent(out) :: nnz
            integer, intent(out) :: indx(*)
            complex(kind=perflibs_r64), intent(out) :: vals(*)
            integer, intent(out) :: info
        end subroutine perflibs_spvec_export_z
    end interface perflibs_spvec_export

    interface perflibs_spvec_gather
        subroutine perflibs_spvec_gather_s(x_d, index_base, n, x_s, flags, info)
            use perflibs_kinds
            implicit none
            real(kind=perflibs_r32), intent(in) :: x_d(*)
            integer, intent(in) :: index_base
            integer, intent(in) :: n
            integer(kind=perflibs_i8), intent(in) :: x_s
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine perflibs_spvec_gather_s

        subroutine perflibs_spvec_gather_d(x_d, index_base, n, x_s, flags, info)
            use perflibs_kinds
            implicit none
            real(kind=perflibs_r64), intent(in) :: x_d(*)
            integer, intent(in) :: index_base
            integer, intent(in) :: n
            integer(kind=perflibs_i8), intent(in) :: x_s
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine perflibs_spvec_gather_d

        subroutine perflibs_spvec_gather_c(x_d, index_base, n, x_s, flags, info)
            use perflibs_kinds
            implicit none
            complex(kind=perflibs_r32), intent(in) :: x_d(*)
            integer, intent(in) :: index_base
            integer, intent(in) :: n
            integer(kind=perflibs_i8), intent(in) :: x_s
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine perflibs_spvec_gather_c

        subroutine perflibs_spvec_gather_z(x_d, index_base, n, x_s, flags, info)
            use perflibs_kinds
            implicit none
            complex(kind=perflibs_r64), intent(in) :: x_d(*)
            integer, intent(in) :: index_base
            integer, intent(in) :: n
            integer(kind=perflibs_i8), intent(in) :: x_s
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine perflibs_spvec_gather_z
    end interface perflibs_spvec_gather

    interface perflibs_spvec_scatter
        subroutine perflibs_spvec_scatter_s(x_s, x_d, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: x_s
            real(kind=perflibs_r32), intent(out) :: x_d(*)
            integer, intent(out) :: info
        end subroutine perflibs_spvec_scatter_s

        subroutine perflibs_spvec_scatter_d(x_s, x_d, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: x_s
            real(kind=perflibs_r64), intent(out) :: x_d(*)
            integer, intent(out) :: info
        end subroutine perflibs_spvec_scatter_d

        subroutine perflibs_spvec_scatter_c(x_s, x_d, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: x_s
            complex(kind=perflibs_r32), intent(out) :: x_d(*)
            integer, intent(out) :: info
        end subroutine perflibs_spvec_scatter_c

        subroutine perflibs_spvec_scatter_z(x_s, x_d, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: x_s
            complex(kind=perflibs_r64), intent(out) :: x_d(*)
            integer, intent(out) :: info
        end subroutine perflibs_spvec_scatter_z
    end interface perflibs_spvec_scatter

    interface perflibs_spvec_update
        subroutine perflibs_spvec_update_s(x, n_updates, indx, vals, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: x
            integer, intent(in) :: n_updates
            integer, intent(in) :: indx(*)
            real(kind=perflibs_r32), intent(in) :: vals(*)
            integer, intent(out) :: info
        end subroutine perflibs_spvec_update_s

        subroutine perflibs_spvec_update_d(x, n_updates, indx, vals, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: x
            integer, intent(in) :: n_updates
            integer, intent(in) :: indx(*)
            real(kind=perflibs_r64), intent(in) :: vals(*)
            integer, intent(out) :: info
        end subroutine perflibs_spvec_update_d

        subroutine perflibs_spvec_update_c(x, n_updates, indx, vals, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: x
            integer, intent(in) :: n_updates
            integer, intent(in) :: indx(*)
            complex(kind=perflibs_r32), intent(in) :: vals(*)
            integer, intent(out) :: info
        end subroutine perflibs_spvec_update_c

        subroutine perflibs_spvec_update_z(x, n_updates, indx, vals, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: x
            integer, intent(in) :: n_updates
            integer, intent(in) :: indx(*)
            complex(kind=perflibs_r64), intent(in) :: vals(*)
            integer, intent(out) :: info
        end subroutine perflibs_spvec_update_z
    end interface perflibs_spvec_update

    interface perflibs_spdot_exec
        subroutine perflibs_spdot_exec_s(x, y, result, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: x
            real(kind=perflibs_r32), intent(in) :: y(*)
            real(kind=perflibs_r32), intent(out) :: result
            integer, intent(out) :: info
        end subroutine perflibs_spdot_exec_s

        subroutine perflibs_spdot_exec_d(x, y, result, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: x
            real(kind=perflibs_r64), intent(in) :: y(*)
            real(kind=perflibs_r64), intent(out) :: result
            integer, intent(out) :: info
        end subroutine perflibs_spdot_exec_d
    end interface perflibs_spdot_exec

    interface perflibs_spdotu_exec
        subroutine perflibs_spdotu_exec_c(x, y, result, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: x
            complex(kind=perflibs_r32), intent(in) :: y(*)
            complex(kind=perflibs_r32), intent(out) :: result
            integer, intent(out) :: info
        end subroutine perflibs_spdotu_exec_c

        subroutine perflibs_spdotu_exec_z(x, y, result, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: x
            complex(kind=perflibs_r64), intent(in) :: y(*)
            complex(kind=perflibs_r64), intent(out) :: result
            integer, intent(out) :: info
        end subroutine perflibs_spdotu_exec_z
    end interface perflibs_spdotu_exec

    interface perflibs_spdotc_exec
        subroutine perflibs_spdotc_exec_c(x, y, result, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: x
            complex(kind=perflibs_r32), intent(in) :: y(*)
            complex(kind=perflibs_r32), intent(out) :: result
            integer, intent(out) :: info
        end subroutine perflibs_spdotc_exec_c

        subroutine perflibs_spdotc_exec_z(x, y, result, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: x
            complex(kind=perflibs_r64), intent(in) :: y(*)
            complex(kind=perflibs_r64), intent(out) :: result
            integer, intent(out) :: info
        end subroutine perflibs_spdotc_exec_z
    end interface perflibs_spdotc_exec

    interface perflibs_spaxpby_exec
        subroutine perflibs_spaxpby_exec_s(alpha, x, beta, y, info)
            use perflibs_kinds
            implicit none
            real(kind=perflibs_r32), intent(in) :: alpha
            integer(kind=perflibs_i8), intent(in) :: x
            real(kind=perflibs_r32), intent(in) :: beta
            real(kind=perflibs_r32), intent(inout) :: y(*)
            integer, intent(out) :: info
        end subroutine perflibs_spaxpby_exec_s

        subroutine perflibs_spaxpby_exec_d(alpha, x, beta, y, info)
            use perflibs_kinds
            implicit none
            real(kind=perflibs_r64), intent(in) :: alpha
            integer(kind=perflibs_i8), intent(in) :: x
            real(kind=perflibs_r64), intent(in) :: beta
            real(kind=perflibs_r64), intent(inout) :: y(*)
            integer, intent(out) :: info
        end subroutine perflibs_spaxpby_exec_d

        subroutine perflibs_spaxpby_exec_c(alpha, x, beta, y, info)
            use perflibs_kinds
            implicit none
            complex(kind=perflibs_r32), intent(in) :: alpha
            integer(kind=perflibs_i8), intent(in) :: x
            complex(kind=perflibs_r32), intent(in) :: beta
            complex(kind=perflibs_r32), intent(inout) :: y(*)
            integer, intent(out) :: info
        end subroutine perflibs_spaxpby_exec_c

        subroutine perflibs_spaxpby_exec_z(alpha, x, beta, y, info)
            use perflibs_kinds
            implicit none
            complex(kind=perflibs_r64), intent(in) :: alpha
            integer(kind=perflibs_i8), intent(in) :: x
            complex(kind=perflibs_r64), intent(in) :: beta
            complex(kind=perflibs_r64), intent(inout) :: y(*)
            integer, intent(out) :: info
        end subroutine perflibs_spaxpby_exec_z
    end interface perflibs_spaxpby_exec

    interface perflibs_spwaxpby_exec
        subroutine perflibs_spwaxpby_exec_s(alpha, x, beta, y, w, info)
            use perflibs_kinds
            implicit none
            real(kind=perflibs_r32), intent(in) :: alpha
            integer(kind=perflibs_i8), intent(in) :: x
            real(kind=perflibs_r32), intent(in) :: beta
            real(kind=perflibs_r32), intent(in) :: y(*)
            real(kind=perflibs_r32), intent(out) :: w(*)
            integer, intent(out) :: info
        end subroutine perflibs_spwaxpby_exec_s

        subroutine perflibs_spwaxpby_exec_d(alpha, x, beta, y, w, info)
            use perflibs_kinds
            implicit none
            real(kind=perflibs_r64), intent(in) :: alpha
            integer(kind=perflibs_i8), intent(in) :: x
            real(kind=perflibs_r64), intent(in) :: beta
            real(kind=perflibs_r64), intent(in) :: y(*)
            real(kind=perflibs_r64), intent(out) :: w(*)
            integer, intent(out) :: info
        end subroutine perflibs_spwaxpby_exec_d

        subroutine perflibs_spwaxpby_exec_c(alpha, x, beta, y, w, info)
            use perflibs_kinds
            implicit none
            complex(kind=perflibs_r32), intent(in) :: alpha
            integer(kind=perflibs_i8), intent(in) :: x
            complex(kind=perflibs_r32), intent(in) :: beta
            complex(kind=perflibs_r32), intent(in) :: y(*)
            complex(kind=perflibs_r32), intent(out) :: w(*)
            integer, intent(out) :: info
        end subroutine perflibs_spwaxpby_exec_c

        subroutine perflibs_spwaxpby_exec_z(alpha, x, beta, y, w, info)
            use perflibs_kinds
            implicit none
            complex(kind=perflibs_r64), intent(in) :: alpha
            integer(kind=perflibs_i8), intent(in) :: x
            complex(kind=perflibs_r64), intent(in) :: beta
            complex(kind=perflibs_r64), intent(in) :: y(*)
            complex(kind=perflibs_r64), intent(out) :: w(*)
            integer, intent(out) :: info
        end subroutine perflibs_spwaxpby_exec_z
    end interface perflibs_spwaxpby_exec

    interface
        subroutine perflibs_spmm_optimize(transA, transB, alpha, A, B, beta, C, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i4), intent(in) :: transA
            integer(kind=perflibs_i4), intent(in) :: transB
            integer(kind=perflibs_i4), intent(in) :: alpha
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(in) :: B
            integer(kind=perflibs_i4), intent(in) :: beta
            integer(kind=perflibs_i8), intent(in) :: C
            integer, intent(out) :: info
        end subroutine perflibs_spmm_optimize
    end interface

    interface
        subroutine perflibs_spadd_optimize(transA, transB, alpha, A, beta, B, C, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i4), intent(in) :: transA
            integer(kind=perflibs_i4), intent(in) :: transB
            integer(kind=perflibs_i4), intent(in) :: alpha
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i4), intent(in) :: beta
            integer(kind=perflibs_i8), intent(in) :: B
            integer(kind=perflibs_i8), intent(in) :: C
            integer, intent(out) :: info
        end subroutine perflibs_spadd_optimize
    end interface

    interface
        subroutine perflibs_spsm_optimize(transA, A, X, alpha, Y, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i4), intent(in) :: transA
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(in) :: X
            integer(kind=perflibs_i4), intent(in) :: alpha
            integer(kind=perflibs_i8), intent(in) :: Y
            integer, intent(out) :: info
        end subroutine perflibs_spsm_optimize
    end interface

    interface
        subroutine perflibs_spsv_optimize(A, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer, intent(out) :: info
        end subroutine perflibs_spsv_optimize
    end interface

    interface
        subroutine perflibs_spelmm_optimize(transA, transB, alpha, A, B, beta, C, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i4), intent(in) :: transA
            integer(kind=perflibs_i4), intent(in) :: transB
            integer(kind=perflibs_i4), intent(in) :: alpha
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(in) :: B
            integer(kind=perflibs_i4), intent(in) :: beta
            integer(kind=perflibs_i8), intent(in) :: C
            integer, intent(out) :: info
        end subroutine perflibs_spelmm_optimize
    end interface

    interface
        subroutine perflibs_sddmm_optimize(transA, transB, alpha, A, B, beta, C, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i4), intent(in) :: transA
            integer(kind=perflibs_i4), intent(in) :: transB
            integer(kind=perflibs_i4), intent(in) :: alpha
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(in) :: B
            integer(kind=perflibs_i4), intent(in) :: beta
            integer(kind=perflibs_i8), intent(in) :: C
            integer, intent(out) :: info
        end subroutine perflibs_sddmm_optimize
    end interface


!   Sparse matrix--vector execution subroutines

    interface perflibs_spmv_exec

        subroutine perflibs_spmv_exec_s(trans, alpha, A, x, beta, y, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i4), intent(in) :: trans
            real(kind=perflibs_r32), intent(in) :: alpha
            integer(kind=perflibs_i8), intent(in) :: A
            real(kind=perflibs_r32), intent(in) :: x(*)
            real(kind=perflibs_r32), intent(in) :: beta
            real(kind=perflibs_r32), intent(inout) :: y(*)
            integer, intent(out) :: info
        end subroutine perflibs_spmv_exec_s

        subroutine perflibs_spmv_exec_d(trans, alpha, A, x, beta, y, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i4), intent(in) :: trans
            real(kind=perflibs_r64), intent(in) :: alpha
            integer(kind=perflibs_i8), intent(in) :: A
            real(kind=perflibs_r64), intent(in) :: x(*)
            real(kind=perflibs_r64), intent(in) :: beta
            real(kind=perflibs_r64), intent(inout) :: y(*)
            integer, intent(out) :: info
        end subroutine perflibs_spmv_exec_d

        subroutine perflibs_spmv_exec_c(trans, alpha, A, x, beta, y, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i4), intent(in) :: trans
            complex(kind=perflibs_r32), intent(in) :: alpha
            integer(kind=perflibs_i8), intent(in) :: A
            complex(kind=perflibs_r32), intent(in) :: x(*)
            complex(kind=perflibs_r32), intent(in) :: beta
            complex(kind=perflibs_r32), intent(inout) :: y(*)
            integer, intent(out) :: info
        end subroutine perflibs_spmv_exec_c

        subroutine perflibs_spmv_exec_z(trans, alpha, A, x, beta, y, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i4), intent(in) :: trans
            complex(kind=perflibs_r64), intent(in) :: alpha
            integer(kind=perflibs_i8), intent(in) :: A
            complex(kind=perflibs_r64), intent(in) :: x(*)
            complex(kind=perflibs_r64), intent(in) :: beta
            complex(kind=perflibs_r64), intent(inout) :: y(*)
            integer, intent(out) :: info
        end subroutine perflibs_spmv_exec_z

    end interface perflibs_spmv_exec

!   Sparse matrix--matrix execution subroutines

    interface perflibs_spmm_exec

        subroutine perflibs_spmm_exec_s(transA, transB, alpha, A, B, beta, C, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i4), intent(in) :: transA
            integer(kind=perflibs_i4), intent(in) :: transB
            real(kind=perflibs_r32), intent(in) :: alpha
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(in) :: B
            real(kind=perflibs_r32), intent(in) :: beta
            integer(kind=perflibs_i8), intent(in) :: C
            integer, intent(out) :: info
        end subroutine perflibs_spmm_exec_s

        subroutine perflibs_spmm_exec_d(transA, transB, alpha, A, B, beta, C, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i4), intent(in) :: transA
            integer(kind=perflibs_i4), intent(in) :: transB
            real(kind=perflibs_r64), intent(in) :: alpha
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(in) :: B
            real(kind=perflibs_r64), intent(in) :: beta
            integer(kind=perflibs_i8), intent(in) :: C
            integer, intent(out) :: info
        end subroutine perflibs_spmm_exec_d

        subroutine perflibs_spmm_exec_c(transA, transB, alpha, A, B, beta, C, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i4), intent(in) :: transA
            integer(kind=perflibs_i4), intent(in) :: transB
            complex(kind=perflibs_r32), intent(in) :: alpha
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(in) :: B
            complex(kind=perflibs_r32), intent(in) :: beta
            integer(kind=perflibs_i8), intent(in) :: C
            integer, intent(out) :: info
        end subroutine perflibs_spmm_exec_c

        subroutine perflibs_spmm_exec_z(transA, transB, alpha, A, B, beta, C, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i4), intent(in) :: transA
            integer(kind=perflibs_i4), intent(in) :: transB
            complex(kind=perflibs_r64), intent(in) :: alpha
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(in) :: B
            complex(kind=perflibs_r64), intent(in) :: beta
            integer(kind=perflibs_i8), intent(in) :: C
            integer, intent(out) :: info
        end subroutine perflibs_spmm_exec_z

    end interface perflibs_spmm_exec

!   Sparse matrix triangular solve execution subroutines

    interface perflibs_spsm_exec

        subroutine perflibs_spsm_exec_s(transA, A, X, alpha, Y, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i4), intent(in) :: transA
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(in) :: X
            real(kind=perflibs_r32), intent(in) :: alpha
            integer(kind=perflibs_i8), intent(in) :: Y
            integer, intent(out) :: info
        end subroutine perflibs_spsm_exec_s

        subroutine perflibs_spsm_exec_d(transA, A, X, alpha, Y, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i4), intent(in) :: transA
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(in) :: X
            real(kind=perflibs_r64), intent(in) :: alpha
            integer(kind=perflibs_i8), intent(in) :: Y
            integer, intent(out) :: info
        end subroutine perflibs_spsm_exec_d

        subroutine perflibs_spsm_exec_c(transA, A, X, alpha, Y, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i4), intent(in) :: transA
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(in) :: X
            complex(kind=perflibs_r32), intent(in) :: alpha
            integer(kind=perflibs_i8), intent(in) :: Y
            integer, intent(out) :: info
        end subroutine perflibs_spsm_exec_c

        subroutine perflibs_spsm_exec_z(transA, A, X, alpha, Y, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i4), intent(in) :: transA
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(in) :: X
            complex(kind=perflibs_r64), intent(in) :: alpha
            integer(kind=perflibs_i8), intent(in) :: Y
            integer, intent(out) :: info
        end subroutine perflibs_spsm_exec_z

    end interface perflibs_spsm_exec

    interface perflibs_spsv_exec

        subroutine perflibs_spsv_exec_s(trans, A, x, alpha, y, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i4), intent(in) :: trans
            integer(kind=perflibs_i8), intent(in) :: A
            real(kind=perflibs_r32), intent(out) :: x(*)
            real(kind=perflibs_r32), intent(in) :: alpha
            real(kind=perflibs_r32), intent(in) :: y(*)
            integer, intent(out) :: info
        end subroutine perflibs_spsv_exec_s

        subroutine perflibs_spsv_exec_d(trans, A, x, alpha, y, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i4), intent(in) :: trans
            integer(kind=perflibs_i8), intent(in) :: A
            real(kind=perflibs_r64), intent(out) :: x(*)
            real(kind=perflibs_r64), intent(in) :: alpha
            real(kind=perflibs_r64), intent(in) :: y(*)
            integer, intent(out) :: info
        end subroutine perflibs_spsv_exec_d

        subroutine perflibs_spsv_exec_c(trans, A, x, alpha, y, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i4), intent(in) :: trans
            integer(kind=perflibs_i8), intent(in) :: A
            complex(kind=perflibs_r32), intent(out) :: x(*)
            complex(kind=perflibs_r32), intent(in) :: alpha
            complex(kind=perflibs_r32), intent(in) :: y(*)
            integer, intent(out) :: info
        end subroutine perflibs_spsv_exec_c

        subroutine perflibs_spsv_exec_z(trans, A, x, alpha, y, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i4), intent(in) :: trans
            integer(kind=perflibs_i8), intent(in) :: A
            complex(kind=perflibs_r64), intent(out) :: x(*)
            complex(kind=perflibs_r64), intent(in) :: alpha
            complex(kind=perflibs_r64), intent(in) :: y(*)
            integer, intent(out) :: info
        end subroutine perflibs_spsv_exec_z

    end interface perflibs_spsv_exec

    interface perflibs_spadd_exec

        subroutine perflibs_spadd_exec_s(transA, transB, alpha, A, beta, B, C, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i4), intent(in) :: transA
            integer(kind=perflibs_i4), intent(in) :: transB
            real(kind=perflibs_r32), intent(in) :: alpha
            integer(kind=perflibs_i8), intent(in) :: A
            real(kind=perflibs_r32), intent(in) :: beta
            integer(kind=perflibs_i8), intent(in) :: B
            integer(kind=perflibs_i8), intent(in) :: C
            integer, intent(out) :: info
        end subroutine perflibs_spadd_exec_s

        subroutine perflibs_spadd_exec_d(transA, transB, alpha, A, beta, B, C, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i4), intent(in) :: transA
            integer(kind=perflibs_i4), intent(in) :: transB
            real(kind=perflibs_r64), intent(in) :: alpha
            integer(kind=perflibs_i8), intent(in) :: A
            real(kind=perflibs_r64), intent(in) :: beta
            integer(kind=perflibs_i8), intent(in) :: B
            integer(kind=perflibs_i8), intent(in) :: C
            integer, intent(out) :: info
        end subroutine perflibs_spadd_exec_d

        subroutine perflibs_spadd_exec_c(transA, transB, alpha, A, beta, B, C, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i4), intent(in) :: transA
            integer(kind=perflibs_i4), intent(in) :: transB
            complex(kind=perflibs_r32), intent(in) :: alpha
            integer(kind=perflibs_i8), intent(in) :: A
            complex(kind=perflibs_r32), intent(in) :: beta
            integer(kind=perflibs_i8), intent(in) :: B
            integer(kind=perflibs_i8), intent(in) :: C
            integer, intent(out) :: info
        end subroutine perflibs_spadd_exec_c

        subroutine perflibs_spadd_exec_z(transA, transB, alpha, A, beta, B, C, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i4), intent(in) :: transA
            integer(kind=perflibs_i4), intent(in) :: transB
            complex(kind=perflibs_r64), intent(in) :: alpha
            integer(kind=perflibs_i8), intent(in) :: A
            complex(kind=perflibs_r64), intent(in) :: beta
            integer(kind=perflibs_i8), intent(in) :: B
            integer(kind=perflibs_i8), intent(in) :: C
            integer, intent(out) :: info
        end subroutine perflibs_spadd_exec_z

    end interface perflibs_spadd_exec

!   Sparse matrix export subroutines

    interface perflibs_spmat_export_csr

        subroutine perflibs_spmat_export_csr_s(A, index_base, m, n, row_ptr, col_indx, vals, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer, intent(in) :: index_base
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, allocatable, intent(out) :: row_ptr(:)
            integer, allocatable, intent(out) :: col_indx(:)
            real(kind=perflibs_r32), allocatable, intent(out) :: vals(:)
            integer, intent(out) :: info
        end subroutine perflibs_spmat_export_csr_s

        subroutine perflibs_spmat_export_csr_d(A, index_base, m, n, row_ptr, col_indx, vals, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer, intent(in) :: index_base
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, allocatable, intent(out) :: row_ptr(:)
            integer, allocatable, intent(out) :: col_indx(:)
            real(kind=perflibs_r64), allocatable, intent(out) :: vals(:)
            integer, intent(out) :: info
        end subroutine perflibs_spmat_export_csr_d

        subroutine perflibs_spmat_export_csr_c(A, index_base, m, n, row_ptr, col_indx, vals, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer, intent(in) :: index_base
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, allocatable, intent(out) :: row_ptr(:)
            integer, allocatable, intent(out) :: col_indx(:)
            complex(kind=perflibs_r32), allocatable, intent(out) :: vals(:)
            integer, intent(out) :: info
        end subroutine perflibs_spmat_export_csr_c

        subroutine perflibs_spmat_export_csr_z(A, index_base, m, n, row_ptr, col_indx, vals, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer, intent(in) :: index_base
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, allocatable, intent(out) :: row_ptr(:)
            integer, allocatable, intent(out) :: col_indx(:)
            complex(kind=perflibs_r64), allocatable, intent(out) :: vals(:)
            integer, intent(out) :: info
        end subroutine perflibs_spmat_export_csr_z

    end interface perflibs_spmat_export_csr

    interface perflibs_spmat_export_csc

        subroutine perflibs_spmat_export_csc_s(A, index_base, m, n, row_indx, col_ptr, vals, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer, intent(in) :: index_base
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, allocatable, intent(out) :: row_indx(:)
            integer, allocatable, intent(out) :: col_ptr(:)
            real(kind=perflibs_r32), allocatable, intent(out) :: vals(:)
            integer, intent(out) :: info
        end subroutine perflibs_spmat_export_csc_s

        subroutine perflibs_spmat_export_csc_d(A, index_base, m, n, row_indx, col_ptr, vals, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer, intent(in) :: index_base
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, allocatable, intent(out) :: row_indx(:)
            integer, allocatable, intent(out) :: col_ptr(:)
            real(kind=perflibs_r64), allocatable, intent(out) :: vals(:)
            integer, intent(out) :: info
        end subroutine perflibs_spmat_export_csc_d

        subroutine perflibs_spmat_export_csc_c(A, index_base, m, n, row_indx, col_ptr, vals, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer, intent(in) :: index_base
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, allocatable, intent(out) :: row_indx(:)
            integer, allocatable, intent(out) :: col_ptr(:)
            complex(kind=perflibs_r32), allocatable, intent(out) :: vals(:)
            integer, intent(out) :: info
        end subroutine perflibs_spmat_export_csc_c

        subroutine perflibs_spmat_export_csc_z(A, index_base, m, n, row_indx, col_ptr, vals, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer, intent(in) :: index_base
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, allocatable, intent(out) :: row_indx(:)
            integer, allocatable, intent(out) :: col_ptr(:)
            complex(kind=perflibs_r64), allocatable, intent(out) :: vals(:)
            integer, intent(out) :: info
        end subroutine perflibs_spmat_export_csc_z

    end interface perflibs_spmat_export_csc

    interface perflibs_spmat_export_coo

        subroutine perflibs_spmat_export_coo_s(A, m, n, nnz, row_indx, col_indx, vals, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, intent(out) :: nnz
            integer, allocatable, intent(out) :: row_indx(:)
            integer, allocatable, intent(out) :: col_indx(:)
            real(kind=perflibs_r32), allocatable, intent(out) :: vals(:)
            integer, intent(out) :: info
        end subroutine perflibs_spmat_export_coo_s

        subroutine perflibs_spmat_export_coo_d(A, m, n, nnz, row_indx, col_indx, vals, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, intent(out) :: nnz
            integer, allocatable, intent(out) :: row_indx(:)
            integer, allocatable, intent(out) :: col_indx(:)
            real(kind=perflibs_r64), allocatable, intent(out) :: vals(:)
            integer, intent(out) :: info
        end subroutine perflibs_spmat_export_coo_d

        subroutine perflibs_spmat_export_coo_c(A, m, n, nnz, row_indx, col_indx, vals, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, intent(out) :: nnz
            integer, allocatable, intent(out) :: row_indx(:)
            integer, allocatable, intent(out) :: col_indx(:)
            complex(kind=perflibs_r32), allocatable, intent(out) :: vals(:)
            integer, intent(out) :: info
        end subroutine perflibs_spmat_export_coo_c

        subroutine perflibs_spmat_export_coo_z(A, m, n, nnz, row_indx, col_indx, vals, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, intent(out) :: nnz
            integer, allocatable, intent(out) :: row_indx(:)
            integer, allocatable, intent(out) :: col_indx(:)
            complex(kind=perflibs_r64), allocatable, intent(out) :: vals(:)
            integer, intent(out) :: info
        end subroutine perflibs_spmat_export_coo_z

    end interface perflibs_spmat_export_coo

    interface perflibs_spmat_export_dense

        subroutine perflibs_spmat_export_dense_s(A, layout, m, n, vals, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i4), intent(in) :: layout
            integer, intent(out) :: m
            integer, intent(out) :: n
            real(kind=perflibs_r32), allocatable, intent(out) :: vals(:)
            integer, intent(out) :: info
        end subroutine perflibs_spmat_export_dense_s

        subroutine perflibs_spmat_export_dense_d(A, layout, m, n, vals, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i4), intent(in) :: layout
            integer, intent(out) :: m
            integer, intent(out) :: n
            real(kind=perflibs_r64), allocatable, intent(out) :: vals(:)
            integer, intent(out) :: info
        end subroutine perflibs_spmat_export_dense_d

        subroutine perflibs_spmat_export_dense_c(A, layout, m, n, vals, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i4), intent(in) :: layout
            integer, intent(out) :: m
            integer, intent(out) :: n
            complex(kind=perflibs_r32), allocatable, intent(out) :: vals(:)
            integer, intent(out) :: info
        end subroutine perflibs_spmat_export_dense_c

        subroutine perflibs_spmat_export_dense_z(A, layout, m, n, vals, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i4), intent(in) :: layout
            integer, intent(out) :: m
            integer, intent(out) :: n
            complex(kind=perflibs_r64), allocatable, intent(out) :: vals(:)
            integer, intent(out) :: info
        end subroutine perflibs_spmat_export_dense_z

    end interface perflibs_spmat_export_dense

    interface perflibs_spmat_export_bsr

        subroutine perflibs_spmat_export_bsr_s(A, block_layout, index_base, m, n, block_size, row_ptr, col_indx, vals, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i4), intent(in) :: block_layout
            integer, intent(in) :: index_base
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, intent(out) :: block_size
            integer, allocatable, intent(out) :: row_ptr(:)
            integer, allocatable, intent(out) :: col_indx(:)
            real(kind=perflibs_r32), allocatable, intent(out) :: vals(:)
            integer, intent(out) :: info
        end subroutine perflibs_spmat_export_bsr_s

        subroutine perflibs_spmat_export_bsr_d(A, block_layout, index_base, m, n, block_size, row_ptr, col_indx, vals, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i4), intent(in) :: block_layout
            integer, intent(in) :: index_base
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, intent(out) :: block_size
            integer, allocatable, intent(out) :: row_ptr(:)
            integer, allocatable, intent(out) :: col_indx(:)
            real(kind=perflibs_r64), allocatable, intent(out) :: vals(:)
            integer, intent(out) :: info
        end subroutine perflibs_spmat_export_bsr_d

        subroutine perflibs_spmat_export_bsr_c(A, block_layout, index_base, m, n, block_size, row_ptr, col_indx, vals, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i4), intent(in) :: block_layout
            integer, intent(in) :: index_base
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, intent(out) :: block_size
            integer, allocatable, intent(out) :: row_ptr(:)
            integer, allocatable, intent(out) :: col_indx(:)
            complex(kind=perflibs_r32), allocatable, intent(out) :: vals(:)
            integer, intent(out) :: info
        end subroutine perflibs_spmat_export_bsr_c

        subroutine perflibs_spmat_export_bsr_z(A, block_layout, index_base, m, n, block_size, row_ptr, col_indx, vals, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i4), intent(in) :: block_layout
            integer, intent(in) :: index_base
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, intent(out) :: block_size
            integer, allocatable, intent(out) :: row_ptr(:)
            integer, allocatable, intent(out) :: col_indx(:)
            complex(kind=perflibs_r64), allocatable, intent(out) :: vals(:)
            integer, intent(out) :: info
        end subroutine perflibs_spmat_export_bsr_z

    end interface perflibs_spmat_export_bsr

    interface perflibs_sprot_exec

        subroutine perflibs_sprot_exec_s(x, y, c, s, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: x
            real(kind=perflibs_r32), intent(inout) :: y(*)
            real(kind=perflibs_r32), intent(in) :: c
            real(kind=perflibs_r32), intent(in) :: s
            integer, intent(out) :: info
        end subroutine perflibs_sprot_exec_s

        subroutine perflibs_sprot_exec_d(x, y, c, s, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: x
            real(kind=perflibs_r64), intent(inout) :: y(*)
            real(kind=perflibs_r64), intent(in) :: c
            real(kind=perflibs_r64), intent(in) :: s
            integer, intent(out) :: info
        end subroutine perflibs_sprot_exec_d

        subroutine perflibs_sprot_exec_c(x, y, c, s, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: x
            complex(kind=perflibs_r32), intent(inout) :: y(*)
            real(kind=perflibs_r32), intent(in) :: c
            complex(kind=perflibs_r32), intent(in) :: s
            integer, intent(out) :: info
        end subroutine perflibs_sprot_exec_c

        subroutine perflibs_sprot_exec_z(x, y, c, s, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: x
            complex(kind=perflibs_r64), intent(inout) :: y(*)
            real(kind=perflibs_r64), intent(in) :: c
            complex(kind=perflibs_r64), intent(in) :: s
            integer, intent(out) :: info
        end subroutine perflibs_sprot_exec_z

        subroutine perflibs_sprot_exec_cs(x, y, c, s, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: x
            complex(kind=perflibs_r32), intent(inout) :: y(*)
            real(kind=perflibs_r32), intent(in) :: c
            real(kind=perflibs_r32), intent(in) :: s
            integer, intent(out) :: info
        end subroutine perflibs_sprot_exec_cs

        subroutine perflibs_sprot_exec_zd(x, y, c, s, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: x
            complex(kind=perflibs_r64), intent(inout) :: y(*)
            real(kind=perflibs_r64), intent(in) :: c
            real(kind=perflibs_r64), intent(in) :: s
            integer, intent(out) :: info
        end subroutine perflibs_sprot_exec_zd

    end interface perflibs_sprot_exec

    interface
        subroutine perflibs_spmat_print_err(A)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
        end subroutine perflibs_spmat_print_err

        subroutine perflibs_spvec_print_err(x)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: x
        end subroutine perflibs_spvec_print_err
    end interface

    interface perflibs_spnorm_exec

        subroutine perflibs_spnorm_exec_cs(A, nrm, result, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i4), intent(in) :: nrm
            real(kind=perflibs_r32), intent(out) :: result
            integer, intent(out) :: info
        end subroutine perflibs_spnorm_exec_cs

        subroutine perflibs_spnorm_exec_zd(A, nrm, result, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i4), intent(in) :: nrm
            real(kind=perflibs_r64), intent(out) :: result
            integer, intent(out) :: info
        end subroutine perflibs_spnorm_exec_zd

    end interface perflibs_spnorm_exec

    interface perflibs_spelmm_exec

        subroutine perflibs_spelmm_exec_s(transA, transB, alpha, A, B, beta, C, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i4), intent(in) :: transA
            integer(kind=perflibs_i4), intent(in) :: transB
            real(kind=perflibs_r32), intent(in) :: alpha
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(in) :: B
            real(kind=perflibs_r32), intent(in) :: beta
            integer(kind=perflibs_i8), intent(in) :: C
            integer, intent(out) :: info
        end subroutine perflibs_spelmm_exec_s

        subroutine perflibs_spelmm_exec_d(transA, transB, alpha, A, B, beta, C, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i4), intent(in) :: transA
            integer(kind=perflibs_i4), intent(in) :: transB
            real(kind=perflibs_r64), intent(in) :: alpha
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(in) :: B
            real(kind=perflibs_r64), intent(in) :: beta
            integer(kind=perflibs_i8), intent(in) :: C
            integer, intent(out) :: info
        end subroutine perflibs_spelmm_exec_d

        subroutine perflibs_spelmm_exec_c(transA, transB, alpha, A, B, beta, C, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i4), intent(in) :: transA
            integer(kind=perflibs_i4), intent(in) :: transB
            complex(kind=perflibs_r32), intent(in) :: alpha
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(in) :: B
            complex(kind=perflibs_r32), intent(in) :: beta
            integer(kind=perflibs_i8), intent(in) :: C
            integer, intent(out) :: info
        end subroutine perflibs_spelmm_exec_c

        subroutine perflibs_spelmm_exec_z(transA, transB, alpha, A, B, beta, C, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i4), intent(in) :: transA
            integer(kind=perflibs_i4), intent(in) :: transB
            complex(kind=perflibs_r64), intent(in) :: alpha
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(in) :: B
            complex(kind=perflibs_r64), intent(in) :: beta
            integer(kind=perflibs_i8), intent(in) :: C
            integer, intent(out) :: info
        end subroutine perflibs_spelmm_exec_z

    end interface perflibs_spelmm_exec

    interface perflibs_sddmm_exec

        subroutine perflibs_sddmm_exec_s(transA, transB, alpha, A, B, beta, C, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i4), intent(in) :: transA
            integer(kind=perflibs_i4), intent(in) :: transB
            real(kind=perflibs_r32), intent(in) :: alpha
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(in) :: B
            real(kind=perflibs_r32), intent(in) :: beta
            integer(kind=perflibs_i8), intent(in) :: C
            integer, intent(out) :: info
        end subroutine perflibs_sddmm_exec_s

        subroutine perflibs_sddmm_exec_d(transA, transB, alpha, A, B, beta, C, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i4), intent(in) :: transA
            integer(kind=perflibs_i4), intent(in) :: transB
            real(kind=perflibs_r64), intent(in) :: alpha
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(in) :: B
            real(kind=perflibs_r64), intent(in) :: beta
            integer(kind=perflibs_i8), intent(in) :: C
            integer, intent(out) :: info
        end subroutine perflibs_sddmm_exec_d

        subroutine perflibs_sddmm_exec_c(transA, transB, alpha, A, B, beta, C, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i4), intent(in) :: transA
            integer(kind=perflibs_i4), intent(in) :: transB
            complex(kind=perflibs_r32), intent(in) :: alpha
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(in) :: B
            complex(kind=perflibs_r32), intent(in) :: beta
            integer(kind=perflibs_i8), intent(in) :: C
            integer, intent(out) :: info
        end subroutine perflibs_sddmm_exec_c

        subroutine perflibs_sddmm_exec_z(transA, transB, alpha, A, B, beta, C, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i4), intent(in) :: transA
            integer(kind=perflibs_i4), intent(in) :: transB
            complex(kind=perflibs_r64), intent(in) :: alpha
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(in) :: B
            complex(kind=perflibs_r64), intent(in) :: beta
            integer(kind=perflibs_i8), intent(in) :: C
            integer, intent(out) :: info
        end subroutine perflibs_sddmm_exec_z

    end interface perflibs_sddmm_exec

    interface perflibs_spscale_exec

        subroutine perflibs_spscale_exec_s(alpha, A, info)
            use perflibs_kinds
            implicit none
            real(kind=perflibs_r32), intent(in) :: alpha
            integer(kind=perflibs_i8), intent(in) :: A
            integer, intent(out) :: info
        end subroutine perflibs_spscale_exec_s

        subroutine perflibs_spscale_exec_d(alpha, A, info)
            use perflibs_kinds
            implicit none
            real(kind=perflibs_r64), intent(in) :: alpha
            integer(kind=perflibs_i8), intent(in) :: A
            integer, intent(out) :: info
        end subroutine perflibs_spscale_exec_d

        subroutine perflibs_spscale_exec_c(alpha, A, info)
            use perflibs_kinds
            implicit none
            complex(kind=perflibs_r32), intent(in) :: alpha
            integer(kind=perflibs_i8), intent(in) :: A
            integer, intent(out) :: info
        end subroutine perflibs_spscale_exec_c

        subroutine perflibs_spscale_exec_z(alpha, A, info)
            use perflibs_kinds
            implicit none
            complex(kind=perflibs_r64), intent(in) :: alpha
            integer(kind=perflibs_i8), intent(in) :: A
            integer, intent(out) :: info
        end subroutine perflibs_spscale_exec_z

    end interface perflibs_spscale_exec

    interface perflibs_sptranspose_exec

        subroutine perflibs_sptranspose_exec(transA, A, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i4), intent(in) :: transA
            integer(kind=perflibs_i8), intent(in) :: A
            integer, intent(out) :: info
        end subroutine perflibs_sptranspose_exec

    end interface perflibs_sptranspose_exec

end module perflibs_sparse
