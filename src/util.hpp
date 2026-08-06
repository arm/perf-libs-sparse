/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include "int.hpp"
#include <perflibs_sparse.h>

#include <cassert>
#include <cmath>
#include <complex>
#include <string>
#include <type_traits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace perflibs::sparse {

// --- type utilities ----------------------------------------------------------

template <typename T> struct remove_complex {
  using type = T;
};

template <typename T> struct remove_complex<std::complex<T>> {
  using type = T;
};

template <typename T> using remove_complex_t = typename remove_complex<T>::type;

template <typename T> struct is_complex : public std::false_type {};
template <typename T>
struct is_complex<std::complex<T>> : public std::true_type {};

template <typename T>
struct is_complex<const std::complex<T>> : public std::true_type {};

template <typename T> inline constexpr bool is_complex_v = is_complex<T>::value;

// --- helpers ----------------------------------------------------------------

// Conjugate helper: identity for real scalars; std::conj for complex.
inline constexpr float conj(float x) noexcept { return x; }
inline constexpr double conj(double x) noexcept { return x; }

inline std::complex<float>
conj(std::complex<float> x) noexcept(noexcept(std::conj(x))) {
  return std::conj(x);
}
inline std::complex<double>
conj(std::complex<double> x) noexcept(noexcept(std::conj(x))) {
  return std::conj(x);
}

template <typename T> static inline bool is_real_nan(T x) {
  return std::isnan(std::real(x));
}

namespace consts {

/// pi in type of FloatType as constexpr
/// example usage   auto circum = pi<double>*2.0*r
template <typename FloatType>
constexpr FloatType pi = FloatType(3.1415926535897932385L);

} // namespace consts

namespace omp {

#ifdef _OPENMP
constexpr bool is_mp = true;
#else
constexpr bool is_mp = false;
#endif

/**
 * Return whether the next parallel level will be a parallel region in which we
 * want to use multithreading.
 *
 * Note: OMP_NESTED is deprecated, and we ignore it.
 */
inline bool get_nested() noexcept {
#ifdef _OPENMP
  // Levels are indexed from zero; return true if current level is smaller than
  // the max.
  return omp_get_active_level() < omp_get_max_active_levels();
#else
  return false;
#endif
}

inline int get_max_threads() noexcept {
#ifdef _OPENMP
  return get_nested() ? omp_get_max_threads() : 1;
#else
  return 1;
#endif
}

inline int get_max_active_levels() noexcept {
#ifdef _OPENMP
  return omp_get_max_active_levels();
#else
  return 1;
#endif
}

inline int get_active_level() noexcept {
#ifdef _OPENMP
  return omp_get_active_level();
#else
  return 0;
#endif
}

inline void set_max_active_levels(int levels) noexcept {
#ifdef _OPENMP
  omp_set_max_active_levels(levels);
#else
  (void)levels;
#endif
}

inline void set_num_threads(int nthreads) noexcept {
#ifdef _OPENMP
  omp_set_num_threads(nthreads);
#else
  (void)nthreads;
#endif
}

inline int get_num_threads() noexcept {
#ifdef _OPENMP
  return omp_get_num_threads();
#else
  return 1;
#endif
}

inline int get_thread_num() noexcept {
#ifdef _OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

} // namespace omp

} // namespace perflibs::sparse

/**
 * Rounds n UP to the nearest multiple of r, if n is not already a multiple
 */
template <typename T1, typename T2>
__attribute__((always_inline)) inline std::common_type_t<T1, T2>
iround(T1 n_in, T2 r_in) {
  using type = std::common_type_t<T1, T2>;
  const type n = n_in;
  const type r = r_in;

  const type diff = n % r;

  if (diff)
    return n - diff + r;

  return n;
}

/**
 * Rounds n UP to the nearest multiple of r, then divides by r.
 */
template <typename IntType1, typename IntType2>
__attribute__((always_inline)) inline std::common_type_t<IntType1, IntType2>
iround_div(IntType1 n, IntType2 r) {
  return iround(n, r) / r;
}

// Information we need to know, not hints. This is discovered on matrix
// creation.
enum perflibs_sparse_matrix_shape_t {
  PERFLIBS_SPARSE_SHAPE_RECTANGULAR = 1000,
  PERFLIBS_SPARSE_SHAPE_UPPER_TRIANGULAR,
  PERFLIBS_SPARSE_SHAPE_LOWER_TRIANGULAR,
  PERFLIBS_SPARSE_SHAPE_DIAGONAL
};
enum perflibs_sparse_matrix_diag_t {
  PERFLIBS_SPARSE_DIAG_UNIT = 2000,
  PERFLIBS_SPARSE_DIAG_KNOWN_UNIT, // we know where the unit diagonal element is
                                   // and don't need to check
  PERFLIBS_SPARSE_DIAG_NON_UNIT,
  PERFLIBS_SPARSE_DIAG_KNOWN_NON_UNIT, // we know where the non-unit diagonal
                                       // element is and don't need to check
  PERFLIBS_SPARSE_DIAG_ZERO // to indicate at least one missing/zero diagonal
                            // element
};

enum sparse_hint_value_internal {
  PERFLIBS_OPERATION_NOTRANS = PERFLIBS_SPARSE_OPERATION_NOTRANS,
  PERFLIBS_OPERATION_CONJNOTRANS = 301,
  PERFLIBS_OPERATION_TRANS = PERFLIBS_SPARSE_OPERATION_TRANS,
  PERFLIBS_OPERATION_CONJTRANS = PERFLIBS_SPARSE_OPERATION_CONJTRANS,
  PERFLIBS_SHAPE_RECTANGULAR = PERFLIBS_SPARSE_SHAPE_RECTANGULAR,
  PERFLIBS_SHAPE_UPPER_TRIANGULAR = PERFLIBS_SPARSE_SHAPE_UPPER_TRIANGULAR,
  PERFLIBS_SHAPE_LOWER_TRIANGULAR = PERFLIBS_SPARSE_SHAPE_LOWER_TRIANGULAR,
  PERFLIBS_SHAPE_DIAGONAL = PERFLIBS_SPARSE_SHAPE_DIAGONAL,
  PERFLIBS_DIAG_UNIT = PERFLIBS_SPARSE_DIAG_UNIT,
  PERFLIBS_DIAG_KNOWN_UNIT = PERFLIBS_SPARSE_DIAG_KNOWN_UNIT,
  PERFLIBS_DIAG_NON_UNIT = PERFLIBS_SPARSE_DIAG_NON_UNIT,
  PERFLIBS_DIAG_KNOWN_NON_UNIT = PERFLIBS_SPARSE_DIAG_KNOWN_NON_UNIT,
  PERFLIBS_DIAG_ZERO = PERFLIBS_SPARSE_DIAG_ZERO
};

static inline sparse_hint_value_internal
to_internal_enum(perflibs_sparse_hint_value ext) {
  switch (ext) {
  case PERFLIBS_SPARSE_OPERATION_NOTRANS:
    return PERFLIBS_OPERATION_NOTRANS;

  case PERFLIBS_SPARSE_OPERATION_TRANS:
    return PERFLIBS_OPERATION_TRANS;

  case PERFLIBS_SPARSE_OPERATION_CONJTRANS:
    return PERFLIBS_OPERATION_CONJTRANS;

  default:
    assert(false && "Invalid external transpose flag");
    return PERFLIBS_OPERATION_NOTRANS;
  }
}

static inline perflibs_sparse_hint_value
to_external_enum(sparse_hint_value_internal in) {
  switch (in) {
  case PERFLIBS_OPERATION_NOTRANS:
    return PERFLIBS_SPARSE_OPERATION_NOTRANS;

  case PERFLIBS_OPERATION_TRANS:
    return PERFLIBS_SPARSE_OPERATION_TRANS;

  case PERFLIBS_OPERATION_CONJTRANS:
    return PERFLIBS_SPARSE_OPERATION_CONJTRANS;

  default:
    assert(false && "Invalid internal transpose flag");
    return PERFLIBS_SPARSE_OPERATION_NOTRANS;
  }
}

/// The formats of sparse matrices
enum spmat_format_t {
  /// Compressed sparse row
  perflibs_format_csr = 0,
  /// Compressed sparse column
  perflibs_format_csc,
  /// Sell-c-sigma
  perflibs_format_scs,
  /// Coordinate
  perflibs_format_coo,
  /// Dense
  perflibs_format_dense,
  /// Supernodal
  perflibs_format_supernodal,
  /// Null matrix representation
  perflibs_format_null,
  /// Identity matrix
  perflibs_format_identity,
  /// Block sparse rows
  perflibs_format_bsr,
};

/// Used to record what SpMM optimizations have been applied to a matrix in the
/// optimize/symbolic phase
enum perflibs_spmm_opt_t {
  /// PERFLIBS_SPARSE_SPMM_STRAT_OPT_NO_STRUCT - i.e. optimization phase doesn't
  /// compute structure
  perflibs_spmm_single_phase,
  /// PERFLIBS_SPARSE_SPMM_STRAT_OPT_PART_STRUCT - i.e. optimization phase
  /// computes
  /// row_ptr only
  perflibs_spmm_part_struct,
  /// PERFLIBS_SPARSE_SPMM_STRAT_OPT_FULL_STRUCT - i.e. optimization phase
  /// computes
  /// row_ptr and col_indx values
  perflibs_spmm_full_struct
};

struct sp_error_t {
  perflibs_int_t perflibs_error_type;
  perflibs_int_t perflibs_error_code;
  std::string err_msg;
};

enum perflibs_datatype {
  PERFLIBS_DATATYPE_SINGLE = 1,
  PERFLIBS_DATATYPE_DOUBLE = 2,
  PERFLIBS_DATATYPE_CPLXSINGLE = 4,
  PERFLIBS_DATATYPE_CPLXDOUBLE = 8
};

/// Gives the options for the different parallel decomposition
/// strategies for a matrix. Not all of the strategies must be
/// honored by all matrix classes
enum class perflibs_parallel_decomp_strategy {
  /// Assign the same number of rows to each thread
  static_rows,
  /// Assign full rows, trying to balance the number of
  /// non-zeros between the threads
  nnz_full_rows,
  /// Assign non-zero values across threads evenly
  nnz
};

namespace perflibs::sparse {

// The banner is used elsewhere
extern const char *banner;

perflibs_status_t update_matrix_error(sp_error_t &error_handle,
                                      perflibs_int_t i);

void print_wrap(std::string str, bool include_banner);

} // namespace perflibs::sparse
