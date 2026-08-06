# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
# affiliates <open-source-office@arm.com></text>
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.24)

# This CMakeLists.txt is intended to be added via:
#   add_subdirectory(fortran_tests)
# after libsparse_* targets are defined in the top-level.
project(fortran_sparse_tests LANGUAGES C Fortran)

include(GNUInstallDirs)

# -----------------------
# Configuration / knobs
# -----------------------
# Which libsparse target to link against:
#   - If empty, we prefer sparse_static if it exists, otherwise sparse_shared.
set(SPARSE_LIB_TARGET "" CACHE STRING "CMake target name of libsparse to link (e.g. sparse_static or sparse_shared)")

# -----------------------
# Resolve sparse target
# -----------------------
if(SPARSE_LIB_TARGET STREQUAL "")
  if(TARGET sparse_shared)
    set(SPARSE_LIB_TARGET sparse_shared)
  elseif(TARGET sparse_static)
    set(SPARSE_LIB_TARGET sparse_static)
  else()
    message(FATAL_ERROR
      "fortran_tests: Could not find sparse_static or sparse_shared.\n"
      "Set -DSPARSE_LIB_TARGET=<target> or ensure libsparse is defined before add_subdirectory(fortran_tests)."
    )
  endif()
else()
  if(NOT TARGET ${SPARSE_LIB_TARGET})
    message(FATAL_ERROR "fortran_tests: SPARSE_LIB_TARGET='${SPARSE_LIB_TARGET}' is not a known CMake target.")
  endif()
endif()

# -----------------------
# Get the Fortran module directory for libsparse
# -----------------------
get_target_property(SPARSE_MOD_DIR ${SPARSE_LIB_TARGET} Fortran_MODULE_DIRECTORY)
if(SPARSE_MOD_DIR MATCHES "-NOTFOUND$" OR SPARSE_MOD_DIR STREQUAL "")
  message(FATAL_ERROR
    "fortran_tests: Target '${SPARSE_LIB_TARGET}' does not have Fortran_MODULE_DIRECTORY set.\n"
    "In the top-level CMakeLists.txt, set e.g.:\n"
    "  set_target_properties(${SPARSE_LIB_TARGET} PROPERTIES\n"
    "    Fortran_MODULE_DIRECTORY \"${CMAKE_BINARY_DIR}/fortran_mod/static\")\n"
  )
endif()

# -----------------------
# C helper: sparse_export_ftn_libfortran.c
# -----------------------
# Build the C helper into an OBJECT library so its object file can be
# linked directly into each test executable.
add_library(fortran_sparse_c_support OBJECT
  "${CMAKE_CURRENT_SOURCE_DIR}/sparse_export_ftn_libfortran.c"
)
if(NOT (SPARSE_ENABLE_COVERAGE AND CMAKE_Fortran_COMPILER_ID STREQUAL "LLVMFlang"))
  sparse_enable_coverage(fortran_sparse_c_support)
endif()

if(APPLE)
  apply_macos_deployment_target(fortran_sparse_c_support)
endif()

target_include_directories(fortran_sparse_c_support PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/../include"
)

target_include_directories(fortran_sparse_c_support PRIVATE $<TARGET_PROPERTY:${_SPARSE_CBLAS_TARGET},INTERFACE_INCLUDE_DIRECTORIES>)

if(SPARSE_ENABLE_INT64)
  target_compile_definitions(fortran_sparse_c_support PRIVATE INTEGER64=1)
endif()

# -----------------------
# Discover Fortran test sources
# -----------------------
file(GLOB FORTRAN_TEST_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.F90"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.f90"
)

if(FORTRAN_TEST_SOURCES STREQUAL "")
  message(WARNING "fortran_tests: No Fortran test sources found under ${CMAKE_CURRENT_SOURCE_DIR}")
endif()

# -----------------------
# Helper: add 4 variants per test source
# -----------------------
function(add_sparse_fortran_test_variants src)
  get_filename_component(_base "${src}" NAME_WE)

  set(_r8_flag
    "$<$<STREQUAL:${CMAKE_Fortran_COMPILER_ID},NVHPC>:-r8>"
    "$<$<STREQUAL:${CMAKE_Fortran_COMPILER_ID},GNU>:-fdefault-real-8>"
    "$<$<STREQUAL:${CMAKE_Fortran_COMPILER_ID},Flang>:-fdefault-real-8>"
    "$<$<STREQUAL:${CMAKE_Fortran_COMPILER_ID},LLVMFlang>:-fdefault-real-8>"
  )

  # Common bits:
  # - Fortran_PREPROCESS ON -> enables preprocessing for .F90
  # - Set Fortran_MODULE_DIRECTORY so the compiler finds .mod from libsparse
  # - link against ${SPARSE_LIB_TARGET}
  # - include the C helper object from fortran_sparse_c_support

  # r4: real (default real kind)
  set(_tgt_r4 "${_base}_r4")
  add_executable(${_tgt_r4} "${src}")
  sparse_enable_coverage(${_tgt_r4})
  set_target_properties(${_tgt_r4} PROPERTIES Fortran_PREPROCESS ON)
  target_compile_definitions(${_tgt_r4} PRIVATE FTYPE=real TEST_ID=r4)
  if(NOT _SPARSE_FORTRAN_INT64_FLAG STREQUAL "")
    target_compile_options(${_tgt_r4} PRIVATE ${_SPARSE_FORTRAN_INT64_FLAG})
  endif()
  set_target_properties(${_tgt_r4} PROPERTIES Fortran_MODULE_DIRECTORY "${SPARSE_MOD_DIR}")
  if(NOT WIN32)
    set_target_properties(${_tgt_r4} PROPERTIES LINKER_LANGUAGE Fortran)
  endif()
  target_sources(${_tgt_r4} PRIVATE $<TARGET_OBJECTS:fortran_sparse_c_support>)
  target_link_libraries(${_tgt_r4} PRIVATE ${SPARSE_LIB_TARGET})
  add_dependencies(${_tgt_r4} ${SPARSE_LIB_TARGET})
  add_test(NAME ${_tgt_r4} COMMAND ${_tgt_r4})

  # r8: real + default-real-8 (gfortran's -fdefault-real-8)
  set(_tgt_r8 "${_base}_r8")
  add_executable(${_tgt_r8} "${src}")
  sparse_enable_coverage(${_tgt_r8})
  set_target_properties(${_tgt_r8} PROPERTIES Fortran_PREPROCESS ON)
  target_compile_definitions(${_tgt_r8} PRIVATE FTYPE=real TEST_ID=r8)
  if(NOT _SPARSE_FORTRAN_INT64_FLAG STREQUAL "")
    target_compile_options(${_tgt_r8} PRIVATE ${_SPARSE_FORTRAN_INT64_FLAG})
  endif()
  set_target_properties(${_tgt_r8} PROPERTIES Fortran_MODULE_DIRECTORY "${SPARSE_MOD_DIR}")
  if(NOT WIN32)
    set_target_properties(${_tgt_r8} PROPERTIES LINKER_LANGUAGE Fortran)
  endif()
  target_compile_options(${_tgt_r8} PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:${_r8_flag}>)
  target_sources(${_tgt_r8} PRIVATE $<TARGET_OBJECTS:fortran_sparse_c_support>)
  target_link_libraries(${_tgt_r8} PRIVATE ${SPARSE_LIB_TARGET})
  add_dependencies(${_tgt_r8} ${SPARSE_LIB_TARGET})
  add_test(NAME ${_tgt_r8} COMMAND ${_tgt_r8})

  # c4: complex (default real kind)
  set(_tgt_c4 "${_base}_c4")
  add_executable(${_tgt_c4} "${src}")
  sparse_enable_coverage(${_tgt_c4})
  set_target_properties(${_tgt_c4} PROPERTIES Fortran_PREPROCESS ON)
  target_compile_definitions(${_tgt_c4} PRIVATE FTYPE=complex TEST_ID=c4)
  if(NOT _SPARSE_FORTRAN_INT64_FLAG STREQUAL "")
    target_compile_options(${_tgt_c4} PRIVATE ${_SPARSE_FORTRAN_INT64_FLAG})
  endif()
  set_target_properties(${_tgt_c4} PROPERTIES Fortran_MODULE_DIRECTORY "${SPARSE_MOD_DIR}")
  if(NOT WIN32)
    set_target_properties(${_tgt_c4} PROPERTIES LINKER_LANGUAGE Fortran)
  endif()
  target_sources(${_tgt_c4} PRIVATE $<TARGET_OBJECTS:fortran_sparse_c_support>)
  target_link_libraries(${_tgt_c4} PRIVATE ${SPARSE_LIB_TARGET})
  add_dependencies(${_tgt_c4} ${SPARSE_LIB_TARGET})
  add_test(NAME ${_tgt_c4} COMMAND ${_tgt_c4})

  # c8: complex + default-real-8
  set(_tgt_c8 "${_base}_c8")
  add_executable(${_tgt_c8} "${src}")
  sparse_enable_coverage(${_tgt_c8})
  set_target_properties(${_tgt_c8} PROPERTIES Fortran_PREPROCESS ON)
  target_compile_definitions(${_tgt_c8} PRIVATE FTYPE=complex TEST_ID=c8)
  if(NOT _SPARSE_FORTRAN_INT64_FLAG STREQUAL "")
    target_compile_options(${_tgt_c8} PRIVATE ${_SPARSE_FORTRAN_INT64_FLAG})
  endif()
  set_target_properties(${_tgt_c8} PROPERTIES Fortran_MODULE_DIRECTORY "${SPARSE_MOD_DIR}")
  if(NOT WIN32)
    set_target_properties(${_tgt_c8} PROPERTIES LINKER_LANGUAGE Fortran)
  endif()
  target_compile_options(${_tgt_c8} PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:${_r8_flag}>)
  target_sources(${_tgt_c8} PRIVATE $<TARGET_OBJECTS:fortran_sparse_c_support>)
  target_link_libraries(${_tgt_c8} PRIVATE ${SPARSE_LIB_TARGET})
  add_dependencies(${_tgt_c8} ${SPARSE_LIB_TARGET})
  add_test(NAME ${_tgt_c8} COMMAND ${_tgt_c8})
endfunction()

# -----------------------
# Build and test everything
# -----------------------
enable_testing()

foreach(src IN LISTS FORTRAN_TEST_SOURCES)
  add_sparse_fortran_test_variants("${src}")
endforeach()

# Convenience target to build all Fortran tests in one go
add_custom_target(build_fortran_tests)
foreach(src IN LISTS FORTRAN_TEST_SOURCES)
  get_filename_component(_base "${src}" NAME_WE)
  add_dependencies(build_fortran_tests
    "${_base}_r4" "${_base}_r8" "${_base}_c4" "${_base}_c8"
  )
endforeach()
