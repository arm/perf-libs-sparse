# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
# affiliates <open-source-office@arm.com></text>
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.24)

project(perflibs_sparse
  VERSION 26.04.0
  LANGUAGES C Fortran CXX ASM
)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# === Cache variables which we want to query during component creation ===
set(SPARSE_COMPILER_ID "${CMAKE_CXX_COMPILER_ID}" CACHE STRING "C++ Compiler name")
set(SPARSE_COMPILER_VERSION "${CMAKE_CXX_COMPILER_VERSION}" CACHE STRING "C++ Compiler version")
mark_as_advanced(SPARSE_COMPILER_ID)
mark_as_advanced(SPARSE_COMPILER_VERSION)
set(SPARSE_FTN_COMPILER_ID "${CMAKE_Fortran_COMPILER_ID}" CACHE STRING "Fortran compiler name")
set(SPARSE_FTN_COMPILER_VERSION "${CMAKE_Fortran_COMPILER_VERSION}" CACHE STRING "Fortran compiler version")
mark_as_advanced(SPARSE_FTN_COMPILER_ID)
mark_as_advanced(SPARSE_FTN_COMPILER_VERSION)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, defaulting to Release")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Make the C++ standard explicit and consistent
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -----------------------
# Options
# -----------------------
option(SPARSE_BUILD_SHARED   "Build shared libsparse" ON)
option(SPARSE_BUILD_STATIC   "Build static libsparse" ON)
option(SPARSE_ENABLE_FORTRAN "Enable Fortran interface/module build" ON)
option(SPARSE_ENABLE_OPENMP "Build with OpenMP and produce *_mp library suffix" OFF)
option(SPARSE_ENABLE_ASAN   "Build with -fsanitize=address and produce *_asan library suffix" OFF)
option(SPARSE_ENABLE_COVERAGE "Build with compiler coverage instrumentation for supported toolchains" OFF)
option(SPARSE_ENABLE_INT64 "Build libsparse with 8-byte default INTEGER (ILP64) and define INTEGER64=1 in C/C++" OFF)
option(SPARSE_ENABLE_ARM64EC "Enable Arm64EC ABI build (--target=arm64ec-pc-windows-msvc)" OFF)
option(SPARSE_ENABLE_TESTS "Enable building of test executables" ON)
option(SPARSE_ENABLE_SVE "Enable SVE kernels/specializations" OFF)
option(SPARSE_USE_ARMPL "Use Arm Performance Libraries (requires env var ARMPL_DIR)" ON)

# Make all binaries go to a bin directory - important for Windows testing of shared lib
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# -----------------------
# ABI / naming: lp64 vs ilp64
# -----------------------
set(_SPARSE_ABI_SUFFIX "lp64")
if(SPARSE_ENABLE_INT64)
  set(_SPARSE_ABI_SUFFIX "ilp64")
endif()

set(_SPARSE_PACKAGE_NAME "perflibs_sparse_${_SPARSE_ABI_SUFFIX}")
set(_SPARSE_PACKAGE_NAMESPACE "${_SPARSE_PACKAGE_NAME}::")
set(_SPARSE_EXPORT_NAME "${_SPARSE_PACKAGE_NAME}Targets")
set(_SPARSE_BLAS_TARGET "${_SPARSE_PACKAGE_NAMESPACE}blas")
set(_SPARSE_CBLAS_TARGET "${_SPARSE_PACKAGE_NAMESPACE}cblas")

if(NOT DEFINED ARMPL_LIB_BASENAME)
  if(SPARSE_ENABLE_INT64)
    set(ARMPL_LIB_BASENAME "armpl_ilp64" CACHE STRING "ArmPL library basename under $ARMPL_DIR/lib (e.g. armpl_lp64, armpl_ilp64)" FORCE)
  else()
    set(ARMPL_LIB_BASENAME "armpl_lp64" CACHE STRING "ArmPL library basename under $ARMPL_DIR/lib (e.g. armpl_lp64, armpl_ilp64)" FORCE)
  endif()
endif()


# -----------------------
# User-overridable default flags (space-separated strings)
# Users can override these at configure time:
#   -DDEFAULT_SPARSE_COMMON_FLAGS="..." -DDEFAULT_SPARSE_CXX_ONLY_FLAGS="..."
# -----------------------
set(DEFAULT_SPARSE_COMMON_FLAGS
  "-fasynchronous-unwind-tables -Wuninitialized -mtune=neoverse-n1 -fPIC -O3 -funroll-loops -nostdlib -Wall -Werror -Wno-missing-braces -Wno-unknown-pragmas -Wno-error=return-type -Wno-alloc-size-larger-than"
  CACHE STRING "Default common C/C++ compile flags (space-separated). Filtered by compiler support."
)

set(DEFAULT_SPARSE_CXX_ONLY_FLAGS
  "-fno-exceptions -fcx-fortran-rules -fno-rtti"
  CACHE STRING "Default C++-only compile flags (space-separated). Filtered by C++ compiler support."
)

# Target OS used by Mako assembly boilerplate. Auto-detect if not set.
set(SPARSE_TARGET_OS "" CACHE STRING "Target OS for Mako-rendered assembly: linux|mac|windows|windows_arm64ec (auto if empty)")
set_property(CACHE SPARSE_TARGET_OS PROPERTY STRINGS "" linux mac windows windows_arm64ec)
if(SPARSE_TARGET_OS STREQUAL "")
  if(APPLE)
    set(SPARSE_TARGET_OS "mac" CACHE STRING "Target OS for Mako-rendered assembly: linux|mac|windows|windows_arm64ec (auto if empty)" FORCE)
  elseif(WIN32)
    if(SPARSE_ENABLE_ARM64EC)
      set(SPARSE_TARGET_OS "windows_arm64ec" CACHE STRING "Target OS for Mako-rendered assembly: linux|mac|windows|windows_arm64ec (auto if empty)" FORCE)
    else()
      set(SPARSE_TARGET_OS "windows" CACHE STRING "Target OS for Mako-rendered assembly: linux|mac|windows|windows_arm64ec (auto if empty)" FORCE)
    endif()
  elseif(UNIX)
    set(SPARSE_TARGET_OS "linux" CACHE STRING "Target OS for Mako-rendered assembly: linux|mac|windows|windows_arm64ec (auto if empty)" FORCE)
  else()
    message(FATAL_ERROR "Unsupported OS: ${CMAKE_SYSTEM_NAME}")
  endif()
endif()
message(STATUS "Building for OS: ${SPARSE_TARGET_OS}")
mark_as_advanced(SPARSE_TARGET_OS)

# -----------------------
# macOS deployment target propagation (C + Fortran)
# -----------------------
if(APPLE)
  # CMAKE_OSX_DEPLOYMENT_TARGET should come from the top-level cache,
  # but ensure it's present for this subdir as well.
  if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "" FORCE)
  endif()

  set(_MACOS_MIN_FLAG "-mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")

  # Helper to apply to any target (C / Fortran objects and final link)
  function(apply_macos_deployment_target tgt)
    target_compile_options(${tgt} PUBLIC
      $<$<COMPILE_LANGUAGE:C>:${_MACOS_MIN_FLAG}>
      $<$<COMPILE_LANGUAGE:Fortran>:${_MACOS_MIN_FLAG}>
    )
    target_link_options(${tgt} PUBLIC
      ${_MACOS_MIN_FLAG}
      LINKER:-w
    )
  endfunction()
endif()

# Build both by default
if(NOT SPARSE_BUILD_SHARED AND NOT SPARSE_BUILD_STATIC)
  message(FATAL_ERROR "At least one of SPARSE_BUILD_SHARED or SPARSE_BUILD_STATIC must be ON.")
endif()

# ---- Windows UCRT naming convention  ----
set(SPARSE_WIN_UCRT_PREFIX "")
if(WIN32)
  # /MT => MultiThreaded => extra "lib" prefix
  # /MD => MultiThreadedDLL => no extra prefix
  if(CMAKE_MSVC_RUNTIME_LIBRARY STREQUAL "MultiThreaded")
    set(SPARSE_WIN_UCRT_PREFIX "lib")
  else()
    set(SPARSE_WIN_UCRT_PREFIX "")
  endif()
endif()

set(SPARSE_LIB_BASENAME "${SPARSE_WIN_UCRT_PREFIX}plsparse_${_SPARSE_ABI_SUFFIX}")

if(WIN32)
  set(SPARSE_ENABLE_SVE OFF CACHE BOOL "Enable SVE kernels/specializations" FORCE)
endif()

# If OpenMP requested -> add compile & link flags
if(SPARSE_ENABLE_OPENMP)
  add_compile_options(-fopenmp)
  add_link_options(-fopenmp)
endif()

# If ASAN requested -> add compile & link flags
if(SPARSE_ENABLE_ASAN)
  add_compile_options(-fsanitize=address)
  add_link_options(-fsanitize=address)
endif()

function(sparse_enable_coverage target_name)
  if(SPARSE_ENABLE_COVERAGE)
    get_target_property(_sparse_target_type ${target_name} TYPE)

    target_compile_options(${target_name} PRIVATE
      $<$<COMPILE_LANG_AND_ID:C,GNU>:--coverage>
      $<$<COMPILE_LANG_AND_ID:CXX,GNU>:--coverage>
      $<$<COMPILE_LANG_AND_ID:Fortran,GNU>:--coverage>
      $<$<COMPILE_LANG_AND_ID:C,Clang,AppleClang>:-fprofile-instr-generate>
      $<$<COMPILE_LANG_AND_ID:C,Clang,AppleClang>:-fcoverage-mapping>
      $<$<COMPILE_LANG_AND_ID:C,Clang,AppleClang>:-Wno-error=pass-failed>
      $<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>:-fprofile-instr-generate>
      $<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>:-fcoverage-mapping>
      $<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>:-Wno-error=pass-failed>
    )

    if(_sparse_target_type STREQUAL "EXECUTABLE" OR
       _sparse_target_type STREQUAL "SHARED_LIBRARY" OR
       _sparse_target_type STREQUAL "MODULE_LIBRARY")
      target_link_options(${target_name} PRIVATE
        $<$<LINK_LANG_AND_ID:C,GNU>:--coverage>
        $<$<LINK_LANG_AND_ID:CXX,GNU>:--coverage>
        $<$<LINK_LANG_AND_ID:Fortran,GNU>:--coverage>
        $<$<LINK_LANG_AND_ID:C,Clang,AppleClang>:-fprofile-instr-generate>
        $<$<LINK_LANG_AND_ID:CXX,Clang,AppleClang>:-fprofile-instr-generate>
      )
    endif()
  endif()
endfunction()

if(SPARSE_ENABLE_ARM64EC)
  if(WIN32)
    add_compile_options(--target=arm64ec-pc-windows-msvc)
    add_link_options(--target=arm64ec-pc-windows-msvc)
    message(STATUS "Arm64EC build ENABLED")
  else()
    message(FATAL_ERROR "Option SPARSE_ENABLE_ARM64EC unsupported on non-Windows system")
  endif()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(blas_fallback)
perflibs_setup_blas(
  PACKAGE_NAME "${_SPARSE_PACKAGE_NAME}"
  BLAS_TARGET "${_SPARSE_BLAS_TARGET}"
  CBLAS_TARGET "${_SPARSE_CBLAS_TARGET}"
  SPARSE_USE_ARMPL "${SPARSE_USE_ARMPL}"
  SPARSE_ENABLE_INT64 "${SPARSE_ENABLE_INT64}"
  ARMPL_LIB_BASENAME "${ARMPL_LIB_BASENAME}"
)

# -----------------------
# Tooling: Python + mako
# -----------------------
find_package(Python3 REQUIRED COMPONENTS Interpreter)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import mako"
  RESULT_VARIABLE _SPARSE_MAKO_RC
  OUTPUT_QUIET
  ERROR_QUIET
)
if(NOT _SPARSE_MAKO_RC EQUAL 0)
  message(FATAL_ERROR
    "Python package 'mako' is required to preprocess src/asm/*.S.mako.\n"
    "Install it with e.g. `python -m pip install mako` (in the environment used by CMake)."
  )
endif()

set(_SPARSE_MAKO_RENDER_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/src/asm/mako_gen.py")

# -----------------------
# Sources
# -----------------------
set(SPARSE_CPP_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/execute_api.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/matrix_api.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/vector_api.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/block_sparse_rows.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/compressed_sparse_columns.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/compressed_sparse_rows.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/convert.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/coordinate_list.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/dense.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/d_opt_sell_c_sigma.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/fortran_api.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/s_opt_sell_c_sigma.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/sell_c_sigma.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/add.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/object_helpers.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/matvec.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/norm.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/solve.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/vector_ops.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/export.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/inf_norm_csr.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/matrix_statistics.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/matmul.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/matmul_gustavson.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/spec.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/spec_sve.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/special.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/matrix_state.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/util.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/vector_state.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/vector_storage.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/statistics.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/supernodal.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/symbolic_matmul_compressed.cpp
)

# -----------------------
# C++ templates processed by Mako: src/*.cpp.mako -> generated .cpp
# -----------------------
set(SPARSE_CPP_TEMPLATES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/opt_sell_c_sigma.cpp.mako
)

set(_SPARSE_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(SPARSE_GENERATED_CPP_SOURCES "")
foreach(_tmpl IN LISTS SPARSE_CPP_TEMPLATES)
  file(RELATIVE_PATH _rel "${CMAKE_CURRENT_SOURCE_DIR}" "${_tmpl}")
  string(REPLACE ".mako" "" _rel_out "${_rel}")           # e.g. src/opt_sell_c_sigma.cpp
  set(_out "${_SPARSE_GENERATED_DIR}/${_rel_out}")         # e.g. <build>/generated/src/opt_sell_c_sigma.cpp
  get_filename_component(_out_dir "${_out}" DIRECTORY)
  file(MAKE_DIRECTORY "${_out_dir}")

  add_custom_command(
    OUTPUT "${_out}"
    COMMAND "${Python3_EXECUTABLE}" "${_SPARSE_MAKO_RENDER_SCRIPT}"
            "${_tmpl}" -o "${_out}"
            --target-os "${SPARSE_TARGET_OS}"
    DEPENDS "${_tmpl}" "${_SPARSE_MAKO_RENDER_SCRIPT}"
    COMMENT "Mako render (cpp): ${_rel}"
    VERBATIM
  )

  list(APPEND SPARSE_GENERATED_CPP_SOURCES "${_out}")
endforeach()

if(SPARSE_GENERATED_CPP_SOURCES)
  # add generated cpp files to the main cpp source list
  list(APPEND SPARSE_CPP_SOURCES ${SPARSE_GENERATED_CPP_SOURCES})

  add_custom_target(sparse_cpp_gen DEPENDS ${SPARSE_GENERATED_CPP_SOURCES})
endif()

# Filter out spec_sve.cpp when SVE disabled
if(NOT SPARSE_ENABLE_SVE)
  list(FILTER SPARSE_CPP_SOURCES EXCLUDE REGEX ".*/spec_sve\\.cpp$")
endif()

file(GLOB_RECURSE SPARSE_ASM_TEMPLATES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/asm/*.S.mako"
)
if(SPARSE_ENABLE_ARM64EC)
  set(SPARSE_ASM_TEMPLATES)
else()
  set(SPARSE_ASM_TEMPLATES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/asm/neon/d_spmv_scs_kernels.S.mako
    ${CMAKE_CURRENT_SOURCE_DIR}/src/asm/neon/s_spmv_scs_kernels.S.mako
    ${CMAKE_CURRENT_SOURCE_DIR}/src/asm/sve/d_spmv_scs_sve_kernels.S.mako
    ${CMAKE_CURRENT_SOURCE_DIR}/src/asm/sve/s_spmv_scs_sve_kernels.S.mako
  )
endif()

if(NOT SPARSE_ENABLE_SVE)
  list(FILTER SPARSE_ASM_TEMPLATES EXCLUDE REGEX ".*/src/asm/sve/.*\\.S\\.mako$")
endif()

set(SPARSE_GENERATED_ASM_SOURCES "")
foreach(_tmpl IN LISTS SPARSE_ASM_TEMPLATES)
  file(RELATIVE_PATH _rel "${CMAKE_CURRENT_SOURCE_DIR}" "${_tmpl}")
  string(REPLACE ".mako" "" _rel_out "${_rel}")
  set(_out "${_SPARSE_GENERATED_DIR}/${_rel_out}")
  get_filename_component(_out_dir "${_out}" DIRECTORY)
  file(MAKE_DIRECTORY "${_out_dir}")

  add_custom_command(
    OUTPUT "${_out}"
    COMMAND "${Python3_EXECUTABLE}" "${_SPARSE_MAKO_RENDER_SCRIPT}"
            "${_tmpl}" -o "${_out}"
            --target-os "${SPARSE_TARGET_OS}"
    DEPENDS "${_tmpl}" "${_SPARSE_MAKO_RENDER_SCRIPT}"
    COMMENT "Mako render: ${_rel}"
    VERBATIM
  )

  # If this is an SVE kernel, compile the generated .S with SVE enabled
  if(SPARSE_ENABLE_SVE)
    if(_rel MATCHES "^src/asm/sve/.*\\.S\\.mako$")
      set_source_files_properties("${_out}" PROPERTIES COMPILE_OPTIONS "-march=armv8-a+sve")
    endif()
  endif()

  list(APPEND SPARSE_GENERATED_ASM_SOURCES "${_out}")
endforeach()

add_custom_target(sparse_asm_gen DEPENDS ${SPARSE_GENERATED_ASM_SOURCES})

# -----------------------
# libmachine.a
# -----------------------
add_library(machine STATIC
  "${CMAKE_CURRENT_SOURCE_DIR}/machine/machine.cpp"
)
set_target_properties(machine PROPERTIES OUTPUT_NAME "plsparse_machine_${_SPARSE_ABI_SUFFIX}")
sparse_enable_coverage(machine)

target_compile_features(machine PUBLIC cxx_std_17)
add_library(machine::machine ALIAS machine)

# -----------------------
# libsparse objects (C++ + generated ASM)
# -----------------------
add_library(sparse_obj OBJECT
  ${SPARSE_CPP_SOURCES}
  ${SPARSE_GENERATED_ASM_SOURCES}
)
add_dependencies(sparse_obj sparse_asm_gen sparse_cpp_gen)
sparse_enable_coverage(sparse_obj)

# Compile this TU with SVE enabled
set_source_files_properties(
  "${CMAKE_CURRENT_SOURCE_DIR}/src/spec_sve.cpp"
  PROPERTIES
    COMPILE_OPTIONS "-march=armv8-a+sve"
)

# Apply -ffast-math to the CSR kernel translation unit (equivalent to the old Makefile's CENTRAL_CXXFLAGS+=-ffast-math for compressed_sparse_rows)
# Do this only for GNU, since with modern LLVM -ffast-math conflicts with -fcx-fortran-rules, which is set project-wide
set_source_files_properties(
  "${CMAKE_CURRENT_SOURCE_DIR}/src/compressed_sparse_rows.cpp"
  PROPERTIES
    COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU>:-ffast-math>>"
)

if(SPARSE_USE_ARMPL)
  target_compile_definitions(sparse_obj PRIVATE SPARSE_USE_ARMPL)
endif()

# On Windows, the public sparse header names ABI complex carrier types
# (_Fcomplex/_Dcomplex). Request the UCRT C complex declarations explicitly so
# those types are visible in C and C++ translation units that include it.
if(WIN32)
  target_compile_definitions(sparse_obj PRIVATE _CRT_USE_C_COMPLEX_H)
endif()

set_property(TARGET sparse_obj PROPERTY POSITION_INDEPENDENT_CODE ON)

set(_SPARSE_INSTALL_INCLUDEDIR
    "${CMAKE_INSTALL_INCLUDEDIR}/plsparse_${_SPARSE_ABI_SUFFIX}")

target_compile_features(sparse_obj PUBLIC cxx_std_17)
target_include_directories(sparse_obj
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${_SPARSE_INSTALL_INCLUDEDIR}>
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
# -----------------------
# Integer model: C/C++ gets -DINTEGER64=1 when ILP64 enabled
# -----------------------
if(SPARSE_ENABLE_INT64)
  target_compile_definitions(sparse_obj PRIVATE INTEGER64=1)
endif()

target_link_libraries(sparse_obj PRIVATE machine::machine)
target_link_libraries(sparse_obj PRIVATE "${_SPARSE_CBLAS_TARGET}")

# -----------------------
# Apply compile-only C/C++ defaults to specific targets (only when user didn't set CMAKE_CXX_FLAGS)
# Flags are checked for compiler support and applied only if supported.
# -----------------------
if("${CMAKE_CXX_FLAGS}" STREQUAL "")
  include(CheckCCompilerFlag)
  include(CheckCXXCompilerFlag)

  # split space-separated cache strings into lists
  string(REGEX REPLACE " +" ";" _SPARSE_COMMON_FLAGS_LIST "${DEFAULT_SPARSE_COMMON_FLAGS}")
  string(REGEX REPLACE " +" ";" _SPARSE_CXX_ONLY_FLAGS_LIST "${DEFAULT_SPARSE_CXX_ONLY_FLAGS}")

  # prepare accepted lists
  set(_SPARSE_ACCEPTED_C_FLAGS "")
  set(_SPARSE_ACCEPTED_CXX_FLAGS "")

  # check common flags for both C and C++ compilers, record support per-language
  foreach(_f IN LISTS _SPARSE_COMMON_FLAGS_LIST)
    # sanitize a variable suffix for CHECK_* result variables
    string(REGEX REPLACE "[^A-Za-z0-9]" "_" _san "${_f}")
    set(_var_c "HAVE_C_FLAG_${_san}")
    set(_var_cxx "HAVE_CXX_FLAG_${_san}")

    # check C flag
    CHECK_C_COMPILER_FLAG("${_f}" ${_var_c})
    if(${${_var_c}})
      list(APPEND _SPARSE_ACCEPTED_C_FLAGS "${_f}")
    endif()

    # check CXX flag
    CHECK_CXX_COMPILER_FLAG("${_f}" ${_var_cxx})
    if(${${_var_cxx}})
      list(APPEND _SPARSE_ACCEPTED_CXX_FLAGS "${_f}")
    endif()
  endforeach()

  # check C++-only flags against the C++ compiler and append to accepted CXX flags
  foreach(_f IN LISTS _SPARSE_CXX_ONLY_FLAGS_LIST)
    string(REGEX REPLACE "[^A-Za-z0-9]" "_" _san "${_f}")
    set(_var_cxx "HAVE_CXX_FLAG_${_san}")
    CHECK_CXX_COMPILER_FLAG("${_f}" ${_var_cxx})
    if(${${_var_cxx}})
      list(APPEND _SPARSE_ACCEPTED_CXX_FLAGS "${_f}")
    endif()
  endforeach()

  # Now apply flags to targets using generator expressions so flags are only used for the correct language.
  # Use $<COMPILE_LANGUAGE:C> and $<COMPILE_LANGUAGE:CXX> and pass each list as separate args (no quoting).
  if(_SPARSE_ACCEPTED_C_FLAGS)
    # add as a language-specific generator expression
    target_compile_options(machine PRIVATE $<$<COMPILE_LANGUAGE:C>:${_SPARSE_ACCEPTED_C_FLAGS}>)
    target_compile_options(sparse_obj PRIVATE $<$<COMPILE_LANGUAGE:C>:${_SPARSE_ACCEPTED_C_FLAGS}>)
  endif()

  if(_SPARSE_ACCEPTED_CXX_FLAGS)
    target_compile_options(machine PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${_SPARSE_ACCEPTED_CXX_FLAGS}>)
    target_compile_options(sparse_obj PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${_SPARSE_ACCEPTED_CXX_FLAGS}>)
  endif()
endif()

# -----------------------
# Compute library filename suffix based on requested options
# -----------------------
if(SPARSE_ENABLE_OPENMP)
  # append _mp
  string(APPEND SPARSE_LIB_BASENAME "_mp")
endif()

if(SPARSE_ENABLE_ASAN)
  # _asan should come after _mp if both are used
  string(APPEND SPARSE_LIB_BASENAME "_asan")
endif()

# -----------------------
# libsparse (shared/static)
# -----------------------
if(SPARSE_BUILD_SHARED)
  add_library(sparse_shared SHARED
    $<TARGET_OBJECTS:sparse_obj>
  )

  # DLL name: <basename><suffix>.dll
  set_target_properties(sparse_shared PROPERTIES OUTPUT_NAME "${SPARSE_LIB_BASENAME}")
  sparse_enable_coverage(sparse_shared)

  if(WIN32)

    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    # Paths (api is at project root; script lives in api/)
    set(API_DIR              "${CMAKE_SOURCE_DIR}/api")
    set(PUBLIC_SYMBOLS_JSON  "${API_DIR}/public_symbols.json")
    set(GEN_DEF_SCRIPT       "${API_DIR}/generate_def_file.py")

    # Put .def in the build tree (current binary dir is in the build dir)
    set(DEF_FILE "${CMAKE_CURRENT_BINARY_DIR}/${SPARSE_LIB_BASENAME}.def")

    add_custom_command(
      OUTPUT "${DEF_FILE}"
      COMMAND "${Python3_EXECUTABLE}" "${GEN_DEF_SCRIPT}"
              "${PUBLIC_SYMBOLS_JSON}"
              --output_file "${DEF_FILE}"
              --lib_name "${SPARSE_LIB_BASENAME}"
      DEPENDS
              "${GEN_DEF_SCRIPT}"
              "${PUBLIC_SYMBOLS_JSON}"
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
      VERBATIM
      COMMENT "Generating ${SPARSE_LIB_BASENAME}.def"
    )

    add_custom_target(gen_def_${SPARSE_LIB_BASENAME} DEPENDS "${DEF_FILE}")

    # Ensure the .def is generated before the DLL links
    add_dependencies(sparse_shared gen_def_${SPARSE_LIB_BASENAME})

    target_sources(sparse_shared PRIVATE ${DEF_FILE})

    # Ensure relink if the .def changes
    set_property(TARGET sparse_shared APPEND PROPERTY LINK_DEPENDS "${DEF_FILE}")

    # Import lib name: <basename><suffix>.dll.lib
    set_target_properties(sparse_shared PROPERTIES
      ARCHIVE_OUTPUT_NAME "${SPARSE_LIB_BASENAME}.dll"
    )
  endif()

  target_link_libraries(sparse_shared PRIVATE machine::machine "${_SPARSE_BLAS_TARGET}" "${_SPARSE_CBLAS_TARGET}")

  target_include_directories(sparse_shared PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${_SPARSE_INSTALL_INCLUDEDIR}>
  )
  add_library(sparse::shared ALIAS sparse_shared)
endif()

if(SPARSE_BUILD_STATIC)
  add_library(sparse_static STATIC
    $<TARGET_OBJECTS:sparse_obj>
  )

  # Static lib name: <basename><suffix>.lib
  set_target_properties(sparse_static PROPERTIES OUTPUT_NAME "${SPARSE_LIB_BASENAME}")
  sparse_enable_coverage(sparse_static)

  target_link_libraries(sparse_static PRIVATE machine::machine "${_SPARSE_BLAS_TARGET}" "${_SPARSE_CBLAS_TARGET}")

  target_include_directories(sparse_static PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${_SPARSE_INSTALL_INCLUDEDIR}>
  )
  add_library(sparse::static ALIAS sparse_static)
endif()

if(SPARSE_ENABLE_INT64)
  if(TARGET sparse_shared)
    target_compile_definitions(sparse_shared PUBLIC INTEGER64=1)
  endif()
  if(TARGET sparse_static)
    target_compile_definitions(sparse_static PUBLIC INTEGER64=1)
  endif()
endif()

# -----------------------
# Optional Fortran interface
# -----------------------
if(SPARSE_ENABLE_FORTRAN)
  enable_language(Fortran)

  # -----------------------
  # Integer model: Fortran default integer size flags (ILP64)
  # -----------------------
  set(_SPARSE_FORTRAN_INT64_FLAG "")
  if(SPARSE_ENABLE_INT64)
    # CMake Fortran compiler IDs to consider:
    # - NVHPC: "NVHPC" (nvfortran)
    # - GNU: "GNU" (gfortran)
    # - LLVMFlang: "LLVMFlang" (new flang)
    if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC")
      set(_SPARSE_FORTRAN_INT64_FLAG "-i8")
    else()
      set(_SPARSE_FORTRAN_INT64_FLAG "-fdefault-integer-8")
    endif()
  endif()

  set(SPARSE_FORTRAN_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/include/perflibs_kinds.f90"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/perflibs_sparse_params.f90"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/perflibs_sparse.f90"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/export_ftn.f90"
  )

  if(TARGET sparse_shared)
    target_sources(sparse_shared PRIVATE ${SPARSE_FORTRAN_SOURCES})

    set_target_properties(sparse_shared PROPERTIES
      Fortran_MODULE_DIRECTORY "${CMAKE_BINARY_DIR}/fortran_mod/shared"
    )

    if(APPLE)
      apply_macos_deployment_target(sparse_shared)
    endif()
  endif()

  if(TARGET sparse_static)
    target_sources(sparse_static PRIVATE ${SPARSE_FORTRAN_SOURCES})

    set_target_properties(sparse_static PROPERTIES
      Fortran_MODULE_DIRECTORY "${CMAKE_BINARY_DIR}/fortran_mod/static"
    )

    if(APPLE)
      apply_macos_deployment_target(sparse_static)
    endif()
  endif()

  # Apply default-integer-8 only to Fortran compilation of libsparse
  if(SPARSE_ENABLE_INT64 AND NOT _SPARSE_FORTRAN_INT64_FLAG STREQUAL "")
    if(TARGET sparse_shared)
      target_compile_options(sparse_shared PUBLIC $<$<COMPILE_LANGUAGE:Fortran>:${_SPARSE_FORTRAN_INT64_FLAG}>)
    endif()
    if(TARGET sparse_static)
      target_compile_options(sparse_static PUBLIC $<$<COMPILE_LANGUAGE:Fortran>:${_SPARSE_FORTRAN_INT64_FLAG}>)
    endif()
  endif()

endif()

# -----------------------
# Tests
# -----------------------
if(SPARSE_ENABLE_TESTS)
  enable_testing()

  set(SPARSE_C_TEST_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/test/bsr_kernel_paths.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/coo_spelmm_paths.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/coo_spmv_paths.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/csr_spmv_omp_paths.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/csc_kernel_paths.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/dense_kernel_paths.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/dense_update_paths.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/format_edge_ops.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/matmul_dispatch_paths_c.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/matmul_dispatch_paths_d.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/matmul_dispatch_paths_s.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/matmul_dispatch_paths_z.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/matmul_variants_c.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/matmul_variants_d.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/matmul_variants_s.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/matmul_variants_z.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/matrix_object_ops.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/matrix_kernel_ops.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/solve_dispatch_paths.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/csr_spsv_omp_paths.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/solve_variants.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/solve_variants_z.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/spmv_basic_d.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/spmm_strategy_z.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/strategy_paths_s.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/strategy_paths_d.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/spvec_ops.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/super_lower.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/super_spsm_lower.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/super_spsm_upper.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/super_upper.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/vector_type_smoke.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/spsm_basic_c.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/spsm_basic_d.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/spsm_basic_s.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/spsm_basic_z.c"
  )

  set(SPARSE_C_TEST_TARGETS "")
  set(SPARSE_TEST_MATH_LIBS "")
  if(UNIX AND NOT APPLE)
    list(APPEND SPARSE_TEST_MATH_LIBS m)
  endif()

  foreach(_src IN LISTS SPARSE_C_TEST_SOURCES)
    get_filename_component(_base "${_src}" NAME_WE)
    set(_tgt "${_base}_test")

    add_executable(${_tgt} "${_src}")
    sparse_enable_coverage(${_tgt})
    target_include_directories(${_tgt} PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )

    # Safe to link explicitly (helps static linkers)
    target_link_libraries(${_tgt} PRIVATE "${_SPARSE_BLAS_TARGET}" "${_SPARSE_CBLAS_TARGET}")
    if(SPARSE_TEST_MATH_LIBS)
      target_link_libraries(${_tgt} PRIVATE ${SPARSE_TEST_MATH_LIBS})
    endif()

    if(TARGET sparse_shared)
      target_link_libraries(${_tgt} PRIVATE sparse_shared)
    elseif(TARGET sparse_static)
      target_link_libraries(${_tgt} PRIVATE sparse_static)
    endif()

    if(SPARSE_ENABLE_INT64)
      target_compile_definitions(${_tgt} PRIVATE INTEGER64=1)
    endif()

    # If user didn't set CMAKE_CXX_FLAGS, give tests the same compile-only defaults
    if("${CMAKE_CXX_FLAGS}" STREQUAL "")
      if(DEFINED _SPARSE_ACCEPTED_C_FLAGS)
        target_compile_options(${_tgt} PRIVATE $<$<COMPILE_LANGUAGE:C>:${_SPARSE_ACCEPTED_C_FLAGS}>)
      endif()
      if(DEFINED _SPARSE_ACCEPTED_CXX_FLAGS)
        target_compile_options(${_tgt} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${_SPARSE_ACCEPTED_CXX_FLAGS}>)
      endif()
    endif()

    add_test(NAME ${_base} COMMAND ${_tgt})
    list(APPEND SPARSE_C_TEST_TARGETS ${_tgt})
  endforeach()

  add_custom_target(build_tests DEPENDS ${SPARSE_C_TEST_TARGETS})
endif()

# -----------------------
# Install
# -----------------------
set(_SPARSE_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/${_SPARSE_PACKAGE_NAME}")

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/perflibs_sparseConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${_SPARSE_PACKAGE_NAME}Config.cmake"
  INSTALL_DESTINATION "${_SPARSE_INSTALL_CMAKEDIR}"
  NO_SET_AND_CHECK_MACRO
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/${_SPARSE_PACKAGE_NAME}ConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY ExactVersion
)

install(FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/include/perflibs_sparse.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/perflibs_status.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/api/public_symbols.json"
  DESTINATION "${_SPARSE_INSTALL_INCLUDEDIR}"
)

install(TARGETS
  machine
  EXPORT "${_SPARSE_EXPORT_NAME}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  INCLUDES DESTINATION "${_SPARSE_INSTALL_INCLUDEDIR}"
)

if(TARGET sparse_shared)
  install(TARGETS sparse_shared
    EXPORT "${_SPARSE_EXPORT_NAME}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    INCLUDES DESTINATION "${_SPARSE_INSTALL_INCLUDEDIR}"
  )
endif()

if(TARGET sparse_static)
  install(TARGETS sparse_static
    EXPORT "${_SPARSE_EXPORT_NAME}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    INCLUDES DESTINATION "${_SPARSE_INSTALL_INCLUDEDIR}"
  )
endif()

if(SPARSE_ENABLE_FORTRAN)
  install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/include/perflibs_kinds.f90"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/perflibs_sparse_params.f90"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/perflibs_sparse.f90"
    DESTINATION "${_SPARSE_INSTALL_INCLUDEDIR}"
  )

  if(TARGET sparse_shared)
    set(_SPARSE_INSTALL_MODDIR "${CMAKE_CURRENT_BINARY_DIR}/fortran_mod/shared")
  else()
    set(_SPARSE_INSTALL_MODDIR "${CMAKE_CURRENT_BINARY_DIR}/fortran_mod/static")
  endif()

  install(DIRECTORY "${_SPARSE_INSTALL_MODDIR}/"
    DESTINATION "${_SPARSE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING
      PATTERN "*.mod"
      PATTERN "*.MOD"
  )

  if(SPARSE_ENABLE_TESTS)
    add_subdirectory(fortran_tests)
  endif()
endif()

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/${_SPARSE_PACKAGE_NAME}Config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/${_SPARSE_PACKAGE_NAME}ConfigVersion.cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/blas_fallback.cmake"
  DESTINATION "${_SPARSE_INSTALL_CMAKEDIR}"
)

install(EXPORT "${_SPARSE_EXPORT_NAME}"
  NAMESPACE "${_SPARSE_PACKAGE_NAMESPACE}"
  DESTINATION "${_SPARSE_INSTALL_CMAKEDIR}"
)
