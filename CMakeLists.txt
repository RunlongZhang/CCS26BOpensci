cmake_minimum_required(VERSION 3.28)
project(aof_tree_artifact LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# All dependencies resolve through standard find_package calls.  When this
# project is configured with the vcpkg toolchain file, the vcpkg.json manifest
# in this directory makes vcpkg download and build OpenSSL, HiGHS, and Eigen
# automatically at configure time -- there are no paths to edit in this file.

find_package(OpenSSL REQUIRED)

find_package(Eigen3 CONFIG QUIET)
if(NOT TARGET Eigen3::Eigen)
    # Fallback for a plain Eigen checkout without vcpkg:
    #   cmake ... -DEIGEN_INCLUDE_DIR=<path to an Eigen source checkout>
    if(NOT DEFINED EIGEN_INCLUDE_DIR)
        message(FATAL_ERROR
            "Eigen not found. Configure with the vcpkg toolchain (recommended) "
            "or pass -DEIGEN_INCLUDE_DIR=<path to an Eigen checkout>.")
    endif()
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${EIGEN_INCLUDE_DIR}")
endif()

find_package(highs CONFIG QUIET)
if(TARGET highs::highs)
    set(HIGHS_TARGET highs::highs)
else()
    # Fallback: locate a HiGHS install discoverable via CMAKE_PREFIX_PATH.
    find_path(HIGHS_INCLUDE_DIR Highs.h PATH_SUFFIXES highs)
    find_library(HIGHS_LIBRARY highs)
    if(NOT HIGHS_INCLUDE_DIR OR NOT HIGHS_LIBRARY)
        message(FATAL_ERROR
            "HiGHS not found. Configure with the vcpkg toolchain (recommended) "
            "or make a HiGHS install discoverable via CMAKE_PREFIX_PATH.")
    endif()
    add_library(highs_imported INTERFACE IMPORTED)
    set_target_properties(highs_imported PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${HIGHS_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${HIGHS_LIBRARY}")
    set(HIGHS_TARGET highs_imported)
endif()

# The sources include <Highs.h> directly; some HiGHS installs export only the
# parent include directory, so add the highs/ subdirectory when it exists.
if(TARGET highs::highs)
    get_target_property(_highs_incs highs::highs INTERFACE_INCLUDE_DIRECTORIES)
    if(_highs_incs)
        foreach(_inc IN LISTS _highs_incs)
            if(EXISTS "${_inc}/highs/Highs.h")
                target_include_directories(highs::highs INTERFACE "${_inc}/highs")
            endif()
        endforeach()
    endif()
endif()

# --- Dataset generator (seeded, duplicate-free) ---
add_executable(FunctionGenerator
    src/FunctionPairMain.cpp
    src/FunctionPairGenerator.cpp)

# --- AOF-tree construction / ADS / authenticated-query benchmark ---
add_executable(AofBench
    src/AofBenchMain.cpp
    src/FunctionPairGenerator.cpp)
target_link_libraries(AofBench PRIVATE OpenSSL::Crypto Eigen3::Eigen)

# --- Verifiable-construction certificate benchmark (needs HiGHS) ---
add_executable(CertBench
    src/CertBenchMain.cpp
    src/FunctionPairGenerator.cpp)
target_link_libraries(CertBench PRIVATE OpenSSL::Crypto Eigen3::Eigen ${HIGHS_TARGET})

# --- I-tree (Simplex) baseline (needs HiGHS; Windows-only sources) ---
add_executable(ITreeSimplex
    src/ITreeSimplexMain.cpp)
target_link_libraries(ITreeSimplex PRIVATE OpenSSL::Crypto Eigen3::Eigen ${HIGHS_TARGET})
