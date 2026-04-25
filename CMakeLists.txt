cmake_minimum_required(VERSION 3.28)
project(aoftree)

# --- Global Settings ---
set(CMAKE_CXX_STANDARD 26)

# 1. FIND EIGEN
# This tells the compiler where to look for <Eigen/Dense>
include_directories("path/to/eigen")

#Library for hashing
find_package(OpenSSL REQUIRED)

# 1. Find HiGHS
find_path(HIGHS_INCLUDE_DIR Highs.h
        PATHS highs/head/loc
)
find_library(HIGHS_LIBRARY highs
        PATHS highs/library/loc
)

if (NOT HIGHS_INCLUDE_DIR OR NOT HIGHS_LIBRARY)
    message(WARNING "HiGHS not found! Targets 'lp_solver' and 'highs_test' will not be built.")
else()
    # --- Target 1: The Main LP Solver Application ---
    add_executable(ITreeSimplex
            ItreeSimplexMain.cpp
            # Helper headers (LPSolver.h, LPTree.h, etc are included implicitly)
            CompactIO.h
    )
    target_include_directories(ITreeSimplex PRIVATE ${HIGHS_INCLUDE_DIR})
    target_link_libraries(ITreeSimplex PRIVATE ${HIGHS_LIBRARY})
    target_link_libraries(ITreeSimplex PRIVATE OpenSSL::SSL OpenSSL::Crypto)

endif()

# --- Target 1: Test data generator for FS-tree ---
add_executable(FunctionGenerator
        FunctionPairMain.cpp
        FunctionPairGenerator.h
        FunctionPairGenerator.cpp
 )

 # --- Target M1: Merkle FsTree ---
 add_executable(Pruned_AOF
        MFsTree.cpp
        PolytopeStructs.h
        PolytopeOps.h
        MFsTree.h
        CompactIO.h
 )
 target_link_libraries(Pruned_AOF PRIVATE OpenSSL::SSL OpenSSL::Crypto)

 # --- Target M2: Merkle FsTree Full---
 add_executable(AOF
        MFsTreeFull.cpp
        PolytopeStructs.h
        PolytopeOps.h
        MFsTreeFull.h
        CompactIO.h
 )
 target_link_libraries(AOF PRIVATE OpenSSL::SSL OpenSSL::Crypto)

 add_executable(format 
        Format.cpp 
 )

 # --- Target M3: Verifiable Merkle FsTree---
 add_executable(Verified_AOF
        VMFsTree.cpp
        PolytopeStructs.h
        PolytopeOps.h
        VMFsTree.h
        CompactIO.h
 )
 target_link_libraries(Verified_AOF PRIVATE OpenSSL::SSL OpenSSL::Crypto)