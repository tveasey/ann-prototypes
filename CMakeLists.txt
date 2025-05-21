cmake_minimum_required(VERSION 3.14)

project("Quantisation Experiments" VERSION 0.1.0 LANGUAGES CXX)

add_library(common SHARED
    src/common/bigvector.cc
    src/common/bruteforce.cc
    src/common/evaluation.cc
    src/common/io.cc
    src/common/utils.cc
)

add_library(fast_k_means SHARED
    src/fast_k_means/osq/limits_optimization.cc
    src/fast_k_means/osq/preconditioner.cc
    src/fast_k_means/osq/utils.cc
    src/fast_k_means/baseline.cc
    src/fast_k_means/common.cc
    src/fast_k_means/hamerly.cc
    src/fast_k_means/hierarchical.cc
    src/fast_k_means/ivf.cc
    src/fast_k_means/local.cc
)
target_link_libraries(fast_k_means common)

add_library(pq SHARED
    src/pq/benchmark.cc
    src/pq/clustering.cc
    src/pq/codebooks.cc
    src/pq/index.cc
    src/pq/stats.cc
    src/pq/subspace.cc
    src/pq/utils.cc
)
target_link_libraries(pq common)

add_library(scalar SHARED
    src/scalar/scalar.cc
    src/scalar/utils.cc
)
target_link_libraries(scalar common)

add_library(soar SHARED
    src/soar/benchmark.cc
    src/soar/ivf.cc
)
target_link_libraries(soar common)

add_executable(run_kmeans_benchmark src/fast_k_means/main.cc)
target_link_libraries(run_kmeans_benchmark common fast_k_means)

add_executable(run_tests
    tests/main.cc
    tests/test_common.cc
    tests/test_observer.cc
    tests/test_pq.cc
    tests/test_scalar.cc
)
target_link_libraries(run_tests common pq scalar)

add_executable(run_benchmark main.cc)
target_link_libraries(run_benchmark common pq scalar soar)

find_package(Boost 1.83.0 REQUIRED COMPONENTS iostreams program_options unit_test_framework)
include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(run_tests ${Boost_LIBRARIES})
target_link_libraries(common ${Boost_LIBRARIES})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp-simd -std=c++17 -march=native -O3")