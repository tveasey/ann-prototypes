cmake_minimum_required(VERSION 3.14)

project("Quantisation Experiments" VERSION 0.1.0 LANGUAGES CXX)

add_library(common SHARED
    src/common/bigvector.cc
    src/common/bruteforce.cc
    src/common/evaluation.cc
    src/common/io.cc
    src/common/utils.cc
)

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
)
target_link_libraries(scalar common)

add_executable(run_tests
    tests/main.cc
    tests/test_common.cc
    tests/test_observer.cc
    tests/test_pq.cc
    tests/test_scalar.cc
)
target_link_libraries(run_tests common pq scalar)

add_executable(run_benchmark main.cc)
target_link_libraries(run_benchmark common pq scalar)

find_package(Boost 1.83.0 REQUIRED COMPONENTS iostreams program_options unit_test_framework)
include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(run_tests ${Boost_LIBRARIES})
target_link_libraries(common ${Boost_LIBRARIES})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp-simd -std=c++17 -march=native -O3")