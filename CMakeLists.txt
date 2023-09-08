cmake_minimum_required(VERSION 3.14)

project(quantisation_experiments)

add_executable(run_quantisation main.cc src/bruteforce.cc src/centroid.cc src/io.cc src/metrics.cc src/pq.cc src/scalar.cc src/utils.cc tests/tests.cc)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -march=native -O3")