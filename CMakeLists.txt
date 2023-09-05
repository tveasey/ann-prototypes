cmake_minimum_required(VERSION 3.14)

project(pq_experiments)

add_executable(run_pq main.cc src/centroid.cc src/io.cc src/metrics.cc src/pq.cc tests/pq_tests.cc)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -march=native -O3")