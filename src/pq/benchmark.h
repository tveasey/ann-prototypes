#pragma once

#include "stats.h"
#include "../common/bigvector.h"
#include "../common/types.h"

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

// Run a benchmark of the PQ brute force search and write the build and search
// stats using writeStats.
void runPqBenchmark(const std::string& tag,
                    Metric metric,
                    float distanceThreshold,
                    std::size_t docsPerCoarseCluster,
                    std::size_t numBooks,
                    std::size_t k,
                    const BigVector& docs,
                    std::vector<float>& queries,
                    const std::function<void(const PqStats&)>& writeStats);


// Run a benchmark of the PQ merge followed by brute force search and write
// the build, merge and search stats using writeStats.
void runPqMergeBenchmark(const std::string& tag,
                         Metric metric,
                         float distanceThreshold,
                         std::size_t docsPerCoarseCluster,
                         std::size_t numBooks,
                         std::size_t k,
                         const BigVector& docs1,
                         const BigVector& docs2,
                         std::vector<float>& queries,
                         const std::function<void(const PqStats&)>& writeStats);