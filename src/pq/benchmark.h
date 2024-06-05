#pragma once

#include "stats.h"
#include "../common/bigvector.h"
#include "../common/types.h"

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

// Run a benchmark of the PQ brute force search and write the search
// stats using writeStats.
void runPQBenchmark(const std::string& tag,
                    Metric metric,
                    float distanceThreshold,
                    std::size_t docsPerCoarseCluster,
                    std::size_t numBooks,
                    std::size_t k,
                    const BigVector& docs,
                    std::vector<float>& queries,
                    const std::function<void(const PQStats&)>& writeStats);
