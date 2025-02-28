#pragma once

#include "../common/bigvector.h"
#include "../common/types.h"

#include <vector>

// Run a benchmark of the PQ brute force search and write the build and search
// stats using writeStats.
void runSoarIVFBenchmark(Metric metric,
                         const BigVector& docs,
                         std::vector<float>& queries,
                         float lambda,
                         std::size_t docsPerCluster,
                         std::size_t k,
                         std::size_t numProbes);
