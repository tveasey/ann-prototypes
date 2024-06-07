#pragma once

#include "../common/evaluation.h"

#include <array>
#include <cstddef>
#include <string>
#include <vector>

// PQ stats
struct PqStats {
    // Test stats
    std::string tag;
    std::string metric;
    std::size_t numQueries;
    std::size_t numDocs;
    std::size_t dim;
    std::size_t numBooks;
    std::size_t k;
    // Brute force stats
    double bfQPS;
    // PQ build stats
    double pqCodeBookBuildTime;
    double pqVectorCompressionRatio;
    double pqCompressionRatio;
    // PQ merge stats
    double pqMergeTime;
    // PQ query stats
    bool normalize;
    std::vector<double> pqQPS;
    std::vector<recalls_t> pqRecalls;
};

void writePqStats(const PqStats& stats);
