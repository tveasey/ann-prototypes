#pragma once

#include "../common/evaluation.h"

#include <array>
#include <cstddef>
#include <string>
#include <vector>

// PQ stats
struct PQStats {
    // Test stats
    std::string tag;
    std::string metric;
    std::size_t numQueries;
    std::size_t numDocs;
    std::size_t dim;
    std::size_t k;
    // Brute force stats
    double bfQPS;
    // PQ build stats
    double pqCodeBookBuildTime;
    double pqCompressionRatio;
    double pqMse;
    // PQ query stats
    bool scann;
    bool normalise;
    std::vector<double> pqQPS;
    std::vector<recalls_t> pqRecalls;
};

void writePQStats(const PQStats& stats);

// Compute the compression ratio of the PQ code.
double computeCompressionRatio(std::size_t dim);
