#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

struct PQStats {
    static constexpr std::size_t MIN_RECALL{0};
    static constexpr std::size_t MAX_RECALL{1};
    static constexpr std::size_t AVG_RECALL{2};
    static constexpr std::array<std::size_t, 6> EXPANSIONS{1, 2, 4, 6, 8, 10};
    using recalls_t = std::array<double, AVG_RECALL + 1>;

    // Test stats
    std::string tag;
    std::size_t numQueries;
    std::size_t numDocs;
    std::size_t k;
    // Brute force stats
    double bfQPS;
    // PQ build stats
    double pqCodeBookBuildTime;
    double pqCompressionRatio;
    double pqMse;
    // PQ query stats
    bool normalise;
    std::vector<double> pqQPS;
    std::vector<recalls_t> pqRecalls;
};

double computeRecall(const std::vector<std::size_t>& nnTrue,
                     std::vector<std::size_t> nnApprox);

double computeCompressionRatio(std::size_t dim);

PQStats::recalls_t computeRecalls(const std::vector<std::vector<std::size_t>>& nnExact,
                                  const std::vector<std::vector<std::size_t>>& nnPQ);
