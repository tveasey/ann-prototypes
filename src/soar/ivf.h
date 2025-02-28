#pragma once

#include "../common/bigvector.h"
#include "../common/types.h"

#include <cstddef>
#include <vector>

class SoarIVFIndex {
public:
    SoarIVFIndex(Metric metric,
                 float lambda,
                 std::size_t dim,
                 std::size_t numClusters,
                 std::size_t numIterations = 5);

    void build(const BigVector& docs);

    std::pair<std::vector<std::size_t>, std::size_t>
    search(const float* query, std::size_t k, std::size_t numProbes) const;

private:
    void maybeNormalizeCentres();

private:
    Metric metric_;
    float lambda_;
    std::size_t dim_;
    std::size_t numClusters_;
    std::size_t numIterations_;
    std::vector<float> centres_;
    std::vector<std::vector<std::size_t>> clustersDocs_;
    std::vector<float> docs_;
};