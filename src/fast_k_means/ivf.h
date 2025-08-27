#pragma once

#include "osq/utils.h"
#include "../common/types.h"

#include <cstddef>
#include <queue>
#include <unordered_set>
#include <vector>

struct QuantizationState {
    float bias;
    std::vector<float> localBias;
    std::vector<int> L1Norms;
    std::vector<Interval> limits;
    std::vector<int> quantizedVectors;
};


class CQuantizedIvfIndex {
public:
    using Point = std::vector<float>;
    using Dataset = std::vector<float>;
    using Centers = std::vector<std::vector<float>>;
    using Clusters = std::vector<QuantizationState>;
    using Assignments = std::vector<std::vector<std::size_t>>;
    using Dimensions = std::vector<std::size_t>;
    using BlockMatrix = std::vector<std::vector<float>>;
    using PermutationMatrix = std::vector<std::vector<std::size_t>>;
    using Topk = std::priority_queue<std::pair<float, std::size_t>>;

public:
    CQuantizedIvfIndex(Metric metric,
                       std::size_t dim,
                       const Dataset& corpus,
                       std::size_t target,
                       std::size_t bits,
                       std::size_t blockDim = 64);

    std::size_t numClusters() const { return clusters_.size(); }

    std::pair<std::unordered_set<std::size_t>, std::size_t>
    search(std::size_t probes,
           std::size_t k,
           std::size_t rerank,
           const Point& query,
           const Dataset& corpus,
           bool useQuantization = true) const;

private:
    void searchCluster(std::size_t cluster,
                       std::size_t k,
                       const Point& query,
                       Topk& topk,
                       std::unordered_set<std::size_t> &uniques) const;
    void buildIndex(const Dataset& corpus, std::size_t target, std::size_t bits);
    float distance(const float* x, const float* y) const;
    void updateTopk(std::size_t k,
                    float d,
                    std::size_t i,
                    Topk& topk,
                    std::unordered_set<std::size_t>* uniques = nullptr) const;

private:
    Metric metric_;
    std::size_t dim_;
    std::size_t bits_;
    PermutationMatrix permutationMatrix_;
    Dimensions dimBlocks_;
    BlockMatrix blocks_;
    Centers centers_;
    Clusters clusters_;
    Assignments assignments_;
};