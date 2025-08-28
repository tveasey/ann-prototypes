#include "ivf.h"

#include "osq/limits_optimization.h"
#include "osq/preconditioner.h"
#include "osq/utils.h"
#include "hierarchical.h"
#include "../common/utils.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <unordered_set>

namespace {

class CEarlyExit {
public:
    void add(float d) {
        moments_.add(d);
    }

    bool exit(std::size_t remaining, float distanceToK) const {
        if (moments_.n() < 12) {
            return false;
        }
        float mean{static_cast<float>(moments_.mean())};
        float var{static_cast<float>(moments_.var())};
        if (var == 0) {
            return mean <= distanceToK;
        }
        float z{(distanceToK - mean) / std::sqrtf(2.0F * var)};
        return 0.5F * (1.0F + std::erff(z)) * remaining < 0.1;
    }

private:
    OnlineMeanAndVariance moments_;
};

std::vector<float> dotAll(const std::vector<float>& x, const std::vector<float>& y) {
    std::size_t dim{x.size()};
    std::vector<float> result(y.size() / dim);
    for (std::size_t i = 0, id = 0; i < result.size(); ++i, id += dim) {
        result[i] = dot(dim, &x[0], &y[id]);
    }
    return result;
}

QuantizationState quantize(Metric metric,
                           std::size_t dim,
                           const std::vector<float>& mean,
                           const std::vector<float>& vectors,
                           std::size_t bits) {

    std::size_t n(vectors.size() / dim);
    auto centeredVectors = vectors;
    centeredVectors = center(dim, std::move(centeredVectors), mean);

    auto limits = optimizeQuantizationLimits(dim, centeredVectors, bits, 0.1F);
    auto quantizedVectors = quantize(dim, centeredVectors, limits, bits);
    auto vectorsL1Norms = l1Norms(dim, quantizedVectors);

    float bias{0.0F};
    std::vector<float> localBias(n);
    switch (metric) {
    case Dot:
    case Cosine:
        bias = dotAllowAlias(dim, &mean[0], &mean[0]);
        localBias = dotAll(mean, vectors);
        break;
    case Euclidean:
        localBias = l2Norms(dim, centeredVectors);
        break;
    }

    return {bias,
            std::move(localBias),
            std::move(vectorsL1Norms),
            std::move(limits),
            std::move(quantizedVectors)};   
}
}

CQuantizedIvfIndex::CQuantizedIvfIndex(Metric metric,
                                       std::size_t dim,
                                       const Dataset& corpus,
                                       std::size_t target,
                                       std::size_t bits,
                                       std::size_t blockDim) :
        metric_{metric}, dim_{dim}, bits_{bits} {

    time([&] {
        std::tie(blocks_, dimBlocks_) = randomOrthogonal(dim, blockDim);
        permutationMatrix_ = permutationMatrix(dim, dimBlocks_, corpus);
    }, "Compute Preconditioner");

    buildIndex(corpus, target, bits);
}

std::pair<std::unordered_set<std::size_t>, std::size_t>
CQuantizedIvfIndex::search(std::size_t probes,
                           std::size_t k,
                           std::size_t rerank,
                           const Point& query,
                           const Dataset& corpus,
                           bool useQuantization) const {

    Point transformedQuery(query);
    applyTransform(dim_, permutationMatrix_, blocks_, dimBlocks_, transformedQuery);

    // Closest centers to the query.
    Topk closest;
    for (std::size_t i = 0; i < centers_.size(); ++i) {
        updateTopk(probes, distance(&centers_[i][0], &transformedQuery[0]), i, closest);
    }

    if (!useQuantization) {
        // Search using true distances and no reranking.
        Topk topk;
        std::unordered_set<std::size_t> uniqueTopk;
        std::size_t comparisons{0};
        while (!closest.empty()) {
            std::size_t cluster{closest.top().second};
            comparisons += assignments_[cluster].size();
            for (auto i : assignments_[cluster]) {
                updateTopk(k, distance(&query[0], &corpus[i * dim_]), i, topk, &uniqueTopk);
            }
            closest.pop();
        }

        return {uniqueTopk, comparisons};
    }

    // Top-"rerank * k" candidates.
    Topk candidates;
    std::unordered_set<std::size_t> uniqueCandidates;
    std::size_t comparisons{0};
    while (!closest.empty()) {
        std::size_t cluster{closest.top().second};
        searchCluster(cluster, k * rerank, transformedQuery, candidates, uniqueCandidates);
        comparisons += assignments_[cluster].size();
        closest.pop();
    }
    
    // Rerank candidates.
    Topk topk;
    std::unordered_set<std::size_t> uniqueTopk;
    for (std::size_t i : uniqueCandidates) {
        updateTopk(k, distance(&query[0], &corpus[i * dim_]), i, topk, &uniqueTopk);
        candidates.pop();
    }

    return {uniqueTopk, comparisons};
}

void CQuantizedIvfIndex::searchCluster(std::size_t cluster,
                                       std::size_t k,
                                       const Point& query,
                                       Topk& topk,
                                       std::unordered_set<std::size_t> &uniques) const {

    std::size_t qbits{bits_ + 3};
    auto q = quantize(metric_, dim_, centers_[cluster], query, qbits);
    const auto& d = clusters_[cluster];
    auto dot = estimateDot(dim_, q.quantizedVectors, d.quantizedVectors,
                           q.L1Norms, d.L1Norms, q.limits, d.limits,
                           qbits, bits_);

    // Note that these are not unbiased estimates of the true similarity.
    // However, since only the order matters constant factors can be ignored.
    // WARNING if a real system mixed representation choices for different
    // data shards it would be a problem. Note we're using a min heap here so
    // convert everything to distances.

    switch (metric_) {
    case Cosine:
    case Dot: {
        float qbias{q.bias - q.localBias[0]};
        for (std::size_t i = 0; i < dot.size(); ++i) {
            float dist{qbias - d.localBias[i] - dot[i]};
            updateTopk(k, dist, assignments_[cluster][i], topk, &uniques);
        }
        break;
    }
    case Euclidean: {
        float qbias{q.localBias[0]};
        for (std::size_t i = 0; i < dot.size(); ++i) {
            float dist{qbias + d.localBias[i] - 2.0F * dot[i]};
            updateTopk(k, dist, assignments_[cluster][i], topk, &uniques);
        }
        break;
    }
    }
}

void CQuantizedIvfIndex::buildIndex(const Dataset& corpus,
                                    std::size_t target,
                                    std::size_t bits) {

    // For cosine the best strategy is to normlise the vectors before quantization.
    // This can be done on-the-fly if desired:
    //   1. The centre calculation needs to use normalised vectors.
    //   2. Each vector needs to be normalised before computing its dot product with
    //      the mean and its intervals.
    //
    // We assume this happens elsewhere.

    HierarchicalKMeansResult result;
    time([&] { result = kMeansHierarchical(dim_, corpus, target); }, "K-Means Hierarchical");
    std::cout << "Average distance to final centers: " << result.computeDispersion(dim_, corpus) << std::endl;

    centers_ = result.finalCenters();
    assignments_ = result.assignments();
    Dataset cluster;
    time([&] {
        for (std::size_t i = 0; i < centers_.size(); ++i) {
            applyTransform(dim_, permutationMatrix_, blocks_, dimBlocks_, centers_[i]);
            cluster.resize(assignments_[i].size() * dim_);
            for (std::size_t j = 0; j < assignments_[i].size(); ++j) {
                std::copy_n(&corpus[assignments_[i][j] * dim_], dim_, &cluster[j * dim_]);
            }
            applyTransform(dim_, permutationMatrix_, blocks_, dimBlocks_, cluster);
            clusters_.emplace_back(quantize(metric_, dim_, centers_[i], cluster, bits));
        }
    }, "Quantization");
}

float CQuantizedIvfIndex::distance(const float* x, const float* y) const {
    switch (metric_) {
    case Cosine:
    case Dot:
        return -dot(dim_, x, y);
    case Euclidean:
        return distanceSq(dim_, x, y);
    }
}

void CQuantizedIvfIndex::updateTopk(std::size_t k,
                                    float d,
                                    std::size_t i,
                                    Topk& topk,
                                    std::unordered_set<std::size_t>* uniques) const {
    if (topk.size() < k) {
        if (uniques != nullptr && uniques->insert(i).second) {
            topk.emplace(d, i);
        } else if (uniques == nullptr) {
            topk.emplace(d, i);
        }
    } else if (d < topk.top().first) {
        if (uniques != nullptr && uniques->insert(i).second) {
            std::size_t j{topk.top().second};
            uniques->erase(j);
            topk.pop();
            topk.emplace(d, i);
        } else if (uniques == nullptr) {
            topk.pop();
            topk.emplace(d, i);
        }
    }
}
