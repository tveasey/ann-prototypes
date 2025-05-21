#include "ivf.h"

#include "osq/limits_optimization.h"
#include "osq/preconditioner.h"
#include "osq/utils.h"
#include "hierarchical.h"
#include "../common/utils.h"

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <unordered_set>

namespace {

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

    // For cosine the best strategy is to normlise the vectors before quantization.
    // This can be done on-the-fly if desired:
    //   1. The centre calculation needs to use normalised vectors.
    //   2. Each vector needs to be normalised before computing its dot product with
    //      the mean and its intervals.

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

    // Compute preconditioner.
    time([&] {
        std::tie(blocks_, dimBlocks_) = randomOrthogonal(dim, blockDim);
        permutationMatrix_ = permutationMatrix(dim, dimBlocks_, corpus);
    }, "Preconditioning");

    buildIndex(dim, corpus, target, bits);
}

std::pair<std::unordered_set<std::size_t>, std::size_t>
CQuantizedIvfIndex::search(std::size_t nProbes,
                           std::size_t k,
                           std::size_t rerank,
                           const Point& query,
                           const Dataset& corpus) const {

    Point transformedQuery(query);
    applyTransform(dim_, permutationMatrix_, blocks_, dimBlocks_, transformedQuery);

    // Closest centers to the query.
    Topk closest;
    for (std::size_t i = 0; i < centers_.size(); ++i) {
        updateTopk(nProbes, distance(&centers_[i][0], &transformedQuery[0]), i, closest);
    }

    // Rerank candidates.
    Topk candidates;
    std::size_t comparisons{0};
    while (!closest.empty()) {
        std::size_t i{closest.top().second};
        closest.pop();
        searchCluster(i, k * rerank, transformedQuery, candidates);
        comparisons += assignments_[i].size();
    }

    // Top-k after reranking.
    // 
    // Note I handle duplicates in a dumb way by gather 2*k. This is just for convenience.
    // In practice you'd just wouldn't add duplicates to the top-k set. I just wanted to
    // reuse updateTopk, I'm not testing performance and the outcome is the same.
    Topk topk;
    while (!candidates.empty()) {
        auto [d, i] = candidates.top();
        candidates.pop();
        updateTopk(2 * k, distance(&query[0], &corpus[i * dim_]), i, topk);
    }

    std::unordered_set<std::size_t> result;
    result.reserve(k);
    while (result.size() < k && !topk.empty()) {
        auto [d, i] = topk.top();
        topk.pop();
        result.insert(i);
    }
    return {std::move(result), comparisons};
}

void CQuantizedIvfIndex::searchCluster(std::size_t cluster,
                                       std::size_t k,
                                       const Point& query,
                                       Topk& topk) const {

    auto q = quantize(metric_, dim_, centers_[cluster], query, bits_ + 3);
    const auto& d = clusters_[cluster];
    auto dot = estimateDot(dim_, q.quantizedVectors, d.quantizedVectors,
                           q.L1Norms, d.L1Norms, q.limits, d.limits,
                           bits_ + 3, bits_);

    // Note that these are not unbiased estimates of the true similarity.
    // However, since only the order matters constant factors can be ignored.
    // WARNING if a real system mixed representation choices for different
    // data shards it would be a problem. Note we're using a min heap here so
    // convert everything to distances.

    switch (metric_) {
    case Dot:
    case Cosine:
        for (std::size_t i = 0; i < dot.size(); ++i) {
            float dist{d.bias - q.localBias[i] - d.localBias[i] - dot[i]};
            updateTopk(k, dist, assignments_[cluster][i], topk);
        }
        break;
    case Euclidean:
        for (std::size_t i = 0; i < dot.size(); ++i) {
            float dist{q.localBias[i] + d.localBias[i] - 2.0F * dot[i]};
            updateTopk(k, dist, assignments_[cluster][i], topk);
        }
        break;
    }
}

void CQuantizedIvfIndex::buildIndex(std::size_t dim,
                                    const Dataset& corpus,
                                    std::size_t target,
                                    std::size_t bits) {

    HierarchicalKMeansResult result;
    time([&] { result = kMeansHierarchical(dim, corpus, target); }, "K-Means Hierarchical");
    std::cout << "Average distance to final centers: " << result.computeDispersion(dim, corpus) << std::endl;

    centers_ = result.finalCenters();
    assignments_ = result.assignments();
    Dataset cluster;
    time([&] {
        for (std::size_t i = 0; i < centers_.size(); ++i) {
            applyTransform(dim, permutationMatrix_, blocks_, dimBlocks_, centers_[i]);
            cluster.resize(assignments_[i].size() * dim);
            for (std::size_t j = 0; j < assignments_[i].size(); ++j) {
                std::copy_n(&corpus[0] + assignments_[i][j] * dim, dim, &cluster[j * dim]);
            }
            applyTransform(dim, permutationMatrix_, blocks_, dimBlocks_, cluster);
            clusters_.emplace_back(quantize(metric_, dim_, centers_[i], cluster, bits));
        }
    }, "Quantization");
}

void CQuantizedIvfIndex::updateTopk(std::size_t k, float d, std::size_t i, Topk& topk) const {
    if (topk.size() < k) {
        topk.emplace(d, i);
    } else if (d < topk.top().first) {
        topk.pop();
        topk.emplace(d, i);
    }
}

float CQuantizedIvfIndex::distance(const float* x, const float* y) const {
    switch (metric_) {
    case Cosine:
        return 1.0F - dot(dim_, x, y);
    case Dot:
        return -dot(dim_, x, y);
    case Euclidean:
        return distanceSq(dim_, x, y);
    }
}
