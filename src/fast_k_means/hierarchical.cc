#include "common.h"
#include "baseline.h"
#include "local.h"

#include <algorithm>
#include <cassert>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

std::size_t pickInitialCenters(std::size_t dim,
                               const Dataset& dataset,
                               std::size_t k,
                               Centers& centers) {

    // Choose data points as random ensuring we have distinct points.

    std::vector<std::size_t> candidates(dataset.size() / dim);
    std::iota(candidates.begin(), candidates.end(), 0);
    std::minstd_rand rng;
    std::shuffle(candidates.begin(), candidates.end(), rng);

    centers.resize(k * dim);
    std::size_t k_{0};
    for (std::size_t i = 0; i < candidates.size() && k_ < k; ++i) {
        std::size_t cand{candidates[i] * dim};
        if (std::any_of(centers.begin(), centers.begin() + k_ * dim,
                        [&dataset, dim, cand](const float& center) {
                            return distanceSq(dim, &dataset[cand], &center) == 0.0F;
                        })) {
            continue;
        }
        std::copy_n(&dataset[cand], dim, &centers[k_++ * dim]);
    }
    centers.resize(k_ * dim);
    return k_;
}

HierarchicalKMeansResult kMeansHierarchical(std::size_t dim,
                                            const Dataset& dataset,
                                            std::size_t targetSize,
                                            std::size_t maxIterations,
                                            std::size_t maxK,
                                            std::size_t samplesPerCluster,
                                            std::size_t clustersPerNeighborhood,
                                            std::size_t depth) {

    std::size_t n{dataset.size() / dim};

    if (n <= targetSize) {
        return {};
    }

    std::size_t k{std::clamp((n + targetSize - 1) / targetSize, 2UL, maxK)};
    std::size_t m{std::min(k * samplesPerCluster * dim, dataset.size())};

    Dataset sample;
    if (m == dataset.size()) {
        sample = dataset;
    } else {
        sample.resize(m);
        std::copy_n(dataset.begin(), m, sample.begin());
    }

    Centers initialCenters;
    k = pickInitialCenters(dim, sample, k, initialCenters);

    HierarchicalKMeansResult result;
    {
        KMeansResult result_{kMeans(
            dim, sample, std::move(initialCenters), k, maxIterations
        )};
        result_.assignRemainingPoints(dim, sample.size(), dataset);
        result = HierarchicalKMeansResult(result_);
    }

    if (k == 1) {
        return result;
    }

    std::vector<std::size_t> counts(result.clusterSizes());
    for (std::size_t c = 0; c < counts.size(); ++c) {
        // Recurse for each cluster which is larger than targetSize.
        // Give ourselves 30% margin for the target size.
        if (10 * counts[c] > 13 * targetSize) {
            result.copyClusterPoints(dim, c, dataset, sample);
            result.updateAssignmentsWithRecursiveSplit(
                dim, c, kMeansHierarchical(dim, sample, targetSize, maxIterations,
                                           maxK, samplesPerCluster,
                                           clustersPerNeighborhood, depth + 1)
            );
        }
    }

    if (depth == 0) {
        result = HierarchicalKMeansResult{kMeansLocal(
            dim, dataset, result.finalCentersFlat(), result.assignmentsFlat(),
            clustersPerNeighborhood, maxIterations
        )};
    }

    return result;
}
