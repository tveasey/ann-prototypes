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
                               std::size_t sampleSize,
                               std::size_t k,
                               Centers& centers) {

    // Choose data points as random ensuring we have distinct points.

    std::vector<std::size_t> candidates(sampleSize / dim);
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
                                            std::size_t maxK,
                                            std::size_t maxFixupIterations,
                                            std::size_t samplesPerCluster,
                                            std::size_t depth) {

    std::size_t n{dataset.size() / dim};

    if (n <= targetSize) {
        return {};
    }

    std::size_t k{std::clamp((n + targetSize / 2) / targetSize, 2UL, maxK)};
    std::size_t m{std::min(k * samplesPerCluster * dim, dataset.size())};

    Centers initialCenters;
    k = pickInitialCenters(dim, dataset, m, k, initialCenters);

    HierarchicalKMeansResult result{
        kMeans(dim, dataset, m, std::move(initialCenters), 2 * maxFixupIterations)
    };

    if (result.effectiveK() == 1) {
        return result;
    }

    std::vector<std::size_t> counts(result.clusterSizes());
    Dataset cluster;
    for (std::size_t c = 0; c < counts.size(); ++c) {
        // Recurse for each cluster which is larger than 1.33 x target. If
        // size < 1.33 x target then |size / 2 - target| > |size - target|.
        if (100 * counts[c] > 133 * targetSize) {
            result.copyClusterPoints(dim, c, dataset, cluster);
            result.updateAssignmentsWithRecursiveSplit(
                dim, c, kMeansHierarchical(dim, cluster, targetSize, maxK,
                                           maxFixupIterations, samplesPerCluster,
                                           depth + 1)
            );
        }
    }

    if (depth == 0) {
        float f{std::min(static_cast<float>(samplesPerCluster) / targetSize, 1.0F)};
        m = dim * (static_cast<std::size_t>(f * dataset.size()) / dim);
        result = HierarchicalKMeansResult{
            kMeansLocal(dim, dataset, m, result.finalCentersFlat(),
                        result.assignmentsFlat(), maxK, maxFixupIterations)
        };
    }

    return result;
}
