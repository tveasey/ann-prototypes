#include "annealing.h"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <vector>

using Points = std::vector<float>;
using ConstPoint = const float*;
using Neighbourhood = std::unordered_map<std::size_t, float>;
using Neighbourhoods = std::vector<Neighbourhood>;
using Distances = std::vector<float>;
using Permutation = std::vector<std::size_t>;

namespace {

float INF{std::numeric_limits<float>::max()};

float euclidean2(std::size_t dim,
                 ConstPoint x,
                 ConstPoint y) {
    float dist{0.0F};
    for (std::size_t d = 0; d < dim; ++d) {
        float diff{x[d] - y[d]};
        dist += diff * diff;
    }
    return dist;
}

Neighbourhoods neighbourhoods(std::size_t dim,
                              const Points& x,
                              std::size_t k) {
    std::size_t n{x.size() / dim};
    Neighbourhoods result(n);
    std::priority_queue<std::pair<float, std::size_t>> knn;
    for (std::size_t i = 0, id = 0; id < x.size(); ++i, id += dim) {
        for (std::size_t j = 0, jd = 0; jd < n * dim; ++j, jd += dim) {
            if (i == j) {
                continue;
            }
            float dist{euclidean2(dim, &x[id], &x[jd])};
            if (knn.size() < k) {
                knn.emplace(dist, jd);
            } else if (dist < knn.top().first) {
                knn.pop();
                knn.emplace(dist, jd);
            }
        }
        while (!knn.empty()) {
            result[i].emplace(knn.top().second, 0.0F);
            knn.pop();
        }
    }
    return result;
}

Distances avgKnnDistance(std::size_t dim,
                         const Points& x,
                         const Neighbourhoods& neighbourhoods) {
    std::size_t n{x.size() / dim};
    Distances avgDistances(n, 0.0F);
    for (std::size_t i = 0, id = 0; id < x.size(); ++i, id += dim) {
        float totalDist{0.0F};
        for (const auto& [jd, _] : neighbourhoods[i]) {
            totalDist += std::sqrtf(euclidean2(dim, &x[id], &x[jd]));
        }
        avgDistances[i] = totalDist / static_cast<float>(neighbourhoods[i].size());
    }
    return avgDistances;
}

void computeWeights(std::size_t dim,
                    const Points& x,
                    const Distances& avgDistances,
                    Neighbourhoods& neighbourhoods) {
    std::size_t n{x.size() / dim};
    for (std::size_t i = 0, id = 0; id < x.size(); ++i, id += dim) {
        float cost{0.0F};
        std::size_t j{0};
        for (auto& [jd, weight] : neighbourhoods[i]) {
            float avg{0.5F * (avgDistances[i] + avgDistances[jd / dim])};
            weight = avg / (1e-8 * avg + std::sqrtf(euclidean2(dim, &x[id], &x[jd])));
        }
    }
}

void recursiveBinaryPartition(std::size_t dim, 
                              std::size_t a,
                              std::size_t b,
                              std::size_t refinements,
                              const Neighbourhoods& neighbourhoods,
                              Permutation& permutation) {

    // Recursive binary partition assigning points to one of the intervals
    // [a, (a+b)/2) or [(a+b)/2, b) based on their neighbourhood assignments.
    // This provides a good global ordering that can be further refined.

    if (b - a <= 1) {
        return;
    }

    std::vector<std::size_t> order(b - a);
    std::vector<float> biases(b - a, 0.0F);
    for (std::size_t i = 0; i < refinements; ++i) {
        // If a neighbour of the i'th point is in
        //   - [a, (a+b)/2) it contributes -weight to cost
        //   - [(a+b)/2, b) it contributes +weight to cost
        //   - else it contributes 0 to cost
        for (std::size_t j = a; j < b; ++j) {
            float bias{0.0F};
            for (const auto& [kd, weight] : neighbourhoods[permutation[j]]) {
                // Get current position of neighbour.
                std::size_t k{permutation[kd / dim]}; 
                if (k >= a && k < (a + b) / 2) {
                    bias -= weight;
                } else if (k >= (a + b) / 2 && k < b) {
                    bias += weight;
                }
            }
            biases[j - a] = bias;
        }

        // Points are currently at permutation[a .. b-1]. Reorder them
        // based on bias (negative means point prefers the first half).
        // Break ties by the current order.
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](std::size_t lhs, std::size_t rhs) {
                      if (biases[lhs] != biases[rhs]) {
                          return biases[lhs] < biases[rhs];
                      }
                      return permutation[a + lhs] < permutation[a + rhs];
                  });

        // Update the permutation.
        for (std::size_t j = a; j < b; ++j) {
            permutation[j] = permutation[a + order[j - a]];
        }
    }

    // Recurse on the two halves.
    std::size_t mid{(a + b) / 2};
    recursiveBinaryPartition(dim, a, mid, refinements, neighbourhoods, permutation);
    recursiveBinaryPartition(dim, mid, b, refinements, neighbourhoods, permutation);
}

float localSearchRefinement(std::size_t dim,
                            std::size_t window,
                            const Neighbourhoods& neighbourhoods,
                            Permutation& permutation) {

    std::size_t n{permutation.size()};

    // For efficiency we approximate use the distance from the centroid
    // as a proxy for delta cost.
    auto approxMinSwapCostSearch = [](const auto& centroids,
                                      auto a, auto b, auto lhs) {
        float minDCost{INF};
        std::size_t rhs{lhs};
        for (std::size_t j = a; j < b; ++j) {
            float dcost{
                std::fabs(j - centroids[lhs - a]) - std::fabs(lhs - centroids[lhs - a]) +
                std::fabs(lhs - centroids[j - a]) - std::fabs(j - centroids[j - a])
            };
            std::tie(minDCost, rhs) = std::min(
                std::make_tuple(minDCost, rhs), std::make_tuple(dcost, j)
            );
        }
        return rhs;
    };

    auto swapCost = [&](auto lhs, auto rhs) {
        if (lhs == rhs) {
            return 0.0F;
        }
        float dcost{0.0F};
        for (const auto& [kd, weight] : neighbourhoods[permutation[lhs]]) {
            std::size_t k{permutation[kd / dim]};
            dcost += weight * (  (std::max(rhs, k) - std::min(rhs, k))
                               - (std::max(lhs, k) - std::min(lhs, k)));
        }
        for (const auto& [kd, weight] : neighbourhoods[permutation[rhs]]) {
            std::size_t k{permutation[kd / dim]};
            dcost += weight * (  (std::max(lhs, k) - std::min(lhs, k))
                               - (std::max(rhs, k) - std::min(rhs, k)));
        }
        return dcost;
    };

    // For a sliding window:
    //   1. Find the points with the highest cost at their current positions.
    //   2. If their centroids are left search the left window for the best swap.
    //   3. If their centroids are right search the right window for the best swap.
    //   4. If any swap reduces cost perform it.
    //   5. Move the window position and repeat.

    std::vector<std::size_t> candidiates;
    std::vector<float> costs;
    std::vector<float> centroids;

    float totalCostReduction{0.0F};
    for (std::size_t i = 0; i < n; i += window / 2) {
        std::size_t a{(i >= window / 2) ? i - window / 2 : 0};
        std::size_t b{std::min(i + window / 2, n)};

        candidiates.resize(b - a);
        std::iota(candidiates.begin(), candidiates.end(), 0);
        costs.resize(b - a, 0.0F);
        centroids.resize(b - a, 0.0F);

        for (std::size_t j = a; j < b; ++j) {
            float cost{0.0F};
            float centroid{0.0F};
            float Z{0.0F};
            for (const auto& [kd, weight] : neighbourhoods[permutation[j]]) {
                std::size_t k{permutation[kd / dim]}; 
                cost += weight * (std::max(j, k) - std::min(j, k));
                centroid += weight * k;
                Z += weight;
            }
            costs[j - a] = cost;
            centroids[j - a] = centroid / Z;
        }

        // Find the highest cost candidates.
        std::partial_sort(
            candidiates.begin(), candidiates.begin() + window / 16,
            candidiates.end(),
            [&](std::size_t lhs, std::size_t rhs) {
                return costs[lhs] > costs[rhs];
            }
        );

        // We ignore indirect impact on costs and centroids even though these
        // can change because we always validate using up-to-date costs when
        // deciding if a swap.

        for (std::size_t j = 0; j < window / 16; ++j) {
            std::size_t maxPos{candidiates[j]};
            if (centroids[maxPos] < static_cast<float>(a + maxPos)) {
                std::size_t cand{
                    approxMinSwapCostSearch(centroids, a, a + maxPos, a + maxPos)
                };
                float dcost{swapCost(a + maxPos, a + cand)};
                if (dcost < 0.0F) {
                    std::swap(permutation[a + maxPos], permutation[a + cand]);
                    std::swap(centroids[maxPos], centroids[cand]);
                    totalCostReduction -= dcost;
                }
            } else if (centroids[maxPos] > static_cast<float>(a + maxPos)) {
                std::size_t cand{
                    approxMinSwapCostSearch(centroids, a + maxPos + 1, b, a + maxPos)
                };
                float dcost{swapCost(a + maxPos, a + cand)};
                if (dcost < 0.0F) {
                    std::swap(permutation[a + maxPos], permutation[a + cand]);
                    std::swap(centroids[maxPos], centroids[cand]);
                    totalCostReduction -= dcost;
                }
            }
        }
    }

    return totalCostReduction;
}

} // namespace

Permutation annealingOrder(std::size_t dim, 
                           const Points& x,
                           std::size_t k,
                           std::size_t refinements) {

    Neighbourhoods neighbourhoods_{neighbourhoods(dim, x, k)};

    Distances avgDistances{avgKnnDistance(dim, x, neighbourhoods_)};

    computeWeights(dim, x, avgDistances, neighbourhoods_);

    std::size_t n{x.size() / dim};
    Permutation permutation(n);
    std::iota(permutation.begin(), permutation.end(), 0);

    recursiveBinaryPartition(dim, 0, n, refinements, neighbourhoods_, permutation);

    // Annealing style local search refinements with decreasing window sizes.
    for (auto window : {128, 64, 32, 16}) {
        float costReduction{localSearchRefinement(dim, window, neighbourhoods_, permutation)};
        std::cout << "Window size: " << window
                  << ", cost reduction: " << costReduction << std::endl;
    }

    return permutation;
}

