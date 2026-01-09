#include "annealing.h"

#include "../common/utils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <unordered_map>
#include <vector>

using Points = std::vector<float>;
using ConstPoint = const float*;
using NodeEdges = std::unordered_map<std::size_t, float>;
using GraphEdges = std::vector<NodeEdges>;
using Distances = std::vector<float>;
using Permutation = std::vector<std::size_t>;

namespace {

float INF{std::numeric_limits<float>::max()};

float sign(float x) {
    return (x > 0) - (x < 0);
}

float pow2(std::size_t x) {
    return static_cast<float>(x * x);
}

float pow2(float x) {
    return x * x;
}

float sigmoid(float x) {
    return 1.0F / (1.0F + std::expf(-x));
}

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

GraphEdges knnEdges(std::size_t dim, const Points& x, std::size_t k) {
    std::size_t n{x.size() / dim};
    GraphEdges edges(n);
    std::priority_queue<std::pair<float, std::size_t>> knn;
    for (std::size_t i = 0, id = 0; id < x.size(); ++i, id += dim) {
        for (std::size_t j = 0, jd = 0; jd < n * dim; ++j, jd += dim) {
            if (i == j) {
                continue;
            }
            float dist{euclidean2(dim, &x[id], &x[jd])};
            if (knn.size() < k) {
                knn.emplace(dist, j);
            } else if (dist < knn.top().first) {
                knn.pop();
                knn.emplace(dist, j);
            }
        }
        while (!knn.empty()) {
            edges[i].emplace(knn.top().second, std::sqrtf(knn.top().first));
            knn.pop();
        }
    }

    // Add an edge to j for each edge from i to j.
    for (std::size_t i = 0; i < n; ++i) {
        for (const auto& [j, dist] : edges[i]) {
            edges[j].emplace(i, dist);
        }
    }

    return edges;
}

Distances avgEdgeLengthByVertex(const GraphEdges& edges) {
    Distances avgDistances(edges.size(), 0.0F);
    for (std::size_t i = 0; i < edges.size(); ++i) {
        float totalDist{0.0F};
        for (const auto& [_, dist] : edges[i]) {
            totalDist += dist;
        }
        avgDistances[i] = totalDist / static_cast<float>(edges[i].size());
    }
    return avgDistances;
}

void computeEdgeWeights(std::size_t dim,
                        const Points& x,
                        const Distances& avgDistances,
                        GraphEdges& edges) {
    std::size_t n{x.size() / dim};
    for (std::size_t i = 0; i < n; ++i) {
        for (auto& [j, weight] : edges[i]) {
            float avg{0.5F * (avgDistances[i] + avgDistances[j])};
            // Cap the maximum weight to 5.
            weight = avg == 0.0 ? 5.0F : avg / (0.2F * avg + weight);
        }
    }
}

float averageVertexEdgeWeight(const GraphEdges& edges) {
    float totalWeight{0.0F};
    std::size_t vertexCount{0};
    for (std::size_t i = 0; i < edges.size(); ++i) {
        for (const auto& [_, weight] : edges[i]) {
            totalWeight += weight;
        }
        ++vertexCount;
    }
    return totalWeight / static_cast<float>(vertexCount);
}

float permutationCost(std::size_t dim,
                      std::size_t k,
                      const GraphEdges& edges,
                      const Permutation& permutation) {
    float cost{0.0F};
    std::size_t n{permutation.size()};
    for (std::size_t i = 0; i < n; ++i) {
        for (const auto& [j_, weight] : edges[permutation[i]]) {
            std::size_t j{permutation[j_]};
            cost += weight * pow2(j - i);
        }
    }
    cost *= 0.5F / static_cast<float>(n * k);
    return cost;
}

void recursiveMinCutPartition(std::size_t dim, 
                              std::size_t a,
                              std::size_t b,
                              const GraphEdges& edges,
                              Permutation& permutation,
                              std::size_t minPartitionSize = 16) {

    // Recursive binary partitioning that tries to minimize the cut cost
    // between the two halves. This is calculated based on the initial
    // positions of the vertices in permutation. Vertices that move between
    // halves are inserted at the position determined by the weighted
    // centroid of their neighbours.

    if (b - a <= minPartitionSize) {
        return;
    }
    
    std::mt19937 rng(21696172);
    auto uniform01 = [&] {
        std::uniform_real_distribution<float> dist(0.0F, 1.0F);
        return dist(rng);
    };

    // This uses an iterative refinement approach to improve the cut. Costs
    // are computed with all vertices at their current positions. Pairs are
    // identified to swap between the two halves and are made with probability
    // monotonic increasing with reduction in cost. Costs are updated after a
    // single round of swaps and the process is repeated for a number for
    // decreasing temperature. It's essentially an annealing variant of the
    // Kernighan–Lin algorithm.
    std::vector<std::size_t> order(b - a);
    std::vector<float> dcosts(b - a);
    std::vector<std::tuple<float, std::size_t>> centroids(b - a);
    std::vector<std::tuple<std::size_t, std::size_t>> swapped;
    float margin{0.08F * averageVertexEdgeWeight(edges)};
    for (float T : {2.0F, 1.0F, 0.5F, 0.25F, 0.125F}) {
        // For each vertex compute the cost of assigning a vertex to the left
        // partition minus the cost of assigning it to the right partition.
        float totalCost{0.0F};
        //swapped.clear();
        for (std::size_t j = a; j < b; ++j) {
            float rcost{0.0F};
            float lcost{0.0F};
            for (const auto& [k_, weight] : edges[permutation[j]]) {
                std::size_t k{permutation[k_]};
                if (k < (a + b) / 2) {
                    rcost += weight;
                } else {
                    lcost += weight;
                }
            }
            // Store the bias in centroids temporarily.
            dcosts[j - a] = lcost - rcost;
            totalCost += j < (a + b) / 2 ? lcost : rcost;
        }

        //std::cout << "  Partition [" << a << ", " << b << ") at temperature "
        //          << T << " cost: "
        //          << totalCost / static_cast<float>(b - a) << std::endl;

        // dcosts is in current vertex order so the left partition is the first
        // (a+b)/2 elements and the right partition is the last (a+b)/2 elements.
        // We want to swap vertices (i, j) for i in left and j in right where
        // d[i] > d[j] to reduce cost.
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.begin() + (b - a) / 2,
                  [&](std::size_t lhs, std::size_t rhs) {
                      return dcosts[lhs] > dcosts[rhs];
                  });
        std::sort(order.begin() + (b - a) / 2, order.end(),
                  [&](std::size_t lhs, std::size_t rhs) {
                      return dcosts[lhs] < dcosts[rhs];
                  });
        for (std::size_t j = 0; j < (b - a) / 2; ++j) {
            std::size_t lhs{order[j]};
            std::size_t rhs{order[(b - a) / 2 + j]};
            float acceptanceLogit{
                (dcosts[lhs] - dcosts[rhs] - margin) / (T * margin)
            };
            if (uniform01() < sigmoid(acceptanceLogit)) {
                std::swap(permutation[a + lhs], permutation[a + rhs]);
                //swapped.emplace_back(lhs, rhs);
            }
        }
    }

    // Recurse on the two halves.
    std::size_t mid{(a + b) / 2};
    recursiveMinCutPartition(dim, a, mid, refinements, edges, permutation);
    recursiveMinCutPartition(dim, mid, b, refinements, edges, permutation);
}

/*float localSearchRefinement(std::size_t dim,
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
        for (const auto& [k_, weight] : neighbourhoods[permutation[lhs]]) {
            std::size_t k{permutation[k_]};
            dcost += weight * (  (std::max(rhs, k) - std::min(rhs, k))
                               - (std::max(lhs, k) - std::min(lhs, k)));
        }
        for (const auto& [k_, weight] : neighbourhoods[permutation[rhs]]) {
            std::size_t k{permutation[k_]};
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
            candidiates.begin(), candidiates.begin() + window / 8,
            candidiates.end(),
            [&](std::size_t lhs, std::size_t rhs) {
                return costs[lhs] > costs[rhs];
            }
        );

        // We ignore indirect impact on costs and centroids even though these
        // can change because we always validate using up-to-date costs when
        // deciding if a swap.

        for (std::size_t j = 0; j < window / 8; ++j) {
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
}*/

} // namespace

Permutation annealingOrder(std::size_t dim, 
                           const Points& x,
                           std::size_t k,
                           std::size_t refinements) {

    GraphEdges edges{knnEdges(dim, x, k)};

    Distances avgDistances{avgEdgeLengthByVertex(edges)};

    computeEdgeWeights(dim, x, avgDistances, edges);

    std::size_t n{x.size() / dim};
    Permutation permutation(n);
    std::iota(permutation.begin(), permutation.end(), 0);

    std::cout << "Average cost before partitioning: "
              << permutationCost(dim, k, edges, permutation) << std::endl;

    recursiveMinCutPartition(dim, 0, n, edges, permutation);

    std::cout << "Average cost after partitioning: "
              << permutationCost(dim, k, edges, permutation) << std::endl;

    return permutation;
}

