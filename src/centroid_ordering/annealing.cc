#include "annealing.h"

#include "hilbert_curve.h"
#include "../common/utils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <vector>

using Points = std::vector<float>;
using ConstPoint = const float*;
using NodeEdges = std::vector<std::pair<std::size_t, float>>;
using GraphEdges = std::vector<NodeEdges>;
using Distances = std::vector<float>;
using Permutation = std::vector<std::size_t>;

namespace {

float INF{std::numeric_limits<float>::max()};

inline float diff(std::size_t i, std::size_t j) {
    return static_cast<float>(i < j ? j - i : i - j);
}

inline float sigmoid(float x) {
    return 1.0F / (1.0F + std::expf(-x));
}

float euclidean2(std::size_t dim, ConstPoint x, ConstPoint y) {
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
    std::vector<std::priority_queue<std::pair<float, std::size_t>>> knn(n);
    for (std::size_t i = 0, id = 0; i < n; ++i, id += dim) {
        for (std::size_t j = 0, jd = 0; j < i; ++j, jd += dim) {
            float dist{euclidean2(dim, &x[id], &x[jd])};
            if (knn[i].size() < k) {
                knn[i].emplace(dist, j);
            } else if (dist < knn[i].top().first) {
                knn[i].pop();
                knn[i].emplace(dist, j);
            }
            if (knn[j].size() < k) {
                knn[j].emplace(dist, i);
            } else if (dist < knn[j].top().first) {
                knn[j].pop();
                knn[j].emplace(dist, i);
            }
        }
    }
    for (std::size_t i = 0; i < n; ++i) {
        while (!knn[i].empty()) {
            auto [dist2, j] = knn[i].top();
            edges[i].emplace_back(j, std::sqrtf(dist2));
            knn[i].pop();
        }
    }

    // Add an edge to j for each edge from i to j.
    for (std::size_t i = 0; i < n; ++i) {
        for (const auto& [j, dist] : edges[i]) {
            edges[j].emplace_back(i, dist);
        }
    }
    // If i -> j and j -> i don't duplicate the edge.
    for (std::size_t i = 0; i < n; ++i) {
        std::sort(edges[i].begin(), edges[i].end());
        edges[i].erase(std::unique(edges[i].begin(), edges[i].end(),
                                   [](const auto& lhs, const auto& rhs) {
                                       return lhs.first == rhs.first;
                                   }), edges[i].end());
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

float permutationCost(const GraphEdges& edges,
                      const Permutation& ptov,
                      const Permutation& vtop) {
    float cost{0.0F};
    std::size_t endpoints{0};
    for (std::size_t i_ = 0; i_ < edges.size(); ++i_) {
        std::size_t i{vtop[i_]};
        for (const auto& [j_, weight] : edges[i_]) {
            std::size_t j{vtop[j_]};
            cost += weight * diff(i, j);
            ++endpoints;
        }
    }
    return cost / static_cast<float>(endpoints);
}

void recursiveMinCutPartition(std::mt19937 &rng,
                              std::size_t a,
                              std::size_t b,
                              const GraphEdges& edges,
                              Permutation& ptov,
                              Permutation& vtop,
                              float margin,
                              std::size_t minPartitionSize = 8) {

    // Recursive binary partitioning to minimize the cut weight at each level.

    if (b - a <= minPartitionSize) {
        return;
    }

    std::uniform_real_distribution<float> dist(0.0F, 1.0F);
    auto uniform01 = [&] { return dist(rng); };

    // This uses an iterative refinement approach to improve the cut. Weights
    // are computed with all vertices at their current positions. Pair swaps
    // are made with probability monotonic increasing with their reduction in
    // cut weight. Cut weights are updated after a single round of swaps and
    // the process repeats with exponentially decreasing temperature. It's
    // essentially an annealing variant of the Kernighan–Lin's algorithm. We
    // allow stale weights because swapping choices are made probabilistically
    // anyway.

    std::size_t mid{(a + b) / 2};

    std::vector<std::size_t> dcostOrder(b - a);
    std::vector<float> dcosts(b - a);
    for (float T : {8.0F, 4.0F, 2.0F, 1.0F, 0.5F, 0.25F}) {

        // For each vertex compute the cost of assigning a vertex to the left
        // partition minus the cost of assigning it to the right partition.
        //float totalCost{0.0F};
        for (std::size_t j = a; j < b; ++j) {
            float rcost{0.0F};
            float lcost{0.0F};
            for (const auto& [k_, weight] : edges[ptov[j]]) {
                (vtop[k_] < mid ? rcost : lcost) += weight;
            }
            dcosts[j - a] = lcost - rcost;
            //totalCost += j < mid ? lcost : rcost;
        }
        //std::cout << "  Partition [" << a << ", " << b << ") at temperature "
        //          << T << " cost: "
        //          << totalCost / static_cast<float>(b - a) << std::endl;

        // dcosts is in current vertex order so the left partition is the first
        // (a+b)/2 elements and the right partition is the last (a+b)/2 elements.
        std::iota(dcostOrder.begin(), dcostOrder.end(), 0);
        std::sort(dcostOrder.begin(), dcostOrder.begin() + (mid - a),
                  [&](std::size_t lhs, std::size_t rhs) {
                      return dcosts[lhs] > dcosts[rhs];
                  });
        std::sort(dcostOrder.begin() + (mid - a), dcostOrder.end(),
                  [&](std::size_t lhs, std::size_t rhs) {
                      return dcosts[lhs] < dcosts[rhs];
                  });

        // We want to swap vertices (i, j) for i in left and j in right where
        // d[i] > d[j] to reduce the cut cost.
        for (std::size_t j = 0; j < mid - a; ++j) {
            std::size_t lhs{dcostOrder[j]};
            std::size_t rhs{dcostOrder[mid - a + j]};
            float acceptanceLogit{
                (dcosts[lhs] - dcosts[rhs] - margin) / (T * margin)
            };
            if (uniform01() < sigmoid(acceptanceLogit)) {
                lhs += a;
                rhs += a;
                std::swap(ptov[lhs], ptov[rhs]);
                vtop[ptov[lhs]] = lhs;
                vtop[ptov[rhs]] = rhs;
            }
            if (acceptanceLogit < -10.0F) {
                // Early exit since further swaps are very unlikely.
                break;
            }
        }
    }

    // Recurse on the 2-split.
    recursiveMinCutPartition(rng, a, mid, edges, ptov, vtop, margin);
    recursiveMinCutPartition(rng, mid, b, edges, ptov, vtop, margin);
}

void local1Opt(const GraphEdges& edges,
               Permutation& ptov,
               Permutation& vtop,
               std::size_t maxRounds = 128) {

    std::size_t swaps;
    std::size_t roundsLeft{maxRounds};

    do {
        swaps = 0;
        for (std::size_t i = 1, j = 0; i < ptov.size(); ++i, ++j) {
            // Try swapping vertex i with it's neighbours to see if cost improves.
            float dcost{0.0F};
            for (const auto& [vk, weight] : edges[ptov[i]]) {
                std::size_t k{vtop[vk]};
                if (k == j) continue; // Skip since edge length is unchanged.
                dcost += weight * (diff(k, j) - diff(k, i));
            }
            for (const auto& [vk, weight] : edges[ptov[j]]) {
                std::size_t k{vtop[vk]};
                if (k == i) continue; // Skip since edge length is unchanged.
                dcost += weight * (diff(k, i) - diff(k, j));
            }
            if (dcost < 0.0F) {
                std::swap(ptov[i], ptov[j]);
                vtop[ptov[i]] = i;
                vtop[ptov[j]] = j;
                ++swaps;
            }
        }
        //std::cout << swaps << " swaps in polishing round, "
        //          << roundsLeft - 1 << " rounds remaining." << std::endl;
    } while (swaps > 0 && --roundsLeft > 0);
}

} // namespace

PermutationCost annealingOrder(std::size_t dim,
                               const Points& x,
                               bool hilbertInitialization,
                               std::size_t k,
                               std::size_t probes) {

    GraphEdges edges{knnEdges(dim, x, k)};

    Distances avgDistances{avgEdgeLengthByVertex(edges)};

    computeEdgeWeights(dim, x, avgDistances, edges);

    // This should be parameterised by degree if it varies significantly
    // between vertices.
    float margin{0.08F * averageVertexEdgeWeight(edges)};

    std::size_t n{x.size() / dim};
    // position -> vertex and vertex -> position maps initialized to identity.
    Permutation ptov(n);
    Permutation vtop(n);
    if (false) {
        // Debug: random initial ordering.
        std::random_device rd;
        std::mt19937 rng(rd());
        std::iota(ptov.begin(), ptov.end(), 0);
        std::shuffle(ptov.begin(), ptov.end(), rng);
        for (std::size_t i = 0; i < n; ++i) {
            vtop[ptov[i]] = i;
        }

    } else if (hilbertInitialization) {
        time([&] { ptov = hilbertOrder(dim, 16, x); }, "Initial Hilbert Ordering");
        for (std::size_t i = 0; i < n; ++i) {
            vtop[ptov[i]] = i;
        }
    } else {
        std::iota(ptov.begin(), ptov.end(), 0);
        std::iota(vtop.begin(), vtop.end(), 0);
    }

    std::random_device rd;
    std::mt19937 rng(rd());

    Permutation minPtov{ptov};
    Permutation minVtop{vtop};
    float minCost{permutationCost(edges, ptov, vtop)};
    std::cout << "Average cost before partitioning: " << minCost << std::endl;

    time([&] {
        for (std::size_t i = 0; i < probes; ++i) {
            recursiveMinCutPartition(rng, 0, n, edges, ptov, vtop, margin);
            float cost{permutationCost(edges, ptov, vtop)};
            //std::cout << "  Cost after probe " << (i + 1) << ": " << cost << std::endl;
            if (cost < minCost) {
                minCost = cost;
                minPtov = ptov;
                minVtop = vtop;
            }
            if (2 * i < probes) {
                std::iota(ptov.begin(), ptov.end(), 0);
                std::iota(vtop.begin(), vtop.end(), 0);
            } else {
                ptov = minPtov;
                vtop = minVtop;
            }
        }
        std::cout << "Average cost after partitioning: " << minCost << std::endl;
    }, "Recursive Min-Cut Partitioning");

    time([&] { local1Opt(edges, minPtov, minVtop); }, "Local 1-Opt Refinement");
    minCost = permutationCost(edges, minPtov, minVtop);
    std::cout << "Average cost after local 1-opt: " << minCost << std::endl;

    return {std::move(minPtov), minCost};
}
