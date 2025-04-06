
#include "local.h"
#include "common.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

namespace {
using Neighbors = std::vector<std::vector<std::size_t>>;
using NeighborQueues = std::vector<std::priority_queue<std::pair<float, std::size_t>>>;
float INF{std::numeric_limits<float>::max()};
}

HierarchicalKMeansResult kMeansLocal(std::size_t dim,
                                     const Dataset& dataset,
                                     std::vector<Centers> centers,
                                     std::vector<std::vector<std::size_t>> assignments,
                                     std::size_t clustersPerNeighborhood,
                                     std::size_t maxIterations) {

    // Swap all points to their nearest cluster center.
    // For each cluster check the 10 nearest neighbour clusters as candidates.

    auto updateNeighbors = [](std::size_t i,
                              std::size_t j,
                              float dsq,
                              auto& candidates) {
        if (candidates[i].size() < 10) {
            candidates[i].emplace(dsq, j);
        } else if (dsq < candidates[i].top().first) {
            candidates[i].pop();
            candidates[i].emplace(dsq, j);
        }
    };

    std::size_t n{dataset.size() / dim};
    std::size_t m{centers.size()};
    Neighbors neighbors(m);
    NeighborQueues neighbors_(m);
    for (std::size_t i = 0; i < centers.size(); ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            float dsq{distanceSq(dim, &centers[i][0], &centers[j][0])};
            updateNeighbors(i, j, dsq, neighbors_);
            updateNeighbors(j, i, dsq, neighbors_);
        }
    }
    for (std::size_t i = 0; i < neighbors_.size(); ++i) {
        neighbors[i].resize(neighbors_[i].size());
        std::size_t j{0};
        while (!neighbors_[i].empty()) {
            neighbors[i][j++] = neighbors_[i].top().second;
            neighbors_[i].pop();
        }
    };

    bool converged{false};
    std::size_t iter{0};
    std::vector<std::size_t> moved;
    Centers nextCenter(dim);

    for (; iter < maxIterations; ++iter) {

        converged = true;

        for (std::size_t i = 0; i < assignments.size(); ++i) {
            auto& assignment = assignments[i];
            if (assignment.size() == 0) {
                continue;
            }

            moved.clear();
            moved.reserve(assignment.size());

            const auto& iNeighbors = neighbors[i];
            for (std::size_t j : assignment) {
                ConstPoint xj{&dataset[j * dim]};
                float dsq{distanceSq(dim, &centers[i][0], xj)};
                std::size_t bestK{i};
                for (std::size_t k : iNeighbors) {
                    float dsqk{distanceSq(dim, &centers[k][0], xj)};
                    if (dsqk < dsq) {
                        dsq = dsqk;
                        bestK = k;
                    }
                }
                if (bestK != i) {
                    moved.push_back(j);
                    assignments[bestK].push_back(j);
                }
            }

            if (moved.size() > 0) {
                std::sort(moved.begin(), moved.end());
                assignment.erase(std::remove_if(
                    assignment.begin(), assignment.end(),
                    [&moved](std::size_t j) {
                        return std::binary_search(moved.begin(), moved.end(), j);
                    }), assignment.end());
                converged = false;
            }
        }
        if (converged) {
            break;
        }

        for (std::size_t i = 0; i < assignments.size(); ++i) {
            auto& assignment = assignments[i];
            if (assignment.size() == 0) {
                continue;
            }
            std::sort(assignment.begin(), assignment.end());
            nextCenter.assign(dim, 0.0F);
            for (std::size_t j : assignment) {
                ConstPoint xj{&dataset[j * dim]};
                #pragma omp simd
                for (std::size_t d = 0; d < dim; ++d) {
                    nextCenter[d] += xj[d];
                }
            }
            std::size_t size{assignment.size()};
            if (size > 0) {
                #pragma omp simd
                for (std::size_t d = 0; d < dim; ++d) {
                    centers[i][d] = nextCenter[d] / size;
                }
            }
        };
    }

    return {std::move(centers), std::move(assignments)};
}