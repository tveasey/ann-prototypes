
#include "local.h"
#include "common.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

#include <iostream>

namespace {
using Neighborhoods = std::vector<std::vector<std::size_t>>;
using NeighborQueues = std::vector<std::priority_queue<std::pair<float, std::size_t>>>;
float INF{std::numeric_limits<float>::max()};

void computeNeighborhoods(std::size_t dim,
                          const Centers& centers,
                          Neighborhoods& neighborhoods,
                          std::size_t clustersPerNeighborhood) {

    auto updateNeighbors = [&](std::size_t id,
                               float dsq,
                               auto& neighborhood) {
        if (neighborhood.size() < clustersPerNeighborhood) {
            neighborhood.emplace(dsq, id);
        } else if (dsq < neighborhood.top().first) {
            neighborhood.pop();
            neighborhood.emplace(dsq, id);
        }
    };

    std::size_t k{neighborhoods.size()};
    NeighborQueues neighborhoods_(k);

    for (std::size_t i = 0, id = 0; i < k; ++i, id += dim) {
        for (std::size_t j = 0, jd = 0; j < i; ++j, jd += dim) {
            float dsq{distanceSq(dim, &centers[id], &centers[jd])};
            updateNeighbors(jd, dsq, neighborhoods_[i]);
            updateNeighbors(id, dsq, neighborhoods_[j]);
        }
    }
    for (std::size_t i = 0; i < k; ++i) {
        neighborhoods[i].resize(neighborhoods_[i].size());
        std::size_t j{0};
        while (!neighborhoods_[i].empty()) {
            neighborhoods[i][j++] = neighborhoods_[i].top().second;
            neighborhoods_[i].pop();
        }
        std::sort(neighborhoods[i].begin(), neighborhoods[i].end());
    };
}

bool stepLloyd(std::size_t dim,
               const Dataset& dataset,
               const Neighborhoods& neighborhoods,
               Centers& centers,
               Centers& nextCenters,
               std::vector<std::size_t>& q,
               std::vector<std::size_t>& a) {

    bool changed{false};

    nextCenters.assign(centers.size(), 0.0F);
    q.assign(centers.size() / dim, 0);

    for (std::size_t i = 0, id = 0; id < dataset.size(); ++i, id += dim) {
        std::size_t currJd{a[i]};
        std::size_t bestJd{currJd};
        float minDsq{distanceSq(dim, &dataset[id], &centers[currJd])};
        for (std::size_t jd : neighborhoods[currJd / dim]) {
            float dsq{distanceSq(dim, &dataset[id], &centers[jd])};
            if (dsq < minDsq) {
                minDsq = dsq;
                bestJd = jd;
            }
        }
        changed |= (a[i] != bestJd);
        a[i] = bestJd;
        ++q[bestJd / dim];
        #pragma omp simd
        for (std::size_t d = 0; d < dim; ++d) {
            nextCenters[bestJd + d] += dataset[id + d];
        }
    }

    for (std::size_t i = 0, id = 0; id < centers.size(); ++i, id += dim) {
        if (q[i] > 0) {
            #pragma omp simd
            for (std::size_t d = 0; d < dim; ++d) {
                centers[id + d] = nextCenters[id + d] / q[i];
            }
        }
    }

    return changed;
}

void assignSpilled(std::size_t dim,
                   const std::vector<float>& dataset,
                   const Neighborhoods& neighborhoods,
                   const std::vector<float>& centers,
                   float lambda,
                   const std::vector<std::size_t>& a,
                   std::vector<std::size_t>& sa) {

    // SOAR uses an adjusted distance for assigning spilled documents which is
    // given by:
    //
    //   soar(x, c) = ||x - c||^2 + lambda * ((x - c_1)^t (x - c))^2 / ||x - c_1||^2
    //
    // Here, x is the document, c is the nearest centroid, and c_1 is the first
    // centroid the document was assigned to. The document is assigned to the
    // cluster with the smallest soar(x, c).

    std::cout << "Assigning spilled documents..." << std::endl;
    sa.resize(a.size());

    std::vector<float> r1(dim);
    for (std::size_t i = 0, id = 0; id < dataset.size(); ++i, id += dim) {
        const auto* xi{&dataset[id]};

        std::size_t currJd{a[i]};
        const float* c1{&centers[currJd]};
        float dsq1{0.0F};
        #pragma omp simd reduction(+:dsq1)
        for (std::size_t j = 0; j < dim; ++j) {
            float diff{xi[j] - c1[j]};
            r1[j] = diff;
            dsq1 += diff * diff;
        }

        std::size_t bestJd{0};
        float minSoar{INF};
        for (std::size_t jd : neighborhoods[currJd / dim]) {
            float dsq2{0.0F};
            float proj{0.0F};
            const float* cj{&centers[jd]};
            #pragma omp simd reduction(+:dsq2, proj)
            for (std::size_t k = 0; k < dim; ++k) {
                float diff{xi[k] - cj[k]};
                float diffProj{r1[k] * diff};
                dsq2 += diff * diff;
                proj += diffProj;
            }
            float soar{dsq2 + lambda * proj * proj / dsq1};
            if (soar < minSoar) {
                bestJd = jd;
                minSoar = soar;
            }
        }

        sa[i] = bestJd;
    }
    std::cout << "Done assigning spilled documents." << std::endl;
}
}

KMeansResult kMeansLocal(std::size_t dim,
                         const Dataset& dataset,
                         Centers centers,
                         std::vector<std::size_t> assignments,
                         std::size_t clustersPerNeighborhood,
                         std::size_t maxIterations,
                         float lambda) {

    std::size_t n{dataset.size() / dim};
    std::size_t m{centers.size()};
    std::size_t k{m / dim};

    if (k == 1 || k >= n) {
        return {k, std::move(centers), std::move(assignments), {}, 0, true};
    }

    Neighborhoods neighborhoods(k);
    computeNeighborhoods(dim, centers, neighborhoods, clustersPerNeighborhood);

    std::size_t iter{0}; // Number of centers
    bool converged{false};
    std::vector<std::size_t> q(k, 0);
    Centers nextCenters(m, 0.0F);
    for (; iter < maxIterations; ++iter) {
        if (!stepLloyd(dim, dataset, neighborhoods,
                       centers, nextCenters, q, assignments)) {
            converged = true;
            break;
        }
    }

    std::vector<std::size_t> spilledAssignments;
    assignSpilled(dim, dataset, neighborhoods, centers,
                  lambda, assignments, spilledAssignments);

    return {k, std::move(centers),
            std::move(assignments), std::move(spilledAssignments),
            iter, converged};
}
