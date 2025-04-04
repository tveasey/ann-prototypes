#include "common.h"

#include <vector>

namespace {

bool stepLloyd(std::size_t dim,
               const Dataset& dataset,
               Centers& centers,
               std::vector<std::size_t>& a) {

    bool changed{false};
    std::vector<std::size_t> q(centers.size() / dim, 0);
    Centers newCenters(centers.size(), 0.0F);
    for (std::size_t i = 0, id = 0; id < dataset.size(); ++i, id += dim) {
        std::size_t bestJd{0};
        float minDsq{std::numeric_limits<float>::max()};
        for (std::size_t jd = 0; jd < centers.size(); jd += dim) {
            float d_sq = distanceSq(dim, &dataset[id], &centers[jd]);
            if (d_sq < minDsq) {
                minDsq = d_sq;
                bestJd = jd;
            }
        }
        changed |= (a[i] != bestJd);
        a[i] = bestJd;
        ++q[bestJd / dim];
        for (std::size_t d = 0; d < dim; ++d) {
            newCenters[bestJd + d] += dataset[id + d];
        }
    }

    for (std::size_t i = 0, id = 0; id < centers.size(); ++i, id += dim) {
        if (q[i] > 0) {
            for (std::size_t d = 0; d < dim; ++d) {
                centers[id + d] = newCenters[id + d] / q[i];
            }
        }
    }

    return changed;
}
}

KMeansResult kMeans(std::size_t dim,
                    const Dataset& dataset,
                    Centers initial_centers,
                    std::size_t k,
                    std::size_t max_iterations = 300) {
    // Initialize assignments
    std::vector<std::size_t> a(dataset.size() / dim, 0);
    Centers centers{std::move(initial_centers)}; // Working copy of centers
    std::size_t iter{0}; // Number of centers
    bool converged{false};
    for (; iter < max_iterations; ++iter) {
        if (!stepLloyd(dim, dataset, centers, a)) {
            converged = true;
            break;
        }
    }

    std::vector<std::size_t> counts(k, 0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        ++counts[a[i] / dim];
    }

    return {k, std::move(centers), std::move(a), iter, converged};
}