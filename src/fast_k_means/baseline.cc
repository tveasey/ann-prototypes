#include "baseline.h"
#include "common.h"

#include <numeric>
#include <vector>

namespace {

bool stepLloyd(std::size_t dim,
               const Dataset& dataset,
               Centers& centers,
               Centers& nextCenters,
               std::vector<std::size_t>& q,
               std::vector<std::size_t>& a) {

    bool changed{false};

    nextCenters.assign(centers.size(), 0.0F);
    q.assign(centers.size() / dim, 0);

    for (std::size_t i = 0, id = 0; id < dataset.size(); ++i, id += dim) {
        std::size_t bestJd{0};
        float minDsq{INF};
        for (std::size_t jd = 0; jd < centers.size(); jd += dim) {
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
}

KMeansResult kMeans(std::size_t dim,
                    const Dataset& dataset,
                    Centers initialCenters,
                    std::size_t k,
                    std::size_t maxIterations) {

    std::size_t n{dataset.size() / dim};

    std::vector<std::size_t> a(n, 0);
    Centers centers{std::move(initialCenters)}; // Current centers
    Centers nextCenters; // Next centers

    if (k == 1) {
        centroid(dim, dataset, &centers[0]);
        return {k, std::move(centers), std::move(a), {}, 0, true};
    }
    if (k >= n) {
        k = n;
        std::iota(a.begin(), a.end(), 0);
        centers.resize(k * dim);
        std::copy_n(dataset.begin(), dataset.size(), centers.begin());
        return {k, std::move(centers), std::move(a), {}, 0, true};
    }

    std::size_t iter{0}; // Number of centers
    bool converged{false};
    std::vector<std::size_t> q(k, 0);
    for (; iter < maxIterations; ++iter) {
        if (!stepLloyd(dim, dataset, centers, nextCenters, q, a)) {
            converged = true;
            break;
        }
    }

    return {k, std::move(centers), std::move(a), {}, iter, converged};
}