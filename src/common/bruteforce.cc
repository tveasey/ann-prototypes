#include "bruteforce.h"

#include "bigvector.h"

#include <cstddef>
#include <queue>
#include <utility>
#include <vector>

namespace {

std::pair<std::vector<std::size_t>, std::vector<float>>
extractIdsAndScores(std::priority_queue<std::pair<float, std::size_t>> topk) {
    std::vector<std::size_t> topkIds(topk.size());
    std::vector<float> topkScores(topk.size());
    for (std::size_t i = topk.size(); !topk.empty(); --i) {
        topkIds[i - 1] = topk.top().second;
        topkScores[i - 1] = topk.top().first;
        topk.pop();
    }
    return {std::move(topkIds), std::move(topkScores)};
}

} // unnamed::

float dotf(std::size_t dim, const float* x, const float* y) {
    float xy{0.0F};
    #pragma omp simd reduction(+:xy)
    for (std::size_t i = 0; i < dim; ++i) {
        xy += x[i] * y[i];
    }
    return xy;
}

std::pair<std::vector<std::size_t>, std::vector<float>>
searchBruteForce(std::size_t k,
                 const BigVector& docs,
                 const std::vector<float>& query) {

    std::size_t dim{query.size()};
    if (dim != docs.dim()) {
        throw std::invalid_argument("query and docs have different dimensions");
    }

    std::priority_queue<std::pair<float, std::size_t>> topk;
    std::size_t id{0};
    for (auto doc : docs) {
        float sim{0.0F};
        auto* data = doc.data();
        #pragma omp simd reduction(+:sim)
        for (std::size_t j = 0; j < dim; ++j) {
            sim += query[j] * data[j];
        }
        float dist{1.0F - sim};
        if (topk.size() < k) {
            topk.push(std::make_pair(dist, id));
        } else if (topk.top().first > dist) {
            topk.pop();
            topk.push(std::make_pair(dist, id));
        }
        ++id;
    }

    return extractIdsAndScores(std::move(topk));
}

std::pair<std::vector<std::size_t>, std::vector<float>>
searchBruteForce(std::size_t k,
                 const std::vector<float>& docs,
                 const std::vector<float>& query) {
    std::size_t dim{query.size()};
    std::priority_queue<std::pair<float, std::size_t>> topk;
    for (std::size_t i = 0, id = 0; i < docs.size(); i += dim, ++id) {
        float sim{0.0F};
        #pragma omp simd reduction(+:sim)
        for (std::size_t j = 0; j < dim; ++j) {
            sim += query[j] * docs[i + j];
        }
        float dist{1.0F - sim};
        if (topk.size() < k) {
            topk.push(std::make_pair(dist, id));
        } else if (topk.top().first > dist) {
            topk.pop();
            topk.push(std::make_pair(dist, id));
        }
    }

    return extractIdsAndScores(std::move(topk));
}
