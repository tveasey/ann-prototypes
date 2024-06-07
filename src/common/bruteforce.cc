#include "bruteforce.h"

#include "bigvector.h"

#include <cstddef>
#include <queue>
#include <utility>
#include <vector>
#include <iostream>

TopK::TopK(std::size_t k) : k_{k} {}

void TopK::add(std::size_t id, float dist) {
    if (topk_.size() < k_) {
        topk_.push(std::make_pair(dist, id));
    } else if (topk_.top().first > dist) {
        topk_.pop();
        topk_.push(std::make_pair(dist, id));
    }
}

std::pair<std::vector<std::size_t>, std::vector<float>> TopK::unpack() {
    std::vector<std::size_t> topkIds(topk_.size());
    std::vector<float> topkScores(topk_.size());
    for (std::size_t i = topk_.size(); i > 0; --i) {
        topkIds[i - 1] = topk_.top().second;
        topkScores[i - 1] = topk_.top().first;
        topk_.pop();
    }
    return {std::move(topkIds), std::move(topkScores)};
}

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

    TopK topk{k};

    std::size_t id{0};
    for (auto doc : docs) {
        float sim{0.0F};
        auto* data = doc.data();
        #pragma omp simd reduction(+:sim)
        for (std::size_t j = 0; j < dim; ++j) {
            sim += query[j] * data[j];
        }
        float dist{1.0F - sim};
        topk.add(id++, dist);
    }

    return topk.unpack();
}

std::pair<std::vector<std::size_t>, std::vector<float>>
searchBruteForce(std::size_t k,
                 const std::vector<float>& docs,
                 const std::vector<float>& query) {

    std::size_t dim{query.size()};
    if ((docs.size() % dim) != 0) {
        throw std::invalid_argument("query and docs have different dimensions");
    }

    TopK topk{k};

    for (std::size_t i = 0, id = 0; i < docs.size(); i += dim, ++id) {
        float sim{0.0F};
        #pragma omp simd reduction(+:sim)
        for (std::size_t j = 0; j < dim; ++j) {
            sim += query[j] * docs[i + j];
        }
        float dist{1.0F - sim};
        topk.add(id, dist);
    }

    return topk.unpack();
}
