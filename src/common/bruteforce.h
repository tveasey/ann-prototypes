#pragma once

#include "bigvector.h"

#include <cstddef>
#include <queue>
#include <vector>

class TopK {
public:
    TopK(std::size_t k);

    void add(std::size_t id, float dist);
    std::pair<std::vector<std::size_t>, std::vector<float>> unpack();

private:
    std::size_t k_{0};
    std::priority_queue<std::pair<float, std::size_t>> topk_;
};

float dotf(std::size_t dim, const float* x, const float* y);

std::pair<std::vector<std::size_t>, std::vector<float>>
searchBruteForce(std::size_t k,
                 const BigVector& docs,
                 const std::vector<float>& query);

std::pair<std::vector<std::size_t>, std::vector<float>>
searchBruteForce(std::size_t k,
                 const std::vector<float>& docs,
                 const std::vector<float>& query);
