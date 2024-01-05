#pragma once

#include "bigvector.h"

#include <cstddef>
#include <vector>

float dotf(std::size_t dim, const float* x, const float* y);

std::pair<std::vector<std::size_t>, std::vector<float>>
searchBruteForce(std::size_t k,
                 const BigVector& docs,
                 const std::vector<float>& query);

std::pair<std::vector<std::size_t>, std::vector<float>>
searchBruteForce(std::size_t k,
                 const std::vector<float>& docs,
                 const std::vector<float>& query);
