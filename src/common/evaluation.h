#pragma once

#include <array>
#include <cstddef>
#include <vector>

constexpr std::array<std::size_t, 5> EXPANSIONS{1, 2, 4, 8, 10};
constexpr std::size_t MIN_RECALL{0};
constexpr std::size_t MAX_RECALL{1};
constexpr std::size_t AVG_RECALL{2};

using recalls_t = std::array<double, AVG_RECALL + 1>;

// Compute the recall of approximate vs exact retrieval.
double computeRecall(const std::vector<std::size_t>& nnTrue,
                     std::vector<std::size_t> nnApprox);

// Compute the min, mean and max recalls of approximate vs exact retrieval.
recalls_t computeRecalls(const std::vector<std::vector<std::size_t>>& nnExact,
                         const std::vector<std::vector<std::size_t>>& nnApprox);
