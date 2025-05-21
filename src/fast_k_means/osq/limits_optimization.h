#pragma once

#include "utils.h"

#include <vector>

std::vector<Interval> optimizeQuantizationLimits(std::size_t dim,
                                               const std::vector<float>& x,
                                               std::size_t bits,
                                               float lambda = 0.1F);

std::pair<std::vector<Interval>, std::vector<int>>
optimizeQuantizationLimitsAndQuantizedVectors(std::size_t dim,
                                             const std::vector<float>& x,
                                             std::size_t bits,
                                             float lambda);