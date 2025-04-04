#pragma once

#include "common.h"

#include <cstddef>

KMeansResult kMeans(std::size_t dim,
                    const Dataset& dataset,
                    Centers initial_centers,
                    std::size_t k,
                    std::size_t max_iterations = 300);