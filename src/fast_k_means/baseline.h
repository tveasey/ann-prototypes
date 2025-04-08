#pragma once

#include "common.h"

#include <cstddef>

KMeansResult kMeans(std::size_t dim,
                    const Dataset& dataset,
                    Centers initialCenters,
                    std::size_t k,
                    std::size_t maxIterations = 8);