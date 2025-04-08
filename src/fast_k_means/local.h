#pragma once

#include "common.h"

#include <cstddef>
#include <vector>

KMeansResult kMeansLocal(std::size_t dim,
                         const Dataset& dataset,
                         Centers centers,
                         std::vector<std::size_t> assignments,
                         std::size_t clustersPerNeighborhood = 32,
                         std::size_t maxIterations = 8,
                         float lambda = 1.0F);
