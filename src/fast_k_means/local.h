#pragma once

#include "common.h"

#include <cstddef>
#include <vector>

HierarchicalKMeansResult kMeansLocal(std::size_t dim,
                                     const Dataset& dataset,
                                     std::vector<Centers> centers,
                                     std::vector<std::vector<std::size_t>> assignments,
                                     std::size_t clustersPerNeighborhood = 16,
                                     std::size_t maxIterations = 300);
