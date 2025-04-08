#pragma once

#include "common.h"

#include <cstddef>

std::size_t pickInitialCenters(std::size_t dim,
                               const Dataset& dataset,
                               std::size_t k,
                               Centers& centers);

HierarchicalKMeansResult kMeansHierarchical(std::size_t dim,
                                            const Dataset& dataset,
                                            std::size_t targetSize = 512,
                                            std::size_t maxIterations = 8,
                                            std::size_t maxK = 128,
                                            std::size_t samplesPerCluster = 256,
                                            std::size_t clustersPerNeighborhood = 32,
                                            std::size_t depth = 0);
