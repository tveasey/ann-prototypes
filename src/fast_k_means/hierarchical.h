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
                                            std::size_t maxIterations = 300,
                                            std::size_t depth = 0);
