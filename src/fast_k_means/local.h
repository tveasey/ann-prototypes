#pragma once

#include "common.h"

#include <cstddef>
#include <vector>

KMeansResult kMeansLocal(std::size_t dim,
                         const Dataset& dataset,
                         std::size_t sampleSize, 
                         Centers centers,
                         std::vector<std::size_t> assignments,
                         std::size_t maxK = 128,
                         std::size_t maxIterations = 8);
