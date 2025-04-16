#pragma once

#include "common.h"

#include <cstddef>

KMeansResult kMeans(std::size_t dim,
                    const Dataset& dataset,
                    std::size_t sampleSize,
                    Centers initialCenters,
                    std::size_t maxIterations = 8);