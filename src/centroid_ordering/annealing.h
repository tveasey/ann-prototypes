#pragma once

#include "common.h"


PermutationCost annealingOrder(std::size_t dim,
                               const Points& x,
                               bool hilbertInitialization = false,
                               std::size_t k = 32,
                               std::size_t probes = 24);