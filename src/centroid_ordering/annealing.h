#pragma once

#include "common.h"

Permutation annealingOrder(std::size_t dim, 
                           const Points& x,
                           std::size_t k = 32,
                           std::size_t probes = 24);