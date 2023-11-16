#pragma once

#include "../common/bigvector.h"

#include <cstdint>
#include <random>
#include <vector>

// Zero pad the vectors so their dimension is a multiple of NUM_BOOKS.
void zeroPad(std::size_t dim, std::vector<float>& vectors);

// Sample documents from docs uniformly at random specified probability.
std::vector<float> sampleDocs(const BigVector& docs,
                              double sampleProbability,
                              std::minstd_rand& rng);
