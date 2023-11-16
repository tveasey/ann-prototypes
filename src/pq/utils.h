#pragma once

#include "../common/bigvector.h"

#include <cstdint>
#include <filesystem>
#include <random>
#include <vector>

// Load vector data from source zero pad and normalize if necessary.
void loadAndPrepareData(std::filesystem::path source, bool normalize);

// Zero pad the vectors so their dimension is a multiple of NUM_BOOKS.
void zeroPad(std::size_t dim, std::vector<float>& vectors);

// Sample documents from docs uniformly at random specified probability.
std::vector<float> sampleDocs(const BigVector& docs,
                              double sampleProbability,
                              std::minstd_rand& rng);
