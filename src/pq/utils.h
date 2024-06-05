#pragma once

#include "../common/bigvector.h"

#include <cstdint>
#include <filesystem>
#include <random>
#include <vector>

// Zero pad the vectors so their dimension is a multiple of numBooks.
std::size_t zeroPad(std::size_t dim,
                    std::size_t numBooks,
                    std::vector<float>& vectors);

// Load vector data from source zero pad and normalize if necessary.
BigVector loadAndPrepareData(const std::filesystem::path& source,
                             std::size_t numBooks,
                             bool normalized);

// Sample documents from docs uniformly at random specified probability.
std::vector<float>
sampleDocs(const BigVector& docs, std::size_t sampleSize, std::minstd_rand rng);
