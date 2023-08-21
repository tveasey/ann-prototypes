#pragma once

#include "storage.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

struct PQStats;

std::size_t readDimension(const std::filesystem::path &source);

std::vector<float> readVectors(std::size_t dim,
                               const std::filesystem::path &source,
                               bool verbose = false);

void writePQStats(const PQStats& stats);