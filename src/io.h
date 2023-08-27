#pragma once

#include "storage.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

struct PQStats;

std::pair<std::vector<float>, std::size_t> readFvecs(const std::filesystem::path& file);

void writePQStats(const PQStats& stats);
