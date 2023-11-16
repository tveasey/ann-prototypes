#pragma once

#include <cstddef>
#include <filesystem>
#include <vector>

std::pair<std::vector<float>, std::size_t> readFvecs(const std::filesystem::path& file);
