#pragma once

#include <cstddef>
#include <filesystem>
#include <functional>
#include <vector>

using TGenerator = std::function<std::vector<float> ()>;

std::pair<std::vector<float>, std::size_t> readFvecs(const std::filesystem::path& file);

std::size_t writeFvecs(const std::filesystem::path& file,
                       int dim,
                       std::size_t numVecs,
                       TGenerator generator);
