#include <filesystem>
#include <vector>

std::pair<std::vector<float>, std::size_t>
readVectors(const std::filesystem::path &source, bool verbose = false);
