#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string_view>
#include <vector>

#include <cmath>
#include <cstdlib>

namespace {
constexpr std::size_t MB{1024 * 1024};

bool toFloat(const std::string str, float& value) {
    if (str.empty()) {
        std::cout << "Missing" << std::endl;
        return false;
    }

    char* endPtr{nullptr};
    errno = 0;
    value = std::strtof(str.c_str(), &endPtr);

    if ((value == 0.0F && errno == EINVAL) ||
         ((value == HUGE_VALF || value == -HUGE_VALF) && errno == ERANGE)) {
        std::cout << "Failed to convert '" << str
                  << "': " << std::strerror(errno) << std::endl;
        return false;
    }

    if (endPtr != nullptr && *endPtr != '\0') {
        std::cout << "Failed to convert '" << str
                  << "': first invalid character " << endPtr << std::endl;
        return false;
    }

    return true;
}
}

std::pair<std::vector<float>, std::size_t>
readVectors(std::filesystem::path &source, bool verbose) {
    if (verbose) {
        std::cout << "Reading " << source << std::endl;
    }
    
    std::vector<float> result;

    std::ifstream reader{source};
    if (!reader.is_open()) {
        std::cout << "File not found" << std::endl;
        return {result, 0};
    }

    result.reserve(10 * MB);

    std::string buffer;
    std::size_t dimension{0};
    for (std::size_t i = 0; std::getline(reader, buffer); /**/) {
        dimension = 0;
        bool failedToParse{false};
        for (std::size_t i = 0, step = 0; i < buffer.size(); i += step + 1, ++dimension) {
            step = std::min(buffer.find(',', i), buffer.size()) - i;
            float value;
            if (!toFloat(buffer.substr(i, step), value)) {
                failedToParse = true;
                break;
            }
            result.push_back(value);
        }
        if (failedToParse) {
            break;
        }
        if (verbose && ++i % 10000 == 0) {
            std::cout << "\rProcessed " << i << " lines" << std::flush;
        }
    }
    if (verbose) {
        std::cout << std::endl;
    }

    result.shrink_to_fit();

    return {std::move(result), dimension};
}