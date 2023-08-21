#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cmath>
#include <cstdlib>

std::size_t readDimension(const std::filesystem::path &source) {

    std::ifstream input{source};
    if (!input.is_open()) {
        std::cout << "File " << source << " not found" << std::endl;
        return 0;
    }

    std::string buffer;
    std::getline(input, buffer);
    try {
        return std::stoi(buffer);
    } catch (const std::invalid_argument& e) {
        std::cout << "Failed to read dimension: " << e.what() << std::endl;
    } catch (const std::out_of_range& e) {
        std::cout << "Failed to read dimension: " << e.what() << std::endl;
    }
    std::abort();
}

std::vector<float> readVectors(std::size_t dim,
                               const std::filesystem::path &source,
                               bool verbose) {
    if (verbose) {
        std::cout << "Reading " << source << "\n" << std::endl;
    }
    
    std::vector<float> result;
    result.reserve(10 * 1024 * 1024);

    std::ifstream input{source};
    if (!input.is_open()) {
        std::cout << "File " << source << " not found" << std::endl;
        return result;
    }

    std::string buffer;
    for (std::size_t i = 0; std::getline(input, buffer); /**/) {
        std::size_t rowDim{0};
        for (std::size_t i = 0, step = 0; i < buffer.size(); i += step + 1, ++rowDim) {
            step = std::min(buffer.find(',', i), buffer.size()) - i;
            try {
                result.push_back(std::stof(buffer.substr(i, step)));
                continue;
            } catch (const std::invalid_argument& e) {
                std::cout << "Failed to read value: " << e.what() << std::endl;
            } catch (const std::out_of_range& e) {
                std::cout << "Failed to read value: " << e.what() << std::endl;
            }
            std::abort();
        }
        if (rowDim != dim) {
            std::cout << "Row missing values " << rowDim << " != " << dim << std::endl;
            std::abort();
        }
        if (verbose && (++i % 10000 == 0)) {
            std::cout << "\rProcessed " << i << " lines" << std::flush;
        }
    }

    if (verbose) {
        std::cout << "\rRead " << result.size()
                  << " vectors       " << std::endl;
    }

    result.shrink_to_fit();

    return result;
}
