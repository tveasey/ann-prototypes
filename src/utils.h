#pragma once

#include <cstddef>
#include <ostream>
#include <random>

template<typename U, typename V>
std::ostream& operator<<(std::ostream& o, const std::pair<U, V>& pair) {
    o << "(" << pair.first << "," << pair.second << ")";
    return o;
}

template<typename T>
std::ostream& operator<<(std::ostream& o, const std::vector<T>& vector) {
    if (vector.empty()) {
        o << "[]";
        return o;
    }

    std::size_t n{vector.size() - 1};

    o << "[";
    for (std::size_t i = 0; i < n; ++i) {
        o << vector[i] << ",";
    }
    o << vector[n] << "]";
    return o;
}

std::vector<std::size_t> uniformSamples(double sampleProbability,
                                        std::size_t n,
                                        std::minstd_rand& rng);

std::vector<float> sampleDocs(double sampleProbability,
                              std::size_t dim,
                              const std::vector<float>& docs,
                              std::minstd_rand& rng);

void normalise(std::size_t dim, std::vector<float>& vectors);

std::vector<float> norms2(std::size_t dim, std::vector<float>& vectors);
