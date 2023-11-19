#pragma once

#include "types.h"

#include <cstddef>
#include <ostream>
#include <random>

const std::string& toString(Metric m);

const std::string& toString(ScalarBits b);

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

std::vector<float> sampleDocs(std::size_t dim,
                              const std::vector<float>& docs,
                              double sampleProbability,
                              std::minstd_rand& rng);

void normalize(std::size_t dim, std::vector<float>& vectors);

std::vector<float> norms2(std::size_t dim, const std::vector<float>& vectors);
