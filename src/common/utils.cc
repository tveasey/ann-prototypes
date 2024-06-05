#include "utils.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>

namespace {
const std::array<std::string, 3> METRICS{"cosine", "dot", "euclidean"};
const std::array<std::string, 4> BITS{"1 bit", "4 bit", "4 bit packed", "8 bit"};

std::vector<std::size_t> uniformSamples(std::size_t n,
                                        double sampleProbability,
                                        std::minstd_rand& rng) {
    if (sampleProbability <= 0.0) {
        return {};
    }

    std::vector<std::size_t> samples;
    if (sampleProbability < 1.0) {
        samples.reserve(static_cast<std::size_t>(1.1 * n * sampleProbability));
        std::geometric_distribution<> geom{sampleProbability};
        for (std::size_t i = geom(rng); i < n; i += 1 + geom(rng)) {
            samples.push_back(i);
        }
    } else {
        samples.resize(n);
        std::iota(samples.begin(), samples.end(), 0);
    }

    return samples;
}

} // unnamed::

const std::string& toString(Metric m) {
    return METRICS[m];
}

const std::string& toString(ScalarBits b) {
    return BITS[b];
}

std::vector<float> sampleDocs(std::size_t dim,
                              const std::vector<float>& docs,
                              double sampleProbability,
                              std::minstd_rand& rng) {
    std::size_t numDocs{docs.size() / dim};
    auto samples = uniformSamples(numDocs, sampleProbability, rng);
    std::vector<float> sampledDocs(dim * samples.size());
    for (std::size_t i = 0; i < samples.size(); ++i) {
        std::size_t sample{samples[i]};
        std::copy_n(docs.begin() + dim * sample, dim, sampledDocs.begin() + dim * i);
    }
    return sampledDocs;
}

void normalize(std::size_t dim, std::vector<float>& vectors) {
    for (std::size_t i = 0; i < vectors.size(); i += dim) {
        float norm{0.0F};
        #pragma omp simd reduction(+:norm)
        for (std::size_t j = 0; j < dim; ++j) {
            norm += vectors[i + j] * vectors[i + j];
        }
        norm = std::sqrtf(norm);
        #pragma omp simd
        for (std::size_t j = 0; j < dim; ++j) {
            vectors[i + j] /= norm;
        }
    }
}

std::vector<float> norms2(std::size_t dim, const std::vector<float>& vectors) {
    std::vector<float> norms2(vectors.size() / dim);
    for (std::size_t i = 0, j = 0; i < vectors.size(); i += dim, ++j) {
        float norm2{0.0F};
        #pragma omp simd reduction(+:norm2)
        for (std::size_t j = 0; j < dim; ++j) {
            norm2 += vectors[i + j] * vectors[i + j];
        }
        norms2[j] = norm2;
    }
    return norms2;
}

Timer::Timer(const std::string& operation,
             std::chrono::duration<double>& duration) :
    operation_{operation},
    duration_{duration},
    start_{std::chrono::steady_clock::now()} {
}

Timer::~Timer() {
    std::chrono::steady_clock::time_point end;
    end = std::chrono::steady_clock::now();
    duration_ = end - start_;
    if (!operation_.empty()) {
        std::cout << operation_ << " took " << duration_.count() << " s" << std::endl;
    }
}

std::chrono::duration<double> time(std::function<void()> f,
                                   const std::string& operation) {
    std::chrono::duration<double> diff{0};
    {
        Timer timer{operation, diff};
        f();
    }
    return diff;
}
