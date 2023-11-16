#include "utils.h"

namespace {
const std::array<std::string, 2> METRICS{"dot", "cosine"}; 
const std::array<std::string, 3> BITS{"4 bit", "4 bit packed", "8 bit"}; 

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
        #pragma clang loop unroll_count(8) vectorize(assume_safety)
        for (std::size_t j = 0; j < dim; ++j) {
            norm += vectors[i + j] * vectors[i + j];
        }
        norm = std::sqrtf(norm);
        #pragma clang loop unroll_count(8) vectorize(assume_safety)
        for (std::size_t j = 0; j < dim; ++j) {
            vectors[i + j] /= norm;
        }
    }
}

std::vector<float> norms2(std::size_t dim, std::vector<float>& vectors) {
    std::vector<float> norms2(vectors.size() / dim);
    for (std::size_t i = 0, j = 0; i < vectors.size(); i += dim, ++j) {
        float norm2{0.0F};
        #pragma clang loop unroll_count(8) vectorize(assume_safety)
        for (std::size_t j = 0; j < dim; ++j) {
            norm2 += vectors[i + j] * vectors[i + j];
        }
        norms2[j] = norm2;
    }
    return norms2;
}
