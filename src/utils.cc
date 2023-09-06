#include "utils.h"

std::vector<std::size_t> uniformSamples(double sampleProbability,
                                        std::size_t n,
                                        std::minstd_rand& rng) {
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

std::vector<float> sampleDocs(double sampleProbability,
                              std::size_t dim,
                              const std::vector<float>& docs,
                              std::minstd_rand& rng) {
    auto samples = uniformSamples(sampleProbability, docs.size() / dim, rng);
    std::vector<float> sampledDocs(dim * samples.size());
    for (std::size_t i = 0; i < samples.size(); ++i) {
        std::size_t sample{samples[i]};
        std::copy_n(docs.begin() + dim * sample, dim, sampledDocs.begin() + dim * i);
    }
    return sampledDocs;
}

void normalise(std::size_t dim, std::vector<float>& vectors) {
    // Ensure vectors are unit (Euclidean) norm.
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
