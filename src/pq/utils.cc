#include "utils.h"

#include "constants.h"
#include "../common/bigvector.h"

#include <algorithm>
#include <random>

void zeroPad(std::size_t dim, std::vector<float>& vectors) {
    if (dim % NUM_BOOKS != 0) {
        std::size_t numVectors{vectors.size() / dim};
        std::size_t paddedDim{NUM_BOOKS * ((dim + NUM_BOOKS - 1) / NUM_BOOKS)};
        vectors.resize(paddedDim * numVectors, 0.0F);
        for (std::size_t i = numVectors; i > 1; --i) {
            std::copy(&vectors[dim * (i - 1)], &vectors[dim * i],
                      &vectors[paddedDim * (i - 1)]);
        }
        std::fill(&vectors[dim], &vectors[paddedDim], 0.0F);
    }
}

std::vector<float> sampleDocs(const BigVector& docs,
                              double sampleProbability,
                              std::minstd_rand& rng) {

    std::size_t dim{docs.dim()};

    if (sampleProbability <= 0.0) {
        return {};
    } else if (sampleProbability >= 1.0) {
        std::vector<float> sampledDocs(docs.size());
        auto sampledDoc = sampledDocs.begin();
        for (auto doc : docs) {
            std::copy(doc.data(), doc.data() + dim, sampledDoc);
            sampledDoc += dim;
        }
        return sampledDocs;
    }

    auto sampleSize =
        static_cast<std::size_t>(sampleProbability * docs.numVectors());

    ReservoirSampler reservoirSampler{dim, sampleSize, rng};
    for (auto doc : docs) {
        reservoirSampler.add(doc.data());
    }
    return std::move(reservoirSampler.sample());
}