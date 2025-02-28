#include "utils.h"

#include "constants.h"
#include "../common/bigvector.h"
#include "../common/utils.h"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <limits>
#include <random>

std::size_t zeroPad(std::size_t dim,
                    std::size_t numBooks,
                    std::vector<float>& vectors) {
    if (dim % numBooks != 0) {
        std::size_t numVectors{vectors.size() / dim};
        std::size_t paddedDim{numBooks * ((dim + numBooks - 1) / numBooks)};
        vectors.resize(paddedDim * numVectors, 0.0F);
        for (std::size_t i = numVectors; i > 1; --i) {
            std::copy(&vectors[dim * (i - 1)], &vectors[dim * i],
                      &vectors[paddedDim * (i - 1)]);
        }
        std::fill(&vectors[dim], &vectors[paddedDim], 0.0F);
        return paddedDim;
    }
    return dim;
}

std::filesystem::path createBigVectorStorage() {
    // Create a temporary file.
    char filename[] = "/tmp/big_vector_storage_XXXXXX";
    int ret{::mkstemp(filename)};
    if (ret == -1) {
        throw std::runtime_error("Couldn't create temporary file.");
    }
    std::cout << "Created temporary file " << filename << std::endl;
    return std::filesystem::path{filename};
}

BigVector loadAndPrepareData(const std::filesystem::path& source,
                             std::size_t numBooks,
                             bool normalized,
                             const std::pair<double, double>& range) {

    return {source, createBigVectorStorage(),
            [numBooks, normalized](std::size_t dim_, std::vector<float>& docs) {
                if (normalized) {
                    normalize(dim_, docs);
                }
                return zeroPad(dim_, numBooks, docs);
           }, range};
}

std::vector<float>
sampleDocs(const BigVector& docs, std::size_t sampleSize, std::minstd_rand rng) {

    std::size_t dim{docs.dim()};

    if (sampleSize == 0) {
        return {};
    } else if (sampleSize >= docs.numVectors()) {
        std::vector<float> sampledDocs(docs.size());
        auto beginSampledDoc = sampledDocs.begin();
        std::vector<Reader> readers;
        readers.reserve(NUM_READERS);
        for (std::size_t i = 0; i < NUM_READERS; ++i) {
            readers.emplace_back([dim, beginSampledDoc](std::size_t pos,
                                                        BigVector::VectorReference doc) {
                std::copy(doc.data(), doc.data() + dim, beginSampledDoc + pos * dim);
            });
        }
        parallelRead(docs, readers);
        return sampledDocs;
    }

    // Sample using reservoir sampling.

    std::size_t numSamplesPerReader{(sampleSize + NUM_READERS - 1) / NUM_READERS};

    std::vector<float> sampledDocs(numSamplesPerReader * NUM_READERS * dim,
                                   std::numeric_limits<float>::quiet_NaN());
    auto beginSampledDoc = sampledDocs.begin();

    std::vector<std::minstd_rand> rngs(NUM_READERS, rng);
    std::vector<ReservoirSampler> samplers;
    std::vector<Reader> sampleReaders;
    samplers.reserve(NUM_READERS);
    sampleReaders.reserve(NUM_READERS);

    for (std::size_t i = 0; i < NUM_READERS; ++i) {
        auto storage = beginSampledDoc + i * numSamplesPerReader * dim;
        samplers.emplace_back(dim, numSamplesPerReader, rngs[i], storage);
        sampleReaders.emplace_back([i, &samplers](std::size_t,
                                                  BigVector::VectorReference doc) {
            samplers[i].add(doc.data());
        });
    }

    parallelRead(docs, sampleReaders);

    sampledDocs.erase(std::remove_if(sampledDocs.begin(), sampledDocs.end(),
                                     [](float f) { return std::isnan(f); }),
                      sampledDocs.end());
    sampledDocs.resize(sampleSize * dim);

    return sampledDocs;
}