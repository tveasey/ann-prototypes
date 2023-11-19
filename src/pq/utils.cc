#include "utils.h"

#include "constants.h"
#include "../common/bigvector.h"
#include "../common/utils.h"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <random>

std::size_t zeroPad(std::size_t dim, std::vector<float>& vectors) {
    if (dim % NUM_BOOKS != 0) {
        std::size_t numVectors{vectors.size() / dim};
        std::size_t paddedDim{NUM_BOOKS * ((dim + NUM_BOOKS - 1) / NUM_BOOKS)};
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

BigVector loadAndPrepareData(const std::filesystem::path& source, bool normalized) {

    // A temporary file for storing the data.
    char filename[] = "/tmp/big_vector_storage_XXXXXX";
    int ret{::mkstemp(filename)};
    if (ret == -1) {
        throw std::runtime_error("Couldn't create temporary file.");
    }
    std::cout << "Created temporary file " << filename << std::endl;

    return {source, filename,
            [normalized](std::size_t dim_, std::vector<float>& docs) {
                if (normalized) {
                    normalize(dim_, docs);
                }
                return zeroPad(dim_, docs);
           }};
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
            readers.emplace_back([dim, &beginSampledDoc](std::size_t pos,
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
        auto& sampler = samplers.back();
        sampleReaders.emplace_back([&sampler](std::size_t,
                                              BigVector::VectorReference doc) {
            sampler.add(doc.data());
        });
    }

    parallelRead(docs, sampleReaders);

    sampledDocs.erase(std::remove_if(sampledDocs.begin(), sampledDocs.end(),
                                     [](float f) { return std::isnan(f); }),
                      sampledDocs.end());
    sampledDocs.resize(sampleSize * dim);

    return sampledDocs;
}