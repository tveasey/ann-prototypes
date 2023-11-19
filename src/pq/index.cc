#include "index.h"

#include "clustering.h"
#include "codebooks.h"
#include "constants.h"
#include "subspace.h"
#include "../common/bigvector.h"
#include "../common/progress_bar.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <ostream>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void centre(std::size_t dim,
            const float* centre,
            float* doc) {
    #pragma clang loop unroll_count(4) vectorize(assume_safety)
    for (std::size_t i = 0; i < dim; ++i) {
        doc[i] = doc[i] - centre[i];
    }
}

void transform(std::size_t dim,
               const std::vector<float>& transformation,
               const float* doc,
               float* projectedDoc) {
    std::fill_n(projectedDoc, dim, 0.0F);
    for (std::size_t i = 0; i < dim; ++i) {
        #pragma clang loop unroll_count(4) vectorize(assume_safety)
        for (std::size_t j = 0; j < dim; ++j) {
            projectedDoc[i] += transformation[i * dim + j] * doc[j];
        }
    }
}

void removeNans(std::vector<std::vector<float>>& samples) {
    for (auto& sample : samples) {
        auto end = std::remove_if(sample.begin(), sample.end(),
                                 [](float x) { return std::isnan(x); });
        sample.erase(end, sample.end());
    }
}

std::vector<Reader>
initializeSampleReaders(std::size_t dim,
                        std::size_t numClusters,
                        std::size_t numSamplesPerCluster,
                        const std::vector<cluster_t>& docsClusters,
                        std::vector<std::minstd_rand>& rngs,
                        std::vector<std::vector<float>>& samples,
                        std::vector<std::vector<ReservoirSampler>>& samplers) {

    // For each thread we write into a non-overlapping region of the
    // sample vector.

    std::size_t numReaders{samplers.size()};
    std::size_t numSamplesPerReader{numSamplesPerCluster / numReaders};

    for (std::size_t i = 0; i < numClusters; ++i) {
        for (std::size_t j = 0; j < numReaders; ++j) {
            auto storage = samples[i].begin() + numSamplesPerReader * j * dim;
            samplers[j].emplace_back(dim, numSamplesPerReader, rngs[j], storage);
        }
    }

    std::vector<Reader> readers;
    readers.reserve(numReaders);
    auto beginDocsClusters = docsClusters.begin();
    for (std::size_t i = 0; i < numReaders; ++i) {
        auto& sampler = samplers[i];
        readers.emplace_back(
            [=, &sampler](std::size_t pos, BigVector::VectorReference doc) {
                auto docCluster = *(beginDocsClusters + pos);
                sampler[docCluster].add(doc.data());
            });
    }

    return readers;
}

} // unnamed::

PqIndex::PqIndex(bool normalized,
                 std::size_t dim,
                 std::vector<float> clustersCentres,
                 std::vector<std::vector<float>> transformations,
                 std::vector<std::vector<float>> codebooksCentres,
                 std::vector<cluster_t> docsClusters,
                 std::vector<code_t> docsCodes)
    : normalized_{normalized}, dim_{dim},
      clustersCentres_(std::move(clustersCentres)),
      transformations_(std::move(transformations)),
      codebooksCentres_(std::move(codebooksCentres)),
      docsClusters_(std::move(docsClusters)),
      docsCodes_(std::move(docsCodes)) {

    this->buildNormsTables();
}

std::pair<std::vector<std::size_t>, std::vector<float>>
PqIndex::search(const std::vector<float>& query, std::size_t k) const {

    // Find the closest k document vector indices to the query vector
    // using similarity tables for each cluster's codebook.

    if (query.size() != dim_) {
        throw std::runtime_error{
            "Query vector has incorrect dimension " +
            std::to_string(query.size()) + " != " + std::to_string(dim_)};
    }

    std::priority_queue<std::pair<float, std::size_t>> topk;

    std::size_t numClusters{clustersCentres_.size() / dim_};
    std::size_t bookDim{dim_ / NUM_BOOKS};

    std::vector<std::vector<float>> simTables(numClusters);
    std::vector<float> clusterSims(numClusters);
    for (std::size_t cluster = 0; cluster < numClusters; ++cluster) {
        auto [clusterSim, simTable] = this->buildSimTable(cluster, query);
        simTables[cluster] = std::move(simTable);
        clusterSims[cluster] = clusterSim;
    }

    for (std::size_t id = 0; id < docsClusters_.size(); ++id) {
        std::size_t cluster{docsClusters_[id]};
        const auto* docCode = &docsCodes_[id * NUM_BOOKS];
        float dist{this->computeDist(clusterSims[cluster],
                                     simTables[cluster],
                                     normsTable_[cluster],
                                     docCode)};
        if (topk.size() < k) {
            topk.push(std::make_pair(dist, id));
        } else if (dist < topk.top().first) {
            topk.pop();
            topk.push(std::make_pair(dist, id));
        }
    }

    // Unpack the top-k and the distances into separate vectors.
    std::vector<std::size_t> topkIds(k);
    std::vector<float> topkDists(k);
    for (std::size_t i = k; i > 0; --i) {
        topkDists[i - 1] = topk.top().first;
        topkIds[i - 1] = topk.top().second;
        topk.pop();
    }

    return {std::move(topkIds), std::move(topkDists)};
}

std::vector<float> PqIndex::decode(std::size_t id) const {

    // Get the encoding of the document at position id.

    if (id >= docsClusters_.size()) {
        throw std::runtime_error{
            "Document id " + std::to_string(id) + " out of range"};
    }

    std::vector<float> result(dim_);

    // Read the nearest centre to the document from each codebook.
    const auto* docCode = &docsCodes_[id * NUM_BOOKS];
    const auto* docCluster = &docsClusters_[id];
    const auto* codeBooks = &codebooksCentres_[*docCluster][0];    
    std::size_t bookDim{dim_ / NUM_BOOKS};
    for (std::size_t b = 0; b < NUM_BOOKS; ++b) {
        const auto* centroid = &codeBooks[(b * BOOK_SIZE + docCode[b]) * bookDim];
        std::copy(centroid, centroid + bookDim, &result[b * bookDim]);
    }

    // Map back into the original coordinate system. We use the fact that
    // the transformation matrix is orthogonal so its inverse is its transpose.
    std::vector<float> projectedResult(dim_, 0.0F);
    const auto* transformation = &transformations_[*docCluster][0];
    for (std::size_t i = 0; i < dim_; ++i) {
        for (std::size_t j = 0; j < dim_; ++j) {
            projectedResult[i] += transformation[j * dim_ + i] * result[j];
        }
    }
    result = std::move(projectedResult);

    // Add on the cluster centre.
    const auto* clusterCentre = &clustersCentres_[*docCluster * dim_];
    for (std::size_t i = 0; i < dim_; ++i) {
        result[i] += clusterCentre[i];
    }

    // If the vectors are normalized then we need to normalize the result.
    if (normalized_) {
        float norm{std::sqrtf(
            std::accumulate(result.begin(), result.end(), 0.0F))};
        for (std::size_t i = 0; i < dim_; ++i) {
            result[i] /= norm;
        }
    }

    return result;
}

double PqIndex::compressionRatio() const {

    // This is the size of the raw data in bytes divided by the size of the
    // index in bytes.

    std::size_t sizeOfClusters{clustersCentres_.size() * sizeof(float)};
    std::size_t sizeOfTransformations{std::accumulate(
        transformations_.begin(), transformations_.end(), 0UL,
        [](std::size_t sum, const auto& t) {
            return sum + t.size() * sizeof(float);
        })};
    std::size_t sizeOfCodebookCentres{std::accumulate(
        codebooksCentres_.begin(), codebooksCentres_.end(), 0UL,
        [](std::size_t sum, const auto& c) {
            return sum + c.size() * sizeof(float);
        })};
    std::size_t sizeOfClusterIds{docsClusters_.size() * sizeof(cluster_t)};
    std::size_t sizeOfCodes{docsCodes_.size() * sizeof(code_t)};
    std::size_t sizeOfNormsTable{std::accumulate(
        normsTable_.begin(), normsTable_.end(), 0UL,
        [](std::size_t sum, const auto& t) {
            return sum + t.size() * sizeof(float);
        })};

    // The raw data is the number of vectors times the dimension times the
    // size of a float.
    std::size_t sizeOfRawData{docsClusters_.size() * dim_ * sizeof(float)};

    return static_cast<double>(sizeOfRawData) /
           static_cast<double>(
               sizeOfClusters + sizeOfTransformations + sizeOfCodebookCentres +
               sizeOfClusterIds + sizeOfCodes + sizeOfNormsTable);
}

void PqIndex::buildNormsTables() {

    // Build a table of the norms of each codebook centre and the dot product
    // of each cluster centre with each codebook centre.

    std::size_t numClusters{clustersCentres_.size() / dim_};
    std::size_t bookDim{dim_ / NUM_BOOKS};

    normsTable_.resize(numClusters, std::vector<float>(2 * BOOK_SIZE * NUM_BOOKS));

    if (normalized_) {
        std::vector<float> projectedCentre(dim_);
        for (std::size_t cluster = 0; cluster < numClusters; ++cluster) {
            auto& normsTable = normsTable_[cluster];
            const auto& codebooksCentres = codebooksCentres_[cluster];
            const auto* clusterCentre = &clustersCentres_[cluster * dim_];
            transform(dim_, transformations_[cluster], clusterCentre, projectedCentre.data());
            for (std::size_t b = 0; b < NUM_BOOKS; ++b) {
                const auto* clusterCentreProj = &projectedCentre[b * bookDim];
                const auto* codebookCentres = &codebooksCentres[b * BOOK_SIZE * bookDim];
                for (std::size_t i = 0; i < BOOK_SIZE; ++i) {
                    float sim{0.0F};
                    float norm2{0.0F};
                    const auto* codebookCentre = codebookCentres + i * bookDim;
                    #pragma clang loop unroll_count(4) vectorize(assume_safety)
                    for (std::size_t j = 0; j < bookDim; ++j) {
                        float cij{codebookCentre[j]};
                        sim += clusterCentreProj[j] * cij;
                        norm2 += cij * cij;
                    }
                    auto t = normsTable.begin() + 2 * (BOOK_SIZE * b + i);
                    t[0] = sim;
                    t[1] = norm2;
                }
            }
        }
    }
}

std::pair<float, std::vector<float>>
PqIndex::buildSimTable(std::size_t cluster, 
                       const std::vector<float> &query) const {

    // Compute the dot product distance from the query to each centre
    // in the codebook.

    std::size_t dim{query.size()};
    std::size_t bookDim{dim / NUM_BOOKS};

    // Transform the query vector into the codebook's coordinate system.
    std::vector<float> projectedQuery(dim, 0.0F);
    transform(dim, transformations_[cluster], query.data(), projectedQuery.data());

    // Compute the dot product distance from the query to each centre in the
    // codebook.
    std::vector<float> simTable(BOOK_SIZE * NUM_BOOKS);
    const auto& codebooksCentres = codebooksCentres_[cluster];
    for (std::size_t b = 0; b < NUM_BOOKS; ++b) {
        const auto* queryProj = &projectedQuery[b * bookDim];
        const auto* codebookCentres = &codebooksCentres[b * BOOK_SIZE * bookDim];
        for (std::size_t i = 0; i < BOOK_SIZE; ++i) {
            float sim{0.0F};
            const auto* codebookCentre = codebookCentres + i * bookDim;
            #pragma clang loop unroll_count(4) vectorize(assume_safety)
            for (std::size_t j = 0; j < bookDim; ++j) {
                sim += queryProj[j] * codebookCentres[j];
            }
            simTable[BOOK_SIZE * b + i] = sim;
        }
    }

    // Compute the dot product distance from the query to cluster centre.
    float sim{0.0F};
    const auto* clusterCentres = &clustersCentres_[cluster];
    for (std::size_t i = 0; i < dim; ++i) {
        sim += query[i] * clusterCentres[i];
    }

    return {sim, std::move(simTable)};
}

float PqIndex::computeDist(float centreSim,
                           const std::vector<float>& simTable,
                           const std::vector<float>& normsTable,
                           const code_t* docCodes) const {

    // We compute the distance as 1 - the dot product similarity. The dot
    // product similarity between the query vector q and the i'th codebook
    // centre is q^t (c + r) = q^t c + q^t r, i.e. it separates into the
    // sum of the similarity between the query and the centre and the
    // similarity between the query and the residual. We can look up the
    // similarity between the query and the nearest centre to the residual
    // in the simTable.
    float sim{centreSim};
    #pragma clang loop unroll_count(8)
    for (std::size_t b = 0; b < NUM_BOOKS; ++b) {
        sim += simTable[BOOK_SIZE * b + docCodes[b]];
    }

    // If the index is normalized then we need to normalize the distance
    // by the norm of the query vector. The norm of the document vector
    // is |c + r| = (|c|^2 + 2 c^t r + |r|^2)^(1/2). By construction the
    // centres are normalized so this simplifies to (1 + 2 c^t r + |r|)^(1/2).
    // We can look up the dot product between the centre and the residual
    // and the norm of the residual in the normsTable.
    if (normalized_) {
        float cdotr{0.0F};
        float rnorm2{0.0F};
        for (std::size_t b = 0; b < NUM_BOOKS; ++b) {
            const auto* t = &normsTable[2 * (BOOK_SIZE * b + docCodes[b])];
            cdotr += t[0];
            rnorm2 += t[1];
        }
        sim /= std::sqrtf(1.0F + 2.0F * cdotr + rnorm2);
    }

    return 1.0F - sim;
}

std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>
buildCodebooksForPqIndex(const BigVector& docs,
                         const std::vector<float>& clustersCentres,
                         const std::vector<cluster_t>& docsClusters) {

    std::size_t dim{docs.dim()};
    std::size_t numDocs{docs.numVectors()};
    std::size_t numClusters{clustersCentres.size() / dim};
    std::vector<std::minstd_rand> rngs(NUM_READERS);
    std::cout << "Building codebooks for " << numClusters << " clusters" << std::endl;

    std::vector<std::vector<float>> transformations;
    transformations.reserve(numClusters);

    // We use a codeblock to ensure the samples are destroyed before we
    // move the next step. This reduces the peak memory usage.
    {
        std::cout << "Computing optimal transforms" << std::endl;

        // Use a random sample of 64 docs per dimension.
        std::vector<float> initialSamples(64 * dim * dim,
                                          std::numeric_limits<float>::quiet_NaN());
        std::vector<std::vector<float>> samples(numClusters, initialSamples);

        std::cout << "Sampling " << 64 * dim << " docs per cluster" << std::endl;
        std::vector<std::vector<ReservoirSampler>> samplers(NUM_READERS);
        auto sampleReaders = initializeSampleReaders(
            dim, numClusters, 64 * dim, docsClusters, rngs, samples, samplers);
        parallelRead(docs, sampleReaders);
        removeNans(samples);

        // Compute optimal transformations for each coarse cluster.
        for (auto& sample : samples) {
            auto [eigVecs, eigVals] = pca(dim, std::move(sample));
            transformations.emplace_back(
                computeOptimalPQSubspaces(dim, eigVecs, eigVals));
        }
    }

    // Prepare the data set for computing code book centres.
    std::vector<std::vector<float>> projectedSamples(numClusters);

    // We use a codeblock to ensure the samplers are destroyed before
    // we move the next step. This reduces the peak memory usage.
    {
        std::cout << "Sampling data to build codebooks" << std::endl;

        // Use a random sample of 128 docs per cluster.
        std::vector<float> initialSamples(128 * BOOK_SIZE * dim,
                                          std::numeric_limits<float>::quiet_NaN());
        std::vector<std::vector<float>> samples(numClusters, initialSamples);

        std::cout << "Sampling " << 128 * BOOK_SIZE << " docs per cluster" << std::endl;
        std::vector<std::vector<ReservoirSampler>> samplers(NUM_READERS);
        auto sampleReaders = initializeSampleReaders(
            dim, numClusters, 128 * BOOK_SIZE, docsClusters, rngs, samples, samplers);
        parallelRead(docs, sampleReaders);
        removeNans(samples);

        // Compute the residuals from the coarse cluster centres and transform
        // them into the optimal subspaces.
        for (std::size_t i = 0; i < numClusters; ++i) {
            auto& transformation = transformations[i];
            auto& projectedSample = projectedSamples[i];
            projectedSample = std::move(samples[i]);
            for (std::size_t j = 0; j < projectedSample.size(); j += dim) {
                centre(dim, &clustersCentres[i * dim], &projectedSample[j]);
            }
            projectedSample = transform(transformation, dim, std::move(projectedSample));
        }
    }

    // Compute the codebook centres for each coarse cluster.
    std::vector<std::vector<float>> codebooksCentres(numClusters);
    std::cout << "Computing optimal transformations" << std::endl;
    ProgressBar progress{numClusters};
    for (std::size_t i = 0; i < numClusters; ++i) {
        codebooksCentres[i] = buildCodebook(dim, projectedSamples[i]).first;
        progress.update();
    }

    return {std::move(transformations), std::move(codebooksCentres)};
}

PqIndex buildPqIndex(const BigVector& docs, bool normalized, float distanceThreshold) {

    std::size_t dim{docs.dim()};
    std::size_t numDocs{docs.numVectors()};

    std::vector<float> clustersCentres;
    std::vector<cluster_t> docsClusters;
    coarseClustering(normalized, docs, clustersCentres, docsClusters);

    std::vector<std::vector<float>> transformations;
    std::vector<std::vector<float>> codebooksCentres;
    std::tie(transformations, codebooksCentres) = 
        buildCodebooksForPqIndex(docs, clustersCentres, docsClusters);

    // Compute the codes for each doc.

    std::vector<Reader> encoders;
    encoders.reserve(NUM_READERS);

    std::vector<code_t> docsCodes(numDocs * NUM_BOOKS);
    auto beginDocCodes = docsCodes.data();
    auto beginDocCluster = docsClusters.begin();
    std::vector<float> centredDoc(dim);
    std::vector<float> projectedDoc(dim);
    for (std::size_t i = 0; i < NUM_READERS; ++i) {
        encoders.emplace_back([=, &clustersCentres, &transformations](
                std::size_t pos,
                BigVector::VectorReference doc) mutable {
            auto docCluster = *(beginDocCluster + pos);
            auto docCodes = beginDocCodes + NUM_BOOKS * pos;
            centredDoc.assign(doc.data(), doc.data() + dim);
            centre(dim, &clustersCentres[docCluster], centredDoc.data());
            transform(dim, transformations[docCluster], centredDoc.data(), projectedDoc.data());
            if (distanceThreshold > 0.0F) {
                anisotropicEncode(projectedDoc, codebooksCentres[docCluster],
                                  distanceThreshold, docCodes);
            } else {
                encode(projectedDoc, codebooksCentres[docCluster], docCodes);
            }
        });
    }

    parallelRead(docs, encoders);

    return {normalized, dim, std::move(clustersCentres), std::move(transformations),
            std::move(codebooksCentres), std::move(docsClusters), std::move(docsCodes)};
}
