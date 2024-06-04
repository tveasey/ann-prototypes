#include "index.h"

#include "clustering.h"
#include "codebooks.h"
#include "constants.h"
#include "subspace.h"
#include "../common/bigvector.h"
#include "../common/bruteforce.h"
#include "../common/progress_bar.h"
#include "../common/types.h"
#include "../common/utils.h"

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
    #pragma omp simd
    for (std::size_t i = 0; i < dim; ++i) {
        doc[i] -= centre[i];
    }
}

void transform(std::size_t dim,
               const std::vector<float>& transformation,
               const float* doc,
               float* transformedDoc) {
    std::fill_n(transformedDoc, dim, 0.0F);
    for (std::size_t i = 0; i < dim; ++i) {
        float sum{0.0F};
        #pragma omp simd reduction(+:sum)
        for (std::size_t j = 0; j < dim; ++j) {
            sum += transformation[i * dim + j] * doc[j];
        }
        transformedDoc[i] = sum;
    }
}

void removeNans(std::vector<std::vector<float>>& samples) {
    for (auto& sample : samples) {
        if (!sample.empty()) {
            auto end = std::remove_if(sample.begin(), sample.end(),
                                     [](float x) { return std::isnan(x); });
            sample.erase(end, sample.end());
        }
    }
}

std::vector<Reader>
initializeSampleReaders(std::size_t dim,
                        std::size_t beginClusters,
                        std::size_t endClusters,
                        std::size_t numSamplesPerCluster,
                        const std::vector<cluster_t>& docsClusters,
                        const std::minstd_rand& rng,
                        std::vector<std::vector<float>>& samples,
                        std::vector<std::vector<ReservoirSampler>>& samplers) {

    // For each thread we write into a non-overlapping region of the
    // sample vector.

    std::size_t numReaders{samplers.size()};
    std::size_t numSamplesPerReader{numSamplesPerCluster / numReaders};

    for (std::size_t i = 0; i < numReaders; ++i) {
        for (std::size_t j = beginClusters; j < endClusters; ++j) {
            auto storage = samples[j].begin() + numSamplesPerReader * i * dim;
            samplers[i][j] = ReservoirSampler{dim, numSamplesPerReader, rng, storage};
        }
    }

    std::vector<Reader> readers;
    readers.reserve(numReaders);
    auto beginDocsClusters = docsClusters.begin();
    for (std::size_t i = 0; i < numReaders; ++i) {
        readers.emplace_back(
            [=, &docsClusters, &samplers](std::size_t pos, BigVector::VectorReference doc) {
                auto docCluster = *(beginDocsClusters + pos);
                if (docCluster >= beginClusters && docCluster < endClusters) {
                    samplers[i][docCluster].add(doc.data());
                }
            });
    }

    return readers;
}

} // unnamed::

PqIndex::PqIndex(Metric metric,
                 std::size_t dim,
                 std::vector<float> clustersCentres,
                 std::vector<std::vector<float>> transformations,
                 std::vector<std::vector<float>> codebooksCentres,
                 std::vector<cluster_t> docsClusters,
                 std::vector<code_t> docsCodes)
    : metric_{metric}, dim_{dim},
      clustersCentres_(std::move(clustersCentres)),
      transformations_(std::move(transformations)),
      codebooksCentres_(std::move(codebooksCentres)),
      docsClusters_(std::move(docsClusters)),
      docsCodes_(std::move(docsCodes)) {

    this->buildNormsTables();
}

std::size_t PqIndex::numCodebooksCentres() const {
    std::size_t bookDim{dim_ / NUM_BOOKS};
    return std::accumulate(codebooksCentres_.begin(),
                           codebooksCentres_.end(), 0,
                           [](std::size_t sum, const auto& codebook) {
                               return sum + codebook.size();
                           }) / bookDim;
}

std::vector<code_t> PqIndex::codes(std::size_t id) const {
    return {docsCodes_.begin() + id * NUM_BOOKS,
            docsCodes_.begin() + (id + 1) * NUM_BOOKS};
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

    std::size_t numClusters{clustersCentres_.size() / dim_};
    std::size_t bookDim{dim_ / NUM_BOOKS};

    std::vector<std::vector<float>> simTables(numClusters);
    std::vector<float> clusterSims(numClusters);
    for (std::size_t cluster = 0; cluster < numClusters; ++cluster) {
        auto [clusterSim, simTable] = this->buildSimTable(cluster, query);
        simTables[cluster] = std::move(simTable);
        clusterSims[cluster] = clusterSim;
    }

    TopK topk{k};
    for (std::size_t id = 0; id < docsClusters_.size(); ++id) {
        std::size_t cluster{docsClusters_[id]};
        const auto* docCode = &docsCodes_[id * NUM_BOOKS];
        float dist{this->computeDist(clusterSims[cluster],
                                     simTables[cluster],
                                     normsTable_[cluster],
                                     docCode)};
        topk.add(id, dist);
    }

    return topk.unpack();
}

std::vector<float> PqIndex::decode(std::size_t id) const {

    // Get the encoding of the document at position id.

    if (id >= docsClusters_.size()) {
        throw std::runtime_error{
            "Document id " + std::to_string(id) + " out of range"};
    }

    // Read the nearest centre to the document from each codebook.
    auto docCluster = docsClusters_[id];
    const auto* docCode = &docsCodes_[id * NUM_BOOKS];
    const auto* codebooks = &codebooksCentres_[docCluster][0];
    std::size_t bookDim{dim_ / NUM_BOOKS};
    std::vector<float> transformedResult(dim_, 0.0F);
    for (std::size_t b = 0; b < NUM_BOOKS; ++b) {
        const auto* centre = &codebooks[(b * BOOK_SIZE + docCode[b]) * bookDim];
        std::copy(centre, centre + bookDim, &transformedResult[b * bookDim]);
    }

    // Transform back into the original coordinate system. We use the fact
    // that the transformation matrix is orthogonal so its inverse is its
    // transpose.
    std::vector<float> result(dim_, 0.0F);
    const auto& transformation = transformations_[docCluster];
    for (std::size_t i = 0; i < dim_; ++i) {
        for (std::size_t j = 0; j < dim_; ++j) {
            result[i] += transformation[j * dim_ + i] * transformedResult[j];
        }
    }

    // Add on the cluster centre.
    const auto* clusterCentre = &clustersCentres_[docCluster * dim_];
    for (std::size_t i = 0; i < dim_; ++i) {
        result[i] += clusterCentre[i];
    }

    // If the vectors are normalized then we need to normalize the result.
    if (metric_ == Cosine) {
        normalize(dim_, result);
    }

    return result;
}

float PqIndex::computeDist(const std::vector<float>& query, std::size_t id) const {

    // Compute the dot product distance between the query vector and the
    // document vector with the given id.

    if (id >= docsClusters_.size()) {
        throw std::runtime_error{
            "Document id " + std::to_string(id) + " out of range"};
    }

    std::size_t cluster{docsClusters_[id]};
    const auto* docCode = &docsCodes_[id * NUM_BOOKS];
    auto [clusterSim, simTable] = this->buildSimTable(cluster, query);
    return this->computeDist(clusterSim, simTable, normsTable_[cluster], docCode);
}

double PqIndex::vectorCompressionRatio() const {
    return static_cast<double>(dim_ * sizeof(float)) /
           static_cast<double>(NUM_BOOKS * sizeof(code_t));
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

    normsTable_.resize(numClusters);

    if (metric_ == Cosine) {
        std::fill_n(normsTable_.begin(), numClusters,
                    std::vector<float>(2 * NUM_BOOKS * BOOK_SIZE));
        std::vector<float> transformedCentre(dim_);
        for (std::size_t cluster = 0; cluster < numClusters; ++cluster) {
            auto& normsTable = normsTable_[cluster];
            const auto& codebooksCentres = codebooksCentres_[cluster];
            const auto* clusterCentre = &clustersCentres_[cluster * dim_];
            transform(dim_, transformations_[cluster], clusterCentre, transformedCentre.data());
            for (std::size_t b = 0; b < NUM_BOOKS; ++b) {
                const auto* clusterCentreProj = &transformedCentre[b * bookDim];
                const auto* codebookCentres = &codebooksCentres[b * BOOK_SIZE * bookDim];
                for (std::size_t i = 0; i < BOOK_SIZE; ++i) {
                    float sim{0.0F};
                    float norm2{0.0F};
                    const auto* codebookCentre = codebookCentres + i * bookDim;
                    #pragma omp simd reduction(+:sim,norm2)
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

    // Compute the dot product distance from the query to each centre in
    // the codebook.
    float sim{0.0F};
    std::vector<float> simTable(BOOK_SIZE * NUM_BOOKS);
    switch (metric_) {
    case Cosine:
    case Dot: {
        // Compute the dot product distance from the query to cluster centre.
        const auto* clusterCentre = &clustersCentres_[cluster * dim];
        for (std::size_t i = 0; i < dim; ++i) {
            sim += query[i] * clusterCentre[i];
        }

        // Transform the query vector into the codebook's coordinate system.
        std::vector<float> transformedQuery(dim, 0.0F);
        transform(dim, transformations_[cluster], query.data(), transformedQuery.data());

        const auto& codebooksCentres = codebooksCentres_[cluster];
        for (std::size_t b = 0; b < NUM_BOOKS; ++b) {
            const auto* queryProj = &transformedQuery[b * bookDim];
            const auto* codebookCentres = &codebooksCentres[b * BOOK_SIZE * bookDim];
            for (std::size_t i = 0; i < BOOK_SIZE; ++i) {
                float sim{0.0F};
                const auto* codebookCentre = codebookCentres + i * bookDim;
                #pragma omp simd reduction(+:sim)
                for (std::size_t j = 0; j < bookDim; ++j) {
                    sim += queryProj[j] * codebookCentre[j];
                }
                simTable[BOOK_SIZE * b + i] = sim;
            }
        }
        break;
    }
    case Euclidean: {
        // The Euclidean difference is |q - (c + r)| = |T * (q - c) - T * r|.
        // We store T * r in the codebook centres so we can compute this by
        // first computing the residual vector q - c and then transforming.
        // Ultimately we compute the squared Euclidean distance by subtracting
        // the sum over codebook distances from 1. So transform the distances
        // to similarities by negating.

        std::vector<float> queryResidual(query);
        centre(dim_, &clustersCentres_[cluster * dim_], queryResidual.data());

        // Transform the residual vector into the codebook's coordinate system.
        std::vector<float> transformedQuery(dim, 0.0F);
        transform(dim, transformations_[cluster], queryResidual.data(), transformedQuery.data());

        // We set this to one so it cancels when we subtract the sum of
        // terms from one to convert similarity to distance.
        sim = 1.0;

        const auto& codebooksCentres = codebooksCentres_[cluster];
        for (std::size_t b = 0; b < NUM_BOOKS; ++b) {
            const auto* queryProj = &transformedQuery[b * bookDim];
            const auto* codebookCentres = &codebooksCentres[b * BOOK_SIZE * bookDim];
            for (std::size_t i = 0; i < BOOK_SIZE; ++i) {
                float dist2{0.0F};
                const auto* codebookCentre = codebookCentres + i * bookDim;
                #pragma omp simd reduction(+:dist2)
                for (std::size_t j = 0; j < bookDim; ++j) {
                    float dij{queryProj[j] - codebookCentre[j]};
                    dist2 += dij * dij;
                }
                simTable[BOOK_SIZE * b + i] = -dist2;
            }
        }
        break;
    }
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
    // by the norm of the document vector. The norm of the document vector
    // is |c + r| = (|c|^2 + 2 c^t r + |r|^2)^(1/2). By construction the
    // centres are normalized so this simplifies to (1 + 2 c^t r + |r|)^(1/2).
    // We can look up the dot product between the centre and the residual
    // and the norm of the residual in the normsTable.
    if (metric_ == Cosine) {
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
    std::minstd_rand rng;
    std::uniform_int_distribution<std::uint_fast32_t> seedDist(
        0, std::numeric_limits<std::uint_fast32_t>::max());
    std::cout << "Building codebooks for " << numClusters << " clusters" << std::endl;

    std::vector<std::vector<float>> transformations;
    transformations.reserve(numClusters);

    // We sample in chunks of 32 clusters to reduce the peak memory usage.
    // We store 64 * dim samples per cluster. For 768 d vectors this amounts
    // to 768 * 64 * 768 * 4 = 144 MB per cluster. So our peak memory usage
    // is 144 MB * 32 = 4.6 GB.
    std::cout << "Computing optimal transforms" << std::endl;
    for (std::size_t i = 0; i < numClusters; i += 32) {

        std::size_t beginClusters{i};
        std::size_t endClusters{i + std::min(NUM_READERS, numClusters - i)};
        std::size_t numSamplesPerCluster{64 * dim};
        std::cout << "Sampling clusters in range "
                  << "[" << beginClusters << "," << endClusters << ") "
                  << "docs per cluster = " << numSamplesPerCluster << std::endl;

        std::vector<float> initialSamples(numSamplesPerCluster * dim,
                                          std::numeric_limits<float>::quiet_NaN());
        std::vector<std::vector<float>> samples(numClusters);
        std::fill(samples.begin() + beginClusters,
                  samples.begin() + endClusters, initialSamples);

        std::vector<std::vector<ReservoirSampler>> samplers(
            NUM_READERS, std::vector<ReservoirSampler>(numClusters));
        auto sampleReaders = initializeSampleReaders(dim, beginClusters, endClusters,
                                                     numSamplesPerCluster, docsClusters,
                                                     std::minstd_rand{seedDist(rng)},
                                                     samples, samplers);
        parallelRead(docs, sampleReaders);
        removeNans(samples);

        // Compute optimal transformations for each coarse cluster.
        for (std::size_t i = beginClusters; i < endClusters; ++i) {
            auto [eigVecs, eigVals] = pca(dim, std::move(samples[i]));
            transformations.emplace_back(
                computeOptimalPQSubspaces(dim, eigVecs, eigVals));
        }
    }

    // Compute the codebook centres for each coarse cluster.
    std::vector<std::vector<float>> codebooksCentres(numClusters);
    std::cout << "Computing codebooks" << std::endl;

    // We sample in chunks of 32 clusters to reduce the peak memory usage.
    // We store 128 * BOOK_SIZE samples per cluster. For 768 d vectors
    // this amounts to 128 * 256 * 768 * 4 = 96 MB per cluster. So our
    // peak memory usage is 96 MB * 32 = 3 GB.
    ProgressBar progress{"Clustering...", numClusters};
    for (std::size_t i = 0; i < numClusters; i += 32) {

        std::size_t beginClusters{i};
        std::size_t endClusters{i + std::min(NUM_READERS, numClusters - i)};
        std::size_t numSamplesPerCluster{128 * BOOK_SIZE};

        std::vector<float> initialSamples(numSamplesPerCluster * dim,
                                          std::numeric_limits<float>::quiet_NaN());
        std::vector<std::vector<float>> samples(numClusters);
        std::fill(samples.begin() + beginClusters,
                  samples.begin() + endClusters, initialSamples);

        std::vector<std::vector<ReservoirSampler>> samplers(
            NUM_READERS, std::vector<ReservoirSampler>(numClusters));
        auto sampleReaders = initializeSampleReaders(dim, beginClusters, endClusters,
                                                     numSamplesPerCluster, docsClusters,
                                                     std::minstd_rand{seedDist(rng)},
                                                     samples, samplers);
        parallelRead(docs, sampleReaders);
        removeNans(samples);

        // Compute the residuals from the coarse cluster centres and transform
        // them to align with the optimal subspaces.
        for (std::size_t j = beginClusters; j < endClusters; ++j) {
            auto& transformation = transformations[j];
            auto& sample = samples[j];
            for (std::size_t k = 0; k < sample.size(); k += dim) {
                centre(dim, &clustersCentres[j * dim], &sample[k]);
            }
            sample = transform(transformation, dim, std::move(sample));
        }

        for (std::size_t j = beginClusters; j < endClusters; ++j) {
            codebooksCentres[j] = buildCodebook(dim, samples[j]).first;
            progress.update();
        }
    }

    return {std::move(transformations), std::move(codebooksCentres)};
}

PqIndex buildPqIndex(const BigVector& docs, Metric metric, float distanceThreshold) {

    std::size_t dim{docs.dim()};
    std::size_t numDocs{docs.numVectors()};

    std::vector<float> clustersCentres;
    std::vector<cluster_t> docsClusters;
    coarseClustering(metric == Cosine, docs, clustersCentres, docsClusters);

    std::vector<std::vector<float>> transformations;
    std::vector<std::vector<float>> codebooksCentres;
    std::tie(transformations, codebooksCentres) =
        buildCodebooksForPqIndex(docs, clustersCentres, docsClusters);

    // Compute the codes for each doc.

    std::vector<Reader> encoders;
    encoders.reserve(NUM_READERS);

    std::vector<code_t> docsCodes(numDocs * NUM_BOOKS);
    auto beginDocsCodes = docsCodes.data();
    auto beginDocsClusters = docsClusters.begin();
    std::vector<float> centredDoc(dim);
    std::vector<float> transformedDoc(dim);
    for (std::size_t i = 0; i < NUM_READERS; ++i) {
        encoders.emplace_back([=, &clustersCentres, &transformations](
                std::size_t pos,
                BigVector::VectorReference doc) mutable {
            auto docCodes = beginDocsCodes + NUM_BOOKS * pos;
            auto docCluster = *(beginDocsClusters + pos);
            const auto& transformation = transformations[docCluster];
            const auto& codebookCentres = codebooksCentres[docCluster];
            centredDoc.assign(doc.data(), doc.data() + dim);
            centre(dim, &clustersCentres[docCluster * dim], centredDoc.data());
            transform(dim, transformation, centredDoc.data(), transformedDoc.data());
            if (distanceThreshold > 0.0F) {
                anisotropicEncode(transformedDoc, codebookCentres, distanceThreshold, docCodes);
            } else {
                encode(transformedDoc, codebookCentres, docCodes);
            }
        });
    }

    parallelRead(docs, encoders);

    return {metric, dim, std::move(clustersCentres), std::move(transformations),
            std::move(codebooksCentres), std::move(docsClusters), std::move(docsCodes)};
}
