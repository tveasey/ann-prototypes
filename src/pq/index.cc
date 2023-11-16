#include "index.h"

#include "clustering.h"
#include "codebooks.h"
#include "constants.h"
#include "subspace.h"
#include "../common/bigvector.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <ostream>
#include <queue>
#include <random>
#include <utility>
#include <vector>

namespace {

void centre(std::size_t dim,
            const float* centre,
            float* doc) {
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
        for (std::size_t j = 0; j < dim; ++j) {
            projectedDoc[i] += transformation[i * dim + j] * doc[j];
        }
    }
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
        const auto* centroid = &codeBooks[(BOOK_SIZE * b + docCode[b]) * bookDim];
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
computeCodebooksForPqIndex(const BigVector& docs,
                           const std::vector<float>& clustersCentres,
                           const std::vector<cluster_t>& docsClusters) {

    std::size_t dim{docs.dim()};
    std::size_t numDocs{docs.numVectors()};
    std::size_t numClusters{clustersCentres.size() / dim};

    std::vector<std::vector<float>> transformations;
    transformations.reserve(numClusters);
    std::minstd_rand rng;
    // We use a codeblock to ensure the samplers are destroyed before we
    // move the next step. This reduces the peak memory usage.
    {
        // Use a random sample of 20 docs per dimension.
        std::vector<ReservoirSampler> samplers(numClusters, {dim, 20 * dim, rng});
        auto i = docsClusters.begin();
        for (auto doc : docs) {
            samplers[*(i++)].add(doc.data());
        }

        // Compute optimal transformations for each coarse cluster.
        for (auto& sampler : samplers) {
            auto& sample = sampler.sample();
            auto [eigVecs, eigVals] = pca(dim, std::move(sample));
            transformations.emplace_back(
                computeOptimalPQSubspaces(dim, NUM_BOOKS, eigVecs, eigVals));
        }
    }

    // Prepare the data set for computing code book centres.
    std::vector<std::vector<float>> projectedSamples(numClusters);

    // We use a codeblock to ensure the samplers are destroyed before we
    // move the next step. This reduces the peak memory usage.
    {
        // Use a random sample of 128 docs per cluster centre.
        //
        // Depending on the expected cluster size this may be more the memory
        // bottleneck. We can always trade runtime for memory by using an outer
        // iteration over cluster ranges.
        std::vector<ReservoirSampler> samplers(numClusters, {dim, 128 * BOOK_SIZE, rng});
        auto i = docsClusters.begin();
        for (auto doc : docs) {
            samplers[*(i++)].add(doc.data());
        }

        // Compute the residuals from the coarse cluster centres and transform
        // them into the optimal subspaces.
        for (std::size_t i = 0; i < clustersCentres.size(); i += dim) {
            auto& transformation = transformations[i];
            auto& projectedSample = projectedSamples[i];
            projectedSample = std::move(samplers[i].sample());
            for (std::size_t j = 0; j < projectedSample.size(); j += dim) {
                centre(dim, &clustersCentres[i], &projectedSample[j]);
            }
            projectedSample = transform(transformation, dim, std::move(projectedSample));
        }
    }

    // Compute the codebook centres for each coarse cluster.
    std::vector<std::vector<float>> codebooksCentres(numClusters);
    for (std::size_t i = 0; i < numClusters; ++i) {
        codebooksCentres[i] = buildCodebook(dim, projectedSamples[i]).first;
    }

    return {std::move(transformations), std::move(codebooksCentres)};
}

PqIndex computePqIndex(bool normalized, const BigVector& docs) {

    std::size_t dim{docs.dim()};
    std::size_t numDocs{docs.numVectors()};

    std::vector<float> clustersCentres;
    std::vector<cluster_t> docsClusters;
    coarseClustering(docs, clustersCentres, docsClusters);
    if (normalized) {
        // Normalize the cluster centres (i.e. spherical k-means).
        for (std::size_t i = 0; i < clustersCentres.size(); i += dim) {
            float norm{std::sqrtf(
                std::accumulate(clustersCentres.begin() + i,
                                clustersCentres.begin() + i + dim,
                                0.0F))};
            for (std::size_t j = i; j < i + dim; ++j) {
                clustersCentres[i] /= norm;
            }
        }
    }

    auto [transformations, codebooksCentres] = 
        computeCodebooksForPqIndex(docs, clustersCentres, docsClusters);

    // Compute the codes for each doc.
    std::vector<code_t> docsCodes(numDocs * NUM_BOOKS);
    auto docCodes = docsCodes.data();
    auto docCluster = docsClusters.begin();
    std::vector<float> centredDoc;
    std::vector<float> projectedDoc;
    for (auto doc : docs) {
        centredDoc.assign(doc.data(), doc.data() + dim);
        centre(dim, &clustersCentres[*docCluster], centredDoc.data());
        transform(dim, transformations[*docCluster], centredDoc.data(), projectedDoc.data());
        writeEncoding(projectedDoc, codebooksCentres[*docCluster], docCodes);
        ++docCluster;
        docCodes += NUM_BOOKS;
    }

    return {normalized, dim, std::move(clustersCentres), std::move(transformations),
            std::move(codebooksCentres), std::move(docsClusters), std::move(docsCodes)};
}
