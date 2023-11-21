#pragma once

#include "constants.h"
#include "types.h"
#include "../common/bigvector.h"

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

class BigVector;

// The PQ index is a data structure that allows to perform approximate nearest
// neighbor search. It is based on the Product Quantization method, which
// consists in splitting the vectors into subvectors and quantizing each
// subvector separately. The quantization is performed by building a codebook
// for each subvector. The codebook is a set of centres and each vector is
// assigned to the closest centre. The code of a vector is the sequence of
// indices of the centres to which the subvectors are assigned.
//
// The PQ index is a two-level index. The first level is a coarse clustering
// of the vectors. The second level is a set of codebooks, one for each cluster
// of the coarse clustering.
//
// We use Optimized Product Quantization (OPQ) to compute the codebooks. OPQ
// is a method to compute the codebooks that minimizes the quantization error.
// It is based on the idea is to compute an orthognal transformation of the
// vectors before quantizing them. We use the parametrization of OPQ. First a
// random sample is used to perform PCA then eigenvectors are greedily added
// each subspace with the minimum product of the eigenvalues so far. This has
// the effect of roughly balancing the data variance across the subspaces.
//
// In total we have the following memory usage:
//   1. The cluster centres use dim * sizeof(float) bytes per cluster.
//   2. The transformations use dim * dim * sizeof(float) bytes per cluster.
//   3. The cluster identifiers use sizeof(cluster_t) bytes per document.
//   4. The codebooks use NUM_BOOKS * sizeof(code_t) bytes per document.
//   5. The codebooks centres use BOOK_SIZE * dim * sizeof(float) bytes
//      per cluster.
//   6. The norms table uses 2 * NUM_BOOKS * BOOK_SIZE * sizeof(float) bytes
//      and per cluster.
//
// For example, suppose sizeof(cluster_t) equals 2 bytes and sizeof(code_t),
// we use 512 * 1024 = 524288 vectors per cluster, and 24 codebooks with 256
// centres per book. Then if we have 100M 768 d vectors, the total number of
// clusters is 100M / 524288 = 191. So the total memory is:
//   1. 191 * 768 * 4 = 0.56 MB bytes for the cluster centres.
//   2. 191 * 768 * 768 * 4 = 430 MB for the transformations.
//   3. 100M * 2 = 200 MB for the cluster identifiers.
//   4. 100M * 24 = 2.2 GB for the codes.
//   5. 191 * 256 * 768 * 4 = 143 MB for the codebooks centres.
//   6. 191 * 2 * 24 * 256 * 4 = 9 MB for the norms table.
//
// In total this is 2.96 GB. The raw vectors use 100M * 768 * 4 = 286 GB.
class PqIndex {
public:
    PqIndex(bool normalized,
            std::size_t dim,
            std::vector<float> clustersCentres,
            std::vector<std::vector<float>> transformations,
            std::vector<std::vector<float>> codebooksCentres,
            std::vector<cluster_t> docsClusters,
            std::vector<code_t> docsCodes);

    // Return the number of clusters.
    std::size_t numClusters() const { return clustersCentres_.size() / dim_; }

    // Return the clusters.
    const std::vector<float>& clustersCentres() const { return clustersCentres_; }

    // Return the cluster of the document vector with give id.
    cluster_t cluster(std::size_t id) const { return docsClusters_[id]; }

    // Return the number of centres.
    std::size_t numCodebooksCentres() const;

    // Return the codebook centres for the given cluster.
    const std::vector<float>& codebookCentres(cluster_t cluster) const {
        return codebooksCentres_[cluster];
    }

    // Return the number of transformations.
    std::size_t numTransformations() const { return transformations_.size(); }

    // Return the transformations for the given cluster.
    const std::vector<float>& transformations(cluster_t cluster) const {
        return transformations_[cluster];
    }

    // Return the number of codes.
    std::size_t numCodes() const { return docsCodes_.size(); }

    // Return the document encoding.
    std::vector<code_t> codes(std::size_t id) const;

    // Perform exact k-nearest neighbor search.
    std::pair<std::vector<std::size_t>, std::vector<float>>
    search(const std::vector<float>& query, std::size_t k) const;

    // Extract the PQ index encoding of the id document's vector.
    std::vector<float> decode(std::size_t id) const;

    // Compute the dot product distance between the query vector and the
    // document vector with the given id.
    float computeDist(const std::vector<float>& query, std::size_t id) const;

    // Compression ratio of the PQ index.
    double compressionRatio() const;

private:
    // Build the table of norms of the codebooks centres.
    //
    // This is used to correct the distance calculation in the case that
    // vectors are normalized.
    void buildNormsTables();

    // Build the table of dot product similarities between the query and
    // the codebooks centres.
    std::pair<float, std::vector<float>>
    buildSimTable(std::size_t cluster, 
                  const std::vector<float> &query) const;

    // Compute the distance between the query and the document with the given
    // codes.
    float computeDist(float centreSim,
                      const std::vector<float>& simTable,
                      const std::vector<float>& normsTable,
                      const code_t* docCodes) const;

private:
    bool normalized_;
    std::size_t dim_;
    std::vector<float> clustersCentres_;
    std::vector<std::vector<float>> transformations_;
    std::vector<std::vector<float>> codebooksCentres_;
    std::vector<cluster_t> docsClusters_;
    std::vector<code_t> docsCodes_;
    std::vector<std::vector<float>> normsTable_;
};

std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>
buildCodebooksForPqIndex(const BigVector& docs,
                         const std::vector<float>& centres,
                         const std::vector<cluster_t>& docsCentres);

PqIndex buildPqIndex(const BigVector& docs,
                     bool normalized,
                     float distanceThreshold = 0.0F);
