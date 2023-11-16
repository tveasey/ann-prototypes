#pragma once

#include <cstddef>

// The number of subspaces to use in the PQ index.
constexpr int NUM_BOOKS{24};
// The number of centres in each codebook.
constexpr int BOOK_SIZE{256};
// The number of iterations to run k-means for when constructing the codebooks.
constexpr std::size_t BOOK_CONSTRUCTION_K_MEANS_ITR{8};
// The number of random restarts of clustering to use when constructing the codebooks.
constexpr std::size_t BOOK_CONSTRUCTION_K_MEANS_RESTARTS{5};
// The number of docs to sample to compute the coarse clustering.
constexpr std::size_t COARSE_CLUSTERING_SAMPLE_SIZE{256 * 1024};
// The expected number of docs for each cluster when computing the coarse clustering.
// This is used to compute the number of clusters to use.
constexpr std::size_t COARSE_CLUSTERING_DOCS_PER_CLUSTER{256 * 1024};
// The number of iterations to run k-means for when computing the coarse clustering.
constexpr std::size_t COARSE_CLUSTERING_KMEANS_ITR{10};
// The number of random restarts of clustering to use when computing the coarse clustering.
constexpr std::size_t COARSE_CLUSTERING_KMEANS_RESTARTS{5};
