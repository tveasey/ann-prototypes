#pragma once

#include "types.h"
#include "../common/bigvector.h"
#include "../pq/constants.h"

#include <cstdint>
#include <random>
#include <set>
#include <vector>

// Initialize subspace cluster centres using the kmeans++ method.
std::pair<std::vector<float>, std::vector<int>>
initKmeansPlusPlus(std::size_t dim,
                   std::size_t numSubspaces,
                   std::size_t numClusters,
                   const std::vector<float>& docs,
                   std::minstd_rand& rng);

// Select the documents ids uniformly at random for the initial codebooks centres.
//
// This is Forgy's method.
std::set<std::size_t> initForgyForBookConstruction(std::size_t numDocs,
                                                   std::minstd_rand& rng);

// Select the documents uniformly at random for the initial codebooks centres.
//
// This is Forgy's method.
std::vector<float> initForgyForBookConstruction(std::size_t dim,
                                                std::size_t numBooks,
                                                const std::vector<float>& docs,
                                                std::minstd_rand& rng);

// Configurable initialisation of the codebooks centres used for testing.
std::vector<float> initForgyForTest(std::size_t dim,
                                    std::size_t numSubspaces,
                                    std::size_t numClusters,
                                    const std::vector<float>& docs,
                                    std::minstd_rand& rng);

// Initialize the codebooks centres using the kmeans++ method.
//
// This selects the first centre uniformly at random and then selects the
// remaining centres with probability proportional to the squared distance
// to the closest selected centre.
std::vector<float> initKmeansPlusPlusForBookConstruction(std::size_t dim,
                                                         std::size_t numBooks,
                                                         const std::vector<float>& docs,
                                                         std::minstd_rand& rng);

// Update the codebooks centres using with one iteration of Lloyd algorihm.
double stepLloydForBookConstruction(std::size_t dim,
                                    std::size_t numBooks,
                                    const std::vector<float>& docs,
                                    std::vector<float>& centres,
                                    std::vector<code_t>& docsCodes);

// Configurable step of Lloyd's algorithm used for testing.
double stepLloydForTest(std::size_t dim,
                        std::size_t numSubspaces,
                        std::size_t numClusters,
                        const std::vector<float>& docs,
                        std::vector<float>& centres,
                        std::vector<code_t>& docsCodes);

// Compute a coarse clustering of the vectors in `docs`.
//
// All vectors are assigned to the closest centre and separate codebooks
// are constructed for each cluster.
void coarseClustering(bool normalized,
                      const BigVector& docs,
                      std::vector<float>& clusterCentres,
                      std::vector<cluster_t>& docsClusters,
                      std::size_t docsPerCluster = COARSE_CLUSTERING_DOCS_PER_CLUSTER);

// Assign the documents to the coarse clusters.
//
// All vectors are assigned to the closest centre and separate codebooks
// are constructed for each cluster.
void assignDocsToCoarseClusters(bool normalized,
                                const BigVector& docs,
                                std::vector<float>& clusterCentres,
                                std::vector<cluster_t>& docsClusters);