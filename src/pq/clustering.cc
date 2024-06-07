#include "clustering.h"

#include "constants.h"
#include "types.h"
#include "utils.h"
#include "../common/utils.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <vector>

std::pair<std::vector<float>, std::vector<int>>
initKmeansPlusPlus(std::size_t dim,
                   std::size_t numSubspaces,
                   std::size_t numClusters,
                   const std::vector<float>& docs,
                   std::minstd_rand& rng) {

    // TODO try migrating to quantized vectors.

    // Use k-means++ initialisation for each subspace independently.
    std::size_t subspaceDim{dim / numSubspaces};
    std::size_t numDocs{docs.size() / dim};
    std::vector<float> centres(numClusters * dim);
    std::vector<int> selectedDocs(numClusters);
    std::vector<float> msds;

    auto selected = selectedDocs.begin();
    auto centre = centres.begin();
    for (std::size_t b = 0; b < dim; b += subspaceDim) {
        // Choose the first centroid uniformly at random.
        std::uniform_int_distribution<> u0n{0, static_cast<int>(numDocs) - 1};
        std::size_t selectedDoc{static_cast<std::size_t>(u0n(rng))};
        *selected = selectedDoc;
        ++selected;
        auto begin = docs.begin() + selectedDoc * dim + b;
        std::copy(begin, begin + subspaceDim, centre);
        centre += subspaceDim;

        msds.assign(numDocs, std::numeric_limits<float>::max());
        for (std::size_t i = 1; i < numClusters; ++i) {
            // Update the squared distance from each document to the nearest
            // centroid.
            auto msd = msds.begin();
            auto lastCentre = centre - subspaceDim;
            for (auto doc = docs.begin(); doc != docs.end(); doc += dim, ++msd) {
                float sd{0.0F};
                auto docProj = doc + b;
                #pragma omp simd reduction(+:sd)
                for (std::size_t j = 0; j < subspaceDim; ++j) {
                    float dij{docProj[j] - lastCentre[j]};
                    sd += dij * dij;
                }
                *msd = std::min(*msd, sd);
            }

            if (std::all_of(msds.begin(), msds.end(),
                            [](float msd) { return msd == 0.0F; })) {
                // If all msds are zero then pick a random document.
                selectedDoc = static_cast<std::size_t>(u0n(rng));
            } else {
                // Sample with probability proportional to the squared distance.
                std::discrete_distribution<std::size_t> discrete{msds.begin(), msds.end()};
                selectedDoc = discrete(rng);
            }

            *selected = selectedDoc;
            ++selected;
            begin = docs.begin() + selectedDoc * dim + b;
            std::copy(begin, begin + subspaceDim, centre);
            centre += subspaceDim;
        }
    }

    return {std::move(centres), std::move(selectedDocs)};
}

namespace {

std::set<std::size_t> initForgy(std::size_t numDocs,
                                std::size_t numClusters,
                                std::minstd_rand& rng) {
    std::set<std::size_t> selection;
    std::uniform_int_distribution<> u0n{0, static_cast<int>(numDocs) - 1};
    while (selection.size() < numClusters) {
        std::size_t cand{static_cast<std::size_t>(u0n(rng))};
        selection.insert(cand);
    }
    return selection;
}

std::vector<float> initForgy(std::size_t dim,
                             std::size_t numSubspaces,
                             std::size_t numClusters,
                             const std::vector<float>& docs,
                             std::minstd_rand& rng) {
    // Use independent Forgy initialisation for each subspace.
    std::size_t subspaceDim{dim / numSubspaces};
    std::size_t numDocs{docs.size() / dim};
    std::vector<float> centres(numClusters * dim);
    auto centre = centres.begin();
    for (std::size_t b = 0; b < numSubspaces; ++b) {
        for (auto i : initForgy(numDocs, numClusters, rng)) {
            auto doc = docs.begin() + i * dim;
            std::copy(doc + b * subspaceDim, doc + (b + 1) * subspaceDim, centre);
            centre += subspaceDim;
        }
    }
    return centres;
}


template<typename CODE>
double stepLloyd(std::size_t dim,
                 std::size_t numSubspaces,
                 std::size_t numClusters,
                 const std::vector<float>& docs,
                 std::vector<float>& centres,
                 std::vector<CODE>& docsCodes) {

    // TODO try migrating to quantized vectors.

    // Since we sum up non-negative losses from each book we can compute
    // the optimal codebooks independently.

    std::size_t numDocs{docs.size() / dim};
    std::size_t subspaceDim{dim / numSubspaces};
    docsCodes.resize(numDocs * numSubspaces);
    std::vector<double> newCentres(numClusters * dim, 0.0);
    std::vector<std::size_t> centreCounts(numClusters * numSubspaces, 0);

    std::size_t pos{0};
    double sumSd{0.0};
    for (auto doc = docs.begin(); doc != docs.end(); /**/) {
        for (std::size_t b = 0; b < numSubspaces; ++b, ++pos, doc += subspaceDim) {
            // Find the nearest centroid.
            int iMsd{0};
            float msd{std::numeric_limits<float>::max()};
            for (std::size_t i = 0; i < numClusters; ++i) {
                float sd{0.0F};
                auto centreProj = &centres[(b * numClusters + i) * subspaceDim];
                #pragma omp simd reduction(+:sd)
                for (std::size_t j = 0; j < subspaceDim; ++j) {
                    float dij{doc[j] - centreProj[j]};
                    sd += dij * dij;
                }
                if (sd < msd) {
                    iMsd = i;
                    msd = sd;
                }
            }

            // Update the centroid.
            auto* newCentre = &newCentres[(b * numClusters + iMsd) * subspaceDim];
            for (std::size_t j = 0; j < subspaceDim; ++j) {
                newCentre[j] += doc[j];
            }
            ++centreCounts[b * numClusters + iMsd];

            // Encode the document.
            docsCodes[pos] = static_cast<CODE>(iMsd);
            sumSd += msd;
        }
    }

    for (std::size_t i = 0; i < centreCounts.size(); ++i) {
        if (centreCounts[i] > 0) {
            for (std::size_t j = 0; j < subspaceDim; ++j) {
                centres[i * subspaceDim + j] = static_cast<float>(
                    newCentres[i * subspaceDim + j] /
                    static_cast<double>(centreCounts[i]));
            }
        }
    }

    return sumSd / static_cast<double>(numDocs);
}

} // unnamed::

std::set<size_t> initForgyForBookConstruction(std::size_t numDocs,
                                              std::minstd_rand& rng) {
    return initForgy(numDocs, BOOK_SIZE, rng);
}

std::vector<float> initForgyForBookConstruction(std::size_t dim,
                                                std::size_t numBooks,
                                                const std::vector<float>& docs,
                                                std::minstd_rand& rng) {
    return initForgy(dim, numBooks, BOOK_SIZE, docs, rng);
}

std::vector<float> initForgyForTest(std::size_t dim,
                                    std::size_t numSubspaces,
                                    std::size_t numClusters,
                                    const std::vector<float>& docs,
                                    std::minstd_rand& rng) {
    return initForgy(dim, numSubspaces, numClusters, docs, rng);
}

std::vector<float> initKmeansPlusPlusForBookConstruction(std::size_t dim,
                                                         std::size_t numBooks,
                                                         const std::vector<float>& docs,
                                                         std::minstd_rand& rng) {
    return initKmeansPlusPlus(dim, numBooks, BOOK_SIZE, docs, rng).first;
}

double stepLloydForBookConstruction(std::size_t dim,
                                    std::size_t numBooks,
                                    const std::vector<float>& docs,
                                    std::vector<float>& centres,
                                    std::vector<code_t>& docsCodes) {
    return stepLloyd(dim, numBooks, BOOK_SIZE, docs, centres, docsCodes);
}

double stepLloydForTest(std::size_t dim,
                        std::size_t numSubspaces,
                        std::size_t numClusters,
                        const std::vector<float>& docs,
                        std::vector<float>& centres,
                        std::vector<code_t>& docsCodes) {
    return stepLloyd(dim, numSubspaces, numClusters, docs, centres, docsCodes);
}

void coarseClustering(bool normalized,
                      const BigVector& docs,
                      std::vector<float>& clusterCentres,
                      std::vector<cluster_t>& docsClusters,
                      std::size_t docsPerCluster) {

    // Use k-means to compute a coarse clustering of the data. We will use
    // one codebook per cluster.

    std::size_t dim{docs.dim()};
    std::size_t numDocs{docs.numVectors()};
    std::cout << "Coarse clustering " << numDocs << " documents..." << std::endl;

    std::minstd_rand rng;
    std::size_t sampleSize{std::min(COARSE_CLUSTERING_SAMPLE_SIZE, numDocs)};
    std::vector<float> sampledDocs{sampleDocs(docs, sampleSize, rng)};

    std::size_t numClusters{std::max(1UL, (numDocs + docsPerCluster - 1) / docsPerCluster)};

    auto ifSphericalKMeansNormalize = [dim, normalized](std::vector<float>& vectors) {
        if (normalized) {
            normalize(dim, vectors);
        }
    };

    // If there is only one cluster then the centre is the mean of the data
    // and the docs are all assigned to cluster zero.
    if (numClusters == 1) {
        clusterCentres.assign(numClusters * dim, 0.0F);
        for (auto doc = sampledDocs.begin(); doc != sampledDocs.end(); doc += docs.dim()) {
            for (std::size_t i = 0; i < dim; ++i) {
                clusterCentres[i] += doc[i];
            }
        }
        for (std::size_t i = 0; i < dim; ++i) {
            clusterCentres[i] /= static_cast<float>(sampleSize);
        }
        ifSphericalKMeansNormalize(clusterCentres);
        docsClusters.resize(numDocs, 0);
        return;
    }

    // A few restarts with Forgy initialisation is good enough.
    double msd{std::numeric_limits<double>::max()};
    for (std::size_t restarts = 0;
         restarts < COARSE_CLUSTERING_KMEANS_RESTARTS;
         ++restarts) {

        auto candidateClusterCentres = initForgy(dim, 1, numClusters, sampledDocs, rng);

        double sd{std::numeric_limits<double>::max()};
        for (std::size_t i = 0; i < COARSE_CLUSTERING_KMEANS_ITR; ++i) {
            sd = stepLloyd(dim, 1, numClusters, sampledDocs, candidateClusterCentres, docsClusters);
            ifSphericalKMeansNormalize(candidateClusterCentres);
        }
        if (sd < msd) {
            msd = sd;
            clusterCentres = std::move(candidateClusterCentres);
        }
    }
    std::cout << "Coarse clustering MSE = " << msd << std::endl;

    assignDocsToCoarseClusters(normalized, docs, clusterCentres, docsClusters);
}

void assignDocsToCoarseClusters(bool normalized,
                                const BigVector& docs,
                                std::vector<float>& clusterCentres,
                                std::vector<cluster_t>& docsClusters) {

    // Assign each document to the nearest centroid and update the centres.

    std::size_t dim{docs.dim()};
    std::size_t numDocs{docs.numVectors()};
    std::size_t numClusters{clusterCentres.size() / dim};

    docsClusters.resize(numDocs, 0);
    std::vector<std::vector<float>> newCentres(
        NUM_READERS, std::vector<float>(numClusters * dim, 0.0F));
    std::vector<std::vector<float>> compensations(
        NUM_READERS, std::vector<float>(numClusters * dim, 0.0F));

    auto ifSphericalKMeansNormalize = [dim, normalized](std::vector<float>& vectors) {
        if (normalized) {
            normalize(dim, vectors);
        }
    };

    std::vector<Reader> clusterWriters;
    clusterWriters.reserve(NUM_READERS);

    for (std::size_t i = 0; i < NUM_READERS; ++i) {

        clusterWriters.emplace_back([&, i](std::size_t pos, BigVector::VectorReference doc) {
            // Find the nearest centroid.
            int iMsd{0};
            float msd{std::numeric_limits<float>::max()};
            for (int i = 0; i < numClusters; ++i) {
                float sd{0.0F};
                #pragma omp simd reduction(+:sd)
                for (std::size_t j = 0; j < dim; ++j) {
                    float dij{clusterCentres[i * dim + j] - doc[j]};
                    sd += dij * dij;
                }
                if (sd < msd) {
                    iMsd = i;
                    msd = sd;
                }
            }

            docsClusters[pos] = static_cast<cluster_t>(iMsd);

            // Use Kahan summation to accumulate the new centres since we can
            // easily reach the limits of float precision for large datasets.
            auto* newCentre = &newCentres[i][iMsd * dim];
            auto* compensation = &compensations[i][iMsd * dim];
            for (std::size_t j = 0; j < dim; ++j) {
                float y{doc[j] - compensation[j]};
                float t{newCentre[j] + y};
                compensation[j] = (t - newCentre[j]) - y;
                newCentre[j] = t;
            }
        });
    }

    parallelRead(docs, clusterWriters);

    std::vector<std::size_t> clusterCounts(numClusters, 0);
    for (std::size_t i = 0; i < numDocs; ++i) {
        ++clusterCounts[docsClusters[i]];
    }

    // Reduce the new centres into the centres by summation and divide by
    // the total document count.
    for (std::size_t i = 0; i < numClusters; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            float sum{0.0F};
            #pragma omp simd reduction(+:sum)
            for (std::size_t k = 0; k < NUM_READERS; ++k) {
                sum += newCentres[k][i * dim + j];
            }
            clusterCentres[i * dim + j] = sum / static_cast<float>(clusterCounts[i]);
        }
    }
    ifSphericalKMeansNormalize(clusterCentres);
}