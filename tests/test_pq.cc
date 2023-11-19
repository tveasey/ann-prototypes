#include "../src/common/utils.h"

#include "../src/pq/clustering.h"
#include "../src/pq/codebooks.h"
#include "../src/pq/constants.h"
#include "../src/pq/index.h"
#include "../src/pq/stats.h"
#include "../src/pq/subspace.h"
#include "../src/pq/utils.h"
#include "../src/common/io.h" 
#include "../src/common/utils.h" 

#include <boost/test/tools/old/interface.hpp>
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

std::vector<float> transpose(const std::vector<float>& matrix, std::size_t dim) {
    std::vector<float> transposed(dim * dim);
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            transposed[i * dim + j] = matrix[j * dim + i];
        }
    }
    return transposed;
};

BOOST_AUTO_TEST_SUITE(pq)

BOOST_AUTO_TEST_CASE(testLoadAndPrepareDocs) {

    auto file = std::filesystem::path(__FILE__).parent_path() / "vectors.fvec";

    auto [vectors, dim] = readFvecs(file);

    dim = zeroPad(dim, vectors);

    {
        BigVector vector{loadAndPrepareData(file, false)};

        BOOST_REQUIRE_EQUAL(vector.dim(), dim);
        BOOST_REQUIRE_EQUAL(vector.numVectors(), vectors.size() / dim);
        BOOST_REQUIRE_EQUAL(vector.size(), vectors.size());

        std::size_t i{0};
        for (auto vec : vector) {
            for (std::size_t j = 0; j < dim; ++j) {
                BOOST_REQUIRE_EQUAL(vec[j], vectors[i++]);
            }
        }
    }

    normalize(dim, vectors);

    {
        // Load again normalizing.
        BigVector vector{loadAndPrepareData(file, true)};

        std::size_t i{0};
        for (auto vec : vector) {
            for (std::size_t j = 0; j < dim; ++j) {
                BOOST_REQUIRE_CLOSE(vec[j], vectors[i++], 1e-4);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testClusteringInitializations) {

    // Test k-means initialization methods.

    // Test that Forgy initialization chooses NUM_BOOKS distinct doc ids
    // from the specified number of docs.
    std::minstd_rand rng;
    auto ids = initForgyForBookConstruction(1000, rng);
    
    BOOST_REQUIRE_EQUAL(ids.size(), BOOK_SIZE);

    // Check that every id is within range.
    for (auto id : ids) {
        BOOST_REQUIRE_LT(id, 1000);
    }

    // Check that if we run multiple times we converge to uniform distribution.
    std::vector<std::size_t> counts(1000, 0);
    for (std::size_t i = 0; i < 5000; ++i) {
        ids = initForgyForBookConstruction(1000, rng);
        for (auto id : ids) {
            ++counts[id];
        }
    }
    auto totalCount = static_cast<double>(
        std::accumulate(counts.begin(), counts.end(), 0UL));
    for (auto count : counts) {
        BOOST_REQUIRE_CLOSE(static_cast<double>(count) / totalCount, 1.0 / 1000.0, 10.0);
    }

    // Test that initialization using vectors chooses BOOK_SIZE vectors in total.
    {
        std::size_t dim{NUM_BOOKS};
        std::vector<float> docs(1000 * dim);
        std::iota(docs.begin(), docs.end(), 0.0F);
        
        auto centres = initForgyForBookConstruction(dim, docs, rng);
        BOOST_REQUIRE_EQUAL(centres.size(), BOOK_SIZE * dim);

        centres = initKmeansPlusPlusForBookConstruction(dim, docs, rng);
        BOOST_REQUIRE_EQUAL(centres.size(), BOOK_SIZE * dim);
    }

    // Test that initialization using vectors chooses the correct subspaces from
    // each vector. To do this we arrange for each document to have NUM_BOOKS
    // dimension. Then we ensure that the ranges for each component do not overlap.
    // Finally, we check that the selected centres only use values from the correct
    // component range.
    {
        std::size_t dim{2 * NUM_BOOKS};
        std::vector<float> docs(1000 * dim);
        for (std::size_t i = 0; i < 1000; ++i) {
            for (std::size_t j = 0; j < dim; ++j) {
                // Use a random number from the range [5 * j, 5 * (j + 1))].
                std::uniform_real_distribution<> uxy{5.0 * j, 5.0 * (j + 1)};
                docs[i * dim + j] = uxy(rng);
            }
        }

        auto centres = initForgyForBookConstruction(dim, docs, rng);
        for (std::size_t b = 0; b < NUM_BOOKS; ++b) {
            for (std::size_t i = 0; i < 2 * BOOK_SIZE; ++i) {
                BOOST_REQUIRE_GE(centres[2 * b * BOOK_SIZE + i], 10.0 * b);
                BOOST_REQUIRE_LT(centres[2 * b * BOOK_SIZE + i], 10.0 * (b + 1));
            }
        }
        centres = initKmeansPlusPlusForBookConstruction(dim, docs, rng);
        for (std::size_t b = 0; b < NUM_BOOKS; ++b) {
            for (std::size_t i = 0; i < 2 * BOOK_SIZE; ++i) {
                BOOST_REQUIRE_GE(centres[2 * b * BOOK_SIZE + i], 10.0 * b);
                BOOST_REQUIRE_LT(centres[2 * b * BOOK_SIZE + i], 10.0 * (b + 1));
            }
        }
    }

    // Check k-means++ initialization works with coincident points.
    {
        std::size_t dim{NUM_BOOKS};
        std::vector<float> docs(1000 * dim);
        std::fill(docs.begin(), docs.end(), 0.0F);
        auto centres = initKmeansPlusPlusForBookConstruction(dim, docs, rng);
        BOOST_REQUIRE_EQUAL(centres.size(), BOOK_SIZE * dim);
    }

    // Arrange from the data to be drawn from a mixture of 20 Gaussians. Check
    // that k-means++ picks points from every cluster with high probability.
    for (std::size_t trial = 0; trial < 10; ++trial) {
        std::size_t dim{NUM_BOOKS};
        std::size_t numClusters{20};
        std::size_t numDocs{200};
        std::vector<float> docs(numDocs * dim);
        std::vector<float> centres(numClusters * dim);
        std::vector<std::size_t> counts(numClusters, 0);
        for (std::size_t i = 0; i < numClusters; ++i) {
            std::normal_distribution<> uxy{20.0 * static_cast<double>(i), 1.0};
            for (std::size_t j = 0; j < dim; ++j) {
                centres[i * dim + j] = uxy(rng);
            }
        }
        for (std::size_t i = 0; i < numDocs; ++i) {
            std::uniform_int_distribution<std::size_t> uxy{0, numClusters - 1};
            std::size_t id{uxy(rng)};
            for (std::size_t j = 0; j < dim; ++j) {
                std::normal_distribution<> uxy{centres[id * dim + j], 1.0};
                docs[i * dim + j] = uxy(rng);
            }
        }

        auto clusterCentres = initKmeansPlusPlusForBookConstruction(dim, docs, rng);
        for (std::size_t b = 0; b < NUM_BOOKS; ++b) {
            for (std::size_t i = 0; i < numClusters; ++i) {
                for (std::size_t j = 0; j < BOOK_SIZE; ++j) {
                    if (std::abs(clusterCentres[b * BOOK_SIZE + j] - 20.0 * i) < 5.0) {
                        ++counts[i];
                        break;
                    }
                }
            }
        }
        for (std::size_t i = 0; i < numClusters; ++i) {
            // Each book should sample each cluster.
            BOOST_REQUIRE_EQUAL(counts[i], NUM_BOOKS);
        }
    }
}

BOOST_AUTO_TEST_CASE(testClusteringStepLloyd) {

    std::size_t dim{10};
    std::size_t numDocs{1000};
    std::minstd_rand rng{0};
    std::uniform_real_distribution<float> u01{0.0F, 1.0F};
    std::vector<float> docs(dim * numDocs);
    for (std::size_t i = 0; i < docs.size(); ++i) {
        docs[i] = u01(rng);
    }

    std::size_t numSubspaces{5};
    std::size_t numClusters{10};
    std::size_t subspaceDim{dim / numSubspaces};

    auto centres = initForgyForTest(dim, numSubspaces, numClusters, docs, rng);

    double lastMsd{std::numeric_limits<double>::max()};

    for (std::size_t i = 0; i < 5; ++i) {

        auto oldCentres = centres;

        std::vector<code_t> docsCodes;
        double msd{stepLloydForTest(dim, numSubspaces, numClusters, docs, centres, docsCodes)};

        // Check that each step of Lloyd's algorithm decreases the MSE.
        BOOST_REQUIRE_LT(msd, lastMsd);
        lastMsd = msd;

        // Check that the centres are the mean of the vectors assigned to them.
        for (std::size_t b = 0; b < numSubspaces; ++b) {
            std::vector<std::size_t> counts(numClusters, 0);
            std::vector<float> centroids(numClusters * subspaceDim, 0.0F);
            for (std::size_t i = 0; i < numDocs; ++i) {
                std::size_t cluster{docsCodes[i * numSubspaces + b]};
                ++counts[cluster];
                for (std::size_t j = 0; j < subspaceDim; ++j) {
                    centroids[cluster * subspaceDim + j] += docs[i * dim + b * subspaceDim + j];
                }
            }
            for (std::size_t i = 0; i < numClusters; ++i) {
                for (std::size_t j = 0; j < subspaceDim; ++j) {
                    centroids[i * subspaceDim + j] /= static_cast<float>(counts[i]);
                }
            }

            auto* subspaceCentres = &centres[b * numClusters * subspaceDim];
            for (std::size_t i = 0; i < numClusters; ++i) {
                for (std::size_t j = 0; j < subspaceDim; ++j) {
                    BOOST_REQUIRE_CLOSE(centroids[i * subspaceDim + j],
                                        subspaceCentres[i * subspaceDim + j], 1e-4);
                }
            }

            // Check that each vector is assigned to the closest centre.
            for (std::size_t i = 0; i < numDocs; ++i) {
                auto* subspaceOldCentres = &oldCentres[b * numClusters * subspaceDim];

                std::size_t iMsd{0};
                float msd{std::numeric_limits<float>::max()};
                for (std::size_t j = 0; j < numClusters; ++j) {
                    float sd{0.0F};
                    for (std::size_t k = 0; k < subspaceDim; ++k) {
                        float di{docs[i * dim + b * subspaceDim + k] - 
                                 subspaceOldCentres[j * subspaceDim + k]};
                        sd += di * di;
                    }
                    if (sd < msd) {
                        iMsd = j;
                        msd = sd;
                    }
                }
                BOOST_REQUIRE_EQUAL(docsCodes[i * numSubspaces + b], iMsd);
            }
        }

        // Compute the MSE for the subspace with the old centres.
        double avgMsd{0.0};
        for (std::size_t b = 0; b < numSubspaces; ++b) {
            double subspaceMsd{0.0};
            for (std::size_t i = 0; i < numDocs; ++i) {
                std::size_t cluster{docsCodes[i * numSubspaces + b]};
                auto* subspaceOldCentres = &oldCentres[b * numClusters * subspaceDim];
                for (std::size_t j = 0; j < subspaceDim; ++j) {
                    float di{docs[i * dim + b * subspaceDim + j] -
                             subspaceOldCentres[cluster * subspaceDim + j]};
                    subspaceMsd += di * di;
                }
            }
            avgMsd += subspaceMsd;
        }
        avgMsd /= static_cast<double>(numDocs);
        BOOST_REQUIRE_CLOSE(avgMsd, msd, 1e-4);
    }
}

BOOST_AUTO_TEST_CASE(testCoarseClustering) {

    char filename[]{"/tmp/test_storage_XXXXXX"};
    int ret{::mkstemp(filename)};
    if (ret == -1) {
        BOOST_FAIL("Couldn't create temporary file");
    }
    std::cout << "Created temporary file " << filename << std::endl;

    std::size_t dim{16};
    std::size_t numDocs{10 * COARSE_CLUSTERING_SAMPLE_SIZE};
    std::minstd_rand rng{0};
    std::uniform_real_distribution<float> u01{0.0F, 1.0F};
    std::filesystem::path tmpFile{filename};
    BigVector docs{dim, numDocs, tmpFile, [&]() { return u01(rng); }};

    std::vector<float> clusterCentres;
    std::vector<cluster_t> docsClusters;
    coarseClustering(false, docs, clusterCentres, docsClusters);

    // Check that the number of clusters is correct.
    BOOST_REQUIRE_EQUAL(clusterCentres.size(), 10 * dim);
    BOOST_REQUIRE_EQUAL(docsClusters.size(), numDocs);

    // Count the number of documents in each cluster and check that it is
    // approximately equal.
    std::vector<std::size_t> counts(10, 0);
    for (auto cluster : docsClusters) {
        ++counts[cluster];
    }
    for (std::size_t i = 0; i < 10; ++i) {
        BOOST_REQUIRE_CLOSE(static_cast<double>(counts[i]) / 
                            static_cast<double>(numDocs), 0.1, 5.0);
    }

    // Test the variance reduction of coarse clustering.
    //
    // To do this we compute the residual trace of the covariance matrix of
    // the data for the coarse clustering. We expect the latter to be smaller.
    // To compute the trace we simply need to sum the square differences of
    // each vector component from the mean vector.

    std::vector<float> mean(dim, 0.0F);
    for (auto doc : docs) {
        for (std::size_t i = 0; i < dim; ++i) {
            mean[i] += doc[i];
        }
    }
    for (std::size_t i = 0; i < dim; ++i) {
        mean[i] /= static_cast<float>(numDocs);
    }
    double residualVariance{0.0};
    for (auto doc : docs) {
        for (std::size_t i = 0; i < dim; ++i) {
            float di{doc[i] - mean[i]};
            residualVariance += di * di;
        }
    }
    residualVariance /= static_cast<double>(numDocs);

    double clusteredResidualVariance{0.0};
    auto docCluster = docsClusters.begin();
    for (auto doc : docs) {
        auto* clusterCentre = &clusterCentres[*(docCluster++) * dim];
        for (std::size_t i = 0; i < dim; ++i) {
            float di{doc[i] - clusterCentre[i]};
            clusteredResidualVariance += di * di;
        }
    }
    clusteredResidualVariance /= static_cast<double>(numDocs);

    std::cout << "Residual variance: " << residualVariance << std::endl;
    std::cout << "Coarse residual variance: " << clusteredResidualVariance << std::endl;

    BOOST_REQUIRE_LT(clusteredResidualVariance, 0.9 * residualVariance);
}

BOOST_AUTO_TEST_CASE(testEncode) {

    std::size_t dim{NUM_BOOKS};
    std::size_t numDocs{100};
    
    // Create some random centres.
    std::vector<float> distinctCentres(BOOK_SIZE);
    std::minstd_rand rng{0};
    for (std::size_t i = 0; i < BOOK_SIZE; ++i) {
        std::uniform_real_distribution<float> u01{0.0F, 5.0F};
        distinctCentres[i] = u01(rng);
    }
    std::vector<float> codebooksCentres(dim * BOOK_SIZE);
    for (std::size_t b = 0; b < NUM_BOOKS; ++b) {
        for (std::size_t i = 0; i < BOOK_SIZE; ++i) {
            codebooksCentres[b * BOOK_SIZE + i] = distinctCentres[i];
        }
    }

    // Each document exactly matches one centre for each subspace. These should
    // be the centres selected.
    std::uniform_int_distribution<std::size_t> uxy{0, BOOK_SIZE - 1};
    std::vector<float> docs(numDocs * dim);
    std::vector<code_t> expectedDocsCodes(numDocs * NUM_BOOKS);
    for (std::size_t i = 0; i < numDocs; ++i) {
        for (std::size_t b = 0; b < NUM_BOOKS; ++b) {
            std::size_t centre{uxy(rng)};
            docs[i * dim + b] = distinctCentres[centre];
            expectedDocsCodes[i * NUM_BOOKS + b] = static_cast<code_t>(centre);
        }
    }

    std::vector<code_t> docsCodes(numDocs * NUM_BOOKS);
    std::vector<float> doc(dim);
    for (std::size_t i = 0; i < numDocs; ++i) {
        std::copy(&docs[i * dim], &docs[(i + 1) * dim], doc.data());
        encode(doc, codebooksCentres, &docsCodes[i * NUM_BOOKS]);
    }

    for (std::size_t i = 0; i < numDocs; ++i) {
        for (std::size_t b = 0; b < NUM_BOOKS; ++b) {
            BOOST_REQUIRE_EQUAL(
                static_cast<std::size_t>(docsCodes[i * NUM_BOOKS + b]),
                static_cast<std::size_t>(expectedDocsCodes[i * NUM_BOOKS + b]));
        }
    }
}

BOOST_AUTO_TEST_CASE(testAnisotropicEncode) {

    std::size_t dim{16 * NUM_BOOKS};
    std::size_t numDocs{10000};
    std::minstd_rand rng{0};
    std::uniform_real_distribution<float> u01{0.0F, 1.0F};
    std::vector<float> docs(dim * numDocs);
    for (std::size_t i = 0; i < docs.size(); ++i) {
        docs[i] = u01(rng);
    }

    auto [eigVecs, eigVals] = pca(dim, docs);
    auto transformation = computeOptimalPQSubspaces(dim, eigVecs, eigVals);
    docs = transform(transformation, dim, std::move(docs));
    auto [centres, docsCodes] = buildCodebook(dim, docs);

    // Compute the anisotropic codes for each document.
    std::vector<code_t> anisotropicDocsCodes(numDocs * NUM_BOOKS);
    std::vector<float> doc(dim);
    for (std::size_t i = 0; i < numDocs; ++i) {
        std::copy(&docs[i * dim], &docs[(i + 1) * dim], doc.data());
        anisotropicEncode(doc, centres, 0.6F, &anisotropicDocsCodes[i * NUM_BOOKS]);
    }

    // Compute the dot product between the original and anisotropic encoded
    // vectors and the original document for each document. This is given by
    // sum of the dot product from each subspace to the corresponding codebook
    // centre.
    std::vector<float> dotProducts(numDocs, 0.0F);
    std::vector<float> pqDotProducts(numDocs, 0.0F);
    std::vector<float> pqDotProductsAnisotropic(numDocs, 0.0F);
    std::size_t bookDim{dim / NUM_BOOKS};
    for (std::size_t i = 0; i < numDocs; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            dotProducts[i] += docs[i * dim + j] * docs[i * dim + j];
        }

        for (std::size_t b = 0; b < NUM_BOOKS; ++b) {
            auto* docProj = &docs[i * dim + b * bookDim];

            std::size_t code{docsCodes[i * NUM_BOOKS + b]};
            auto* centre = &centres[(b * BOOK_SIZE + code) * bookDim];
            for (std::size_t j = 0; j < bookDim; ++j) {
                pqDotProducts[i] += centre[j] * docProj[j];
            }

            code = anisotropicDocsCodes[i * NUM_BOOKS + b];
            centre = &centres[(b * BOOK_SIZE + code) * bookDim];
            for (std::size_t j = 0; j < bookDim; ++j) {
                pqDotProductsAnisotropic[i] += centre[j] * docProj[j];
            }
        }
    }

    float avgPqDiff{0.0F};
    float avgPqAnisotropicDiff{0.0F};
    for (std::size_t i = 0; i < numDocs; ++i) {
        avgPqDiff += std::fabsf(dotProducts[i] - pqDotProducts[i]);
        avgPqAnisotropicDiff += std::fabsf(dotProducts[i] - pqDotProductsAnisotropic[i]);
    }
    avgPqDiff /= static_cast<float>(numDocs);
    avgPqAnisotropicDiff /= static_cast<float>(numDocs);
    std::cout << "Average PQ diff: " << avgPqDiff << std::endl;
    std::cout << "Average PQ anisotropic diff: " << avgPqAnisotropicDiff << std::endl;

    // We expect the anisotropic codes to result in lower dot product
    // error for vectors close to the original document.
    BOOST_REQUIRE_LT(avgPqAnisotropicDiff, 0.75 * avgPqDiff);
}

BOOST_AUTO_TEST_CASE(testBuildCodebook) {

    std::size_t dim{2 * NUM_BOOKS};
    std::size_t numDocs{1000};
    std::minstd_rand rng{0};
    std::uniform_real_distribution<float> u01{0.0F, 1.0F};
    std::vector<float> docs(dim * numDocs);
    for (std::size_t i = 0; i < docs.size(); ++i) {
        docs[i] = u01(rng);
    }

    auto [centres, docsCodes] = buildCodebook(dim, docs);

    // Check that the codebook has the expected size.
    BOOST_REQUIRE_EQUAL(centres.size(), dim * BOOK_SIZE);
    BOOST_REQUIRE_EQUAL(docsCodes.size(), numDocs * NUM_BOOKS);

    // Check that the resconstruction error is much less than encoding
    // with random centres.

    std::vector<float> randomCentres(dim * BOOK_SIZE);
    for (std::size_t i = 0; i < randomCentres.size(); ++i) {
        randomCentres[i] = u01(rng);
    }
    std::vector<code_t> randomCentresDocsCodes(numDocs * NUM_BOOKS);
    {
        std::vector<float> doc(dim);
        for (std::size_t i = 0; i < numDocs; ++i) {
            std::copy(&docs[i * dim], &docs[(i + 1) * dim], doc.data());
            encode(doc, randomCentres, &docsCodes[i * NUM_BOOKS]);
        }
    }

    double avgMse{0.0};
    double avgRandomMse{0.0};
    std::size_t bookDim{dim / NUM_BOOKS};
    for (std::size_t i = 0; i < numDocs; ++i) {
        for (std::size_t b = 0; b < NUM_BOOKS; ++b) {
            std::size_t code{docsCodes[i * NUM_BOOKS + b]};
            auto doc = &docs[i * dim + b * bookDim];
            auto centre = &centres[(b * BOOK_SIZE + code) * bookDim];
            for (std::size_t j = 0; j < bookDim; ++j) {
                float dj{doc[j] - centres[j]};
                avgMse += dj * dj;
            }
        }
        for (std::size_t b = 0; b < NUM_BOOKS; ++b) {
            std::size_t code{randomCentresDocsCodes[i * NUM_BOOKS + b]};
            auto doc = &docs[i * dim + b * bookDim];
            auto centre = &randomCentres[(b * BOOK_SIZE + code) * bookDim];
            for (std::size_t j = 0; j < dim; ++j) {
                float dj{doc[j] - centre[j]};
                avgRandomMse += dj * dj;
            }
        }
    }
    avgMse /= static_cast<double>(numDocs);
    avgRandomMse /= static_cast<double>(numDocs);

    std::cout << "Clustered MSE: " << avgMse << std::endl;
    std::cout << "Random MSE:    " << avgRandomMse << std::endl;

    BOOST_REQUIRE_LT(avgMse, avgRandomMse / 40.0);
}

BOOST_AUTO_TEST_CASE(testCentreData) {

    std::size_t dim{10};
    std::size_t numDocs{1000};
    std::minstd_rand rng{0};
    std::uniform_real_distribution<float> u01{0.0F, 1.0F};
    std::vector<float> docs(dim * numDocs);
    for (std::size_t i = 0; i < docs.size(); ++i) {
        docs[i] = u01(rng);
    }
    auto centredDocs = centreData(dim, std::move(docs));

    // Check that the mean of each dimension is zero.
    for (std::size_t i = 0; i < dim; ++i) {
        float mean{0.0F};
        for (std::size_t j = 0; j < numDocs; ++j) {
            mean += centredDocs[j * dim + i];
        }
        BOOST_REQUIRE_LT(std::fabsf(mean), 1e-3);
    }
}

BOOST_AUTO_TEST_CASE(testCovarianceMatrix) {

    // Check some invariants of some random data's covariance matrix.
    std::size_t dim{10};
    std::size_t numDocs{1000};
    std::minstd_rand rng{0};
    std::uniform_real_distribution<float> u01{0.0F, 1.0F};
    std::vector<float> docs(dim * numDocs);
    for (std::size_t i = 0; i < docs.size(); ++i) {
        docs[i] = u01(rng);
    }

    // Centre the data.
    docs = centreData(dim, std::move(docs));

    auto cov = covarianceMatrix(dim, docs);

    // Check that the covariance matrix is symmetric.
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = i + 1; j < dim; ++j) {
            BOOST_REQUIRE_CLOSE(cov[i * dim + j], cov[j * dim + i], 1e-6);
        }
    }

    // Check that the covariance matrix is positive semi-definite. A sufficient
    // condition for this is that all the diagonals are non-negative and the
    // matrix is diagonally dominant.
    for (std::size_t i = 0; i < dim; ++i) {
        BOOST_REQUIRE_GE(cov[i * dim + i], 0.0);
        double sum{0.0};
        for (std::size_t j = 0; j < dim; ++j) {
            if (i != j) {
                sum += std::fabs(cov[i * dim + j]);
            }
        }
        BOOST_REQUIRE_GE(cov[i * dim + i], sum);
    }
}

BOOST_AUTO_TEST_CASE(testPca) {

    std::size_t dim{10};
    std::size_t numDocs{1000};
    std::minstd_rand rng{0};
    std::uniform_real_distribution<float> u01{0.0F, 1.0F};
    std::vector<float> docs(dim * numDocs);
    for (std::size_t i = 0; i < docs.size(); ++i) {
        docs[i] = u01(rng);
    }

    auto [eigVecs, eigValues] = pca(dim, docs);

    // Check that the eigenvalues are in descending order.
    for (std::size_t i = 1; i < dim; ++i) {
        BOOST_REQUIRE_GE(eigValues[i - 1], eigValues[i]);
    }

    // Check that the eigenvectors are orthogonal.
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = i + 1; j < dim; ++j) {
            float dot{0.0F};
            for (std::size_t k = 0; k < dim; ++k) {
                dot += eigVecs[i * dim + k] * eigVecs[j * dim + k];
            }
            BOOST_REQUIRE_LT(std::fabsf(dot), 1e-4);
        }
    }

    // Check that the eigenvectors are normalized.
    for (std::size_t i = 0; i < dim; ++i) {
        float norm{0.0F};
        for (std::size_t j = 0; j < dim; ++j) {
            norm += eigVecs[i * dim + j] * eigVecs[i * dim + j];
        }
        BOOST_REQUIRE_CLOSE(norm, 1.0F, 1e-4);
    }

    // Check that the eigenvectors multiplied by the data covariance matrix
    // satisfy the definition of an eigenvector.
    docs = centreData(dim, std::move(docs));
    auto cov = covarianceMatrix(dim, docs);
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            float dot{0.0F};
            for (std::size_t k = 0; k < dim; ++k) {
                dot += cov[j * dim + k] * eigVecs[i * dim + k];
            }
            BOOST_REQUIRE_CLOSE(dot, eigValues[i] * eigVecs[i * dim + j], 1e-4);
        }
    }
}

BOOST_AUTO_TEST_CASE(testComputeOptimalPQSubspaces) {

    std::size_t dim{3 * NUM_BOOKS};
    std::size_t numDocs{1000};
    std::minstd_rand rng{0};
    std::vector<float> docs(dim * numDocs);
    for (std::size_t i = 0; i < numDocs; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            std::uniform_real_distribution<float> u0n{0.0F, 0.1F * static_cast<float>(j + 1)};
            docs[i * dim + j] = u0n(rng);
        }
    }

    auto [eigVecs, eigValues] = pca(dim, docs);

    auto transformation = computeOptimalPQSubspaces(dim, eigVecs, eigValues);
    
    // Check that the optimal PQ transformations are orthogonal.
    for (std::size_t i = 0; i < NUM_BOOKS; ++i) {
        for (std::size_t j = i + 1; j < NUM_BOOKS; ++j) {
            float dot{0.0F};
            for (std::size_t k = 0; k < dim; ++k) {
                dot += transformation[i * dim + k] * transformation[j * dim + k];
            }
            BOOST_REQUIRE_LT(std::fabsf(dot), 1e-4);
        }
    }

    // Check that the optimal PQ transformations are normalized.
    for (std::size_t i = 0; i < NUM_BOOKS; ++i) {
        float norm{0.0F};
        for (std::size_t j = 0; j < dim; ++j) {
            norm += transformation[i * dim + j] * transformation[i * dim + j];
        }
        BOOST_REQUIRE_CLOSE(norm, 1.0F, 1e-4);
    }

    // Compute the data variance per subspace.
    std::vector<float> originalVariances(NUM_BOOKS, 0.0F);
    std::vector<float> transformedVariance(NUM_BOOKS, 0.0F);
    docs = centreData(dim, std::move(docs));
    for (std::size_t b = 0; b < NUM_BOOKS; ++b) {
        for (std::size_t i = 0; i < numDocs; ++i) {
            std::array<float, 3> x{docs[i * dim +  3 * b],
                                   docs[i * dim + (3 * b + 1)],
                                   docs[i * dim + (3 * b + 2)]};
            originalVariances[b] += x[0] * x[0] + x[1] * x[1] + x[2] * x[2];

            std::array<float, 3> xt{0.0F, 0.0F, 0.0F};
            for (std::size_t k = 0; k < dim; ++k) {
                xt[0] += docs[i * dim + k] * transformation[(3 * b)     * dim + k];
                xt[1] += docs[i * dim + k] * transformation[(3 * b + 1) * dim + k];
                xt[2] += docs[i * dim + k] * transformation[(3 * b + 2) * dim + k];
            }
            transformedVariance[b] += xt[0] * xt[0] + xt[1] * xt[1] + xt[2] * xt[2];
        }
    }

    // Check that the range of subspace variances of the transformed data is
    // significantly smaller than the original data.
    auto [originalMin, originalMax] = 
        std::minmax_element(originalVariances.begin(), originalVariances.end());
    auto [transformedMin, transformedMax] =
        std::minmax_element(transformedVariance.begin(), transformedVariance.end());
    float originalVarianceRange{*originalMax - *originalMin};
    float transformedVarianceRange{*transformedMax - *transformedMin};
    BOOST_REQUIRE_LT(transformedVarianceRange, 0.15 * originalVarianceRange);
}

BOOST_AUTO_TEST_CASE(testOptimalCodebooks) {

    std::size_t dim{3 * NUM_BOOKS};
    std::size_t numDocs{128 * BOOK_SIZE};
    std::minstd_rand rng{0};
    std::vector<float> docs(dim * numDocs);
    for (std::size_t i = 0; i < numDocs; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            std::uniform_real_distribution<float> u0n{0.0F, 0.1F * static_cast<float>(j + 1)};
            docs[i * dim + j] = u0n(rng);
        }
    }

    // Compute the resconstruction error using the original data.
    auto [centres, docsCodes] = buildCodebook(dim, docs);

    double avgMse{0.0};
    std::size_t bookDim{dim / NUM_BOOKS};
    for (std::size_t i = 0; i < numDocs; ++i) {
        float docMse{0.0F};
        for (std::size_t b = 0; b < NUM_BOOKS; ++b) {
            std::size_t code{docsCodes[i * NUM_BOOKS + b]};
            auto doc = &docs[i * dim + b * bookDim];
            auto centre = &centres[(b * BOOK_SIZE + code) * bookDim];
            for (std::size_t j = 0; j < bookDim; ++j) {
                float dj{doc[j] - centre[j]};
                avgMse += dj * dj;
                docMse += dj * dj;
            }
        }
    }
    avgMse /= static_cast<double>(numDocs);

    // Compute the rectruction error using the transformed data.

    auto [eigVecs, eigVals] = pca(dim, docs);
    auto transformation = computeOptimalPQSubspaces(dim, eigVecs, eigVals);
    docs = transform(transformation, dim, std::move(docs));
    std::tie(centres, docsCodes) = buildCodebook(dim, docs);

    double avgTransformedMse{0.0};
    for (std::size_t i = 0; i < numDocs; ++i) {
        for (std::size_t b = 0; b < NUM_BOOKS; ++b) {
            std::size_t code{docsCodes[i * NUM_BOOKS + b]};
            auto doc = &docs[i * dim + b * bookDim];
            auto centre = &centres[(b * BOOK_SIZE + code) * bookDim];
            double mse{0.0};
            for (std::size_t j = 0; j < bookDim; ++j) {
                float dj{doc[j] - centre[j]};
                avgTransformedMse += dj * dj;
                mse += dj * dj;
            }
        }
    }
    avgTransformedMse /= static_cast<double>(numDocs);

    std::cout << "Original MSE:    " << avgMse << std::endl;
    std::cout << "Transformed MSE: " << avgTransformedMse << std::endl;

    BOOST_REQUIRE_LT(avgTransformedMse, 0.6 * avgMse);
}

BOOST_AUTO_TEST_CASE(testTransform) {

    std::size_t dim{3 * NUM_BOOKS};
    std::size_t numDocs{1000};
    std::minstd_rand rng{0};
    std::vector<float> docs(dim * numDocs);
    for (std::size_t i = 0; i < numDocs; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            std::uniform_real_distribution<float> u0n{0.0F, 0.1F * static_cast<float>(j + 1)};
            docs[i * dim + j] = u0n(rng);
        }
    }

    // We check that transforming by:
    //   1. The identity yields the original docs,
    //   2. A multiple of the identity yields scales the original docs,
    //   3. The reflected identity reverses the docs.

    std::vector<float> identity(dim * dim, 0.0F);
    for (std::size_t i = 0; i < dim; ++i) {
        identity[i * dim + i] = 1.0F;
    }
    auto transformedDocs = transform(identity, dim, docs);
    for (std::size_t i = 0; i < docs.size(); ++i) {
        BOOST_REQUIRE_CLOSE(transformedDocs[i], docs[i], 1e-4);
    }

    std::vector<float> scale(dim * dim, 0.0F);
    for (std::size_t i = 0; i < dim; ++i) {
        scale[i * dim + i] = 2.0F;
    }
    transformedDocs = transform(scale, dim, docs);
    for (std::size_t i = 0; i < docs.size(); ++i) {
        BOOST_REQUIRE_CLOSE(transformedDocs[i], 2.0F * docs[i], 1e-4);
    }

    std::vector<float> reflect(dim * dim, 0.0F);
    for (std::size_t i = 0; i < dim; ++i) {
        reflect[i * dim + (dim - i - 1)] = 1.0F;
    }
    transformedDocs = transform(reflect, dim, docs);
    for (std::size_t i = 0; i < docs.size(); i += dim) {
        for (std::size_t j = 0; j < dim; ++j) {
            BOOST_REQUIRE_CLOSE(transformedDocs[i + j], docs[i + dim - j - 1], 1e-4);
        }
    }
}

BOOST_AUTO_TEST_CASE(testSampleDocs) {

    char filename[]{"/tmp/test_storage_XXXXXX"};
    int ret{::mkstemp(filename)};
    if (ret == -1) {
        BOOST_FAIL("Couldn't create temporary file");
    }
    std::cout << "Created temporary file " << filename << std::endl;

    // Create a BigVector using a random generator.
    std::size_t dim{10};
    std::size_t numDocs{1000};
    std::minstd_rand rng{0};
    std::uniform_real_distribution<float> u01{0.0F, 1.0F};
    std::filesystem::path tmpFile{filename};
    BigVector docs{dim, numDocs, tmpFile, [&] { return u01(rng); }};

    // Sample 5% of the docs.
    auto sampledDocs = sampleDocs(docs, 50, rng);

    // Check we get the correct number.
    BOOST_REQUIRE_EQUAL(sampledDocs.size(), static_cast<std::size_t>(50 * dim));

    // Check that the vectors we sample are all in the original set.
    auto hash_combine = [](std::size_t seed, std::size_t value) {
        seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    };
    auto hash = [&](const auto* begin, const auto* end) {
        std::size_t seed{0};
        std::hash<float> hasher;
        for (auto x = begin; x != end; ++x) {
            seed = hash_combine(seed, hasher(*x));
        }
        return seed;
    };
    std::unordered_set<std::size_t> hashes;
    std::hash<float> hasher;
    for (auto vec : docs) {
        hashes.insert(hash(vec.data(), vec.data() + dim));
    }
    for (std::size_t i = 0; i < sampledDocs.size(); i += dim) {
        BOOST_REQUIRE_EQUAL(hashes.count(hash(sampledDocs.data() + i,
                                              sampledDocs.data() + i + dim)), 1);
    }
}

BOOST_AUTO_TEST_CASE(testZeroPad) {
    std::vector<float> vectors(2 * NUM_BOOKS - 2, 1.0F);
    zeroPad(NUM_BOOKS - 1, vectors);

    BOOST_REQUIRE_EQUAL(vectors.size(), 2 * NUM_BOOKS);

    // Check for each vector the first the first NUM_BOOKS - 1 components
    // are unchanged and the last component is zero.
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < NUM_BOOKS - 1; ++j) {
            BOOST_REQUIRE_EQUAL(vectors[i * NUM_BOOKS + j], 1.0F);
        }
        BOOST_REQUIRE_EQUAL(vectors[i * NUM_BOOKS + NUM_BOOKS - 1], 0.0F);
    }
}

BOOST_AUTO_TEST_CASE(testBuildCodebooksForPqIndex) {

    char filename[]{"/tmp/test_storage_XXXXXX"};
    int ret{::mkstemp(filename)};
    if (ret == -1) {
        BOOST_FAIL("Couldn't create temporary file");
    }
    std::cout << "Created temporary file " << filename << std::endl;

    // Create a BigVector using a random generator.
    std::size_t dim{2 * NUM_BOOKS};
    std::size_t bookDim{dim / NUM_BOOKS};
    std::size_t numDocs{6 * COARSE_CLUSTERING_DOCS_PER_CLUSTER};
    std::minstd_rand rng{0};
    std::uniform_real_distribution<float> u01{0.0F, 1.0F};
    std::filesystem::path tmpFile{filename};
    BigVector docs{dim, numDocs, tmpFile, [&] { return u01(rng); }};

    std::vector<float> clusterCentres;
    std::vector<cluster_t> docsClusters;
    coarseClustering(false, docs, clusterCentres, docsClusters);

    auto [transformations, codebooksCentres] =
        buildCodebooksForPqIndex(docs, clusterCentres, docsClusters);

    // We should have a transformation and codebook for each cluster.
    BOOST_REQUIRE_EQUAL(transformations.size(), 6);
    BOOST_REQUIRE_EQUAL(codebooksCentres.size(), 6);

    // Each transform should be a dim * dim matrix.
    for (auto& transform : transformations) {
        BOOST_REQUIRE_EQUAL(transform.size(), dim * dim);
    }

    // Each codebook should have NUM_BOOKS * BOOK_SIZE centres.
    std::size_t numCl{clusterCentres.size() / dim};
    for (auto& codebook : codebooksCentres) {
        BOOST_REQUIRE_EQUAL(codebook.size(), NUM_BOOKS * BOOK_SIZE * bookDim);
    }

    // Check the quantization error in the calculation of the dot products
    // between 1000 randomly selected pairs of documents is small.

    float avgRelativeError{0.0F};

    for (std::size_t i = 0; i < 1000; ++i) {
        // Choose a random pair of documents.
        std::uniform_int_distribution<std::size_t> uxy{0, numDocs - 1};
        std::size_t x{uxy(rng)};
        std::size_t y{uxy(rng)};
        std::vector<float> docX(dim);
        std::vector<float> docY(dim);

        // Read the documents from the BigVector via random access iterators.
        const auto* beginX = (*(docs.begin() + x)).data();
        const auto* beginY = (*(docs.begin() + y)).data();
        std::copy(beginX, beginX + dim, docX.begin());
        std::copy(beginY, beginY + dim, docY.begin());

        // Compute the dot product between the documents.
        float dot{0.0F};
        for (std::size_t j = 0; j < dim; ++j) {
            dot += docX[j] * docY[j];
        }

        std::size_t clusterX{docsClusters[x]};
        std::size_t clusterY{docsClusters[y]};
        const auto& transformationX = transformations[clusterX];
        const auto& transformationY = transformations[clusterY];
        const auto& codebooksCentresX = codebooksCentres[clusterX];
        const auto& codebooksCentresY = codebooksCentres[clusterY];
        const auto* clusterCentreX = &clusterCentres[clusterX * dim];
        const auto* clusterCentreY = &clusterCentres[clusterY * dim];

        // Encode the documents.
        std::vector<code_t> docsCodesX(NUM_BOOKS);
        std::vector<code_t> docsCodesY(NUM_BOOKS);
        std::vector<float> residualX(dim);
        std::vector<float> residualY(dim);
        for (std::size_t j = 0; j < dim; ++j) {
            residualX[j] = docX[j] - clusterCentreX[j];
            residualY[j] = docY[j] - clusterCentreY[j];
        }
        residualX = transform(transformationX, dim, residualX);
        residualY = transform(transformationY, dim, residualY);
        encode(residualX, codebooksCentresX, docsCodesX.data());
        encode(residualY, codebooksCentresY, docsCodesY.data());

        // Compute the dot product between the documents from the coarse cluster
        // centres and the corresponding codebooks approximations. This is given
        // by (c_x + r_x)^t (c_y + r_y). The residuals r_x and r_y are read from
        // the document codes then transformed by the inverse of the corresponding
        // transformation matrices. Since these are othogonal, the inverse is the
        // transpose.
        std::vector<float> quantizedResidualX(dim);
        std::vector<float> quantizedResidualY(dim);
        for (std::size_t b = 0; b < NUM_BOOKS; ++b) {
            std::size_t codeX{docsCodesX[b]};
            std::size_t codeY{docsCodesY[b]};
            auto* codebookCentreX = &codebooksCentresX[(b * BOOK_SIZE + codeX) * bookDim];
            auto* codebookCentreY = &codebooksCentresY[(b * BOOK_SIZE + codeY) * bookDim];
            for (std::size_t j = 0; j < bookDim; ++j) {
                quantizedResidualX[b * bookDim + j] = codebookCentreX[j];
                quantizedResidualY[b * bookDim + j] = codebookCentreY[j];
            }
        }
        quantizedResidualX = transform(transpose(transformationX, dim),
                                       dim, quantizedResidualX);
        quantizedResidualY = transform(transpose(transformationY, dim),
                                       dim, quantizedResidualY);

        float quantizedDot{0.0F};
        for (std::size_t j = 0; j < dim; ++j) {
            quantizedDot += (clusterCentreX[j] + quantizedResidualX[j]) *
                            (clusterCentreY[j] + quantizedResidualY[j]);
        }

        avgRelativeError += std::fabsf(dot - quantizedDot) / std::fabsf(dot);
    }

    avgRelativeError /= 1000.0F;

    std::cout << "Average relative error: " << avgRelativeError << std::endl;
    BOOST_REQUIRE_LT(avgRelativeError, 0.01);
}

BOOST_AUTO_TEST_SUITE_END()

} // unnamed::