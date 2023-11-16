#include "../src/common/utils.h"

#include "../src/pq/clustering.h"
#include "../src/pq/codebooks.h"
#include "../src/pq/constants.h"
#include "../src/pq/index.h"
#include "../src/pq/stats.h"
#include "../src/pq/subspace.h"
#include "../src/pq/utils.h"

#include <boost/test/tools/old/interface.hpp>
#include <boost/test/unit_test.hpp>

#include <algorithm>
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

BOOST_AUTO_TEST_SUITE(pq)

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
    // Check that each step of Lloyd's algorithm decreases the MSE.
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

    // Check that the eigenvectors are normalised.
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

    auto transformation = computeOptimalPQSubspaces(dim, NUM_BOOKS, eigVecs, eigValues);
    
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

    // Check that the optimal PQ transformations are normalised.
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
    BOOST_REQUIRE_LT(transformedVarianceRange, 0.1 * originalVarianceRange);
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

    char filename[] = "/tmp/prefXXXXXX";
    int ret{::mkstemp(filename)};
    if (ret == -1) {
        BOOST_FAIL("Couldn't create temporary file");
    }
    std::cout << "Created temporary file " << filename << std::endl;
    std::filesystem::path tmpFile{filename};

    // Create a BigVector using a random generator.
    std::size_t dim{10};
    std::size_t numDocs{1000};
    std::minstd_rand rng{0};
    std::uniform_real_distribution<float> u01{0.0F, 1.0F};
    BigVector docs{dim, numDocs, tmpFile, [&] { return u01(rng); }};

    // Sample 5% of the docs.
    double sampleProbability{0.05};
    auto sampledDocs = sampleDocs(docs, sampleProbability, rng);

    // Check we get the correct number.
    BOOST_REQUIRE_EQUAL(sampledDocs.size(),
                        static_cast<std::size_t>(sampleProbability * numDocs * dim));

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

BOOST_AUTO_TEST_SUITE_END()

} // unnamed::