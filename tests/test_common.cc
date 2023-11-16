#include "../src/common/bruteforce.h"
#include "../src/common/evaluation.h"
#include "../src/common/io.h"
#include "../src/common/bigvector.h"
#include "../src/common/utils.h"

#include <algorithm>
#include <boost/test/tools/old/interface.hpp>
#include <boost/test/unit_test.hpp>

#include <boost/test/unit_test_suite.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

BOOST_AUTO_TEST_SUITE(common)

BOOST_AUTO_TEST_CASE(testBigVector) {
    // Create a temporary file.
    char filename[] = "/tmp/prefXXXXXX";
    int ret{::mkstemp(filename)};
    if (ret == -1) {
        BOOST_FAIL("Couldn't create temporary file");
    }
    std::cout << "Created temporary file " << filename << std::endl;

    std::filesystem::path tmpFile{filename};

    // Write some data to the file.
    std::vector<float> data(100);
    std::iota(data.begin(), data.end(), 0.0f);
    std::ofstream ofs{tmpFile, std::ios::binary};
    ofs.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    ofs.close();

    // Create a BigVector from the file.
    BigVector vec{10, 10, tmpFile};

    BOOST_REQUIRE_EQUAL(vec.dim(), 10);
    BOOST_REQUIRE_EQUAL(vec.numVectors(), 10);
    BOOST_REQUIRE_EQUAL(vec.size(), 100);
    float i{0.0F};
    for (auto vec : vec) {
        for (std::size_t j = 1.0; j < 10.0; j += 1.0F) {
            BOOST_REQUIRE_EQUAL(vec[j], i + j);
        }
        i += 10.0F;
    }

    // The temporary file will be deleted when the BigVector is destroyed.
}

BOOST_AUTO_TEST_CASE(testNormalize) {
    std::minstd_rand rng{0};
    std::uniform_real_distribution<float> u01{0.0F, 1.0F};
    std::vector<float> vectors(1000);
    for (std::size_t i = 0; i < vectors.size(); ++i) {
        vectors[i] = u01(rng);
    }
    normalise(10, vectors);
    for (std::size_t i = 0; i < vectors.size(); i += 10) {
        float norm{0.0F};
        for (std::size_t j = 0; j < 10; ++j) {
            norm += vectors[i + j] * vectors[i + j];
        }
        BOOST_REQUIRE_CLOSE(norm, 1.0F, 1e-4);
    }
}

BOOST_AUTO_TEST_CASE(testReadFvecs) {
    auto file = std::filesystem::path(__FILE__).parent_path() / "vectors.fvec";
    auto [vectors, dim] = readFvecs(file);
    BOOST_REQUIRE_EQUAL(dim, 4);
    std::ostringstream result;
    result << vectors;
    BOOST_REQUIRE_EQUAL(result.str(), "[-1.1,2.1,0.3,1.7,1.2,3.1,-0.9,1.8]");
}

BOOST_AUTO_TEST_CASE(testRecall) {
    // Test that the recall is calculated correctly.
    std::vector<std::size_t> exact{0, 1, 2, 5, 4, 3, 6, 9, 8, 7};
    std::vector<std::size_t> approx{1, 0, 2, 4, 6, 7, 17, 11, 14, 8};

    BOOST_REQUIRE_EQUAL(computeRecall(exact, approx), 7.0 / 10.0);
}

BOOST_AUTO_TEST_CASE(testSampleDocs) {

    std::minstd_rand rng{0};
    std::uniform_real_distribution<float> u01{0.0F, 1.0F};
    std::size_t dim{10};
    std::size_t numVectors{1000};
    std::vector<float> docs(dim * numVectors);
    for (std::size_t i = 0; i < docs.size(); ++i) {
        docs[i] = u01(rng);
    }

    std::vector<float> sampledVectors{sampleDocs(dim, docs, 0.0, rng)};
    BOOST_REQUIRE_EQUAL(sampledVectors.size(), 0);

    sampledVectors = sampleDocs(dim, docs, 1.0, rng);
    BOOST_REQUIRE_EQUAL(sampledVectors.size(), docs.size());
    BOOST_REQUIRE_EQUAL_COLLECTIONS(sampledVectors.begin(), sampledVectors.end(),
                                    docs.begin(), docs.end());
    
    sampledVectors = sampleDocs(dim, docs, 0.1, rng);
    BOOST_REQUIRE_EQUAL(sampledVectors.size() % dim, 0);
    BOOST_REQUIRE_EQUAL(sampledVectors.size(), docs.size() / 10);

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
    for (std::size_t i = 0; i < docs.size(); i += dim) {
        hashes.insert(hash(docs.data() + i, docs.data() + i + dim));
    }
    for (std::size_t i = 0; i < sampledVectors.size(); i += dim) {
        BOOST_REQUIRE_EQUAL(hashes.count(hash(sampledVectors.data() + i,
                                              sampledVectors.data() + i + dim)), 1);
    }
}

BOOST_AUTO_TEST_CASE(testSearchBruteForce) {

    // Test that for a collection of known vectors the brute force search
    // the true nearest neighbours. Note that this uses the dot product as
    // the distance metric.

    std::size_t dim{3};
    std::vector<float> data{ 1.0F, 0.0F,  1.0F,
                             0.0F, 1.0F,  0.0F,
                            -1.0F, 0.0F, -1.0F};
    std::vector<std::vector<float>> queries{
        { 1.0F, 0.0F, 1.0F},
        { 0.0F, 1.0F, 0.0F},
        {-1.0F, 0.0F, -1.0F}};
    
    std::vector<std::size_t> expected{0, 1, 2};
    for (std::size_t i = 0; i < queries.size(); ++i) {
        auto [ids, scores] = searchBruteForce(1, data, queries[i]);
        BOOST_REQUIRE_EQUAL(ids.size(), 1);
        BOOST_REQUIRE_EQUAL(scores.size(), 1);
        BOOST_REQUIRE_EQUAL(ids[0], expected[i]);
        BOOST_REQUIRE_EQUAL(scores[0],
                            1.0 - std::transform_reduce(data.data() + expected[i] * dim,
                                                        data.data() + (expected[i] + 1) * dim,
                                                        queries[i].data(), 0));
    }
}

BOOST_AUTO_TEST_SUITE_END()

} // unnamed::