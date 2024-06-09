#include "../src/common/bruteforce.h"
#include "../src/common/evaluation.h"
#include "../src/common/io.h"
#include "../src/common/bigvector.h"
#include "../src/common/utils.h"

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

BOOST_AUTO_TEST_SUITE(common)

std::filesystem::path createTemporaryFile() {
    // Create a temporary file.
    char filename[] = "/tmp/test_storage_XXXXXX";
    int ret{::mkstemp(filename)};
    if (ret == -1) {
        BOOST_FAIL("Couldn't create temporary file");
    }
    std::cout << "Created temporary file " << filename << std::endl;
    return std::filesystem::path{filename};
}

BOOST_AUTO_TEST_CASE(testBigVector) {

    std::filesystem::path tmpFile{createTemporaryFile()};
    std::size_t dim{10};
    std::size_t numVectors{10};
    BigVector vec{dim, numVectors, tmpFile, [i = 0]() mutable {
        return i++;
    }};

    BOOST_REQUIRE_EQUAL(vec.dim(), 10);
    BOOST_REQUIRE_EQUAL(vec.numVectors(), 10);
    BOOST_REQUIRE_EQUAL(vec.size(), 100);
    float i{0.0F};
    for (auto vec : vec) {
        for (std::size_t j = 0.0; j < 10.0; j += 1.0F) {
            BOOST_REQUIRE_EQUAL(vec[j], i + j);
        }
        i += 10.0F;
    }
}

BOOST_AUTO_TEST_CASE(testBigVectorReadRange) {

    auto fvecs = std::filesystem::path(__FILE__).parent_path() / "vectors.fvec";

    std::vector<std::pair<double, double>> ranges{{0.0, 0.5}, {0.5, 1.0}};
    std::vector<std::vector<float>> expected{{-1.1F, 2.1F, 0.3F, 1.7F}, {1.2F, 3.1F, -0.9F, 1.8F}};

    for (std::size_t i = 0; i < ranges.size(); ++i) {
        std::filesystem::path tmpFile{createTemporaryFile()};
        BigVector vec{
            fvecs, tmpFile,
            [](std::size_t dim, std::vector<float>&) { return dim; },
            ranges[i]};

        BOOST_REQUIRE_EQUAL(vec.dim(), 4);
        BOOST_REQUIRE_EQUAL(vec.numVectors(), 1);
        BOOST_REQUIRE_EQUAL(vec.size(), 4);
        for (auto vec : vec) {
            for (std::size_t j = 0; j < 4; j += 1) {
                BOOST_REQUIRE_CLOSE(vec[j], expected[i][j], 1e-6);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testBigVectorMerge) {

    std::size_t dim{10};
    std::size_t numVectors{10};
    std::filesystem::path tmpFile1{createTemporaryFile()};
    BigVector vec1{dim, numVectors, tmpFile1, [i = 0]() mutable {
        return i++;
    }};
    std::filesystem::path tmpFile2{createTemporaryFile()};
    BigVector vec2{dim, numVectors, tmpFile2, [i = dim * numVectors]() mutable {
        return i++;
    }};

    std::filesystem::path tmpFile3{createTemporaryFile()};
    BigVector merged{merge(vec1, vec2, tmpFile3)};

    BOOST_REQUIRE_EQUAL(merged.dim(), 10);
    BOOST_REQUIRE_EQUAL(merged.numVectors(), 20);
    BOOST_REQUIRE_EQUAL(merged.size(), 200);
    float i{0.0F};
    for (auto vec : merged) {
        for (std::size_t j = 0.0; j < 10.0; j += 1.0F) {
            BOOST_REQUIRE_EQUAL(vec[j], i + j);
        }
        i += 10.0F;
    }
}

BOOST_AUTO_TEST_CASE(testParallelReadBigVector) {

    // Create a temporary file.
    char filename[] = "/tmp/test_storage_XXXXXX";
    int ret{::mkstemp(filename)};
    if (ret == -1) {
        BOOST_FAIL("Couldn't create temporary file");
    }
    std::cout << "Created temporary file \"" << filename << "\"" << std::endl;

    std::size_t dim{512};
    std::size_t numVectors{1000000};
    std::filesystem::path tmpFile{filename};
    BigVector vec{dim, numVectors, tmpFile, [dim, i = std::uint64_t{0}]() mutable {
        return i++ / dim;
    }};

    // Read the vectors in parallel checking that the results are correct.
    std::vector<Reader> checker;
    checker.reserve(32);
    std::vector<bool> passed(32, true);
    std::vector<std::vector<std::size_t>> visited(32);
    for (std::size_t i = 0; i < 2; ++i) {
        checker.emplace_back([i, dim, &passed, &visited](std::size_t pos,
                                                         BigVector::VectorReference vec) {
            for (std::size_t j = 0; j < dim; ++j) {
                passed[i] = passed[i] && (vec[j] == static_cast<float>(pos));
            }
            visited[i].push_back(pos);
        });
    }

    parallelRead(vec, checker);

    BOOST_REQUIRE(std::all_of(passed.begin(), passed.end(), [](bool x) { return x; }));

    // Merge the visited vectors.
    std::vector<std::size_t> merged;
    merged.reserve(numVectors);
    for (auto& v : visited) {
        merged.insert(merged.end(), v.begin(), v.end());
    }

    // Check we visited every vector.
    std::sort(merged.begin(), merged.end());
    BOOST_REQUIRE_EQUAL(merged.size(), numVectors);
    for (std::size_t i = 0; i < numVectors; ++i) {
        BOOST_REQUIRE_EQUAL(merged[i], i);
    }
}

BOOST_AUTO_TEST_CASE(testNormalize) {

    std::minstd_rand rng{0};
    std::uniform_real_distribution<float> u01{0.0F, 1.0F};
    std::vector<float> vectors(1000);
    for (std::size_t i = 0; i < vectors.size(); ++i) {
        vectors[i] = u01(rng);
    }
    normalize(10, vectors);
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

BOOST_AUTO_TEST_CASE(testReservoirSample) {

    // Create a temporary file.
    char filename[] = "/tmp/test_storage_XXXXXX";
    int ret{::mkstemp(filename)};
    if (ret == -1) {
        BOOST_FAIL("Couldn't create temporary file");
    }
    std::cout << "Created temporary file \"" << filename << "\"" << std::endl;

    std::size_t dim{96};
    std::size_t numVectors{1500000};
    std::filesystem::path tmpFile{filename};
    BigVector vec{dim, numVectors, tmpFile, [dim, i = std::uint64_t{0}]() mutable {
        return static_cast<float>(i++ / dim);
    }};

    std::vector<float> samples(100 * dim, std::numeric_limits<float>::quiet_NaN());
    std::minstd_rand rng{0};
    ReservoirSampler sampler{dim, 100, rng, samples.begin()};

    for (auto vec : vec) {
        sampler.add(vec.data());
    }

    BOOST_REQUIRE(std::none_of(samples.begin(), samples.end(),
                               [](float x) { return std::isnan(x); }));

    // Check that the vectors we sample are all in the original set.
    // We can do this by checking that each vector is a constant vector
    // whose components are integers in the range [0, 1500000).
    for (std::size_t i = 0; i < samples.size(); i += dim) {
        float value{samples[i]};
        BOOST_REQUIRE(std::all_of(samples.begin() + i, samples.begin() + i + dim,
                                  [value](float x) { return x == value; }));
        BOOST_REQUIRE_CLOSE(value, std::round(value), 1e-6);
        BOOST_REQUIRE(value >= 0.0F);
        BOOST_REQUIRE(value < 1500000.0F);
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