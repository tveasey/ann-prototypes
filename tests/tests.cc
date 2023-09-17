#include "../src/bruteforce.h"
#include "../src/io.h"
#include "../src/pq.h"
#include "../src/scalar.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {
template<typename T>
std::vector<T> toVector(std::priority_queue<T>& queue) {
    std::vector<T> result;
    result.reserve(queue.size());
    while (!queue.empty()) {
        result.push_back(queue.top());
        queue.pop();
    }
    return result;
}

bool testReadFvecs() {
    auto file = std::filesystem::path(__FILE__).parent_path() / "vectors.fvec";
    auto [vectors, dim] = readFvecs(file);
    if (dim != 4) {
        std::cout << "FAILED: output " << dim << " != 4" << std::endl;
        return false;
    }
    std::ostringstream result;
    result << vectors;
    if (result.str() != "[-1.1,2.1,0.3,1.7,1.2,3.1,-0.9,1.8]") {
        std::cout << "FAILED: output " << vectors << std::endl;
        return false;
    }
    return true;
}

bool testPacking() {

    std::minstd_rand rng;
    std::uniform_int_distribution<> u{0, 15};

    for (std::size_t t = 0; t < 100; ++t) {
        std::vector<std::uint8_t> x(45);
        std::vector<std::uint8_t> xp((x.size() + 1) / 2);
        std::generate_n(x.begin(), x.size(), [&] { return u(rng); });

        std::size_t rem{x.size() & 0x1F};
        std::size_t dim{x.size() - rem};
        for (std::size_t i = 0; i < x.size(); i += 32) {
            pack4B(32, &x[i], &xp[i >> 1]);
        }
        pack4B(rem, &x[dim], &xp[dim >> 1]);

        std::vector<std::uint8_t> xu(x.size());
        for (std::size_t i = 0; i < x.size(); i += 32) {
            unpack4B(32, &xp[i >> 1], &xu[i]);
        }
        unpack4B(rem, &xp[dim >> 1], &xu[dim]);

        if (x != xu) {
            std::cout << "FAILED: mismatch " << std::vector<int>(x.begin(), x.end())
                      << " != " << std::vector<int>(xu.begin(), xu.end()) << std::endl;
            return false;
        }
    }
    return true;
}

bool testDot4B() {

    std::minstd_rand rng;
    std::uniform_int_distribution<> u{0, 15};

    for (std::size_t t = 0; t < 100; ++t) {
        std::vector<std::uint8_t> x(32);
        std::vector<std::uint8_t> y(x.size());
        std::vector<std::uint8_t> yp(x.size() / 2);
        std::generate_n(x.begin(), x.size(), [&] { return u(rng); });
        std::generate_n(y.begin(), y.size(), [&] { return u(rng); });

        std::uint32_t expectedDot{0};
        for (std::size_t i = 0; i < x.size(); ++i) {
            expectedDot += x[i] * y[i];
        }

        auto dot = dot4B(x.size(), x.data(), y.data());

        for (std::size_t i = 0; i < y.size(); i += 32) {
            pack4B(32, &y[i], &yp[i >> 1]);
        }

        auto dotp = dot4BP(x.size(), x.data(), yp.data());

        if (dot != expectedDot) {
            std::cout << "FAILED: mismatch " << dot << " != " << expectedDot << std::endl;
            return false;
        }
        if (dotp != expectedDot) {
            std::cout << "FAILED: packed mismatch " << dotp << " != " << expectedDot << std::endl;
            return false;
        }
    }

    // Remainder.
    for (std::size_t t = 0; t < 100; ++t) {
        std::vector<std::uint8_t> x(47);
        std::vector<std::uint8_t> y(x.size());
        std::vector<std::uint8_t> yp((x.size() + 1) / 2);
        std::generate_n(x.begin(), x.size(), [&] { return u(rng); });
        std::generate_n(y.begin(), y.size(), [&] { return u(rng); });

        std::uint32_t expectedDot{0};
        for (std::size_t i = 0; i < x.size(); ++i) {
            expectedDot += x[i] * y[i];
        }

        auto dot = dot4B(x.size(), x.data(), y.data());

        for (std::size_t i = 0; i < (y.size() & ~0x1F); i += 32) {
            pack4B(32, &y[i], &yp[i >> 1]);
        }
        pack4B(y.size() & 0x1F, &y[y.size() & ~0x1F], &yp[(y.size() & ~0x1F) >> 1]);

        auto dotp = dot4BP(x.size(), x.data(), yp.data());

        if (dot != expectedDot) {
            std::cout << "FAILED: mismatch " << dot << " != " << expectedDot << std::endl;
            return false;
        }
        if (dotp != expectedDot) {
            std::cout << "FAILED: packed mismatch " << dotp << " != " << expectedDot << std::endl;
            return false;
        }
    }

    // Long.
    for (std::size_t t = 0; t < 100; ++t) {
        std::vector<std::uint8_t> x(5030);
        std::vector<std::uint8_t> y(x.size());
        std::vector<std::uint8_t> yp((x.size() + 1) / 2);
        std::generate_n(x.begin(), x.size(), [&] { return u(rng); });
        std::generate_n(y.begin(), y.size(), [&] { return u(rng); });

        std::uint32_t expectedDot{0};
        for (std::size_t i = 0; i < x.size(); ++i) {
            expectedDot += x[i] * y[i];
        }

        auto dot = dot4B(x.size(), x.data(), y.data());

        for (std::size_t i = 0; i < (y.size() & ~0x1F); i += 32) {
            pack4B(32, &y[i], &yp[i >> 1]);
        }
        pack4B(y.size() & 0xF, &y[y.size() & ~0x1F], &yp[(y.size() & ~0x1F) >> 1]);

        auto dotp = dot4BP(x.size(), x.data(), yp.data());

        if (dot != expectedDot) {
            std::cout << "FAILED: mismatch " << dot << " != " << expectedDot << std::endl;
            return false;
        }
        if (dotp != expectedDot) {
            std::cout << "FAILED: packed mismatch " << dotp << " != " << expectedDot << std::endl;
            return false;
        }
    }

    return true;
}

bool testDot8B() {

    std::minstd_rand rng;
    std::uniform_int_distribution<> u{0, 15};

    for (std::size_t t = 0; t < 100; ++t) {
        std::vector<std::uint16_t> x(128);
        std::vector<std::uint8_t> y(x.size());
        std::generate_n(x.begin(), x.size(), [&] { return u(rng); });
        std::generate_n(y.begin(), y.size(), [&] { return u(rng); });

        std::uint32_t expectedDot{0};
        for (std::size_t i = 0; i < x.size(); ++i) {
            expectedDot += x[i] * y[i];
        }

        auto dot = dot8B(x.size(), x.data(), y.data());

        if (dot != expectedDot) {
            std::cout << "FAILED: mismatch " << dot << " != " << expectedDot << std::endl;
            return false;
        }
    }

    return true;
}

std::uint32_t dot8B(std::size_t dim,
                    const std::uint16_t*__restrict x,
                    const std::uint8_t*__restrict y);


bool testQuantiles() {
    std::vector<float> docs(1000);
    std::iota(docs.begin(), docs.end(), 0.0F);
    auto q = quantiles(100, docs, 0.9);
    if (q.first != 50.0F || q.second != 950.0F) {
        std::cout << "FAILED: unexpected quantiles " << q << std::endl;
        return false;
    }
    std::reverse(docs.begin(), docs.end());
    q = quantiles(100, docs, 0.9);
    if (q.first != 50.0F || q.second != 950.0F) {
        std::cout << "FAILED: unexpected quantiles " << q << std::endl;
        return false;
    }
    return true;
}

bool testScalar8BitQuantise() {
    std::size_t dim{128};

    std::minstd_rand rng;
    std::uniform_real_distribution<> u{-5.0, 5.0};

    for (std::size_t i = 0; i < 1000; ++i) {

        float lower{-4.9};
        float upper{4.9};

        std::vector<float> doc(dim);
        std::generate_n(doc.begin(), dim, [&] { return u(rng); });

        auto [quantised, p1] = scalarQuantise8B({lower, upper}, dim, doc);

        auto dequantised = scalarDequantise8B({lower, upper}, dim, quantised);

        float norm{0.0F};
        float avg{0.0F};
        float rmse{0.0F};
        for (std::size_t i = 0; i < dim; ++i) {
            float r{doc[i] - dequantised[i]};
            norm += doc[i] * doc[i];
            avg += r;
            rmse += r * r;
        }
        avg /= static_cast<float>(dim);
        rmse = std::sqrtf(rmse / static_cast<float>(dim));
        float tol{4.0F * rmse / std::sqrt(static_cast<float>(dim))};
        if (avg > tol) {
            std::cout << "FAILED: maybe biased " << avg << " > " << tol << std::endl;
            return false;
        }
        if (rmse > 0.001F * norm) {
            std::cout << "FAILED: large error " << rmse << " > " << 0.001F * norm << std::endl;
            return false;
        }
    }


    std::vector<float> docs(300 * dim);
    for (std::size_t i = 0; i < 10; ++i) {
        std::generate_n(docs.begin() + i * docs.size() / 20, docs.size() / 20,
                        [&] { return 5.0F + 2.0F * static_cast<float>(i) + u(rng); });
    }
    normalise(dim, docs);
    float lower{*std::min_element(docs.begin(), docs.end())};
    float upper{*std::max_element(docs.begin(), docs.end())};
    float invScale{(upper - lower) / 255.0F};
    float p2{invScale * invScale};

    auto [quantised, p1] = scalarQuantise8B({lower, upper}, dim, docs);
    auto dequantised = scalarDequantise8B({lower, upper}, dim, quantised);

    float ei{0.0F};
    float ef{0.0F};
    for (std::size_t i = 0, id = 0; i < docs.size(); i += dim) {
        for (std::size_t j = i; j < docs.size(); j += dim) {
            float sim{0.0F};
            std::uint32_t simqi{0};
            float simqf{0.0F};
            for (std::size_t k = 0; k < dim; ++k) {
                sim   += docs[i + k] * docs[j + k];
                simqi += static_cast<std::uint16_t>(quantised[i + k]) *
                         static_cast<std::uint16_t>(quantised[j + k]);
                simqf += dequantised[i + k] * dequantised[j + k];
            }
            float simq{p1[i / dim] + p1[j / dim] + p2 * static_cast<float>(simqi)};
            ei += std::fabs(sim - simq);
            ef += std::fabs(sim - simqf);
        }
    }

    if (ei > 0.5F * ef) {
        std::cout << "FAILED: dot error " << ei << " > " << 0.5F * ef << std::endl;
        return false;
    }
    return true;
}

bool testZeroPad() {
    std::vector<float> vectors{1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F,
                               2.0F, 2.0F, 2.0F, 2.0F, 2.0F, 2.0F, 2.0F, 2.0F, 2.0F};
    zeroPad(9, vectors);

    if (vectors.size() != 2 * numBooks()) {
        std::cout << "FAILED: output " << vectors << std::endl;
        return false;
    }

    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 9; ++j) {
            if (vectors[i * numBooks() + j] != static_cast<float>(i + 1)) {
                std::cout << "FAILED: output " << vectors << std::endl;
                return false;
            }
        }
        for (std::size_t j = 9; j < numBooks(); ++j) {
            if (vectors[i * numBooks() + j] != 0.0F) {
                std::cout << "FAILED: output " << vectors << std::endl;
                return false;
            }
        }
    }
    return true;
}

bool testNormalise() {
    std::minstd_rand rng;
    std::normal_distribution<> norm(2.0, 1.0);

    std::size_t dim{80};
    std::vector<float> vectors(800);
    std::generate_n(vectors.begin(), vectors.size(), [&] { return norm(rng); });

    normalise(dim, vectors);

    bool passed{true};
    for (std::size_t i = 0; i < vectors.size(); i += dim) {
        float norm{0.0F};
        for (std::size_t j = 0; j < dim; ++j) {
            norm += vectors[j] * vectors[j];
        }
        if (std::fabs(norm - 1.0F) > 1e-6) {
            std::cout << "FAILED: norm " << norm << std::endl;
            passed = false;
            break;
        }
    }
    return passed;
}

bool testSearchBruteForce() {
    std::minstd_rand rng;
    std::normal_distribution<> norm(2.0, 1.0);

    std::size_t dim{80};
    std::vector<float> docs(10 * dim);
    std::generate_n(docs.begin(), docs.size(), [&] { return norm(rng); });
    std::vector<float> query(dim);
    std::generate_n(query.begin(), query.size(), [&] { return norm(rng); });

    normalise(dim, docs);
    normalise(dim, query);

    std::priority_queue<std::pair<float, std::size_t>> topk;
    searchBruteForce(5, docs, query, topk);

    auto result = toVector(topk);
    std::vector<std::size_t> indices(5);
    std::transform(result.begin(), result.end(),
                    indices.begin(),
                    [](const auto& p) { return p.second; });
    std::sort(indices.begin(), indices.end());

    bool passed{true};
    if (result.size() != 5) {
        std::cout << "FAILED: output " << result << std::endl;
        passed = false;
    }
    for (std::size_t i = 1; i < result.size(); ++i) {
        if (result[i].first > result[i-1].first) {
            std::cout << "FAILED: output " << result << std::endl;
            passed = false;
            break;
        }
    }
    for (std::size_t i = 0; i < 10; ++i) {
        if (!std::binary_search(indices.begin(), indices.end(), i)) {
            float sim{0.0F};
            for (std::size_t j = 0; j < dim; ++j) {
                sim += query[j] * docs[i * dim + j];
            }
            float dist{1.0F - sim};
            if (dist < result[0].first) {
                std::cout << "FAILED: dist " << dist
                          << " < " << result[0].first << std::endl;
                passed = false;
                break;
            }
        }
    }
    return passed;
}

bool testInitForgy() {
    std::minstd_rand rng;

    bool passed{true};
    for (std::size_t i = 0; i < 10; ++i) {
        auto result = initForgy(1000, rng);
        std::vector<std::size_t> centres(result.begin(), result.end());

        if (centres.size() != bookSize()) {
            std::cout << "FAILED: centres " << centres << std::endl;
            passed = false;
            break;
        }
        for (auto centre : centres) {
            if (centre > 999) {
                std::cout << "FAILED: centres " << centres << std::endl;
                passed = false;
                break;
            }
        }
    }

    std::size_t bookDim{2};
    std::size_t dim{bookDim * numBooks()};
    std::vector<float> bookValues(numBooks());
    std::iota(bookValues.begin(), bookValues.end(), 0.0F);

    std::vector<float> docs(1000 * dim);
    for (std::size_t i = 0; i < docs.size(); i += dim) {
        for (std::size_t b = 0; b < numBooks(); ++b) {
            for (std::size_t j = 0; j < bookDim; ++j) {
                docs[i + b * bookDim + j] = bookValues[b];
            }
        }
    }

    auto centres = initForgy(dim, docs, rng);

    if (centres.size() != bookSize() * dim) {
        std::cout << "FAILED: size " << centres.size()
                  << " != " << bookSize() * dim << std::endl;
        passed = false;
    }
    for (std::size_t b = 0; b < numBooks(); ++b) {
        for (std::size_t i = 0; i < bookSize(); ++i) {
            for (std::size_t j = 0; j < bookDim; ++j) {
                float actual{centres[(b * bookSize() + i) * bookDim + j]};
                if (actual != bookValues[b]) {
                    std::cout << "FAILED: book " << b << "@" << i << " "
                              << actual << " != " << bookValues[b] << std::endl;
                    passed = false;
                }
            }
        }
    }

    return passed;
}

bool testStepLloyd() {
    std::minstd_rand rng;

    std::size_t bookDim{2};
    std::size_t dim{bookDim * numBooks()};
    std::vector<float> bookCentres(numBooks());
    for (std::size_t b = 1; b <= numBooks(); ++b) {
        bookCentres[b - 1] = 1000.0F * static_cast<float>(b);
    }

    std::vector<float> docs(20 * bookSize() * dim);
    std::normal_distribution<> norm{0.0, 0.1};
    std::uniform_int_distribution<> cluster{0, bookSize() - 1};
    for (std::size_t i = 0; i < docs.size(); i += dim) {
        for (std::size_t b = 0; b < numBooks(); ++b) {
            float offset{bookCentres[b] + static_cast<float>(2 * cluster(rng) - bookSize())};
            for (std::size_t j = 0; j < bookDim; ++j) {
                docs[i + b * bookDim + j] = offset + static_cast<float>(norm(rng));
            }
        }
    }

    auto centres = initForgy(dim, docs, rng);

    std::vector<code_t> docsCodes(20 * bookSize() * numBooks());
    auto calculateErrors = [&] {
        auto doc = docs.begin();
        float maxMse{0.0F};
        float totMse{0.0F};
        for (std::size_t i = 0; i < docsCodes.size(); i += numBooks(), doc += dim) {
            for (std::size_t b = 0; b < numBooks(); ++b) {
                float mse{0.0F};
                for (std::size_t j = 0; j < bookDim; ++j) {
                    mse += std::powf(
                        docs[b * bookDim + j] -
                        centres[(b * bookSize() + docsCodes[i + b]) * bookDim + j], 2.0F);
                }
                mse = std::sqrtf(mse);
                maxMse = std::max(maxMse, mse);
                totMse += mse;
            }
        }
        return std::make_pair(maxMse, totMse);
    };

    // Check usual k-means invariants.
    bool passed{true};
    float lastMaxMse{std::numeric_limits<float>::max()};
    float lastTotMse{std::numeric_limits<float>::max()};
    for (std::size_t i = 0; i < 5; ++i) {
        stepLloyd(dim, docs, centres, docsCodes);
        auto [maxMse, totMse] = calculateErrors();
        if (maxMse > lastMaxMse) {
            std::cout << "FAILED: " << maxMse << " > " << lastMaxMse << std::endl;
            passed = false;
        }
        if (totMse > lastTotMse) {
            std::cout << "FAILED: " << totMse << " > " << lastTotMse << std::endl;
            passed = false;
        }
    }

    return passed;
}

bool testStepScann() {
    std::minstd_rand rng;

    std::size_t bookDim{3};
    std::size_t dim{bookDim * numBooks()};

    std::vector<float> docs(20 * bookSize() * dim);
    std::normal_distribution<> norm{10.0, 2.0};
    std::generate_n(docs.begin(), docs.size(), [&] { return norm(rng); });
    auto docsNorms2 = norms2(dim, docs);

    auto centresKMeans = initForgy(dim, docs, rng);
    auto centresScann = centresKMeans;

    std::vector<code_t> docsCodesKMeans(20 * bookSize() * numBooks());
    for (std::size_t i = 0; i < 20; ++i) {
        stepLloyd(dim, docs, centresKMeans, docsCodesKMeans);
    }

    std::vector<code_t> docsCodesScann(20 * bookSize() * numBooks());
    for (std::size_t i = 0; i < 20; ++i) {
        stepScann(0.4F, dim, docs, docsNorms2, centresScann, docsCodesScann);
    }

    return true;
}

bool testBuildCodeBook() {
    std::minstd_rand rng;
    std::normal_distribution<> norm(0.0, 2.0);

    std::size_t bookDim{2};
    std::size_t dim{bookDim * numBooks()};
    std::vector<float> docs(1000 * dim);
    std::generate_n(&docs[0], docs.size(), [&] { return norm(rng); });

    normalise(dim, docs);

    auto [codeBooks, docsCodes] = buildCodeBook(dim, 1.0, docs, 10);

    float rmse{std::sqrtf(quantisationMseLoss(dim, codeBooks, docs, docsCodes))};

    if (rmse > 0.1) {
        std::cout << "FAILED: dist " << rmse << " > 0.03" << std::endl;
        return false;
    }
    return true;
}

bool testBuildDistTable() {
    std::minstd_rand rng;
    std::normal_distribution<> norm(2.0, 1.0);

    std::size_t bookDim{10};
    std::size_t dim{bookDim * numBooks()};
    std::vector<float> codeBooks(bookSize() * dim);
    std::vector<float> query(dim);

    std::generate_n(codeBooks.begin(), codeBooks.size(), [&] { return norm(rng); });
    std::generate_n(query.begin(), query.size(), [&] { return norm(rng); });

    normalise(dim, codeBooks);
    normalise(dim, query);

    std::vector<float> table{buildDistTable(codeBooks, query)};

    std::vector<float> expectedTable;
    for (std::size_t b = 0; b < numBooks(); ++b) {
        for (std::size_t i = 0; i < bookSize(); ++i) {
            float sim{0.0F};
            for (std::size_t j = 0; j < bookDim; ++j) {
                sim += codeBooks[(b * bookSize() + i) * bookDim + j] *
                       query[b * bookDim + j];
            }
            expectedTable.push_back(sim);
        }
    }

    bool passed{true};
    if (table.size() != expectedTable.size()) {
        std::cout << "FAILED: size " << table.size()
                  << " != " << expectedTable.size() << std::endl;
        passed = false;
    } else {
        for (std::size_t i = 0; i < table.size(); ++i) {
            if (std::fabs(table[i] - expectedTable[i]) > 1e-6) {
                std::cout << "FAILED: dist@" << i << " " << table[i]
                          << " != " << expectedTable[i] << std::endl;
                passed = false;
                break;
            }
        }
    }
    return passed;
}

#define RUN_TEST(x)                                      \
    do {                                                 \
    std::cout << "Running " << #x << " ";                \
    if ((x)()) { std::cout << "PASSED!" << std::endl;} } \
    while (false)
}

void runUnitTests() {
    RUN_TEST(testReadFvecs);
    RUN_TEST(testPacking);
    RUN_TEST(testDot4B);
    RUN_TEST(testDot8B);
    RUN_TEST(testQuantiles);
    RUN_TEST(testScalar8BitQuantise);
    RUN_TEST(testZeroPad);
    RUN_TEST(testNormalise);
    RUN_TEST(testSearchBruteForce);
    RUN_TEST(testInitForgy);
    RUN_TEST(testStepLloyd);
    RUN_TEST(testStepScann);
    RUN_TEST(testBuildCodeBook);
    RUN_TEST(testBuildDistTable);
}