#include "../src/pq.h"
#include "../src/io.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
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

bool testReadDimension() {
    auto file = std::filesystem::path(__FILE__).parent_path() / "dim-vectors.txt";
    auto dim = readDimension(file);
    if (dim != 4) {
        std::cout << "FAILED: output " << dim << " != 4" << std::endl;
        return false;
    }
    return true;
}

bool testReadVectors() {
    auto file = std::filesystem::path(__FILE__).parent_path() / "vectors.csv";
    auto vectors = readVectors(4, file);
    std::ostringstream result;
    result << vectors;
    if (result.str() != "[-1.1,2.1,0.3,1.7,1.2,3.1,-0.9,1.8]") {
        std::cout << "FAILED: output " << vectors << std::endl;
        return false;
    }
    return true;
}

bool testZeroPad() {
    std::vector<float> vectors{1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F,
                               2.0F, 2.0F, 2.0F, 2.0F, 2.0F, 2.0F, 2.0F, 2.0F, 2.0F};
    zeroPad(9, vectors);

    std::ostringstream result;
    result << vectors;
    if (result.str() != "[1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,2,2,2,2,2,2,2,2,2,0,0,0,0,0,0,0]") {
        std::cout << "FAILED: output " << vectors << std::endl;
        return false;
    }
    return true;
}

bool testNormalize() {
    std::minstd_rand rng;
    std::normal_distribution<> norm(2.0, 1.0);

    std::size_t dim{80};
    std::vector<float> vectors(800);
    std::generate_n(vectors.begin(), vectors.size(), [&] { return norm(rng); });

    normalize(dim, vectors);

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

    normalize(dim, docs);
    normalize(dim, query);

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
        auto result = initForgy(10, 100, rng);
        std::vector<std::size_t> centres(result.begin(), result.end());

        if (centres.size() != 10) {
            std::cout << "FAILED: centres " << centres << std::endl;
            passed = false;
            break;
        }
        for (auto centre : centres) {
            if (centre > 99) {
                std::cout << "FAILED: centres " << centres << std::endl;
                passed = false;
                break;
            }
        }
    }

    std::size_t dim{80};
    std::size_t bookDim{dim / 8};

    std::vector<float> docs(8000);
    for (std::size_t i = 0; i < docs.size(); i += dim) {
        for (std::size_t j = 0; j < dim; ++j) {
            docs[i + j] = static_cast<float>(i * bookDim + j / bookDim);
        }
    }

    auto result = initForgy(5, bookDim, dim, docs, rng);

    if (result.size() != 5 * dim) {
        std::cout << "FAILED: size " << result.size()
                  << " != " << 5 * dim << std::endl;
        passed = false;
    }
    for (std::size_t i = 0; i < result.size(); i += 5 * bookDim) {
        std::size_t expectedBook{i / (5 * bookDim)};
        for (std::size_t j = 0; j < 5 * bookDim; ++j) {
            std::size_t book{static_cast<std::size_t>(result[i+j]) % bookDim};
            if (book != expectedBook) {
                std::cout << "FAILED: book@" << j << " " << book
                          << " != " << expectedBook << std::endl;
                passed = false;
                break;
            }
        }
    }

    return passed;
}

bool testStepLloyd() {
    std::size_t numBooks{4};
    std::size_t bookSize{16};
    std::size_t dim{8};
    std::size_t bookDim{dim / numBooks};

    std::minstd_rand rng;
    std::normal_distribution<> norm(0.0, 0.01);
    std::uniform_int_distribution<> uniform(0, bookSize - 1);

    std::vector<float> centres(numBooks * bookSize * bookDim);
    std::vector<float> docs(5 * bookSize * dim);

    for (std::size_t b = 0; b < numBooks; ++b) {
        for (std::size_t i = 0; i < bookSize; ++i) {
            std::size_t book{b * bookSize + i};
            centres[book * bookDim + 0] = std::cos(2.0 * 3.14159 * i / bookSize);
            centres[book * bookDim + 1] = std::sin(2.0 * 3.14159 * i / bookSize);
        }
    }

    std::vector<float> counts(numBooks * bookSize, 0.0F);
    std::vector<float> expectedCentres(centres.size(), 0.0F);
    std::vector<code_t> expectedDocsCodes;

    for (std::size_t i = 0; i < docs.size(); i += dim) {
        std::vector<int> codes(numBooks);
        std::vector<float> noise(dim);
        std::generate_n(&codes[0], numBooks, [&] { return uniform(rng); });
        std::generate_n(&noise[0], dim, [&] { return norm(rng); });
        for (std::size_t b = 0; b < numBooks; ++b) {
            std::size_t book{b * bookSize + codes[b]};
            counts[book] += 1.0F;
            float alpha{(counts[book] - 1.0F) / counts[book]};
            float beta{1.0F - alpha};
            float norm{0.0F};
            for (std::size_t j = 0; j < bookDim; ++j) {
                docs[i + b * bookDim + j] =
                    centres[book * bookDim + j] + noise[b * bookDim + j];
                norm += docs[i + b * bookDim + j] * docs[i + b * bookDim + j];
            }
            norm = std::sqrt(norm);
            for (std::size_t j = 0; j < bookDim; ++j) {
                docs[i + b * bookDim + j] /= norm;
                expectedCentres[book * bookDim + j] = 
                    alpha * expectedCentres[book * bookDim + j] +
                    beta * docs[i + b * bookDim + j];
            }
            expectedDocsCodes.push_back(static_cast<code_t>(codes[b] - offset()));
        }
    }

    std::vector<code_t> docsCodes(expectedDocsCodes);

    stepLloyd(numBooks, bookSize, dim, docs, centres, docsCodes);

    bool passed{true};
    for (std::size_t i = 0; i < centres.size(); ++i) {
        if (std::fabs(centres[i] - expectedCentres[i]) > 1e-6) {
            std::cout << "FAILED: centre@" << i << " " << centres[i]
                      << " != " << expectedCentres[i] << std::endl;
            passed = false;
            break;
        }
    }
    for (std::size_t i = 0; i < docsCodes.size(); ++i) {
        if (docsCodes[i] != expectedDocsCodes[i]) {
            std::cout << "FAILED: centre@" << i << " "
                      << static_cast<int>(docsCodes[i]) << " != "
                      << static_cast<int>(expectedCentres[i]) << std::endl;
            passed = false;
            break;
        }
    }
    
    return passed;
}

bool testComputeDispersion() {
    std::size_t dim{5};
    std::vector<float> centres{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 2.0,
                               3.0, 3.0, 1.0, 3.0, 1.0, 0.0, 4.0, 0.0, 4.0, 4.0};
    std::vector<float> docs{1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 2.0, 1.0, 2.0, 2.0,
                            3.0, 2.5, 3.0, 3.0, 3.5, 0.0, 4.0, 0.0, 4.0, 4.0,
                            1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 4.0,
                            3.5, 2.5, 3.5, 2.0, 3.5, 4.0, 5.0, 4.0, 4.0, 3.0};
    std::vector<code_t> docsCentres{static_cast<code_t>(0 - offset()),
                                    static_cast<code_t>(1 - offset()),
                                    static_cast<code_t>(2 - offset()),
                                    static_cast<code_t>(3 - offset()),
                                    static_cast<code_t>(0 - offset()),
                                    static_cast<code_t>(1 - offset()),
                                    static_cast<code_t>(2 - offset()),
                                    static_cast<code_t>(3 - offset())};

    normalize(dim, centres);
    normalize(dim, docs);

    float dispersion{computeDispersion(dim, centres, docs, docsCentres)};

    if (std::fabs(dispersion - 3.13844) > 1e-6) {
        std::cout << "FAILED: " << dispersion << " != 3.13844" << std::endl;
        return false;
    }
    return true;
}

bool testBuildCodeBook() {
    std::minstd_rand rng;
    std::normal_distribution<> norm(0.0, 2.0);

    std::size_t dim{16};
    std::size_t bookDim{dim / numBooks()};
    std::vector<float> docs(1000 * 16);
    std::generate_n(&docs[0], docs.size(), [&] { return norm(rng); });

    normalize(dim, docs);

    auto [codeBooks, docsCodes] = buildCodeBook(dim, docs);

    float avgDist{0.0F};
    float count{0.0F};
    for (std::size_t i = 0; i < docsCodes.size(); i += numBooks()) {
        float sim{0.0F};
        for (std::size_t b = 0; b < numBooks(); ++b) {
            std::size_t book{b * bookSize() + (offset() + docsCodes[i + b])};
            for (std::size_t j = 0; j < bookDim; ++j) {
                sim += codeBooks[book * bookDim + j] * docs[(i + b) * bookDim + j];
            }
        }
        avgDist += 1.0F - sim;
        count += 1.0F;
    }
    avgDist /= count;
    if (avgDist > 0.03) {
        std::cout << "FAILED: dist " << avgDist << " > 0.03" << std::endl;
        return false;
    }
    return true;
}

bool testBuildDistTable() {
    std::minstd_rand rng;
    std::normal_distribution<> norm(2.0, 1.0);

    std::size_t dim{80};
    std::size_t bookDim{dim / numBooks()};
    std::vector<float> codeBooks(bookSize() * dim);
    std::vector<float> query(dim);

    std::generate_n(codeBooks.begin(), codeBooks.size(), [&] { return norm(rng); });
    std::generate_n(query.begin(), query.size(), [&] { return norm(rng); });

    normalize(dim, codeBooks);
    normalize(dim, query);

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

bool testComputeDist() {
    std::minstd_rand rng;
    std::normal_distribution<> norm(2.0, 1.0);

    std::vector<float> table(numBooks() * bookSize());
    std::vector<int> rawCodes{249,
                              1 * bookSize() + 127,
                              2 * bookSize() + 75,
                              3 * bookSize() + 211,
                              4 * bookSize() + 3,
                              5 * bookSize() + 179,
                              6 * bookSize() + 98,
                              7 * bookSize() + 33};

    std::generate_n(&table[0], table.size(), [&] { return norm(rng); });

    std::vector<code_t> codes;
    float expectedDist{0.0F};
    for (auto i : rawCodes) {
        codes.push_back(static_cast<code_t>((i % bookSize()) - offset()));
        expectedDist += table[i];
    }
    expectedDist = 1.0F - expectedDist;

    float dist{computeDist(table, &codes[0])};

    if (std::fabs(dist - expectedDist) > 1e-6) {
        std::cout << "FAILED: dist " << dist << " != " << expectedDist << std::endl;
        return false;
    }
    return true;
}

#define RUN_TEST(x)                                      \
    do {                                                 \
    std::cout << "Running " << #x << " ";                \
    if ((x)()) { std::cout << "PASSED!" << std::endl;} } \
    while (false)
}

void runUnitTests() {
    RUN_TEST(testReadDimension);
    RUN_TEST(testReadVectors);
    RUN_TEST(testZeroPad);
    RUN_TEST(testNormalize);
    RUN_TEST(testSearchBruteForce);
    RUN_TEST(testInitForgy);
    RUN_TEST(testStepLloyd);
    RUN_TEST(testComputeDispersion);
    RUN_TEST(testBuildCodeBook);
    RUN_TEST(testBuildDistTable);
    RUN_TEST(testComputeDist);
}