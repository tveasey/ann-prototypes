#include "pq.h"
#include "metrics.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <ios>
#include <iostream>
#include <limits>
#include <map>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// Notes
// -----
//
// * Hardcode dot product: other metrics should perform similarly.
// * Data-oriented design to maximize memory access efficiency.
// * Hardcode all loop limits where possible since this helps the
//   compiler vectorise code.
// * Brute force k-means performs well for moderate k.

namespace {
constexpr int NUM_BOOKS{8};
constexpr int BOOK_SIZE{256};
constexpr int OFFSET{[] {
    if constexpr (sizeof(code_t) == 1) {
        return 127;
    }
    return std::min((1 << (8 * sizeof(code_t) - 1)) - BOOK_SIZE, 0);
}()};
constexpr std::size_t K_MEANS_ITR{12};
}

int numBooks() {
    return NUM_BOOKS;
}

int bookSize() {
    return BOOK_SIZE;
}

int offset() {
    return OFFSET;
}

std::size_t kMeansItr() {
    return K_MEANS_ITR;
}

void zeroPad(std::size_t dim, std::vector<float>& vectors) {
    // Zero pad the vectors so their dimension is a multiple of NUM_BOOKS.
    if (dim % NUM_BOOKS != 0) {
        std::size_t numVectors{vectors.size() / dim};
        std::size_t paddedDim{NUM_BOOKS * ((dim + NUM_BOOKS - 1) / NUM_BOOKS)};
        vectors.resize(paddedDim * numVectors, 0.0F);
        for (std::size_t i = numVectors; i > 1; --i) {
            std::copy(&vectors[dim * (i - 1)], &vectors[dim * i],
                      &vectors[paddedDim * (i - 1)]);
        }
        std::fill(&vectors[dim], &vectors[paddedDim], 0.0F);
    }
}

void normalize(std::size_t dim, std::vector<float>& vectors) {
    // Ensure vectors are unit (Euclidean) norm.
    for (std::size_t i = 0; i < vectors.size(); i += dim) {
        float norm{0.0F};
        #pragma clang loop vectorize(assume_safety)
        for (std::size_t j = 0; j < dim; ++j) {
            norm += vectors[i + j] * vectors[i + j];
        }
        norm = std::sqrt(norm);
        #pragma clang loop vectorize(assume_safety)
        for (std::size_t j = 0; j < dim; ++j) {
            vectors[i + j] /= norm;
        }
    }
}

std::set<std::size_t> initForgy(std::size_t k,
                                std::size_t numDocs,
                                std::minstd_rand& rng) {
    std::set<std::size_t> selection;
    int n(static_cast<int>(numDocs));
    std::uniform_int_distribution<> u0n{0, n - 1};
    while (selection.size() < k) {
        std::size_t cand{static_cast<std::size_t>(u0n(rng))};
        selection.insert(cand);
    }
    return selection;
}

std::vector<float> initForgy(std::size_t k,
                             std::size_t bookDim,
                             std::size_t dim,
                             const std::vector<float>& docs,
                             std::minstd_rand& rng) {
    auto selection = initForgy(k, docs.size() / dim, rng);
    std::vector<float> centres(k * dim);
    auto centre = centres.begin();
    for (std::size_t b = 0; b < NUM_BOOKS; ++b) {
        for (auto i : selection) {
            const auto* doc = &docs[i * dim];
            std::copy(doc + b * bookDim, doc + (b + 1) * bookDim, centre);
            centre += bookDim;
        }
    }
    return centres;
}

void stepLloyd(std::size_t numBooks,
               std::size_t bookSize,
               std::size_t dim,
               const std::vector<float>& docs,
               std::vector<float>& centres,
               std::vector<code_t>& docsCodes) {

    // Since we sum up non-negative losses from each book we can compute
    // the optimal codebooks independently.

    std::size_t bookDim{dim / numBooks};

    std::vector<std::size_t> bookCounts(numBooks * bookSize, 0);
    std::vector<float> newCentres(centres.size(), 0.0F);

    std::size_t pos{0};
    for (auto doc = docs.begin(); doc != docs.end(); doc += dim, pos += numBooks) {
        for (std::size_t b = 0; b < numBooks; ++b) {
            // Find the nearest centroid.
            int nearestCentre{0};
            float minDist{std::numeric_limits<float>::max()};
            for (int i = 0; i < bookSize; ++i) {
                float dist{0.0F};
                for (std::size_t j = 0; j < bookDim; ++j) {
                    float dij{centres[(b * bookSize + i) * bookDim + j] - doc[b * bookDim + j]};
                    dist += dij * dij;
                }
                if (dist < minDist) {
                    nearestCentre = i;
                    minDist = dist;
                }
            }

            // Update the centroid.
            std::size_t nearestBook{b * bookSize + nearestCentre};
            auto& bookCount = bookCounts[nearestBook];
            auto* newCentre = &newCentres[nearestBook * bookDim];
            // Switch to double so epsilon << 1 / "largest possible cluster".
            double alpha{static_cast<double>(bookCount) / static_cast<double>(bookCount + 1)};
            double beta{1.0 - alpha};
            for (std::size_t j = 0; j < bookDim; ++j) {
                newCentre[j] = static_cast<float>(alpha * newCentre[j] + beta * doc[b * bookDim + j]);
            }
            ++bookCount;

            // Encode the document.
            docsCodes[pos + b] = static_cast<code_t>(nearestCentre - OFFSET);
        }
    }

    centres = std::move(newCentres);
}

void stepScann(float t,
               std::size_t numBooks,
               std::size_t bookSize,
               std::size_t dim,
               const std::vector<float>& docs,
               std::vector<float>& centres,
               std::vector<code_t>& docsCodes) {

    // Since we sum up non-negative losses from each book we can compute
    // the optimal codebooks independently.

    std::size_t bookDim{dim / numBooks};

    // See theorem 3.4 https://arxiv.org/pdf/1908.10396.pdf. This assumes
    // vectors are normalised.
    float eta{static_cast<float>(dim - 1) * t / (1 - t)};

    std::vector<float> bookCounts(numBooks * bookSize, 0.0F);
    std::vector<float> newCentres(centres.size(), 0.0F);

    std::size_t pos{0};
    for (auto doc = docs.begin(); doc != docs.end(); doc += dim, pos += numBooks) {
        for (std::size_t b = 0; b < numBooks; ++b) {
            // Find the centroid which minimizes the anisotropic loss.
            int nearestCentre{0};
            float minDist{std::numeric_limits<float>::max()};
            for (int i = 0; i < bookSize; ++i) {
                float distParallel{0.0F};
                float distPerpendicular{0.0F};
                for (std::size_t j = 0; j < bookDim; ++j) {
                    float cij{centres[(b * bookSize + i) * bookDim + j]};
                    float xj{doc[b * bookDim + j]};
                    float dij{xj - cij};
                    // TODO we need to have per book vector norms.
                    float dpij{xj * xj * dij};
                    distParallel += dpij;
                    distPerpendicular += dij - dpij;
                }
                if (eta * distParallel + distPerpendicular < minDist) {
                    nearestCentre = i;
                    minDist = eta * distParallel + distPerpendicular;
                }
            }

            // TODO Centroid update needs to maintain partition statistics.

            // Encode the document.
            docsCodes[pos + b] = static_cast<code_t>(nearestCentre - OFFSET);
        }
    }

    centres = std::move(newCentres);
}

std::pair<std::vector<float>, std::vector<code_t>>
buildCodeBook(std::size_t dim,
              const std::vector<float>& docs,
              std::size_t iterations) {
    std::size_t bookDim{dim / NUM_BOOKS};
    std::minstd_rand rng;
    std::vector<float> codeBooks{initForgy(BOOK_SIZE, bookDim, dim, docs, rng)};
    std::vector<code_t> docsCodes(docs.size() / bookDim);
    for (std::size_t i = 0; i < iterations; ++i) {
        stepLloyd(NUM_BOOKS, BOOK_SIZE, dim, docs, codeBooks, docsCodes);
    }
    return std::make_pair(std::move(codeBooks), std::move(docsCodes));
}

std::pair<std::vector<float>, std::vector<code_t>>
buildCodeBookScann(float t,
                   std::size_t dim,
                   const std::vector<float>& docs,
                   std::size_t iterations) {
    // Initialize with k-means.
    auto [codeBooks, docsCodes] = buildCodeBook(dim, docs, iterations / 2);

    // Fine-tune with anisotropic loss.
    for (std::size_t i = 0; i < iterations / 2 + iterations % 2; ++i) {
        stepScann(t, NUM_BOOKS, BOOK_SIZE, dim, docs, codeBooks, docsCodes);
    }

    return std::make_pair(std::move(codeBooks), std::move(docsCodes));
}

double quantisationMse(std::size_t dim,
                       const std::vector<float>& codeBooks,
                       const std::vector<float>& docs,
                       const std::vector<code_t>& docsCodes) {

    double count{0.0};
    double avgMse{0.0};

    std::size_t bookDim{dim / NUM_BOOKS};
    auto docCodes = docsCodes.begin();
    for (auto doc = docs.begin();
         doc != docs.end();
         doc += dim, docCodes += NUM_BOOKS, count += 1.0) {

        float mse{0.0F};
        for (std::size_t b = 0; b < NUM_BOOKS; ++b) {
            #pragma clang loop vectorize(assume_safety)
            for (std::size_t i = 0; i < bookDim; ++i) {
                float di{doc[b * bookDim + i] -
                         codeBooks[(BOOK_SIZE * b + OFFSET + docCodes[b]) * bookDim + i]};
                mse += di * di;
            }
        }

        double alpha{count / (count + 1.0)};
        double beta{1.0 - alpha};
        avgMse = alpha * avgMse + beta * mse;
    }

    return avgMse;
}

std::vector<float> buildDistTable(const std::vector<float>& codeBooks,
                                  const std::vector<float>& query) {
    std::size_t bookDim{query.size() / NUM_BOOKS};
    std::vector<float> distTable(NUM_BOOKS * BOOK_SIZE);
    for (std::size_t b = 0; b < NUM_BOOKS; ++b) {
        for (std::size_t i = 0; i < BOOK_SIZE; ++i) {
            float sim{0.0F};
            #pragma clang loop vectorize(assume_safety)
            for (std::size_t j = 0; j < bookDim; ++j) {
                sim += query[b * bookDim + j] *
                       codeBooks[(BOOK_SIZE * b + i) * bookDim + j];
            }
            distTable[BOOK_SIZE * b + i] = sim;
        }
    }
    return distTable;
}

std::vector<float> buildDistNorm2Table(const std::vector<float>& codeBooks,
                                       const std::vector<float>& query) {
    std::size_t bookDim{query.size() / NUM_BOOKS};
    std::vector<float> distTable(2 * NUM_BOOKS * BOOK_SIZE);
    for (std::size_t b = 0; b < NUM_BOOKS; ++b) {
        for (std::size_t i = 0; i < BOOK_SIZE; ++i) {
            float sim{0.0F};
            float norm2{0.0F};
            #pragma clang loop vectorize(assume_safety)
            for (std::size_t j = 0; j < bookDim; ++j) {
                float cij{codeBooks[(BOOK_SIZE * b + i) * bookDim + j]};
                sim += query[b * bookDim + j] * cij;
                norm2 += cij * cij;
            }
            auto* t = &distTable[2 * (BOOK_SIZE * b + i)];
            t[0] = sim;
            t[1] = norm2;
        }
    }
    return distTable;    
}

float computeDist(const std::vector<float>& distTable,
                  const code_t* docCode) {
    float sim{0.0F};
    #pragma clang loop unroll_count(8)
    for (std::size_t b = 0; b < NUM_BOOKS; ++b) {
        sim += distTable[BOOK_SIZE * b + OFFSET + docCode[b]];
    }
    return 1.0F - sim;
}

float computeNormedDist(const std::vector<float>& distTable,
                        const code_t* docCode) {
    float sim{0.0F};
    float norm2{0.0F};
    #pragma clang loop unroll_count(8)
    for (std::size_t b = 0; b < NUM_BOOKS; ++b) {
        const auto* t = &distTable[2 * (BOOK_SIZE * b + OFFSET + docCode[b])];
        sim += t[0];
        norm2 += t[1];
    }
    // Project the quantised document vector back onto unit sphere.
    return 1.0F - sim / std::sqrt(norm2);
}

void searchPQ(std::size_t k,
              const std::vector<float>& codeBooks,
              const std::vector<code_t>& docsCodes,
              const std::vector<float>& query,
              bool normalise,
              std::priority_queue<std::pair<float, std::size_t>>& topk) {

    std::size_t dim{query.size()};
    std::size_t bookDim{dim / NUM_BOOKS};

    auto distTable = normalise ?
        buildDistNorm2Table(codeBooks, query) :
        buildDistTable(codeBooks, query);

    std::size_t docId{0};
    auto docCodes = docsCodes.begin();
    for (/**/; docId < k; ++docId, docCodes += NUM_BOOKS) {
        float dist{normalise ?
                   computeNormedDist(distTable, &(*docCodes)) :
                   computeDist(distTable, &(*docCodes))};
        topk.push(std::make_pair(dist, docId));
    }
    for (/**/; docCodes != docsCodes.end(); ++docId, docCodes += NUM_BOOKS) {
        float dist{normalise ?
                   computeNormedDist(distTable, &(*docCodes)) :
                   computeDist(distTable, &(*docCodes))};
        if (dist < topk.top().first) {
            topk.pop();
            topk.push(std::make_pair(dist, docId));
        }
    }
}

void searchBruteForce(std::size_t k,
                      const std::vector<float>& docs,
                      const std::vector<float>& query,
                      std::priority_queue<std::pair<float, std::size_t>>& topk) {
    std::size_t dim{query.size()};

    for (std::size_t i = 0; i < docs.size(); i += dim) {
        float sim{0.0F};
        #pragma clang loop vectorize(assume_safety)
        for (std::size_t j = 0; j < dim; ++j) {
            sim += query[j] * docs[i+j];
        }
        float dist{1.0F - sim};
        if (topk.size() < k) {
            topk.push(std::make_pair(dist, i / dim));
        } else if (topk.top().first > dist) {
            topk.pop();
            topk.push(std::make_pair(dist, i / dim));
        }
    }
}

void runPQBenchmark(const std::string& tag,
                    std::size_t k,
                    std::size_t dim,
                    std::vector<float>& docs,
                    std::vector<float>& queries,
                    bool normalise,
                    const std::function<void(const PQStats&)>& writeStats) {

    normalize(dim, docs);
    normalize(dim, queries);
    zeroPad(dim, docs);
    zeroPad(dim, queries);

    std::chrono::duration<double> diff{0};
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;

    std::vector<float> query(dim);
    std::priority_queue<std::pair<float, std::size_t>> topk;

    std::size_t nq{queries.size() / dim};
    std::size_t nd{docs.size() / dim};
    std::cout << std::setprecision(3)
              << "query count = " << nq << ", doc count = " << nd
              << ", dimension = " << dim << std::endl;
    std::cout << std::boolalpha << "top-k = " << k
              << ", normalise = " << normalise << std::endl;

    std::vector<std::vector<std::size_t>> nnExact(nq, std::vector<std::size_t>(k));
    for (std::size_t i = 0; i < queries.size(); i += dim) {
        std::copy(queries.begin() + i, queries.begin() + i + dim, query.begin());

        start = std::chrono::high_resolution_clock::now();
        searchBruteForce(k, docs, query, topk);
        end = std::chrono::high_resolution_clock::now();
        diff += std::chrono::duration<double>(end - start);

        for (std::size_t j = 1; j <= k && !topk.empty(); ++j) {
            nnExact[i / dim][k - j] = topk.top().second;
            topk.pop();
        }
    }
    std::cout << "Brute force took " << diff.count() << "s" << std::endl;

    PQStats stats{tag, nq, nd, k};
    stats.normalise = normalise;
    stats.bfQPS = static_cast<std::size_t>(
        std::round(static_cast<double>(nq) / diff.count()));

    start = std::chrono::high_resolution_clock::now();
    auto [codeBooks, docsCodes] = buildCodeBook(dim, docs, K_MEANS_ITR);
    end = std::chrono::high_resolution_clock::now();
    diff = std::chrono::duration<double>(end - start);
    std::cout << "Building codebooks took " << diff.count() << "s" << std::endl;

    stats.pqMse = quantisationMse(dim, codeBooks, docs, docsCodes);
    std::cout << "Quantisation MSE = " << stats.pqMse << std::endl;

    stats.pqCodeBookBuildTime = diff.count();
    stats.pqCompressionRatio = computeCompressionRatio(dim);

    std::vector<std::size_t> expansion{1, 2, 4, 6, 8, 10};

    for (std::size_t m : PQStats::EXPANSIONS) {

        std::vector<std::vector<std::size_t>> nnPQ(nq, std::vector<std::size_t>(m * k));

        diff = std::chrono::duration<double>{0};
        for (std::size_t i = 0; i < queries.size(); i += dim) {
            std::copy(queries.begin() + i, queries.begin() + i + dim, query.begin());

            start = std::chrono::high_resolution_clock::now();
            searchPQ(m * k, codeBooks, docsCodes, query, normalise, topk);
            end = std::chrono::high_resolution_clock::now();
            diff += std::chrono::duration<double>(end - start);

            for (std::size_t j = 1; j <= m * k && !topk.empty(); ++j) {
                nnPQ[i / dim][m * k - j] = topk.top().second;
                topk.pop();
            }
        }

        stats.pqQPS.push_back(std::round(static_cast<double>(nq) / diff.count()));
        stats.pqRecalls.push_back(computeRecalls(nnExact, nnPQ));
        std::cout << "PQ search took " << diff.count() << "s, "
                  << "average recall @ " << k << "|" << m * k << " = "
                  << stats.pqRecalls.back()[PQStats::AVG_RECALL] << std::endl;
    }

    writeStats(stats);
}