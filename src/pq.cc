#include "pq.h"

#include <algorithm>
#include <chrono>
#include <cmath>
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

void zeroPad(std::size_t dim, std::vector<float>& vectors) {
    // Zero pad the vectors so their dimension is a multiple of NUM_BOOKS.
    if (dim % NUM_BOOKS != 0) {
        std::size_t numVectors{vectors.size() / dim};
        std::size_t paddedDim{NUM_BOOKS * ((dim + 7) / NUM_BOOKS)};
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
            const float* doc = &docs[i * dim];
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

    std::vector<float> bookCounts(numBooks * bookSize, 0.0F);
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
            float alpha{bookCount / (bookCount + 1.0F)};
            float beta{1.0F - alpha};
            for (std::size_t j = 0; j < bookDim; ++j) {
                newCentre[j] = alpha * newCentre[j] + beta * doc[b * bookDim + j];
            }
            bookCount += 1.0F;

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
    float eta{static_cast<float>(dim- 1) * t / 1 - t};

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
                    float ci{centres[(b * bookSize + i) * bookDim + j]};
                    float xi{doc[b * bookDim + j]};
                    float r{xi - ci};
                    // TODO we need to have per book vector norms.
                    float rp{xi * xi * r};
                    distParallel += rp;
                    distPerpendicular += r - rp;
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

float computeDispersion(std::size_t dim,
                        const std::vector<float>& centres,
                        const std::vector<float>& docs,
                        const std::vector<code_t>& docsCentres) {
    float dispersion{0.0F};
    for (std::size_t i = 0; i < docsCentres.size(); ++i) {
        float dist{0.0F};
        for (std::size_t j = 0; j < dim; ++j) {
            float d{centres[(OFFSET + docsCentres[i]) * dim + j] - docs[i * dim + j]};
            dist += d * d;
        }
        dispersion += std::sqrt(dist);
    }
    return dispersion;
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
    std::size_t bookDim{dim / NUM_BOOKS};
    std::minstd_rand rng;
    std::vector<float> codeBooks{initForgy(BOOK_SIZE, bookDim, dim, docs, rng)};
    std::vector<code_t> docsCodes(docs.size() / bookDim);

    // Initialize with k-means.
    for (std::size_t i = 0; i < iterations / 2; ++i) {
        stepLloyd(NUM_BOOKS, BOOK_SIZE, dim, docs, codeBooks, docsCodes);
    }

    // Fine-tune with anisotropic loss.
    for (std::size_t i = 0; i < iterations / 2 + iterations % 2; ++i) {
        stepScann(t, NUM_BOOKS, BOOK_SIZE, dim, docs, codeBooks, docsCodes);
    }

    return std::make_pair(std::move(codeBooks), std::move(docsCodes));
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

float computeDist(const std::vector<float>& distTable,
                  const code_t* docCode) {
    float dist{0.0F};
    #pragma clang loop unroll_count(NUM_BOOKS)
    for (std::size_t b = 0; b < NUM_BOOKS; ++b) {
        dist += distTable[BOOK_SIZE * b + OFFSET + docCode[b]];
    }
    return 1.0 - dist;
}

void searchPQ(std::size_t k,
              const std::vector<float>& codeBooks,
              const std::vector<code_t>& docsCodes,
              const std::vector<float>& query,
              std::priority_queue<std::pair<float, std::size_t>>& topk) {

    std::size_t dim{query.size()};
    std::size_t bookDim{dim / NUM_BOOKS};

    auto distTable = buildDistTable(codeBooks, query);

    std::size_t docId{0};
    auto docCodes = docsCodes.begin();
    for (/**/; docId < k; ++docId, docCodes += NUM_BOOKS) {
        topk.push(std::make_pair(computeDist(distTable, &(*docCodes)), docId));
    }
    for (/**/; docCodes != docsCodes.end(); ++docId, docCodes += NUM_BOOKS) {
        float dist{computeDist(distTable, &(*docCodes))};
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

void runPQBenchmark(std::size_t dim,
                    std::vector<float>& docs,
                    std::vector<float>& queries) {

    normalize(dim, docs);
    normalize(dim, queries);
    zeroPad(dim, docs);
    zeroPad(dim, queries);

    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;
    std::chrono::duration<double> diff{0};
    std::vector<float> query(dim);
    std::priority_queue<std::pair<float, std::size_t>> topk;

    std::vector<std::vector<std::pair<float, std::size_t>>> nearestExact(queries.size() / dim);
    for (std::size_t i = 0; i < queries.size(); i += dim) {
        std::copy(&queries[i], &queries[i + dim], &query[0]);

        start = std::chrono::high_resolution_clock::now();
        searchBruteForce(10, docs, query, topk);
        end = std::chrono::high_resolution_clock::now();
        diff += std::chrono::duration<double>(end - start);

        while (!topk.empty()) {
            nearestExact[i / dim].push_back(topk.top());
            topk.pop();
        }
    }

    std::cout << "Brute force search duration " << diff.count() << "s" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    auto [codeBooks, docsCodes] = buildCodeBook(dim, docs);
    end = std::chrono::high_resolution_clock::now();
    diff = std::chrono::duration<double>(end - start);

    std::cout << "Building codebooks duration " << diff.count() << "s" << std::endl;

    diff = std::chrono::duration<double>{0};
    std::vector<std::vector<std::pair<float, std::size_t>>> nearestPQ(queries.size() / dim);
    for (std::size_t i = 0; i < queries.size(); i += dim) {
        std::copy(&queries[i], &queries[i + dim], &query[0]);

        start = std::chrono::high_resolution_clock::now();
        searchPQ(10, codeBooks, docsCodes, query, topk);
        end = std::chrono::high_resolution_clock::now();
        diff += std::chrono::duration<double>(end - start);

        while (!topk.empty()) {
            nearestPQ[i / dim].push_back(topk.top());
            topk.pop();
        }
    }

    std::cout << "PQ search duration " << diff.count() << "s" << std::endl;
}