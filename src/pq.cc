#include "pq.h"
#include "metrics.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <ios>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// Notes
// -----
// * Hardcode dot product: other metrics should perform similarly.
// * Data-oriented design to maximize memory access efficiency.
// * Hardcode all loop limits where possible since this helps the
//   compiler vectorise code.
// * Brute force k-means performs well for moderate k.

namespace {
constexpr int NUM_BOOKS{24};
constexpr int BOOK_SIZE{256};
constexpr std::size_t K_MEANS_ITR{8};
std::array<std::string, 2> METRICS{"dot", "cosine"}; 
}

int numBooks() {
    return NUM_BOOKS;
}

int bookSize() {
    return BOOK_SIZE;
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

void normalise(std::size_t dim, std::vector<float>& vectors) {
    // Ensure vectors are unit (Euclidean) norm.
    for (std::size_t i = 0; i < vectors.size(); i += dim) {
        float norm{0.0F};
        #pragma clang loop unroll_count(8) vectorize(assume_safety)
        for (std::size_t j = 0; j < dim; ++j) {
            norm += vectors[i + j] * vectors[i + j];
        }
        norm = std::sqrtf(norm);
        #pragma clang loop unroll_count(8) vectorize(assume_safety)
        for (std::size_t j = 0; j < dim; ++j) {
            vectors[i + j] /= norm;
        }
    }
}

std::vector<float> norms2(std::size_t dim, std::vector<float>& vectors) {
    std::vector<float> norms2(vectors.size() / dim);
    for (std::size_t i = 0, j = 0; i < vectors.size(); i += dim, ++j) {
        float norm2{0.0F};
        #pragma clang loop unroll_count(8) vectorize(assume_safety)
        for (std::size_t j = 0; j < dim; ++j) {
            norm2 += vectors[i + j] * vectors[i + j];
        }
        norms2[j] = norm2;
    }
    return norms2;
}

double quantisationMseLoss(std::size_t dim,
                           const std::vector<float>& codeBooks,
                           const std::vector<float>& docs,
                           const std::vector<code_t>& docsCodes) {

    std::size_t bookDim{dim / NUM_BOOKS};

    double totalMse{0.0};
    std::size_t count{0};

    auto code = docsCodes.begin();
    for (auto doc = docs.begin(); doc != docs.end(); ++count) {
        float mse{0.0F};
        for (std::size_t b = 0; b < NUM_BOOKS; ++b, ++code, doc += bookDim) {
            #pragma clang loop unroll_count(4) vectorize(assume_safety)
            for (std::size_t i = 0; i < bookDim; ++i) {
                float di{doc[i] - codeBooks[(BOOK_SIZE * b + *code) * bookDim + i]};
                mse += di * di;
            }
        }
        totalMse += mse;
    }
    return totalMse / static_cast<double>(count);
}

double quantisationScannLoss(float t,
                             std::size_t dim,
                             const std::vector<float>& codeBooks,
                             const std::vector<float>& docs,
                             const std::vector<float>& docsNorms2,
                             const std::vector<code_t>& docsCodes) {

    std::size_t bookDim{dim / NUM_BOOKS};

    double totalLoss{0.0};
    std::size_t count{0};

    auto code = docsCodes.begin();
    auto docNorm2 = docsNorms2.begin();
    for (auto doc = docs.begin(); doc != docs.end(); ++count, ++docNorm2) {

        float n2{*docNorm2};
        float t2{t * t / n2};
        float eta{static_cast<float>(dim - 1) * t2 / (1 - t2)};

        float loss{0.0F};
        for (std::size_t b = 0; b < NUM_BOOKS; ++b, ++code, doc += bookDim) {
            #pragma clang loop vectorize(assume_safety)
            for (std::size_t i = 0; i < bookDim; ++i) {
                float ci{codeBooks[(BOOK_SIZE * b + *code) * bookDim + i]};
                float xi{doc[i]};
                float ri{xi - ci};
                float riPar{xi * xi * ri / n2};
                float riPerp{ri - riPar};
                loss += eta * riPar * riPar + riPerp * riPerp;
            }
        }
        totalLoss += loss;
    }
    return totalLoss / static_cast<double>(count);
}

std::set<std::size_t> initForgy(std::size_t numDocs,
                                std::minstd_rand& rng) {
    std::set<std::size_t> selection;
    std::uniform_int_distribution<> u0n{0, static_cast<int>(numDocs) - 1};
    while (selection.size() < BOOK_SIZE) {
        std::size_t cand{static_cast<std::size_t>(u0n(rng))};
        selection.insert(cand);
    }
    return selection;
}

std::vector<float> initForgy(std::size_t dim,
                             const std::vector<float>& docs,
                             std::minstd_rand& rng) {
    std::size_t bookDim{dim / NUM_BOOKS};
    std::size_t numDocs{docs.size() / dim};
    auto selection = initForgy(numDocs, rng);
    std::vector<float> centres(BOOK_SIZE * dim);
    auto centre = centres.begin();
    for (std::size_t b = 0; b < NUM_BOOKS; ++b) {
        for (auto i = selection.begin(); i != selection.end(); ++i, centre += bookDim) {
            auto doc = docs.begin() + *i * dim;
            std::copy(doc + b * bookDim, doc + (b + 1) * bookDim, centre);
        }
    }
    return centres;
}

void stepLloyd(std::size_t dim,
               const std::vector<float>& docs,
               std::vector<float>& centres,
               std::vector<code_t>& docsCodes) {

    // Since we sum up non-negative losses from each book we can compute
    // the optimal codebooks independently.

    std::size_t bookDim{dim / NUM_BOOKS};
    std::vector<double> newCentres(BOOK_SIZE * dim, 0.0);
    std::vector<std::size_t> centreCounts(BOOK_SIZE * NUM_BOOKS, 0);

    std::size_t pos{0};
    for (auto doc = docs.begin(); doc != docs.end(); /**/) {
        for (std::size_t b = 0; b < NUM_BOOKS; ++b, ++pos, doc += bookDim) {
            // Find the nearest centroid.
            int iMinLoss{0};
            float minLoss{std::numeric_limits<float>::max()};
            for (int i = BOOK_SIZE * b; i < BOOK_SIZE * (b + 1); ++i) {
                float loss{0.0F};
                for (std::size_t j = 0; j < bookDim; ++j) {
                    float dij{centres[i * bookDim + j] - doc[j]};
                    loss += dij * dij;
                }
                if (loss < minLoss) {
                    iMinLoss = i;
                    minLoss = loss;
                }
            }

            // Update the centroid.
            auto* newCentre = &newCentres[iMinLoss * bookDim];
            for (std::size_t j = 0; j < bookDim; ++j) {
                newCentre[j] += doc[j];
            }
            ++centreCounts[iMinLoss];

            // Encode the document.
            docsCodes[pos] = static_cast<code_t>(iMinLoss - BOOK_SIZE * b);
        }
    }

    for (std::size_t i = 0; i < centreCounts.size(); ++i) {
        if (centreCounts[i] > 0) {
            for (std::size_t j = 0; j < bookDim; ++j) {
                centres[i * bookDim + j] = static_cast<float>(
                    newCentres[i * bookDim + j] /
                    static_cast<double>(centreCounts[i]));
            }
        }
    }
}

void stepScann(float t,
               std::size_t dim,
               const std::vector<float>& docs,
               const std::vector<float>& docsNorms2,
               std::vector<float>& centres,
               std::vector<code_t>& docsCodes) {

    // Since we sum up non-negative losses from each book we can compute
    // the optimal codebooks independently.

    std::size_t bookDim{dim / NUM_BOOKS};

    // Retain incorrect k-means centroid calculation for now.
    std::vector<double> newCentres(BOOK_SIZE * dim, 0.0);
    std::vector<std::size_t> centreCounts(BOOK_SIZE * NUM_BOOKS, 0);
    std::vector<float> l0(NUM_BOOKS);

    std::size_t pos{0};
    auto docNorm2 = docsNorms2.begin();
    for (auto doc = docs.begin(); doc != docs.end(); ++docNorm2) {

        // See theorem 3.4 https://arxiv.org/pdf/1908.10396.pdf.
        float n2{*docNorm2};
        float t2{t * t / n2};
        float eta{static_cast<float>(dim - 1) * t2 / (1 - t2)};
        float scale{(eta - 1.0F) / n2};

        // Precompute the residual projected onto the document vector.
        float l{0.0F};
        for (std::size_t b = 0; b < NUM_BOOKS; ++b) {
            int i0 = BOOK_SIZE * b + docsCodes[pos + b];
            l0[b] = 0.0F;
            for (std::size_t j = 0; j < bookDim; ++j) {
                float xj{doc[b * bookDim + j]};
                float cij{centres[i0 * bookDim + j]};
                l0[b] += xj * (xj - cij);
            }
            l += l0[b];
        }
        for (std::size_t b = 0; b < NUM_BOOKS; ++b) {
            l0[b] = l - l0[b];
        }

        for (std::size_t b = 0; b < NUM_BOOKS; ++b, ++pos, doc += bookDim) {
            // Find the centroid assignment that minimizes anisotropic loss.
            int iMinLoss{0};
            float minLoss{std::numeric_limits<float>::max()};
            for (int i = BOOK_SIZE * b; i < BOOK_SIZE * (b + 1); ++i) {
                float l1{l0[b]};
                float l2{0.0F};
                for (std::size_t j = 0; j < bookDim; ++j) {
                    float cij{centres[i  * bookDim + j]};
                    float xj{doc[j]};
                    float rij{xj - cij};
                    l1 += xj * rij;
                    l2 += rij * rij;
                }
                float loss{l2 + scale * l1 * l1};
                if (loss < minLoss) {
                    iMinLoss = i;
                    minLoss = loss;
                }
            }

            auto* newCentre = &newCentres[iMinLoss * bookDim];
            for (std::size_t j = 0; j < bookDim; ++j) {
                newCentre[j] += doc[j];
            }
            ++centreCounts[iMinLoss];

            // Encode the document.
            docsCodes[pos] = static_cast<code_t>(iMinLoss - BOOK_SIZE * b);
        }
    }

    for (std::size_t i = 0; i < centreCounts.size(); ++i) {
        if (centreCounts[i] > 0) {
            for (std::size_t j = 0; j < bookDim; ++j) {
                centres[i * bookDim + j] = static_cast<float>(
                    newCentres[i * bookDim + j] /
                    static_cast<double>(centreCounts[i]));
            }
        }
    }
}

std::vector<std::size_t> uniformSamples(double sampleProbability,
                                        std::size_t n,
                                        std::minstd_rand& rng) {
    std::vector<std::size_t> samples;
    if (sampleProbability < 1.0) {
        samples.reserve(static_cast<std::size_t>(1.1 * n * sampleProbability));
        std::geometric_distribution<> geom{sampleProbability};
        for (std::size_t i = geom(rng); i < n; i += 1 + geom(rng)) {
            samples.push_back(i);
        }
    } else {
        samples.resize(n);
        std::iota(samples.begin(), samples.end(), 0);
    }
    return samples;
}

std::vector<float> sampleDocs(double sampleProbability,
                              std::size_t dim,
                              const std::vector<float>& docs,
                              std::minstd_rand& rng) {
    auto samples = uniformSamples(sampleProbability, docs.size() / dim, rng);
    std::vector<float> sampledDocs(dim * samples.size());
    for (std::size_t i = 0; i < samples.size(); ++i) {
        std::size_t sample{samples[i]};
        std::copy_n(docs.begin() + dim * sample, dim, sampledDocs.begin() + dim * i);
    }
    return sampledDocs;
}

std::vector<float> initCodeBooks(std::size_t dim,
                                 const std::vector<float>& docs,
                                 const loss_t& loss) {

    // Random restarts with aggressive downsample.

    std::vector<float> minLossCodeBooks;
    double minLoss{std::numeric_limits<double>::max()};

    std::minstd_rand rng;

    // Using 20 vectors per centroid is enough to get reasonable estimates.
    std::size_t numDocs{docs.size() / dim};
    double sampleProbability{
        std::min(20 * BOOK_SIZE / static_cast<double>(numDocs), 1.0)};
    auto sampledDocs = sampleDocs(sampleProbability, dim, docs, rng);

    std::vector<code_t> docsCodes(NUM_BOOKS * numDocs);
    for (std::size_t trial = 0; trial < 5; ++trial) {
        std::vector<float> codeBooks{initForgy(dim, sampledDocs, rng)};
        // Four iterations is largely converged.
        for (std::size_t i = 0; i < 4; ++i) {
            stepLloyd(dim, sampledDocs, codeBooks, docsCodes);
        }
        double trialLoss{loss(dim, codeBooks, sampledDocs, docsCodes)};
        if (trialLoss < minLoss) {
            minLossCodeBooks = std::move(codeBooks);
            minLoss = trialLoss;
        }
    }    

    return minLossCodeBooks;
}

std::pair<std::vector<float>, std::vector<code_t>>
buildCodeBook(std::size_t dim,
              double sampleProbability,
              const std::vector<float>& docs,
              std::size_t iterations,
              const loss_t& loss) {

    std::size_t bookDim{dim / NUM_BOOKS};
    std::vector<float> codeBooks{initCodeBooks(dim, docs, loss)};
    std::vector<code_t> docsCodes(docs.size() / bookDim);

    if (sampleProbability < 1.0) {
        std::minstd_rand rng;
        auto sampledDocs = sampleDocs(sampleProbability, dim, docs, rng);
        for (std::size_t i = 0; i < iterations - 1; ++i) {
            stepLloyd(dim, sampledDocs, codeBooks, docsCodes);
        }
    } else {
        for (std::size_t i = 0; i < iterations - 1; ++i) {
            stepLloyd(dim, docs, codeBooks, docsCodes);
        }
    }

    // One full step to compute the correct document codes.
    stepLloyd(dim, docs, codeBooks, docsCodes);

    return std::make_pair(std::move(codeBooks), std::move(docsCodes));
}

std::pair<std::vector<float>, std::vector<code_t>>
buildCodeBookScann(float t,
                   std::size_t dim,
                   double sampleProbability,
                   const std::vector<float>& docs,
                   const std::vector<float>& docsNorms2,
                   std::size_t iterations) {

    // Initialize with k-means.
    auto [codeBooks, docsCodes] =
        buildCodeBook(dim, sampleProbability, docs, iterations / 2,
                      [&](std::size_t dim,
                          const std::vector<float>& codeBooks,
                          const std::vector<float>& docs,
                          const std::vector<code_t>& docsCodes) {
                          return quantisationScannLoss(t, dim, codeBooks, docs, docsNorms2, docsCodes);
                      });

    // Fine-tune with anisotropic loss.
    iterations = iterations / 2 + iterations % 2;
    if (sampleProbability < 1.0) {
        std::minstd_rand rng;
        auto samples = uniformSamples(sampleProbability, docs.size() / dim, rng);
        std::vector<float> sampledDocs(dim * samples.size());
        std::vector<float> sampledDocsNorms(samples.size());
        for (std::size_t i = 0; i < samples.size(); ++i) {
            std::size_t sample{samples[i]};
            std::copy_n(docs.begin() + dim * sample, dim, sampledDocs.begin() + dim * i);
            sampledDocsNorms[i] = docsNorms2[sample];
        }
        for (std::size_t i = 0; i < iterations - 1; ++i) {
            stepScann(t, dim, sampledDocs, sampledDocsNorms, codeBooks, docsCodes);
        }
    } else {
        for (std::size_t i = 0; i < iterations - 1; ++i) {
            stepScann(t, dim, docs, docsNorms2, codeBooks, docsCodes);
        }
    }

    // One full step to compute the correct document codes.
    stepScann(t, dim, docs, docsNorms2, codeBooks, docsCodes);

    return std::make_pair(std::move(codeBooks), std::move(docsCodes));
}

std::vector<float> buildDistTable(const std::vector<float>& codeBooks,
                                  const std::vector<float>& query) {
    std::size_t bookDim{query.size() / NUM_BOOKS};
    std::vector<float> distTable(BOOK_SIZE * NUM_BOOKS);
    for (std::size_t b = 0; b < NUM_BOOKS; ++b) {
        for (std::size_t i = 0; i < BOOK_SIZE; ++i) {
            float sim{0.0F};
            #pragma clang loop unroll_count(8) vectorize(assume_safety)
            for (std::size_t j = 0; j < bookDim; ++j) {
                float cij{codeBooks[(BOOK_SIZE * b + i) * bookDim + j]};
                sim += query[b * bookDim + j] * cij;
            }
            distTable[BOOK_SIZE * b + i] = sim;
        }
    }
    return distTable;
}

std::vector<float> buildDistNorm2Table(const std::vector<float>& codeBooks,
                                       const std::vector<float>& query) {
    std::size_t bookDim{query.size() / NUM_BOOKS};
    std::vector<float> distTable(2 * BOOK_SIZE * NUM_BOOKS);
    for (std::size_t b = 0; b < NUM_BOOKS; ++b) {
        for (std::size_t i = 0; i < BOOK_SIZE; ++i) {
            float sim{0.0F};
            float norm2{0.0F};
            #pragma clang loop unroll_count(4) vectorize(assume_safety)
            for (std::size_t j = 0; j < bookDim; ++j) {
                float cij{codeBooks[(BOOK_SIZE * b + i) * bookDim + j]};
                sim += query[b * bookDim + j] * cij;
                // TODO the norm calculation can be cached for all queries.
                norm2 += cij * cij;
            }
            auto t = distTable.begin() + 2 * (BOOK_SIZE * b + i);
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
        sim += distTable[BOOK_SIZE * b + docCode[b]];
    }
    return 1.0F - sim;
}

float computeNormedDist(const std::vector<float>& distTable,
                        const code_t* docCode) {
    float sim{0.0F};
    float qnorm2{0.0F};
    #pragma clang loop unroll_count(4)
    for (std::size_t b = 0; b < NUM_BOOKS; ++b) {
        const auto* t = &distTable[2 * (BOOK_SIZE * b + docCode[b])];
        sim += t[0];
        qnorm2 += t[1];
    }
    // Project the quantised representation onto the unit sphere.
    return 1.0F - sim / std::sqrtf(qnorm2);
}

std::vector<float> encoded(std::size_t dim,
                           const std::vector<float>& codeBooks,
                           const code_t* docCode) {
    std::size_t bookDim{dim / NUM_BOOKS};
    std::vector<float> result(dim);
    for (std::size_t b = 0; b < NUM_BOOKS; ++b) {
        const auto* centroid = &codeBooks[(BOOK_SIZE * b + docCode[b]) * bookDim];
        std::copy(centroid, centroid + bookDim, &result[b * bookDim]);
    }
    return result;
}

void searchPQ(std::size_t k,
              const std::vector<float>& codeBooks,
              const std::vector<code_t>& docsCodes,
              const std::vector<float>& docsNorms,
              const std::vector<float>& query,
              bool normalise,
              std::priority_queue<std::pair<float, std::size_t>>& topk) {

    std::size_t dim{query.size()};
    std::size_t bookDim{dim / NUM_BOOKS};

    auto distTable = normalise ?
        buildDistNorm2Table(codeBooks, query) :
        buildDistTable(codeBooks, query);

    auto docCodes = docsCodes.begin();
    std::size_t id{0};
    for (/**/; id < k; docCodes += NUM_BOOKS, ++id) {
        float dist{normalise ?
                   computeNormedDist(distTable, &(*docCodes)) :
                   computeDist(distTable, &(*docCodes))};
        topk.push(std::make_pair(dist, id));
    }
    for (/**/; docCodes != docsCodes.end(); docCodes += NUM_BOOKS, ++id) {
        float dist{normalise ?
                   computeNormedDist(distTable, &(*docCodes)) :
                   computeDist(distTable, &(*docCodes))};
        if (dist < topk.top().first) {
            topk.pop();
            topk.push(std::make_pair(dist, id));
        }
    }
}

void searchBruteForce(std::size_t k,
                      const std::vector<float>& docs,
                      const std::vector<float>& query,
                      std::priority_queue<std::pair<float, std::size_t>>& topk) {
    std::size_t dim{query.size()};
    for (std::size_t i = 0, id = 0; i < docs.size(); i += dim, ++id) {
        float sim{0.0F};
        #pragma clang loop unroll_count(8) vectorize(assume_safety)
        for (std::size_t j = 0; j < dim; ++j) {
            sim += query[j] * docs[i + j];
        }
        float dist{1.0F - sim};
        if (topk.size() < k) {
            topk.push(std::make_pair(dist, id));
        } else if (topk.top().first > dist) {
            topk.pop();
            topk.push(std::make_pair(dist, id));
        }
    }
}

void runPQBenchmark(const std::string& tag,
                    bool scann,
                    Metric metric,
                    std::size_t k,
                    std::size_t dim,
                    std::vector<float>& docs,
                    std::vector<float>& queries,
                    const std::function<void(const PQStats&)>& writeStats) {

    static_assert(BOOK_SIZE - 1 <= std::numeric_limits<code_t>::max(),
                  "You need to increase code_t size");

    if (metric == Cosine) {
        normalise(dim, docs);
        normalise(dim, queries);
    }

    auto docsNorms2 = norms2(dim, docs);

    zeroPad(dim, docs);
    zeroPad(dim, queries);

    std::chrono::duration<double> diff{0};
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;

    std::vector<float> query(dim);
    std::priority_queue<std::pair<float, std::size_t>> topk;

    std::size_t numQueries{queries.size() / dim};
    std::size_t numDocs{docs.size() / dim};
    std::cout << std::setprecision(5)
              << "query count = " << numQueries
              << ", doc count = " << numDocs
              << ", dimension = " << dim << std::endl;
    std::cout << std::boolalpha
              << "metric = " << METRICS[metric]
              << ", top-k = " << k
              << ", scann = " << scann
              << ", normalise = " << (metric == Cosine)
              << ", book count = " << NUM_BOOKS
              << ", book size = " << BOOK_SIZE << std::endl;

    std::vector<std::vector<std::size_t>> nnExact(numQueries, std::vector<std::size_t>(k));
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

    PQStats stats{tag, METRICS[metric], numQueries, numDocs, dim, k};
    stats.scann = scann;
    stats.normalise = (metric == Cosine);
    stats.bfQPS = std::round(static_cast<double>(numQueries) / diff.count());

    start = std::chrono::high_resolution_clock::now();
    // Using 200 vectors per centroid is sufficient to build the code book.
    double sampleProbability{200 * NUM_BOOKS / static_cast<double>(numDocs)};
    auto [codeBooks, docsCodes] = scann ?
        buildCodeBookScann(0.2, dim, sampleProbability, docs, docsNorms2, K_MEANS_ITR) :
        buildCodeBook(dim, sampleProbability, docs, K_MEANS_ITR);
    end = std::chrono::high_resolution_clock::now();
    diff = std::chrono::duration<double>(end - start);
    std::cout << "Building codebooks took " << diff.count() << "s" << std::endl;

    stats.pqMse = quantisationMseLoss(dim, codeBooks, docs, docsCodes);
    std::cout << "Quantisation MSE = " << stats.pqMse << std::endl;

    stats.pqCodeBookBuildTime = diff.count();
    stats.pqCompressionRatio = computeCompressionRatio(dim);

    for (std::size_t m : PQStats::EXPANSIONS) {

        std::vector<std::vector<std::size_t>> nnPQ(numQueries, std::vector<std::size_t>(m * k));

        diff = std::chrono::duration<double>{0};
        for (std::size_t i = 0; i < queries.size(); i += dim) {
            std::copy(queries.begin() + i, queries.begin() + i + dim, query.begin());

            start = std::chrono::high_resolution_clock::now();
            searchPQ(m * k, codeBooks, docsCodes, docsNorms2, query, metric == Cosine, topk);
            end = std::chrono::high_resolution_clock::now();
            diff += std::chrono::duration<double>(end - start);

            for (std::size_t j = 1; j <= m * k && !topk.empty(); ++j) {
                nnPQ[i / dim][m * k - j] = topk.top().second;
                topk.pop();
            }
        }

        stats.pqQPS.push_back(std::round(static_cast<double>(numQueries) / diff.count()));
        stats.pqRecalls.push_back(computeRecalls(nnExact, nnPQ));
        std::cout << "PQ search took " << diff.count() << "s, "
                  << "average recall@" << k << "|" << m * k << " = "
                  << stats.pqRecalls.back()[PQStats::AVG_RECALL] << std::endl;
    }

    writeStats(stats);
}