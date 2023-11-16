#include "pq.h"

#include "bruteforce.h"
#include "metrics.h"
#include "pq_clustering.h"
#include "pq_constants.h"
#include "utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
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
              << "metric = " << toString(metric)
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

    PQStats stats{tag, toString(metric), numQueries, numDocs, dim, k};
    stats.scann = scann;
    stats.normalise = (metric == Cosine);
    stats.bfQPS = std::round(static_cast<double>(numQueries) / diff.count());

    start = std::chrono::high_resolution_clock::now();
    // Using 128 vectors per centroid is sufficient to build the code book.
    double sampleProbability{128 * NUM_BOOKS / static_cast<double>(numDocs)};
    auto [codeBooks, docsCodes] = buildCodeBook(dim, sampleProbability, docs, K_MEANS_ITR);
    end = std::chrono::high_resolution_clock::now();
    diff = std::chrono::duration<double>(end - start);
    std::cout << "Building codebooks took " << diff.count() << "s" << std::endl;

    stats.pqMse = quantisationMseLoss(dim, codeBooks, docs, docsCodes);
    std::cout << "Quantisation MSE = " << stats.pqMse << std::endl;

    stats.pqCodeBookBuildTime = diff.count();
    stats.pqCompressionRatio = computeCompressionRatio(dim);

    for (std::size_t m : PQStats::EXPANSIONS) {

        std::vector<std::vector<std::size_t>> nnPQ(
            numQueries, std::vector<std::size_t>(m * k, numDocs + 1));

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