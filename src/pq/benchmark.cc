#include "benchmark.h"

#include "constants.h"
#include "index.h"
#include "stats.h"
#include "types.h"
#include "utils.h"
#include "../common/bruteforce.h"
#include "../common/progress_bar.h"
#include "../common/types.h"
#include "../common/utils.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <functional>
#include <iostream>
#include <memory>
#include <queue>
#include <vector>

void runPQBenchmark(const std::string& tag,
                    Metric metric,
                    float distanceThreshold,
                    std::size_t docsPerCoarseCluster,
                    std::size_t numBooks,
                    std::size_t k,
                    const BigVector& docs,
                    std::vector<float>& queries,
                    const std::function<void(const PQStats&)>& writeStats) {

    static_assert(BOOK_SIZE - 1 <= std::numeric_limits<code_t>::max(),
                  "You need to increase code_t size");

    std::size_t dim{docs.dim()};

    if (queries.size() % dim != 0) {
        throw std::invalid_argument("Invalid query size");
    }
    if (metric == Cosine) {
        normalize(dim, queries);
    }

    std::size_t numQueries{queries.size() / dim};
    std::size_t numDocs{docs.size() / dim};
    std::chrono::duration<double> diff{0};

    std::cout << "Building PQ index..." << std::endl;
    PqIndex index{[&] {
        Timer timer{"Building PQ index", diff};
        return buildPqIndex(docs, metric, docsPerCoarseCluster, numBooks, distanceThreshold);
    }()};
    std::cout << "PQ index built in " << diff.count() << " s" << std::endl;

    PQStats stats{tag, toString(metric), numQueries, numDocs, dim, numBooks, k};
    stats.pqCodeBookBuildTime = diff.count();
    stats.pqVectorCompressionRatio = index.vectorCompressionRatio();
    stats.pqCompressionRatio = index.compressionRatio();

    std::cout << std::setprecision(5)
              << "query count = " << numQueries
              << ", doc count = " << numDocs
              << ", dimension = " << dim << std::endl;
    std::cout << std::boolalpha
              << "metric = " << toString(metric)
              << ", top-k = " << k
              << ", normalized = " << (metric == Cosine)
              << ", book count = " << numBooks
              << ", book size = " << BOOK_SIZE
              << ", vector compression ratio = " << stats.pqVectorCompressionRatio
              << ", compression ratio = " << stats.pqCompressionRatio << std::endl;

    std::vector<float> query(docs.dim());
    std::vector<std::vector<std::size_t>> topkExact(numQueries);

    diff = std::chrono::duration<double>{0};
    std::unique_ptr<ProgressBar> progress{
        std::make_unique<ProgressBar>("Bruteforce...", numQueries)};
    for (std::size_t i = 0, j = 0; i < queries.size(); i += dim, ++j) {
        std::copy(queries.begin() + i, queries.begin() + i + dim, query.begin());
        diff += time([&] { topkExact[j] = searchBruteForce(k, docs, query).first; });
        progress->update();
    }
    progress.reset();
    std::cout << "Brute force took " << diff.count() << " s" << std::endl;

    stats.normalize = (metric == Cosine);
    stats.bfQPS = std::round(static_cast<double>(numQueries) / diff.count());

    for (std::size_t m : EXPANSIONS) {

        std::vector<std::vector<std::size_t>> topkPq(numQueries);

        diff = std::chrono::duration<double>{0};
        progress = std::make_unique<ProgressBar>("PQ search...", numQueries);
        for (std::size_t i = 0, j = 0; i < queries.size(); i += dim, ++j) {
            std::copy(queries.begin() + i, queries.begin() + i + dim, query.begin());
            diff += time([&] { topkPq[j] = index.search(query, m * k).first; });
            progress->update();
        }
        progress.reset();
        std::cout << "PQ search took " << diff.count() << " s" << std::endl;

        stats.pqQPS.push_back(std::round(static_cast<double>(numQueries) / diff.count()));
        stats.pqRecalls.push_back(computeRecalls(topkExact, topkPq));
        std::cout << "Average recall@" << k << "|" << m * k << " = "
                  << stats.pqRecalls.back()[AVG_RECALL] << std::endl;
    }
    for (std::size_t i = 0; i < EXPANSIONS.size(); ++i) {
        std::size_t m{EXPANSIONS[i]};
        std::cout << "Recall@" << k << "|" << m * k << " = "
                  << stats.pqRecalls[i][AVG_RECALL] << std::endl;
    }

    writeStats(stats);
}