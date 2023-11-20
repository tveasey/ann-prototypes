#include "benchmark.h"

#include "constants.h"
#include "index.h"
#include "stats.h"
#include "types.h"
#include "utils.h"
#include "../common/bruteforce.h"
#include "../common/types.h"
#include "../common/utils.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <functional>
#include <iostream>
#include <queue>
#include <vector>

namespace {

// RAII timer
class Timer {
public:
    Timer(const std::string& operation,
          std::chrono::duration<double>& duration) :
        operation_{operation},
        duration_{duration},
        start_{std::chrono::high_resolution_clock::now()} {
    }

    Timer(const Timer&) = delete;
    Timer& operator=(const Timer&) = delete;

    ~Timer() {
        std::chrono::high_resolution_clock::time_point end;
        end = std::chrono::high_resolution_clock::now();
        duration_ = end - start_;
        if (!operation_.empty()) {
            std::cout << operation_ << " took " << duration_.count() << "s" << std::endl;
        }
    }

private:
    const std::string& operation_;
    std::chrono::duration<double>& duration_;
    std::chrono::high_resolution_clock::time_point start_;
};

std::chrono::duration<double> time(std::function<void()> f,
                                   const std::string& operation = "") {
    std::chrono::duration<double> diff{0};
    {
        Timer timer{operation, diff};
        f();
    }
    return diff;
}

} // unnamed::

void runPQBenchmark(const std::string& tag,
                    Metric metric,
                    float distanceThreshold,
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

    bool normalized{metric == Cosine};
    std::size_t numQueries{queries.size() / dim};
    std::size_t numDocs{docs.size() / dim};
    std::chrono::duration<double> diff{0};

    std::cout << "Building PQ index..." << std::endl;
    PqIndex index{[&] {
        Timer timer{"Building PQ index", diff};
        return buildPqIndex(docs, normalized, distanceThreshold);
    }()};

    PQStats stats{tag, toString(metric), numQueries, numDocs, dim, k};
    stats.pqCodeBookBuildTime = diff.count();
    stats.pqCompressionRatio = index.compressionRatio();

    std::cout << std::setprecision(5)
              << "query count = " << numQueries
              << ", doc count = " << numDocs
              << ", dimension = " << dim << std::endl;
    std::cout << std::boolalpha
              << "metric = " << toString(metric)
              << ", top-k = " << k
              << ", normalize = " << (metric == Cosine)
              << ", book count = " << NUM_BOOKS
              << ", book size = " << BOOK_SIZE << std::endl;

    std::vector<float> query(docs.dim());
    std::vector<std::vector<std::size_t>> topkExact(numQueries);

    diff = std::chrono::duration<double>{0};
    for (std::size_t i = 0; i < queries.size(); i += dim) {
        std::copy(queries.begin() + i, queries.begin() + i + dim, query.begin());
        diff += time([&] { topkExact[i] = searchBruteForce(k, docs, query).first; });
    }
    std::cout << "Brute force took " << diff.count() << "s" << std::endl;

    stats.normalize = (metric == Cosine);
    stats.bfQPS = std::round(static_cast<double>(numQueries) / diff.count());

    for (std::size_t m : EXPANSIONS) {

        std::vector<std::vector<std::size_t>> topkPq(numQueries);

        diff = std::chrono::duration<double>{0};
        for (std::size_t i = 0; i < queries.size(); i += dim) {
            std::copy(queries.begin() + i, queries.begin() + i + dim, query.begin());
            diff += time([&] {topkPq[i] = index.search(query, m * k).first; });
        }
        std::cout << "PQ search took " << diff.count() << "s" << std::endl;
    
        stats.pqQPS.push_back(std::round(static_cast<double>(numQueries) / diff.count()));
        stats.pqRecalls.push_back(computeRecalls(topkExact, topkPq));
        std::cout << "Average recall@" << k << "|" << m * k << " = "
                  << stats.pqRecalls.back()[AVG_RECALL] << std::endl;
    }

    writeStats(stats);
}