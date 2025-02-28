#include "benchmark.h"
#include "ivf.h"

#include "../common/bruteforce.h"
#include "../common/evaluation.h"
#include "../common/progress_bar.h"
#include "../common/types.h"
#include "../common/utils.h"

#include <iostream>
#include <vector>

namespace {

void runQueries(const BigVector& docs,
                const std::vector<float>& queries,
                const SoarIVFIndex& index,
                std::size_t k,
                std::size_t numProbes) {

    std::size_t numQueries{queries.size() / docs.dim()};
    std::size_t dim{docs.dim()};
    std::cout << "query count = " << numQueries
              << ", doc count = " << docs.numVectors()
              << ", dimension = " << dim << std::endl;

    std::vector<std::vector<std::size_t>> topkExact(numQueries);

    auto diff = std::chrono::duration<double>{0};
    std::unique_ptr<ProgressBar> progress{
        std::make_unique<ProgressBar>("Bruteforce...", numQueries)};
    std::vector<float> query(dim);
    for (std::size_t i = 0, j = 0; i < queries.size(); i += dim, ++j) {
        std::copy(queries.begin() + i, queries.begin() + i + dim, query.begin());
        diff += time([&] { topkExact[j] = searchBruteForce(k, docs, query).first; });
        progress->update();
    }
    progress.reset();
    std::cout << "Brute force took " << diff.count() << " s" << std::endl;

    std::vector<std::vector<std::size_t>> topkSoarIVF(numQueries);
    for (std::size_t m : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
        std::cout << "Running IVF search with " << m * numProbes << " probes" << std::endl;
        diff = std::chrono::duration<double>{0};
        progress = std::make_unique<ProgressBar>("IVF search...", numQueries);
        std::size_t averageNumberComparisons{0};
        for (std::size_t i = 0, j = 0; i < queries.size(); i += dim, ++j) {
            std::size_t numberOfComparisons{0};
            diff += time([&] {
                std::tie(topkSoarIVF[j], numberOfComparisons) =
                    index.search(queries.data() + i, k, m * numProbes);
            });
            averageNumberComparisons += numberOfComparisons;
            progress->update();
        }
        averageNumberComparisons = (averageNumberComparisons + numQueries / 2) / numQueries;
        progress.reset();
        std::cout << "IVF search took " << diff.count() << " s" << std::endl;
        std::cout << "QPS = " << numQueries / diff.count() << std::endl;
        std::cout << "Average recall@" << k << " = "
                  << computeRecalls(topkExact, topkSoarIVF)[AVG_RECALL] << std::endl;
        std::cout << "Average number of comparisons = " << averageNumberComparisons << std::endl;
    }
}
    
} // unnamed::
    
void runSoarIVFBenchmark(Metric metric,
                         const BigVector& docs,
                         std::vector<float>& queries,
                         std::size_t k,
                         std::size_t docsPerCluster,
                         float lambda,
                         std::size_t numProbes) {

    std::cout << "Running Soar IVF benchmark..." << std::endl;
    std::cout << "metric = " << toString(metric)
              << ", top-k = " << k
              << ", normalized = " << (metric == Cosine) << std::endl;

    std::size_t numClusters{
        std::max(1UL, (docs.numVectors() + docsPerCluster - 1) / docsPerCluster)};
    std::cout << "numVectors = " << docs.numVectors() << std::endl;
    std::cout << "docsPerCluster = " << docsPerCluster << std::endl;
    std::cout << "numClusters = " << numClusters << std::endl;

    SoarIVFIndex index{metric, lambda, docs.dim(), numClusters};

    std::cout << "Building IVF index..." << std::endl;
    auto duration = time([&] { index.build(docs); });
    std::cout << "Building IVF index took " << duration.count() << " s" << std::endl;

    runQueries(docs, queries, index, k, numProbes);
}