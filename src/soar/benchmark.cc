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
    std::cout << std::setprecision(5)
                << "query count = " << numQueries
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

    diff = std::chrono::duration<double>{0};
    progress = std::make_unique<ProgressBar>("PQ search...", numQueries);
    for (std::size_t i = 0, j = 0; i < queries.size(); i += dim, ++j) {
        diff += time([&] { topkSoarIVF[j] = index.search(queries.data() + i * dim, k, numProbes); });
        progress->update();
    }
    progress.reset();
    std::cout << "PQ search took " << diff.count() << " s" << std::endl;
    std::cout << "QPS = " << numQueries / diff.count() << std::endl;
    std::cout << "Average recall@" << k << " = " << computeRecalls(topkExact, topkSoarIVF)[AVG_RECALL] << std::endl;
}
    
} // unnamed::
    
void runSoarIVFBenchmark(Metric metric,
                         const BigVector& docs,
                         std::vector<float>& queries,
                         float lambda,
                         std::size_t docsPerCluster,
                         std::size_t k,
                         std::size_t numProbes) {

    std::cout << "Running Soar IVF benchmark..." << std::endl;
    std::cout << "metric = " << toString(metric)
              << ", top-k = " << k
              << ", normalized = " << (metric == Cosine) << std::endl;

    std::size_t numClusters{
        std::max(1UL, (docs.numVectors() + docsPerCluster - 1) / docsPerCluster)};

    SoarIVFIndex index{metric, lambda, docs.dim(), numClusters};

    std::cout << "Building IVF index..." << std::endl;
    index.build(docs);
    std::cout << "IVF index built" << std::endl;

    runQueries(docs, queries, index, k, numProbes);
}