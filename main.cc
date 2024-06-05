#include "src/common/bigvector.h"
#include "src/common/io.h"
#include "src/common/types.h"
#include "src/pq/benchmark.h"
#include "src/pq/constants.h"
#include "src/pq/utils.h"
#include "src/scalar/scalar.h"
#include "src/common/utils.h"

#include <boost/program_options.hpp>

#include <fstream>
#include <iostream>
#include <optional>
#include <random>

namespace {

void loadAndRunPQBenchmark(const std::string& dataset,
                           Metric metric,
                           float distanceThreshold,
                           std::size_t docsPerCoarseCluster,
                           std::size_t dimensionsPerCode) {

    auto root = std::filesystem::path(__FILE__).parent_path();

    std::cout << "Loading queries from "
              << (root / "data" / ("queries-" + dataset + ".fvec")) << std::endl;
    auto [queries, qdim] = readFvecs(root / "data" / ("queries-" + dataset + ".fvec"));
    std::cout << "Loaded " << queries.size() / qdim << " queries of dimension " << qdim << std::endl;

    // We pad to a multiple of the number of books so round up.
    std::size_t numBooks{(qdim + dimensionsPerCode - 1) / dimensionsPerCode};

    zeroPad(qdim, numBooks, queries);

    std::cout << "Loading corpus from "
              << (root / "data" / ("corpus-" + dataset + ".fvec")) << std::endl;
    BigVector docs{loadAndPrepareData(
        root / "data" / ("corpus-" + dataset + ".fvec"), numBooks, metric == Cosine)};
    std::cout << "Loaded " << docs.numVectors() << " vectors of dimension " << docs.dim() << std::endl;

    if (qdim != docs.dim()) {
        throw std::runtime_error("Dimension mismatch");
    }
    if (docs.numVectors() == 0 || queries.empty()) {
        return;
    }

    runPQBenchmark(dataset, metric, distanceThreshold, docsPerCoarseCluster,
                   numBooks, 10, docs, queries, writePQStats);
}

void loadAndRunScalarBenchmark(const std::string& dataset, Metric metric, ScalarBits bits) {
    auto root = std::filesystem::path(__FILE__).parent_path();
    auto [docs, ddim] = readFvecs(root / "data" / ("corpus-" + dataset + ".fvec"));
    auto [queries, qdim] = readFvecs(root / "data" / ("queries-" + dataset + ".fvec"));
    if (ddim != qdim) {
        throw std::runtime_error("Dimension mismatch");
    }
    if (docs.empty() || queries.empty()) {
        return;
    }
    runScalarBenchmark(dataset, metric, bits, 10, qdim, docs, queries);
}

} // unnamed::

int main(int argc, char* argv[]) {

    std::optional<ScalarBits> scalar;
    Metric metric{Cosine};
    float distanceThreshold{0.0F};
    std::size_t docsPerCoarseCluster{COARSE_CLUSTERING_DOCS_PER_CLUSTER};
    std::size_t dimensionsPerCode{8};
    std::string dataset;

    boost::program_options::options_description desc("Usage: run_benchmark\nOptions");
    desc.add_options()
        ("help,h", "Show this help")
        ("scalar,s", boost::program_options::value<std::string>(),
            "Use 1, 4, 4P or 8 bit scalar quantisation. If not supplied then run PQ")
        ("run,r", boost::program_options::value<std::string>(),
            "Run a test dataset")
        ("metric,m", boost::program_options::value<std::string>()->default_value("cosine"),
            "The metric, must be cosine, dot or euclidean with which to compare vectors")
        ("distance", boost::program_options::value<float>()->default_value(0.0F),
            "The ScaNN threshold used for computing the parallel distance cost multiplier")
        ("docs-per-coarse-cluster", boost::program_options::value<std::size_t>()->default_value(COARSE_CLUSTERING_DOCS_PER_CLUSTER),
            "The number of documents per coarse cluster in the PQ index")
        ("dimensions-per-code", boost::program_options::value<std::size_t>()->default_value(8),
            "The number of dimensions per code in the PQ index");

    try {
        boost::program_options::variables_map vm;
        boost::program_options::store(
            boost::program_options::parse_command_line(argc, argv, desc), vm);
        boost::program_options::notify(vm);
        if (vm.count("help")) {
            std::cerr << desc << std::endl;
            return 0;
        }
        if (vm.count("scalar")) {
            auto s = vm["scalar"].as<std::string>();
            if (s == "1") {
                scalar = B1;
            } else if (s == "4") {
                scalar = B4;
            } else if (s == "4P") {
                scalar = B4P;
            } else if (s == "8") {
                scalar = B8;
            } else if (s != "None") {
                throw boost::program_options::error("Invalid scalar quantisation");
            }
        }
        if (vm.count("run")) {
            dataset = vm["run"].as<std::string>();
        }
        if (vm.count("metric")) {
            auto m = vm["metric"].as<std::string>();
            if (m == "cosine") {
                metric = Cosine;
            } else if (m == "dot") {
                metric = Dot;
            } else if (m == "euclidean") {
                metric = Euclidean;
            } else {
                throw boost::program_options::error("Invalid metric");
            }
        }
        if (vm.count("distance")) {
            distanceThreshold = vm["distance"].as<float>();
        }
        if (vm.count("docs-per-coarse-cluster")) {
            docsPerCoarseCluster = vm["docs-per-coarse-cluster"].as<std::size_t>();
            if (docsPerCoarseCluster == 0) {
                throw boost::program_options::error("Invalid docs per coarse cluster");
            }
        }
        if (vm.count("dimensions-per-code")) {
            dimensionsPerCode = vm["dimensions-per-code"].as<std::size_t>();
            if (dimensionsPerCode == 0) {
                throw boost::program_options::error("Invalid dimensions per code");
            }
        }
    } catch (const boost::program_options::error& e) {
        std::cerr << "Error parsing command line: " << e.what() << std::endl;
        std::cerr << desc << std::endl;
        return 1;
    }

    if (!dataset.empty()) {
        try {
            if (scalar != std::nullopt) {
                loadAndRunScalarBenchmark(dataset, metric, *scalar);
            } else {
                loadAndRunPQBenchmark(dataset, metric, distanceThreshold,
                                      docsPerCoarseCluster, dimensionsPerCode);
            }
        } catch (const std::exception& e) {
            std::cerr << "Caught exception: " << e.what() << std::endl;
            return 1;
        }
    }

    return 0;
}
