#include "src/common/bigvector.h"
#include "src/common/io.h"
#include "src/common/types.h"
#include "src/pq/benchmark.h"
#include "src/pq/constants.h"
#include "src/pq/utils.h"
#include "src/scalar/scalar.h"
#include "src/scalar/utils.h"
#include "src/soar/benchmark.h"

#include <boost/program_options.hpp>
#include <boost/program_options/value_semantic.hpp>

#include <algorithm>
#include <iostream>
#include <optional>
#include <random>

namespace {

void generateBenchmark(const std::string& dataset,
                      int dimension,
                      std::size_t numVecs) {

    auto root = std::filesystem::path(__FILE__).parent_path();

    std::cout << "Writing " << numVecs << " vectors with dimension " << dimension
              << " to " << (root / "data" / (dataset + ".fvec")) << std::endl;

    std::vector<float> result(dimension);
    std::minstd_rand rng;
    std::uniform_real_distribution<float> u01{0.0F, 1.0F};
    auto randDocs = [&] {
        std::generate_n(result.begin(), dimension, [&] { return u01(rng); });
        return result;
    };
    writeFvecs(root / "data" / (dataset + ".fvec"), dimension, numVecs, randDocs);

    std::cout << "Wrote " << numVecs << " vectors with dimension " << dimension
              << " to " << (root / "data" / (dataset + ".fvec")) << std::endl;
}

void loadAndRunPqBenchmark(const std::string& dataset,
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

    runPqBenchmark(dataset, metric, distanceThreshold, docsPerCoarseCluster,
                   numBooks, 10, docs, queries, writePqStats);
}

void loadAndRunPqMergeBenchmark(const std::string& dataset,
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
    BigVector docs1{loadAndPrepareData(
        root / "data" / ("corpus-" + dataset + ".fvec"), numBooks, metric == Cosine, {0.0, 0.5})};
    BigVector docs2{loadAndPrepareData(
        root / "data" / ("corpus-" + dataset + ".fvec"), numBooks, metric == Cosine, {0.5, 1.0})};
    std::cout << "Loaded " << (docs1.numVectors() + docs2.numVectors())
              << " vectors of dimension " << docs1.dim() << std::endl;

    if (qdim != docs1.dim()) {
        throw std::runtime_error("Dimension mismatch");
    }
    if (docs1.numVectors() == 0 || docs2.numVectors() == 0 || queries.empty()) {
        return;
    }

    runPqMergeBenchmark(dataset, metric, distanceThreshold, docsPerCoarseCluster,
                        numBooks, 10, docs1, docs2, queries, writePqStats);
}

void testSoarIVF(const std::string& dataset,
                 Metric metric,
                 float lambda,
                 std::size_t docsPerCluster) {
    
    auto root = std::filesystem::path(__FILE__).parent_path();

    std::cout << "Loading queries from "
              << (root / "data" / ("queries-" + dataset + ".fvec")) << std::endl;
    auto [queries, qdim] = readFvecs(root / "data" / ("queries-" + dataset + ".fvec"));
    std::cout << "Loaded " << queries.size() / qdim << " queries of dimension " << qdim << std::endl;
    
    std::cout << "Loading corpus from "
              << (root / "data" / ("corpus-" + dataset + ".fvec")) << std::endl;
    BigVector docs{loadAndPrepareData(
        root / "data" / ("corpus-" + dataset + ".fvec"), 1, metric == Cosine)};
    std::cout << "Loaded " << docs.numVectors() << " vectors of dimension " << docs.dim() << std::endl;
    
    if (qdim != docs.dim()) {
        throw std::runtime_error("Dimension mismatch");
    }
    if (docs.numVectors() == 0 || queries.empty()) {
        return;
    }

    runSoarIVFBenchmark(metric, docs, queries, lambda, docsPerCluster, 10, docsPerCluster);
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

    bool generate{false};
    int dimension{1024};
    std::size_t numVecs{16UL * 1024UL * 1024UL};
    std::optional<ScalarBits> scalar;
    Metric metric{Cosine};
    bool merge{false};
    float distanceThreshold{0.0F};
    std::size_t docsPerCoarseCluster{COARSE_CLUSTERING_DOCS_PER_CLUSTER};
    std::size_t dimensionsPerCode{8};
    std::string dataset;

    boost::program_options::options_description desc("Usage: run_benchmark\nOptions");
    desc.add_options()
        ("help,h", "Show this help")
        ("generate,g", boost::program_options::bool_switch(),
            "Generate random data with the specified vector count and dimension")
        ("dim,d", boost::program_options::value<int>()->default_value(1024),
            "The dimension of the data to generate")
        ("num-vecs,v", boost::program_options::value<std::size_t>()->default_value(16UL * 1024UL * 1024UL),
            "The number of document vectors to generate")
        ("scalar,s", boost::program_options::value<std::string>(),
            "Use 1, 4, 4P or 8 bit scalar quantisation. If not supplied then run PQ")
        ("run,r", boost::program_options::value<std::string>(),
            "Run a test dataset")
        ("metric,m", boost::program_options::value<std::string>()->default_value("cosine"),
            "The metric, must be cosine, dot or euclidean with which to compare vectors")
        ("merge", boost::program_options::bool_switch(),
            "Run the merge benchmark instead of the standard benchmark")
        ("perp-distance-threshold", boost::program_options::value<float>()->default_value(0.0F),
            "The ScaNN threshold used for computing the parallel distance cost multiplier")
        ("docs-per-coarse-cluster", boost::program_options::value<std::size_t>()->default_value(COARSE_CLUSTERING_DOCS_PER_CLUSTER),
            "The number of documents per coarse cluster in the PQ index")
        ("dimensions-per-code", boost::program_options::value<std::size_t>()->default_value(16),
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
        if (vm.count("generate")) {
            generate = vm["generate"].as<bool>();
        }
        if (vm.count("dim")) {
            dimension = vm["dim"].as<int>();
            if (dimension <= 0) {
                throw boost::program_options::error("Invalid dimension");
            }
        }
        if (vm.count("num-vecs")) {
            numVecs = vm["num-vecs"].as<std::size_t>();
            if (numVecs == 0) {
                throw boost::program_options::error("Invalid number of vectors");
            }
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
        if (vm.count("merge")) {
            merge = vm["merge"].as<bool>();
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

    if (generate) {
        generateBenchmark(dataset, dimension, numVecs);
    } else if (!dataset.empty()) {
        try {
            if (scalar != std::nullopt) {
                loadAndRunScalarBenchmark(dataset, metric, *scalar);
            } else if (merge) {
                loadAndRunPqMergeBenchmark(dataset, metric, distanceThreshold,
                                           docsPerCoarseCluster, dimensionsPerCode);
            } else {
                loadAndRunPqBenchmark(dataset, metric, distanceThreshold,
                                      docsPerCoarseCluster, dimensionsPerCode);
            }
        } catch (const std::exception& e) {
            std::cerr << "Caught exception: " << e.what() << std::endl;
            return 1;
        }
    }

    return 0;
}
