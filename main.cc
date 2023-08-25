#include "src/io.h"
#include "src/pq.h"
#include "tests/pq_tests.h"

#include <fstream>
#include <iostream>

namespace {
void runSmokeTest() {

    std::minstd_rand rng;
    std::normal_distribution<> norm(2.0, 1.0);

    // 100k 256d corpus vectors.
    std::vector<float> docs(25600000);
    std::generate_n(docs.begin(), docs.size(), [&] { return norm(rng); });

    // 100 256d query vectors.
    std::vector<float> queries(25600);
    std::generate_n(queries.begin(), queries.size(), [&] { return norm(rng); });

    runPQBenchmark("smoke", Cosine, 10, 256, docs, queries);
}

void runExample(const std::string& dataset, Metric metric) {

    auto root = std::filesystem::path(__FILE__).parent_path();
    auto dim = readDimension(
        root / "data" / ("dim-" + dataset + ".txt"));
    auto docs = readVectors(
        dim, root / "data" / ("corpus-" + dataset + ".csv"), true);
    auto queries = readVectors(
        dim, root / "data" / ("queries-" + dataset + ".csv"), true);

    runPQBenchmark(dataset, metric, 10, dim, docs, queries, writePQStats);
}

std::string usage() {
    return "run_pq [-h,--help] [-u,--unit] [-s,--smoke] [-r,--run DATASET] [-n, --norm]\n"
           "\t--help\t\tShow this help\n"
           "\t--unit\t\tRun the unit tests\n"
           "\t--smoke\t\tRun the smoke test\n"
           "\t--run DATASET\tRun a test dataset\n"
           "\t--metric METRIC\tThe metric, must be cosine or dot, with which to compare vectors\n"
           "\t--norm\t\tNormalise quantised document vectors";
}
}

int main(int argc, char* argv[]) {

    bool unit{false};
    bool smoke{false};
    Metric metric{Cosine};
    std::string dataset;

    for (int i = 1; i < argc; ++i) {
        std::string arg{argv[i]};
        if (arg == "-h" || arg == "--help") {
            std::cout << usage() << std::endl;
            return 0;
        } else if (arg == "-u" || arg == "--unit") {
            unit = true;
        } else if (arg == "-s" || arg == "--smoke") {
            smoke = true;
        } else if (arg == "-r" || arg == "--run") {
            if (i + 1 == argc) {
                std::cerr << "Missing dataset. Usage:\n\n" << usage() << std::endl;
                return 1;
            }
            dataset = argv[i + 1];
        } else if (arg == "-m" || arg == "--metric") {
            if (i + 1 == argc) {
                std::cerr << "Missing metric. Usage:\n\n" << usage() << std::endl;
                return 1;
            }
            if (std::strcmp(argv[i + 1], "cosine") == 0) {
                metric = Cosine;
            } else if (std::strcmp(argv[i + 1], "dot") == 0) {
                metric = Dot;
            } else {
                std::cerr << "Bad metric " << argv[i + 1] << ". Usage:\n\n" << usage() << std::endl;
                return 1;
            }
        }
    }

    if (unit) {
        runUnitTests();
    }
    if (smoke) {
        runSmokeTest();
    }
    if (!dataset.empty()) {
        runExample(dataset, metric);
    }

    return 0;
}
