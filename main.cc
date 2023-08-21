#include "src/pq.h"
#include "src/io.h"
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

    runPQBenchmark(256, docs, queries);
}

void runExample(const std::string& dataset) {

    auto root = std::filesystem::path(__FILE__).parent_path();
    auto dim = readDimension(
        root / "data" / ("dim-" + dataset + ".txt"));
    auto docs = readVectors(
        dim, root / "data" / ("corpus-" + dataset + ".csv"), true);
    auto queries = readVectors(
        dim, root / "data" / ("queries-" + dataset + ".csv"), true);

    runPQBenchmark(dim, docs, queries);
}

std::string usage() {
    return "run_pq [-h,--help] [-u,--unit] [-s,--smoke] [-r,--run DIR]\n"
           "\t--help\t\tShow this help\n"
           "\t--unit\t\tRun the unit tests\n"
           "\t--smoke\t\tRun the smoke test\n"
           "\t--run DATASET\tRun a test dataset";
}
}

int main(int argc, char* argv[]) {

    bool unit{false};
    bool smoke{false};
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
            if (i +1 == argc) {
                std::cerr << "Bad input. Usage:\n\n" << usage() << std::endl;
                return 1;
            }
            dataset = argv[i + 1];
        }
    }

    if (unit) {
        runUnitTests();
    }
    if (smoke) {
        runSmokeTest();
    }
    if (!dataset.empty()) {
        runExample(dataset);
    }

    return 0;
}
