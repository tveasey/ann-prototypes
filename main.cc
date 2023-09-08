#include "src/io.h"
#include "src/pq.h"
#include "src/scalar.h"
#include "src/utils.h"
#include "tests/tests.h"

#include <fstream>
#include <iostream>
#include <optional>

namespace {
void loadAndRunPQBenchmark(const std::string& dataset, Metric metric, bool scann) {
    auto root = std::filesystem::path(__FILE__).parent_path();
    auto [docs, ddim] = readFvecs(root / "data" / ("corpus-" + dataset + ".fvec"));
    auto [queries, qdim] = readFvecs(root / "data" / ("queries-" + dataset + ".fvec"));
    if (ddim != qdim) {
        std::cout << "Dimension mismatch " << ddim << " != " << qdim << std::endl;
        return;
    }
    runPQBenchmark(dataset, scann, metric, 10, qdim, docs, queries, writePQStats);
}

void loadAndRunScalarBenchmark(const std::string& dataset, Metric metric, ScalarBits bits) {
    auto root = std::filesystem::path(__FILE__).parent_path();
    auto [docs, ddim] = readFvecs(root / "data" / ("corpus-" + dataset + ".fvec"));
    auto [queries, qdim] = readFvecs(root / "data" / ("queries-" + dataset + ".fvec"));
    if (ddim != qdim) {
        std::cout << "Dimension mismatch " << ddim << " != " << qdim << std::endl;
        return;
    }
    runScalarBenchmark(dataset, metric, bits, 10, qdim, docs, queries);
}

std::string usage() {
    return "run_quantisation [-h,--help] [-u,--unit] [-s,--scalar] [--scann] [-r,--run DATASET] [-m, --metric METRIC]\n"
           "\t--help\t\tShow this help\n"
           "\t--unit\t\tRun the unit tests (default false)\n"
           "\t--scalar N\tUse 4 or 8 bit scalar quantisation (default None)\n"
           "\t--scann\t\tUse anisotrpoic loss when building code books (default false)\n"
           "\t--run DATASET\tRun a test dataset\n"
           "\t--metric METRIC\tThe metric, must be cosine or dot, with which to compare vectors (default cosine)";
}
}

int main(int argc, char* argv[]) {

    bool unit{false};
    bool smoke{false};
    std::optional<ScalarBits> scalar;
    bool scann{false};
    Metric metric{Cosine};
    std::string dataset;

    for (int i = 1; i < argc; ++i) {
        std::string arg{argv[i]};
        if (arg == "-h" || arg == "--help") {
            std::cout << usage() << std::endl;
            return 0;
        } else if (arg == "-u" || arg == "--unit") {
            unit = true;
        } else if (arg == "-s" || arg == "--scalar") {
            if (i + 1 == argc) {
                std::cerr << "Missing dataset. Usage:\n\n" << usage() << std::endl;
                return 1;
            }
            if (std::strcmp(argv[i + 1], "4") == 0) {
                scalar = Scalar4Bit;
            } else if (std::strcmp(argv[i + 1], "8") == 0) {
                scalar = Scalar8Bit;
            } else {
                std::cerr << "Unsupported bits " << argv[i + 1] << ". Usage:\n\n" << usage() << std::endl;
                return 1;
            }
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
        } else if (arg == "--scann") {
            scann = true;
        }
    }

    if (unit) {
        runUnitTests();
    }
    if (!dataset.empty()) {
        if (scalar != std::nullopt) {
            loadAndRunScalarBenchmark(dataset, metric, *scalar);
        } else {
            loadAndRunPQBenchmark(dataset, metric, scann);
        }
    }

    return 0;
}
