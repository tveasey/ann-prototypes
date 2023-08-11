#include "src/pq.h"
#include "tests/pq_tests.h"

#include <fstream>
#include <iostream>

namespace {
void runExample(std::string dir) {
    std::ifstream file{dir.c_str(), std::ios_base::binary};
    int vecSize;
    file.read(reinterpret_cast<char*>(&vecSize), 4);
    std::cout << vecSize << std::endl;
    // file.read(reinterpret_cast<char*>(docHashes.data()), docHashesBytes);
}

void runSmokeTest() {

    std::minstd_rand rng;
    std::normal_distribution<> norm(2.0, 1.0);

    // 10k 256d corpus vectors. 
    std::vector<float> docs(25600000);
    std::generate_n(docs.begin(), docs.size(), [&] { return norm(rng); });

    // 100 256d query vectors.
    std::vector<float> queries(25600);
    std::generate_n(queries.begin(), queries.size(), [&] { return norm(rng); });

    runPQBenchmark(256, docs, queries);
}

std::string usage() {
    return "pq [-h,--help] [-u,--unit] [-s,--smoke] [-r,--run DIR]\n"
           "\t--help\t\tShow this help\n"
           "\t--unit\t\tRun the unit tests\n"
           "\t--smoke\t\tRun the smoke test\n"
           "\t--run DIR\tRun the example from the gist in DIR";
}
}

int main(int argc, char* argv[]) {

    bool unit{false};
    bool smoke{false};
    std::string dir;

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
            dir = argv[i + 1];
        }
    }

    if (unit) {
        runUnitTests();
    }
    if (smoke) {
        runSmokeTest();
    }
    if (!dir.empty()) {
        runExample(dir);
    }

    return 0;
}
