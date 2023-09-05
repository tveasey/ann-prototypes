#include "metrics.h"
#include "pq.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sys/stat.h>
#include <sys/types.h>

std::pair<std::vector<float>, std::size_t> readFvecs(const std::filesystem::path& file) {
    auto fileStr = file.u8string();
    auto* f = std::fopen(fileStr.c_str(), "r");
    if (f == nullptr) {
        std::cout << "Couldn't open " << file << std::endl;
        return {};
    }

    int dim;
    std::fread(&dim, 1, sizeof(int), f);
    std::fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    std::size_t sz = st.st_size;
    if (sz % ((dim + 1) * 4) != 0) {
        std::cout << "File must contain a whole number of vectors" << std::endl;
        return {};
    }

    std::size_t n{sz / ((dim + 1) * 4)};
    std::vector<float> vectors(n * (dim + 1));
    std::size_t nr{std::fread(vectors.data(), sizeof(float), n * (dim + 1), f)};
    if (nr != n * (dim + 1)) {
        std::cout << "Only read " << (nr / dim) << " out of "
                  << n << " vectors" << std::endl;
        return {};
    }

    // Shift to remove row headers.
    auto *x = vectors.data();
    for (std::size_t i = 0; i < n; i++) {
        std::memmove(x + i * dim, x + 1 + i * (dim + 1), dim * sizeof(*x));
    }
    vectors.resize(n * dim);

    std::fclose(f);

    return {std::move(vectors), static_cast<std::size_t>(dim)};
}

void writePQStats(const PQStats& stats) {
    auto fileExists = [](const auto& file) {
        std::ifstream test(file);
        return test.good();
    };

    auto root = std::filesystem::path(__FILE__).parent_path().parent_path();
    auto statsFile = root / "output" / "stats.csv";
    if (!fileExists(statsFile)) {
        // Header
        std::ofstream writer(statsFile, std::ios_base::out);
        writer << "tag,metric,num_queries,num_docs,dim,num_books,book_size,"
               << "top_k,bf_qps,pq_build_time,pq_k_means_itr,pq_compression,"
               << "pq_mse,pq_scann,pq_normalise";
        for (const auto& m : PQStats::EXPANSIONS) {
            writer << ",pq_qps_" << m;
        }
        for (const auto& m : PQStats::EXPANSIONS) {
            writer << ",pq_query_min_recall_" << m
                   << ",pq_query_max_recall_" << m
                   << ",pq_query_avg_recall_" << m;
        }
        writer << std::endl;
    }

    std::ofstream writer(statsFile, std::ios_base::app);
    writer << stats.tag << "," << stats.metric << "," << stats.numQueries << ","
           << stats.numDocs << "," << stats.dim << "," << numBooks() << ","
           << bookSize() << "," << stats.k << "," << stats.bfQPS << ","
           << stats.pqCodeBookBuildTime << "," << kMeansItr() << ","
           << stats.pqCompressionRatio << "," << stats.pqMse << ","
           << stats.scann << "," << stats.normalise;
    for (const auto& qps : stats.pqQPS) {
        writer << "," << qps;
    }
    for (const auto& recalls : stats.pqRecalls) {
        writer << "," << recalls[PQStats::MIN_RECALL]
               << "," << recalls[PQStats::MAX_RECALL]
               << "," << recalls[PQStats::AVG_RECALL];
    }
    writer << std::endl;
}
