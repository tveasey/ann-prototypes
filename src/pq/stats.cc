#include "stats.h"

#include "constants.h"
#include "types.h"
#include "../common/evaluation.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>


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
               << "pq_mse,pq_normalize";
        for (const auto& m : EXPANSIONS) {
            writer << ",pq_qps_" << m;
        }
        for (const auto& m : EXPANSIONS) {
            writer << ",pq_query_min_recall_" << m
                   << ",pq_query_max_recall_" << m
                   << ",pq_query_avg_recall_" << m;
        }
        writer << std::endl;
    }

    std::ofstream writer(statsFile, std::ios_base::app);
    writer << stats.tag << "," << stats.metric << "," << stats.numQueries << ","
           << stats.numDocs << "," << stats.dim << "," << NUM_BOOKS << ","
           << BOOK_SIZE << "," << stats.k << "," << stats.bfQPS << ","
           << stats.pqCodeBookBuildTime << "," << BOOK_CONSTRUCTION_K_MEANS_ITR << ","
           << stats.pqCompressionRatio << "," << stats.pqMse << ","
           << stats.normalize;
    for (const auto& qps : stats.pqQPS) {
        writer << "," << qps;
    }
    for (const auto& recalls : stats.pqRecalls) {
        writer << "," << recalls[MIN_RECALL]
               << "," << recalls[MAX_RECALL]
               << "," << recalls[AVG_RECALL];
    }
    writer << std::endl;
}
