#include "metrics.h"
#include "pq.h"

#include <algorithm>
#include <cmath>

double computeRecall(const std::vector<std::size_t>& nnTrue,
                     std::vector<std::size_t> nnApprox) {

    std::sort(nnApprox.begin(), nnApprox.end());

    double hits{0.0};
    for (auto i : nnTrue) {
        hits += std::binary_search(nnApprox.begin(), nnApprox.end(), i) ? 1.0 : 0.0;
    }

    return hits / static_cast<double>(nnTrue.size());
}

double computeCompressionRatio(std::size_t dim) {
    return static_cast<double>(dim * sizeof(float)) /
           static_cast<double>(numBooks() * sizeof(code_t));
}

PQStats::recalls_t computeRecalls(const std::vector<std::vector<std::size_t>>& nnExact,
                                  const std::vector<std::vector<std::size_t>>& nnPQ) {
    PQStats::recalls_t recalls;
    recalls[PQStats::MIN_RECALL] = 1.0;
    recalls[PQStats::MAX_RECALL] = 0.0;
    recalls[PQStats::AVG_RECALL] = 0.0;
    for (std::size_t i = 0; i < nnExact.size(); ++i) {
        auto recall = computeRecall(nnExact[i], nnPQ[i]);
        recalls[PQStats::MIN_RECALL] = std::min(recalls[PQStats::MIN_RECALL], recall);
        recalls[PQStats::MAX_RECALL] = std::max(recalls[PQStats::MAX_RECALL], recall);
        recalls[PQStats::AVG_RECALL] += recall;
    }
    recalls[PQStats::AVG_RECALL] /= static_cast<double>(nnExact.size());
    return recalls;
}
