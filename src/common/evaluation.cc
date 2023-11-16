#include "evaluation.h"

#include <algorithm>

double computeRecall(const std::vector<std::size_t>& nnTrue,
                     std::vector<std::size_t> nnApprox) {
    std::sort(nnApprox.begin(), nnApprox.end());
    return static_cast<double>(
               std::count_if(nnTrue.begin(), nnTrue.end(),
                             [&nnApprox](std::size_t i) {
                                 return std::binary_search(nnApprox.begin(), nnApprox.end(), i);
                             })) / static_cast<double>(nnTrue.size());
}

recalls_t computeRecalls(const std::vector<std::vector<std::size_t>>& nnExact,
                         const std::vector<std::vector<std::size_t>>& nnApprox) {
    recalls_t recalls;
    recalls[MIN_RECALL] = 1.0;
    recalls[MAX_RECALL] = 0.0;
    recalls[AVG_RECALL] = 0.0;
    for (std::size_t i = 0; i < nnExact.size(); ++i) {
        auto recall = computeRecall(nnExact[i], nnApprox[i]);
        recalls[MIN_RECALL] = std::min(recalls[MIN_RECALL], recall);
        recalls[MAX_RECALL] = std::max(recalls[MAX_RECALL], recall);
        recalls[AVG_RECALL] += recall;
    }
    recalls[AVG_RECALL] /= static_cast<double>(nnExact.size());
    return recalls;
}
