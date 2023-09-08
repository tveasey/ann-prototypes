#include "bruteforce.h"

void searchBruteForce(std::size_t k,
                      const std::vector<float>& docs,
                      const std::vector<float>& query,
                      std::priority_queue<std::pair<float, std::size_t>>& topk) {
    std::size_t dim{query.size()};
    for (std::size_t i = 0, id = 0; i < docs.size(); i += dim, ++id) {
        float sim{0.0F};
        #pragma clang loop unroll_count(8) vectorize(assume_safety)
        for (std::size_t j = 0; j < dim; ++j) {
            sim += query[j] * docs[i + j];
        }
        float dist{1.0F - sim};
        if (topk.size() < k) {
            topk.push(std::make_pair(dist, id));
        } else if (topk.top().first > dist) {
            topk.pop();
            topk.push(std::make_pair(dist, id));
        }
    }
}
