#include "utils.h"

#include "../common/utils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <tuple>
#include <utility>
#include <vector>
#include <iostream>

std::tuple<float, float, double> maximize(std::function<double (float, float)> f,
                                          std::size_t nProbes,
                                          const std::pair<float, float>& xrange,
                                          const std::pair<float, float>& yrange) {

    auto [xmin, xmax] = xrange;
    auto [ymin, ymax] = yrange;

    // Generate a set of candidate points. Note we sample at regular intervals
    // from a fine random grid. This gives quasi-uniform sampling.
    std::vector<std::pair<float, float>> candidates;
    candidates.reserve(nProbes * nProbes + 1);
    std::minstd_rand rng;
    std::uniform_real_distribution<float> u01{0.0, 1.0};
    std::size_t nGrid{2 * nProbes};
    std::vector<float> xcands(nGrid);
    std::vector<float> ycands(nGrid);
    for (std::size_t i = 0; i < nGrid; ++i) {
        float x{xmin + u01(rng) * (xmax - xmin)};
        float y{ymin + u01(rng) * (ymax - ymin)};
        xcands[i] = x;
        ycands[i] = y;
    }
    std::sort(xcands.begin(), xcands.end());
    std::sort(ycands.begin(), ycands.end());
    xcands.erase(std::unique(xcands.begin(), xcands.end()), xcands.end());
    ycands.erase(std::unique(ycands.begin(), ycands.end()), ycands.end());
    std::size_t dx{xcands.size() / nProbes};
    std::size_t dy{ycands.size() / nProbes};
    candidates.emplace_back(0.5F * (xmin + xmax), 0.5F * (ymin + ymax));
    for (std::size_t i = 0; i < nProbes; ++i) {
        for (std::size_t j = 0; j < nProbes; ++j) {
            candidates.emplace_back(xcands[i * dx], ycands[j * dy]);
        }
    }

    // Perform a greedy search over the candidates at each stage picking the
    // candidate with the maximum upper bound for the function. The upper bound
    // is computed by extrapolating using an estimate of the Lipschitz constant.
    std::vector<std::size_t> visited(nProbes);
    std::vector<std::size_t> remainder(candidates.size());
    std::vector<double> fxy(nProbes);
    std::vector<double> ub(candidates.size(), std::numeric_limits<double>::lowest());
    double l{0.0};

    // Initialize the first two probes at random.
    std::uniform_int_distribution<std::size_t> u0n{0, candidates.size() - 1};
    visited[0] = u0n(rng);
    do {
        visited[1] = u0n(rng);
    }
    while (visited[1] == visited[0]);
    if (visited[0] > visited[1]) {
        std::swap(visited[0], visited[1]);
    }
    std::iota(remainder.begin(), remainder.end(), 0);
    remainder.erase(remainder.begin() + visited[1]);
    remainder.erase(remainder.begin() + visited[0]);
    auto [x0, y0] = candidates[visited[0]];
    auto [x1, y1] = candidates[visited[1]];
    double f0{f(x0, y0)};
    double f1{f(x1, y1)};
    fxy[0] = f0;
    fxy[1] = f1;
    l = std::abs(f1 - f0) / std::hypot(x1 - x0, y1 - y0);

    for (auto i : remainder) {
        auto [x, y] = candidates[i];
        ub[i] = std::min(f0 + l * std::hypot(x - x0, y - y0),
                         f1 + l * std::hypot(x - x1, y - y1));
    }
    for (std::size_t i = 2; i < nProbes; ++i) {
        auto next = std::max_element(ub.begin(), ub.end()) - ub.begin();
        auto [x, y] = candidates[next];
        double fi{f(x, y)};

        visited[i] = next;
        fxy[i] = fi;
        ub[next] = std::numeric_limits<double>::lowest();
        remainder.erase(std::lower_bound(remainder.begin(), remainder.end(), next));

        // Update the Lipschitz constant.
        for (std::size_t j = 0; j < i; ++j) {
            auto [xj, yj] = candidates[visited[j]];
            double g{std::abs(fi - fxy[j]) / std::hypot(x - xj, y - yj)};
            l = std::max(l, g);
        }

        // Update the candidate upper bounds.
        for (auto j : remainder) {
            auto [xj, yj] = candidates[j];
            for (std::size_t k = 0; k <= i; ++k) {
                auto [xk, yk] = candidates[visited[k]];
                ub[j] = std::min(ub[j], fxy[k] + l * std::hypot(xj - xk, yj - yk));
            }
        }
    }

    auto best = std::max_element(fxy.begin(), fxy.end()) - fxy.begin();
    auto [xbest, ybest] = candidates[visited[best]];
    return {xbest, ybest, fxy[best]};
}
