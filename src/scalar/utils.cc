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

std::tuple<float, float, double> maximize(std::function<double (float, float)> f,
                                          std::size_t nProbes,
                                          const std::pair<float, float>& xrange,
                                          const std::pair<float, float>& yrange) {

    // Heavily based on https://arxiv.org/pdf/1703.02628.pdf.

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
    std::iota(remainder.begin(), remainder.end(), 0);
    double l{0.0};

    // Initialize with some random probes.
    for (std::size_t p = 0; p < nProbes / 4; ++p) {
        std::uniform_int_distribution<std::size_t> u0n{0, candidates.size() - p - 1};
        std::size_t next{remainder[u0n(rng)]};
        visited[p] = next;
        remainder.erase(std::lower_bound(remainder.begin(), remainder.end(), next));
        auto [x, y] = candidates[next];
        fxy[p] = f(x, y);
        for (std::size_t i = 0; i < p; ++i) {
            auto [xi, yi] = candidates[visited[i]];
            double li{std::abs(fxy[p] - fxy[i]) / std::hypot(x - xi, y - yi)};
            l = std::max(l, li);
        }
    }

    // Compute the upper bounds for the remaining candidates.
    std::vector<double> ub(candidates.size(), std::numeric_limits<double>::lowest());
    for (auto i : remainder) {
        auto [xi, yi] = candidates[i];
        double ubi{std::numeric_limits<double>::max()};
        for (std::size_t j = 0; j < nProbes / 4; ++j) {
            auto [xj, yj] = candidates[visited[j]];
            ubi = std::min(ubi, fxy[j] + l * std::hypot(xi - xj, yi - yj));
        }
        ub[i] = ubi;
    }

    for (std::size_t p = nProbes / 4; p < nProbes; ++p) {
        auto next = std::max_element(ub.begin(), ub.end()) - ub.begin();
        auto [x, y] = candidates[next];
        double fp{f(x, y)};

        visited[p] = next;
        fxy[p] = fp;
        ub[next] = std::numeric_limits<double>::lowest();
        remainder.erase(std::lower_bound(remainder.begin(), remainder.end(), next));

        // Update the Lipschitz constant.
        for (std::size_t i = 0; i < p; ++i) {
            auto [xj, yj] = candidates[visited[i]];
            double li{std::abs(fp - fxy[i]) / std::hypot(x - xj, y - yj)};
            l = std::max(l, li);
        }

        // Update the candidate upper bounds.
        for (auto i : remainder) {
            auto [xi, yi] = candidates[i];
            for (std::size_t j = 0; j <= p; ++j) {
                auto [xk, yk] = candidates[visited[j]];
                ub[i] = std::min(ub[i], fxy[j] + l * std::hypot(xi - xk, yi - yk));
            }
        }
    }

    auto best = std::max_element(fxy.begin(), fxy.end()) - fxy.begin();
    auto [xbest, ybest] = candidates[visited[best]];
    return {xbest, ybest, fxy[best]};
}