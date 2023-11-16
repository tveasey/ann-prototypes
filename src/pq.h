#pragma once

#include "pq_types.h"
#include "utils.h"

#include <cstdint>
#include <functional>
#include <ostream>
#include <queue>
#include <random>
#include <set>
#include <utility>
#include <vector>

struct PQStats;

std::vector<float> buildDistTable(const std::vector<float>& codeBooks,
                                  const std::vector<float>& query);

std::vector<float> buildDistNorm2Table(const std::vector<float>& codeBooks,
                                       const std::vector<float>& query);

float computeDist(const std::vector<float>& distTable,
                  const code_t* docCode);

float computeNormedDist(const std::vector<float>& distTable,
                        const code_t* docCode);

std::vector<float> encoded(std::size_t dim,
                           const std::vector<float>& codeBooks,
                           const code_t* docCode);

void searchPQ(std::size_t k,
              const std::vector<float>& codeBooks,
              const std::vector<code_t>& docsCodes,
              const std::vector<float>& docsNorms,
              const std::vector<float>& query,
              bool normalise,
              std::priority_queue<std::pair<float, std::size_t>>& topk);

void runPQBenchmark(const std::string& tag,
                    Metric metric,
                    std::size_t k,
                    std::size_t dim,
                    std::vector<float>& docs,
                    std::vector<float>& queries,
                    const std::function<void(const PQStats&)>& writeStats = [](const PQStats&) {});
