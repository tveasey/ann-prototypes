#pragma once

#include "stats.h" 
#include "../common/bigvector.h"
#include "../common/types.h"

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

void runPQBenchmark(const std::string& tag,
                    Metric metric,
                    std::size_t k,
                    const BigVector& docs,
                    std::vector<float>& queries,
                    const std::function<void(const PQStats&)>& writeStats);
