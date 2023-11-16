#pragma once

#include "../common/types.h"

#include <queue>
#include <tuple>
#include <utility>
#include <vector>

std::uint32_t dot8B(std::size_t dim,
                    const std::uint8_t*__restrict x,
                    const std::uint8_t*__restrict y);

std::uint32_t dot4B(std::size_t dim,
                    const std::uint8_t*__restrict x,
                    const std::uint8_t*__restrict y);

std::uint32_t dot4BP(std::size_t dim,
                     const std::uint8_t*__restrict x,
                     const std::uint8_t*__restrict y);

void pack4B(std::size_t dim,
             const std::uint8_t*__restrict raw,
             std::uint8_t*__restrict packed);

void unpack4B(std::size_t dim,
              const std::uint8_t*__restrict packed,
              std::uint8_t*__restrict raw);

std::pair<float, float>
quantiles(std::size_t dim, const std::vector<float>& vectors, float ci);

std::pair<std::vector<std::uint8_t>, std::vector<float>>
scalarQuantise8B(const std::pair<float, float>& range,
                 std::size_t dim,
                 const std::vector<float>& dequantised);

std::vector<float> scalarDequantise8B(const std::pair<float, float>& range,
                                      std::size_t dim,
                                      const std::vector<std::uint8_t>& quantised);

std::pair<std::vector<std::uint8_t>, std::vector<float>>
scalarQuantise4B(const std::pair<float, float>& range,
                 bool pack,
                 std::size_t dim,
                 const std::vector<float>& dequantised);

std::vector<float> scalarDequantise4B(const std::pair<float, float>& range,
                                      bool packed,
                                      std::size_t dim,
                                      const std::vector<std::uint8_t>& quantised);


void runScalarBenchmark(const std::string& tag,
                        Metric metric,
                        ScalarBits bits,
                        std::size_t k,
                        std::size_t dim,
                        std::vector<float>& docs,
                        std::vector<float>& queries);
