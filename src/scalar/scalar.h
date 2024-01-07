#pragma once

#include "../common/types.h"

#include <cstdint>
#include <queue>
#include <string>
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

std::uint32_t dot1B(std::size_t dim,
                    const std::uint32_t*__restrict x,
                    const std::uint32_t*__restrict y);

void pack4B(std::size_t dim,
            const std::uint8_t*__restrict raw,
            std::uint8_t*__restrict packed);

void unpack4B(std::size_t dim,
              const std::uint8_t*__restrict packed,
              std::uint8_t*__restrict raw);

std::pair<float, float>
quantiles(std::size_t dim, const std::vector<float>& vectors, float ci);

std::pair<float, float> computeBuckets1B(std::size_t dim, std::vector<float>& docs);

std::pair<std::vector<std::uint8_t>, std::vector<float>>
scalarQuantise8B(const std::pair<float, float>& range,
                 std::size_t dim,
                 const std::vector<float>& dequantised);

std::vector<float> scalarDequantise8B(const std::pair<float, float>& range,
                                      std::size_t dim,
                                      const std::vector<std::uint8_t>& quantised);

void searchScalarQuantise8B(std::size_t k,
                            const std::pair<float, float>& range,
                            const std::vector<std::uint8_t>& docs,
                            const std::vector<float>& p1,
                            const std::vector<float>& query,
                            std::priority_queue<std::pair<float, std::size_t>>& topk);

std::pair<std::vector<std::uint8_t>, std::vector<float>>
scalarQuantise4B(const std::pair<float, float>& range,
                 bool pack,
                 std::size_t dim,
                 const std::vector<float>& dequantised);

std::vector<float> scalarDequantise4B(const std::pair<float, float>& range,
                                      bool packed,
                                      std::size_t dim,
                                      const std::vector<std::uint8_t>& quantised);

void searchScalarQuantise4B(std::size_t k,
                            const std::pair<float, float>& range,
                            bool packed,
                            const std::vector<std::uint8_t>& docs,
                            const std::vector<float>& p1,
                            const std::vector<float>& query,
                            std::priority_queue<std::pair<float, std::size_t>>& topk);

std::pair<std::vector<std::uint32_t>, std::vector<float>>
scalarQuantise1B(const std::pair<float, float>& bucketCentres,
                 std::size_t dim,
                 const std::vector<float>& dequantised);

std::vector<float> scalarDequantise1B(const std::pair<float, float>& bucketCentres,
                                      std::size_t dim,
                                      const std::vector<std::uint32_t>& quantised);

void searchScalarQuantise1B(std::size_t k,
                            const std::pair<float, float>& bucketCentres,
                            const std::vector<std::uint32_t>& docs,
                            const std::vector<float>& p1,
                            const std::vector<float>& query,
                            std::priority_queue<std::pair<float, std::size_t>>& topk);

void runScalarBenchmark(const std::string& tag,
                        Metric metric,
                        ScalarBits bits,
                        std::size_t k,
                        std::size_t dim,
                        std::vector<float>& docs,
                        std::vector<float>& queries);
