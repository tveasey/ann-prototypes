#include "scalar.h"

#include "bruteforce.h"
#include "metrics.h"
#include "utils.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <queue>
#include <tuple>

namespace {
#if defined(__ARM_NEON__)

#include <arm_neon.h>

std::uint32_t dot4BM(std::size_t dim,
                     const std::uint8_t*__restrict x,
                     const std::uint8_t*__restrict y) {
    // This special case assumes that the vector dimension is:
    //   1. A multiple of 16
    //   2. Less than 256 * 16 = 4096 (or sums have the risk of overflowing).
    //
    // Both these cases will commonly hold. We call this as a subroutine for
    // the general implementation.

    uint16x8_t xysuml{vdupq_n_u16(0)};
    uint16x8_t xysumh{vdupq_n_u16(0)};

    for (std::size_t i = 0; i < dim; i += 16) {
        // Read into 16 x 8 bit vectors.
        uint8x16_t xb{vld1q_u8(x + i)};
        uint8x16_t yb{vld1q_u8(y + i)};
        // Multiply.
        uint8x16_t xyb{vmulq_u8(xb, yb)};
        // Split into 2 x 8 x 16 bit vectors in which type we accumulate.
        uint16x8_t xybl{vmovl_u8(vget_low_u8(xyb))};
        uint16x8_t xybh{vmovl_u8(vget_high_u8(xyb))};
        // Accumulate.
        xysuml = vaddq_u16(xysuml, xybl);
        xysumh = vaddq_u16(xysumh, xybh);
    }

    return vaddlvq_u16(xysuml) + vaddlvq_u16(xysumh);
}

#else

std::uint32_t dot4BM(std::size_t dim,
                     const std::uint8_t*__restrict x,
                     const std::uint8_t*__restrict y) {

    // Tell the compiler dim contraints.
    dim = std::min(dim, static_cast<std::size_t>(4096)) & ~0xF;

    std::uint32_t xy{0};
    #pragma clang loop unroll_count(16) vectorize(assume_safety)
    for (std::size_t i = 0; i < dim; ++i) {
        xy += x[i] * y[i];
    }
    return xy;
}

#endif

std::uint32_t dot4BR(std::size_t dim,
                     const std::uint8_t*__restrict x,
                     const std::uint8_t*__restrict y) {
    std::uint32_t xy{0};
    #pragma clang loop unroll_count(16) vectorize(assume_safety)
    for (std::size_t i = 0; i < dim; ++i) {
        xy += x[i] * y[i];
    }
    return xy;
}
}

std::uint32_t dot4B(std::size_t dim,
                    const std::uint8_t*__restrict x,
                    const std::uint8_t*__restrict y) {
    std::size_t remainder{dim & 0xF};
    dim -= remainder;
    std::uint32_t xy{0};
    for (std::size_t i = 0; i < dim; i += 4096) {
        xy += dot4BM(std::min(dim, i + 4096), x + i, y + i);
    }
    if (remainder > 0) {
        xy += dot4BR(remainder, x + dim, y + dim);
    }
    return xy;
}

std::pair<float, float>
quantiles(std::size_t dim, const std::vector<float>& docs, float ci) {

    std:size_t numDocs{docs.size() / dim};

    std::minstd_rand rng;
    double p{std::min(25000.0 / static_cast<double>(numDocs), 1.0)};
    auto sampled = sampleDocs(p, dim, docs, rng);

    auto lower = static_cast<std::size_t>(
        std::round(static_cast<double>(sampled.size()) * 0.5 * (1.0 - ci)));
    auto upper = static_cast<std::size_t>(
        std::round(static_cast<double>(sampled.size()) * 0.5 * (1.0 + ci)));

    std::nth_element(sampled.begin(), sampled.begin() + lower, sampled.end());
    std::nth_element(sampled.begin() + lower + 1, sampled.begin() + upper, sampled.end());

    return {sampled[lower], sampled[upper]};
}

std::pair<std::vector<std::uint8_t>, std::vector<float>>
scalarQuantise8B(const std::pair<float, float>& range,
                 std::size_t dim,
                 const std::vector<float>& dequantised) {

    auto [lower, upper] = range;
    std:size_t numDocs{dequantised.size() / dim};
    float scale{255.0F / (upper - lower)};
    float invScale{(upper - lower) / 255.0F};

    std::vector<float> p1(numDocs, 0.0F);
    std::vector<std::uint8_t> quantised(dequantised.size());

    for (std::size_t i = 0, k = 0; i < dequantised.size(); i += dim, ++k) {
        #pragma clang unroll_count(2) vectorise(assume_safety)
        for (std::size_t j = i; j < i + dim; ++j) {
            float x{dequantised[j]};
            float dx{std::clamp(x, lower, upper) - lower};
            float dxs{scale * dx};
            float dxq{invScale * std::round(dxs)};
            //p1[k] += lower * (lower / 2.0F + dxq);
            p1[k] += lower * (x - lower / 2.0F) + (dx - dxq) * dxq;
            quantised[j] = static_cast<std::uint8_t>(std::round(dxs));
        }
    }
    return {std::move(quantised), std::move(p1)};
}

std::vector<float> scalarDequantise8B(const std::pair<float, float>& range,
                                      std::size_t dim,
                                      const std::vector<std::uint8_t>& quantised) {
    auto [lower, upper] = range;
    std::vector<float> dequantised(quantised.size());
    float scale{(upper - lower) / 255.0F};
    #pragma clang unroll_count(8) vectorise(assume_safety)
    for (std::size_t i = 0; i < quantised.size(); ++i) {
        dequantised[i] = lower + scale * static_cast<float>(quantised[i]);
    }
    return dequantised;
}

void searchScalarQuantise8B(std::size_t k,
                            const std::pair<float, float>& range,
                            const std::vector<std::uint8_t>& docs,
                            std::vector<float> p1,
                            const std::vector<float>& query,
                            std::priority_queue<std::pair<float, std::size_t>>& topk) {

    auto [lower, upper] = range;
    std::size_t dim{query.size()};
    float scale{255.0F / (upper - lower)};
    float invScale{(upper - lower) / 255.0F};

    std::vector<std::uint16_t> qq(dim);
    float q1{0.0F};
    float p2{invScale * invScale};
    for (std::size_t i = 0; i < dim; ++i) {
        float x{query[i]};
        float dx{std::clamp(x, lower, upper) - lower};
        float dxs{scale * dx};
        float dxq{invScale * std::round(dxs)};
        //q1 += lower * (lower / 2.0F + dxq);
        q1 += lower * (x - lower / 2.0F) + (dx - dxq) * dxq;
        qq[i] = static_cast<std::uint16_t>(std::round(dxs));
    }

    for (std::size_t i = 0, id = 0; i < docs.size(); i += dim, ++id) {
        std::uint32_t simi{0};
        // TODO handcraft a vectorised version of this loop.
        #pragma clang loop unroll_count(8) vectorize(assume_safety)
        for (std::size_t j = 0; j < dim; ++j) {
            simi += qq[j] * static_cast<std::uint16_t>(docs[i + j]);
        }
        float sim{p1[id] + q1 + p2 * static_cast<float>(simi)};
        float dist{1.0F - sim};
        if (topk.size() < k) {
            topk.push(std::make_pair(dist, id));
        } else if (topk.top().first > dist) {
            topk.pop();
            topk.push(std::make_pair(dist, id));
        }
    }
}

// THESE 4 BIT IMPLEMENTATIONS DO NOT PACK THE DOC VECTORS.

std::pair<std::vector<std::uint8_t>, std::vector<float>>
scalarQuantise4B(const std::pair<float, float>& range,
                 std::size_t dim,
                 const std::vector<float>& dequantised) {

    auto [lower, upper] = range;
    std:size_t numDocs{dequantised.size() / dim};
    float scale{15.0F / (upper - lower)};
    float invScale{(upper - lower) / 15.0F};

    std::vector<float> p1(numDocs, 0.0F);
    std::vector<std::uint8_t> quantised(dequantised.size());

    for (std::size_t i = 0, k = 0; i < dequantised.size(); i += dim, ++k) {        
        #pragma clang unroll_count(2) vectorise(assume_safety)
        for (std::size_t j = i; j < i + dim; ++j) {
            float x{dequantised[j]};
            float dx{std::clamp(x, lower, upper) - lower};
            float dxs{scale * dx};
            float dxq{invScale * std::round(dxs)};
            //p1[k] += lower * (lower / 2.0F + dxq);
            p1[k] += lower * (x - lower / 2.0F) + (dx - dxq) * dxq;
            quantised[j] = static_cast<std::uint8_t>(std::round(dxs));
        }
    }
    return {std::move(quantised), std::move(p1)};
}

std::vector<float> scalarDequantise4B(const std::pair<float, float>& range,
                                      std::size_t dim,
                                      const std::vector<std::uint8_t>& quantised) {
    constexpr std::uint8_t MASK{0xF};
    auto [lower, upper] = range;
    std::vector<float> dequantised(quantised.size());
    float scale{(upper - lower) / 15.0F};
    #pragma clang unroll_count(4) vectorise(assume_safety)
    for (std::size_t i = 0; i < quantised.size(); ++i) {
        dequantised[i] = lower + scale * static_cast<float>(quantised[i]);
    }
    return dequantised;
}

void searchScalarQuantise4B(std::size_t k,
                            const std::pair<float, float>& range,
                            const std::vector<std::uint8_t>& docs,
                            std::vector<float> p1,
                            const std::vector<float>& query,
                            std::priority_queue<std::pair<float, std::size_t>>& topk) {

    auto [lower, upper] = range;
    std::size_t dim{query.size()};
    float scale{15.0F / (upper - lower)};
    float invScale{(upper - lower) / 15.0F};

    std::vector<std::uint8_t> qq(dim);
    float q1{0.0F};
    float p2{invScale * invScale};
    for (std::size_t i = 0; i < dim; ++i) {
        float x{query[i]};
        float dx{std::clamp(x, lower, upper) - lower};
        float dxs{scale * dx};
        float dxq{invScale * std::round(dxs)};
        //q1 += lower * (lower / 2.0F + dxq);
        q1 += lower * (x - lower / 2.0F) + (dx - dxq) * dxq;
        qq[i] = static_cast<std::uint8_t>(std::round(dxs));
    }

    for (std::size_t i = 0, id = 0; i < docs.size(); i += dim, ++id) {
        std::uint32_t simi{dot4B(dim, &qq[0], &docs[i])};
        float sim{p1[id] + q1 + p2 * static_cast<float>(simi)};
        float dist{1.0F - sim};
        if (topk.size() < k) {
            topk.push(std::make_pair(dist, id));
        } else if (topk.top().first > dist) {
            topk.pop();
            topk.push(std::make_pair(dist, id));
        }
    }
}

void runScalarBenchmark(const std::string& tag,
                        Metric metric,
                        ScalarBits bits,
                        std::size_t k,
                        std::size_t dim,
                        std::vector<float>& docs,
                        std::vector<float>& queries) {

    if (metric == Cosine) {
        normalise(dim, docs);
        normalise(dim, queries);
    }

    auto docsNorms2 = norms2(dim, docs);

    std::chrono::duration<double> diff{0};
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;

    std::vector<float> query(dim);
    std::priority_queue<std::pair<float, std::size_t>> topk;

    std::size_t numQueries{queries.size() / dim};
    std::size_t numDocs{docs.size() / dim};
    std::cout << std::setprecision(5)
              << "query count = " << numQueries
              << ", doc count = " << numDocs
              << ", dimension = " << dim << std::endl;
    std::cout << std::boolalpha
              << "metric = " << toString(metric)
              << ", quantisation = " << toString(bits)
              << ", top-k = " << k
              << ", normalise = " << (metric == Cosine) << std::endl;

    std::vector<std::vector<std::size_t>> nnExact(numQueries, std::vector<std::size_t>(k));
    for (std::size_t i = 0; i < queries.size(); i += dim) {
        std::copy(queries.begin() + i, queries.begin() + i + dim, query.begin());

        start = std::chrono::high_resolution_clock::now();
        searchBruteForce(k, docs, query, topk);
        end = std::chrono::high_resolution_clock::now();
        diff += std::chrono::duration<double>(end - start);

        for (std::size_t j = 1; j <= k && !topk.empty(); ++j) {
            nnExact[i / dim][k - j] = topk.top().second;
            topk.pop();
        }
    }
    std::cout << "Brute force took " << diff.count() << "s" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    auto range = quantiles(dim, docs, 0.999);
    auto [quantisedDocs, p1] = [&] {
        switch (bits) {
        case Scalar4Bit:
            return scalarQuantise4B(range, dim, docs);
        case Scalar8Bit:
            return scalarQuantise8B(range, dim, docs);
        }
    }();
    end = std::chrono::high_resolution_clock::now();
    diff = std::chrono::duration<double>(end - start);
    std::cout << "(lower, upper) = " << range << std::endl;
    std::cout << "Quantisation took " << diff.count() << "s" << std::endl;

    std::vector<std::vector<std::size_t>> nnSQ(
        numQueries, std::vector<std::size_t>(k + 40, numDocs + 1));

    for (std::size_t a = 0; a <= 40; a += 10) {
        diff = std::chrono::duration<double>{0};
        for (std::size_t i = 0; i < queries.size(); i += dim) {
            std::copy(queries.begin() + i, queries.begin() + i + dim, query.begin());

            start = std::chrono::high_resolution_clock::now();
            switch (bits) {
            case Scalar4Bit:
                searchScalarQuantise4B(k + a, range, quantisedDocs, p1, query, topk);
                break;
            case Scalar8Bit:
                searchScalarQuantise8B(k + a, range, quantisedDocs, p1, query, topk);
                break;
            }
            end = std::chrono::high_resolution_clock::now();
            diff += std::chrono::duration<double>(end - start);

            for (std::size_t j = 1; j <= k + a && !topk.empty(); ++j) {
                nnSQ[i / dim][k + a - j] = topk.top().second;
                topk.pop();
            }
        }

        auto recalls = computeRecalls(nnExact, nnSQ);
        std::cout << "Quantised search took " << diff.count() << "s, "
                  << "average recall@" << k << "|" << k + a << " = "
                  << recalls[PQStats::AVG_RECALL] << std::endl;
    }
}

// WIP

std::pair<std::vector<std::uint8_t>, std::vector<float>>
scalarQuantise4BMc(std::size_t dim,
                   const std::vector<float>& dequantised,
                   const std::vector<std::tuple<std::size_t, float, float>>& channels) {
 
    std::vector<std::uint8_t> quantised((dequantised.size() + 1) / 2);
 
    for (std::size_t i = 0; i < quantised.size(); ++i) {
        for (const auto& channel : channels) {
            auto [width, lower, upper] = channel;
            float scale{15.0F / (upper - lower)};
            for (std::size_t j = 0; j < width; ++i, ++j) {
                float r1{std::clamp(dequantised[2 * i], lower, upper) - lower};
                float r2{std::clamp(dequantised[2 * i + 1], lower, upper) - lower};
                quantised[i] = 
                    (static_cast<std::uint8_t>(std::round(scale * r1)) << 4) +
                    (static_cast<std::uint8_t>(std::round(scale * r2)));
            }
        }
    }

    // TODO
    return {std::move(quantised), std::vector<float>()};
}

std::vector<float>
scalarDequantise4BMc(std::size_t dim,
                     const std::vector<std::uint8_t>& quantised,
                     const std::vector<std::tuple<std::size_t, float, float>>& channels) {
    std::vector<float> dequantised(2 * quantised.size());
    #pragma clang unroll_count(8) vectorise(assume_safety)
    for (std::size_t i = 0; i < quantised.size(); ++i) {
        for (const auto& channel : channels) {
            auto [width, lower, upper] = channel;
            for (std::size_t j = 0; j < width; ++i, ++j) {
                dequantised[2 * i] =
                    lower + (upper - lower) * static_cast<float>(quantised[i] >> 4) / 15.0F;
                dequantised[2 * i + 1] =
                    lower + (upper - lower) * static_cast<float>(quantised[i] & 0xF) / 15.0F;
            }
        }
    }
    return dequantised;
}
