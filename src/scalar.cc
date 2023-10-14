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

// Vectorised dot product implementation.

#include <arm_neon.h>

std::uint32_t dot8B16(std::size_t dim,
                      const std::uint8_t*__restrict x,
                      const std::uint8_t*__restrict y) {

    uint32x4_t xysuml{vdupq_n_u32(0)};
    uint32x4_t xysumh{vdupq_n_u32(0)};

    for (std::size_t i = 0; i < dim; i += 16) {
        // Read into 16 x 8 bit vectors.
        uint8x16_t xb{vld1q_u8(x + i)};
        uint8x16_t yb{vld1q_u8(y + i)};
        // Multiply.
        uint16x8_t xybl{vmull_u8(vget_low_u8(xb), vget_low_u8(yb))};
        uint16x8_t xybh{vmull_u8(vget_high_u8(xb), vget_high_u8(yb))};
        // Accumulate 4 x 32 bit vectors (adding adjacent 16 bit lanes).
        xysuml = vpadalq_u16(xysuml, xybl);
        xysumh = vpadalq_u16(xysumh, xybh);
    }

    return vaddvq_u32(vaddq_u32(xysuml, xysumh));
}

// Implements dot product for the first 16 * floor(dim / 16) components
// of a vector. If dim > 4096 the vector must be blocked.
std::uint32_t dot4B32(std::size_t dim,
                      const std::uint8_t*__restrict x,
                      const std::uint8_t*__restrict y) {
    // This special case assumes that the vector dimension is:
    //   1. A multiple of 32
    //   2. Less than 256 * 16 = 4096 (or sums have the risk of overflowing).
    //
    // Both these cases will commonly hold. We call this as a subroutine for
    // the general implementation.

    uint16x8_t xysuml{vdupq_n_u16(0)};
    uint16x8_t xysumh{vdupq_n_u16(0)};

    for (std::size_t i = 0; i < dim; i += 32) {
        // Read into 16 x 8 bit vectors.
        uint8x16_t xbl{vld1q_u8(x + i)};
        uint8x16_t xbh{vld1q_u8(x + i + 16)};
        uint8x16_t ybl{vld1q_u8(y + i)};
        uint8x16_t ybh{vld1q_u8(y + i + 16)};
        // Multiply.
        uint8x16_t xybl{vmulq_u8(xbl, ybl)};
        uint8x16_t xybh{vmulq_u8(xbh, ybh)};
        // Accumulate 8 x 32 bit vectors (adding adjacent 16 bit lanes).
        xysuml = vpadalq_u8(xysuml, xybl);
        xysumh = vpadalq_u8(xysumh, xybh);
    }

    return vaddlvq_u16(xysuml) + vaddlvq_u16(xysumh);
}

// Implements dot product for the first 16 * floor(dim / 32) components
// of a vector. If dim > 4096 the vector must be blocked. Requires that
// the y vector has been packed.
std::uint32_t dot4BP32(std::size_t dim,
                       const std::uint8_t*__restrict x,
                       const std::uint8_t*__restrict y) {

    // We assume that the values for y are packed into the upper and lower
    // 4 bits of each element of the vector and that they are reordered so
    // we can unpack them efficiently. In particular, we assume for each
    // block of 16 integers we store the 0-15 elements in the lower bits and
    // the 16-31 elements in the upper bits.

    uint16x8_t xysuml{vdupq_n_u16(0)};
    uint16x8_t xysumh{vdupq_n_u16(0)};
    uint8x16_t m{vdupq_n_u8(0xF)};

    for (std::size_t i = 0; i < dim; i += 32) {
        // Read into 16 x 8 bit vector.
        uint8x16_t xbl{vld1q_u8(x + i)};
        // Read into 16 x 8 bit vector.
        uint8x16_t xbh{vld1q_u8(x + i + 16)};
        // Read into 16 x 8 bit vector.
        uint8x16_t ybp{vld1q_u8(y + (i >> 1))};
        // Mask to extract lower 4 bits in 8 x 16 bit vector.
        uint8x16_t ybl{vandq_u8(ybp, m)};
        // Right shift to extract upper 4 bits in 8 x 16 bit vector.
        uint8x16_t ybh{vshrq_n_u8(ybp, 4)};
        // Multiply.
        uint8x16_t xybl{vmulq_u8(xbl, ybl)};
        uint8x16_t xybh{vmulq_u8(xbh, ybh)};
        // Accumulate 8 x 16 bit vectors (adding adjacent 8 bit lanes).
        xysuml = vpadalq_u8(xysuml, xybl);
        xysumh = vpadalq_u8(xysumh, xybh);
    }

    return vaddlvq_u16(xysuml) + vaddlvq_u16(xysumh);
}

#else

// Fallback dot product implementation.

std::uint32_t dot8B16(std::size_t dim,
                      const std::uint8_t*__restrict x,
                      const std::uint8_t*__restrict y) {
    // Tell the compiler dim contraints.
    dim = dim & ~0xF;
    std::uint32_t xy{0};
    #pragma clang loop unroll_count(8) vectorize(assume_safety)
    for (std::size_t i = 0; i < dim; ++i) {
        xy += static_cast<std::uint16_t>(x[i]) * static_cast<std::uint16_t>(y[i]);
    }
    return xy;
}

std::uint32_t dot4B16(std::size_t dim,
                      const std::uint8_t*__restrict x,
                      const std::uint8_t*__restrict y) {
    // Tell the compiler dim contraints.
    dim = std::min(dim, 4096U) & ~0xF;
    std::uint32_t xy{0};
    #pragma clang loop unroll_count(16) vectorize(assume_safety)
    for (std::size_t i = 0; i < dim; ++i) {
        xy += x[i] * y[i];
    }
    return xy;
}

std::uint32_t dot4BP32(std::size_t dim,
                       const std::uint8_t*__restrict x,
                       const std::uint8_t*__restrict y) {
    // Tell the compiler dim contraints.
    dim = std::min(dim, 4096U) & ~0x1F;
    std::uint32_t xy{0};
    for (std::size_t i = 0; i < dim; i += 32) {
        #pragma clang loop unroll_count(8) vectorize(assume_safety)
        for (std::size_t j = 0; j < 16; ++j) {
            xy += x[i + j] * (y[(i >> 1) + j] & 0xF);
        }
        #pragma clang loop unroll_count(8) vectorize(assume_safety)
        for (std::size_t j = 0; j < 16; ++j) {
            xy += x[i + j + 16] * (y[(i >> 1) + j] >> 4);
        }
    }
    return xy;
}

#endif

// Remainder handling for dot product.

std::uint32_t dot8BR(std::size_t dim,
                     const std::uint8_t*__restrict x,
                     const std::uint8_t*__restrict y) {
    // Tell the compiler dim contraints.
    dim = dim & 0xF;
    std::uint32_t xy{0};
    #pragma clang loop vectorize(assume_safety)
    for (std::size_t i = 0; i < dim; ++i) {
        xy += static_cast<std::uint16_t>(x[i]) * static_cast<std::uint16_t>(y[i]);
    }
    return xy;
}

std::uint32_t dot4BR(std::size_t dim,
                     const std::uint8_t*__restrict x,
                     const std::uint8_t*__restrict y) {
    // Tell the compiler dim contraints.
    dim = dim & 0xF;
    std::uint32_t xy{0};
    #pragma clang loop vectorize(assume_safety)
    for (std::size_t i = 0; i < dim; ++i) {
        xy += x[i] * y[i];
    }
    return xy;
}

std::uint32_t dot4BPR(std::size_t dim,
                      const std::uint8_t*__restrict x,
                      const std::uint8_t*__restrict y) {
    // Tell the compiler dim contraints.
    dim = dim & 0x1F;
    std::size_t rem{dim & 0x1};
    dim -= rem;
    std::uint32_t xy{0};
    #pragma clang loop vectorize(assume_safety)
    for (std::size_t i = 0; i < dim; i += 2) {
        xy += x[i]     * (y[i >> 1] & 0xF);
        xy += x[i + 1] * (y[i >> 1] >> 4);
    }
    if (rem > 0) {
        xy += x[dim] * y[dim >> 1];
    }
    return xy;
}

void pack4B32(const std::uint8_t*__restrict block,
              std::uint8_t*__restrict packedBlock) {
    for (std::size_t i = 0; i < 16; ++i) {
        packedBlock[i] = (block[16 + i] << 4) + block[i];
    }
}

void pack4BR(std::size_t dim,
             const std::uint8_t*__restrict remainder,
             std::uint8_t*__restrict packedRemainder) {
    dim = dim & 0x1F;
    std::size_t rem{dim & 0x1};
    dim -= rem;
    for (std::size_t i = 0; i < dim; i += 2) {
        packedRemainder[i >> 1] = (remainder[i + 1] << 4) + remainder[i];
    }
    if (rem > 0) {
        packedRemainder[dim >> 1] = remainder[dim];
    }
}

void unpack4B32(const std::uint8_t*__restrict packedBlock,
                std::uint8_t*__restrict block) {
    for (std::size_t i = 0; i < 16; ++i) {
        block[i]      = packedBlock[i] & 0xF;
        block[i + 16] = packedBlock[i] >> 4;
    }
}

void unpack4BR(std::size_t dim,
               const std::uint8_t*__restrict packedRemainder,
               std::uint8_t*__restrict remainder) {
    dim = dim & 0xF;
    std::size_t rem{dim & 0x1};
    dim -= rem;
    for (std::size_t i = 0; i < dim; i += 2) {
        remainder[i]     = packedRemainder[i >> 1] & 0xF;
        remainder[i + 1] = packedRemainder[i >> 1] >> 4;
    }
    if (rem > 0) {
        remainder[dim] = rem * packedRemainder[dim >> 1];
    }
}
}

std::uint32_t dot8B(std::size_t dim,
                    const std::uint8_t*__restrict x,
                    const std::uint8_t*__restrict y) {
    std::size_t rem{dim & 0xF};
    dim -= rem;
    return dot8B16(dim, x, y) + (rem > 0 ? dot8BR(rem, x + dim, y + dim) : 0);
}

std::uint32_t dot4B(std::size_t dim,
                    const std::uint8_t*__restrict x,
                    const std::uint8_t*__restrict y) {
    std::size_t rem{dim & 0x1F};
    dim -= rem;
    std::uint32_t xy{0};
    for (std::size_t i = 0; i < dim; i += 4096) {
        xy += dot4B32(std::min(dim - i, 4096UL), x + i, y + i);
    }
    if (rem > 0) {
        xy += dot4BR(rem, x + dim, y + dim);
    }
    return xy;
}

std::uint32_t dot4BP(std::size_t dim,
                     const std::uint8_t*__restrict x,
                     const std::uint8_t*__restrict y) {
    std::size_t rem{dim & 0x1F};
    dim -= rem;
    std::uint32_t xy{0};
    for (std::size_t i = 0; i < dim; i += 4096) {
        xy += dot4BP32(std::min(dim - i, 4096UL), x + i, y + (i >> 1));
    }
    if (rem > 0) {
        xy += dot4BPR(rem, x + dim, y + (dim >> 1));
    }
    return xy;
}

void pack4B(std::size_t dim,
            const std::uint8_t*__restrict raw,
            std::uint8_t*__restrict packed) {
    if (dim == 32) {
        pack4B32(raw, packed);
    } else {
        pack4BR(dim, raw, packed);
    }
}

void unpack4B(std::size_t dim,
              const std::uint8_t*__restrict packed,
              std::uint8_t*__restrict raw) {
    if (dim == 32) {
        unpack4B32(packed, raw);
    } else {
        unpack4BR(dim, packed, raw);
    }
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

    for (std::size_t i = 0, id = 0; i < dequantised.size(); i += dim, ++id) {
        #pragma clang unroll_count(2) vectorize(assume_safety)
        for (std::size_t j = i; j < i + dim; ++j) {
            float x{dequantised[j]};
            float dx{std::clamp(x, lower, upper) - lower};
            float dxs{scale * dx};
            float dxq{invScale * std::round(dxs)};
            //p1[k] += lower * (lower / 2.0F + dxq);
            p1[id] += lower * (x - lower / 2.0F) + (dx - dxq) * dxq;
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
    #pragma clang unroll_count(8) vectorize(assume_safety)
    for (std::size_t i = 0; i < quantised.size(); ++i) {
        dequantised[i] = lower + scale * static_cast<float>(quantised[i]);
    }
    return dequantised;
}

namespace {
void searchScalarQuantise8B(std::size_t k,
                            const std::pair<float, float>& range,
                            const std::vector<std::uint8_t>& docs,
                            const std::vector<float>& p1,
                            const std::vector<float>& query,
                            std::priority_queue<std::pair<float, std::size_t>>& topk) {

    auto [lower, upper] = range;
    std::size_t dim{query.size()};
    float scale{255.0F / (upper - lower)};
    float invScale{(upper - lower) / 255.0F};

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
        std::uint32_t simi{dot8B(dim, &qq[0], &docs[i])};
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
}

std::pair<std::vector<std::uint8_t>, std::vector<float>>
scalarQuantise4B(const std::pair<float, float>& range,
                 bool pack,
                 std::size_t dim,
                 const std::vector<float>& dequantised) {

    auto [lower, upper] = range;
    std:size_t numDocs{dequantised.size() / dim};
    std::size_t dimq{pack ? (dim + 1) >> 1 : dim};
    float scale{15.0F / (upper - lower)};
    float invScale{(upper - lower) / 15.0F};

    std::vector<float> p1(numDocs, 0.0F);
    std::vector<std::uint8_t> quantised(dequantised.size());

    auto xd = dequantised.begin();
    auto xq = quantised.begin();
    std::array<std::uint8_t, 32> block;
    for (std::size_t id = 0; id < numDocs; ++id, xq += dimq) {
         for (std::size_t j = 0, k = 0; j < dim; xd += k) {
            for (k = 0; j < dim && k < 32; ++j, ++k) {
                float x{xd[k]};
                float dx{std::clamp(x, lower, upper) - lower};
                float dxs{scale * dx};
                float dxq{invScale * std::round(dxs)};
                //p1[k] += lower * (lower / 2.0F + dxq);
                p1[id] += lower * (x - lower / 2.0F) + (dx - dxq) * dxq;
                block[k] = static_cast<std::uint8_t>(std::round(dxs));
            }
            if (pack) {
                pack4B(k, &block[0], &xq[(j - k) >> 1]);
            } else {
                std::copy_n(&block[0], k, &xq[j - k]);
            }
        }
    }

    return {std::move(quantised), std::move(p1)};
}

std::vector<float> scalarDequantise4B(const std::pair<float, float>& range,
                                      bool packed,
                                      std::size_t dim,
                                      const std::vector<std::uint8_t>& quantised) {
    auto [lower, upper] = range;
    std::size_t dimq{packed ? (dim + 1) >> 1 : dim};
    float scale{(upper - lower) / 15.0F};

    std::vector<float> dequantised(quantised.size());

    const std::uint8_t* xq = &quantised[0];
    std::array<std::uint8_t, 32> block;
    for (auto xd = dequantised.begin(); xd != dequantised.end(); /**/) {
        if (packed) {
            for (std::size_t i = 0; i < dim; i += 32) {
                std::size_t j{std::min(dim - i, 32UL)};
                unpack4B(j, xq, &block[0]);
                for (std::size_t k = 0; k < j; ++k, ++xd) {
                    *xd = lower + scale * static_cast<float>(block[k]);
                }
                xq += (j + 1) >> 1;
            }
        } else {
            for (std::size_t i = 0; i < dim; ++i, ++xq, ++xd) {
                *xd = lower + scale * static_cast<float>(*xq);
            }
        }
    }

    return dequantised;
}

namespace {
void searchScalarQuantise4B(std::size_t k,
                            const std::pair<float, float>& range,
                            bool packed,
                            const std::vector<std::uint8_t>& docs,
                            const std::vector<float>& p1,
                            const std::vector<float>& query,
                            std::priority_queue<std::pair<float, std::size_t>>& topk) {

    auto [lower, upper] = range;
    std::size_t dim{query.size()};
    std::size_t dimq{packed ? (dim + 1) >> 1 : dim};
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

    for (std::size_t i = 0, id = 0; i < docs.size(); i += dimq, ++id) {
        std::uint32_t simi{packed ?
                           dot4BP(dim, &qq[0], &docs[i]) :
                           dot4B(dim, &qq[0], &docs[i])};
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
        case B4:
            return scalarQuantise4B(range, false, dim, docs);
        case B4P:
            return scalarQuantise4B(range, true, dim, docs);
        case B8:
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
            case B4:
                searchScalarQuantise4B(k + a, range, false, quantisedDocs, p1, query, topk);
                break;
            case B4P:
                searchScalarQuantise4B(k + a, range, true, quantisedDocs, p1, query, topk);
                break;
            case B8:
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
