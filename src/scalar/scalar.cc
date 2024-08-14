#include "scalar.h"

#include "utils.h"
#include "../common/bruteforce.h"
#include "../common/evaluation.h"
#include "../common/utils.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <ios>
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <tuple>
#include <utility>

namespace {

class OnlineMeanAndVariance {
public:
    OnlineMeanAndVariance() : n_{0}, mean_{0.0F}, sse_{0.0F} {}

    void add(float x) {
        ++n_;
        double delta{x - mean_};
        mean_ += delta / static_cast<double>(n_);
        sse_ += delta * (x - mean_);
    }

    double mean() const {
        return mean_;
    }

    double variance() const {
        return sse_ / (n_ - 1);
    }

private:
    std::size_t n_{0};
    double mean_{0.0};
    double sse_{0.0};
};

#if defined(__ARM_NEON__)

// Vectorized quantised dot product implementations.

#include <arm_neon.h>

// Implements dot product for the first 64 * floor(dim / 64) components
// of a vector.
std::uint32_t dot8B32(std::size_t dim,
                      const std::uint8_t*__restrict x,
                      const std::uint8_t*__restrict y) {

    // We have contension in the instruction pipeline on the accumulation
    // registers if we use too few.
    uint32x4_t xysum00{vdupq_n_u32(0)};
    uint32x4_t xysum16{vdupq_n_u32(0)};
    uint32x4_t xysum32{vdupq_n_u32(0)};
    uint32x4_t xysum48{vdupq_n_u32(0)};

    // Avoid overflow handling instructions which come with unsigned int.
    // Some unrolling gives around 4% performance improvement.
    #pragma clang loop unroll_count(2)
    for (int i = 0; i < static_cast<int>(dim); i += 32) {
        // Read into 16 x 8 bit vectors.
        uint8x16_t xbl{vld1q_u8(x + i)};
        uint8x16_t xbh{vld1q_u8(x + i + 16)};
        uint8x16_t ybl{vld1q_u8(y + i)};
        uint8x16_t ybh{vld1q_u8(y + i + 16)};
        // Multiply.
        uint16x8_t xyb00{vmull_u8(vget_low_u8(xbl),  vget_low_u8(ybl))};
        uint16x8_t xyb16{vmull_u8(vget_high_u8(xbl), vget_high_u8(ybl))};
        uint16x8_t xyb32{vmull_u8(vget_low_u8(xbh),  vget_low_u8(ybh))};
        uint16x8_t xyb48{vmull_u8(vget_high_u8(xbh), vget_high_u8(ybh))};
        // Accumulate 4 x 32 bit vectors (adding adjacent 16 bit lanes).
        xysum00 = vpadalq_u16(xysum00, xyb00);
        xysum16 = vpadalq_u16(xysum16, xyb16);
        xysum32 = vpadalq_u16(xysum32, xyb32);
        xysum48 = vpadalq_u16(xysum48, xyb48);
    }

    return vaddvq_u32(vaddq_u32(vaddq_u32(xysum00, xysum16),
                                vaddq_u32(xysum32, xysum48)));
}

// Implements dot product for the first 64 * floor(dim / 64) components
// of a vector. If dim > 4096 the vector must be blocked.
std::uint32_t dot4B64(std::size_t dim,
                      const std::uint8_t*__restrict x,
                      const std::uint8_t*__restrict y) {
    // This special case assumes that the vector dimension is:
    //   1. A multiple of 32
    //   2. Less than 256 * 16 = 4096 (or sums have the risk of overflowing).
    //
    // Both these cases will commonly hold. We call this as a subroutine for
    // the general implementation.

    uint16x8_t xysum00{vdupq_n_u16(0)};
    uint16x8_t xysum16{vdupq_n_u16(0)};
    uint16x8_t xysum32{vdupq_n_u16(0)};
    uint16x8_t xysum48{vdupq_n_u16(0)};

    // Some unrolling gives around 4% performance improvement.
    #pragma clang loop unroll_count(2)
    for (int i = 0; i < static_cast<int>(dim); i += 64) {
        // Read into 16 x 8 bit vectors.
        uint8x16_t xb00{vld1q_u8(x + i)};
        uint8x16_t xb16{vld1q_u8(x + i + 16)};
        uint8x16_t xb32{vld1q_u8(x + i + 32)};
        uint8x16_t xb48{vld1q_u8(x + i + 48)};
        uint8x16_t yb00{vld1q_u8(y + i)};
        uint8x16_t yb16{vld1q_u8(y + i + 16)};
        uint8x16_t yb32{vld1q_u8(y + i + 32)};
        uint8x16_t yb48{vld1q_u8(y + i + 48)};
        // Multiply.
        uint8x16_t xyb00{vmulq_u8(xb00, yb00)};
        uint8x16_t xyb16{vmulq_u8(xb16, yb16)};
        uint8x16_t xyb32{vmulq_u8(xb32, yb32)};
        uint8x16_t xyb48{vmulq_u8(xb48, yb48)};
        // Accumulate 8 x 32 bit vectors (adding adjacent 16 bit lanes).
        xysum00 = vpadalq_u8(xysum00, xyb00);
        xysum16 = vpadalq_u8(xysum16, xyb16);
        xysum32 = vpadalq_u8(xysum32, xyb32);
        xysum48 = vpadalq_u8(xysum48, xyb48);
    }

    return vaddlvq_u16(vaddq_u16(xysum00, xysum16)) +
           vaddlvq_u16(vaddq_u16(xysum32, xysum48));
}

// Implements dot product for the first 64 * floor(dim / 64) components
// of a vector. If dim > 4096 the vector must be blocked. Requires that
// the y vector has been packed.
std::uint32_t dot4BP64(std::size_t dim,
                       const std::uint8_t*__restrict x,
                       const std::uint8_t*__restrict y) {

    // We assume that the values for y are packed into the upper and lower
    // 4 bits of each element of the vector and that they are reordered so
    // we can unpack them efficiently. In particular, we assume for each
    // block of 16 integers we store the 0-15 elements in the lower bits and
    // the 16-31 elements in the upper bits.

    uint16x8_t xysum00{vdupq_n_u16(0)};
    uint16x8_t xysum16{vdupq_n_u16(0)};
    uint16x8_t xysum32{vdupq_n_u16(0)};
    uint16x8_t xysum48{vdupq_n_u16(0)};
    uint8x16_t m{vdupq_n_u8(0xF)};

    for (int i = 0; i < static_cast<int>(dim); i += 64) {
        // Read into 16 x 8 bit vector.
        uint8x16_t xb00{vld1q_u8(x + i)};
        uint8x16_t xb16{vld1q_u8(x + i + 16)};
        uint8x16_t xb32{vld1q_u8(x + i + 32)};
        uint8x16_t xb48{vld1q_u8(x + i + 48)};
        uint8x16_t ybpl{vld1q_u8(y + (i >> 1))};
        uint8x16_t ybph{vld1q_u8(y + (i >> 1) + 16)};
        // Unpack (lanewise mask and shift) into 16 x 8 bit vectors.
        uint8x16_t yb00{vandq_u8(ybpl, m)};
        uint8x16_t yb16{vshrq_n_u8(ybpl, 4)};
        uint8x16_t yb32{vandq_u8(ybph, m)};
        uint8x16_t yb48{vshrq_n_u8(ybph, 4)};
        // Multiply.
        uint8x16_t xyb00{vmulq_u8(xb00, yb00)};
        uint8x16_t xyb16{vmulq_u8(xb16, yb16)};
        uint8x16_t xyb32{vmulq_u8(xb32, yb32)};
        uint8x16_t xyb48{vmulq_u8(xb48, yb48)};
        // Accumulate 8 x 16 bit vectors (adding adjacent 8 bit lanes).
        xysum00 = vpadalq_u8(xysum00, xyb00);
        xysum16 = vpadalq_u8(xysum16, xyb16);
        xysum32 = vpadalq_u8(xysum32, xyb32);
        xysum48 = vpadalq_u8(xysum48, xyb48);
    }

    return vaddlvq_u16(vaddq_u16(xysum00, xysum16)) +
           vaddlvq_u16(vaddq_u16(xysum32, xysum48));
}

#else

// Fallback dot product implementation.

std::uint32_t dot8B32(std::size_t dim,
                      const std::uint8_t*__restrict x,
                      const std::uint8_t*__restrict y) {
    // Tell the compiler dim contraints.
    dim = dim & ~0x1F;
    std::uint32_t xy{0};
    #pragma clang loop unroll_count(8) vectorize(assume_safety)
    for (std::size_t i = 0; i < dim; ++i) {
        xy += static_cast<std::uint16_t>(x[i]) * static_cast<std::uint16_t>(y[i]);
    }
    return xy;
}

std::uint32_t dot4B64(std::size_t dim,
                      const std::uint8_t*__restrict x,
                      const std::uint8_t*__restrict y) {
    // Tell the compiler dim contraints.
    dim = std::min(dim, 4096UL) & ~0x3F;
    std::uint32_t xy{0};
    #pragma clang loop unroll_count(16) vectorize(assume_safety)
    for (std::size_t i = 0; i < dim; ++i) {
        xy += x[i] * y[i];
    }
    return xy;
}

std::uint32_t dot4BP64(std::size_t dim,
                       const std::uint8_t*__restrict x,
                       const std::uint8_t*__restrict y) {
    // Tell the compiler dim contraints.
    dim = std::min(dim, 4096UL) & ~0x3F;
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
    dim = dim & 0x1F;
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
    dim = dim & 0x3F;
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
    dim = dim & 0x3F;
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
    dim = dim & 0x3F;
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

} // unnamed::

std::uint32_t dot8B(std::size_t dim,
                    const std::uint8_t*__restrict x,
                    const std::uint8_t*__restrict y) {
    std::size_t rem{dim & 0x1F};
    dim -= rem;
    return dot8B32(dim, x, y) + (rem > 0 ? dot8BR(rem, x + dim, y + dim) : 0);
}

std::uint32_t dot4B(std::size_t dim,
                    const std::uint8_t*__restrict x,
                    const std::uint8_t*__restrict y) {
    std::size_t rem{dim & 0x3F};
    dim -= rem;
    std::uint32_t xy{0};
    for (std::size_t i = 0; i < dim; i += 4096) {
        xy += dot4B64(std::min(dim - i, 4096UL), x + i, y + i);
    }
    if (rem > 0) {
        xy += dot4BR(rem, x + dim, y + dim);
    }
    return xy;
}

std::uint32_t dot4BP(std::size_t dim,
                     const std::uint8_t*__restrict x,
                     const std::uint8_t*__restrict y) {
    std::size_t rem{dim & 0x3F};
    dim -= rem;
    std::uint32_t xy{0};
    for (std::size_t i = 0; i < dim; i += 4096) {
        xy += dot4BP64(std::min(dim - i, 4096UL), x + i, y + (i >> 1));
    }
    if (rem > 0) {
        xy += dot4BPR(rem, x + dim, y + (dim >> 1));
    }
    return xy;
}

std::uint32_t dot1B(std::size_t dim,
                    const std::uint32_t*__restrict x,
                    const std::uint32_t*__restrict y) {
    std::uint32_t xy{0};
    #pragma clang loop vectorize(assume_safety)
    for (std::size_t i = 0; i < dim; ++i) {
        std::uint32_t xandy{x[i] & y[i]};
        xy += __builtin_popcount(xandy);
    }
    return xy;
}

void pack4B(std::size_t dim,
            const std::uint8_t*__restrict raw,
            std::uint8_t*__restrict packed) {
    if (dim == 64) {
        pack4B32(raw, packed);
        pack4B32(raw + 32, packed + 16);
    } else {
        pack4BR(dim, raw, packed);
    }
}

void unpack4B(std::size_t dim,
              const std::uint8_t*__restrict packed,
              std::uint8_t*__restrict raw) {
    if (dim == 64) {
        unpack4B32(packed, raw);
        unpack4B32(packed + 16, raw + 32);
    } else {
        unpack4BR(dim, packed, raw);
    }
}

std::pair<float, float>
quantiles(std::size_t dim, const std::vector<float>& docs, float ci) {

    std:size_t numDocs{docs.size() / dim};

    std::minstd_rand rng;
    double p{std::min(25000.0 / static_cast<double>(numDocs), 1.0)};
    auto sampled = sampleDocs(dim, docs, p, rng);

    auto lower = static_cast<std::size_t>(
        std::round(static_cast<double>(sampled.size()) * 0.5 * (1.0 - ci)));
    auto upper = static_cast<std::size_t>(
        std::round(static_cast<double>(sampled.size()) * 0.5 * (1.0 + ci)));

    std::nth_element(sampled.begin(), sampled.begin() + lower, sampled.end());
    std::nth_element(sampled.begin() + lower + 1, sampled.begin() + upper, sampled.end());

    return {sampled[lower], sampled[upper]};
}

std::pair<float, float>
computeQuantisationInterval(std::size_t dim,
                            const std::vector<float>& docs,
                            const std::pair<float, float>& lowerRange,
                            const std::pair<float, float>& upperRange,
                            quantised_nearest_neighbours_t quantisedNearestNeighbours) {

    // We use the following calibration procedure:
    //   1. Randomly sample 1000 documents
    //   2. For each document find the closest 10 documents in the sample set.
    //   3. Perform blackbox maximization of the coefficient of determination
    //      for the quantized vs the float dot products.

    std::size_t numDocs{docs.size() / dim};
    std::size_t numSamples{std::min(numDocs, 1000UL)};
    std::size_t numNeighbours{std::min(numDocs - 1, 10UL)};

    // Sample the documents using random shuffle of the document indices.
    std::vector<float> sampledDocs(numSamples * dim);
    if (numSamples < numDocs) {
        std::vector<std::size_t> docIds(numDocs);
        std::iota(docIds.begin(), docIds.end(), 0);
        std::minstd_rand rng;
        std::shuffle(docIds.begin(), docIds.end(), rng);
        docIds.resize(numSamples);
        std::sort(docIds.begin(), docIds.end());
        for (std::size_t i = 0; i < numSamples; ++i) {
            std::copy_n(docs.begin() + docIds[i] * dim, dim, sampledDocs.begin() + i * dim);
        }
    } else {
        std::copy(docs.begin(), docs.end(), sampledDocs.begin());
    }

    // Find the nearest neighbours for each sampled document.
    std::vector<std::vector<std::size_t>> neighbours(numSamples);
    {
        std::vector<float> query(dim);
        for (std::size_t i = 0; i < sampledDocs.size(); i += dim) {
            std::copy_n(sampledDocs.begin() + i, dim, query.begin());
            std::size_t doc{i / dim};
            neighbours[doc] = searchBruteForce(numNeighbours, sampledDocs, query).first;
            std::sort(neighbours[doc].begin(), neighbours[doc].end());
        }
    }

    // Perform a black box optimization of the quantization interval maximizing
    // the coefficient of determination between the quantised and float dot
    // product (computed for the top-10 nearest matches).
    std::vector<float> query(dim);
    std::vector<float> queryNeighbours(numNeighbours * dim);
    auto correlation = [&] (float l, float u) {
        OnlineMeanAndVariance cod;
        for (std::size_t k = 0; k < sampledDocs.size(); k += dim) {
            std::copy_n(sampledDocs.begin() + k, dim, query.begin());

            std::size_t doc{k / dim};
            std::size_t queryNumNeighbours{neighbours[doc].size()};
            queryNeighbours.resize(queryNumNeighbours * dim);
            auto queryNeighbour = queryNeighbours.begin();
            for (auto n : neighbours[doc]) {
                std::copy_n(sampledDocs.begin() + n * dim, dim, queryNeighbour);
                queryNeighbour += dim;
            }
            std::priority_queue<std::pair<float, std::size_t>> topk;
            quantisedNearestNeighbours(dim, queryNeighbours, query, {l, u}, topk);

            OnlineMeanAndVariance scores;
            OnlineMeanAndVariance errors;
            while (!topk.empty()) {
                auto [oneMinusXyq, id] = topk.top();
                topk.pop();
                float xy{dotf(dim, &query[0], &queryNeighbours[id * dim])};
                scores.add(xy);
                errors.add((1.0F - oneMinusXyq) - xy);
            }
            cod.add(1.0 - errors.variance() / scores.variance());
        }
        return cod.mean();
    };

    auto [lmin, lmax] = lowerRange;
    auto [umin, umax] = upperRange;
    auto [lCoarse, uCoarse, corrMaxCoarse] = maximize(correlation, 32, {lmin, lmax}, {umin, umax});
    std::cout << "Coarse max correlation: " << corrMaxCoarse 
              << ", interval = [" << lCoarse << ", " << uCoarse << "]" << std::endl;
    std::tie(lmin, lmax) = std::make_pair(lCoarse - (lmax - lmin) / 12.0F, lCoarse + (lmax - lmin) / 12.0F);
    std::tie(umin, umax) = std::make_pair(uCoarse - (umax - umin) / 12.0F, uCoarse + (umax - umin) / 12.0F);
    auto [lFine, uFine, corrMaxFine] = maximize(correlation, 16, {lmin, lmax}, {umin, umax});
    std::cout << "Fine max correlation: " << corrMaxFine
              << ", interval = [" << lFine << ", " << uFine << "]" << std::endl;

    return corrMaxFine < corrMaxCoarse ?
           std::make_pair(lFine, uFine) : std::make_pair(lCoarse, uCoarse);
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
            float dx{x - lower};
            float dxc{std::clamp(x, lower, upper) - lower};
            float dxs{scale * dxc};
            float dxq{invScale * std::round(dxs)};
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
        float dx{x - lower};
        float dxc{std::clamp(x, lower, upper) - lower};
        float dxs{scale * dxc};
        float dxq{invScale * std::round(dxs)};
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
    std::array<std::uint8_t, 64> block;
    for (std::size_t id = 0; id < numDocs; ++id, xq += dimq) {
         for (std::size_t j = 0, k = 0; j < dim; xd += k) {
            for (k = 0; j < dim && k < 64; ++j, ++k) {
                float x{xd[k]};
                float dx{x - lower};
                float dxc{std::clamp(x, lower, upper) - lower};
                float dxs{scale * dxc};
                float dxq{invScale * std::round(dxs)};
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
    std::array<std::uint8_t, 64> block;
    for (auto xd = dequantised.begin(); xd != dequantised.end(); /**/) {
        if (packed) {
            for (std::size_t i = 0; i < dim; i += 64) {
                std::size_t j{std::min(dim - i, 64UL)};
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
        float dx{x - lower};
        float dxc{std::clamp(x, lower, upper) - lower};
        float dxs{scale * dxc};
        float dxq{invScale * std::round(dxs)};
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

std::pair<std::vector<std::uint8_t>, std::vector<float>>
scalarQuantise1B(const std::pair<float, float>& bucketCentres,
                 std::size_t dim,
                 const std::vector<float>& dequantised) {

    auto [lower, upper] = bucketCentres;
    std::size_t dimq{4 * ((dim + 31) / 32)};
    std:size_t numDocs{dequantised.size() / dim};
    float scale{1.0F / (upper - lower)};
    float invScale{upper - lower};

    std::vector<float> p1(numDocs, 0.0F);
    std::vector<std::uint8_t> quantised(numDocs * dimq, 0);
    auto* xq32 = reinterpret_cast<std::uint32_t*>(&quantised[0]);
    for (std::size_t i = 0, id = 0; i < dequantised.size(); i += dim, ++id) {
        for (std::size_t j = i; j < i + dim; ++xq32) {
            std::uint32_t xq{0};
            for (std::size_t k = 0; j < i + dim && k < 32; ++j, ++k) {
                float x{dequantised[j]};
                float dx{x - lower};
                float dxc{std::clamp(x, lower, upper) - lower};
                float dxs{scale * dxc};
                float dxq{invScale * std::round(dxs)};
                p1[id] += lower * (x - lower / 2.0F) + (dx - dxq) * dxq;
                xq |= static_cast<std::uint32_t>(std::round(dxs)) << k;
            }
            *xq32 = xq;
        }
    }

    return {std::move(quantised), std::move(p1)};
}

std::vector<float> scalarDequantise1B(const std::pair<float, float>& bucketCentres,
                                      std::size_t dim,
                                      const std::vector<std::uint8_t>& quantised) {

    auto [lower, upper] = bucketCentres;
    std::size_t dimq{4 * (dim + 31) / 32};
    std::size_t numDocs{quantised.size() / dimq};
    float scale{upper - lower};

    std::vector<float> dequantised(dim * numDocs);

    const auto* xq32 = reinterpret_cast<const std::uint32_t*>(&quantised[0]);
    for (auto xd = dequantised.begin(); xd != dequantised.end(); /**/) {
        for (std::size_t i = 0; i < dim; i += 32, ++xq32) {
            std::size_t j{std::min(dim - i, 32UL)};
            for (std::size_t k = 0; k < j; ++k, ++xd) {
                *xd = lower + scale * static_cast<float>((*xq32 >> k) & 0x1);
            }
        }
    }

    return dequantised;
}

void searchScalarQuantise1B(std::size_t k,
                            const std::pair<float, float>& bucketCentres,
                            const std::vector<std::uint8_t>& docs,
                            const std::vector<float>& p1,
                            const std::vector<float>& query,
                            std::priority_queue<std::pair<float, std::size_t>>& topk) {

    auto [lower, upper] = bucketCentres;
    std::size_t dim{query.size()};
    std::size_t dimq{4 * ((dim + 31) / 32)};
    float scale{1.0F / (upper - lower)};
    float invScale{upper - lower};

    std::vector<std::uint8_t> qq(dimq, 0);
    float q1{0.0F};
    float p2{invScale * invScale};
    {
        auto* qq32 = reinterpret_cast<std::uint32_t*>(&qq[0]);
        for (std::size_t i = 0; i < dim; ++qq32) {
            std::uint32_t xq{0};
            for (std::size_t j = 0; i < dim && j < 32; ++i, ++j) {
                float x{query[i]};
                float dx{x - lower};
                float dxc{std::clamp(x, lower, upper) - lower};
                float dxs{scale * dxc};
                float dxq{invScale * std::round(dxs)};
                q1 += lower * (x - lower / 2.0F) + (dx - dxq) * dxq;
                xq |= static_cast<std::uint32_t>(std::round(dxs)) << j;
            }
            *qq32 = xq;
        }
    }

    for (std::size_t i = 0, id = 0; i < docs.size(); i += dimq, ++id) {
        auto qq32 = reinterpret_cast<const std::uint32_t*>(&qq[0]);
        auto dq32 = reinterpret_cast<const std::uint32_t*>(&docs[i]);
        std::uint32_t simi{dot1B(dim, qq32, dq32)};
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

namespace {

std::pair<std::pair<float, float>, std::pair<float, float>>
candidateQuantizationIntervals(std::size_t dim,
                               const std::vector<float>& docs,
                               ScalarBits bits) {
    switch (bits) {
    case B1: {
        auto [al, bu] = quantiles(dim, docs, 0.8);
        auto [au, bl] = quantiles(dim, docs, 0.2);
        return {{al, au}, {bl, bu}};
    }
    case B4:
    case B4P:
    case B8: {
        float c{1.0F / static_cast<float>(dim + 1)};
        auto [al, bu] = quantiles(dim, docs, 1.0F - 1.0F / static_cast<float>(dim + 1));
        auto [au, bl] = quantiles(dim, docs, 1.0F - std::min(32UL, dim / 10) / static_cast<float>(dim + 1));
        return {{al, au}, {bl, bu}};
    }
    }
}

void nearestNeighbours1B(std::size_t dim,
                         const std::vector<float>& docs,
                         const std::vector<float>& query,
                         const std::pair<float, float>& range,
                         std::priority_queue<std::pair<float, std::size_t>>& topk) {
    auto [nq, pn] = scalarQuantise1B(range, dim, docs);
    searchScalarQuantise1B(docs.size() / dim, range, nq, pn, query, topk);
}

void nearestNeighbours4B(std::size_t dim,
                         const std::vector<float>& docs,
                         const std::vector<float>& query,
                         const std::pair<float, float>& range,
                         std::priority_queue<std::pair<float, std::size_t>>& topk) {
    auto [nq, pn] = scalarQuantise4B(range, false, dim, docs);
    searchScalarQuantise4B(docs.size() / dim, range, false, nq, pn, query, topk);
}

void nearestNeighbours8B(std::size_t dim,
                         const std::vector<float>& docs,
                         const std::vector<float>& query,
                         const std::pair<float, float>& range,
                         std::priority_queue<std::pair<float, std::size_t>>& topk) {
    auto [nq, pn] = scalarQuantise8B(range, dim, docs);
    searchScalarQuantise8B(docs.size() / dim, range, nq, pn, query, topk);
}

} // unnamed::

void runScalarBenchmark(const std::string& tag,
                        Metric metric,
                        ScalarBits bits,
                        std::size_t k,
                        std::size_t dim,
                        std::vector<float>& docs,
                        std::vector<float>& queries) {

    if (metric == Cosine) {
        normalize(dim, docs);
        normalize(dim, queries);
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
              << ", normalize = " << (metric == Cosine) << std::endl;

    std::vector<std::vector<std::size_t>> nnExact(numQueries);
    for (std::size_t i = 0; i < queries.size(); i += dim) {
        std::copy(queries.begin() + i, queries.begin() + i + dim, query.begin());
        start = std::chrono::steady_clock::now();
        nnExact[i / dim] = searchBruteForce(k, docs, query).first;
        end = std::chrono::steady_clock::now();
        diff += std::chrono::duration<double>(end - start);
    }
    std::cout << "Brute force took " << diff.count() << " s" << std::endl;

    start = std::chrono::steady_clock::now();
    auto range = [&] {
        auto [l, u] = candidateQuantizationIntervals(dim, docs, bits);
        switch (bits) {
        case B1: 
            return computeQuantisationInterval(dim, docs, l, u, nearestNeighbours1B);
        case B4:
        case B4P:
            return computeQuantisationInterval(dim, docs, l, u, nearestNeighbours4B);
        case B8:
            return computeQuantisationInterval(dim, docs, l, u, nearestNeighbours8B);
        }
    }();
    auto [quantisedDocs, p1] = [&] {
        switch (bits) {
        case B1:
            return scalarQuantise1B(range, dim, docs);
        case B4:
            return scalarQuantise4B(range, false, dim, docs);
        case B4P:
            return scalarQuantise4B(range, true, dim, docs);
        case B8:
            return scalarQuantise8B(range, dim, docs);
        }
    }();
    end = std::chrono::steady_clock::now();
    diff = std::chrono::duration<double>(end - start);
    std::cout << "Quantisation took " << diff.count() << " s" << std::endl;

    int last{90};
    std::vector<std::vector<std::size_t>> nnSQ(
        numQueries, std::vector<std::size_t>(k + last, numDocs + 1));

    for (std::size_t a : {0, 10, 20, 30, 40, last}) {
        diff = std::chrono::duration<double>{0};
        for (std::size_t i = 0; i < queries.size(); i += dim) {
            std::copy(queries.begin() + i, queries.begin() + i + dim, query.begin());

            start = std::chrono::steady_clock::now();
            switch (bits) {
            case B1:
                searchScalarQuantise1B(k + a, range, quantisedDocs, p1, query, topk);
                break;
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
            end = std::chrono::steady_clock::now();
            diff += std::chrono::duration<double>(end - start);

            for (std::size_t j = 1; j <= k + a && !topk.empty(); ++j) {
                nnSQ[i / dim][k + a - j] = topk.top().second;
                topk.pop();
            }
        }

        auto recalls = computeRecalls(nnExact, nnSQ);
        std::cout << "Quantised search took " << diff.count() << " s, "
                  << "average recall@" << k << "|" << k + a << " = "
                  << recalls[AVG_RECALL] << std::endl;
    }
}
