#include "utils.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>

std::ostream& operator<<(std::ostream& os, const OnlineMeanAndVariance& moments) {
    os << "(n = " << moments.n() << ", mean = " << moments.mean() << ", var = " << moments.var() << ")";
    return os;
}

std::ostream& operator<<(std::ostream& os, const Interval& interval) {
    os << "[" << interval.min() << ", " << interval.max() << "]";
    return os;
}

float dotAllowAlias(std::size_t dim, const float* x, const float* y) {
    float result(0.0F);
    #pragma omp simd reduction(+:result)
    for (std::size_t i = 0; i < dim; ++i) {
        result += x[i] * y[i];
    }
    return result;
}

namespace {
int dotAllowAlias(std::size_t dim, const int* x, const int* y) {
    int result(0);
    #pragma omp simd reduction(+:result)
    for (std::size_t i = 0; i < dim; ++i) {
        result += x[i] * y[i];
    }
    return result;
}
}

void matrixVectorMultiply(std::size_t dim, const float* m, const float* x, float* y) {
    for (std::size_t i = 0; i < dim; ++i) {
        y[i] = dotAllowAlias(dim, m + i * dim, x);
    }
}

void normalize(std::size_t dim, std::vector<float>& vectors) {
    for (std::size_t i = 0; i < vectors.size(); i += dim) {
        float norm(0.0F);
        #pragma omp simd reduction(+:norm)
        for (std::size_t j = 0; j < dim; ++j) {
            norm += vectors[i + j] * vectors[i + j];
        }
        norm = std::sqrtf(norm);
        #pragma omp simd
        for (std::size_t j = 0; j < dim; ++j) {
            vectors[i + j] /= norm;
        }
    }
}

std::vector<float> componentMeans(std::size_t dim, const std::vector<float>& x) {
    std::vector<OnlineMeanAndVariance> moments(dim);
    for (std::size_t i = 0; i < x.size(); i += dim) {
        for (std::size_t j = 0; j < dim; ++j) {
            moments[j].add(x[i + j]);
        }
    }
    std::vector<float> means(dim);
    for (std::size_t i = 0; i < dim; ++i) {
        means[i] = moments[i].mean();
    }
    return std::move(means);
}

std::vector<float> center(std::size_t dim,
                          std::vector<float> x,
                          const std::vector<float>& means) {
    for (std::size_t i = 0; i < x.size(); i += dim) {
        #pragma omp simd
        for (std::size_t j = 0; j < dim; ++j) {
            x[i + j] -= means[j];
        }
    }
    return std::move(x);
}

void quantize(std::size_t dim,
              const float* x,
              const Interval& limits,
              std::size_t bits,
              int* quantized) {
    float nSteps((1 << bits) - 1);
    float a(limits.min());
    float b(limits.max());
    float step((b - a) / nSteps);
    float stepInv(1.0F / step);
    for (std::size_t i = 0; i < dim; ++i) {
        float xi(std::clamp(x[i], a, b));
        quantized[i] = static_cast<int>(std::round((xi - a) * stepInv));
    }
}

std::vector<int> quantize(std::size_t dim,
                          const std::vector<float>& x,
                          const std::vector<Interval>& limits,
                          std::size_t bits) {
    float nSteps((1 << bits) - 1);
    std::vector<int> quantized(x.size());
    for (std::size_t i = 0; i < x.size(); i += dim) {
        float a(limits[i / dim].min());
        float b(limits[i / dim].max());
        float step((b - a) / nSteps);
        float stepInv(1.0F / step);
        for (std::size_t j = 0; j < dim; ++j) {
            float xi = std::clamp(x[i + j], a, b);
            quantized[i + j] = static_cast<int>(std::round((xi - a) * stepInv));
        }
    }
    return quantized;
}

std::vector<float> dequantize(std::size_t dim,
                              const std::vector<int>& xq,
                              const std::vector<Interval>& limits,
                              std::size_t bits) {
    float nSteps((1 << bits) - 1);
    std::vector<float> dequantized(xq.size());
    for (std::size_t i = 0; i < xq.size(); i += dim) {
        float a(limits[i / dim].min());
        float b(limits[i / dim].max());
        float step((b - a) / nSteps);
        for (std::size_t j = 0; j < dim; ++j) {
            dequantized[i + j] = a + step * static_cast<float>(xq[i + j]);
        }
    }
    return dequantized;
}

double quantizationMSE(std::size_t dim,
                       const float* x,
                       const int* xq,
                       const Interval &limits,
                       std::size_t bits) {
    double mse{0.0};
    float nSteps((1 << bits) - 1);
    float a(limits.min());
    float b(limits.max());
    float step((b - a) / nSteps);
    for (std::size_t i = 0; i < dim; ++i) {
    float xqi{a + step * static_cast<float>(xq[i])};
        mse += (x[i] - xqi) * (x[i] - xqi);
    }
    return mse;
}

std::vector<int> l1Norms(std::size_t dim, const std::vector<int>& x) {
    std::vector<int> norms(x.size() / dim);
    for (std::size_t i = 0; i < norms.size(); ++i) {
        norms[i] = std::accumulate(x.data() + i * dim,
                                   x.data() + (i + 1) * dim, int(0));
    }
    return norms;
}

std::vector<float> l2Norms(std::size_t dim,
                           const std::vector<float>& x) {
    std::vector<float> norms(x.size() / dim, 0.0F);
    for (std::size_t i = 0; i < norms.size(); ++i) {
        float norm(0.0F);
        #pragma omp simd reduction(+:norm)
        for (std::size_t j = 0; j < dim; ++j) {
            norms[i] += x[i * dim + j] * x[i * dim + j];
        }
        norms[i] = std::sqrtf(norm);
    }
    return norms;
}

std::vector<float> estimateDot(std::size_t dim,
                               const std::vector<int>& x,
                               const std::vector<int>& y,
                               const std::vector<int>& xL1,
                               const std::vector<int>& yL1,
                               const std::vector<Interval>& xlimits,
                               const std::vector<Interval>& ylimits,
                               std::size_t xbits,
                               std::size_t ybits) {
    float xscale(1.0F / static_cast<float>((1 << xbits) - 1));
    float yscale(1.0F / static_cast<float>((1 << ybits) - 1));
    std::vector<float> dots;
    dots.reserve((x.size() / dim) * (y.size() / dim));
    for (std::size_t i = 0; i < x.size(); i += dim) {
        auto vi = i / dim;
        float ax(xlimits[vi].min());
        float lx(xscale * xlimits[vi].length());
        float x1(xL1[vi]);
        for (std::size_t j = 0; j < y.size(); j += dim) {
            auto vj = j / dim;
            float ay(ylimits[vj].min());
            float ly(yscale * ylimits[vj].length());
            float dot_(ax * ay * static_cast<float>(dim) +
                       ay * lx * static_cast<float>(x1) +
                       ax * ly * static_cast<float>(yL1[vj]) +
                       lx * ly * static_cast<float>(dotAllowAlias(dim, x.data() + i, y.data() + j)));
            dots.push_back(dot_);
        }
    }
    return dots;
}

void modifiedGramSchmidt(std::size_t dim, std::vector<double>& m) {
    for (std::size_t i = 0; i < dim; ++i) {
        double norm(0.0);
        auto* mi = m.data() + i * dim;
        #pragma omp simd reduction(+:norm)
        for (std::size_t j = 0; j < dim; ++j) {
            norm += mi[j] * mi[j];
        }
        norm = std::sqrtf(norm);
        if (norm == 0.0) {
            continue;
        }
        #pragma omp simd
        for (std::size_t j = 0; j < dim; ++j) {
            mi[j] /= norm;
        }
        for (std::size_t k = i + 1; k < dim; ++k) {
            double dotik(0.0);
            auto* mk = m.data() + k * dim;
            #pragma omp simd reduction(+:dotik)
            for (std::size_t j = 0; j < dim; ++j) {
                dotik += mi[j] * mk[j];
            }
            #pragma omp simd
            for (std::size_t j = 0; j < dim; ++j) {
                mk[j] -= dotik * mi[j];
            }
        }
    }
}

std::pair<std::vector<std::vector<float>>, std::vector<std::size_t>>
randomOrthogonal(std::size_t dim, std::size_t blockDim) {
    blockDim = std::min(dim, blockDim);
    std::size_t nblocks{dim / blockDim};
    std::size_t rem{dim % blockDim};

    std::vector<std::vector<float>> blocks(nblocks + (rem > 0 ? 1 : 0));
    std::vector<std::size_t> dimBlocks(nblocks + (rem > 0 ? 1 : 0));

    std::mt19937 gen(215873873);
    std::normal_distribution<double> norm(0.0, 1.0);

    std::vector<double> m(blockDim * blockDim);
    for (std::size_t i = 0; i < nblocks; ++i) {
        std::generate_n(m.begin(), blockDim * blockDim, [&norm, &gen] { return norm(gen); });
        modifiedGramSchmidt(blockDim, m);
        blocks[i] = std::move(std::vector<float>(m.begin(), m.end()));
        dimBlocks[i] = blockDim;
    }
    if (rem == 0) {
        return {std::move(blocks), std::move(dimBlocks)};
    }

    m.resize(rem * rem);
    std::generate_n(m.begin(), rem * rem, [&norm, &gen] { return norm(gen); });
    modifiedGramSchmidt(rem, m);
    blocks[nblocks] = std::move(std::vector<float>(m.begin(), m.end()));
    dimBlocks[nblocks] = rem;

    return {std::move(blocks), std::move(dimBlocks)};
}
