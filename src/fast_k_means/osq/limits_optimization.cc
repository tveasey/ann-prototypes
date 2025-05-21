#include "limits_optimization.h"

#include "utils.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {
std::pair<std::vector<float>, std::vector<float>>
vectorRange(std::size_t dim, std::vector<float> x) {
    std::vector<float> min(x.size() / dim, std::numeric_limits<float>::max());
    std::vector<float> max(x.size() / dim, std::numeric_limits<float>::lowest());
    for (std::size_t i = 0; i < x.size(); i += dim) {
        auto& min_ = min[i / dim];
        auto& max_ = max[i / dim];
        for (std::size_t j = 0; j < dim; ++j) {
            min_ = std::min(min_, x[i + j]);
            max_ = std::max(max_, x[i + j]);
        }
    }
    return {std::move(min), std::move(max)};
}

std::vector<OnlineMeanAndVariance>
vectorMoments(std::size_t dim, std::vector<float> x) {
    std::vector<OnlineMeanAndVariance> moments(x.size() / dim);
    for (std::size_t i = 0; i < x.size(); /**/) {
        auto& vectorMoments = moments[i / dim];
        for (std::size_t j = 0; j < dim; ++i, ++j) {
            vectorMoments.add(x[i]);
        }
    }
    return moments;
}

std::vector<Interval> MINIMUM_MSE_GRID{
    {-0.798, 0.798}, {-1.493, 1.493}, {-2.051, 2.051}, {-2.514, 2.514},
    {-2.916, 2.916}, {-3.278, 3.278}, {-3.611, 3.611}, {-3.922, 3.922}};

class OptimizeQuantizationLimits {
public:
    OptimizeQuantizationLimits(std::size_t bits,
                               float lambda = 0.1F,
                               std::size_t nIters = 5,
                               bool debug = false) :
        bits_(bits), lambda_(lambda), nIters_(nIters),
        nPoints_(static_cast<std::size_t>(std::pow(2, bits))) {
    }

    Interval compute(const Interval& initialInterval,
                     const float* x0, const float* xn) const {

        // We want to optimize anisotropic error.
        //
        // We proceed by an iterative algorithm which alternates between:
        //   1. Assigning components to their nearest grid point,
        //   2. Updating the grid points to minimize the error.

        std::size_t dim(xn - x0);
        float xx(0.0F);
        for (std::size_t i = 0; i < dim; ++i) {
            float xi(x0[i]);
            xx += xi * xi;
        }

        Interval interval(initialInterval);
        float error(quantizationError(interval, xx, x0, xn));
        for (std::size_t i = 0; i < nIters_; ++i) {
            Interval lastInterval(interval);
            interval = step(interval, xx, x0, xn);
            float lastError(error);
            error = quantizationError(interval, xx, x0, xn);
            if (interval.closeTo(lastInterval, 1e-8)) {
                break;
            }
            // Unlike k-means the error is not guaranteed to decrease at each step
            // because rounding to the nearest grid point is not the snapping which
            // minimizes anisotropic error. Once it starts increasing it nearly
            // always continues to do so. So we stop as soon as we see an increase.
            if (error > lastError) {
                interval = lastInterval;
                break;
            }
        }
        return interval;
    }

    Interval step(const Interval& interval,
                  float xx,
                  const float* x0, const float* xn) const {
        std::size_t dim(xn - x0);
        float a(interval.min());
        float b(interval.max());
        float scale((1.0F - lambda_) / xx);
        float stepInv((static_cast<float>(nPoints_) - 1.0F) / (b - a));

        // Compute the derivatives of the different loss function terms w.r.t.
        // the parameters a and b.
        double daa(0.0);
        double dab(0.0);
        double dbb(0.0);
        double dax(0.0);
        double dbx(0.0);
        for (std::size_t i = 0; i < dim; ++i) {
            float xi(x0[i]);
            auto k = std::round((std::clamp(xi, a, b) - a) * stepInv);
            auto s = k / static_cast<float>(nPoints_ - 1);
            daa += (1.0 - s) * (1.0 - s);
            dab += (1.0 - s) * s;
            dbb += s * s;
            dax += xi * (1.0 - s);
            dbx += xi * s;
        }

        double M[]{scale * dax * dax + lambda_ * daa,
                   scale * dax * dbx + lambda_ * dab,
                   scale * dbx * dbx + lambda_ * dbb};
        double c[]{dax, dbx};

        double det(M[0] * M[2] - M[1] * M[1]);

        if (det == 0.0) {
            return Interval(a, b);
        }

        float aOpt((M[2] * c[0] - M[1] * c[1]) / det);
        float bOpt((M[0] * c[1] - M[1] * c[0]) / det);
        return Interval(aOpt, bOpt);
    }

    float quantizationError(const Interval& interval,
                            float xx,
                            const float* x0, const float* xn) const {
        std::size_t dim(xn - x0);
        float a(interval.min());
        float b(interval.max());
        float step((b - a) / (static_cast<float>(nPoints_) - 1.0F));
        float stepInv(1.0F / step);
        float xe(0.0);
        float e(0.0);
        for (std::size_t i = 0; i < dim; ++i) {
            float xi(x0[i]);
            float xiq(a + step * std::round((std::clamp(xi, a, b) - a) * stepInv));
            xe += xi * (xi - xiq);
            e  += (xi - xiq) * (xi - xiq);
        }
        return (1.0F - lambda_) * xe * xe / xx + lambda_ * e;
    }

private:
    std::size_t bits_;
    float lambda_;
    std::size_t nIters_;
    std::size_t nPoints_;
};

// This does a more expensive optimization of the limits by trying to minimize the
// anisotropic error when snapping the vector to the grid. It does the snapping
// to grid points coordinate-wise, which won't necessarily minimize the error, but
// empirically the error nearly always decreases or stays the same per step. We
// can't enumerate candidate grid points because the number is exponential in the
// number of dimensions. It gives us a small improvement in nDCG@10 almost across
// the board using brute force.
class OptimizeQuantizationLimitsAndQuantizedVectors {
public:
    OptimizeQuantizationLimitsAndQuantizedVectors(std::size_t bits,
                                                  float lambda = 0.1F,
                                                  std::size_t nIters = 5,
                                                  bool debug = false) :
        bits_(bits), lambda_(lambda), nIters_(nIters),
        nPoints_(static_cast<std::size_t>(std::pow(2, bits))) {
    }

    const std::vector<int>& quantized() const {
        return quantized_;
    }

    Interval compute(const Interval& initialInterval,
                     const float* x0, const float* xn) {

        // We want to optimize anisotropic error.
        //
        // We proceed by an iterative algorithm which alternates between:
        //   1. Assigning components to their nearest grid point,
        //   2. Updating the grid points to minimize the error.

        std::size_t dim(xn - x0);

        initializeQuantized(initialInterval, x0, xn);
        float xx(0.0F);
        for (std::size_t i = 0; i < dim; ++i) {
            float xi(x0[i]);
            xx += xi * xi;
        }

        Interval interval(initialInterval);
        float error(quantizationError(interval, xx, x0, xn));
        for (std::size_t i = 0; i < nIters_; ++i) {
            Interval lastInterval(interval);
            interval = step(interval, xx, x0, xn);
            float lastError(error);
            error = quantizationError(interval, xx, x0, xn);
            if (interval.closeTo(lastInterval, 1e-8)) {
                break;
            }
            // This nearly always decreases or stays the same with the improved snapping
            // criterion, but we check just in case because our snapping is heuristic.
            if (error > lastError) {
                interval = lastInterval;
                break;
            }
        }
        return interval;
    }

    void initializeQuantized(const Interval& interval, const float* x0, const float* xn) {
        std::size_t dim(xn - x0);
        quantized_.resize(dim);
        float a(interval.min());
        float b(interval.max());
        float step((b - a) / (static_cast<float>(nPoints_) - 1.0F));
        float stepInv(1.0F / step);
        for (std::size_t i = 0; i < dim; ++i) {
            float xi(x0[i]);
            quantized_[i] = static_cast<int>(
                std::round((std::clamp(xi, a, b) - a) * stepInv));
        }
    }

    Interval step(const Interval& interval,
                  float xx,
                  const float* x0, const float* xn) {
        std::size_t dim(xn - x0);
        float a(interval.min());
        float b(interval.max());
        float scale((1.0F - lambda_) / xx);
        float step((b - a) / (static_cast<float>(nPoints_) - 1.0F));
        float stepInv(1.0F / step);

        float xe(0.0);
        float e(0.0);
        for (std::size_t i = 0; i < dim; ++i) {
            float xi(x0[i]);
            float xiq(a + step * std::round((std::clamp(xi, a, b) - a) * stepInv));
            xe += xi * (xi - xiq);
            e  += (xi - xiq) * (xi - xiq);
        }

        // Compute the derivatives of the different loss function terms w.r.t.
        // the parameters a and b.
        double daa(0.0);
        double dab(0.0);
        double dbb(0.0);
        double dax(0.0);
        double dbx(0.0);
        for (std::size_t i = 0; i < dim; ++i) {
            float xi(x0[i]);
            float xi_((std::clamp(xi, a, b) - a) * stepInv);
            float xiq(a + step * std::round(xi_));
            float xil(a + step * std::floor(xi_));
            float xiu(a + step * std::ceil(xi_));
            float xel(xe + xi * (xiq - xil));
            float xeu(xe + xi * (xiq - xiu));
            float el(e + (xi - xil) * (xi - xil) - (xi - xiq) * (xi - xiq));
            float eu(e + (xi - xiu) * (xi - xiu) - (xi - xiq) * (xi - xiq));
            float errorl((1.0F - lambda_) * xel * xel / xx + lambda_ * el);
            float erroru((1.0F - lambda_) * xeu * xeu / xx + lambda_ * eu);
            if (errorl < erroru) {
                quantized_[i] = static_cast<int>(std::floor(xi_));
                xe = xel;
                e = el;
            } else {
                quantized_[i] = static_cast<int>(std::ceil(xi_));
                xe = xeu;
                e = eu;
            }
            auto s = static_cast<float>(quantized_[i]) / static_cast<float>(nPoints_ - 1);
            daa += (1.0 - s) * (1.0 - s);
            dab += (1.0 - s) * s;
            dbb += s * s;
            dax += xi * (1.0 - s);
            dbx += xi * s;
        }

        double M[]{scale * dax * dax + lambda_ * daa,
                   scale * dax * dbx + lambda_ * dab,
                   scale * dbx * dbx + lambda_ * dbb};
        double c[]{dax, dbx};

        double det(M[0] * M[2] - M[1] * M[1]);

        if (det == 0.0) {
            return Interval(a, b);
        }

        float aOpt((M[2] * c[0] - M[1] * c[1]) / det);
        float bOpt((M[0] * c[1] - M[1] * c[0]) / det);
        return Interval(aOpt, bOpt);
    }

    float quantizationError(const Interval& interval,
                            float xx,
                            const float* x0, const float* xn) const {
        std::size_t dim(xn - x0);
        float a(interval.min());
        float b(interval.max());
        float step((b - a) / (static_cast<float>(nPoints_) - 1.0F));
        float stepInv(1.0F / step);
        float xe(0.0);
        float e(0.0);
        for (std::size_t i = 0; i < dim; ++i) {
            float xi(x0[i]);
            float xiq(a + step * static_cast<float>(quantized_[i]));
            xe += xi * (xi - xiq);
            e  += (xi - xiq) * (xi - xiq);
        }
        return (1.0F - lambda_) * xe * xe / xx + lambda_ * e;
    }

private:
    std::size_t bits_;
    float lambda_;
    std::size_t nIters_;
    std::size_t nPoints_;
    std::vector<int> quantized_;
};
}

std::vector<Interval>
initialQuantizationLimits(std::size_t dim, const std::vector<float>& x, std::size_t bits) {
    auto amin(*std::min_element(x.begin(), x.end()));
    auto bmax(*std::max_element(x.begin(), x.end()));
    auto moments = vectorMoments(dim, x);
    std::vector<Interval> limits(x.size() / dim);
    for (std::size_t i = 0; i < moments.size(); ++i) {
        float mean(moments[i].mean());
        float std(std::sqrtf(moments[i].var()));
        limits[i] = (mean + std * MINIMUM_MSE_GRID[bits - 1]).truncate(amin, bmax);
    }
    return limits;
}

std::vector<Interval>
optimizeQuantizationLimits(std::size_t dim,
                          const std::vector<float>& x,
                          std::size_t bits,
                          float lambda) {
    auto limits = initialQuantizationLimits(dim, x, bits);
    OptimizeQuantizationLimits optimizer(bits, lambda);
    for (std::size_t i = 0; i < limits.size(); ++i) {
        limits[i] = optimizer.compute(limits[i], x.data() + i * dim, x.data() + (i + 1) * dim);
    }
    return limits;
}

std::pair<std::vector<Interval>, std::vector<int>>
optimizeQuantizationLimitsAndQuantizedVectors(std::size_t dim,
                                             const std::vector<float>& x,
                                             std::size_t bits,
                                             float lambda) {
    auto limits = initialQuantizationLimits(dim, x, bits);
    std::vector<int> quantized(x.size());
    OptimizeQuantizationLimitsAndQuantizedVectors optimizer(bits, lambda);
    for (std::size_t i = 0; i < limits.size(); ++i) {
        limits[i] = optimizer.compute(limits[i], x.data() + i * dim, x.data() + (i + 1) * dim);
        for (std::size_t j = 0; j < dim; ++j) {
            quantized[i * dim + j] = optimizer.quantized()[j];
        }
    }
    return {std::move(limits), std::move(quantized)};
}
