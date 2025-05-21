#pragma once

#include <ostream>
#include <utility>
#include <vector>

class OnlineMeanAndVariance {
public:
    OnlineMeanAndVariance() : n_(0), mean_(0), var_(0) {}

    unsigned int n() const { return n_; }
    double mean() const { return mean_; }
    double var() const { return var_ / n_; }

    void add(double x) {
        ++n_;
        double delta = x - mean_;
        mean_ += delta / static_cast<double>(n_);
        var_ += delta * (x - mean_);
    }

private:
    std::size_t n_;
    double mean_;
    double var_;
};

std::ostream& operator<<(std::ostream& os, const OnlineMeanAndVariance& moments);

class Interval {
public:
    Interval() : min_(0.0F), max_(0.0F) {};
    Interval(float min, float max) : min_(min), max_(max) {}

    float min() const { return min_; }
    float max() const { return max_; }
    float length() const { return max_ - min_; }
    bool closeTo(const Interval& rhs, float epsilon) const {
        return std::abs(min_ - rhs.min_) < epsilon && std::abs(max_ - rhs.max_) < epsilon;
    }

    Interval& truncate(float min, float max) {
        min_ = std::max(min_, min);
        max_ = std::min(max_, max);
        return *this;
    }
    Interval& operator+=(float rhs) {
        min_ += rhs;
        max_ += rhs;
        return *this;
    }
    Interval& operator*=(float scalar) {
        min_ *= scalar;
        max_ *= scalar;
        return *this;
    }

private:
    float min_;
    float max_;
};

inline Interval operator+(float lhs, Interval rhs) {
    rhs += lhs;
    return rhs;
}
inline Interval operator*(float rhs, Interval lhs) {
    lhs *= rhs;
    return lhs;
}

std::ostream& operator<<(std::ostream& os, const Interval& interval);

float dotAllowAlias(std::size_t dim, const float* x, const float* y);

void matrixVectorMultiply(std::size_t dim, const float* m, const float* x, float* y);

void normalize(std::size_t dim, std::vector<float>& vectors);

std::vector<float> componentMeans(std::size_t dim, const std::vector<float>& x);

std::vector<float> center(std::size_t dim,
                          std::vector<float> x,
                          const std::vector<float>& means);

void quantize(std::size_t dim,
              const float* x,
              const Interval& limits,
              std::size_t bits,
              int* quantized);

std::vector<int> quantize(std::size_t dim,
                          const std::vector<float>& x,
                          const std::vector<Interval>& limits,
                          std::size_t bits);

std::vector<float> dequantize(std::size_t dim,
                              const std::vector<int>& xq,
                              const std::vector<Interval>& limits,
                              std::size_t bits);

double quantizationMSE(std::size_t dim,
                       const float* x,
                       const int* xq,
                       const Interval& limits,
                       std::size_t bits);

std::vector<int> l1Norms(std::size_t dim, const std::vector<int>& x);

std::vector<float> l2Norms(std::size_t dim,
                           const std::vector<float>& limits);

std::vector<float> estimateDot(std::size_t dim,
                               const std::vector<int>& x,
                               const std::vector<int>& y,
                               const std::vector<int>& xL1,
                               const std::vector<int>& yL1,
                               const std::vector<Interval>& xlimits,
                               const std::vector<Interval>& ylimits,
                               std::size_t xbits,
                               std::size_t ybits);

void modifiedGramSchmidt(std::size_t dim, std::vector<double>& m);

std::pair<std::vector<std::vector<float>>, std::vector<std::size_t>>
randomOrthogonal(std::size_t dim, std::size_t blockDim = 64);