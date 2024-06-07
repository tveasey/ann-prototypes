#pragma once

#include <boost/iostreams/device/mapped_file.hpp>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// A wrapper around a boost::iostream::memory_mapped_file which provides
// a random access const iterator over the vectors.
//
// Note this is not intended to be persistent and so the memory mapped file
// is removed in the destructor.
class BigVector {
public:
    // A reference to a vector in the memory mapped file.
    //
    // This is used as the value_type of the const_iterator.
    class VectorReference {
    public:
        VectorReference() = default;
        VectorReference(const float* data, std::size_t dim) : data_{data}, dim_{dim} {
        }

        const float& operator[](std::size_t i) const {
            return data_[i];
        }
        const float* data() const {
            return data_;
        }
        std::size_t dim() const {
            return dim_;
        }

    private:
        const float* data_{nullptr};
        std::size_t dim_{0};
    };

    // A random access const iterator over the vectors in the memory mapped file.
    class const_iterator {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = const float;
        using difference_type = std::ptrdiff_t;
        using reference = VectorReference;

        const_iterator() = default;
        const_iterator(const float* data, std::size_t dim)
            : data_{data}, dim_{dim} {
        }

        reference operator*() const {
            return {data_, dim_};
        }
        reference operator[](difference_type n) const {
            return {data_ + n * dim_, dim_};
        }

        const_iterator& operator++() {
            data_ += dim_;
            return *this;
        }
        const_iterator operator++(int) {
            const_iterator tmp{*this};
            ++*this;
            return tmp;
        }
        const_iterator& operator--() {
            data_ -= dim_;
            return *this;
        }
        const_iterator operator--(int) {
            const_iterator tmp{*this};
            --*this;
            return tmp;
        }

        const_iterator& operator+=(difference_type n) {
            data_ += n * dim_;
            return *this;
        }
        const_iterator& operator-=(difference_type n) {
            data_ -= n * dim_;
            return *this;
        }

        friend const_iterator operator+(const_iterator lhs, difference_type rhs) {
            lhs += rhs;
            return lhs;
        }
        friend const_iterator operator+(difference_type lhs, const_iterator rhs) {
            rhs += lhs;
            return rhs;
        }
        friend const_iterator operator-(const_iterator lhs, difference_type rhs) {
            lhs -= rhs;
            return lhs;
        }
        friend difference_type operator-(const_iterator lhs, const_iterator rhs) {
            return (lhs.data_ - rhs.data_) / lhs.dim_;
        }

        friend bool operator==(const_iterator lhs, const_iterator rhs) {
            return lhs.data_ == rhs.data_;
        }
        friend bool operator!=(const_iterator lhs, const_iterator rhs) {
            return !(lhs == rhs);
        }
        friend bool operator<(const_iterator lhs, const_iterator rhs) {
            return lhs.data_ < rhs.data_;
        }
        friend bool operator>(const_iterator lhs, const_iterator rhs) {
            return rhs < lhs;
        }
        friend bool operator<=(const_iterator lhs, const_iterator rhs) {
            return !(rhs < lhs);
        }
        friend bool operator>=(const_iterator lhs, const_iterator rhs) {
            return !(lhs < rhs);
        }
        friend void swap(const_iterator& lhs, const_iterator& rhs) {
            std::swap(lhs.data_, rhs.data_);
            std::swap(lhs.dim_, rhs.dim_);
        }

    private:
        const float* data_{nullptr};
        std::size_t dim_{0};
    };

    using TGenerator = std::function<float ()>;
    using TPrepare = std::function<std::size_t(std::size_t, std::vector<float>&)>;

public:
    // Create with a generator function.
    BigVector(std::size_t dim,
              std::size_t numVectors,
              const std::filesystem::path& storage,
              const TGenerator& generator);
    // Create a BigVector reading the vectors from from an numpy fvecs file.
    BigVector(const std::filesystem::path& fvecs,
              const std::filesystem::path& storage,
              TPrepare prepare = [](std::size_t dim, std::vector<float>&) { return dim; });

    ~BigVector();

    BigVector(const BigVector&) = delete;
    BigVector& operator=(const BigVector&) = delete;

    // Normalize the vectors in place.
    void normalize();
    // The dimension of the vectors.
    std::size_t dim() const {
        return dim_;
    }
    // The number of vectors.
    std::size_t numVectors() const {
        return numVectors_;
    }
    // The total number of floats in the memory mapped file.
    std::size_t size() const {
        return dim_ * numVectors_;
    }
    // Get the i'th document.
    VectorReference operator[](std::size_t i) const {
        return {data_ + i * dim_, dim_};
    }
    // An iterator over the vectors of a memory mapped file.
    const_iterator begin() const {
        return {data_, dim_};
    }
    // The end iterator over the vectors of a memory mapped file.
    const_iterator end() const {
        return {data_ + dim_ * numVectors_, dim_};
    }

private:
    void create_memory_mapped_file(std::size_t dim,
                                   std::size_t numVectors,
                                   const std::filesystem::path& path);

    std::size_t dim_;
    std::size_t numVectors_;
    std::filesystem::path storage_;
    boost::iostreams::mapped_file file_;
    float* data_;
};

using Reader = std::function<void (std::size_t, BigVector::VectorReference)>;

// Merge two big vectors into a new big vector.
//
// Note there are clearly more efficient strategies but this gives us some
// sort of lower bound on the performance.
BigVector merge(const BigVector& a, const BigVector& b, std::filesystem::path storage);

// Read all the vectors in parallel.
//
// It arranges to read multiple disjoint regions of the memory mapped file
// concurrently to maximize IOPS.
void parallelRead(const BigVector& docs, std::vector<Reader>& readers);

// Read all the vectors in a specified range parallel.
//
// It arranges to read multiple disjoint regions of the memory mapped file
// concurrently to maximize IOPS.
void parallelRead(const BigVector& docs,
                  std::size_t begin,
                  std::size_t end,
                  std::vector<Reader>& readers);

// A simple reservoir sampler which stores the samples flat in a std::vector.
//
// This reads samples into a portion of a vector which is managed by the
// caller. The caller is responsible for ensuring that the vector is large
// enough to hold the samples.
class ReservoirSampler {
public:
    ReservoirSampler() = default;
    ReservoirSampler(std::size_t dim,
                     std::size_t sampleSize,
                     const std::minstd_rand& rng,
                     std::vector<float>::iterator beginSamples)
        : dim_{dim},
          sampleSize_{sampleSize},
          rng_{rng},
          beginSamples_{beginSamples} {
    }

    // Sample a document.
    void add(const float* doc) {
        if (dim_ == 0) {
            return;
        }
        ++numDocs_;
        if (numDocs_ < sampleSize_) {
            std::copy(doc, doc + dim_, beginSamples_ + (numDocs_ - 1) * dim_);
        } else {
            std::uniform_int_distribution<std::size_t> u0n{0, numDocs_ - 1};
            std::size_t pos{u0n(rng_)};
            if (pos < sampleSize_) {
                std::copy(doc, doc + dim_, beginSamples_ + pos * dim_);
            }
        }
    }

private:
    std::size_t dim_{0};
    std::size_t sampleSize_{0};
    std::size_t numDocs_{0};
    std::minstd_rand rng_;
    std::vector<float>::iterator beginSamples_;
};
