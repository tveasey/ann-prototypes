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
#include <utility>

// A wrapper around a boost::iostream::memory_mapped_file which provides
// a random access const iterator over the vectors.
class BigVector {
public:
    // A reference to a vector in the memory mapped file.
    //
    // This is used as the value_type of the const_iterator.
    class VectorReference {
    public:
        VectorReference() = default;
        VectorReference(const float* data, std::size_t dim)
            : data_{data},
              dim_{dim} {
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
    using TPad = std::function<void(std::size_t, std::vector<float>&)>;

public:
    // Create a BigVector with the given dimensions and number of vectors.
    BigVector(std::size_t dim,
              std::size_t numVectors,
              const std::filesystem::path& storage);
    // Create with a generator function.
    BigVector(std::size_t dim,
              std::size_t numVectors,
              const std::filesystem::path& storage,
              const TGenerator& generator);
    // Create a BigVector reading the vectors from from an numpy fvecs file.
    BigVector(const std::filesystem::path& fvecs,
              const std::filesystem::path& storage,
              TPad zeroPad = [](std::size_t, std::vector<float>&) {});

    // Note that the file is automatically closed when the memory_mapped_file
    // object is destroyed.

    BigVector(const BigVector&) = delete;
    BigVector& operator=(const BigVector&) = delete;

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
    boost::iostreams::mapped_file file_;
    float* data_;
};

// A simple reservoir sampler which stores the samples flat in a std::vector.
class ReservoirSampler {
public:
    ReservoirSampler(std::size_t dim, std::size_t sampleSize, std::minstd_rand& rng)
        : dim_{dim},
          sampleSize_{sampleSize},
          rng_{rng} {
        sample_.resize(sampleSize_ * dim_);
    }

    // Sample a document.
    void add(const float* doc) {
        ++numDocs_;
        if (sample_.size() < sampleSize_) {
            std::copy(doc, doc + dim_, sample_.begin() + (numDocs_ - 1) * dim_);
        } else {
            std::uniform_int_distribution<std::size_t> u0n{0, numDocs_ - 1};
            std::size_t pos{u0n(rng_)};
            if (pos < sampleSize_) {
                std::copy(doc, doc + dim_, sample_.begin() + pos * dim_);
            }
        }
    }

    std::vector<float>& sample() { return sample_; }

private:
    std::size_t dim_;
    std::size_t sampleSize_;
    std::minstd_rand& rng_;
    std::size_t numDocs_{0};
    std::vector<float> sample_;
};
