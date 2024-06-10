#include "bigvector.h"

#include <boost/iostreams/device/mapped_file.hpp>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/_types/_seek_set.h>
#include <utility>

#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>


BigVector::BigVector(std::size_t dim,
                     std::size_t numVectors,
                     const std::filesystem::path& storage,
                     const TGenerator& generator)
    : dim_{dim},
      numVectors_{numVectors},
      storage_{storage} {
    this->create_memory_mapped_file(dim, numVectors, storage);
    for (std::size_t i = 0; i < numVectors_; ++i) {
        for (std::size_t j = 0; j < dim_; ++j) {
            data_[i * dim_ + j] = generator();
        }
    }
}

BigVector::BigVector(const std::filesystem::path& fvecs,
                     const std::filesystem::path& storage,
                     TPrepare prepare,
                     const std::pair<double, double>& range)
    : storage_{storage} {

    // Create from an numpy fvecs without ever copying the full data
    // into process memory.

    auto fvecsStr = fvecs.u8string();
    auto* source = std::fopen(fvecsStr.c_str(), "r");
    if (source == nullptr) {
        throw std::runtime_error("Couldn't open " + fvecsStr);
    }

    // Read the header to get the number of dimensions.
    {
        int dim;
        std::fread(&dim, 1, sizeof(int), source);
        std::fseek(source, 0, SEEK_SET);
        dim_ = dim;
    }

    // Use the file size to get the number of vectors. Importantly we assume
    // the vectors were saved as 32 bit floats.
    struct stat st;
    fstat(fileno(source), &st);
    auto bytes = st.st_size;
    if (bytes % ((dim_ + 1) * 4) != 0) {
        throw std::runtime_error("File must contain a whole number of vectors");
    }
    numVectors_ = bytes / ((dim_ + 1) * 4);

    std::size_t start{0};
    std::size_t end{numVectors_};
    if (range.first > 0.0 || range.second < 1.0) {
        auto [a, b] = range;
        a = std::clamp(a, 0.0, 1.0);
        b = std::clamp(b, a, 1.0);
        if (a > 0.0) {
            start = static_cast<std::size_t>(a * numVectors_);
        }
        if (b < 1.0) {
            end = static_cast<std::size_t>(b * numVectors_);
        }
        numVectors_ = end - start;
        if (std::fseek(source, start * (dim_ + 1) * sizeof(float), SEEK_CUR) != 0) {
            throw std::runtime_error("Failed to seek to " + std::to_string(start));
        }
    }

    // Dry run to estimate the padded dimension.
    std::vector<float> doc(dim_, 0.0F);
    std::size_t preparedDim{prepare(dim_, doc)};

    this->create_memory_mapped_file(preparedDim, numVectors_, storage);

    // Read the data in 1MB chunks then drop the dimension header from each
    // row and write the data to the memory mapped file.
    std::size_t chunkSize{(1024 * 1024 / (4 * (dim_ + 1)))};
    std::vector<float> chunk(chunkSize * (dim_ + 1));
    std::size_t numChunks{(numVectors_ + chunkSize - 1) / chunkSize};
    auto* writePos = data_;
    for (std::size_t i = 0; i < numChunks; ++i) {
        chunkSize = std::min(chunkSize, (numVectors_ - i * chunkSize));
        chunk.resize(chunkSize * (dim_ + 1));
        std::size_t floatsRead{std::fread(
            chunk.data(), sizeof(float), chunkSize * (dim_ + 1), source)};
        if (floatsRead != chunkSize * (dim_ + 1)) {
            throw std::runtime_error(
                "Only read " + std::to_string(floatsRead) + " out of " +
                std::to_string(chunkSize * (dim_ + 1)) + " floats");
        }

        // Shift to remove row headers.
        for (std::size_t i = 0; i < chunkSize; i++) {
            std::memmove(chunk.data() + i * dim_,
                         chunk.data() + 1 + i * (dim_ + 1),
                         dim_ * sizeof(float));
        }
        chunk.resize(chunkSize * dim_);

        // Any other preparation, such as zero padding or normalization.
        if (prepare(dim_, chunk) != preparedDim) {
            std::fclose(source);
            throw std::runtime_error("The vector dimension must be the same for all chunks");
        }

        // Write the chunk to the memory mapped file.
        std::memcpy(writePos, chunk.data(), chunkSize * preparedDim * sizeof(float));
        writePos += chunkSize * preparedDim;
    }

    dim_ = preparedDim;

    std::fclose(source);
}

BigVector::~BigVector() {
    file_.close();
    // Remove the file.
    if (std::filesystem::remove(storage_)) {
        std::cout << "Removed temporary file " << storage_ << std::endl;
    } else {
        std::cerr << "Failed to remove temporary file " << storage_ << std::endl;
    }
}

void BigVector::normalize() {
    for (std::size_t i = 0; i < numVectors_; ++i) {
        float* begin = data_ + i * dim_;
        float* end = begin + dim_;
        float norm{0.0F};
        #pragma omp simd reduction(+:norm)
        for (auto* x = begin; x != end; ++x) {
            norm += *x * *x;
        }
        norm = std::sqrtf(norm);
        #pragma omp simd
        for (auto* x = begin; x != end; ++x) {
            *x /= norm;
        }
    }
}

void BigVector::create_memory_mapped_file(std::size_t dim,
                                          std::size_t numVectors,
                                          const std::filesystem::path& storage) {
    // Create a file which is large enough to hold the vectors.
    std::size_t fileSize{dim * numVectors * sizeof(float)};
    boost::iostreams::mapped_file_params  params;
    params.path = storage.u8string();
    params.new_file_size = fileSize;
    params.mode = (std::ios_base::out | std::ios_base::in);
    file_.open(params);
    if (!file_.is_open()) {
        throw std::runtime_error("Failed to open file " + storage.u8string());
    }
    data_ = reinterpret_cast<float*>(file_.data());
}

namespace {

class Copier {
public:
    Copier(std::size_t offset, std::size_t dim, float* begin) :
        offset_{offset}, dim_{dim}, begin_{begin} {
    }

    void operator()(std::size_t id, BigVector::VectorReference doc) {
        std::copy(doc.data(), doc.data() + dim_, begin_ + (id - offset_) * dim_);
    }

private:
    std::size_t offset_;
    std::size_t dim_;
    float* begin_;
};

class YieldFrom {
private:
    constexpr static std::size_t BUFFER_VECTORS{128UL * 1024UL};

public:
    YieldFrom(std::vector<const BigVector*> source) :
        source_{std::move(source)},
        offset_{0}, pos_{0} {
        buffer_.resize(BUFFER_VECTORS * source_[0]->dim());
        curr_ = source_.data();
        this->fillBuffer();
    }

    float operator()() {
        if (pos_ == read_) {
            if (offset_ == (*curr_)->numVectors()) {
                ++curr_;
                offset_ = 0;
            }
            this->fillBuffer();
            pos_ = 0;
        }
        return buffer_[pos_++];
    }

private:
    void fillBuffer() {
        std::size_t dim{(*curr_)->dim()};
        std::size_t end{std::min((*curr_)->numVectors(), offset_ + BUFFER_VECTORS)};
        std::size_t numReaders{std::clamp((end - offset_) / 10, 1UL, 32UL)};
        std::vector<Reader> copiers(numReaders, Copier{offset_, dim, buffer_.data()});
        parallelRead(**curr_, offset_, end, copiers);
        read_ = (end - offset_) * dim;
        offset_ = end;
    }

private:
    std::vector<const BigVector*> source_;
    const BigVector** curr_{nullptr};
    std::vector<float> buffer_;
    std::size_t offset_{0};
    std::size_t pos_{0};
    std::size_t read_{0};
};

} // unnamed::

BigVector merge(const BigVector& a, const BigVector& b, std::filesystem::path storage) {
    if (a.dim() != b.dim()) {
        throw std::invalid_argument("The dimensions of the vectors must match");
    }
    return {a.dim(), a.numVectors() + b.numVectors(), storage, YieldFrom{{&a, &b}}};
}

void parallelRead(const BigVector& docs, std::vector<Reader>& readers) {
    parallelRead(docs, 0, docs.numVectors(), readers);
}

void parallelRead(const BigVector& docs,
                  std::size_t begin,
                  std::size_t end,
                  std::vector<Reader>& readers) {

    std::size_t numReaders{readers.size()};
    std::size_t blocksize{(end - begin + numReaders - 1) / numReaders};

    // Compared to the cost of reading a large number of vectors from disk
    // the cost of creating the threads is negligible so we don't bother
    // reusing them.
    std::vector<std::thread> threads;
    threads.reserve(numReaders);
    for (std::size_t i = 0; i < numReaders; ++i) {
        threads.emplace_back([begin, end, i, blocksize, &readers, &docs]() {
            std::size_t blockBegin{begin + i * blocksize};
            std::size_t blockEnd{std::min(begin + (i + 1) * blocksize, end)};
            auto beginDocs = docs.begin() + blockBegin;
            auto endDocs = docs.begin() + blockEnd;
            std::size_t id{blockBegin};
            for (auto doc = beginDocs; doc != endDocs; ++id, ++doc) {
                readers[i](id, *doc);
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }
}
