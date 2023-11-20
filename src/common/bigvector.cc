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
                     TPrepare prepare)
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

    // Dry run to estimate the padded dimension.
    std::vector<float> doc(dim_, 0.0F);
    std::size_t preparedDim{prepare(dim_, doc)};

    this->create_memory_mapped_file(preparedDim, numVectors_, storage);

    // Read the data into the memory mapped file in 1MB chunks and then drop
    // the dimension header from each row and write the data to the memory
    // mapped file.
    std::size_t chunkSize{(1024UL * 1024UL / (4 * (dim_ + 1)))};
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

        // Shift to remove row headers and zero pad to a multiple of the
        // number of books.
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

void parallelRead(const BigVector& docs, std::vector<Reader>& readers) {

    std::size_t numReaders{readers.size()};
    std::size_t blocksize{(docs.numVectors() + numReaders - 1) / numReaders};

    // Compared to the cost of reading a large number of vectors from disk
    // the cost of creating the threads is negligible so we don't bother
    // reusing them.
    std::vector<std::thread> threads;
    threads.reserve(numReaders);
    for (std::size_t i = 0; i < numReaders; ++i) {
        threads.emplace_back([i, blocksize, &readers, &docs]() {
            std::size_t blockBegin{i * blocksize};
            std::size_t blockEnd{std::min((i + 1) * blocksize, docs.numVectors())};
            auto endDocs = docs.begin() + blockEnd;
            std::size_t id{blockBegin};
            for (auto doc = docs.begin() + blockBegin; doc != endDocs; ++id, ++doc) {
                readers[i](id, *doc);
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }
}
