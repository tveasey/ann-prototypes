#include "bigvector.h"

#include <boost/iostreams/device/mapped_file.hpp>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>


BigVector::BigVector(std::size_t dim,
                     std::size_t numVectors,
                     const std::filesystem::path& storage)
    : dim_{dim},
      numVectors_{numVectors} {
    this->create_memory_mapped_file(dim, numVectors, storage);
}

BigVector::BigVector(std::size_t dim,
                     std::size_t numVectors,
                     const std::filesystem::path& storage,
                     const TGenerator& generator)
    : dim_{dim},
      numVectors_{numVectors} {
    this->create_memory_mapped_file(dim, numVectors, storage);
    for (std::size_t i = 0; i < numVectors_; ++i) {
        for (std::size_t j = 0; j < dim_; ++j) {
            data_[i * dim_ + j] = generator();
        }
    }
}

BigVector::BigVector(const std::filesystem::path& fvecs,
                     const std::filesystem::path& storage,
                     TPad zeroPad) {

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

    create_memory_mapped_file(dim_, numVectors_, storage);

    // Read the data into the memory mapped file in 1MB chunks and then drop
    // the dimension header from each row and write the data to the memory
    // mapped file.
    std::size_t chunkSize{(1024UL * 1024UL / (4 * (dim_ + 1)))};
    std::vector<float> chunk(chunkSize * (dim_ + 1));
    std::size_t numChunks{numVectors_ / chunkSize};
    for (std::size_t i = 0; i < numChunks; ++i) {
        std::size_t currentChunkSize{
            std::min(chunkSize, (numVectors_ - i * chunkSize))};

        chunk.resize(currentChunkSize * (dim_ + 1));
        std::size_t floatsRead{std::fread(
            chunk.data(), sizeof(float), currentChunkSize * (dim_ + 1), source)};
        if (floatsRead != currentChunkSize * (dim_ + 1)) {
            throw std::runtime_error(
                "Only read " + std::to_string(floatsRead) + " out of " +
                std::to_string(currentChunkSize * (dim_ + 1)) + " floats");
        }

        // Shift to remove row headers and zero pad to a multiple of the
        // number of books.
        for (std::size_t i = 0; i < currentChunkSize; i++) {
            std::memmove(chunk.data() + i * dim_,
                         chunk.data() + 1 + i * (dim_ + 1),
                         dim_ * sizeof(float));
        }
        chunk.resize(currentChunkSize * dim_);
        zeroPad(dim_, chunk);

        // Write the chunk to the memory mapped file.
        std::memcpy(data_ + i * chunkSize * dim_,
                    chunk.data(),
                    chunkSize * dim_ * sizeof(float));
    }

    std::fclose(source);
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
