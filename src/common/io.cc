#include "io.h"
#include "progress_bar.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>

std::pair<std::vector<float>, std::size_t> readFvecs(const std::filesystem::path& file) {
    auto fileStr = file.u8string();
    auto* f = std::fopen(fileStr.c_str(), "r");
    if (f == nullptr) {
        std::cout << "Couldn't open " << file << std::endl;
        return {};
    }

    int dim;
    std::fread(&dim, 1, sizeof(int), f);
    std::fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    std::size_t sz = st.st_size;
    if (sz % ((dim + 1) * 4) != 0) {
        std::cout << "File must contain a whole number of vectors" << std::endl;
        return {};
    }

    std::size_t n{sz / ((dim + 1) * 4)};
    std::vector<float> vectors(n * (dim + 1));
    std::size_t nr{std::fread(vectors.data(), sizeof(float), n * (dim + 1), f)};
    if (nr != n * (dim + 1)) {
        std::cout << "Only read " << (nr / dim) << " out of "
                  << n << " vectors" << std::endl;
        return {};
    }

    // Shift to remove row headers.
    auto *x = vectors.data();
    for (std::size_t i = 0; i < n; i++) {
        std::memmove(x + i * dim, x + 1 + i * (dim + 1), dim * sizeof(*x));
    }
    vectors.resize(n * dim);

    std::fclose(f);

    return {std::move(vectors), static_cast<std::size_t>(dim)};
}

std::size_t writeFvecs(const std::filesystem::path& file,
                       int dim,
                       std::size_t numVecs,
                       TGenerator generator) {
    auto fileStr = file.u8string();
    auto* f = std::fopen(fileStr.c_str(), "w");
    if (f == nullptr) {
        std::cout << "Couldn't open " << file << std::endl;
    }

    std::size_t bytes{0};
    ProgressBar progress("Writing...", numVecs);
    for (std::size_t i = 0; i < numVecs; ++i) {
        auto vec = generator();
        bytes += std::fwrite(&dim, sizeof(int), 1, f);
        bytes += std::fwrite(vec.data(), sizeof(float), dim, f);
        progress.update();
    }

    std::fclose(f);

    return bytes;
}
