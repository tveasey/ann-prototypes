#include "../common/utils.h"
#include "baseline.h"
#include "common.h"
#include "hamerly.h"
#include "hierarchical.h"

#include <filesystem>
#include <iostream>
#include <optional>
#include <utility>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>

namespace {
enum class Method {
    KMEANS_LLOYD,
    KMEANS_HAMERLY,
    KMEANS_HIERARCHICAL
};

std::pair<std::vector<float>, std::size_t> readFvecs(const std::filesystem::path& file,
                                                     std::optional<int> dim_) {
    auto fileStr = file.u8string();
    auto* f = std::fopen(fileStr.c_str(), "r");
    if (f == nullptr) {
        std::cout << "Couldn't open " << file << std::endl;
        return {};
    }

    int dim;
    int dimPadded;
    if (dim_) {
        dim = dimPadded = *dim_;
    } else {
        std::fread(&dim, 1, sizeof(int), f);
        dimPadded = dim + 1;
        std::fseek(f, 0, SEEK_SET);
    }
        struct stat st;
        fstat(fileno(f), &st);
        std::size_t sz = st.st_size;
        if (sz % (dimPadded * 4) != 0) {
        std::cout << "File must contain a whole number of vectors" << std::endl;
        return {};
    }

    std::size_t n(sz / (dimPadded * 4));
    std::vector<float> vectors(n * dimPadded);
    std::size_t nr(std::fread(vectors.data(), sizeof(float), n * dimPadded, f));
    if (nr != n * dimPadded) {
        std::cout << "Only read " << (nr / dimPadded) << " out of "
        << n << " vectors" << std::endl;
        return {};
    }

    // Shift to remove row headers.
    if (dim != dimPadded) {
        auto *x = vectors.data();
        for (std::size_t i = 0; i < n; i++) {
            std::memmove(x + i * dim, x + 1 + i * dimPadded, dim * sizeof(*x));
        }
        vectors.resize(n * dim);
    }

    std::fclose(f);

    return {std::move(vectors), static_cast<std::size_t>(dim)};
}
}

int main(int argc, char** argv) {
    std::optional<int> dim_;
    std::vector<std::string> files;
    std::size_t k{100};
    Method method{Method::KMEANS_HIERARCHICAL};
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-h") {
            std::cout << "Usage: " << argv[0] << " [-d <dim>] [-k <clusters>] -f <file1> <file1> ..." << std::endl;
            return 0;
        } else if (std::string(argv[i]) == "-d") {
            dim_ = std::stoul(argv[++i]);
        } else if (std::string(argv[i]) == "-k") {
            k = std::stoul(argv[++i]);
        } else if (std::string(argv[i]) == "-f") {
            for (++i; i < argc; ++i) {
                files.emplace_back(argv[i]);
            }
        } else if (std::string(argv[i]) == "-m") {
            std::string methodStr(argv[++i]);
            if (methodStr == "lloyd") {
                method = Method::KMEANS_LLOYD;
            } else if (methodStr == "hamerly") {
                method = Method::KMEANS_HAMERLY;
            } else if (methodStr == "hierarchical") {
                method = Method::KMEANS_HIERARCHICAL;
            } else {
                std::cout << "Unknown method: " << methodStr << std::endl;
                return 1;
            }
        }
    }

    std::size_t dim;
    Dataset data;
    for (const auto& file : files) {
        auto [x, d] = readFvecs(file, dim_);
        std::cout << "Read " << x.size() / d << " vectors of dimension " << d << std::endl;
        if (x.empty()) {
            std::cout << "Failed to read corpus from " << file << std::endl;
            return 1;
        }
        data.insert(data.end(), x.begin(), x.end());
        dim = d;
    }
    //data.resize(std::min(100 * k * dim, data.size()));
    std::cout << "Read " << data.size() / dim << " corpus vectors of dimension " << dim << std::endl;

    // --- Choose Initial Centers (e.g., first k points) ---
    Centers initialCenters;
    for (int i = 0; i < k * dim; ++i) {
        initialCenters.push_back(data[i]);
    }
    
    // --- Run K-Means ---
    std::cout << "Running K-Means with k=" << k << "..." << std::endl;
    switch (method) {
        case Method::KMEANS_LLOYD: {
            std::cout << "Using Lloyd's algorithm" << std::endl;
            KMeansResult result;
            time([&] { result = kMeans(dim, data, initialCenters, k, 32); }, "K-Means Lloyd");
            std::cout << "\n--- Results ---" << result.print() << std::endl;
            std::cout << "Average distance to final centers: " << result.computeDispersion(dim, data) << std::endl;
            break;
        }
        case Method::KMEANS_HAMERLY: {
            std::cout << "Using Hamerly's algorithm" << std::endl;
            KMeansResult result;
            time([&] { result = kMeansHamerly(dim, data, initialCenters, k, 32); }, "K-Means Hamerly");
            std::cout << "\n--- Results ---" << result.print() << std::endl;
            std::cout << "Average distance to final centers: " << result.computeDispersion(dim, data) << std::endl;
            break;
        }
        case Method::KMEANS_HIERARCHICAL: {
            std::cout << "Using Hierarchical K-Means" << std::endl;
            HierarchicalKMeansResult result;
            time([&] { result = kMeansHierarchical(dim, data, 512, 32); }, "K-Means Hierarchical");
            std::cout << "\n--- Results ---" << result.print() << std::endl;
            std::cout << "Average distance to final centers: " << result.computeDispersion(dim, data) << std::endl;
            break;
        }
    }
}

    