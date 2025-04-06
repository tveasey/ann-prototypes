#include "baseline.h"
#include "common.h"
#include "hamerly.h"
#include "hierarchical.h"
#include "../common/utils.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <optional>
#include <queue>
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

float ivfRecall(std::size_t dim,
                const Dataset& data,
                const HierarchicalKMeansResult& result) {

    // Test IVF recall.

    using Queue = std::priority_queue<std::pair<float, std::size_t>>;

    auto updateTopk = [](std::size_t i,
                         std::size_t j,
                         float dsq,
                         std::size_t k,
                         auto& queue) {
        if (queue.size() < k) {
            queue.emplace(dsq, j);
        } else if (dsq < queue.top().first) {
            queue.pop();
            queue.emplace(dsq, j);
        }
    };

    std::size_t n{data.size() / dim};
    std::size_t m{std::min(n, 50UL)};
    std::vector<float> queries(m * dim);
    std::copy_n(data.begin(), m * dim, queries.begin());

    std::vector<Queue> nearestPoints(m);
    for (std::size_t i = 0, id = 0; id < queries.size(); ++i, id += dim) {
        for (std::size_t j = 0, jd = 0; jd < data.size(); ++j, jd += dim) {
            if (i == j) {
                continue;
            }
            float dsq{distanceSq(dim, &queries[id], &data[jd])};
            updateTopk(i, j, dsq, 10, nearestPoints[i]);
        }
    }

    std::vector<Queue> nearestClusters(m);
    std::size_t numNearestClusters{
        std::max((3 * result.finalCenters().size()) / 100, 1UL)
    };
    for (std::size_t i = 0, id = 0; id < queries.size(); ++i, id += dim) {
        for (std::size_t j = 0; j < result.finalCenters().size(); ++j) {
            float dsq{distanceSq(dim, &queries[id], &result.finalCenters()[j][0])};
            updateTopk(i, j, dsq, numNearestClusters, nearestClusters[i]);
        }
    }

    float averageRecall{0.0F};
    std::vector<std::size_t> nearest;
    std::vector<std::size_t> ivf;
    for (std::size_t i = 0; i < m; ++i) {
        nearest.clear();
        while (!nearestPoints[i].empty()) {
            nearest.push_back(nearestPoints[i].top().second);
            nearestPoints[i].pop();
        }
        ivf.clear();
        while (!nearestClusters[i].empty()) {
            std::size_t cluster{nearestClusters[i].top().second};
            ivf.insert(ivf.end(),
                       result.assignments()[cluster].begin(),
                       result.assignments()[cluster].end());
            nearestClusters[i].pop();
        }
        std::sort(ivf.begin(), ivf.end());
        auto hits = std::count_if(
            nearest.begin(), nearest.end(),
            [&ivf](std::size_t j) {
                return std::binary_search(ivf.begin(), ivf.end(), j);
            });
        averageRecall += static_cast<float>(hits) / nearest.size();
    }
    averageRecall /= m;
    return averageRecall;
}
}

int main(int argc, char** argv) {
    std::optional<int> dim_;
    std::vector<std::string> files;
    std::size_t k{100};
    Method method{Method::KMEANS_HIERARCHICAL};
    bool parameterSearch{false};
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-h") {
            std::cout << "Usage: " << argv[0] << std::endl;
            std::cout << " [-m <method>] [-p] [-k <clusters>] [-d <dim>] -f <file1> <file1> ..." << std::endl;
            std::cout << "  -m <method>       : kmeans method (lloyd, hamerly, hierarchical)" << std::endl;
            std::cout << "  -p                : parameter search" << std::endl;
            std::cout << "  -k <clusters>     : number of clusters (default: 100)" << std::endl;
            std::cout << "  -d <dim>          : dimension of vectors (default: auto)" << std::endl;
            std::cout << "  -f <file1> <file2>: input files (required)" << std::endl;
            std::cout << "  -h                : help" << std::endl;
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
        } else if (std::string(argv[i]) == "-p") {
            parameterSearch = true;
        } else {
            std::cout << "Unknown option: " << argv[i] << std::endl;
            return 1;
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
    std::cout << "Read " << data.size() / dim << " corpus vectors of dimension " << dim << std::endl;

    if (parameterSearch) {
        std::vector<std::vector<std::size_t>> parameters;
        std::vector<float> dispersions;
        std::vector<double> seconds;

        for (std::size_t maxK : {120, 128, 136}) {
            for (std::size_t samplesPerCluster : {256, 384, 512}) {
                for (std::size_t maxIterations : {6, 8, 10}) {
                    std::cout << "Running parameter search with maxK=" << maxK
                              << ", samplesPerCluster=" << samplesPerCluster
                              << ", maxIterations=" << maxIterations << std::endl;
                    HierarchicalKMeansResult result;
                    auto took = time([&] {
                        result = kMeansHierarchical(dim, data, 512, maxIterations, maxK, samplesPerCluster);
                    }, "K-Means Hierarchical").count();
                    float dispersion{result.computeDispersion(dim, data)};
                    std::cout << "Took " << took << " seconds, dispersion: " << dispersion << std::endl;
                    parameters.push_back({maxK, samplesPerCluster, maxIterations});
                    seconds.push_back(took);
                    dispersions.push_back(dispersion);
                }
            }
        }

        double minDispersion{*std::min_element(dispersions.begin(), dispersions.end())};
        double minSeconds{*std::min_element(seconds.begin(), seconds.end())};
        std::cout << "Min dispersion: " << minDispersion << std::endl;
        std::cout << "Min seconds: " << minSeconds << std::endl;
        std::transform(dispersions.begin(), dispersions.end(), dispersions.begin(),
                       [minDispersion](double d) { return d / minDispersion; });
        std::transform(seconds.begin(), seconds.end(), seconds.begin(),
                       [minSeconds](double s) { return s / minSeconds; });

        std::vector<std::size_t> bestParameters;
        double bestScore{std::numeric_limits<double>::max()};
        for (std::size_t i = 0; i < parameters.size(); ++i) {
            double score{10.0 * dispersions[i] + seconds[i]};
            if (score < bestScore) {
                bestScore = score;
                bestParameters = parameters[i];
            }
        }

        std::cout << "Parameters: " << parameters << std::endl;
        std::cout << "Dispersions: " << dispersions << std::endl;
        std::cout << "Seconds: " << seconds << std::endl;
        std::cout << "Best parameters: maxK=" << bestParameters[0]
                  << ", samplesPerCluster=" << bestParameters[1]
                  << ", maxIterations=" << bestParameters[2] << std::endl;
        std::cout << "Best score: " << bestScore << std::endl;
        return 0;
    }

    // --- Downsample for raw clustering ---
    std::vector<float> sample(std::min(256 * k * dim, data.size()));
    std::copy_n(data.begin(), sample.size(), sample.begin());

    // --- Choose Initial Centers (e.g., first k points) ---
    Centers initialCenters;
    k = pickInitialCenters(dim, sample, k, initialCenters);

    // --- Run K-Means ---
    switch (method) {
        case Method::KMEANS_LLOYD: {
            std::cout << "Running K-Means with k=" << k << "..." << std::endl;
            std::cout << "Using Lloyd's algorithm" << std::endl;
            KMeansResult result;
            time([&] {
                result = kMeans(dim, sample, initialCenters, k, 8);
                result.assignRemainingPoints(dim, sample.size(), data);
            }, "K-Means Lloyd");
            std::cout << "\n--- Results ---" << result.print() << std::endl;
            std::cout << "Average distance to final centers: " << result.computeDispersion(dim, data) << std::endl;
            break;
        }
        case Method::KMEANS_HAMERLY: {
            std::cout << "Running K-Means with k=" << k << "..." << std::endl;
            std::cout << "Using Hamerly's algorithm" << std::endl;
            KMeansResult result;
            time([&] {
                result = kMeansHamerly(dim, sample, initialCenters, k, 32);
                result.assignRemainingPoints(dim, sample.size(), data);
            }, "K-Means Hamerly");
            std::cout << "\n--- Results ---" << result.print() << std::endl;
            std::cout << "Average distance to final centers: " << result.computeDispersion(dim, data) << std::endl;
            break;
        }
        case Method::KMEANS_HIERARCHICAL: {
            std::cout << "Running K-Means..." << std::endl;
            std::cout << "Using Hierarchical K-Means" << std::endl;
            HierarchicalKMeansResult result;
            time([&] { result = kMeansHierarchical(dim, data, 256, 8); }, "K-Means Hierarchical");
            std::cout << "\n--- Results ---" << result.print() << std::endl;
            std::cout << "Average distance to final centers: " << result.computeDispersion(dim, data) << std::endl;
            std::cout << "Testing IVF recall..." << std::endl;
            std::cout << "Average recall: " << ivfRecall(dim, data, result) << std::endl;
            break;
        }
    }

    std::cout << "--- End Results ---" << std::endl;
}

