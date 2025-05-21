#include "baseline.h"
#include "common.h"
#include "hierarchical.h"
#include "ivf.h"
#include "../common/utils.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <optional>
#include <queue>
#include <random>
#include <unordered_map>
#include <utility>
#include <unordered_set>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>

namespace {
enum class Method {
    KMEANS_LLOYD,
    KMEANS_HIERARCHICAL,
    IVF
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

void ivfRecall(Metric metric,
               std::size_t dim,
               std::vector<std::size_t> percentages,
               std::size_t target,
               std::size_t bits,
               std::size_t rerank,
               const Dataset& data) {

    using Queue = std::priority_queue<std::pair<float, std::size_t>>;

    auto distance = [&](ConstPoint p1, ConstPoint p2) {
        switch (metric) {
        case Cosine:
            return 1.0F - dot(dim, p1, p2);
        case Dot:
            return -dot(dim, p1, p2);
        case Euclidean:
            return distanceSq(dim, p1, p2);
        }
    };

    auto updateTopk = [](std::size_t k,
                         float d,
                         std::size_t i,
                         auto& queue) {
        if (queue.size() < k) {
            queue.emplace(d, i);
        } else if (d < queue.top().first) {
            queue.pop();
            queue.emplace(d, i);
        }
    };

    std::size_t k{10};

    // Pick at most 75 random queries.
    std::size_t n{data.size() / dim};
    std::size_t m{std::min(n, 75UL)};
    std::vector<float> queries(m * dim);
    std::minstd_rand rng;
    std::uniform_int_distribution<std::size_t> u0n(0, n - 1);
    for (std::size_t i = 0; i < m; ++i) {
        std::size_t j{u0n(rng)};
        std::copy_n(&data[j * dim], dim, &queries[i * dim]);
    }

    // Brute force search.
    std::vector<std::vector<std::size_t>> actual(m);
    Queue topk;
    for (std::size_t i = 0, id = 0; id < queries.size(); ++i, id += dim) {
        for (std::size_t j = 0, jd = 0; jd < data.size(); ++j, jd += dim) {
            updateTopk(k, distance(&queries[id], &data[jd]), j, topk);
        }
        while (!topk.empty()) {
            actual[i].push_back(topk.top().second);
            topk.pop();
        }
    }

    CQuantizedIvfIndex index{metric, dim, data, target, bits, 32};
    
    std::vector<float> query(dim);
    std::unordered_set<std::size_t> found;
    std::size_t comparisons{0};
    for (auto percentage : percentages) {
        std::size_t probes{
            (percentage * index.numClusters()) / 100
        };
        float averageRecall{0.0F};
        std::size_t averageComparisons{0};

        for (std::size_t i = 0, id = 0; i < m; ++i, id += dim) {
            std::copy_n(&queries[id], dim, &query[0]);
            std::tie(found, comparisons) = index.search(probes, k, rerank, query, data);
            auto hits = std::count_if(
                actual[i].begin(), actual[i].end(),
                [&found](std::size_t j) {
                    return found.find(j) != found.end();
                });
            averageRecall += static_cast<float>(hits) / actual[i].size();
            averageComparisons += comparisons;
        }
        averageRecall /= m;
        averageComparisons /= m;
        std::cout << "IVF: recall = " << averageRecall
                  << ", comparisons = " << averageComparisons
                  << ", % compared = " << static_cast<float>(averageComparisons) / n << std::endl;
    }
}

}

int main(int argc, char** argv) {
    std::optional<int> dim_;
    std::vector<std::string> files;
    std::size_t target{384};
    std::size_t k{100};
    std::size_t bits{1};
    std::size_t rerank{5};
    Metric metric{Cosine};
    Method method{Method::KMEANS_HIERARCHICAL};
    bool parameterSearch{false};
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-h") {
            std::cout << "Usage: " << argv[0] << std::endl;
            std::cout << " [--metric <metric>] [--method <method>] [-p] [-k <clusters>] [-b <bits>] "
                      << "[-r <rerank>] [-d <dim>] -f <file1> <file1> ..." << std::endl;
            std::cout << "  --metric <metric> : distance metric (cosine, euclidean, mip)" << std::endl;
            std::cout << "  --method <method> : kmeans method (lloyd, hierarchical)" << std::endl;
            std::cout << "  -s                : target cluster size" << std::endl;
            std::cout << "  -p                : parameter search" << std::endl;
            std::cout << "  -k <clusters>     : number of clusters (default: 100)" << std::endl;
            std::cout << "  -b <bits>         : number of quantization bits (default: 1)" << std::endl;
            std::cout << "  -r <rerank>       : the multiple of k to rerank (default: 5)" << std::endl;
            std::cout << "  -d <dim>          : dimension of vectors (default: auto)" << std::endl;
            std::cout << "  -f <file1> <file2>: input files (required)" << std::endl;
            std::cout << "  -h                : help" << std::endl;
            return 0;
        } else if (std::string(argv[i]) == "-d") {
            dim_ = std::stoul(argv[++i]);
        } else if (std::string(argv[i]) == "-k") {
            k = std::stoul(argv[++i]);
        } else if (std::string(argv[i]) == "-b") {
            bits = std::stoul(argv[++i]);
        } else if (std::string(argv[i]) == "-r") {
            rerank = std::stoul(argv[++i]);
        } else if (std::string(argv[i]) == "-f") {
            for (++i; i < argc; ++i) {
                files.emplace_back(argv[i]);
            }
        } else if (std::string(argv[i]) == "-s") {
            target = std::stoul(argv[++i]);
        } else if (std::string(argv[i]) == "--method") {
            std::string methodStr(argv[++i]);
            if (methodStr == "lloyd") {
                method = Method::KMEANS_LLOYD;
            } else if (methodStr == "hierarchical") {
                method = Method::KMEANS_HIERARCHICAL;
            } else if (methodStr == "ivf") {
                method = Method::IVF;
            } else {
                std::cout << "Unknown method: " << methodStr << std::endl;
                return 1;
            }
        } else if (std::string(argv[i]) == "--metric") {
            std::string metricStr(argv[++i]);
            if (metricStr == "cosine") {
                metric = Cosine;
            } else if (metricStr == "mip") {
                metric = Dot;
            } else if (metricStr == "euclidean") {
                metric = Euclidean;
            } else {
                std::cout << "Unknown metric: " << metricStr << std::endl;
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

    if (metric == Cosine) {
        std::cout << "Normalizing data..." << std::endl;
        normalize(dim, data);
    }
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
                        result = kMeansHierarchical(
                            dim, data, 512, maxK, maxIterations, samplesPerCluster
                    );
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

    k = data.size() / dim / 512;

    // --- Downsample for raw clustering ---
    std::size_t sampleSize(std::min(256 * k * dim, data.size()));

    // --- Choose Initial Centers (e.g., first k points) ---
    Centers initialCenters;
    pickInitialCenters(dim, data, sampleSize, k, initialCenters);

    // --- Run K-Means ---
    switch (method) {
        case Method::KMEANS_LLOYD: {
            std::cout << "Running K-Means with k=" << k << "..." << std::endl;
            std::cout << "Using Lloyd's algorithm" << std::endl;
            KMeansResult result;
            time([&] {
                result = kMeans(dim, data, sampleSize, initialCenters, 8);
            }, "K-Means Lloyd");
            std::cout << "\n--- Results ---" << result.print() << std::endl;
            std::cout << "Average distance to final centers: " << result.computeDispersion(dim, data) << std::endl;
            break;
        }
        case Method::KMEANS_HIERARCHICAL: {
            std::cout << "Running K-Means..." << std::endl;
            std::cout << "Using Hierarchical K-Means" << std::endl;
            HierarchicalKMeansResult result;
            time([&] { result = kMeansHierarchical(dim, data, target); }, "K-Means Hierarchical");
            std::cout << "\n--- Results ---" << result.print() << std::endl;
            std::cout << "Average distance to final centers: " << result.computeDispersion(dim, data) << std::endl;
            break;
        }
        case Method::IVF: {
            std::cout << "Testing IVF recall..." << std::endl;
            ivfRecall(metric, dim, {1, 2, 3, 4, 5, 6}, target, bits, rerank, data);
            break;
        }
    }

    std::cout << "--- End Results ---" << std::endl;
}

