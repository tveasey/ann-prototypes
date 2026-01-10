#include "optimal_order.h"

#include "hierarchical.h"
#include "../common/utils.h"
#include "../centroid_ordering/annealing.h"

#include <cstddef>
//#include <iomanip>
#include <iostream>

void reorderCentroids(std::size_t dim,
                      const Dataset& corpus,
                      std::size_t target,
                      std::size_t trials) {

    auto computeCosineSimilarity = [&](const Permutation& permutation,
                                       const Dataset& centers) {
        std::size_t k{centers.size() / dim};
        std::vector<std::vector<float>> simMatrix(k, std::vector<float>(k, 1.0F));
        for (std::size_t i = 0; i < k; ++i) {
            for (std::size_t j = i + 1; j < k; ++j) {
                float sim{dot(dim,
                              &centers[dim * permutation[i]],
                              &centers[dim * permutation[j]])};
                simMatrix[i][j] = sim;
                simMatrix[j][i] = sim;
            }
        }
        return simMatrix;
    };

    std::cout << "Running K-Means..." << std::endl;
    std::cout << "Using Hierarchical K-Means" << std::endl;
    HierarchicalKMeansResult result;
    time([&] { result = kMeansHierarchical(dim, corpus, target); }, "K-Means Hierarchical");
    std::cout << "Average distance to final centers: " << result.computeDispersion(dim, corpus) << std::endl;

    std::cout << "Computing optimal centroid ordering..." << std::endl;
    std::cout << "\n--- Results ---" << result.print() << std::endl;
    for (std::size_t t = 0; t < trials; ++t) {
        std::cout << "Trial " << (t + 1) << " / " << trials << std::endl;
        Permutation permutation;
        time([&] { permutation = annealingOrder(dim, result.finalCentersFlat()); }, "Annealing Order");
    }

    // Compute cosine similarity martix before and after reordering.
    /*
    // Normalize centroids.
    Centers normalizedCenters{result.finalCentersFlat()};
    normalize(dim, normalizedCenters);

    Permutation identityPermutation(permutation.size());
    std::iota(identityPermutation.begin(), identityPermutation.end(), 0UL);

    auto simMatrixBefore{computeCosineSimilarity(identityPermutation, normalizedCenters)};
    std::cout << "Cosine Similarity Matrix Before Reordering:" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "before = [" << std::endl;
    for (const auto& row : simMatrixBefore) {
        std::cout << "[" << row[0];
        for (std::size_t j = 1; j < row.size(); ++j) {
            std::cout << ", " << row[j];
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "];" << std::endl;

    auto simMatrixAfter{computeCosineSimilarity(permutation, normalizedCenters)};
    std::cout << "Cosine Similarity Matrix After Reordering:" << std::endl;
    std::cout << "after = [ " << std::endl;
    for (const auto& row : simMatrixAfter) {
        std::cout << "[" << row[0];
        for (std::size_t j = 1; j < row.size(); ++j) {
            std::cout << ", " << row[j];
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "];" << std::endl;
    */
}