#include "preconditioner.h"

#include "utils.h"

#include <cstddef>
#include <numeric>
#include <vector>

std::vector<std::vector<std::size_t>>
permutationMatrix(std::size_t dim,
                  const std::vector<std::size_t>& dimBlocks,
                  const std::vector<float>& vectors) {

    if (dimBlocks.size() == 1) {
        std::vector<std::size_t> indices(dim);
        std::iota(indices.begin(), indices.end(), 0);
        return {std::move(indices)};
    }

    // Use a greedy approach to pick assignments to blocks that equalizes their variance.

    std::vector<OnlineMeanAndVariance> moments(dim);
    for (std::size_t i = 0; i < vectors.size(); i += dim) {
        for (std::size_t j = 0; j < dim; ++j) {
            moments[j].add(vectors[i + j]);
        }
    }
    std::vector<std::size_t> indices(dim);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](const auto& lhs, const auto& rhs) {
                  return moments[lhs].var() > moments[rhs].var();
              });

    std::vector<std::vector<std::size_t>> permutationMatrix(dimBlocks.size());

    std::vector<double> variances(dimBlocks.size(), 0.0);
    for (std::size_t i : indices) {
        auto j = std::min_element(variances.begin(), variances.end()) - variances.begin();
        permutationMatrix[j].push_back(i);
        variances[j] = (
            permutationMatrix[j].size() == dimBlocks[j] ?
            std::numeric_limits<double>::max() : // Prevent further assignments.
            variances[j] + moments[i].var()
        );
    }
    for (std::size_t i = 0; i < permutationMatrix.size(); ++i) {
        std::sort(permutationMatrix[i].begin(), permutationMatrix[i].end());
    }

    return permutationMatrix;
}

void applyTransform(std::size_t dim,
                    const std::vector<std::vector<std::size_t>>& permutationMatrix,
                    const std::vector<std::vector<float>>& blocks,
                    const std::vector<std::size_t>& dimBlocks,
                    std::vector<float>& vectors) {

    if (blocks.size() == 1) {
        std::vector<float> x(dimBlocks[0]);
        for (std::size_t i = 0; i < vectors.size(); i += dim) {
            matrixVectorMultiply(dimBlocks[0], blocks[0].data(), vectors.data() + i, x.data());
            std::copy(x.begin(), x.end(), vectors.begin() + i);
        }
        return;
    }

    std::vector<float> v(dim);
    std::vector<float> x;
    std::vector<float> y;
    for (std::size_t i = 0; i < vectors.size(); /**/) {
        // We need a copy of the vector because we're permuting it in place.
        std::copy(vectors.begin() + i, vectors.begin() + i + dim, v.begin());
        for (std::size_t j = 0; j < blocks.size(); ++j) {
            auto& block = blocks[j];
            auto dimBlock = dimBlocks[j];
            x.resize(dimBlock);
            y.resize(dimBlock);
            std::transform(permutationMatrix[j].begin(), permutationMatrix[j].end(),
                           x.begin(),
                           [&](std::size_t k) { return v[k]; });
            matrixVectorMultiply(dimBlock, block.data(), x.data(), y.data());
            std::copy(y.begin(), y.end(), vectors.begin() + i);
            i += dimBlock;
        }
    }
}
    