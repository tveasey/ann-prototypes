#include "utils.h"

#include "constants.h"
#include "../common/utils.h"

#include "../Eigen/Dense"
#include "../Eigen/SVD"

#include <cstdint>
#include <functional>
#include <iostream>
#include <queue>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

std::vector<float> centreData(std::size_t dim,
                              std::vector<float> data) {
    std::vector<float> centroid(dim, 0.0F);
    std::size_t numDocs{data.size() / dim};
    for (std::size_t i = 0; i < data.size(); i += dim) {
        #pragma clang loop unroll_count(4) vectorize(assume_safety)
        for (std::size_t j = 0; j < dim; ++j) {
            centroid[j] += data[i + j];
        }
    }
    for (std::size_t j = 0; j < dim; ++j) {
        centroid[j] /= static_cast<float>(numDocs);
    }
    for (std::size_t i = 0; i < data.size(); i += dim) {
        #pragma clang loop unroll_count(4) vectorize(assume_safety)
        for (std::size_t j = 0; j < dim; ++j) {
            data[i + j] -= centroid[j];
        }
    }
    return std::move(data);
} 

std::vector<double> covarianceMatrix(std::size_t dim,
                                     const std::vector<float>& data) {

    std::size_t numDocs{data.size() / dim};
    std::vector<double> cov(dim * dim, 0.0F);
    for (std::size_t i = 0; i < data.size(); i += dim) {
        for (std::size_t j = 0; j < dim; ++j) {
            #pragma clang loop unroll_count(4) vectorize(assume_safety)
            for (std::size_t k = 0; k < dim; ++k) {
                cov[j * dim + k] += data[i + j] * data[i + k];
            }
        }
    }

    for (std::size_t j = 0; j < dim; ++j) {
        #pragma clang loop unroll_count(4) vectorize(assume_safety)
        for (std::size_t k = 0; k < dim; ++k) {
            cov[j * dim + k] /= static_cast<double>(numDocs);
        }
    }


    return cov;
}

std::pair<std::vector<double>, std::vector<double>>
pca(std::size_t dim, std::vector<float> docs) {

    if (docs.size() % dim != 0) {
        throw std::invalid_argument("docs.size() % dim != 0");
    }

    docs = centreData(dim, std::move(docs));
    auto cov = covarianceMatrix(dim, docs);

    // Compute the eigenvectors and eigenvalues of the covariance matrix
    // using Eigen's JacobiSVD.
    Eigen::Map<Eigen::MatrixXd> covMat(cov.data(), dim, dim);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(covMat, Eigen::ComputeThinU | Eigen::ComputeThinV);

    // Copy the eigenvectors and eigenvalues into std::vector.
    std::vector<double> eigVals(dim, 0.0F);
    std::vector<double> eigVecs(dim * dim, 0.0F);
    Eigen::Map<Eigen::VectorXd> eigValsMat(eigVals.data(), dim);
    Eigen::Map<Eigen::MatrixXd> eigVecsMat(eigVecs.data(), dim, dim);
    eigValsMat = svd.singularValues();
    eigVecsMat = svd.matrixU();

    // Sort the eigenvectors and eigenvalues in descending order.
    std::vector<std::size_t> sortedEigValIdx(dim);
    std::iota(sortedEigValIdx.begin(), sortedEigValIdx.end(), 0);
    std::sort(sortedEigValIdx.begin(), sortedEigValIdx.end(),
              [&eigVals](std::size_t i, std::size_t j) {
                  return std::make_pair(eigVals[i], i) > std::make_pair(eigVals[j], j);
              });
    std::vector<double> sortedEigVals(dim, 0.0F);
    std::vector<double> sortedEigVecs(dim * dim, 0.0F);
    for (std::size_t i = 0; i < dim; ++i) {
        std::size_t k{sortedEigValIdx[i]};
        sortedEigVals[i] = eigVals[k];
        #pragma clang loop unroll_count(4) vectorize(assume_safety)
        for (std::size_t j = 0; j < dim; ++j) {
            sortedEigVecs[i * dim + j] = eigVecs[k * dim + j];
        }
    }

    return std::make_pair(std::move(sortedEigVecs), std::move(sortedEigVals));
}

std::vector<float>
computeOptimalPQSubspaces(std::size_t dim,
                          const std::vector<double>& eigVecs,
                          const std::vector<double>& eigVals) {

    // Compute the optimal PQ transformations using the method described in
    // "Optimized Product Quantization for Approximate Nearest Neighbor Search"
    // by Tiezheng Ge, Kaiming He, Qifa Ke, and Jian Sun.

    // We assume that we have all eigenvectors and eigenvaluess of the data
    // covariance matrix which is rank full and they sorted descending by
    // eigenvalue.
    if (eigVecs.size() != dim * dim) {
        throw std::invalid_argument("eigVecs.size() != dim * dim");
    }
    if (eigVals.size() != dim) {
        throw std::invalid_argument("eigVals.size() != dim");
    }
    if (!std::is_sorted(eigVals.begin(), eigVals.end(),
                        std::greater<double>())) {
        throw std::invalid_argument("eigVals not sorted in descending order");
    }

    // If the largest eigenvalue is 0 then fall back to the identity.
    double maxEigVal{eigVals.back()};
    if (maxEigVal <= 0.0) {
        std::vector<float> transformation(dim * dim, 0.0F);
        for (std::size_t i = 0; i < dim; ++i) {
            transformation[i * dim + i] = 1.0F;
        }
        return transformation;
    }

    // We assume without loss of generality that numSubspaces divides dim
    // because we can always zero-pad the document vectors to make this the
    // case.
    if (dim % NUM_BOOKS != 0) {
        throw std::invalid_argument("dim % numSubspaces != 0");
    }

    std::size_t subspaceDim{dim / NUM_BOOKS};

    // We rescale eigenvalues so that the smalled is 1.0. This means that
    // each time we add an eigenvector to a subspace we increase its product
    // of eigenvalues.
    double logMinEigVal{std::log(std::max(eigVals.back(), 1e-8 * maxEigVal))};

    // Assign the first numSubspaces eigenvectors (one to each book).
    std::vector<std::vector<std::size_t>> assignments(NUM_BOOKS);
    std::priority_queue<std::pair<double, std::size_t>,
                        std::vector<std::pair<double, std::size_t>>,
                        std::greater<std::pair<double, std::size_t>>> logProdEigVals;
    for (std::size_t i = 0; i < NUM_BOOKS; ++i) {
        assignments[i].push_back(i * dim);
        double eigValAdj{std::max(eigVals[i], logMinEigVal)};
        logProdEigVals.push(std::make_pair(std::log(eigValAdj) - logMinEigVal, i));
    }

    // Use a greedy bin packing algorithm heuristic where at each step the
    // eigenvector of the largest remaining eigenvalue is assigned to the
    // subspace with the minimum product of eigenvalues.
    for (std::size_t i = NUM_BOOKS; i < dim; ++i) {
        while (!logProdEigVals.empty()) {
            auto [minLogProdEigVals, j] = logProdEigVals.top();
            logProdEigVals.pop();
            if (assignments[j].size() == subspaceDim) {
                continue;
            }
            assignments[j].push_back(i * dim);
            double eigValAdj{std::max(eigVals[i], logMinEigVal)};
            minLogProdEigVals += std::log(eigValAdj) - logMinEigVal;
            logProdEigVals.push(std::make_pair(minLogProdEigVals, j));
            break;
        }
    }

    // Compute the optimal PQ transformations.
    std::vector<float> transformation(dim * dim, 0.0F);
    for (std::size_t i = 0, pos = 0; i < assignments.size(); ++i) {
        for (auto j : assignments[i]) {
            for (std::size_t k = 0; k < dim; ++k) {
                transformation[pos + k] = static_cast<float>(eigVecs[j + k]);
            }
            pos += dim;
        }
    }

    return transformation;
}

std::vector<float> transform(const std::vector<float>& transformation,
                             std::size_t dim,
                             std::vector<float> docs) {
    if (transformation.size() != dim * dim) {
        throw std::invalid_argument("transformation.size() != dim * dim");
    }
    if (docs.size() % dim != 0) {
        throw std::invalid_argument("docs.size() % dim != 0");
    }

    std::size_t numDocs{docs.size() / dim};
    std::vector<float> projected(dim, 0.0F);
    for (std::size_t i = 0; i < docs.size(); i += dim) {
        projected.assign(dim, 0.0F);
        for (std::size_t j = 0; j < dim; ++j) {
            #pragma clang loop unroll_count(4) vectorize(assume_safety)
            for (std::size_t k = 0; k < dim; ++k) {
                projected[j] += transformation[j * dim + k] * docs[i + k];
            }
        }
        std::copy(projected.begin(), projected.end(), docs.begin() + i);
    }
    return std::move(docs);
}
