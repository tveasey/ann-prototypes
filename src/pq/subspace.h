#pragma once

#include <cstddef>
#include <utility>
#include <vector>

// Compute the mean of the given vectors and subtract it from each vector.
std::vector<float> centreData(std::size_t dim,
                              std::vector<float> data);

// Compute the covariance matrix of the given vectors.
//
// Returns a vector of length dim * dim containing the covariance matrix
// stored flat.
std::vector<double> covarianceMatrix(std::size_t dim,
                                     const std::vector<float>& data);

// Compute the principal components of the given vectors.
//
// These are the eigenvectors of the covariance matrix of the vectors.
// Returns a pair of vectors, the first containing the eigenvectors
// stored flat and the second containing the corresponding eigenvalues.
// They are sorted in descending order of eigenvalue.
std::pair<std::vector<double>, std::vector<double>>
pca(std::size_t dim, std::vector<float> docs);

// Compute the optimal PQ transformations for the given eigenvectors and
// eigenvalues. The eigenvectors are assumed to be sorted in descending
// order of eigenvalue.
//
// This uses the parameteric approach from the paper:
// "Optimized Product Quantization" by Tiezheng Ge, Kaiming He†, Qifa Ke,
// and Jian Sun.
//
// Returns the orthogonal transformation matrix stored flat.
std::vector<float> computeOptimalPQSubspaces(std::size_t dim,
                                             std::size_t numSubspaces,
                                             const std::vector<double>& eigVecs,
                                             const std::vector<double>& eigVals);

// Transform the given vectors using the given transformation.
std::vector<float> transform(const std::vector<float>& transformation,
                             std::size_t dim,
                             std::vector<float> docs);
