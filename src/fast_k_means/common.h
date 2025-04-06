#pragma once

#include <cstddef>
#include <cassert>
#include <string>
#include <vector>

/*#include <iostream>
template<typename T>
class SafeVector {
public:
    SafeVector() = default;
    SafeVector(std::size_t size) : data_(size) {}
    SafeVector(std::size_t size, T value) : data_(size, value) {}
    SafeVector(std::vector<T> data) : data_(std::move(data)) {}
    SafeVector(SafeVector&&) = default;
    SafeVector(const SafeVector&) = default;
    SafeVector& operator=(SafeVector&&) = default;
    SafeVector& operator=(const SafeVector&) = default;

    T& operator[](std::size_t index) { if (index >= data_.size()) {std::cout << "out of bounds" << std::endl;}; return data_[index]; }
    const T& operator[](std::size_t index) const { if (index >= data_.size()) {std::cout << "out of bounds" << std::endl;}; return data_[index]; }
    typename std::vector<T>::iterator begin() { return data_.begin(); }
    typename std::vector<T>::iterator end() { return data_.end(); }
    typename std::vector<T>::const_iterator begin() const { return data_.begin(); }
    typename std::vector<T>::const_iterator end() const { return data_.end(); }
    std::size_t size() const { return data_.size(); }
    void resize(std::size_t size) { data_.resize(size); }
    void clear() { data_.clear(); }
    void push_back(const T& value) { data_.push_back(value); }
    void assign(std::size_t size, float value) { data_.assign(size, value); }
    void assign(std::size_t size) { data_.assign(size, 0.0F); }

private:
    std::vector<T> data_;
};*/

using Point = float*;
using ConstPoint = const float*;
using Dataset = std::vector<float>;
using Centers = std::vector<float>;

// Calculates the squared Euclidean distance between two points.
// Assumes p1 and p2 have the same dimension.
float distanceSq(std::size_t dim,
                 ConstPoint __restrict p1,
                 ConstPoint __restrict p2);

/// Calculates the centroid of a dataset.
void centroid(std::size_t dim, const Dataset& dataset, Point centroid);

// This class encapsulates the result of the k-means clustering algorithm.
class KMeansResult {
public:
    KMeansResult() = default;
    KMeansResult(std::size_t numClusters,
                 Centers centers,
                 std::vector<std::size_t> assignments,
                 std::size_t iterationsRun,
                 bool converged);

    std::size_t numClusters() const { return numClusters_; }
    std::vector<std::size_t> clusterSizes() const;
    const Centers& finalCenters() const { return finalCenters_; }
    const std::vector<std::size_t>& assignments() const { return assignments_; }
    std::size_t iterationsRun() const { return iterationsRun_; }
    bool converged() const { return converged_; }
    void assignRemainingPoints(std::size_t dim,
                               std::size_t beginUnassigned,
                               const Dataset& dataset);
    float computeDispersion(std::size_t dim, const Dataset& dataset) const;
    std::pair<float, float> clusterSizeMoments() const;
    std::string print() const;

private:
    std::size_t numClusters_{0};
    Centers finalCenters_;
    std::vector<std::size_t> assignments_;
    std::size_t iterationsRun_{0};
    bool converged_{false};
};

// This class encapsulates the result of the hierarchical k-means clustering
// algorithm.
class HierarchicalKMeansResult {
public:
    HierarchicalKMeansResult() = default;
    HierarchicalKMeansResult(const KMeansResult& result);
    HierarchicalKMeansResult(std::vector<Centers> finalCenters,
                             std::vector<std::vector<std::size_t>> assignments);

    std::size_t numClusters() const { return finalCenters_.size(); }
    std::vector<std::size_t> clusterSizes() const;
    const std::vector<Centers>& finalCenters() const { return finalCenters_; }
    const std::vector<std::vector<std::size_t>>& assignments() const { return assignments_; }
    void copyClusterPoints(std::size_t dim,
                           std::size_t cluster,
                           const Dataset& dataset,
                           Dataset& copy) const;
    void updateAssignmentsWithRecursiveSplit(std::size_t dim,
                                             std::size_t cluster,
                                             HierarchicalKMeansResult splitClusters);
    float computeDispersion(std::size_t dim, const Dataset& dataset) const;
    std::pair<float, float> clusterSizeMoments() const;
    std::string print() const;

private:
    std::vector<Centers> finalCenters_;
    std::vector<std::vector<std::size_t>> assignments_;
};
