#pragma once

#include <cstddef>
#include <cassert>
#include <limits>
#include <string>
#include <vector>

using Point = float*;
using ConstPoint = const float*;
using Dataset = std::vector<float>;
using Centers = std::vector<float>;

constexpr float INF{std::numeric_limits<float>::max()};

// Calculates the dot product of two points.
// Assumes p1 and p2 have the same dimension.
float dot(std::size_t dim,
          ConstPoint __restrict p1,
          ConstPoint __restrict p2);

// Calculates the squared Euclidean distance between two points.
// Assumes p1 and p2 have the same dimension.
float distanceSq(std::size_t dim,
                 ConstPoint __restrict p1,
                 ConstPoint __restrict p2);

// Calculates the SOAR distance between two points.
// Assumes r, x, and c have the same dimension.
// The rnorm parameter is the squared norm of the vector r.
float distanceSoar(std::size_t dim,
                   ConstPoint __restrict r,
                   ConstPoint __restrict x,
                   ConstPoint __restrict c,
                   float rnorm,
                   float lambda = 1.0F);

// Calculates the centroid of a dataset.
void centroid(std::size_t dim, const Dataset& dataset, Point centroid);

// This class encapsulates the result of the k-means clustering algorithm.
class KMeansResult {
public:
    KMeansResult() = default;
    KMeansResult(std::size_t numClusters,
                 Centers centers,
                 std::vector<std::size_t> assignments,
                 std::vector<std::size_t> spilledAssignments,
                 std::size_t iterationsRun,
                 bool converged);

    std::size_t numClusters() const { return numClusters_; }
    std::vector<std::size_t> clusterSizes() const;
    const Centers& finalCenters() const { return finalCenters_; }
    const std::vector<std::size_t>& assignments() const { return assignments_; }
    const std::vector<std::size_t>& spilledAssignments() const { return spilledAssignments_; }
    std::size_t iterationsRun() const { return iterationsRun_; }
    bool converged() const { return converged_; }
    float computeDispersion(std::size_t dim, const Dataset& dataset) const;
    std::vector<float> quantizationErrors(std::size_t dim,
                                          const Dataset& dataset) const;
    std::pair<float, float> clusterSizeMoments() const;
    std::string print() const;

private:
    std::size_t numClusters_{0};
    Centers finalCenters_;
    std::vector<std::size_t> assignments_;
    std::vector<std::size_t> spilledAssignments_;
    std::size_t iterationsRun_{0};
    bool converged_{false};
};

// This class encapsulates the result of the hierarchical k-means clustering
// algorithm.
class HierarchicalKMeansResult {
public:
    HierarchicalKMeansResult() = default;
    HierarchicalKMeansResult(const KMeansResult& result);

    std::size_t numClusters() const { return finalCenters_.size(); }
    std::size_t effectiveK() const;
    std::vector<std::size_t> clusterSizes() const;
    const std::vector<Centers>& finalCenters() const { return finalCenters_; }
    const std::vector<std::vector<std::size_t>>& assignments() const { return assignments_; }
    Centers finalCentersFlat() const;
    std::vector<std::size_t> assignmentsFlat() const;
    void copyClusterPoints(std::size_t dim,
                           std::size_t cluster,
                           const Dataset& dataset,
                           Dataset& copy) const;
    void updateAssignmentsWithRecursiveSplit(std::size_t dim,
                                             std::size_t cluster,
                                             HierarchicalKMeansResult splitClusters);
    float computeDispersion(std::size_t dim, const Dataset& dataset) const;
    std::vector<float> quantizationErrors(std::size_t dim,
                                          const Dataset& dataset) const;
    std::pair<float, float> clusterSizeMoments() const;
    std::string print() const;

private:
    std::vector<Centers> finalCenters_;
    std::vector<std::vector<std::size_t>> assignments_;
};
