#include "common.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <numeric>
#include <utility>

float distanceSq(std::size_t dim,
                 ConstPoint __restrict p1,
                 ConstPoint __restrict p2) {
    float dsq{0.0F};
    #pragma omp simd reduction(+:dsq)
    for (std::size_t i = 0; i < dim; ++i) {
        float diff{p1[i] - p2[i]};
        dsq += diff * diff;
    }
    return dsq;
}

void centroid(std::size_t dim, const Dataset& dataset, Point centroid) {
    std::fill_n(centroid, dim, 0.0F);
    std::size_t n{dataset.size()};
    for (std::size_t id = 0; id < n; id += dim) {
        ConstPoint xi{&dataset[id]};
        #pragma omp simd
        for (std::size_t d = 0; d < dim; ++d) {
            centroid[d] += xi[d];
        }
    }
    #pragma omp simd
    for (std::size_t d = 0; d < dim; ++d) {
        centroid[d] = (centroid[d] * dim) / n;
    }
}


KMeansResult::KMeansResult(std::size_t numClusters,
                           Centers centers,
                           std::vector<std::size_t> assignments,
                           std::vector<std::size_t> spilledAssignments,
                           std::size_t iterationsRun,
                           bool converged)
    : numClusters_(numClusters),
      finalCenters_(std::move(centers)),
      assignments_(std::move(assignments)),
      spilledAssignments_(std::move(spilledAssignments)),
      iterationsRun_(iterationsRun),
      converged_(converged) {
}

std::vector<std::size_t> KMeansResult::clusterSizes() const {
    std::vector<std::size_t> sizes(numClusters_, 0);
    std::size_t dim{finalCenters_.size() / numClusters_};
    for (std::size_t i = 0; i < assignments_.size(); ++i) {
        ++sizes[assignments_[i] / dim];
    }
    return sizes;
}

void KMeansResult::assignRemainingPoints(std::size_t dim,
                                         std::size_t beginUnassigned,
                                         const Dataset& dataset) {

    if (beginUnassigned >= dataset.size()) {
        return;
    }

    Centers newFinalCenters(finalCenters_);
    std::vector<std::size_t> sizes(clusterSizes());
    for (std::size_t i = 0, id = 0; i < sizes.size(); ++i, id += dim) {
        for (std::size_t d = 0; d < dim; ++d) {
            newFinalCenters[id + d] *= sizes[i];
        }
    }
    assignments_.resize(dataset.size() / dim);
    for (std::size_t i = beginUnassigned / dim, id = beginUnassigned;
         id < dataset.size();
         ++i, id += dim) {
        std::size_t bestJd{0};
        float minDsq{std::numeric_limits<float>::max()};
        for (std::size_t jd = 0; jd < finalCenters_.size(); jd += dim) {
            float dsq{distanceSq(dim, &dataset[id], &finalCenters_[jd])};
            if (dsq < minDsq) {
                minDsq = dsq;
                bestJd = jd;
            }
        }
        assignments_[i] = bestJd;
        #pragma omp simd
        for (std::size_t d = 0; d < dim; ++d) {
            newFinalCenters[bestJd + d] += dataset[id + d];
        }
        ++sizes[bestJd / dim];
    }
    for (std::size_t i = 0, id = 0; id < finalCenters_.size(); ++i, id += dim) {
        if (sizes[i] > 0) {
            #pragma omp simd
            for (std::size_t d = 0; d < dim; ++d) {
                finalCenters_[id + d] = newFinalCenters[id + d] / sizes[i];
            }
        }
    }
}

float KMeansResult::computeDispersion(std::size_t dim, const Dataset& dataset) const {
    float totalDispersion{0.0F};
    std::size_t n{assignments_.size()};
    for (std::size_t i = 0, id = 0; i < n; ++i, id += dim) {
        std::size_t cluster{assignments_[i]};
        totalDispersion +=  distanceSq(dim, &dataset[id], &finalCenters_[cluster]);
    }
    return totalDispersion / n;
}

std::pair<float, float> KMeansResult::clusterSizeMoments() const {
    std::vector<std::size_t> sizes(clusterSizes());
    float mean{0.0F};
    for (std::size_t size : sizes) {
        mean += size;
    }
    mean /= sizes.size();
    float variance{0.0F};
    for (std::size_t size : sizes) {
        float diff{static_cast<float>(size) - mean};
        variance += diff * diff;
    }
    return {mean, std::sqrtf(variance / sizes.size())};
}

std::string KMeansResult::print() const {
    std::string result;
    result += "\nConverged: ";
    result += (converged_ ? "Yes" : "No");
    result += "\nIterations Run: " + std::to_string(iterationsRun_);
    result += "\nNumber of clusters: " + std::to_string(finalCenters_.size());
    auto moments = clusterSizeMoments();
    result += "\nCluster size moments: mean = " + std::to_string(moments.first) + " sd = " +
              std::to_string(moments.second);
    return result;
}

HierarchicalKMeansResult::HierarchicalKMeansResult(const KMeansResult& result)
    : finalCenters_(result.numClusters()),
      assignments_(result.numClusters()) {

    std::size_t dim{result.finalCenters().size() / result.numClusters()};
    for (std::size_t i = 0; i < result.assignments().size(); ++i) {
        std::size_t cluster{result.assignments()[i] / dim};
        assignments_[cluster].push_back(i);
    }
    for (std::size_t i = 0; i < result.spilledAssignments().size(); ++i) {
        std::size_t spilled{result.spilledAssignments()[i] / dim};
        assignments_[spilled].push_back(i);
    }
    for (auto& assignment : assignments_) {
        std::sort(assignment.begin(), assignment.end());
    }
    for (std::size_t i = 0, id = 0; i < finalCenters_.size(); ++i, id += dim) {
        finalCenters_[i].resize(dim);
        std::copy_n(&result.finalCenters()[id], dim, &finalCenters_[i][0]);
    }
}

std::size_t HierarchicalKMeansResult::effectiveK() const {
    return std::count_if(
        assignments_.begin(), assignments_.end(),
        [](const std::vector<std::size_t>& cluster) {
            return !cluster.empty();
        });
}

std::vector<std::size_t> HierarchicalKMeansResult::clusterSizes() const {
    std::vector<std::size_t> sizes(assignments_.size(), 0);
    for (std::size_t i = 0; i < assignments_.size(); ++i) {
        sizes[i] = assignments_[i].size();
    }
    return sizes;
}

Centers HierarchicalKMeansResult::finalCentersFlat() const {
    if (finalCenters_.empty()) {
        return {};
    }
    std::size_t dim{finalCenters_[0].size()};
    Centers flatCenters(finalCenters_.size() * dim);
    for (std::size_t i = 0, id = 0; i < finalCenters_.size(); ++i, id += dim) {
        std::copy_n(&finalCenters_[i][0], dim, &flatCenters[id]);
    }
    return flatCenters;
}

std::vector<std::size_t> HierarchicalKMeansResult::assignmentsFlat() const {
    if (assignments_.empty()) {
        return {};
    }
    std::size_t n{std::accumulate(
        assignments_.begin(), assignments_.end(), 0UL,
        [](std::size_t sum, const std::vector<std::size_t>& cluster) {
            return sum + cluster.size();
        })};
    std::size_t dim{finalCenters_[0].size()};
    std::vector<std::size_t> flatAssignments(n);
    for (std::size_t i = 0, id = 0; i < assignments_.size(); ++i, id += dim) {
        for (std::size_t j : assignments_[i]) {
            flatAssignments[j] = id;
        }
    }
    return flatAssignments;
}

void HierarchicalKMeansResult::copyClusterPoints(std::size_t dim,
                                                 std::size_t cluster,
                                                 const Dataset& dataset,
                                                 Dataset& copy) const {
    
    std::size_t n{assignments_[cluster].size()};
    copy.resize(n * dim);
    auto* copy_{&copy[0]};
    for (std::size_t i : assignments_[cluster]) {
        std::size_t id{i * dim};
        std::copy_n(&dataset[id], dim, copy_);
        copy_ += dim;
    }
}

void HierarchicalKMeansResult::updateAssignmentsWithRecursiveSplit(std::size_t dim,
                                                                   std::size_t cluster,
                                                                   HierarchicalKMeansResult splitClusters) {

    std::size_t last{finalCenters_.size()};
    finalCenters_.resize(finalCenters_.size() + splitClusters.finalCenters_.size() - 1);
    finalCenters_[cluster] = splitClusters.finalCenters_[0];
    for (std::size_t i = 1; i < splitClusters.finalCenters_.size(); ++i) {
        finalCenters_[last + i - 1] = std::move(splitClusters.finalCenters_[i]);
    }
    auto clusterAssignments = assignments_[cluster];
    auto copyAssignments = [&clusterAssignments](const std::vector<std::size_t>& assignments) {
        std::vector<std::size_t> result(assignments.size());
        std::transform(assignments.begin(), assignments.end(),
                       result.begin(),
                       [&clusterAssignments](std::size_t point) {
                           return clusterAssignments[point];
                       });
        return result;
    };
    last = assignments_.size();
    assignments_.resize(assignments_.size() + splitClusters.assignments_.size() - 1);
    assignments_[cluster] = copyAssignments(splitClusters.assignments_[0]);
    for (std::size_t i = 1; i < splitClusters.assignments_.size(); ++i) {
        assignments_[last + i - 1] = copyAssignments(splitClusters.assignments_[i]);
    }
}

float HierarchicalKMeansResult::computeDispersion(std::size_t dim, const Dataset& dataset) const {
    float totalDispersion{0.0F};
    std::size_t n{0};
    for (std::size_t i = 0; i < assignments_.size(); ++i) {
        for (std::size_t j : assignments_[i]) {
            totalDispersion += distanceSq(dim, &dataset[j * dim], &finalCenters_[i][0]);
            ++n;
        }
    }
    return totalDispersion / n;
}

std::pair<float, float> HierarchicalKMeansResult::clusterSizeMoments() const {
    std::vector<std::size_t> sizes(clusterSizes());
    float mean{0.0F};
    for (std::size_t size : sizes) {
        mean += size;
    }
    mean /= sizes.size();
    float variance{0.0F};
    for (std::size_t size : sizes) {
        float diff{static_cast<float>(size) - mean};
        variance += diff * diff;
    }
    return {mean, std::sqrtf(variance / sizes.size())};
}

std::string HierarchicalKMeansResult::print() const {
    std::string result;
    result += "\nNumber of clusters: " + std::to_string(finalCenters_.size());
    auto moments = clusterSizeMoments();
    result += "\nCluster size moments: mean = " + std::to_string(moments.first) + " sd = " +
              std::to_string(moments.second);
    return result;
}
