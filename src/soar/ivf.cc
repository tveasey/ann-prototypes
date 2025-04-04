#include "ivf.h"

#include "../common/progress_bar.h"
#include "../common/utils.h"

#include <cstddef>
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {
std::vector<float> initKmeansPlusPlus(std::size_t dim,
                                      std::size_t numClusters,
                                      const std::vector<float>& docs,
                                      std::minstd_rand& rng) {

    // Use k-means++ initialisation for each subspace independently.
    std::size_t numDocs{docs.size() / dim};
    std::vector<float> centres(numClusters * dim);
    std::vector<int> selectedDocs(numClusters);
    std::vector<float> msds;

    auto centre = centres.begin();

    // Choose the first centroid uniformly at random.
    std::uniform_int_distribution<> u0n{0, static_cast<int>(numDocs) - 1};
    std::size_t selectedDoc{static_cast<std::size_t>(u0n(rng))};
    auto begin = docs.begin() + selectedDoc * dim;
    std::copy(begin, begin + dim, centre);
    centre += dim;

    msds.assign(numDocs, std::numeric_limits<float>::max());
    for (std::size_t i = 1; i < numClusters; ++i) {
        // Update the squared distance from each document to the nearest centroid.
        auto msd = msds.begin();
        auto lastCentre = centre - dim;
        for (auto doc = docs.begin(); doc != docs.end(); doc += dim, ++msd) {
            float sd{0.0F};
            #pragma omp simd reduction(+:sd)
            for (std::size_t j = 0; j < dim; ++j) {
                float dlj{doc[j] - lastCentre[j]};
                sd += dlj * dlj;
            }
            *msd = std::min(*msd, sd);
        }

        if (std::all_of(msds.begin(), msds.end(),
                        [](float msd) { return msd == 0.0F; })) {
            // If all msds are zero then pick a random document.
            selectedDoc = static_cast<std::size_t>(u0n(rng));
        } else {
            // Sample with probability proportional to the squared distance.
            std::discrete_distribution<std::size_t> discrete{msds.begin(), msds.end()};
            selectedDoc = discrete(rng);
        }

        begin = docs.begin() + selectedDoc * dim;
        std::copy(begin, begin + dim, centre);
        centre += dim;
    }

    return centres;
}

double stepLloyd(std::size_t dim,
                 std::size_t numClusters,
                 const std::vector<float>& docs,
                 std::vector<float>& centres,
                 std::vector<std::size_t>& docsCodes) {

    std::size_t numDocs{docs.size() / dim};
    docsCodes.resize(numDocs);
    std::vector<double> newCentres(numClusters * dim, 0.0);
    std::vector<std::size_t> centreCounts(numClusters, 0);

    std::size_t pos{0};
    double sumsd{0.0};
    for (auto doc = docs.begin(); doc != docs.end(); ++pos, doc += dim) {
        // Find the nearest centroid.
        std::size_t imsd{0};
        float msd{std::numeric_limits<float>::max()};
        for (std::size_t i = 0; i < numClusters; ++i) {
            float sd{0.0F};
            auto ci = &centres[i * dim];
            #pragma omp simd reduction(+:sd)
            for (std::size_t j = 0; j < dim; ++j) {
                float dij{doc[j] - ci[j]};
                sd += dij * dij;
            }
            if (sd < msd) {
                imsd = i;
                msd = sd;
            }
        }

        // Update the centroid.
        auto* newCentre = &newCentres[imsd * dim];
        for (std::size_t j = 0; j < dim; ++j) {
            newCentre[j] += doc[j];
        }
        ++centreCounts[imsd];

        // Encode the document.
        docsCodes[pos] = static_cast<std::size_t>(imsd);
        sumsd += msd;
    }

    for (std::size_t i = 0; i < centreCounts.size(); ++i) {
        if (centreCounts[i] > 0) {
            auto* centre = &centres[i * dim];
            auto* newCentre = &newCentres[i * dim];
            for (std::size_t j = 0; j < dim; ++j) {
                centre[j] = static_cast<float>(
                    newCentre[j] / static_cast<double>(centreCounts[i]));
            }
        }
    }

    return sumsd / static_cast<double>(numDocs);
}

void assignSpilledDocs(float lambda,
                      const std::vector<float>& docs,
                      const std::vector<float>& centres,
                      const std::vector<std::size_t>& docsCodes,
                      std::vector<std::size_t>& spilledDocsCodes) {

    // SOAR uses an adjusted distance for assigning spilled documents which is
    // given by:
    //
    //   d_soar(x, c) = ||x - c||^2 + lambda * ((x - c_1)^t (x - c))^2 / ||x - c_1||^2
    //
    // Here, x is the document, c is the nearest centroid, and c_1 is the first
    // centroid the document was assigned to. The document is assigned to the
    // cluster with the smallest d_soar(x, c).

    std::size_t dim{docs.size() / docsCodes.size()};
    std::size_t numClusters{centres.size() / dim};
    spilledDocsCodes.resize(docsCodes.size());
    
    std::vector<float> r1(dim);
    for (std::size_t i = 0; i < docsCodes.size(); ++i) {
        const auto* doc = &docs[i * dim];

        float n1{0.0F};
        auto c1 = &centres[docsCodes[i] * dim];
        #pragma omp simd reduction(+:n1)
        for (std::size_t j = 0; j < dim; ++j) {
            r1[j] = doc[j] - c1[j];
            n1 += r1[j] * r1[j];
        }
        
        std::size_t imdsoar{0};
        float mdsoar{std::numeric_limits<float>::max()};
        for (std::size_t j = 0; j < numClusters; ++j) {
            float sd{0.0F};
            float proj{0.0F};
            auto cj = &centres[j * dim];
            #pragma omp simd reduction(+:sd, proj)
            for (std::size_t k = 0; k < dim; ++k) {
                float djk{doc[k] - cj[k]};
                float projjk{r1[k] * djk};
                sd += djk * djk;
                proj += projjk;
            }
            float dsoar{sd + lambda * proj * proj / n1};
            if (dsoar < mdsoar) {
                imdsoar = j;
                mdsoar = dsoar;
            }
        }

        spilledDocsCodes[i] = imdsoar;
    }
}

} // unnamed::

SoarIVFIndex::SoarIVFIndex(Metric metric,
                           float lambda,
                           std::size_t dim,
                           std::size_t numClusters,
                           std::size_t numIterations) :
    metric_{metric}, lambda_{lambda}, dim_{dim},
    numClusters_(numClusters), numIterations_{numIterations} {
}

void SoarIVFIndex::build(const BigVector& docs) {
    std::cout << "Building IVF index for " << docs.numVectors() << " documents"
              << ", using " << numClusters_ << " clusters" << std::endl;

    // Just do everything in memory (we're not testing anything massive).
    docs_.clear();
    docs_.resize(docs.size());
    for (std::size_t i = 0; i < docs.numVectors(); ++i) {
        std::copy(docs[i].data(), docs[i].data() + dim_, &docs_[i * dim_]);
    }

    // This can work on a subset of the documents of size O(numClusters_) if we switch
    // to keeping the documents on disk.
    std::minstd_rand rng;
    auto diff = time([&] { centres_ = initKmeansPlusPlus(dim_, numClusters_, docs_, rng); });
    std::cout << "Initialised centres took " << diff.count() << " s" << std::endl;
    std::vector<std::size_t> docsCodes;
    std::unique_ptr<ProgressBar> progress{
        std::make_unique<ProgressBar>("Clustering...", numIterations_)};
    double sdlast{stepLloyd(dim_, numClusters_, docs_, centres_, docsCodes)};
    progress->update();
    maybeNormalizeCentres();
    for (int i = 0; i < numIterations_ - 1; ++i) {
        double sd{stepLloyd(dim_, numClusters_, docs_, centres_, docsCodes)};
        progress->update();
        maybeNormalizeCentres();
        if (sd > (1.0 - 1e-6) * sdlast) {
            break;
        }
        sdlast = sd;
    }
    progress.reset();

    // Do SOAR based spill assignments.
    std::vector<std::size_t> spilledDocsCodes;
    if (lambda_ > 0.0F) {
        diff = time([&]{
            assignSpilledDocs(lambda_, docs_, centres_, docsCodes, spilledDocsCodes);
        });
        std::cout << "Assigning spilled documents took " << diff.count() << " s" << std::endl;
    }

    // Fill in the clusters' documents.
    clustersDocs_.resize(numClusters_);
    for (std::size_t i = 0; i < docsCodes.size(); ++i) {
        clustersDocs_[docsCodes[i]].push_back(i);
    }
    for (std::size_t i = 0; i < spilledDocsCodes.size(); ++i) {
        clustersDocs_[spilledDocsCodes[i]].push_back(i);
    }

    // Improve cache locality for search. We should switch to sorting the documents
    // when we do things properly.
    for (auto& cluster : clustersDocs_) {
        std::sort(cluster.begin(), cluster.end());
    }
}

std::pair<std::vector<std::size_t>, std::size_t>
SoarIVFIndex::search(const float* query, std::size_t k, std::size_t numProbes) const {
    // Find the nearest numProbes clusters.
    std::priority_queue<std::pair<float, std::size_t>> nearest;
    for (std::size_t i = 0; i < numProbes; ++i) {
        nearest.emplace(std::numeric_limits<float>::max(),
                        std::numeric_limits<std::size_t>::max());
    }
    for (std::size_t i = 0; i < numClusters_; ++i) {
        float sim{0.0F};
        auto* centre = &centres_[i * dim_];
        #pragma omp simd reduction(+:sim)
        for (std::size_t j = 0; j < dim_; ++j) {
            sim += query[j] * centre[j];
        }
        float dist{1.0F - sim};
        if (dist < nearest.top().first) {
            nearest.pop();
            nearest.emplace(dist, i);
        }
    }
    std::vector<std::size_t> nearestClusters(numProbes);
    for (std::size_t i = 0; i < numProbes; ++i) {
        nearestClusters[i] = nearest.top().second;
        nearest.pop();
    }

    // Find the nearest k documents searching the nearest clusters.
    for (std::size_t i = 0; i < k; ++i) {
        nearest.emplace(std::numeric_limits<float>::max(),
                        std::numeric_limits<std::size_t>::max());
    }
    std::unordered_set<std::size_t> uniqueDocs;
    std::size_t numberOfComparisions{0};
    for (auto cluster : nearestClusters) {
        numberOfComparisions += clustersDocs_[cluster].size();
        for (auto i : clustersDocs_[cluster]) {
            if (uniqueDocs.find(i) != uniqueDocs.end()) {
                continue;
            }
            float sim{0.0F};
            const auto* doc = &docs_[i * dim_];
            #pragma omp simd reduction(+:sim)
            for (std::size_t j = 0; j < dim_; ++j) {
                sim += query[j] * doc[j];
            }
            float dist{1.0F - sim};
            if (dist < nearest.top().first) {
                uniqueDocs.erase(nearest.top().second);
                uniqueDocs.insert(i);
                nearest.pop();
                nearest.emplace(dist, i);
            }
        }
    }
    std::vector<std::size_t> nearestDocs(k);
    for (std::size_t i = 0; i < k; ++i) {
        nearestDocs[i] = nearest.top().second;
        nearest.pop();
    }

    return {std::move(nearestDocs), numberOfComparisions};
}

void SoarIVFIndex::maybeNormalizeCentres() {
    if (metric_ == Cosine) {
        normalize(dim_, centres_);
    }
}
