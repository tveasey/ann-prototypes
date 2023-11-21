#include "codebooks.h"

#include "constants.h"
#include "clustering.h"
#include "types.h"
#include "utils.h"
#include "../common/utils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

namespace {

std::vector<float> initCodebookCentres(std::size_t dim,
                                       const std::vector<float>& docs) {

    // Random restarts with aggressive downsample.

    std::vector<float> minMseCodebookCentres;
    double minMse{std::numeric_limits<double>::max()};

    std::minstd_rand rng;

    // Using 32 vectors per centroid is enough to get reasonable estimates.
    std::size_t numDocs{docs.size() / dim};
    double p{std::min(32 * BOOK_SIZE / static_cast<double>(numDocs), 1.0)};
    auto sampledDocs = sampleDocs(dim, docs, p, rng);

    std::vector<code_t> docsCodes(NUM_BOOKS * numDocs);
    for (std::size_t restart = 0;
         restart < BOOK_CONSTRUCTION_K_MEANS_RESTARTS;
         ++restart) {
        std::vector<float> codebookCentres{
            initKmeansPlusPlusForBookConstruction(dim, sampledDocs, rng)};
        // K-means converges pretty quickly so we use half the number
        // of iterations we do for final clustering to select the best
        // initialization.
        double mse;
        for (std::size_t i = 0; 2 * i < BOOK_CONSTRUCTION_K_MEANS_ITR; ++i) {
            mse = stepLloydForBookConstruction(dim, sampledDocs, codebookCentres, docsCodes);
        }
        if (mse < minMse) {
            minMseCodebookCentres = std::move(codebookCentres);
            minMse = mse;
        }
    }

    return minMseCodebookCentres;
}

} // unnamed::

std::pair<std::vector<float>, std::vector<code_t>>
buildCodebook(std::size_t dim, const std::vector<float>& docs) {
    std::size_t bookDim{dim / NUM_BOOKS};
    std::vector<float> codebookCentres{initCodebookCentres(dim, docs)};
    std::vector<code_t> docsCodes(docs.size() / bookDim);
    double lastMse{std::numeric_limits<double>::max()};
    for (std::size_t i = 0; i < BOOK_CONSTRUCTION_K_MEANS_ITR; ++i) {
        double mse{stepLloydForBookConstruction(dim, docs, codebookCentres, docsCodes)};
        if (std::abs(lastMse - mse) < 1e-4 * mse) {
            break;
        }
        lastMse = mse;
    }
    return std::make_pair(std::move(codebookCentres), std::move(docsCodes));
}

void encode(const std::vector<float>& doc,
            const std::vector<float>& codebooksCentres,
            code_t* codes) {

    // For each book, find the closest centroid and append the corresponding
    // code for the book.
    std::size_t dim{doc.size()};
    std::size_t bookDim{dim / NUM_BOOKS};
    for (std::size_t b = 0; b < NUM_BOOKS; ++b) {
        std::size_t iMsd{0};
        float msd{std::numeric_limits<float>::max()};
        auto docProj = doc.begin() + b * bookDim;
        auto codebookCentres = codebooksCentres.begin() + b * bookDim * BOOK_SIZE;
        for (std::size_t i = 0; i < BOOK_SIZE; ++i) {
            float sd{0.0F};
            auto codebookCentre = codebookCentres + i * bookDim;
            #pragma clang loop unroll_count(4) vectorize(assume_safety)
            for (std::size_t j = 0; j < bookDim; ++j) {
                float dij{docProj[j] - codebookCentre[j]};
                sd += dij * dij;
            }
            if (sd < msd) {
                iMsd = i;
                msd = sd;
            }
        }
        codes[b] = static_cast<code_t>(iMsd);
    }
}

void anisotropicEncode(const std::vector<float>& doc,
                       const std::vector<float>& codebooksCentres,
                       float threshold,
                       code_t* codes) {

    // For each book, find the closest centroid and append the corresponding
    // code for the book.
    //
    // To measure the distance between a vector and a centroid, we use the
    // an anisotropic distance function which is the sum of the squared
    // parallel and perpendicular distances. The parallel distance is scaled
    // based on the threshold model proposed in https://arxiv.org/pdf/1908.10396.pdf.
    // Specifically, the threshold T is used to compute the parallel distance
    // scale factor as:
    //
    //   T^2 * / (1 - T^2) * (d - 1)
    //
    // Note that the distance function separates into independent functions
    // over each codebook. This means we can compute the minimum distance by
    // finding the minimum distance for each codebook independently.
    //
    // Specifically, if |x|^2 denotes the squared L2 norm of doc vector x,
    // which is decomposed into subspaces as [x_1, ..., x_n], and y_{i,j} is
    // the j'th centre of the i'th codebook, then the distance function we
    // wish to minimize when selecting each j is:
    //
    //   d_i = |x_i - y_{i,j}|^2 + (n - 1) / |x|^2 * (x^t (y_{i,j} - x_i))^2
    //
    // Here, n is parallelScale.

    std::size_t dim{doc.size()};
    std::size_t bookDim{dim / NUM_BOOKS};
    float n2{norms2(dim, doc)[0]};
    float scale{threshold * threshold / (1.0F - threshold * threshold) * (dim - 1.0F)};

    for (std::size_t b = 0; b < NUM_BOOKS; ++b) {
        std::size_t iMsd{0};
        float msd{std::numeric_limits<float>::max()};
        auto docProj = doc.begin() + b * bookDim;
        auto codebookCentres = codebooksCentres.begin() + b * bookDim * BOOK_SIZE;
        for (std::size_t i = 0; i < BOOK_SIZE; ++i) {
            float sd{0.0F};
            auto codebookCentre = codebookCentres + i * bookDim;
            #pragma clang loop unroll_count(4) vectorize(assume_safety)
            for (std::size_t j = 0; j < bookDim; ++j) {
                float dij{docProj[j] - codebookCentre[j]};
                sd += dij * dij;
            }
            float parallel{0.0F};
            #pragma clang loop unroll_count(4) vectorize(assume_safety)
            for (std::size_t j = 0; j < bookDim; ++j) {
                parallel += docProj[j] * (codebookCentre[j] - docProj[j]);
            }
            sd += (scale - 1.0F) * parallel * parallel / n2;
            if (sd < msd) {
                iMsd = i;
                msd = sd;
            }
        }
        codes[b] = static_cast<code_t>(iMsd);
    }
}