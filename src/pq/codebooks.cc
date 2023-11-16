#include "codebooks.h"

#include "constants.h"
#include "clustering.h"
#include "types.h"
#include "utils.h"
#include "../common/utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

double quantisationMseLoss(std::size_t dim,
                           const std::vector<float>& codebooksCentres,
                           const std::vector<float>& docs,
                           const std::vector<code_t>& docsCodes) {

    std::size_t bookDim{dim / NUM_BOOKS};

    double totalMse{0.0};
    std::size_t count{0};

    auto code = docsCodes.begin();
    for (auto doc = docs.begin(); doc != docs.end(); ++count) {
        float mse{0.0F};
        for (std::size_t b = 0; b < NUM_BOOKS; ++b, ++code, doc += bookDim) {
            #pragma clang loop unroll_count(4) vectorize(assume_safety)
            for (std::size_t i = 0; i < bookDim; ++i) {
                float di{doc[i] - codebooksCentres[(BOOK_SIZE * b + *code) * bookDim + i]};
                mse += di * di;
            }
        }
        totalMse += mse;
    }
    return totalMse / static_cast<double>(count);
}

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

std::pair<std::vector<float>, std::vector<code_t>>
buildCodebook(std::size_t dim, const std::vector<float>& docs) {
    std::size_t bookDim{dim / NUM_BOOKS};
    std::vector<float> codebookCentres{initCodebookCentres(dim, docs)};
    std::vector<code_t> docsCodes(docs.size() / bookDim);
    for (std::size_t i = 0; i < BOOK_CONSTRUCTION_K_MEANS_ITR; ++i) {
        stepLloydForBookConstruction(dim, docs, codebookCentres, docsCodes);
    }
    return std::make_pair(std::move(codebookCentres), std::move(docsCodes));
}

void writeEncoding(const std::vector<float>& doc,
                   const std::vector<float>& codebooksCentres,
                   code_t* codes) {

    // For each book, find the closest centroid and append the corresponding
    // code for the book.
    std::size_t dim{doc.size()};
    for (std::size_t b = 0; b < NUM_BOOKS; ++b) {
        std::size_t bookDim{dim / NUM_BOOKS};
        std::size_t bookSize{BOOK_SIZE * bookDim};
        std::size_t bookOffset{b * bookSize};
        std::size_t bookCodeOffset{b * BOOK_SIZE};
        std::size_t iMinSd{0};
        float minSd{std::numeric_limits<float>::max()};
        auto docProj = doc.begin() + dim * b;
        auto codebookCentre = codebooksCentres.begin() + bookOffset;
        for (std::size_t c = 0; c < bookSize; c += bookDim) {
            float sd{0.0F};
            #pragma clang loop unroll_count(4) vectorize(assume_safety)
            for (std::size_t i = 0; i < bookDim; ++i) {
                float di{docProj[i] - codebookCentre[c + i]};
                sd += di * di;
            }
            if (sd < minSd) {
                iMinSd = c;
                minSd = sd;
            }
        }
        codes[b] = static_cast<code_t>(iMinSd / bookDim);
    }
}
