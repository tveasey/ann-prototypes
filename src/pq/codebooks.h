#pragma once

#include "types.h"
#include "../common/bigvector.h"

#include <utility>
#include <vector>

// Build a codebook for the docs vectors.
//
// Returns a pair of vectors. The first vector contains the codebook centres
// and the second vector contains the codes for each document. These are
// stored flat. The codebook centres comprise the first BOOK_SIZE centres
// each with dimension dim / numBooks. The codes contain numBooks codes for
// each document, i.e. the code of the closest centre to each doc.
std::pair<std::vector<float>, std::vector<code_t>>
buildCodebook(std::size_t dim, std::size_t numBooks, const std::vector<float>& docs);

// Update the codebook for the docs vectors.
std::pair<std::vector<float>, std::vector<code_t>>
updateCodebook(std::size_t dim,
               std::size_t numBooks,
               const std::vector<float>& docs,
               std::vector<float> codebookCentres);

// Write the encoding of a single document to the codes array.
void encode(const std::vector<float>& doc,
            const std::vector<float>& codebooks,
            std::size_t numBooks,
            code_t* codes);

// Write the anisotropic encoding of a single document to the codes array.
void anisotropicEncode(const std::vector<float>& doc,
                       const std::vector<float>& codebooksCentres,
                       std::size_t numBooks,
                       float threshold,
                       code_t* codes);
