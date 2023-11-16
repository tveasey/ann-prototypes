#pragma once

#include "types.h"
#include "../common/bigvector.h"

#include <utility>
#include <vector>

// Build a codebook for the docs vectors.
//
// Retruns a pair of vectors. The first vector contains the codebook centres
// and the second vector contains the codes for each document. These are
// stored flat. The codebook centres comprise the first BOOK_SIZE centres
// each with dimension dim / NUM_BOOKS. The codes contain NUM_BOOKS codes
// for each document, i.e. the code of the closest centre to each doc.
std::pair<std::vector<float>, std::vector<code_t>>
buildCodebook(std::size_t dim, const std::vector<float>& docs);

// Write the encoding of a single document to the codes array.
void writeEncoding(const std::vector<float>& doc,
                   const std::vector<float>& codebooks,
                   code_t* codes);
