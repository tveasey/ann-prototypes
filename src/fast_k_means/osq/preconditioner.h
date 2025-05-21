#include <cstddef>
#include <vector>

// The permutation matrix is used to assign components of the vectors to blocks.
// The permutation is chosen to equalize the variance of the blocks.
std::vector<std::vector<std::size_t>>
permutationMatrix(std::size_t dim,
                  const std::vector<std::size_t>& dimBlocks,
                  const std::vector<float>& vectors);

// Apply a random orthogonal transformation to the vectors. This comprises a
// permutation of the components followed by a blockwise random rotation.
void applyTransform(std::size_t dim,
                    const std::vector<std::vector<std::size_t>>& permutationMatrix,
                    const std::vector<std::vector<float>>& blocks,
                    const std::vector<std::size_t>& dimBlocks,
                    std::vector<float>& vectors);