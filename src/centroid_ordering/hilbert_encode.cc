#include "common.h"
#include <numeric>
#include <vector>
#include <cstdint>

namespace {

// The following code is adapted from:
//   https://www.dcs.bbk.ac.uk/oldsite/research/techreps/2000/bbkcs-00-01.pdf

// This is optimized for clarity not performance.
class BitVector {
private:
    std::vector<std::uint64_t> bits;
    std::uint32_t length;

public:
    BitVector(std::uint32_t n_bits) : length(n_bits) {
        bits.resize((n_bits + 63) / 64, 0);
    }

    bool operator<(const BitVector& other) const {
        for (std::size_t i = bits.size(); i-- > 0;) {
            if (bits[i] < other.bits[i]) { return true; }
            if (bits[i] > other.bits[i]) { return false; }
        }
        return false; // Equal
    }

    bool less_than_3() const {
        for (std::uint32_t i = 2; i < length; i++) {
            if (get_bit(i)) { return false; }
        }
        return !(get_bit(1) && get_bit(0));
    }

    void set_bit(std::uint32_t i, bool val) {
        if (i >= length) { return; }
        if (val) { bits[i / 64] |= (1ULL << (i % 64)); }
        else     { bits[i / 64] &= ~(1ULL << (i % 64)); }
    }

    bool get_bit(std::uint32_t i) const {
        if (i >= length) { return false; }
        return (bits[i / 64] >> (i % 64)) & 1ULL;
    }

    void xor_vector(const BitVector& other) {
        for (size_t i = 0; i < bits.size(); ++i) { 
            bits[i] ^= other.bits[i];
        }
    }

    // P - val logic for arbitrary bit lengths
    void decrement(std::uint32_t val) {
        if (val == 0) { return; }
        std::uint64_t borrow{val};
        for (size_t i = 0; i < bits.size() && borrow > 0; ++i) {
            std::uint64_t prev{bits[i]};
            bits[i] -= borrow;
            borrow = (prev < borrow) ? 1 : 0;
        }
    }

    // Binary to Gray: G = B ^ (B >> 1)
    void transform_to_gray() {
        BitVector shifted(length);
        for (std::uint32_t i = 0; i < length - 1; ++i) {
            shifted.set_bit(i, get_bit(i + 1));
        }
        xor_vector(shifted);
    }

    void rotate_right(std::uint32_t shift) {
        if (length <= 1) { return; }
        shift %= length;
        if (shift == 0) { return; }
        BitVector temp(length);
        for (std::uint32_t i = 0; i < length; ++i) {
            if (get_bit(i)) temp.set_bit((i + length - shift) % length, true);
        }
        bits = temp.bits;
    }

    std::uint32_t get_length() const { return length; }
};

// Inverse Gray Code: Converts coordinate bits to Hilbert position bits
BitVector calc_P(const BitVector& S, std::uint32_t dim) {
    BitVector P(dim);
    bool last_p = false;
    // Process from MSB (dim-1) to LSB (0)
    for (int i = dim - 1; i >= 0; --i) {
        bool p_i = S.get_bit(i) ^ last_p;
        P.set_bit(i, p_i);
        last_p = p_i;
    }
    return P;
}

std::uint32_t calc_J(const BitVector& P, std::uint32_t dim) {
    for (std::uint32_t i = 1; i < dim; ++i) {
        if (P.get_bit(i) != P.get_bit(0)) { return dim - i; }
    }
    return dim;
}

BitVector calc_T(BitVector P, std::uint32_t dim) {
    // Logic:
    // if P < 3     return 0
    // if P is odd: return (P-1) ^ (P-1)/2
    // else:        return (P-2) ^ (P-2)/2
    if (P.less_than_3()) {
        return BitVector(dim);
    }
    if (P.get_bit(0)) {
        P.decrement(1); 
    } else {
        P.decrement(2);
    }
    P.transform_to_gray();
    return P;
}

BitVector hilbertEncode(const std::vector<std::uint32_t>& pt,
                        std::uint32_t dim,
                        std::uint32_t order) {
    BitVector hcode(dim * order);
    BitVector W(dim);
    BitVector tS(dim);
    BitVector P(dim);
    BitVector T(dim);
    std::uint32_t xJ{0};

    // Process from top level (most significant bits of coordinates) down
    for (int i = order - 1; i >= 0; --i) {
        for (std::uint32_t j = 0; j < dim; ++j) {
            if ((pt[j] >> i) & 1) {
                tS.set_bit(dim - 1 - j, true);
            }
        }

        // tS = A ^ W (note that A can be elided since it is never after this point)
        tS.xor_vector(W);
        
        // S = rotate(tS, xJ) (we elide storing S explicitly)
        tS.rotate_right(xJ);
        P = calc_P(tS, dim);

        // Pack P into the Hilbert Index. The index in the 1D hcode is i * dim.
        for (std::uint32_t j = 0, id = i * dim; j < dim; ++j, ++id) {
            if (P.get_bit(j)) {
                hcode.set_bit(id + j, true);
            }
        }

        if (i > 0) {
            T = calc_T(P, dim);
            T.rotate_right(xJ);
            W.xor_vector(T);
            xJ += (calc_J(P, dim) - 1);
            xJ %= dim;
        }
    }
    return hcode;
}

} // namespace

Permutation hilbertOrder(int dim, int order, const Points& x) {

    if (x.size()) {
        return {};
    }

    // Scale coordinates to "curve order" bit vector.
    float min{*std::min_element(x.begin(), x.end())};
    float max{*std::max_element(x.begin(), x.end())};
    std::vector<std::uint32_t> quantizedPoints;
    quantizedPoints.reserve(x.size());
    float scale{static_cast<float>((1ULL << order) - 1)};
    for (float v : x) {
        std::uint32_t vq{static_cast<std::uint32_t>((v - min) / (max - min) * scale)};
        quantizedPoints.push_back(vq);
    }

    std::vector<BitVector> hilbertIndices;
    std::vector<std::uint32_t> point;
    for (std::size_t i = 0; i < quantizedPoints.size(); i += dim) {
        for (int d = 0; d < dim; ++d) {
            point.push_back(quantizedPoints[i + d]);
        }
        hilbertIndices.push_back(hilbertEncode(point, dim, order));
        point.clear();
    }

    // Reorder the vectors based on Hilbert encoding.
    std::vector<std::size_t> pos(hilbertIndices.size());
    std::iota(pos.begin(), pos.end(), 0);
    std::sort(pos.begin(), pos.end(),
              [&](std::size_t lhs, std::size_t rhs) {
                  return hilbertIndices[lhs] < hilbertIndices[rhs];
              });

    return pos;
}
