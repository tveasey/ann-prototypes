#include "../src/common/utils.h"

#include "../src/common/io.h"
#include "../src/common/utils.h"
#include "../src/scalar/scalar.h"

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

namespace {

BOOST_AUTO_TEST_SUITE(scalar)

BOOST_AUTO_TEST_CASE(testDot1B) {

    std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<std::uint32_t> u01{0, 1};
    std::size_t dim{100};
    std::vector<std::uint32_t> x(dim);
    std::vector<std::uint32_t> y(dim);
    
    for (std::size_t t = 0; t < 100; ++t) {
        // Generate some random binary vectors.
        std::generate(x.begin(), x.end(), [&] { return u01(rng); });
        std::generate(y.begin(), y.end(), [&] { return u01(rng); });

        auto expected = std::inner_product(x.begin(), x.end(), y.begin(), 0);

        // Pack into 32 bit integers.
        for (std::size_t i = 0; i < dim; /**/) {
            std::vector<std::uint32_t> packedX(dim / 32, 0);
            std::vector<std::uint32_t> packedY(dim / 32, 0);
            for (std::size_t j = 0; i < dim && j < 32; ++i, ++j) {
                packedX[i / 32] |= x[i] << j;
                packedY[i / 32] |= y[i] << j;
            }
        }
        std::uint32_t actual{dot1B(dim, x.data(), y.data())};

        BOOST_CHECK_EQUAL(expected, actual);
    }
}

BOOST_AUTO_TEST_CASE(testScalarQuantise1B) {

    std::mt19937_64 rng{std::random_device{}()};
    std::uniform_real_distribution<float> u010{0.0F, 10.0F};
    std::size_t dim{100};
    std::vector<float> x(dim);
    
    for (std::size_t t = 0; t < 100; ++t) {
        std::generate(x.begin(), x.end(), [&] { return u010(rng); });

        auto [xq, _] = scalarQuantise1B({2.5F, 7.5F}, dim, x);

        // Check each quantised component is rounding to the correct bucket centre.
        for (std::size_t i = 0; i < dim; /**/) {
            for (std::size_t j = 0; i < dim && j < 32; ++i, ++j) {
                std::uint32_t mask(1 << j);
                if (x[i] >= 5.0F) {
                    BOOST_CHECK_EQUAL(xq[i / 32] & mask, mask);
                } else {
                    BOOST_CHECK_EQUAL(xq[i / 32] & mask, 0);
                }
            }
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()

} // unnamed::