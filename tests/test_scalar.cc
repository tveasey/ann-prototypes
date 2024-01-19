#include "../src/common/utils.h"

#include "../src/common/io.h"
#include "../src/common/utils.h"
#include "../src/scalar/scalar.h"
#include "../src/scalar/utils.h"

#include <boost/test/tools/old/interface.hpp>
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

namespace {

BOOST_AUTO_TEST_SUITE(scalar)

BOOST_AUTO_TEST_CASE(testQuantiles) {

    // Generate uniformly distributed vectors.
    std::mt19937_64 rng;
    std::uniform_real_distribution<float> u010{0.0F, 10.0F};
    std::size_t dim{100};
    std::vector<float> x(1000 * dim);
    std::generate(x.begin(), x.end(), [&] { return u010(rng); });

    // Check the quantiles are correct.
    auto [q1p5, q2p5] = quantiles(dim, x, 0.5F);
    BOOST_CHECK_CLOSE(q1p5, 2.5F, 1.0F);
    BOOST_CHECK_CLOSE(q2p5, 7.5F, 1.0F);
    auto [q1p9, q2p9] = quantiles(dim, x, 0.9F);
    BOOST_CHECK_CLOSE(q1p9, 0.5F, 1.0F);
    BOOST_CHECK_CLOSE(q2p9, 9.5F, 1.0F);
}

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

BOOST_AUTO_TEST_CASE(testDot4B) {

    std::minstd_rand rng{std::random_device{}()};
    std::uniform_int_distribution<> u{0, 15};

    for (std::size_t t = 0; t < 100; ++t) {
        std::vector<std::uint8_t> x(64);
        std::vector<std::uint8_t> y(x.size());
        std::vector<std::uint8_t> yp(x.size() / 2);
        std::generate_n(x.begin(), x.size(), [&] { return u(rng); });
        std::generate_n(y.begin(), y.size(), [&] { return u(rng); });

        std::uint32_t expectedDot{0};
        for (std::size_t i = 0; i < x.size(); ++i) {
            expectedDot += x[i] * y[i];
        }

        auto dot = dot4B(x.size(), x.data(), y.data());

        for (std::size_t i = 0; i < y.size(); i += 64) {
            pack4B(64, &y[i], &yp[i >> 1]);
        }

        auto dotp = dot4BP(x.size(), x.data(), yp.data());

        BOOST_REQUIRE_EQUAL(expectedDot, dot);
        BOOST_REQUIRE_EQUAL(expectedDot, dotp);
    }

    // Remainder.
    for (std::size_t t = 0; t < 100; ++t) {
        std::vector<std::uint8_t> x(79);
        std::vector<std::uint8_t> y(x.size());
        std::vector<std::uint8_t> yp((x.size() + 1) / 2);
        std::generate_n(x.begin(), x.size(), [&] { return u(rng); });
        std::generate_n(y.begin(), y.size(), [&] { return u(rng); });

        std::uint32_t expectedDot{0};
        for (std::size_t i = 0; i < x.size(); ++i) {
            expectedDot += x[i] * y[i];
        }

        auto dot = dot4B(x.size(), x.data(), y.data());

        std::size_t remainder{x.size() & 0x3F};
        std::size_t n{x.size() - remainder};
        for (std::size_t i = 0; i < n; i += 64) {
            pack4B(64, &y[i], &yp[i >> 1]);
        }
        pack4B(remainder, &y[n], &yp[n >> 1]);

        auto dotp = dot4BP(x.size(), x.data(), yp.data());

        BOOST_REQUIRE_EQUAL(expectedDot, dot);
        BOOST_REQUIRE_EQUAL(expectedDot, dotp);
    }

    // Long.
    for (std::size_t t = 0; t < 100; ++t) {
        std::vector<std::uint8_t> x(5030);
        std::vector<std::uint8_t> y(x.size());
        std::vector<std::uint8_t> yp((x.size() + 1) / 2);
        std::generate_n(x.begin(), x.size(), [&] { return u(rng); });
        std::generate_n(y.begin(), y.size(), [&] { return u(rng); });

        std::uint32_t expectedDot{0};
        for (std::size_t i = 0; i < x.size(); ++i) {
            expectedDot += x[i] * y[i];
        }

        auto dot = dot4B(x.size(), x.data(), y.data());

        std::size_t remainder{x.size() & 0x3F};
        std::size_t n{x.size() - remainder};
        for (std::size_t i = 0; i < n; i += 64) {
            pack4B(64, &y[i], &yp[i >> 1]);
        }
        pack4B(remainder, &y[n], &yp[n >> 1]);

        auto dotp = dot4BP(x.size(), x.data(), yp.data());

        BOOST_REQUIRE_EQUAL(expectedDot, dot);
        BOOST_REQUIRE_EQUAL(expectedDot, dotp);
    }
}

BOOST_AUTO_TEST_CASE(testDot8B) {

    std::minstd_rand rng{std::random_device{}()};
    std::uniform_int_distribution<> u{0, 15};

    for (std::size_t t = 0; t < 100; ++t) {
        std::vector<std::uint8_t> x(128);
        std::vector<std::uint8_t> y(x.size());
        std::generate_n(x.begin(), x.size(), [&] { return u(rng); });
        std::generate_n(y.begin(), y.size(), [&] { return u(rng); });

        std::uint32_t expectedDot{0};
        for (std::size_t i = 0; i < x.size(); ++i) {
            expectedDot += x[i] * y[i];
        }

        auto dot = dot8B(x.size(), x.data(), y.data());

        BOOST_REQUIRE_EQUAL(expectedDot, dot);
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

        const std::uint32_t* xq32{reinterpret_cast<const std::uint32_t*>(xq.data())};

        // Check each quantised component is rounding to the correct bucket centre.
        for (std::size_t i = 0; i < dim; /**/) {
            for (std::size_t j = 0; i < dim && j < 32; ++i, ++j) {
                std::uint32_t mask(1 << j);
                if (x[i] >= 5.0F) {
                    BOOST_CHECK_EQUAL(xq32[i / 32] & mask, mask);
                } else {
                    BOOST_CHECK_EQUAL(xq32[i / 32] & mask, 0);
                }
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(test4BPacking) {

    std::minstd_rand rng{std::random_device{}()};
    std::uniform_int_distribution<> u{0, 15};

    for (std::size_t t = 0; t < 100; ++t) {
        std::vector<std::uint8_t> x(77);
        std::vector<std::uint8_t> xp((x.size() + 1) / 2);
        std::generate_n(x.begin(), x.size(), [&] { return u(rng); });

        std::size_t rem{x.size() & 0x1F};
        std::size_t dim{x.size() - rem};
        for (std::size_t i = 0; i < x.size(); i += 64) {
            pack4B(64, &x[i], &xp[i >> 1]);
        }
        pack4B(rem, &x[dim], &xp[dim >> 1]);

        std::vector<std::uint8_t> xu(x.size());
        for (std::size_t i = 0; i < x.size(); i += 64) {
            unpack4B(64, &xp[i >> 1], &xu[i]);
        }
        unpack4B(rem, &xp[dim >> 1], &xu[dim]);

        BOOST_REQUIRE_EQUAL(x, xu);
    }
}

BOOST_AUTO_TEST_CASE(test_maximize) {

    auto pi = std::acos(-1.0);
    auto f = [&](float x, float y) {
        return (2.0 + std::cos(x - 1.0) + std::cos(y - 1.0)) /
               (1.0 + std::abs(x - 1.0) + std::abs(y - 1.0));
    };

    auto [x, y, fbest] = maximize(f, 24, {-3.0F, 5.0F}, {-3.0F, 5.0F});
    BOOST_REQUIRE_CLOSE(x, 1.0F, 3.0F);
    BOOST_REQUIRE_CLOSE(y, 1.0F, 3.0F);

    std::minstd_rand rng;
    std::uniform_real_distribution<float> u{-3.0F, 5.0F};
    for (std::size_t t = 0; t < 10; ++t) {
        double fmax{std::numeric_limits<double>::lowest()};
        for (std::size_t i = 0; i < 24; ++i) {
            auto xi = u(rng);
            auto yi = u(rng);
            auto fi = f(xi, yi);
            if (fi > fmax) {
                fmax = fi;
            }
        }
        BOOST_REQUIRE_LT(fmax, fbest);
    }
}

BOOST_AUTO_TEST_SUITE_END()

} // unnamed::