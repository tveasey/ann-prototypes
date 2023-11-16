#define BOOST_TEST_MODULE tests
// Defining BOOST_TEST_MODULE usually auto-generates main(), but we don't want
// this as we need custom initialisation to allow for output in both console and
// Boost.Test XML formats
#define BOOST_TEST_NO_MAIN

#include "test_observer.h"

#define BOOST_TEST_ALTERNATIVE_INIT_API
#include <boost/test/unit_test.hpp>

int main(int argc, char** argv) {
    TestObserver observer;
    boost::unit_test::framework::register_observer(observer);
    int result{boost::unit_test::unit_test_main(&init_unit_test, argc, argv)};
    boost::unit_test::framework::deregister_observer(observer);
    return result;
}
