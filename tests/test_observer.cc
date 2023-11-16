#include "test_observer.h"

#include <boost/test/tree/test_unit.hpp>

void TestObserver::test_unit_start(const boost::unit_test::test_unit& test) {
    const std::string& unitName{test.full_name()};
    if (unitName.find('/') == std::string::npos) {
        // This gets called for suites as well as test cases - ignore these
        return;
    }
    std::cout << '+' << std::string(unitName.length() + 4, '-') << '+' << std::endl;
    std::cout << "|  " << unitName << "  |" << std::endl;
    std::cout << '+' << std::string(unitName.length() + 4, '-') << '+' << std::endl;
}

void TestObserver::test_unit_finish(const boost::unit_test::test_unit& test,
                                    unsigned long elapsed) {
    const std::string& unitName{test.full_name()};
    if (unitName.find('/') == std::string::npos) {
        // This gets called for suites as well as test cases - ignore these
        return;
    }
    std::cout << "Unit test timing - " << unitName << " took " << (elapsed / 1000) << "ms" << std::endl;
}
