#pragma once

#include <boost/test/tree/observer.hpp>

#include <string>
#include <iostream>

class TestObserver : public boost::unit_test::test_observer {
public:
    // Called at the start of each suite and at the start of each test case.
    void test_unit_start(const boost::unit_test::test_unit& test) override;

    // Called at the end of each suite and at the end of each test case.
    void test_unit_finish(const boost::unit_test::test_unit& test,
                          unsigned long elapsed) override;
};
