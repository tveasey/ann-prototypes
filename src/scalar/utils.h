#include <cstddef>
#include <functional>
#include <tuple>
#include <utility>

std::tuple<float, float, double> maximize(std::function<double (float, float)> f,
                                          std::size_t nProbes,
                                          const std::pair<float, float>& xrange,
                                          const std::pair<float, float>& yrange);