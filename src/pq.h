#pragma once

#include <cstdint>
#include <functional>
#include <ostream>
#include <queue>
#include <random>
#include <set>
#include <utility>
#include <vector>

struct PQStats;

using code_t = std::uint8_t;
using loss_t = std::function<double(std::size_t,
                                    const std::vector<float>&,
                                    const std::vector<float>&,
                                    const std::vector<code_t>&)>;
enum Metric {
    Dot,
    Cosine
};

template<typename U, typename V>
std::ostream& operator<<(std::ostream& o, const std::pair<U, V>& pair) {
    o << "(" << pair.first << "," << pair.second << ")";
    return o;
}

template<typename T>
std::ostream& operator<<(std::ostream& o, const std::vector<T>& vector) {
    if (vector.empty()) {
        o << "[]";
        return o;
    }

    std::size_t n{vector.size() - 1};

    o << "[";
    for (std::size_t i = 0; i < n; ++i) {
        o << vector[i] << ",";
    }
    o << vector[n] << "]";
    return o;
}

int numBooks();

int bookSize();

std::size_t kMeansItr();

void zeroPad(std::size_t dim, std::vector<float>& vectors);

void normalise(std::size_t dim, std::vector<float>& vectors);

std::vector<float> norms2(std::size_t dim, std::vector<float>& vectors);

double quantisationMseLoss(std::size_t dim,
                           const std::vector<float>& codeBooks,
                           const std::vector<float>& docs,
                           const std::vector<code_t>& docsCodes);

double quantisationScannLoss(float t,
                             std::size_t dim,
                             const std::vector<float>& codeBooks,
                             const std::vector<float>& docs,
                             const std::vector<float>& docsNorms2,
                             const std::vector<code_t>& docsCodes);

std::set<std::size_t> initForgy(std::size_t numDocs,
                                std::minstd_rand& rng);

std::vector<float> initForgy(std::size_t dim,
                             const std::vector<float>& docs,
                             std::minstd_rand& rng);

void stepLloyd(std::size_t dim,
               const std::vector<float>& docs,
               std::vector<float>& centres,
               std::vector<code_t>& docsCodes);

void stepScann(float t,
               std::size_t dim,
               const std::vector<float>& docs,
               const std::vector<float>& docsNorms2,
               std::vector<float>& centres,
               std::vector<code_t>& docsCodes);

std::vector<float> sampleDocs(double sampleProbability,
                              std::size_t dim,
                              const std::vector<float>& docs,
                              std::minstd_rand& rng);

std::vector<float> initCodeBooks(std::size_t dim,
                                 const std::vector<float>& docs,
                                 const loss_t& loss);

std::pair<std::vector<float>, std::vector<code_t>>
buildCodeBook(std::size_t dim,
              double sampleProbability,
              const std::vector<float>& docs,
              std::size_t iterations,
              const loss_t& loss = quantisationMseLoss);

std::pair<std::vector<float>, std::vector<code_t>>
buildCodeBookScann(float t,
                   std::size_t dim,
                   double sampleProbability,
                   const std::vector<float>& docs,
                   const std::vector<float>& docsNorms2,
                   std::size_t iterations);

std::vector<float> buildDistTable(const std::vector<float>& codeBooks,
                                  const std::vector<float>& query);

std::vector<float> buildDistNorm2Table(const std::vector<float>& codeBooks,
                                       const std::vector<float>& query);

float computeDist(const std::vector<float>& distTable,
                  const code_t* docCode);

float computeNormedDist(const std::vector<float>& distTable,
                        const code_t* docCode);

std::vector<float> encoded(std::size_t dim,
                           const std::vector<float>& codeBooks,
                           const code_t* docCode);

void searchPQ(std::size_t k,
              const std::vector<float>& codeBooks,
              const std::vector<code_t>& docsCodes,
              const std::vector<float>& docsNorms,
              const std::vector<float>& query,
              bool normalise,
              std::priority_queue<std::pair<float, std::size_t>>& topk);

void searchBruteForce(std::size_t k,
                      const std::vector<float>& docs,
                      const std::vector<float>& query,
                      std::priority_queue<std::pair<float, std::size_t>>& topk);

void runPQBenchmark(const std::string& tag,
                    bool scann,
                    Metric metric,
                    std::size_t k,
                    std::size_t dim,
                    std::vector<float>& docs,
                    std::vector<float>& queries,
                    const std::function<void(const PQStats&)>& writeStats = [](const PQStats&) {});
