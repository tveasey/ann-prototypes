#include <cstdint>
#include <ostream>
#include <queue>
#include <random>
#include <set>
#include <utility>
#include <vector>

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

void zeroPad(std::size_t dim, std::vector<float>& vectors);

void normalize(std::size_t dim, std::vector<float>& vectors);

std::set<std::size_t> initForgy(std::size_t k,
                                std::size_t numDocs,
                                std::minstd_rand& rng);

std::vector<float> initForgy(std::size_t k,
                             std::size_t bookDim,
                             std::size_t dim,
                             const std::vector<float>& docs,
                             std::minstd_rand& rng);

void stepLloyd(std::size_t numBooks,
               std::size_t bookSize,
               std::size_t dim,
               const std::vector<float>& docs,
               std::vector<float>& centres,
               std::vector<std::int8_t>& docsCodes);

float computeDispersion(std::size_t dim,
                        const std::vector<float>& centres,
                        const std::vector<float>& docs,
                        const std::vector<std::int8_t>& docsCentres);

std::pair<std::vector<float>, std::vector<std::int8_t>>
buildCodeBook(std::size_t dim,
              const std::vector<float>& docs);

std::vector<float> buildDistTable(const std::vector<float>& codeBooks,
                                  const std::vector<float>& query);

float computeDist(const std::vector<float>& distTable,
                  const std::int8_t* docCode);

void searchPQ(std::size_t k,
              const std::vector<float>& codeBooks,
              const std::vector<std::int8_t>& docsCodes,
              const std::vector<float>& query,
              std::priority_queue<std::pair<float, std::size_t>>& topk);

void searchBruteForce(std::size_t k,
                      const std::vector<float>& docs,
                      const std::vector<float>& query,
                      std::priority_queue<std::pair<float, std::size_t>>& topk);

void runPQBenchmark(std::size_t dim,
                    std::vector<float>& docs,
                    std::vector<float>& queries);
