#include "common.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

namespace {
float INF{std::numeric_limits<float>::max()};

// --- Algorithm 3: Point-All-Ctrs ---
// Finds the closest and second closest centers for a point x_i.
// Updates a_i (assignment), u_i (upper bound to closest), l_i (lower bound to second closest).
void pointAllCtrs(std::size_t dim,
                  ConstPoint xi,
                  const Centers& centers,
                  std::size_t& ai,
                  float& ui,
                  float& li) {
    std::size_t k{centers.size()};

    float minDsq{INF};
    float secondMinDsq{INF};
    std::size_t bestJd{0};

    for (std::size_t jd = 0; jd < k; jd += dim) {
        float dsq{distanceSq(dim, xi, &centers[jd])};
        if (dsq < minDsq) {
            secondMinDsq = minDsq; // Old min becomes second min
            minDsq = dsq;
            bestJd = jd;
        } else if (dsq < secondMinDsq) {
            secondMinDsq = dsq;
        }
    }

    ai = bestJd;
    ui = std::sqrtf(minDsq);
    li = std::sqrtf(secondMinDsq);
}


// --- Algorithm 2: Initialize ---
// Initializes assignments, bounds, counts, and center sums.
void initialize(std::size_t dim,
                const Centers& centers,
                const Dataset& dataset,
                std::vector<std::size_t>& q,   // Out: counts per cluster
                Centers& cPrime,              // Out: vector sums per cluster
                std::vector<float>& u,         // Out: upper bounds for points
                std::vector<float>& l,         // Out: lower bounds for points
                std::vector<std::size_t>& a) { // Out: assignments for points

    std::size_t n{dataset.size()};
    std::size_t k{centers.size()};

    // Initialize counts and sums
    q.assign(k / dim, 0);
    cPrime.assign(k, 0.0F);

    // Resize point-specific vectors
    u.resize(n / dim);
    l.resize(n / dim);
    a.resize(n / dim);

    // Assign initial clusters and calculate initial bounds
    for (std::size_t i = 0, id = 0; id < n; ++i, id += dim) {
        ConstPoint xi{&dataset[id]}; // Reference to the current point
        pointAllCtrs(dim, xi, centers, a[i], u[i], l[i]);
        ++q[a[i] / dim];
        Point cPrimeAi{&cPrime[a[i]]}; // Reference to the center sum for this cluster
        #pragma omp simd
        for(std::size_t d = 0; d < dim; ++d) {
            cPrimeAi[d] += xi[d];
        }
    }
}

// --- Algorithm 4: Move-Centers ---
// Calculates new center locations based on sums (cPrime) and counts (q).
// Calculates how far each center moved (p).
// Resets cPrime and q for the next iteration's accumulation.
void moveCenters(std::size_t dim,
                 const Centers& cPrime,
                 const std::vector<std::size_t>& q,
                 Centers& centers,            // In/Out: centers (updated)
                 std::vector<float>& p) {     // Out: distances centers moved (squared)

    std::size_t k{centers.size()};
    p.resize(k / dim);

    for (std::size_t i = 0, id = 0; id < k; ++i, id += dim) {
        float dsq{0.0F};
        std::size_t qi{q[i]};
        if (qi > 0) {
            Point cId{&centers[id]};
            ConstPoint cPrimeId{&cPrime[id]};
            #pragma omp simd reduction(+:dsq)
            for(std::size_t d = 0; d < dim; ++d) {
                float cNew{cPrimeId[d] / qi};
                float diff{cNew - cId[d]};
                dsq += diff * diff;
                cId[d] = cNew;
            }
        }
        p[i] = std::sqrtf(dsq);
    }
}

// --- Algorithm 5: Update-Bounds ---
// Updates upper (u) and lower (l) bounds based on center movements (p).
void updateBounds(std::size_t dim,
                  const std::vector<float>& p,
                  const std::vector<std::size_t>& a,
                  std::vector<float>& u,       // In/Out: upper bounds
                  std::vector<float>& l) {     // In/Out: lower bounds

    std::size_t k{p.size()};

    // Find the largest and second largest center movements
    float maxP{-1.0F};
    float secondMaxP{-1.0F};
    std::size_t r{0}; // index of max movement
    for (std::size_t i = 0; i < k; ++i) {
        if (p[i] > maxP) {
            secondMaxP = maxP;
            maxP = p[i];
            r = i;
        } else if (p[i] > secondMaxP) {
            secondMaxP = p[i];
        }
    }

    // Update bounds for each point
    for (std::size_t i = 0; i < u.size(); ++i) {
        std::size_t ai{a[i] / dim};
        u[i] += p[ai];
        l[i] = std::max(l[i] - ((r == ai) ? secondMaxP : maxP), 0.0F);
    }
}

}

// --- Algorithm 1: K-means (Hamerly Variant) ---
KMeansResult kMeansHamerly(std::size_t dim,
                           const Dataset& dataset,
                           Centers initialCenters,
                           std::size_t k,
                           std::size_t maxIterations) {

    std::size_t n{dataset.size()};

    // Basic validation
    if (n == 0 || k == 0 || dim == 0 ||
        initialCenters.size() != static_cast<std::size_t>(k * dim)) {
        std::cerr << "Error: Invalid input to kMeansHamerly (dataset size, k, dimensions, or initial centers mismatch)." << std::endl;
        return {{}, {}, {0}, {}, 0, false};
    }
    if (initialCenters.size() % dim != 0) {
        std::cerr << "Error: Initial center dimension mismatch." << std::endl;
        return {{}, {}, {0}, {}, 0, false};
    }
    if (dataset.size() % dim != 0) {
        std::cerr << "Error: Data point dimension mismatch." << std::endl;
        return {{}, {}, {0}, {}, 0, false};
    }

    // --- Data Structures ---
    Centers centers{std::move(initialCenters)}; // Working copy of centers
    std::vector<std::size_t> a;        // Assignments a(i)
    std::vector<float> u;              // Upper bounds u(i) (squared distance)
    std::vector<float> l;              // Lower bounds l(i) (squared distance)
    std::vector<std::size_t> q;        // Counts q(j)
    Centers cPrime;                    // Sums c'(j)
    std::vector<float> p;              // Center movement p(j) (squared distance)
    std::vector<float> s;              // Dist to closest center s(j) (squared distance)
    s.resize(k);

    // --- Step 1: Initialize ---
    initialize(dim, centers, dataset, q, cPrime, u, l, a);

    // Re-calculate centers based on initial assignment (good practice)
    moveCenters(dim, cPrime, q, centers, p);

    if (k <= 1) {
        // Special case for k=1: assign all points to the first center
        return {k, std::move(centers), std::move(a), {}, 0, true};
    }

    // Bounds need initial update after first move
    updateBounds(dim, p, a, u, l);

    // --- Step 2: Main Loop ---
    bool converged{false};
    std::size_t iter{0};
    std::size_t kd{k * dim}; // Number of centers
    for (; iter < maxIterations; ++iter) {
        std::size_t switched{0};
        std::size_t recomputed{0};
        bool changed{false}; // Track if any assignment changed in this iteration

        // --- Step 3: Update s(j) - distance to closest *other* center ---
        s.assign(k, INF); // Reset s(j) to max value
        for (std::size_t i = 0, id = 0; id < kd; ++i, id += dim) {
            ConstPoint center{&centers[id]}; // Reference to the current center
            for (std::size_t j = 0, jd = 0; jd < id; ++j, jd += dim) {
                float dsq{distanceSq(dim, center, &centers[jd])};
                s[i] = std::min(s[i], dsq);
                s[j] = std::min(s[j], dsq);
            }
        }
        std::transform(s.begin(), s.end(), s.begin(),
                       [](float val) { return std::sqrtf(val); });

        // --- Step 5: Iterate through points ---
        for (std::size_t i = 0, id = 0; id < n; ++i, id += dim) {
            // --- Step 6: Calculate m ---
            float m{std::max(s[a[i] / dim] / 2.0F, l[i])}; // Using squared distances

            // --- Step 7: First Bound Test ---
            if (u[i] > m) {
                // --- Step 8: Tighten Upper Bound ---
                 // Recalculate distance to current assigned center
                u[i] = std::sqrtf(distanceSq(dim, &dataset[id], &centers[a[i]]));

                // --- Step 9: Second Bound Test ---
                if (u[i] > m) {
                    std::size_t aOld{a[i]}; // Store old assignment

                    // --- Step 11: Point-All-Ctrs ---
                    // Find true closest and second closest, update a(i), u(i), l(i)
                    pointAllCtrs(dim, &dataset[id], centers, a[i], u[i], l[i]);

                    std::size_t aNew{a[i]}; // New assignment

                    // --- Step 12 & 13: Check for change and update tracking ---
                    if (aOld != aNew) {
                        changed = true;
                        ++q[aNew / dim];
                        --q[aOld / dim];
                        Point cPrimeOld{&cPrime[aOld]};
                        Point cPrimeNew{&cPrime[aNew]};
                        ConstPoint xi{&dataset[id]}; // Reference to the current point
                        #pragma omp simd
                        for (std::size_t d = 0; d < dim; ++d) {
                            cPrimeNew[d] += xi[d];
                            cPrimeOld[d] -= xi[d];
                        }
                        ++switched; // Count how many points were changed
                    }
                    ++recomputed; // Count how many points were recomputed
                }
            }
        }

        //std::cout << "% switched: " << (static_cast<float>(switched * dim) / n) * 100.0F << std::endl;
        //std::cout << "% recomputed: " << (static_cast<float>(recomputed * dim) / n) * 100.0F << std::endl;

        // --- Check Convergence ---
        if (!changed) {
            converged = true;
            break; // Exit main loop
        }

        // --- Step 14: Move Centers ---
        moveCenters(dim, cPrime, q, centers, p);

        // --- Step 15: Update Bounds ---
        updateBounds(dim, p, a, u, l);

    } // End main loop (iterations)

    return {k, std::move(centers), std::move(a), {}, iter, converged};
}
