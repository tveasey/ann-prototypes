#include "common.h"

#include <algorithm>
#include <cassert> // For basic error checking
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
    if (k == 0) {
        return; // No centers
    }

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
    ui = minDsq; // Use squared distance for bounds internally

    // Handle k=1 case where there's no second closest center
    li = (k > dim) ? secondMinDsq : 0.0F;
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

    if (n == 0 || k == 0) {
         // Handle empty input if necessary
         std::cerr << "Warning: Empty dataset, centers, or zero dimension in Initialize." << std::endl;
         q.assign(k / dim, 0);
         cPrime.assign(k, 0.0F);
         u.resize(n / dim);
         l.resize(n / dim);
         a.resize(n / dim);
         return;
    }

    // Initialize counts and sums
    q.assign(k / dim, 0);
    cPrime.assign(k, 0.0F);

    // Resize point-specific vectors
    u.resize(n / dim);
    l.resize(n / dim);
    a.resize(n / dim);

    // Assign initial clusters and calculate initial bounds
    for (std::size_t i = 0, id = 0; id < n; id += dim, ++i) {
        assert(id < dataset.size() && i < a.size() && i < u.size() && i < l.size());
        ConstPoint xi{&dataset[id]}; // Reference to the current point
        pointAllCtrs(dim, xi, centers, a[i], u[i], l[i]);
        assert(a[i] / dim < q.size() && a[i] < cPrime.size() - dim);
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
    if (k == 0) {
        return;
    }

    p.resize(k / dim);

    std::vector<float> cOld(dim, 0.0F); // Temporary storage for old center
    for (std::size_t i = 0, id = 0; id < k; ++i, id += dim) {
        std::copy(&centers[id], &centers[id] + dim, &cOld[0]); // Copy old center
        if (q[i] > 0) {
            for(std::size_t d = 0; d < dim; ++d) {
                centers[id + d] = cPrime[id + d] / q[i];
            }
            p[i] = distanceSq(dim, &cOld[0], &centers[id]);
        } else {
            // Handle empty cluster: keep center as is, movement is 0
            // Alternative strategies exist (e.g., re-initialize randomly,
            // or assign the point farthest from its center), but keeping
            // it simple here.
            p[i] = 0.0F;
        }
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
    if (k == 0) {
        return;
    }

    // Find the largest (r) and second largest (r_prime) center movements
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

    // Ensure secondMaxP is non-negative if initialized to -1
    secondMaxP = std::max(secondMaxP, 0.0F);

    // Update bounds for each point
    for (std::size_t i = 0; i < u.size(); ++i) {
        // Update upper bound (using sqrt because p is squared, but u/l are also squared internally)
        // Correction: p, u, l are all squared distances, so add directly.
        std::size_t ai{a[i] / dim}; // assignment of point i
        u[i] += p[ai];

        // Update lower bound
        if (r == ai) {
            // Subtract second largest movement if point belongs to fastest moving cluster
            // Note: We stored squared distances. Subtraction is valid if we assume
            // the triangle inequality holds approx. for squared distances too (less strict),
            // or if we interpret bounds as squared bounds. Let's stick to squared bounds.
            l[i] -= secondMaxP;
        } else {
            // Subtract largest movement otherwise
            l[i] -= maxP;
        }
        // Ensure lower bound doesn't become negative (can happen due to approximations)
        if (l[i] < 0.0F) {
            l[i] = 0.0F;
        }
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
        return {{}, {}, {0}, 0, false};
    }
    if (initialCenters.size() % dim != 0) {
        std::cerr << "Error: Initial center dimension mismatch." << std::endl;
        return {{}, {}, {0}, 0, false};
    }
    if (dataset.size() % dim != 0) {
        std::cerr << "Error: Data point dimension mismatch." << std::endl;
        return {{}, {}, {0},  0, false};
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

    if (k == 1) {
        // Special case for k=1: assign all points to the first center
        return {k, std::move(centers), std::move(a), 0, true};
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
        for (std::size_t i = 0, id = 0; id < kd; ++i, id += dim) {
            assert(id < centers.size() && i < s.size() - dim);
            s[i] = INF;
            for (std::size_t iPrime = 0, idPrime = 0; idPrime < id; ++iPrime, idPrime += dim) {
                assert(idPrime < centers.size() - dim);
                float dsq{distanceSq(dim, &centers[id], &centers[idPrime])};
                if (dsq < s[i]) {
                    s[i] = dsq;
                } else if (dsq < s[iPrime]) {
                    s[iPrime] = dsq;
                }
            }
            // Handle k=1 case
            if (k <= 1) {
               s[i] = 0.0F;
            }
        }

        // --- Step 5: Iterate through points ---
        for (std::size_t i = 0, id = 0; id < n; ++i, id += dim) {
            // --- Step 6: Calculate m ---
            float m{std::max(s[a[i] / dim] / 2.0F, l[i])}; // Using squared distances

            // --- Step 7: First Bound Test ---
            if (u[i] > m) {
                // --- Step 8: Tighten Upper Bound ---
                 // Recalculate distance to current assigned center
                u[i] = distanceSq(dim, &dataset[id], &centers[a[i]]);

                // --- Step 9: Second Bound Test ---
                if (u[i] > m) {
                    std::size_t a_old{a[i]}; // Store old assignment

                    // --- Step 11: Point-All-Ctrs ---
                    // Find true closest and second closest, update a(i), u(i), l(i)
                    pointAllCtrs(dim, &dataset[id], centers, a[i], u[i], l[i]);

                    std::size_t a_new{a[i]}; // New assignment

                    // --- Step 12 & 13: Check for change and update tracking ---
                    if (a_old != a_new) {
                        changed = true;
                        // We're moving a point from cluster a_old to a[i].
                        // Update counts and vector sums.
                        ++q[a_new / dim];
                        --q[a_old / dim];
                        for (std::size_t d = 0; d < dim; ++d) {
                            cPrime[a_new + d] += dataset[id + d];
                            cPrime[a_old + d] -= dataset[id + d];
                        }
                        ++switched; // Count how many points were changed
                    }
                    ++recomputed; // Count how many points were recomputed
                }
            }
        } // End loop through points

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

    return {k, std::move(centers), std::move(a), iter, converged};
}
