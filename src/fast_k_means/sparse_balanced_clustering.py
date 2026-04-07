"""
Sparse Mini-Batch Balanced Clustering via NQT Optimal Transport.

Each point routes mass to its m nearest centroids (support set computed via
HNSW approximate nearest neighbours).  Mini-batches subsample points uniformly;
per-batch sparse Sinkhorn on active centroids only.  Centroid and dual updates
use Robbins-Monro averaging.

Cost normalisation: ε values are relative — actual ε = ε_rel * median_cost,
computed from the initial support sets and updated on refresh.
"""

import time
import numpy as np
from typing import NamedTuple

import hnswlib


# ---------------------------------------------------------------------------
# NQT link functions
# ---------------------------------------------------------------------------

def exp_nqt(z: np.ndarray) -> np.ndarray:
    """NQT exp link function: exp(z) = (1 + 2^(z - floor(z))) * 2^floor(z)"""
    z = np.asarray(z, dtype=np.float32)
    fl = np.floor(z)
    return np.ldexp(np.float32(1.0) + z - fl, fl.astype(np.int32))


def log_nqt(x: np.ndarray) -> np.ndarray:
    """NQT log link function: log(x) = floor(log2(x)) + log2(x / 2^floor(log2(x)))"""
    x = np.asarray(x, dtype=np.float32)
    fl = np.floor(np.log2(np.maximum(x, 1e-30)))
    return fl + np.ldexp(x, (-fl).astype(np.int32)) - np.float32(1.0)


# ---------------------------------------------------------------------------
# HNSW-based support set computation
# ---------------------------------------------------------------------------

def build_centroid_index(centroids: np.ndarray,
                         ef_construction: int = 200,
                         M: int = 16) -> tuple[hnswlib.Index, float]:
    """
    Build an HNSW index over centroids.

    Returns:
        index: hnswlib.Index ready for knn_query
        build_time: wall-clock seconds for index construction
    """
    k, d = centroids.shape
    t0 = time.perf_counter()
    index = hnswlib.Index(space='l2', dim=d)
    index.init_index(max_elements=k, ef_construction=ef_construction, M=M)
    index.add_items(centroids, np.arange(k))
    index.set_ef(min(24, k))  # ef >= m for recall
    build_time = time.perf_counter() - t0
    return index, build_time


def query_support_sets(index: hnswlib.Index,
                       points: np.ndarray,
                       m: int) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Query HNSW index to find m nearest centroids per point.

    Returns:
        indices: (N, m) int array — column indices of nearest centroids
        costs: (N, m) float array — squared distances to those centroids
        query_time: wall-clock seconds for the query
    """
    t0 = time.perf_counter()
    indices, costs = index.knn_query(points, k=m)
    query_time = time.perf_counter() - t0
    return indices.astype(np.int64), costs.astype(np.float32), query_time


def compute_support_sets(points: np.ndarray,
                         centroids: np.ndarray,
                         m: int,
                         ef_construction: int = 200,
                         M: int = 16) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Full support set computation: build HNSW index + query all points.

    Returns:
        indices: (N, m) int array
        costs: (N, m) float array
        build_time: seconds to build the index
        query_time: seconds to query all points
    """
    index, build_time = build_centroid_index(
        centroids, ef_construction=ef_construction, M=M
    )
    indices, costs, query_time = query_support_sets(index, points, m)
    return indices, costs, build_time, query_time


# ---------------------------------------------------------------------------
# Sparse balanced Sinkhorn
# ---------------------------------------------------------------------------

def sparse_balanced_sinkhorn(
    support_indices: np.ndarray,
    cost_values: np.ndarray,
    active_centroids: np.ndarray,
    k: int,
    v: np.ndarray,
    epsilon: float,
    max_iter: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Balanced Sinkhorn on sparse support, restricted to active centroids.

    Remaps support indices to a dense [0, n_active) space so all column
    operations are O(n_active) not O(k).

    Args:
        support_indices: (n_b, m) centroid indices per point
        cost_values: (n_b, m) costs per point
        active_centroids: (n_active,) sorted indices of centroids reachable from batch
        k: total number of centroids
        v: (k,) current column duals (warm-start)
        epsilon: absolute regularisation strength
        max_iter: Sinkhorn iterations (2-3 for inner loop, more for final)

    Returns:
        P_data: (n_b, m) sparse assignment values (P_ij for j in support)
        v: (k,) updated column duals (only active entries modified)
    """
    n_b, _ = support_indices.shape
    n_active = len(active_centroids)
    a = np.float32(1.0 / n_b)  # row target
    b = np.float32(1.0 / k)    # column target
    log_b = log_nqt(np.array([b], dtype=np.float32))[0]

    # Remap support_indices to dense [0, n_active) space
    global_to_local = np.empty(k, dtype=np.int64)
    global_to_local[active_centroids] = np.arange(n_active)
    local_support = global_to_local[support_indices]  # (n_b, m)

    # Work on dense active-only dual vector
    v_active = v[active_centroids].copy()  # (n_active,) float32

    for _ in range(max_iter):
        # --- Row update ---
        v_local = v_active[local_support]  # (n_b, m)
        scores = (v_local - cost_values) / epsilon
        scores -= scores.max(axis=1, keepdims=True)
        U = exp_nqt(scores)
        row_sums = np.maximum(U.sum(axis=1, keepdims=True), np.float32(1e-30))
        P_data = (a * U) / row_sums

        # --- Column update: scatter into dense n_active array ---
        col_marginals = np.zeros(n_active, dtype=np.float32)
        np.add.at(col_marginals, local_support.ravel(), P_data.ravel())

        active_mask = col_marginals > 1e-30
        v_active[active_mask] += epsilon * (
            log_b - log_nqt(col_marginals[active_mask])
        )

    # Final row normalisation
    v_local = v_active[local_support]
    scores = (v_local - cost_values) / epsilon
    scores -= scores.max(axis=1, keepdims=True)
    U = exp_nqt(scores)
    row_sums = np.maximum(U.sum(axis=1, keepdims=True), np.float32(1e-30))
    P_data = (a * U) / row_sums

    # Write back to full dual vector
    v = v.copy()
    v[active_centroids] = v_active

    return P_data, v


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class ClusteringResult(NamedTuple):
    """Result of sparse balanced clustering."""
    centroids: np.ndarray
    assignments: np.ndarray
    inertia_history: list
    imbalance_history: list
    timing: dict  # build/query/train breakdown


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------

def sparse_balanced_clustering(
    points: np.ndarray,
    centroids: np.ndarray,
    m: int = 8,
    epsilon: float = 0.1,
    epsilon_final: float = 0.01,
    batch_size: int = 2048,
    eta_0: float | None = None,
    eta_decay_epochs: float = 4.0,
    sinkhorn_iters: int = 2,
    n_epochs: int = 8,
    support_refresh_epochs: int = 1,
    polish_iters: int = 100,
    epoch_fraction: float = 1.0,
    reduced_dim: int | None = None,
    verbose: bool = True,
) -> ClusteringResult:
    """
    Sparse mini-batch balanced clustering.

    ε values are relative: actual ε = ε_rel * median_nearest_cost.

    Args:
        points: (N, d)
        centroids: (k, d) initial centroids
        m: support set size (nearest centroids per point), power of 2
        epsilon: initial ε (relative to median cost)
        epsilon_final: final ε after annealing (relative to median cost)
        batch_size: points per mini-batch (n_b)
        eta_0: initial Robbins-Monro step size
        eta_decay_epochs: η decay timescale in epochs
        sinkhorn_iters: per-batch Sinkhorn iterations (2-3 recommended)
        n_epochs: full passes over the data
        support_refresh_epochs: recompute support sets every N epochs (0=never)
        polish_iters: number of final polishing iterations
        epoch_fraction: fraction of points to sample per epoch (0, 1].
                        Support sets are only queried for the sampled points.
                        All schedules (ε, η) remain unchanged.
        reduced_dim: if set, project points to this dimension via SVD for all
                     internal computation (HNSW, Sinkhorn, centroid updates).
                     Final centroids are recomputed in full-d from assignments.
        verbose: print progress
    """
    N, d_full = points.shape
    k = len(centroids)
    K = N // k

    points_full = np.asarray(points, dtype=np.float32)
    centroids_full_init = np.asarray(centroids, dtype=np.float32)

    # --- Dimension reduction via SVD ---
    if reduced_dim is not None and reduced_dim < d_full:
        t_svd = time.perf_counter()
        n_sample = min(N, 8192)
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(N, size=n_sample, replace=False)
        sample = points_full[sample_idx]
        mean_vec = sample.mean(axis=0)
        _, _, Vt = np.linalg.svd(sample - mean_vec, full_matrices=False)
        projection = Vt[:reduced_dim].T.astype(np.float32)  # (d_full, reduced_dim)
        points = (points_full - mean_vec) @ projection
        centroids = ((centroids_full_init - mean_vec) @ projection).copy()
        d = reduced_dim
        t_svd = time.perf_counter() - t_svd
        if verbose:
            print(f"SVD dimension reduction: {d_full} → {d} "
                  f"(sample={n_sample}, {t_svd*1000:.1f}ms)")
    else:
        points = points_full
        centroids = centroids_full_init.copy()
        d = d_full
        projection = None
        mean_vec = None

    v = np.zeros(k, dtype=np.float32)  # column duals

    # Epoch subsampling
    N_epoch = max(int(N * epoch_fraction), batch_size)

    # Derive η decay timescale from epochs
    batches_per_epoch = max((N_epoch + batch_size - 1) // batch_size, 8)
    batch_size = (N_epoch + batches_per_epoch - 1) // batches_per_epoch
    eta_0 = eta_0 or 8 * batch_size / N_epoch
    t0 = eta_decay_epochs * batches_per_epoch  # decay timescale in iterations

    eps_final = epsilon_final if (epsilon_final is not None and epsilon_final < epsilon) else epsilon

    # --- Timing accumulators ---
    total_build_time = 0.0
    total_query_time = 0.0
    n_refreshes = 0

    # Support arrays — lazily filled per epoch sample
    support_indices = np.empty((N, m), dtype=np.int64)
    support_costs = np.empty((N, m), dtype=np.float32)
    support_valid = np.zeros(N, dtype=bool)  # which points have current support

    # Initial support: build index, query epoch-sized sample
    if verbose:
        print(f"Computing initial support sets (m={m})...")
    centroid_index, bt = build_centroid_index(centroids)
    total_build_time += bt
    epoch_idx = np.random.choice(N, size=N_epoch, replace=False)
    si, sc, qt = query_support_sets(centroid_index, points[epoch_idx], m)
    support_indices[epoch_idx] = si
    support_costs[epoch_idx] = sc
    support_valid[epoch_idx] = True
    total_query_time += qt

    # Global cost normalisation: median nearest-centroid distance
    cost_scale = max(np.median(sc[:, 0]), 1e-12)

    if verbose:
        print(f"  HNSW build: {bt*1000:.1f}ms, query: {qt*1000:.1f}ms")
        dim_str = f"{d}" if projection is None else f"{d} (reduced from {d_full})"
        print(f"  {N}x{k}, d={dim_str}, epoch_fraction={epoch_fraction}")
        print(f"  cost_scale (median): {cost_scale:.4f}")
        print(f"  ε: {epsilon:.4f} → {eps_final:.4f} (×cost_scale = "
              f"{epsilon*cost_scale:.4f} → {eps_final*cost_scale:.4f})")
        print(f"  η: {eta_0:.4f}, t₀={t0:.0f} iters ({eta_decay_epochs} epochs)")
        print(f"  Sinkhorn: {sinkhorn_iters} inner")
        print(f"  Batches/epoch: {batches_per_epoch}, batch_size: {batch_size}")
        if support_refresh_epochs > 0:
            print(f"  Support refresh: every {support_refresh_epochs} epochs")

    inertia_history = []
    imbalance_history = []
    t = 0

    total_train_time = 0

    for epoch in range(n_epochs):
        frac = epoch / max(n_epochs - 1, 1)
        eps_rel = epsilon * (eps_final / epsilon) ** frac

        # Sample points for this epoch
        epoch_idx = np.random.choice(N, size=N_epoch, replace=False)
        epoch_idx = np.sort(epoch_idx)  # for better memory access patterns

        t_refresh_support_start = time.perf_counter()

        # Refresh support sets: rebuild index periodically, query new points
        if (support_refresh_epochs > 0 and epoch > 0 and epoch % support_refresh_epochs == 0):
            centroid_index, bt = build_centroid_index(centroids)
            total_build_time += bt
            support_valid[:] = False  # invalidate all on index rebuild
            n_refreshes += 1

        # Query support for epoch points that are stale
        need_query = epoch_idx[~support_valid[epoch_idx]]
        if len(need_query) > 0:
            si, sc, qt = query_support_sets(centroid_index, points[need_query], m)
            support_indices[need_query] = si
            support_costs[need_query] = sc
            support_valid[need_query] = True
            total_query_time += qt
            cost_scale = max(np.median(sc[:, 0]), 1e-12)
            if verbose and epoch > 0:
                print(f"  [refresh #{n_refreshes}: queried {len(need_query)} pts, "
                      f"query={qt:.2f}s, cost_scale={cost_scale:.4f}]")

        refresh_support_time = time.perf_counter() - t_refresh_support_start
        total_train_time += refresh_support_time

        t_epoch_start = time.perf_counter()

        eps_abs = eps_rel * cost_scale
        perm = np.random.permutation(N_epoch)

        for batch_start in range(0, N_epoch, batch_size):
            batch_end = min(batch_start + batch_size, N_epoch)
            batch_idx = np.sort(epoch_idx[perm[batch_start:batch_end]])

            # Robbins-Monro step size
            eta = eta_0 / (1.0 + t / t0) if t0 > 0 else eta_0

            # Gather: one sequential read of centroid array
            batch_support = support_indices[batch_idx]  # (nb, m)
            batch_pts = points[batch_idx]               # (nb, d)
            active_centroids = np.unique(batch_support)
            n_active = len(active_centroids)
            active_cents = centroids[active_centroids]  # (n_active, d)

            # Remap support indices to dense [0, n_active)
            global_to_local = np.empty(k, dtype=np.int64)
            global_to_local[active_centroids] = np.arange(n_active)
            local_support = global_to_local[batch_support]  # (nb, m)

            # Compute costs
            cents_at_support = centroids[batch_support]  # (nb, m, d)
            diff = batch_pts[:, None, :] - cents_at_support
            batch_costs = np.einsum('imd,imd->im', diff, diff)

            # Sinkhorn on local indices
            P_data, v_new = sparse_balanced_sinkhorn(
                local_support, batch_costs,
                active_centroids=np.arange(n_active),
                k=n_active, v=v[active_centroids],
                epsilon=eps_abs, max_iter=sinkhorn_iters,
            )

            # Scatter: one write back to dual and centroid arrays
            # Dual update
            v[active_centroids] = (
                (1 - eta) * v[active_centroids] + eta * v_new
            )
            # Centroid update into n_active-sized buffers
            weight_sums = np.zeros(n_active, dtype=np.float32)
            np.add.at(weight_sums, local_support.ravel(), P_data.ravel())
            wp = np.einsum('im,id->imd', P_data, batch_pts)
            weighted_pts = np.zeros((n_active, d), dtype=np.float32)
            np.add.at(weighted_pts, local_support.ravel(), wp.reshape(-1, d))
            mask = weight_sums > 1e-7
            active_cents[mask] = (
                (1 - eta) * active_cents[mask]
                + eta * weighted_pts[mask] / weight_sums[mask, None]
            )
            centroids[active_centroids] = active_cents

            t += 1

        epoch_time = time.perf_counter() - t_epoch_start
        total_train_time += epoch_time

        if verbose:
            # End-of-epoch diagnostics: nearest-centroid on epoch sample
            sample_sup = support_indices[epoch_idx]  # (N_epoch, m)
            sample_costs = support_costs[epoch_idx]
            assignments_sample = sample_sup[np.arange(N_epoch), np.argmin(sample_costs, axis=1)]
            dists_sample = sample_costs.min(axis=1)
            counts = np.bincount(assignments_sample, minlength=k)
            # Scale counts to full N for comparable σ
            counts_scaled = (counts.astype(np.float64) * N / N_epoch).astype(np.int64)
            inertia = np.mean(dists_sample)
            inertia_history.append((epoch, t, inertia))
            imbalance_history.append((counts_scaled.min(), counts_scaled.max(), counts_scaled.std()))
            print(f"  epoch {epoch:2d} ({t:4d} iters) ({epoch_time:.2f}s)"
                  f" | η={eta:.4f} | ε_rel={eps_rel:.4f} | inertia≈{inertia:.4f} | "
                  f"bal≈({counts_scaled.min()},{counts_scaled.max()}) σ≈{counts_scaled.std():.1f}")
        else:
            print(f"  epoch {epoch:2d} ({t:4d} iters) "
                  f"({refresh_support_time:.2f}s refresh) "
                  f"({epoch_time:.2f}s SH) "
                  f"| η={eta:.4f} | ε_rel={eps_rel:.4f}")

    # --- Dual polishing: full-batch Sinkhorn with frozen centroids ---
    # Refresh support sets so they reflect final centroid positions
    if verbose:
        print("  Refreshing support sets for polishing...")
    support_indices, support_costs, bt, qt = compute_support_sets(
        points, centroids, m
    )
    total_build_time += bt
    total_query_time += qt
    total_train_time += bt + qt
    cost_scale = max(np.median(support_costs[:, 0]), 1e-12)
    if verbose:
        print(f"  [polish refresh: build={bt:.2f}s, query={qt:.2f}s, "
              f"cost_scale={cost_scale:.4f}]")

    if verbose:
        print("  Dual polishing (full-batch Sinkhorn)...")
    t_polish_start = time.perf_counter()

    # Recompute exact costs on final support sets
    polish_costs = np.empty((N, m), dtype=np.float32)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        pts_i = points[start:end]
        cents_i = centroids[support_indices[start:end]]  # (nb, m, d)
        diff = pts_i[:, None, :] - cents_i
        polish_costs[start:end] = np.einsum('imd,imd->im', diff, diff)

    # ε during polishing
    eps_polish = eps_final * cost_scale

    a_full = np.float32(1.0 / N)
    b_full = np.float32(1.0 / k)
    log_b = log_nqt(np.array([b_full], dtype=np.float32))[0]

    v_polish = v.copy()
    prev_sigma = float('inf')

    for it in range(polish_iters):
        frac = it / max(polish_iters - 1, 1)

        # Row update: P_ij ∝ exp_nqt((v_j - c_ij) / ε), row-normalised
        v_at_support = v_polish[support_indices]  # (N, m)
        scores = (v_at_support - polish_costs) / eps_polish
        scores -= scores.max(axis=1, keepdims=True)
        U = exp_nqt(scores)
        row_sums = np.maximum(U.sum(axis=1, keepdims=True), np.float32(1e-30))
        P_full = (a_full * U) / row_sums  # (N, m)

        # Column marginals
        col_marg = np.zeros(k, dtype=np.float32)
        np.add.at(col_marg, support_indices.ravel(), P_full.ravel())

        # Column update
        active = col_marg > 1e-30
        v_polish[active] += eps_polish * (log_b - log_nqt(col_marg[active]))

        # Periodic diagnostics + early stopping
        if it % 10 == 0 or it == polish_iters - 1:
            best = np.argmax(P_full, axis=1)
            asgn = support_indices[np.arange(N), best]
            cnts = np.bincount(asgn, minlength=k)
            sigma = cnts.std()
            if verbose:
                print(f"    polish iter {it:3d}: ε_rel={eps_polish/cost_scale:.4f} "
                      f"bal=({cnts.min()},{cnts.max()}) σ={sigma:.1f}")
            if sigma > prev_sigma and it >= 20:
                if verbose:
                    print(f"    early stop: σ increased ({prev_sigma:.1f} → {sigma:.1f})")
                break
            prev_sigma = sigma

    total_polish_time = time.perf_counter() - t_polish_start
    if verbose:
        print(f"    polish time: {total_polish_time:.2f}s")

    # --- Final assignment from polished transport plan ---
    best_in_support = np.argmax(P_full, axis=1)
    assignments = support_indices[np.arange(N), best_in_support]

    # Reconstruct full-d centroids from assignments
    if projection is not None:
        centroids_out = np.zeros((k, d_full), dtype=np.float32)
        for j in range(k):
            mask_j = assignments == j
            if mask_j.any():
                centroids_out[j] = points_full[mask_j].mean(axis=0)
        centroids = centroids_out


    counts = np.bincount(assignments, minlength=k)
    final_inertia = np.mean(
        np.sum((points_full - centroids[assignments]) ** 2, axis=1)
    )

    # Timing summary
    timing = {
        'total_build': total_build_time,
        'total_query': total_query_time,
        'total_support': total_build_time + total_query_time,
        'total_train': total_train_time,
        'total_polish': total_polish_time,
        'n_refreshes': n_refreshes,
        'n_iters': t,
    }

    if verbose:
        print(f"  Final inertia: {final_inertia:.4f}")
        print(f"  Balance: ({counts.min()},{counts.max()}) σ={counts.std():.1f}, target={K}")
        print(f"  Timing: build={total_build_time*1000:.1f}ms total, "
              f"query={total_query_time*1000:.1f}ms total, "
              f"train={total_train_time:.2f}s ({n_refreshes} refreshes), "
              f"polish={total_polish_time:.2f}s")

    return ClusteringResult(
        centroids=centroids,
        assignments=assignments,
        inertia_history=inertia_history,
        imbalance_history=imbalance_history,
        timing=timing,
    )
