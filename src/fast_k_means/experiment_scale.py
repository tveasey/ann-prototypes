"""
Experiment: support set refresh frequency sweep with HNSW.

N=16384, k=128, d=128, single Gaussian IVF data.
Compares m=8 with various refresh frequencies against m=16 and m=32 baselines.
Reports HNSW build vs query timing breakdown.
"""

import argparse
import time
from pathlib import Path

import numpy as np
from sparse_balanced_clustering import sparse_balanced_clustering


# ---------------------------------------------------------------------------
# Hierarchical k-means init
# ---------------------------------------------------------------------------

def _small_kmeans(points, k, n_iters=10, rng=None, k_means_plus_plus=False):
    """Run k-means on a small set of points (within a single cluster split)."""
    if rng is None:
        rng = np.random.RandomState()
    N, d = points.shape
    if N <= k:
        return points.copy(), np.arange(N)
    best_dist = np.full(N, np.inf)
    first = rng.randint(N)
    centroids = np.empty((k, d))
    centroids[0] = points[first]
    if k_means_plus_plus:
        for i in range(1, k):
            d2 = np.sum((points - centroids[i-1]) ** 2, axis=1)
            np.minimum(best_dist, d2, out=best_dist)
            p = np.maximum(best_dist, 1e-30)
            p = p / p.sum()
            centroids[i] = points[rng.choice(N, p=p)]
    else:
        indices = np.arange(N)
        np.random.shuffle(indices)
        for i in range(1, k):
            centroids[i] = points[indices[i]]
    for i in range(n_iters):
        asgn = np.empty(N, dtype=np.int64)
        chunk = max(1, min(5000, N))
        for s in range(0, N if i == n_iters - 1 else 256 * k, chunk):
            e = min(s + chunk, N)
            dists = np.sum((points[s:e, None] - centroids[None, :]) ** 2, axis=-1)
            asgn[s:e] = np.argmin(dists, axis=1)
        for j in range(k):
            mask = asgn == j
            if mask.sum() > 0:
                centroids[j] = points[mask].mean(axis=0)
    return centroids, asgn


def hierarchical_kmeans(points, target_size=128, max_split=128, n_iters=10, seed=42):
    """Hierarchical k-means: repeatedly split the largest cluster."""
    rng = np.random.RandomState(seed)
    N = len(points)
    k_target = N // target_size

    assignments = np.zeros(N, dtype=np.int64)
    centroids = [points.mean(axis=0)]
    cluster_points = {0: np.arange(N)}

    next_id = 1
    step = 0

    while len(centroids) < k_target:
        largest_id = max(cluster_points.keys(), key=lambda c: len(cluster_points[c]))
        largest_pts = cluster_points[largest_id]
        largest_size = len(largest_pts)

        n_sub = min(largest_size // target_size, max_split)
        n_sub = max(n_sub, 2)

        remaining = k_target - len(centroids) + 1
        n_sub = min(n_sub, remaining)

        if n_sub < 2 or largest_size < 4:
            break
        n_sub = max(n_sub, 2)

        sub_cents, sub_asgn = _small_kmeans(
            points[largest_pts], n_sub, n_iters=n_iters, rng=rng
        )

        del cluster_points[largest_id]
        centroids[largest_id] = sub_cents[0]
        cluster_points[largest_id] = largest_pts[sub_asgn == 0]
        assignments[largest_pts[sub_asgn == 0]] = largest_id

        for s in range(1, n_sub):
            new_id = next_id
            next_id += 1
            centroids.append(sub_cents[s])
            mask = sub_asgn == s
            cluster_points[new_id] = largest_pts[mask]
            assignments[largest_pts[mask]] = new_id

        step += 1
        if step % 10 == 0 or len(centroids) >= k_target:
            counts = np.array([len(v) for v in cluster_points.values()])
            print(f"    step {step}: k={len(centroids)}, "
                  f"bal=({counts.min()},{counts.max()}) σ={counts.std():.1f}")

    # Compact
    id_map = {}
    compact_centroids = []
    for i, cid in enumerate(sorted(cluster_points.keys())):
        id_map[cid] = i
        compact_centroids.append(centroids[cid])
    compact_centroids = np.array(compact_centroids)
    for i in range(N):
        assignments[i] = id_map[assignments[i]]

    return compact_centroids, assignments


def compute_inertia(points, centroids, assignments):
    return np.mean(np.sum((points - centroids[assignments]) ** 2, axis=1))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(points: np.ndarray, centroids: np.ndarray,
        seed: int = 0, m: int = 32, n_epochs: int = 8,
        epsilon: float = 0.1, epsilon_final: float = 0.01,
        batch_size: int = 2048, eta_0: float | None = None,
        eta_decay_epochs: float = 4.0, sinkhorn_iters: int = 3,
        support_refresh_epochs: int = 1, polish_iters: int = 100,
        epoch_fraction: float = 1.0, reduced_dim: int | None = None) -> dict:
    np.random.seed(seed)

    result = sparse_balanced_clustering(
        points, centroids.copy(),
        m=m, epsilon=epsilon, epsilon_final=epsilon_final,
        batch_size=batch_size,
        eta_0=eta_0, eta_decay_epochs=eta_decay_epochs,
        sinkhorn_iters=sinkhorn_iters,
        n_epochs=n_epochs,
        support_refresh_epochs=support_refresh_epochs,
        polish_iters=polish_iters,
        epoch_fraction=epoch_fraction,
        reduced_dim=reduced_dim,
        verbose=True,
    )

    k = len(centroids)
    inertia = compute_inertia(points, result.centroids, result.assignments)
    counts = np.bincount(result.assignments, minlength=k)

    return {
        'inertia': inertia,
        'bal_min': counts.min(),
        'bal_max': counts.max(),
        'bal_std': counts.std(),
        'inertia_history': result.inertia_history,
        'imbalance_history': result.imbalance_history,
        'timing': result.timing,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_fves(fname):
    x = np.fromfile(fname, dtype='int32')
    d = x[0]
    x = x.reshape(-1, d + 1)[:, 1:].copy()
    x = x.view('float32')
    return x

def generate_synthetic_data(N, seed=42):
    rng = np.random.RandomState(seed)
    d = 256
    d_intr = 20
    low_d = rng.randn(N, d_intr)
    A = rng.randn(d_intr, d)
    Q, _ = np.linalg.qr(A.T)
    Q = Q[:, :d_intr].T
    points = low_d @ Q + rng.randn(N, d) * 0.05
    return points

def main(corpus_filename: Path | None, target_size: int = 256) -> None:
    """Run the experiment."""

    if corpus_filename is not None:
        print(f"Loading {corpus_filename}...")
        points = load_fves(corpus_filename)
    else:
        points = generate_synthetic_data(4*16384)

    K = target_size
    N, d = points.shape
    k = N // K

    print(f"\nHierarchical k-means warm-start (target_size={K})...")
    t0 = time.time()
    warm_cents, warm_asgn = hierarchical_kmeans(points, target_size=K, max_split=128, seed=42)
    t_init = time.time() - t0
    warm_inertia = compute_inertia(points, warm_cents, warm_asgn)
    warm_counts = np.bincount(warm_asgn, minlength=k)
    print(f"  Warm inertia: {warm_inertia:.4f}, balance: ({warm_counts.min()},{warm_counts.max()}) "
          f"σ={warm_counts.std():.1f}, time={t_init:.1f}s")

    # Configs: (name, m, refresh_epochs, batch_size, reduced_dim, epoch_fraction, polish_iters)
    configs = [
        ('frac=0.25', 16, 1, 2048, 128, 0.25, 150),
    ]

    all_results = {}
    for name, m_val, refresh, bs, rd, epoch_frac, polish_iters in configs:
        print(f"\n{'='*80}")
        print(f"  {name}, m={m_val}, ref@{refresh}, S=3, ε=0.1→0.01, 16 epochs")
        print(f"{'='*80}")

        r = run(points, warm_cents, seed=10000, m=m_val,
                support_refresh_epochs=refresh, batch_size=bs,
                reduced_dim=rd, epoch_fraction=epoch_frac,
                polish_iters=polish_iters)
        delta = r['inertia'] - warm_inertia
        tm = r['timing']
        print(f"\n  RESULT: inertia={r['inertia']:.4f} (Δ={delta:+.4f}), "
              f"bal=({r['bal_min']},{r['bal_max']}) σ={r['bal_std']:.1f}")
        print(f"  HNSW: build={tm['total_build']*1000:.1f}ms, "
              f"query={tm['total_query']*1000:.1f}ms, "
              f"refreshes={tm['n_refreshes']}")
        print(f"  Train: {tm['total_train']:.2f}s, "
              f"per-iter: {tm['total_train']/tm['n_iters']*1000:.2f}ms")
        all_results[name] = r

    # Summary table
    print(f"\n{'='*120}")
    print(f"SUMMARY  N={N}, k={k}, K={K}, d={d}")
    print(f"warm inertia={warm_inertia:.4f} (hierarchical k-means)")
    print(f"{'='*120}")
    print(f"{'config':<15} | {'inertia':>10} | {'Δ':>8} | {'bal':>9} | {'σ':>6} | "
          f"{'train':>8} | {'build_ms':>8} | {'query_ms':>8} | {'#refresh':>8} | "
          f"{'ms/iter':>8}")
    print("-" * 120)

    for name, r in all_results.items():
        delta = r['inertia'] - warm_inertia
        tm = r['timing']
        ms_iter = tm['total_train'] / tm['n_iters'] * 1000
        print(f"{name:<15} | {r['inertia']:>10.4f} | {delta:>+8.4f} | "
              f"({r['bal_min']:>3},{r['bal_max']:>3}) | {r['bal_std']:>6.1f} | "
              f"{tm['total_train']:>7.2f}s | "
              f"{tm['total_build']*1000:>8.1f} | {tm['total_query']*1000:>8.1f} | "
              f"{tm['n_refreshes']:>8} | {ms_iter:>7.2f}")

    # Trajectory
    print(f"\n{'='*120}")
    print("INERTIA + BALANCE TRAJECTORY")
    print(f"{'='*120}")
    for name, r in all_results.items():
        print(f"\n  {name}:")
        for ep, (_, iters, inertia) in enumerate(r['inertia_history']):
            bal = r['imbalance_history'][ep]
            delta = inertia - warm_inertia
            print(f"    epoch {ep:2d} ({iters:5d} iters): inertia={inertia:.4f} "
                  f"(Δ={delta:+.4f}) bal=({bal[0]:.0f},{bal[1]:.0f}) σ={bal[2]:.1f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run scalable balanced clustering experiments.")
    parser.add_argument('--size', type=int, default=256, help="Target cluster size K")
    parser.add_argument('--corpus', type=Path, default=None,
                        help="Path to corpus file (not used in this experiment, synthetic data is generated)")
    args = parser.parse_args()

    main(args.corpus, args.size)
