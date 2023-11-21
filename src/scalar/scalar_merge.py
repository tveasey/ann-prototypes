import math
import numpy as np
from sklearn.cluster import KMeans


# Merge functionality

def read_fvecs(file: str) -> np.ndarray:
    x = np.fromfile(file, dtype=np.float32)
    dim = x.view(np.int32)[0]
    assert dim > 0
    x = x.reshape(-1, 1 + dim)
    if not all(x.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + file)
    x = x[:, 1:]
    return x

def sample(x: np.ndarray, n: int = 25000) -> np.ndarray:
    if x.shape[0] < n:
        return x
    return x[np.random.choice(x.shape[0], n), :]

def quantise(x: np.ndarray,
             lower: float,
             upper: float,
             pow_2_bits: float = 256) -> np.ndarray:
    x_q = np.clip(x, a_min=lower, a_max=upper)
    x_q = np.round(pow_2_bits * (x_q - lower) / (upper - lower))
    return x_q

def quantise_all(x: list[np.ndarray],
                 q: list[np.ndarray],
                 pow_2_bits: float = 256.0) -> list[np.ndarray]:
    return [quantise(x_p, q_p[0], q_p[1], pow_2_bits) for x_p, q_p in zip(x, q)]

def dequantise(x_q: np.ndarray,
               lower: float,
               upper: float,
               pow_2_bits: float = 256.0) -> np.ndarray:
    return lower + (upper - lower) * x_q / pow_2_bits

def dequantise_all(x_q: list[np.ndarray],
                   q: list[np.ndarray],
                   pow_2_bits: float = 256.0) -> list[np.ndarray]:
    return [dequantise(x_p, q_p[0], q_p[1], pow_2_bits) for x_p, q_p in zip(x_q, q)]

def channel_quantise(x: np.ndarray,
                     interval: float,
                     n_channels: int = 3,
                     pow_2_bits: float = 16.0) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    m = np.array(np.mean(x, axis=0)).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_channels, n_init="auto").fit(m)
    indices = np.array(range(len(m)))
    channels = [indices[kmeans.labels_ == i] for i in range(n_channels)]
    q_c = [central_confidence_interval(x[:,c], interval) for c in channels]
    x_q = quantise_all([x[:,c] for c in channels], q_c, pow_2_bits)
    x_q = np.concatenate(x_q, axis=1)
    return x_q, q_c, channels

def channel_dequantise(x_q: np.ndarray,
                       q: list[np.ndarray],
                       channels: list[np.ndarray],
                       pow_2_bits: float = 16.0) -> np.ndarray:
    widths = np.cumsum([0] + [len(c) for c in channels])
    x_q = [x_q[:,widths[i-1]:widths[i]] for i in range(1, len(widths))]
    x_c = dequantise_all(x_q, q, pow_2_bits)
    x = np.array([])
    x.resize((x_q[0].shape[0], widths[-1]))
    for i, channel in enumerate(channels):
        x[:,channel] = x_c[i]
    return x

def central_confidence_interval(x: np.ndarray, interval: float) -> np.ndarray:
    lower = np.quantile(x, 0.5 * (1.0 - interval))
    upper = np.quantile(x, 0.5 * (1.0 + interval))
    return np.array([lower, upper])

def merged_quantiles(q: list[np.ndarray], n: np.ndarray) -> np.ndarray:
    # Use weighted linear combination. This works better with the logic
    # to requantize if the segments are very imbalanced.
    return sum(n_p * q_p for n_p, q_p in zip(n, q)) / np.sum(n)

def recompute_merged_quantiles(x_q: list[np.ndarray],
                               q: list[np.ndarray],
                               interval: float) -> np.ndarray:
    # Sample partitions in proportion to their size. This means we should
    # avoid requantizing the large segments if they're very imbalanced.
    tot = sum(x.shape[0] for x in x_q)
    w_p = [x.shape[0] / tot for x in x_q]
    x_s = dequantise_all([sample(x, int(math.ceil(25000 * w))) for x,w in zip(x_q, w_p)], q)
    return central_confidence_interval(np.concatenate(x_s, axis=0), interval)

def should_recompute_quantiles(q: list[np.ndarray],
                               new_q: np.ndarray) -> bool:
    # Averaging breaks down if quantiles are sihnificantly different.
    tol = (new_q[1] - new_q[0]) / 32.0
    return sum(1 for qp in q if np.max(np.abs(new_q - qp)) > tol) > 0

def should_requantise(q: list[np.ndarray],
                      new_q: np.ndarray) -> np.ndarray:
    # We use a tolerance which ensures that the error introduced is
    # small compared to the quantisation error. Also it will be mean
    # zero as well so should not accumulate over multiple merges.
    tols = np.array([0.2 * (q_p[1] - q_p[0]) / 256.0 for q_p in q])
    return np.array([
        i for i, (q_p, tol) in enumerate(zip(q, tols))
        if np.max(np.abs(new_q - q_p)) > tol
    ])

def merge_quantisation(x_q: list[np.ndarray],
                       q: list[np.ndarray],
                       interval: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:

    n = np.array([x.shape[0] for x in x_q])
    q_m = merged_quantiles(q, n)

    if should_recompute_quantiles(q, q_m):
        q_m = recompute_merged_quantiles(x_q, q, interval)
    x_m = [np.array([]) for _ in range(len(x_q))]

    requantise = should_requantise(q, q_m)
    copy = [i for i in range(len(x_q)) if i not in requantise]

    n_requantise = sum(n[i] for i in requantise)
    for i in requantise:
        x_m[i] = quantise(
            dequantise(x_q[i], q[i][0], q[i][1]), q_m[0], q_m[1]
        )
    for i in copy:
        x_m[i] = x_q[i]

    return np.concatenate(x_m, axis=0), q_m, n_requantise / sum(n)


# Test functionality

def random_partition(x: np.ndarray,
                     partitions: list[int]) -> list[np.ndarray]:
    index = np.array([i for i in range(x.shape[0])])
    np.random.shuffle(index)
    index = [index[partitions[i-1]:partitions[i]] for i in range(1, len(partitions))]
    for i in index:
        i.sort()
    return [x[i,:] for i in index]

def sorted_partition(x: np.ndarray,
                     partitions: list[int],
                     asc = True) -> list[np.ndarray]:
    norms = np.linalg.norm(x, axis=1)
    index = np.argsort(norms)
    if asc:
        index = np.flip(index)
    index = [index[partitions[i-1]:partitions[i]] for i in range(1, len(partitions))]
    for i in index:
        i.sort()
    return [x[i,:] for i in index]

def cluster_partition(x: np.ndarray,
                      n_partitions: int) -> list[np.ndarray]:
    kmeans = KMeans(n_clusters=n_partitions, n_init=5, max_iter=20).fit(x)
    return [x[kmeans.labels_ == i] for i in range(n_partitions)]


def compute_quantisation_rmse(x: np.ndarray,
                              x_q: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.power(x - x_q, 2.0), axis=1))

def compute_quantisation_theta(x: np.ndarray,
                               x_q: np.ndarray) -> np.ndarray:
    # The angle between the vectors in radians
    return np.arccos(
        np.sum(x * x_q, axis=1) /
        np.sqrt(np.sum(np.power(x, 2.0), axis=1) * np.sum(np.power(x_q, 2.0), axis=1))
    )
