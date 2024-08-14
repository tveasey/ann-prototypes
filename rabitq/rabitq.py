import math
import argparse
from pathlib import Path
from tqdm import tqdm

import numpy as np


def _centre_data(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Centre the data by subtracting the mean.
    """
    centre = np.mean(data, axis=0)
    return centre, data - centre


def _normalize(data: np.ndarray) -> np.ndarray:
    """
    Normalize the data.
    """
    if len(data.shape) == 1:
        return data / np.linalg.norm(data)
    return data / np.linalg.norm(data, axis=1)[:, None]


def _binarize(data: np.ndarray) -> np.ndarray:
    """
    Convert the data to a bit vector.
    """
    return np.array(
        data > 0, dtype=np.float32
    )  # Don't really care about packing properly


def _quantize(
    q: np.ndarray, num_bits: int, rng: np.random.Generator
) -> tuple[float, float, np.ndarray]:
    """
    Quantize the query.
    """
    m = 2**num_bits - 1
    v_l = np.min(q)
    v_u = np.max(q)
    inv_delta = m / (v_u - v_l) # multiply is much cheaper than divide
    return (
        v_l,
        v_u,
        np.floor(inv_delta * (q - v_l) + rng.uniform(0.0, 1.0), dtype=np.float32),
    )  # Don't care about packing properly here

def _simulate_quantization(data: np.ndarray, num_bits: int, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate quantization noise.
    """
    l, u, data = _quantize(data, num_bits, rng)
    delta = (u - l) / (2**num_bits - 1)
    return l + delta * data

def plot_constant_distributions(q: np.ndarray, o_r: np.ndarray):

    from matplotlib import pyplot as plt

    d = len(q)
    sqrt_d = math.sqrt(d)

    centre, o = _centre_data(o_r)
    o = _normalize(o)
    q = q - centre
    q = _normalize(q)
    x_b = _binarize(o)

    s_b = x_b.sum(axis=1)
    o_o_q = (o * (2 * x_b - 1)).sum(axis=1) / sqrt_d
    o_n = np.linalg.norm(o_r - centre, axis=1)
    c_o_r = np.dot(centre, o_r.T)

    plt.figure(1)
    plt.hist(s_b, bins=100)
    plt.title("distribution of sum of binary components")
    plt.figure(2)
    plt.hist(o_o_q, bins=100)
    plt.title("distribution of dot of document with quantized document")
    plt.figure(3)
    plt.hist(o_n, bins=100)
    plt.title("distribution of norm of residual of document from its centre")
    plt.figure(4)
    plt.hist(c_o_r, bins=100)
    plt.title("distribution of dot of centre with document")
    plt.show()

def normalized_residual_dot(
        q: np.ndarray,
        o_r: np.ndarray,
        constant_quantisation_bits: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    d = len(q)
    sqrt_d = math.sqrt(d)

    centre, o = _centre_data(o_r)
    o = _normalize(o)
    q = q - centre
    q = _normalize(q)

    rng = np.random.Generator(np.random.PCG64())

    x_b = _binarize(o)
    v_l, v_u, q_u = _quantize(q, 4, rng)

    o_o_q = (o * (2 * x_b - 1)).sum(axis=1) / sqrt_d
    if constant_quantisation_bits is not None:
        o_o_q = _simulate_quantization(o_o_q, 8, rng)

    dot_q = np.dot(q_u, x_b.T)

    delta = (v_u - v_l) / (2**4 - 1)

    # Undo the scaling and shifting applied to the query and the data.
    est_dot = (
        2 * delta / sqrt_d * dot_q
        + 2 * v_l / sqrt_d * x_b.sum(axis=1)
        - delta / sqrt_d * q_u.sum()
        - sqrt_d * v_l
    ) / o_o_q

    return est_dot, centre


def mip(q: np.ndarray,
        o_r: np.ndarray,
        constant_quantisation_bits: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the corrected maximum inner product.
    """

    true_dot = np.dot(q, o_r.T)

    est_dot, centre = normalized_residual_dot(q, o_r, constant_quantisation_bits)

    q_n = np.linalg.norm(q - centre)
    o_n = np.linalg.norm(o_r - centre, axis=1)
    c_o_r = np.dot(centre, o_r.T)
    if constant_quantisation_bits is not None:
        o_n = _simulate_quantization(o_n, 8, np.random.Generator(np.random.PCG64()))
        c_o_r = _simulate_quantization(c_o_r, 8, np.random.Generator(np.random.PCG64()))

    est_dot_mip = (
        q_n * o_n * est_dot
        + c_o_r
        + np.dot(q, centre)
        - np.dot(centre, centre)
    )

    return true_dot, est_dot_mip

def fvecs_read(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute the corrected maximum inner product.")
    parser.add_argument("--query-file", type=str, help="The query vectors file.")
    parser.add_argument("--docs-file", type=str, help="The docs vectors file.")
    parser.add_argument("--top-k", type=int, default=10, help="The top-k value (default 10).")
    parser.add_argument("--rerank-multiple", type=int, default=5, help="The multiple to rerank (default 5).")
    parser.add_argument("--constant-quantisation-bits", type=int, default=None, help="The number of bits to quantize the constants.")
    parser.add_argument("--plot", action="store_true", help="Plot the distributions of the constants.")
    args = parser.parse_args()

    Path(args.query_file).resolve(strict=True)
    Path(args.docs_file).resolve(strict=True)

    q = fvecs_read(args.query_file)
    o = fvecs_read(args.docs_file)

    if args.plot:
        plot_constant_distributions(q, o)
        exit(0)

    k = args.top_k
    m = args.rerank_multiple * k

    recalls = []
    for i in tqdm(range(min(30, len(q))), desc="Calculating recall"):
        true_dot, est_dot_mip = mip(q[i,:], o)

        reranked = np.argpartition(true_dot, -k)[-k:]
        top_k = np.argpartition(est_dot_mip, -m)[-m:]

        recalls.append(len(set(reranked).intersection(set(top_k)))*100/k)

    print(f"Recall@{k}|{m}: {np.mean(recalls)}")
