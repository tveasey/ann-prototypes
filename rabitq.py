import math

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
    delta = (v_u - v_l) / m
    return (
        v_l,
        v_u,
        np.floor((q - v_l) / delta + rng.uniform(0.0, 1.0), dtype=np.float32),
    )  # Don't care about packing properly here


def normalized_residual_dot(q: np.ndarray, o_r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = len(q)
    sqrt_d = math.sqrt(d)

    centre, o = _centre_data(o_r)
    o = _normalize(o)
    q = q - centre
    q = _normalize(q)

    x_b = _binarize(o)
    v_l, v_u, q_u = _quantize(q, 4, np.random.Generator(np.random.PCG64()))

    o_o_q = (o * (2 * x_b - 1)).sum(axis=1) / sqrt_d

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


def mip(q: np.ndarray, o_r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the corrected maximum inner product.
    """

    true_dot = np.dot(q, o_r.T)

    est_dot, centre = normalized_residual_dot(q, o_r)

    q_n = np.linalg.norm(q - centre)
    o_n = np.linalg.norm(o_r - centre, axis=1)

    est_dot_mip = (
        q_n * o_n * est_dot
        + np.dot(centre, o_r.T)
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
