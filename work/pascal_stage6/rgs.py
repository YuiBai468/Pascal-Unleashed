from dataclasses import dataclass

import numpy as np


BLOCK = 32
LIMIT = 1904
GROUPS = (32, 64, 128, 256, 512)


@dataclass
class Encoding:
    a: np.ndarray
    b: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    group: int


def block_view(matrix):
    rows, k = matrix.shape
    padded_k = (k + BLOCK - 1) // BLOCK * BLOCK
    block_data = np.zeros((rows, padded_k), dtype=np.float32)
    block_data[:, :k] = matrix
    return block_data.reshape(rows, -1, BLOCK)


def median_exponent(maxima):
    exponents = np.zeros(maxima.shape[1], dtype=np.float32)
    for block, values in enumerate(maxima.T):
        nonzero = values[values > 0]
        if nonzero.size:
            exponents[block] = np.median(np.log2(nonzero))
    return exponents


def encode_channels(block_data, scales, k):
    q = np.rint(block_data / scales[:, :, None]).astype(np.int32)
    if np.max(np.abs(q)) > LIMIT:
        raise ArithmeticError('Reciprocal scale failed to cover the encoded values')
    h = np.rint(q.astype(np.float32) / 16).astype(np.int32)
    g = q - 15 * h
    packed = np.empty((q.shape[0], (k + 3) // 4, 2), dtype=np.int32)
    for index, channel in enumerate((h, g)):
        byte_values = channel.reshape(q.shape[0], -1).astype(np.int8)
        packed[:, :, index] = byte_values.view(np.int32)[:, :packed.shape[1]]
    return packed


def encode(a, b_transposed, group, reciprocal=True):
    if not BLOCK <= group <= 8192 or group % BLOCK or a.shape[1] != b_transposed.shape[1]:
        raise ValueError('Expected equal K dimensions and a supported group size')
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b_transposed)):
        raise ValueError('The prototype requires finite FP16 operands')
    aa, bb = block_view(a), block_view(b_transposed)
    ma, mb = np.max(np.abs(aa), axis=2) / LIMIT, np.max(np.abs(bb), axis=2) / LIMIT
    blocks = ma.shape[1]
    stride = group // BLOCK
    groups = (blocks + stride - 1) // stride
    alpha = np.empty((groups, a.shape[0]), dtype=np.float32)
    beta = np.empty((groups, b_transposed.shape[0]), dtype=np.float32)
    sa, sb = np.empty_like(ma), np.empty_like(mb)

    differences = median_exponent(ma) - median_exponent(mb) if reciprocal and stride > 1 else None
    for index, start in enumerate(range(0, blocks, stride)):
        physical = slice(start, start + stride)
        ratio = 1.0
        if differences is not None:
            shift = np.rint((differences[physical] - differences[start]) / 2).astype(np.int32)
            ratio = np.exp2(shift).astype(np.float32)
        # The zero-scale fallback changes only all-zero rows or columns.
        ar = np.max(ma[:, physical] / ratio, axis=1)
        br = np.max(mb[:, physical] * ratio, axis=1)
        ar[ar == 0] = 1
        br[br == 0] = 1
        alpha[index], beta[index] = ar, br
        sa[:, physical] = ar[:, None] * ratio
        sb[:, physical] = br[:, None] / ratio
    return Encoding(encode_channels(aa, sa, a.shape[1]),
                    encode_channels(bb, sb, a.shape[1]), alpha, beta, group)


def channels(packed, k):
    return [np.ascontiguousarray(packed[:, :, i]).view(np.int8)[:, :k].astype(np.int64)
            for i in range(2)]


def grouped_reference(encoded, k):
    ah, ag = channels(encoded.a, k)
    bh, bg = channels(encoded.b, k)
    c = np.zeros((ah.shape[0], bh.shape[0]), dtype=np.float32)
    for index, start in enumerate(range(0, k, encoded.group)):
        block = slice(start, start + encoded.group)
        z = 15 * (ah[:, block] @ bh[:, block].T) + ag[:, block] @ bg[:, block].T
        scale = (16 * encoded.alpha[index, :, None]) * encoded.beta[index, None, :]
        c = (z.astype(np.float32).astype(np.float64) * scale + c.astype(np.float64)).astype(np.float32)
    return c


def sample_operands(shape, kind='normal', seed=42):
    m, n, k = shape
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((m, k), dtype=np.float32)
    b = rng.standard_normal((n, k), dtype=np.float32)
    if kind in ('reciprocal', 'common_spike'):
        shifts = np.resize(np.array((-5, 4, 0, 6, -4, 3, -2, 1)), (k + 31) // 32)
        factors = np.repeat(np.exp2(shifts).astype(np.float32), 32)[:k]
        a *= factors
        b *= 1 / factors if kind == 'reciprocal' else factors
    return a.astype(np.float16), b.astype(np.float16)
