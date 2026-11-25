"""Screen integer codes that embed several FP16 values in one hyperplane."""

import argparse
from itertools import combinations_with_replacement, product
import json
import math

import numpy as np

from bench import SOURCE_DIR
from plane_lattice import build_offsets
from rgs import sample_operands


def embedding(normal):
    """Construct an orthonormal basis of the hyperplane perpendicular to normal."""
    n = np.asarray(normal, dtype=np.float64)
    n /= np.linalg.norm(n)
    v = n.copy()
    v[-1] += 1
    reflect = np.eye(len(n)) - 2 * np.outer(v, v) / np.dot(v, v)
    return n, reflect[:, :-1].astype(np.float32)


def build_cases(normal, penalties, bins):
    n, p = embedding(normal)
    candidates = np.asarray(list(product(range(-5, 6), repeat=len(n))), dtype=np.int32)
    projected = candidates @ p
    transverse = candidates @ n
    allowed = (np.linalg.norm(projected, axis=1) < 1.5) & (np.abs(transverse) <= 6)
    projected, transverse, candidates = projected[allowed], transverse[allowed], candidates[allowed]
    axes = [(np.arange(bins) + .5) / bins - .5] * len(n)
    fractions = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, len(n))
    position = fractions @ p
    height = fractions @ n
    costs = np.sum(projected ** 2, axis=1) - 2 * position @ projected.T
    extra = transverse ** 2 - 2 * height[:, None] * transverse
    return p, [candidates[np.argmin(costs + weight * extra, axis=1)] for weight in penalties]


def encode(matrix, projection, lut, bins, margin=None):
    count = projection.shape[1]
    projected = matrix.reshape(len(matrix), -1, count) @ projection.T
    if margin is None:
        margin = 127 - np.abs(lut).max()
    scale = np.max(np.abs(projected), axis=(1, 2)) / margin
    scale[scale == 0] = 1
    units = projected / scale[:, None, None]
    cell = np.floor(units * bins + bins / 2).astype(np.int32)
    key = cell[:, :, 0] % bins
    for axis in range(1, cell.shape[2]):
        key = key * bins + cell[:, :, axis] % bins
    encoded = cell // bins + lut[key]
    if encoded.min() < -128 or encoded.max() > 127:
        raise ValueError('Encoding overflow')
    return encoded.reshape(len(matrix), -1).astype(np.float64), scale


def test_case(inputs, projection, table, bins):
    a, b, gold, norm = inputs
    ac, sa = encode(a, projection, table, bins)
    bc, sb = encode(b, projection, table, bins)
    return float(np.linalg.norm((ac @ bc.T) * (sa[:, None] * sb) - gold) / norm)


def operands(rows, k, seed):
    a, b = [item.astype(np.float32) for item in sample_operands((rows, rows, k), seed=seed)]
    gold = a.astype(np.float64) @ b.astype(np.float64).T
    return a, b, gold, np.linalg.norm(gold)


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--max-coeff', type=int, default=7)
    cli.add_argument('--input-count', type=int, choices=(3, 4), default=3)
    cli.add_argument('--output', default='hyperplane-cpu.json')
    args = cli.parse_args()
    values = range(1, args.max_coeff + 1)
    normals = [n for n in combinations_with_replacement(values, args.input_count + 1) if math.gcd(*n) == 1]
    k = 4032
    training = operands(54, k, 4489)
    weight_grid = (.0003, .0007, .0015, .003, .006)
    rows = []
    initial_bins = 8 if args.input_count == 3 else 4
    for index, normal in enumerate(normals):
        p, tables = build_cases(normal, weight_grid, initial_bins)
        for penalty, table in zip(weight_grid, tables):
            rows.append(dict(normal=normal, penalty=penalty, bins=initial_bins,
                             error=test_case(training, p, table, initial_bins)))
        if (index + 1) % 50 == 0:
            print(index + 1, len(normals), min(rows, key=lambda row: row['error']), flush=True)
    rows.sort(key=lambda row: row['error'])
    fine = []
    held_out = [operands(128, k, seed) for seed in (4129, 1171)]
    by_normal = list(dict.fromkeys(row['normal'] for row in rows))[:12]
    for normal in by_normal:
        for bins in ((8, 12, 16) if args.input_count == 3 else (4, 8)):
            for penalty in weight_grid:
                n, p = embedding(normal)
                table = build_offsets(n, p, penalty, bins)
                errors = [test_case(pair, p, table, bins) for pair in held_out]
                fine.append(dict(normal=normal, penalty=penalty, bins=bins, errors=errors, error=float(np.mean(errors))))
        print(normal, min((row for row in fine if row['normal'] == normal), key=lambda row: row['error']), flush=True)
    result = dict(screen=rows, validated=sorted(fine, key=lambda row: row['error']))
    (SOURCE_DIR / 'results' / args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
