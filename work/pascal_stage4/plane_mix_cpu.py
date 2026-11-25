"""Screen mixed 2→3 and 1→2 lattice codes against a full FP64 product."""

import argparse
import json

import numpy as np

from bench import SOURCE_DIR
from plane_lattice import build, line
from rgs import sample_operands


def code(projected, scale, table):
    units = projected / scale[:, None, None]
    cells = np.floor(units * 16 + 8).astype(np.int32)
    index = cells[:, :, 0] & 15
    for axis in range(1, cells.shape[2]):
        index = (index << 4) | (cells[:, :, axis] & 15)
    encoded = (cells >> 4) + table[index]
    if encoded.min(initial=0) < -128 or encoded.max(initial=0) > 127:
        raise ValueError('int8 code out of range')
    return encoded.reshape(len(projected), -1).astype(np.float64)


def pack(matrix, plane_basis, plane_lut, line_basis, line_lut, dual):
    plane_projection = matrix[:, dual:].reshape(len(matrix), -1, 2) @ plane_basis.T
    line_projection = matrix[:, :dual, None] @ line_basis.T
    peak = np.maximum(np.max(abs(plane_projection), axis=(1, 2), initial=0),
                      np.max(abs(line_projection), axis=(1, 2), initial=0))
    margin = 127 - max(np.max(abs(plane_lut)), np.max(abs(line_lut)))
    scale = np.where(peak > 0, peak / margin, 1)
    return (np.concatenate((code(plane_projection, scale, plane_lut),
                            code(line_projection, scale, line_lut)), axis=1), scale)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=290)
    args = parser.parse_args()
    a, b = [x.astype(np.float32) for x in sample_operands((192, 192, 4096), seed=args.seed)]
    reference = a.astype(np.float64) @ b.astype(np.float64).T
    scale = np.linalg.norm(reference)
    p, lut = build((3, 4, 5), .002, 16)
    records = []
    for n in ((2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (5, 8), (7, 8), (8, 9)):
        for penalty in (.0001, .0005, .001, .002):
            q, offsets = line(n, penalty, 16)
            for count in (512, 1024, 1536):
                left, sa = pack(a, p, lut, q, offsets, count)
                right, sb = pack(b, p, lut, q, offsets, count)
                approx = (left @ right.T) * (sa[:, None] * sb)
                error = np.linalg.norm(approx - reference) / scale
                records.append(dict(normal=n, penalty=penalty, count=count, error=float(error)))
        best = min(records[-12:], key=lambda row: row['error'])
        print(best, flush=True)
    output = SOURCE_DIR / 'results' / 'plane-mix-cpu.json'
    output.write_text(json.dumps(records, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
