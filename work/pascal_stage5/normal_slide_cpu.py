"""Use the normal direction to fit lattice coordinates inside signed bytes."""

from itertools import combinations
import json

import numpy as np

from bench import SOURCE_DIR
from hyperplane_cpu import embedding, operands
from plane_lattice import basis, build_offsets


def quantize(matrix, normal, projection, lookup, bins, blend):
    normal = np.asarray(normal, dtype=np.int32)
    coordinates = matrix.reshape(len(matrix), -1, projection.shape[1]) @ projection.T
    peak = np.max(np.abs(coordinates), axis=(1, 2))
    face_peak = np.zeros(len(matrix), dtype=np.float32)
    for a, b in combinations(range(len(normal)), 2):
        face = (normal[b] * coordinates[:, :, a] - normal[a] * coordinates[:, :, b])
        np.maximum(face_peak, np.max(np.abs(face), axis=1) / (normal[a] + normal[b]), out=face_peak)
    offsets = np.abs(lookup).max()
    margin = 127 - int(np.ceil(normal.max() / 2 + offsets + .5))
    scale = np.maximum(face_peak, blend * peak) / margin
    scale[scale == 0] = 1
    units = coordinates / scale[:, None, None]
    cells = np.floor(units * bins + bins / 2).astype(np.int32)
    keys = cells[:, :, 0] % bins
    for axis in range(1, cells.shape[2]):
        keys = keys * bins + cells[:, :, axis] % bins
    codes = cells // bins + lookup[keys]
    lower = np.max(-((127 + codes) // normal), axis=2)
    upper = np.min((127 - codes) // normal, axis=2)
    if np.any(lower > upper):
        raise ValueError('Insufficient margin for normal slide')
    slide = np.minimum(np.maximum(0, lower), upper)
    codes += slide[:, :, None] * normal
    return (codes.reshape(len(matrix), -1).astype(np.float64), scale,
            float(np.count_nonzero(slide) / slide.size), float(np.mean(scale)))


def main():
    datasets = [operands(144, 4080, seed) for seed in (2109, 4129, 1171)]
    rows = []
    normals = ((3, 4, 5), (29, 39, 51), (3, 4, 5, 6), (4, 5, 6, 8),
               (4, 6, 7, 9), (2, 3, 4, 5), (3, 4, 5, 6, 7))
    for normal in normals:
        n, p = basis(normal) if len(normal) == 3 else embedding(normal)
        bins = 16 if len(normal) < 5 else 8
        for penalty in (.001, .002, .003, .004, .006, .008, .012):
            lut = build_offsets(n, p, penalty, bins)
            for blend in (.78, .81, .84, .87, .90, .94):
                errors, fractions, scales = [], [], []
                for a, b, gold, denominator in datasets:
                    pa, sa, fa, ga = quantize(a, normal, p, lut, bins, blend)
                    pb, sb, fb, gb = quantize(b, normal, p, lut, bins, blend)
                    estimate = (pa @ pb.T) * (sa[:, None] * sb)
                    errors.append(float(np.linalg.norm(estimate - gold) / denominator))
                    fractions.append((fa+fb)/2)
                    scales.append((ga+gb)/2)
                rows.append(dict(normal=normal, penalty=penalty, bins=bins, blend=blend,
                                 errors=errors, error=float(np.mean(errors)),
                                 shifted=float(np.mean(fractions)), mean_scale=float(np.mean(scales))))
        print(normal, *sorted((r for r in rows if r['normal']==normal),key=lambda r:r['error'])[:2], flush=True)
        (SOURCE_DIR / 'results' / 'normal-slide-cpu.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')


if __name__=='__main__':
    main()
