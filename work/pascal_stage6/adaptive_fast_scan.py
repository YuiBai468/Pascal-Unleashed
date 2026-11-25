"""Screen the mixed signed-byte encoder's scale and two-byte direction."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from bench import SOURCE_DIR
from centered_mix_scan import codes, faces, projection
from plane_lattice import basis, build_offsets, line
from rgs import sample_operands


def operands(rows, seed):
    return tuple(x.astype(np.float32) for x in sample_operands((rows, rows, 8192), seed=seed))


def prepare(x, dual, plane, line_spec, blend, line_blend):
    a, b = x[:, :dual], x[:, dual:]
    line_coords = projection(a, line_spec[0])
    plane_coords = projection(b, plane[0])
    peak = np.maximum(faces(line_coords, line_spec[2], line_blend), faces(plane_coords, plane[2], blend))
    rms = np.sqrt(np.mean(x*x, axis=1))
    return (a, b), (line_coords, plane_coords), peak, rms


def packed(prepared, specs, margin, rms_cap):
    raw, projected, peak, rms = prepared
    limit = np.minimum(peak, rms_cap*rms) if rms_cap else peak
    scale = np.where(limit > 0, limit/margin, 1)
    parts = []
    for x, p, (_, table, normal) in zip(raw, projected, specs):
        q = codes(p, normal, table, scale, margin, False, True)
        parts.append((x, q*scale[:, None]))
    return parts


def moments(x, z, original_width):
    x = x.reshape(-1, original_width).astype(np.float64)
    z = z.reshape(len(x), -1)
    return x.T@x/len(x), z.T@z/len(x), x.T@z/len(x)


def population_error(left, right):
    total = reference = 0.
    for index, width in enumerate((1, 2)):
        sa, qa, ca = moments(*left[index], width)
        sb, qb, cb = moments(*right[index], width)
        power = np.sum(sa*sb)
        groups = left[index][0].shape[1]//width
        reference += groups*power
        total += groups*(power + np.sum(qa*qb) - 2*np.sum(ca*cb))
    return float(np.sqrt(max(total, 0)/reference))


def full_error(left, right):
    actual = sum(a[1]@b[1].T for a,b in zip(left,right))
    expected = sum(a[0].astype(np.float64)@b[0].astype(np.float64).T for a,b in zip(left,right))
    return float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dual', type=int, default=2048)
    p.add_argument('--rows', type=int, default=64)
    p.add_argument('--output', type=Path, default=SOURCE_DIR/'results'/'slide-stage10-adaptive-cpu.json')
    args = p.parse_args()
    shapes = ((3,4,5), (2,3,4), (3,5,7), (4,5,6), (5,7,8), (7,9,11))
    line_shapes = ((3,4), (4,5), (5,6), (7,9), (8,11))
    lines = {n:(*line(n, .0001, 16), n) for n in line_shapes}
    planes = {}
    for n, penalty in itertools.product(shapes, (.0015,.0025,.004)):
        normal, p = basis(n)
        planes[n,penalty] = (p, build_offsets(normal,p,penalty,16), n)
    data = operands(args.rows, 2109)
    records = []
    baseline = dict(plane=(3,4,5), line=(3,4), penalty=.0025, blend=.81, line_blend=.89, margin=128, rms_cap=0.)
    for n, penalty in planes:
        for l in line_shapes if n==(3,4,5) else ((3,4),):
            specs = lines[l], planes[n,penalty]
            for blend in (.77,.81,.85):
                prepared = [prepare(x,args.dual,specs[1],specs[0],blend,.89) for x in data]
                for cap,rms in ((126,0.),(128,0.),(130,0.),(128,2.65),(128,2.8),(128,2.95)):
                    result = population_error(*(packed(x,specs,cap,rms) for x in prepared))
                    records.append(dict(plane=n,line=l,penalty=penalty,blend=blend,line_blend=.89,margin=cap,rms_cap=rms,
                                        estimate=result))
        print(n, penalty, min(records,key=lambda r:r['estimate']),flush=True)
    candidates = sorted(records,key=lambda r:r['estimate'])[:12]+[baseline]
    checks = [operands(112,seed) for seed in (2109,43,101)]
    for r in candidates:
        specs = lines[r['line']], planes[r['plane'],r['penalty']]
        r['validation'] = [full_error(*(packed(prepare(x,args.dual,specs[1],specs[0],r['blend'],r['line_blend']),
                                               specs,r['margin'],r['rms_cap']) for x in pair)) for pair in checks]
        print(r,flush=True)
    args.output.write_text(json.dumps(dict(dual=args.dual,candidates=candidates,records=records),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
