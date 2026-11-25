"""Screen exact low-rank range reduction before a two-byte GEMM."""

import argparse
import json

import numpy as np

from bench import SOURCE_DIR
from clip_ar2_cpu import ar2_digits
from rgs import sample_operands


def shift_blocks(x, width, floor, trim):
    groups = x.reshape(len(x), -1, width)
    hi = groups.max(axis=2)
    lo = groups.min(axis=2)
    radius = np.sqrt(np.mean(x*x, axis=1))
    half_range = (hi-lo)*.5
    peak = np.maximum(radius*floor, half_range.max(axis=1)*trim)
    lower = hi-peak[:, None]
    upper = lo+peak[:, None]
    shifts = np.where(lower <= upper, np.clip(0.0, lower, upper), (hi+lo)*.5)
    reduced = (groups-shifts[:, :, None]).reshape(x.shape)
    core = np.clip(reduced, -peak[:, None], peak[:, None])
    return core, shifts, peak, reduced-core


def trial(a, b, ref, width, floor, trim, tables):
    core_a, u, peak_a, tail_a = shift_blocks(a, width, floor, trim)
    core_b, v, peak_b, tail_b = shift_blocks(b, width, floor, trim)
    n, k = a.shape
    sums_b = b.reshape(n, -1, width).sum(axis=2)
    sums_a = (core_a+tail_a).reshape(n, -1, width).sum(axis=2)
    correction = u@sums_b.T+sums_a@v.T
    core_error = core_a@core_b.T+correction-ref
    uncapped = (core_a+tail_a)@(core_b+tail_b).T+correction
    np.testing.assert_allclose(uncapped, ref, rtol=0, atol=1e-9)
    norm = np.linalg.norm(ref)
    values = []
    for base, model in tables:
        if model is None:
            sa, sb = peak_a/(base*127), peak_b/(base*127)
            aw = ar2_digits(np.rint(core_a/sa[:, None]).astype(np.int32), base)
            bw = ar2_digits(np.rint(core_b/sb[:, None]).astype(np.int32), base)
            weights = (base*(base-1), base)
            label = str(base)
        else:
            step = float(model['step'])
            aw, bw = [], []
            sa, sb = (peak*abs(1/(step*(len(model['a'])-1))) for peak in (peak_a, peak_b))
            for x, s, label, destination in ((core_a, sa, 'a', aw), (core_b, sb, 'b', bw)):
                entries = model[label]
                index = np.minimum(len(entries)-1, np.rint(np.abs(x)/s[:, None]).astype(np.int32))
                destination.extend((entries[index]*np.sign(x)[:, :, None]).transpose(2, 0, 1))
            sa, sb = sa/step, sb/step
            weights = model['weights']
            label = 'trained'
        accum = sum(w*(ad.astype(np.float64)@bd.astype(np.float64).T) for w, ad, bd in zip(weights, aw, bw))
        delta = accum*(sa[:, None]*sb)-core_a@core_b.T
        values.append(dict(encoder=label, error=float(np.linalg.norm(core_error+delta)/norm),
                           exact_tails=float(np.linalg.norm(delta)/norm)))
    return dict(width=width, rank=k//width, floor=floor, trim=trim,
                mean_peak=float((peak_a.mean()+peak_b.mean())*.5),
                tails=int(np.count_nonzero(tail_a)+np.count_nonzero(tail_b)),
                tail_error=float(np.linalg.norm(core_error)/norm), models=values)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rows', type=int, default=96)
    p.add_argument('--k', type=int, default=16384)
    p.add_argument('--seeds', type=int, nargs='+', default=(2109,))
    args = p.parse_args()
    with np.load(SOURCE_DIR/'results'/'folded-lane-0.npz', allow_pickle=False) as store:
        model = dict(store)
    tables = [(14, None), (15, None), (16, None), (14, model)]
    rows = []
    for seed in args.seeds:
        a, b = (v.astype(np.float64) for v in sample_operands((args.rows,args.rows,args.k),seed=seed))
        ref = a@b.T
        for width in (32,64,128,256):
            for floor, trim in ((0.,1.),(3.,1.),(3.16,.99),(3.16,.97),(3.20,.975),(3.20,1.)):
                result=trial(a,b,ref,width,floor,trim,tables)
                result['seed']=seed
                rows.append(result)
                print({**result, 'models': min(result['models'],key=lambda v:v['error'])},flush=True)
                (SOURCE_DIR/'results'/'bucket-shift-scan.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
