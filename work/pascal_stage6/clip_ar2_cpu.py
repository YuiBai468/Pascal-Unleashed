"""Evaluate two-byte AR2 with separately restored Gaussian tails."""

from itertools import product
import json

import numpy as np

from bench import SOURCE_DIR
from rgs import sample_operands


def capacity(base):
    q=np.arange(base*128,dtype=np.int32)
    h=np.rint(q.astype(np.float64)/base).astype(np.int32)
    g=q-(base-1)*h
    out=np.flatnonzero((h>127)|(g>127)|(g<-127))
    return int(out[0]-1)


def ar2_digits(q, base):
    h=np.rint(q.astype(np.float64)/base).astype(np.int32)
    g=q-(base-1)*h
    # Move along the lattice line only when the nearest high digit overflows the mixed digit.
    adjustment=(g>127).astype(np.int32)-(g<-127).astype(np.int32)
    h+=adjustment
    g-=(base-1)*adjustment
    if np.max(abs(h))>127 or np.max(abs(g))>127:
        raise AssertionError('Packing failure')
    return h,g


def quantize(x, cutoff, base, cap):
    scale=cutoff/cap
    core=np.clip(x,-cutoff[:,None],cutoff[:,None])
    q=np.rint(core/scale[:,None]).astype(np.int32)
    h,g=ar2_digits(q,base)
    return h.astype(np.float64)*scale[:,None],g.astype(np.float64)*scale[:,None],core


def matmul_error(spec, test):
    base,factor,cap=spec
    a,b,rad_a,rad_b,gold,norm=test
    ha,ga,ca=quantize(a,rad_a*factor,base,cap)
    hb,gb,cb=quantize(b,rad_b*factor,base,cap)
    error=(base*base-base)*(ha@hb.T)+base*(ga@gb.T)-ca@cb.T
    delta_a,delta_b=a-ca,b-cb
    # Test one cheaper sparse patch, using only the high byte for the dense factor.
    coarse_a = base*ha
    coarse_b = base*hb + delta_b
    coarse_error=error+delta_a@(coarse_b-b).T+(coarse_a-ca)@delta_b.T
    bound=float(np.linalg.norm(error)/norm)
    approximation=float(np.linalg.norm(coarse_error)/norm)
    return bound, approximation, np.count_nonzero(delta_a)/len(a), np.count_nonzero(delta_b)/len(b)


def main():
    full=[]
    for seed in (2109,4129,31401):
        a,b=(v.astype(np.float64) for v in sample_operands((128,128,8192),seed=seed))
        gold=a@b.T
        full.append((a,b,np.sqrt(np.mean(a*a,axis=1)),np.sqrt(np.mean(b*b,axis=1)),gold,np.linalg.norm(gold)))
    records=[]
    for base,factor in product(range(11,23),np.arange(2.75,3.51,.05)):
        cap=capacity(base)
        r=matmul_error((base,float(factor),cap),full[0])
        records.append(dict(base=base,clip=float(factor),cap=cap,exact_tail=r[0],coarse_tail=r[1],counts=r[2:]))
    feasible=[r for r in records if r['exact_tail']<.00104]
    ranked=sorted(feasible,key=lambda r:(max(r['counts']),r['coarse_tail']))[:14]
    ranked.extend(sorted(records,key=lambda r:r['coarse_tail'])[:4])
    for r in ranked:
        r['validation']=[matmul_error((r['base'],r['clip'],r['cap']),t) for t in full[1:]]
        print(r,flush=True)
    (SOURCE_DIR/'results'/'clip-ar2-cpu.json').write_text(json.dumps(dict(screen=records,shortlist=ranked),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
