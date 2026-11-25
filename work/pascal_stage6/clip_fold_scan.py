"""Scan the full two-byte AR2 range, then check promising candidates on full products."""

import json
from math import erfc, sqrt

import numpy as np

from bench import SOURCE_DIR
from clip_ar2_cpu import ar2_digits, capacity, matmul_error
from rgs import sample_operands


def expected_error(x, probability, base, cap, cutoff):
    clipped=np.minimum(x,cutoff)
    step=cutoff/cap
    q=np.rint(clipped/step).astype(np.int32)
    digits=np.column_stack(ar2_digits(q,base))*step
    weights=np.array([base*(base-1),base],dtype=np.float64)
    energy=probability@(clipped**2)
    gain=(digits.T@(probability*clipped))/energy
    residual=digits-clipped[:,None]*gain
    covariance=residual.T@(probability[:,None]*residual)
    mixed=gain*weights
    bias=(gain@mixed-1)*energy
    variance=bias*bias+2*energy*(mixed@covariance@mixed)+weights@(covariance**2)@weights
    return sqrt(variance)/(probability@(x*x))


def test_case(seed):
    a,b=(v.astype(np.float64) for v in sample_operands((128,128,8192),seed=seed))
    ref=a@b.T
    return a,b,np.sqrt(np.mean(a*a,axis=1)),np.sqrt(np.mean(b*b,axis=1)),ref,np.linalg.norm(ref)


def main():
    x=(np.arange(32768,dtype=np.float64)+.5)/4096
    probability=np.exp(-.5*x*x)
    probability/=probability.sum()
    records=[]
    for base in range(11,21):
        q=np.arange(-base*127,base*127+1)
        h,g=ar2_digits(q,base)
        np.testing.assert_array_equal((base-1)*h+g,q)
        for folded,cap in ((False,capacity(base)-1),(True,base*127)):
            for cutoff in np.arange(2.98,3.241,.01):
                error=expected_error(x,probability,base,cap,cutoff)
                records.append(dict(base=base,clip=float(cutoff),cap=cap,fold=folded,
                                    error=error,tail_fraction=erfc(cutoff/sqrt(2))))
    selected=[]
    for folded in (False,True):
        feasible=[r for r in records if r['fold']==folded and r['error']<=.001015]
        selected.extend(sorted(feasible,key=lambda r:(-r['clip'],r['error']))[:10])
    tests=[test_case(seed) for seed in (2109,43,101)]
    for row in selected:
        spec=tuple(row[name] for name in ('base','clip','cap'))
        row['validation']=[matmul_error(spec,test)[:2] for test in tests]
        print(row,flush=True)
    output=SOURCE_DIR/'results'/'clip-fold-scan.json'
    output.write_text(json.dumps(dict(screen=records,shortlist=selected),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
