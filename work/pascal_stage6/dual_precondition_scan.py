"""Test an exact dual K transform computed from sparse near-tail constraints."""

import argparse
import json

import numpy as np

from bench import SOURCE_DIR
from rotation_tail_scan import stats


def constraints(x, cutoff, guard, sign):
    by_axis=[]
    for j in range(x.shape[1]):
        indices=np.flatnonzero(abs(x[:,j])>=cutoff-guard)
        values=x[indices,j]
        samples=x[indices]
        if len(indices):
            goal=np.clip(values,-cutoff+guard/4,cutoff-guard/4)
            gram=samples@samples.T
            gram.flat[::len(indices)+1]+=1e-3
            inverse=np.linalg.solve(gram,samples).T
            by_axis.append((j,samples,inverse,sign*(goal-values)))
    return by_axis


def step(x,y,cutoff,guard,sweeps):
    left=constraints(x,cutoff,guard,1)
    right=constraints(y,cutoff,guard,-1)
    d=x.shape[1]
    delta=np.zeros((d,d),dtype=np.float32)
    for _ in range(sweeps):
        for view,targets in ((delta,left),(delta.T,right)):
            for j,samples,inverse,wanted in targets:
                current=view[:,j]
                current+=inverse@(wanted-samples@current)
    best=(x,y,0.)
    best_loss=sum(stats(v,cutoff)[1]**2 for v in (x,y))
    identity=np.eye(d,dtype=np.float32)
    for slope in (1.,.65):
        change=identity+np.float32(slope)*delta
        changed_x=x@change
        changed_y=y@np.linalg.solve(change.T,identity)
        loss=sum(stats(v,cutoff)[1]**2 for v in (changed_x,changed_y))
        if loss<best_loss:
            best=(changed_x,changed_y,slope)
            best_loss=loss
    return *best,len(left)+len(right)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--rows',type=int,default=8192)
    p.add_argument('--widths',type=int,nargs='+',default=(32,48,64,96,128))
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--sweeps',type=int,default=6)
    p.add_argument('--cutoff',type=float,default=3.2)
    args=p.parse_args()
    results=[]
    for d in args.widths:
        for seed in (2109,43):
            rng=np.random.default_rng(seed)
            x,y=(rng.standard_normal((args.rows,d),dtype=np.float32).astype(np.float16).astype(np.float32)
                 for _ in range(2))
            gold=x[:128].astype(np.float64)@y[:128].astype(np.float64).T
            report=dict(rows=args.rows,depth=d,seed=seed,cutoff=args.cutoff,initial=[stats(v,args.cutoff) for v in (x,y)],
                        rounds=[])
            for _ in range(args.rounds):
                x,y,slope,active=step(x,y,args.cutoff,.07,args.sweeps)
                actual=x[:128].astype(np.float64)@y[:128].astype(np.float64).T
                invariant=float(np.linalg.norm(actual-gold)/np.linalg.norm(gold))
                if invariant>1e-5:
                    raise AssertionError(f'Nonmatching inverse transform: {invariant}')
                report['rounds'].append(dict(stats=[stats(v,args.cutoff) for v in (x,y)],
                                             slope=slope,active_axes=active,invariance_error=invariant))
            results.append(report)
            print(report,flush=True)
            dest=SOURCE_DIR/'results'/'dual-precondition-scan.json'
            dest.write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
