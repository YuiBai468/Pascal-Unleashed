"""See how much tail energy can be removed by rotating the common K axis."""

import argparse
import json
from time import perf_counter

import numpy as np

from bench import SOURCE_DIR


def stats(x, cutoff):
    residual=np.maximum(np.abs(x)-cutoff,0)
    return float(np.count_nonzero(residual)/residual.size),float(np.sqrt(np.mean(residual**2)))


def rotate(x, cutoff):
    cols=x.shape[1]
    residual=x-np.clip(x,-cutoff,cutoff)
    if not residual.any():
        return x,0.
    cov=np.empty((cols,cols),dtype=np.float32)
    indices=[]
    for j in range(cols):
        active=np.flatnonzero(residual[:,j])
        indices.append(active)
        cov[:,j]=x[active].T@residual[active,j]
    skew=(cov-cov.T)*.5
    steep=np.sum(skew**2,dtype=np.float64)
    if steep==0:
        return x,0.
    curve=sum(np.linalg.norm(x[active]@skew[:,j])**2 for j,active in enumerate(indices))
    initial=steep/max(curve,1e-20)
    best_x=x
    best_loss=stats(x,cutoff)[1]
    best_step=0.
    eye=np.eye(cols,dtype=np.float32)
    for fraction in (1.15,.85,.55,.30):
        delta=np.float32(initial*fraction/2)*skew
        q=np.linalg.solve(eye+delta,eye-delta)
        trial=x@q
        loss=stats(trial,cutoff)[1]
        if loss<best_loss:
            best_x,best_loss,best_step=trial,loss,initial*fraction
    return best_x,best_step


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--rows',type=int,default=16384)
    parser.add_argument('--cols',nargs='+',type=int,default=(32,64,128,256))
    parser.add_argument('--iterations',type=int,default=4)
    parser.add_argument('--cutoff',type=float,default=3.18)
    args=parser.parse_args()
    output=SOURCE_DIR/'results'/f'rotation-tail-{args.rows}.json'
    records=[]
    for d in args.cols:
        for seed in (2109,43):
            rng=np.random.default_rng(seed)
            x=rng.standard_normal((args.rows,d),dtype=np.float32).astype(np.float16).astype(np.float32)
            start=perf_counter()
            measures=[dict(density=stats(x,args.cutoff)[0],rms=stats(x,args.cutoff)[1])]
            energy=np.linalg.norm(x,axis=1)
            for _ in range(args.iterations):
                x,step=rotate(x,args.cutoff)
                density,rms=stats(x,args.cutoff)
                measures.append(dict(density=density,rms=rms,step=step))
            np.testing.assert_allclose(np.linalg.norm(x,axis=1),energy,atol=3e-5,rtol=3e-6)
            row=dict(rows=args.rows,depth=d,seed=seed,cutoff=args.cutoff,seconds=perf_counter()-start,measures=measures)
            records.append(row)
            print(row,flush=True)
            output.write_text(json.dumps(records,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
