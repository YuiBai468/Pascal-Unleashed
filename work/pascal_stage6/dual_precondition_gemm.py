"""Measure full-product error when dual K transforms replace clipping corrections."""

import argparse
import json
from time import perf_counter

import numpy as np

from bench import SOURCE_DIR
from dual_precondition_scan import step
from rgs import sample_operands


def encode(x, table, cutoff):
    indices=np.minimum(len(table)-1,np.rint(np.abs(x)*((len(table)-1)/cutoff)).astype(np.int32))
    digits=table[indices]*np.where(x<0,-1,1)[:,:,None]
    return tuple(np.ascontiguousarray(digits[:,:,lane],dtype=np.float32) for lane in (0,1))


def trial(a,b,model,cutoff,width,rounds,sweeps):
    start=perf_counter()
    scales=tuple(np.sqrt(np.mean(x.astype(np.float32)**2,axis=1)) for x in (a,b))
    x=a.astype(np.float32)/scales[0][:,None]
    y=b.astype(np.float32)/scales[1][:,None]
    initial=[int(np.count_nonzero(abs(t)>cutoff)) for t in (x,y)]
    for k in range(0,x.shape[1],width):
        end=min(k+width,x.shape[1])
        xx,yy=x[:,k:end],y[:,k:end]
        for _ in range(rounds):
            xx,yy,_,_=step(xx,yy,cutoff,.07,sweeps)
        x[:,k:end]=xx
        y[:,k:end]=yy
    runtime=perf_counter()-start
    final=[int(np.count_nonzero(abs(t)>cutoff)) for t in (x,y)]
    left=encode(x,model[0],cutoff)
    right=encode(y,model[1],cutoff)
    weights=model[2]*(cutoff/127)**2
    exact_b=b.astype(np.float64)
    energy=error=precondition_error=0.
    for k in range(0,len(a),256):
        sl=slice(k,k+256)
        gold=a[sl].astype(np.float64)@exact_b.T
        scale=scales[0][sl,None]*scales[1][None,:]
        decoded=sum(w*(l[sl]@r.T) for w,l,r in zip(weights,left,right))*scale
        transformed=(x[sl]@y.T)*scale
        e=decoded-gold
        t=transformed-gold
        energy+=np.vdot(gold,gold)
        error+=np.vdot(e,e)
        precondition_error+=np.vdot(t,t)
    return dict(relative_frobenius=float(np.sqrt(error/energy)),clipping_counts_initial=initial,
                clipping_counts_final=final,cpu_transform_seconds=runtime,
                invariant_relative_frobenius=float(np.sqrt(precondition_error/energy)),verified_elements=len(a)*len(b))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--rows',type=int,default=2048)
    p.add_argument('--depth',type=int,default=4096)
    p.add_argument('--widths',nargs='+',type=int,default=(32,64))
    p.add_argument('--seeds',nargs='+',type=int,default=(2109,43,101))
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--sweeps',type=int,default=6)
    args=p.parse_args()
    with np.load(SOURCE_DIR/'results'/'folded-lane-0.npz',allow_pickle=False) as data:
        model=(data['a'],data['b'],data['weights'])
        cutoff=json.loads(str(data['metadata']))['clip']
    results=[]
    for seed in args.seeds:
        a,b=sample_operands((args.rows,args.rows,args.depth),seed=seed)
        for d in args.widths:
            report=dict(rows=args.rows,depth=args.depth,width=d,seed=seed,rounds=args.rounds,clip=cutoff)
            report.update(trial(a,b,model,cutoff,d,args.rounds,args.sweeps))
            results.append(report)
            print(report,flush=True)
            dest=SOURCE_DIR/'results'/'dual-precondition-gemm.json'
            dest.write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
