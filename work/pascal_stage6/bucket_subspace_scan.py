"""Fit a shared low-dimensional basis to row bounds and check exact restoration."""

import argparse
import json

import numpy as np

from bench import SOURCE_DIR
from rgs import sample_operands


def directions(width, rank):
    positions = np.arange(width)
    if rank == 1:
        return np.ones((width, 1))
    if rank == 2:
        angle = (2*np.pi/width)*positions
        return np.column_stack((np.cos(angle), np.sin(angle)))
    return (1-2*((positions[:, None] >> np.arange(rank)) & 1))/np.sqrt(rank)


def move_to_box(x, basis, cutoff, passes):
    rows, k = x.shape
    width, rank = basis.shape
    blocks = x.reshape(-1,width)
    bounds = np.repeat(cutoff,k//width)
    active = np.flatnonzero(np.max(abs(blocks),axis=1)>bounds)
    raw = blocks[active]
    limit = bounds[active]
    u = np.zeros((len(active),rank))
    r = np.arange(len(active))
    for _ in range(passes):
        residual=raw-u@basis.T
        worst=np.argmax(abs(residual),axis=1)
        value=residual[r,worst]
        move=np.sign(value)*np.maximum(abs(value)-limit,0)
        u+=move[:,None]*basis[worst]
    coeff=np.zeros((len(blocks),rank))
    coeff[active]=u
    shifted=blocks.copy()
    shifted[active]-=u@basis.T
    shifted=shifted.reshape(x.shape)
    peak=np.maximum(cutoff,np.max(abs(shifted),axis=1))
    return shifted,peak,coeff.reshape(rows,-1)


def encode(x,high,table):
    scale=high/127
    index=np.minimum(len(table)-1,np.rint(abs(x)/scale[:,None]*(len(table)-1)/127).astype(np.int32))
    return table[index]*np.sign(x)[:,:,None]*scale[:,None,None]


def trial(a,b,ref,basis,clip,passes,model):
    rows,k=a.shape
    width,rank=basis.shape
    caps=[clip*np.sqrt(np.mean(x*x,axis=1)) for x in (a,b)]
    (left,la,u),(right,lb,v)=[move_to_box(x,basis,c,passes) for x,c in zip((a,b),caps)]
    sum_b=(b.reshape(rows,-1,width)@basis).reshape(rows,-1)
    sum_a=(left.reshape(rows,-1,width)@basis).reshape(rows,-1)
    correction=u@sum_b.T+sum_a@v.T
    np.testing.assert_allclose(left@right.T+correction,ref,atol=2e-10,rtol=0)
    ea,eb=encode(left,la,model['a']),encode(right,lb,model['b'])
    actual=correction+sum(w*(ea[:,:,i]@eb[:,:,i].T) for i,w in enumerate(model['weights']))
    normalized_peak=np.r_[la/caps[0],lb/caps[1]]*clip
    return dict(width=width,rank=rank,clip=clip,passes=passes,rank_total=rank*k//width,
                peak_mean=float(np.mean(normalized_peak)),peak_p90=float(np.quantile(normalized_peak,.9)),
                relative=float(np.linalg.norm(actual-ref)/np.linalg.norm(ref)))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--rows',type=int,default=128)
    p.add_argument('--k',type=int,default=16384)
    p.add_argument('--seeds',type=int,nargs='+',default=(2109,))
    args=p.parse_args()
    with np.load(SOURCE_DIR/'results'/'folded-lane-0.npz',allow_pickle=False) as file:
        model=dict(file)
    records=[]
    for seed in args.seeds:
        a,b=(x.astype(np.float64) for x in sample_operands((args.rows,args.rows,args.k),seed=seed))
        ref=a@b.T
        for width,rank in ((128,1),(128,2),(256,2),(256,4),(512,4),(512,5),(1024,4)):
            basis=directions(width,rank)
            for clip,passes in ((3.18,2),(3.18,4),(3.13,4),(3.13,8),(3.10,8)):
                result=trial(a,b,ref,basis,clip,passes,model)
                result['seed']=seed
                records.append(result)
                print(result,flush=True)
                (SOURCE_DIR/'results'/'bucket-subspace-scan.json').write_text(json.dumps(records,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
