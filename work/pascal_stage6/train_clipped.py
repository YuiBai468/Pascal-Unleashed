"""Choose two-byte codebooks for the clipped distribution used in GPU residual correction."""

import argparse
import json
from math import erfc,sqrt,pi

import numpy as np

from bench import SOURCE_DIR
from clip_ar2_cpu import ar2_digits, capacity
from rgs import sample_operands
from train_two_lanes import GRID,LIMIT,STEP,train


def initial(base,fold=False,grid=GRID):
    cap=base*127 if fold else capacity(base)-1
    q=np.rint(grid*(cap/LIMIT)).astype(np.int32)
    h,g=ar2_digits(q,base)
    return np.column_stack((h,g)).astype(np.int8),np.array([base*(base-1),base])*(LIMIT/cap)**2


def validate(model,test,cutoff,step=STEP):
    a,b,ref,scale_a,scale_b,norm=test
    clipped=[np.clip(x,-(s*cutoff)[:,None],(s*cutoff)[:,None]) for x,s in ((a,scale_a),(b,scale_b))]
    ab,bb=clipped
    tables=[]
    for x,scale,code in ((ab,scale_a,model[0]),(bb,scale_b,model[1])):
        sc=scale*cutoff/LIMIT
        digits=np.minimum(len(code)-1,np.rint(np.abs(x)/(sc[:,None]*step)).astype(np.int32))
        packed=code[digits]*np.sign(x)[:,:,None]*sc[:,None,None]
        tables.append(packed)
    ca,cb=tables
    approx=sum(w*(ca[:,:,i]@cb[:,:,i].T) for i,w in enumerate(model[2]))
    target=ab@bb.T
    return float(np.linalg.norm(approx-target)/norm)


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument('--iterations',type=int,default=16)
    cli.add_argument('--fold',action='store_true')
    cli.add_argument('--bases',nargs='+',type=int,default=(12,13,14,15,16))
    cli.add_argument('--cutoffs',nargs='+',type=float,default=(3.,3.04,3.08,3.12,3.16,3.20))
    cli.add_argument('--bins',type=int,default=64)
    args=cli.parse_args()
    step=1/args.bins
    grid=np.arange(LIMIT*args.bins+1,dtype=np.float64)*step
    bank=[]
    for seed in (43,101,4129):
        a,b=(t.astype(np.float64) for t in sample_operands((128,128,8192),seed=seed))
        bank.append((a,b,a@b.T,np.sqrt(np.mean(a*a,axis=1)),np.sqrt(np.mean(b*b,axis=1)),
                     np.linalg.norm(a@b.T)))
    records=[]
    candidates=[]
    prefix='folded' if args.fold else 'clipped'
    for cutoff in args.cutoffs:
        sigma=LIMIT/cutoff
        probability=np.exp(-.5*(grid/sigma)**2)*(step/sigma)*sqrt(2/pi)
        probability[0]*=.5
        probability[-1]=erfc((LIMIT-.5*step)/(sqrt(2)*sigma))
        probability/=probability.sum()
        for base in args.bases:
            model=train(probability,0,args.iterations,start=initial(base,args.fold,grid),grid=grid)
            report=dict(base=base,clip=cutoff,estimated=model[-1],errors=[validate(model,test,cutoff,step) for test in bank],
                        bins=args.bins,fold=args.fold,
                        probabilities='normalized half-normal with clipped mass',weights=model[2].tolist())
            records.append(report)
            candidates.append((cutoff if max(report['errors'])<.000998 else 0,-max(report['errors']),len(records)-1,model))
            print(report,flush=True)
        dest=SOURCE_DIR/'results'
        (dest/f'trained-{prefix}.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
        for p,(_,_,i,model) in enumerate(sorted(candidates,key=lambda x:x[:2],reverse=True)[:4]):
            np.savez(dest/f'{prefix}-lane-{p}.npz',a=model[0],b=model[1],weights=model[2],step=step,
                     metadata=json.dumps(records[i]))


if __name__=='__main__':
    main()
