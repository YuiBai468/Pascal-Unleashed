"""Fit a two-byte AR2 lookup to a clipped left input and a peak-scaled right input."""

import argparse
import json

import numpy as np

from bench import SOURCE_DIR
from rgs import sample_operands
from train_clipped import initial
from train_two_lanes import features, nearest, optimum


def normalize(x, cutoff, bins):
    peak = (np.sqrt(np.mean(x*x,axis=1))*cutoff if cutoff is not None else np.max(abs(x),axis=1))
    scale = np.maximum(peak,1e-20)/127
    clipped = np.clip(x,-peak[:,None],peak[:,None])
    index = np.minimum(127*bins,np.rint(abs(x)*(bins/scale[:,None])).astype(np.int32))
    sign = np.sign(x)
    return scale, index, sign, clipped


def distribution(x, cutoff, bins):
    scale,index,_,_=normalize(x,cutoff,bins)
    weights = np.broadcast_to(scale[:,None]**2,x.shape).ravel()
    counts = np.bincount(index.ravel(),weights=weights,minlength=127*bins+1)
    return counts/counts.sum()


def learn(a,b,weights,prob_a,prob_b,grid,steps):
    af,bf=features(a,prob_a,grid),features(b,prob_b,grid)
    for _ in range(steps):
        a=nearest(grid,*bf,weights)
        af=features(a,prob_a,grid)
        b=nearest(grid,*af,weights)
        bf=features(b,prob_b,grid)
        weights=optimum(af,bf)
    return a,b,weights


def problem(rows,seed,clip,bins):
    a,b=(x.astype(np.float64) for x in sample_operands((rows,rows,16384),seed=seed))
    left,right=normalize(a,clip,bins),normalize(b,None,bins)
    coarse=np.rint(b/right[0][:,None])*right[0][:,None]
    restored=(a-left[-1])@coarse.T
    expected=a@b.T
    return left,right,restored,expected,float(np.linalg.norm(expected))


def error(model, problem):
    a,b,weights=model
    left,right,correction,expected,norm=problem
    packed=[]
    for table,(scale,index,sign,_) in zip((a,b),(left,right)):
        packed.append((table[index]*sign[:,:,None])*scale[:,None,None])
    approx=sum(w*(packed[0][:,:,i]@packed[1][:,:,i].T) for i,w in enumerate(weights))
    return float(np.linalg.norm(approx+correction-expected)/norm)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--clip',type=float,default=3.45)
    p.add_argument('--steps',type=int,default=12)
    p.add_argument('--rows',type=int,default=96)
    args=p.parse_args()
    bins=128
    grid=np.arange(127*bins+1,dtype=np.float64)/bins
    p_a,p_b=(v.astype(np.float32) for v in sample_operands((512,512,16384),seed=20261125))
    prob_a=distribution(p_a,args.clip,bins)
    prob_b=distribution(p_b,None,bins)
    with np.load(SOURCE_DIR/'results'/'folded-lane-0.npz',allow_pickle=False) as baseline:
        original=(baseline['a'],baseline['b'],baseline['weights'])
    checks=[problem(args.rows,seed,args.clip,bins) for seed in (2109,43,101)]
    reference=[error(original,test) for test in checks]
    records=[dict(base=14,clip=args.clip,mode='archived',errors=reference)]
    candidates=[]
    print(records[0],flush=True)
    for base in (13,14,15,16,17):
        code,w=initial(base,True,grid)
        model=learn(code,code,w,prob_a,prob_b,grid,args.steps)
        errors=[error(model,test) for test in checks]
        row=dict(base=base,clip=args.clip,bins=bins,fold=True,mode='asymmetric',errors=errors,
                 weights=model[2].tolist())
        records.append(row)
        candidates.append((max(errors),row,model))
        print(row,flush=True)
    _,settings,(a,b,w)=min(candidates,key=lambda r:r[0])
    stem=f'asymmetric-stage10-{args.clip:.2f}'.replace('.','p')
    dest=SOURCE_DIR/'results'
    (dest/f'{stem}.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    if max(settings['errors'])<max(reference):
        np.savez(dest/f'{stem}.npz',a=a,b=b,weights=w,step=1/bins,metadata=json.dumps(settings))


if __name__=='__main__':
    main()
