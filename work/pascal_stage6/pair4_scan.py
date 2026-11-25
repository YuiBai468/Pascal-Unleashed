"""Search four-byte signed embeddings of a two-dimensional input block."""

import argparse
import json
from itertools import product
from math import cos,sin,sqrt

import numpy as np

from bench import SOURCE_DIR
from rgs import sample_operands


def model(t, angle):
    r=np.array(((cos(angle),-sin(angle)),(sin(angle),cos(angle))),dtype=np.float64)
    c=1/sqrt(1+t*t)
    projection=c*np.concatenate((np.eye(2),t*r),axis=0)
    normal=c*np.concatenate((-t*r.T,np.eye(2)),axis=0)
    axes=np.concatenate((np.eye(2),r),axis=0)
    support=127*(np.abs(axes).sum(axis=1)+t*np.abs(axes@r.T).sum(axis=1))
    return r,c,projection,normal,axes/support[:,None]


def lookup(t, spec, penalty, bins, radius):
    r,c,p,n,_=spec
    slice_4d=np.array(list(product(range(-radius,radius+1),repeat=2)))
    high=-t*slice_4d@r
    small=np.array(list(product((-1,0,1),repeat=2)))
    around=np.rint(high).astype(np.int32)[:,None,:]+small[None,:,:]
    lower=np.broadcast_to(slice_4d[:,None,:],around.shape)
    offsets=np.concatenate((around,lower),axis=2).reshape(-1,4)
    tangent=offsets@p
    normal=offsets@n
    keep=np.sum(tangent*tangent,axis=1)<1.75**2
    offsets,tangent,normal=offsets[keep],tangent[keep],normal[keep]
    axis=(np.arange(bins)+.5)/bins-.5
    phase=np.stack(np.meshgrid(axis,axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,4)
    norm=np.sum(tangent*tangent,axis=1)+penalty*np.sum(normal*normal,axis=1)
    target_t=phase@p
    target_n=phase@n
    answer=[]
    for i in range(0,len(phase),256):
        scores=norm-2*(target_t[i:i+256]@tangent.T+penalty*(target_n[i:i+256]@normal.T))
        answer.append(offsets[np.argmin(scores,axis=1)])
    return np.concatenate(answer).astype(np.int16)


def constrained(x, t, spec, penalty, radius):
    r,c,*_=spec
    z=x/c
    initial=x*c
    clipped_h=np.clip(initial,-127,127)
    center=(t*(z-clipped_h)+penalty*t*clipped_h)/(t*t+penalty)
    low_center=np.clip(np.rint(center@r.T),-127,127)
    grid=np.array(list(product(range(-radius,radius+1),repeat=2)))
    chosen=[]
    for i in range(0,len(x),64):
        target=z[i:i+64,None,:]
        at=np.clip(low_center[i:i+64,None,:]+grid,-127,127)
        v=at@r
        high=np.clip(np.rint((target-t*(1-penalty)*v)/(1+penalty*t*t)),-127,127)
        tangent=high+t*v-target
        normal=v-t*high
        err=np.sum(tangent*tangent+penalty*normal*normal,axis=2)
        best=np.argmin(err,axis=1)
        rows=np.arange(len(best))
        chosen.append(np.column_stack((high[rows,best],at[rows,best])))
    return np.concatenate(chosen).astype(np.int16)


def encode(x,t,spec,penalty,bins,lut,high_sigma,radius):
    r,c,p,n,axes=spec
    pairs=x.reshape(len(x),-1,2)
    rms=np.sqrt(np.mean(x*x,axis=1))
    required=np.max(np.abs((pairs/c)@axes.T),axis=(1,2))
    scale=np.maximum(rms*c/high_sigma,required*1.002)
    scale[scale==0]=1.
    scaled=pairs/scale[:,None,None]
    target=scaled@p.T
    near=np.rint(target).astype(np.int16)
    frac=np.minimum(bins-1,((target-near+.5)*bins).astype(np.int32))
    addr=((frac[:,:,0]*bins+frac[:,:,1])*bins+frac[:,:,2])*bins+frac[:,:,3]
    digits=near+lut[addr]
    invalid=(np.abs(digits)>127).any(axis=2)
    if invalid.any():
        digits[invalid]=constrained(scaled[invalid],t,spec,penalty,radius)
    return digits.reshape(len(x),-1).astype(np.float64),scale,float(invalid.mean())


def trial(a,b,gold,t,spec,penalty,bins,lut,high_sigma,radius):
    h,sa,fa=encode(a,t,spec,penalty,bins,lut,high_sigma,radius)
    g,sb,fb=encode(b,t,spec,penalty,bins,lut,high_sigma,radius)
    out=(h@g.T)*(sa[:,None]*sb)
    return float(np.linalg.norm(out-gold)/np.linalg.norm(gold)),(fa+fb)/2


def dataset(shape,seed):
    a,b=(v.astype(np.float64) for v in sample_operands(shape,seed=seed))
    return a,b,a@b.T


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bins',type=int,default=8)
    p.add_argument('--radius',type=int,default=9)
    p.add_argument('--shape',nargs=3,type=int,default=(52,52,4096))
    args=p.parse_args()
    test=dataset(args.shape,2109)
    out=SOURCE_DIR/'results'/'pair4-scan.json'
    records=[]
    for t,angle in product((.20,.26,.32,.40,.49),(.17,.39,.64,.7853981633974483)):
        spec=model(t,angle)
        for penalty in (.00006,.00015,.00035,.0008):
            lut=lookup(t,spec,penalty,args.bins,args.radius)
            for high_sigma in (34,39,44,49):
                error,fraction=trial(*test,t,spec,penalty,args.bins,lut,high_sigma,args.radius)
                records.append(dict(t=t,angle=angle,penalty=penalty,sigma=high_sigma,error=error,boundary=fraction))
        print(t,angle,min(records[-16:],key=lambda x:x['error']),flush=True)
        out.write_text(json.dumps(records,indent=2),encoding='utf-8')
    verification=[dataset((120,120,8192),seed) for seed in (2109,43,101)]
    for row in sorted(records,key=lambda x:x['error'])[:8]:
        t,angle,penalty,high_sigma=(row[k] for k in ('t','angle','penalty','sigma'))
        spec=model(t,angle)
        lut=lookup(t,spec,penalty,12,14)
        row['validated']=[trial(*test,t,spec,penalty,12,lut,high_sigma,14) for test in verification]
        print('validation',row,flush=True)
        out.write_text(json.dumps(records,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
