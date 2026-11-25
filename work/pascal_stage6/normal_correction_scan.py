"""Measure tangent error after spending a separate dot product on the normal component."""

import itertools
import json
from statistics import median

import numpy as np

from bench import SOURCE_DIR
from hyperplane_cpu import embedding
from rgs import sample_operands


def table_for_tube(n,p,extent,bins):
    dim=len(n)
    core=np.rint(np.arange(-extent,extent+.25,.25)[:,None]*n).astype(np.int16)
    neighbors=np.array(list(itertools.product(range(-1,2),repeat=dim)),dtype=np.int16)
    offsets=np.unique((core[:,None,:]+neighbors).reshape(-1,dim),axis=0)
    projected=offsets@p
    normal=offsets@n
    mask=(np.sum(projected**2,axis=1)<=1.5**2)&(np.abs(normal)<extent)
    offsets,projected,normal=offsets[mask],projected[mask],normal[mask]
    axis=(np.arange(bins)+.5)/bins-.5
    mesh=np.stack(np.meshgrid(*([axis]*dim),indexing='ij'),axis=-1).reshape(-1,dim)
    tangent=mesh@p
    normal_target=mesh@n
    def select(penalty):
        norm=np.sum(projected**2,axis=1)+penalty*normal**2
        out=[]
        for i in range(0,len(mesh),512):
            a=tangent[i:i+512]@projected.T
            r=normal_target[i:i+512,None]*normal
            out.append(offsets[np.argmin(norm-2*(a+penalty*r),axis=1)])
        return np.concatenate(out)
    return select


def encode(x,p,n,lut,bins,cap):
    dim=len(n)
    physical=x.reshape(len(x),-1,dim-1)@p.T
    scale=np.max(np.abs(physical),axis=(1,2))/cap
    scale[scale==0]=1
    units=physical/scale[:,None,None]
    nearest=np.rint(units).astype(np.int16)
    fractions=np.clip(np.floor((units-nearest+.5)*bins).astype(np.int32),0,bins-1)
    address=fractions[:,:,0]
    for j in range(1,dim):
        address=address*bins+fractions[:,:,j]
    codes=nearest+lut[address]
    bad=np.any(abs(codes)>127,axis=2)
    codes[bad]=nearest[bad]
    raw_normal=codes@n
    return codes.reshape(len(x),-1).astype(np.float64),raw_normal,scale,float(bad.mean())


def quantized(x,bits):
    if bits==1:
        return np.sign(x)*np.mean(abs(x),axis=1,keepdims=True)
    maximum=(1<<(bits-1))-1
    scale=np.max(abs(x),axis=1,keepdims=True)/maximum
    scale[scale==0]=1
    return np.rint(x/scale)*scale


def trial(data,n,p,lut,bins,cap):
    a,b,gold,denominator=data
    ac,ar,sa,fa=encode(a,p,n,lut,bins,cap)
    bc,br,sb,fb=encode(b,p,n,lut,bins,cap)
    scale=sa[:,None]*sb
    raw=ac@bc.T
    exact=ar@br.T
    def error(x):
        return float(np.linalg.norm((raw-x)*scale-gold)/denominator)
    return dict(exact=error(exact),uncorrected=error(0),fallback=(fa+fb)/2,
                corrected={str(bits):error(quantized(ar,bits)@quantized(br,bits).T) for bits in (1,2,3,8)},
                normal_rms=float(np.sqrt(np.mean(ar**2))))


def main():
    shape=(112,112,4080)
    tests=[]
    for seed in (2109,43,101):
        a,b=(x.astype(np.float64) for x in sample_operands(shape,seed=seed))
        gold=a@b.T
        tests.append((a,b,gold,np.linalg.norm(gold)))
    records=[]
    directions=((7,9,11),(29,39,51),(149,191,233),(9,11,13,15),
                (29,33,41,47),(149,173,181,211),(31,37,41,47,53))
    for direction in directions:
        n,p=embedding(direction)
        bins=16 if len(n)==3 else 8 if len(n)==4 else 4
        for radius in (16,32,56):
            options=table_for_tube(n,p,radius,bins)
            for penalty in (0,1e-5,1e-4):
                lookup=options(penalty)
                for cap in (88,104,116):
                    row=dict(direction=direction,radius=radius,penalty=penalty,cap=cap)
                    row.update(trial(tests[0],n,p,lookup,bins,cap))
                    records.append(row)
        print(direction,min((r for r in records if r['direction']==direction),key=lambda r:r['exact']),flush=True)
        (SOURCE_DIR/'results'/'normal-correction-scan.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    for dim in (3,4,5):
        ranked=sorted((r for r in records if len(r['direction'])==dim),key=lambda r:r['exact'])[:3]
        for row in ranked:
            n,p=embedding(row['direction'])
            bins=16 if dim==3 else 8 if dim==4 else 4
            lookup=table_for_tube(n,p,row['radius'],bins)(row['penalty'])
            row['validation']=[trial(t,n,p,lookup,bins,row['cap']) for t in tests[1:]]
            print(dim,median([row['exact'],*(x['exact'] for x in row['validation'])]),row,flush=True)
    (SOURCE_DIR/'results'/'normal-correction-scan.json').write_text(json.dumps(records,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
