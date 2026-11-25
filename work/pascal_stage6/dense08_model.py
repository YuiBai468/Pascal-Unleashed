"""Tune scalar two-byte lattice codes whose dot product uses one INT8 GEMM."""

import json
from math import gcd

import numpy as np

from bench import SOURCE_DIR
from rgs import sample_operands


def codebook(u,v,cap,penalty):
    digits=np.arange(-127,128,dtype=np.int32)
    h,g=np.meshgrid(digits,digits,indexing='ij')
    q=(u*h+v*g).ravel()
    r=(-v*h+u*g).ravel()
    order=np.lexsort((abs(r),q))
    ordered=q[order]
    first=np.r_[True,np.diff(ordered)!=0]
    best=order[first]
    p=q[best]
    r=r[best]
    width=min(len(p),65)
    targets=np.arange(cap+1,dtype=np.int32)
    start=np.clip(np.searchsorted(p,targets)-width//2,0,len(p)-width)
    indices=start[:,None]+np.arange(width)
    err=(p[indices]-targets[:,None])**2+penalty*r[indices]**2
    chosen=best[indices[np.arange(len(targets)),np.argmin(err,axis=1)]]
    table=np.column_stack((h.ravel()[chosen],g.ravel()[chosen])).astype(np.int8)
    table[0]=0
    return table


def data(seed,shape):
    a,b=(v.astype(np.float64) for v in sample_operands(shape,seed=seed))
    out=a@b.T
    return a,b,out,np.linalg.norm(out)


def error(test,table,d,cutoff):
    a,b,gold,norm=test
    cap=len(table)-1
    packed=[]
    for x in (a,b):
        maxabs=np.max(abs(x),axis=1)
        limit=np.minimum(maxabs,cutoff*np.sqrt(np.mean(x*x,axis=1)))
        scale=limit/cap
        key=np.minimum(cap,np.rint(abs(x)/scale[:,None]).astype(np.int32))
        code=(table[key]*np.sign(x)[:,:,None]).reshape(len(x),-1)
        packed.append((code,scale))
    (h,sa),(g,sb)=packed
    c=(h@g.T)*(d*sa[:,None]*sb)
    return float(np.linalg.norm(c-gold)/norm)


def main():
    test=data(2109,(48,48,4096))
    results=[]
    for u in range(4,18):
        for v in range(1,min(u,10)):
            if gcd(u,v)!=1:
                continue
            for fraction in (.75,.84,.93,1.):
                cap=int(127*(u+v)*fraction)
                for penalty in (.00035,.001,.0025):
                    digits=codebook(u,v,cap,penalty)
                    distance=error(test,digits,u*u+v*v,np.inf)
                    results.append(dict(u=u,v=v,cap=cap,penalty=penalty,error=distance))
        dest=SOURCE_DIR/'results'/'dense08-model-scan.json'
        dest.write_text(json.dumps(results,indent=2),encoding='utf-8')
        print(u,min(results,key=lambda x:x['error']),flush=True)
    verify=[data(seed,(112,112,8192)) for seed in (2109,43,101)]
    best=[]
    for row in sorted(results,key=lambda x:x['error'])[:16]:
        u,v,cap,penalty=(row[k] for k in ('u','v','cap','penalty'))
        digits=codebook(u,v,cap,penalty)
        row['validation']=[error(test,digits,u*u+v*v,np.inf) for test in verify]
        best.append((max(row['validation']),row,digits))
    for i,(_,row,digits) in enumerate(sorted(best,key=lambda x:x[0])[:4]):
        print(row,flush=True)
        np.savez(SOURCE_DIR/'results'/f'dense08-model-{i}.npz',table=digits,metadata=json.dumps(row))
    dest.write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
