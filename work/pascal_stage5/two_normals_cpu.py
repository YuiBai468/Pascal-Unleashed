"""Check whether a joint six-to-eight lattice code can replace two triplet codes."""

from itertools import product
import json

import numpy as np

from bench import SOURCE_DIR
from hyperplane_cpu import operands


def reduce_basis(metric):
    rank=len(metric)
    integer=np.eye(rank,dtype=np.int64)

    def gram():
        q, r=np.linalg.qr(metric@integer)
        return q,r

    _,r=gram()
    k=1
    while k<rank:
        for i in range(k-1,-1,-1):
            repeat=int(np.rint(r[i,k]/r[i,i]))
            if repeat:
                integer[:,k]-=repeat*integer[:,i]
                _,r=gram()
        if r[k,k]**2 + r[k-1,k]**2 >= .96*r[k-1,k-1]**2:
            k+=1
        else:
            integer[:,[k-1,k]]=integer[:,[k,k-1]]
            _,r=gram()
            k=max(1,k-1)
    q,r=gram()
    return integer,q,r


def setup(normal,penalty):
    q, _=np.linalg.qr(normal,mode='complete')
    n,p=q[:,:2],q[:,2:]
    weight=np.eye(len(normal))-(1-np.sqrt(penalty))*(n@n.T)
    return p,weight,reduce_basis(weight)


def nearest(target, weighted_target, weight, basis, quality):
    integers,q,r=basis
    y=weighted_target@q
    best=np.zeros_like(target,dtype=np.int32)
    cost=np.full(len(y),np.inf)
    for delta in range(-quality,quality+1):
        coefficients=np.zeros_like(y)
        for j in range(y.shape[1]-1,-1,-1):
            tail=coefficients[:,j+1:]@r[j,j+1:]
            coefficients[:,j]=np.rint((y[:,j]-tail)/r[j,j])
            if j==y.shape[1]-1:
                coefficients[:,j]+=delta
        candidate=coefficients@integers.T
        total=np.sum(((candidate-target)@weight)**2,axis=1)
        updated=total<cost
        best[updated]=candidate[updated]
        cost[updated]=total[updated]
    return best


def encode(x, normal, model, beta, quality):
    p, metric, basis=model
    xa=x.astype(np.float32).reshape(len(x),-1,p.shape[1])
    position=xa@p.T
    scale=beta*np.max(np.abs(position),axis=(1,2))/124
    scale[scale==0]=1
    z=(position/scale[:,None,None]).reshape(-1,len(normal))
    best=nearest(z,z,metric,basis,quality)
    overflow=np.any(abs(best)>127,axis=1)
    increments=np.asarray(list(product(range(-7,8),repeat=2)))@normal.T
    selected=np.flatnonzero(overflow)
    for start in range(0,len(selected),512):
        ii=selected[start:start+512]
        base=best[ii,None,:]+increments[None,:,:]
        clip=np.clip(base,-127,127)
        score=np.sum(((clip-z[ii,None,:])@metric)**2,axis=2)
        best[ii]=clip[np.arange(len(ii)),np.argmin(score,axis=1)]
    return best.reshape(len(x),-1).astype(np.float64),scale


def test(data,normal,model,beta,quality):
    a,b,gold,norm=data
    aa,sa=encode(a,normal,model,beta,quality)
    bb,sb=encode(b,normal,model,beta,quality)
    return float(np.linalg.norm((aa@bb.T)*(sa[:,None]*sb)-gold)/norm)


def main():
    rng=np.random.default_rng(317)
    centered=np.array([[3,4,5,6,0,0,0,0],[0,0,0,0,3,4,5,6]],dtype=np.int32).T
    candidates=[centered]
    for levels in (1,2,3):
        for _ in range(22):
            change=rng.integers(-levels,levels+1,centered.shape,dtype=np.int32)
            candidates.append(centered+change)
    data=operands(34,4092,1709)
    records=[]
    for i,normal in enumerate(candidates):
        for penalty in (.002,.004,.006):
            model=setup(normal,penalty)
            for beta in (.85,.91):
                err=test(data,normal,model,beta,1)
                records.append((err,i,penalty,beta))
    selected=sorted(records)[:12]
    print('screen',selected,flush=True)
    held_out=[operands(112,4092,seed) for seed in (4129,31401)]
    checked=[]
    for _,i,penalty,beta in selected:
        model=setup(candidates[i],penalty)
        errors=[test(data,candidates[i],model,beta,2) for data in held_out]
        checked.append(dict(coefficients=candidates[i].tolist(),penalty=penalty,beta=beta,errors=errors,mean=float(np.mean(errors))))
        print(i,penalty,beta,errors,flush=True)
    output=SOURCE_DIR/'results'/'two-normals-cpu.json'
    output.write_text(json.dumps(dict(screen=sorted(records),validated=checked),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
