"""Separate tangent error from normal leakage and screen attainable byte-widths."""

from itertools import combinations, product
import json

import numpy as np

from bench import SOURCE_DIR
from bounded_slide_cpu import lookup_codes
from plane_lattice import basis, build_offsets, line
from rgs import sample_operands


def setup(direction, weight, bins=16):
    n = np.asarray(direction,dtype=np.float64)
    n /= np.linalg.norm(n)
    p, lut = (line(direction,weight,bins) if len(direction)==2 else
              (lambda basis_parts: (basis_parts[1],build_offsets(*basis_parts,weight,bins)))(basis(direction)))
    return np.asarray(direction,dtype=np.int32), p.astype(np.float64), n, lut, bins, weight


def gauge(projected,direction):
    result=np.zeros(projected.shape[:-1])
    for i,j in combinations(range(len(direction)),2):
        result=np.maximum(result,abs(direction[j]*projected[...,i]-direction[i]*projected[...,j])/
                          (direction[i]+direction[j]))
    return result


def describe(x,spec,desired_scale,clipped=False,masked=False):
    m,p,n,lut,bins,weight=spec
    projected=x@p.T
    extent=gauge(projected,m)
    limit=127-np.abs(lut).max()-.5
    keep=extent<=desired_scale*limit
    # Resolve constrained point candidates by re-quantizing near the translated target.
    units=projected/desired_scale
    q=lookup_codes(units,lut,bins)
    border=np.any(abs(q)>127,axis=-1)
    if np.any(border):
        z=units[border]
        first=np.max((-limit-z)/m,axis=1)
        last=np.min((limit-z)/m,axis=1)
        center=np.where(first<=last,np.minimum(np.maximum(0,first),last),(first+last)*.5)
        candidate=q[border]
        lower=np.max(np.ceil((-127-candidate)/m),axis=1)
        upper=np.min(np.floor((127-candidate)/m),axis=1)
        slide=np.where(lower<=upper,np.minimum(np.maximum(0,lower),upper),np.rint((lower+upper)*.5))
        normal_q=np.clip(candidate+slide[:,None]*m,-127,127).astype(np.int32)
        best=normal_q.copy()
        best_cost=np.sum(((best-z)@p)**2,axis=1)+weight*(best@n)**2
        if clipped:
            for offset in (0,.15,-.15,.3,-.3,.55,-.55):
                translated=z+(center+offset)[:,None]*m
                code=np.clip(lookup_codes(translated,lut,bins),-127,127)
                cost=np.sum(((code-z)@p)**2,axis=1)+weight*(code@n)**2
                good=cost<best_cost
                best[good],best_cost[good]=code[good],cost[good]
        q[border]=best
    tangent=(q@p)*desired_scale
    residue=(q@n)*desired_scale
    if masked:
        tangent[~keep]=x[~keep]
        residue[~keep]=0
    return tangent,residue,keep


def accuracy(x,tangent,residue):
    # Exact expected product MSE for independently sampled vectors from the empirical distribution.
    y=np.column_stack((tangent,residue))
    inv=1/len(x)
    mx=(x.T@x)*inv
    cross=(x.T@y)*inv
    my=(y.T@y)*inv
    mt=(tangent.T@tangent)*inv
    tx=(x.T@tangent)*inv
    reference=np.sum(mx**2)
    total=np.sum(my**2)+reference-2*np.sum(cross**2)
    plane=np.sum(mt**2)+reference-2*np.sum(tx**2)
    normal=my[-1,-1]**2
    return [float(np.sqrt(max(value,0)/reference)) for value in (total,plane,normal)]


def main():
    gaussian=sample_operands((128,128,8192),seed=126081)[0].astype(np.float64)
    results=[]
    for direction in ((1,2),(2,3),(3,4),(4,5),(4,7),(5,7),(5,8),(6,7),(7,8),(8,9),
                      (2,3,5),(3,4,5),(4,5,6),(4,5,7),(3,5,6),(6,8,9),(29,39,51)):
        d=len(direction)-1
        x=gaussian.reshape(-1,d)[::8].copy()
        values=(.00002,.00007,.0002,.0006,.0012) if d==1 else (.0005,.001,.002,.003,.005)
        for weight in values:
            spec=setup(direction,weight)
            for scale in (.015,.0165,.018,.0195,.021,.0225,.024,.0255,.028,.034,.039,.044):
                for masked in ((False,) if d==1 else (False,True)):
                    t,r,keep=describe(x,spec,scale,True,masked)
                    errors=accuracy(x,t,r)
                    results.append(dict(direction=direction,weight=weight,scale=scale,masked=masked,errors=errors,
                                        outside=float(np.mean(~keep))))
        selected=[r for r in results if r['direction']==direction and not r['masked']]
        print(direction,*(min((r for r in selected if r['scale']==s),key=lambda r:r['errors'][0])
                         for s in (.018,.0225,.0255,.039)),flush=True)
        path=SOURCE_DIR/'results'/'code-characterization.json'
        path.write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
