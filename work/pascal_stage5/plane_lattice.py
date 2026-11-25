"""Two FP16 coordinates projected onto three signed bytes, sharing a batch scale."""

import itertools
import json

import numpy as np

from bench import SOURCE_DIR
from rgs import sample_operands


def basis(normal):
    n=np.asarray(normal,dtype=np.float64)
    n/=np.linalg.norm(n)
    u=np.array([n[1],-n[0],0.0]);u/=np.linalg.norm(u)
    return n,np.stack((u,np.cross(n,u)),axis=1).astype(np.float32)


def build(normal, penalty, bins):
    n,p=basis(normal)
    return p, build_offsets(n, p, penalty, bins)


def line(normal, penalty, bins):
    n = np.asarray(normal, dtype=np.float64)
    n /= np.linalg.norm(n)
    p = np.array([[n[1]], [n[0] * -1]], dtype=np.float32)
    return p, build_offsets(n, p, penalty, bins)


def build_offsets(n, p, penalty, bins):
    dimensions = len(n)
    candidates=np.array(list(itertools.product(range(-6,7),repeat=dimensions)),dtype=np.int32)
    proj=candidates@p.astype(np.float64)
    signed=candidates@n
    valid=(np.linalg.norm(proj,axis=1)<1.5)&(abs(signed)<=7)
    candidates,proj,signed=candidates[valid],proj[valid],signed[valid]
    axis=(np.arange(bins)+.5)/bins-.5
    frac=np.stack(np.meshgrid(*([axis]*dimensions),indexing='ij'),axis=-1).reshape(-1,dimensions)
    positions=frac@p.astype(np.float64)
    normals=frac@n
    last=np.sum(proj**2,axis=1)+penalty*signed**2
    indices=[]
    for start in range(0,len(frac),2048):
        sl=slice(start,start+2048)
        score=last-2*(positions[sl]@proj.T+penalty*normals[sl,None]*signed)
        indices.append(np.argmin(score,axis=1))
    return candidates[np.concatenate(indices)]


def encode(x,p,lut,bins,margin):
    co=x.astype(np.float32).reshape(len(x),-1,2)@p.T
    s=np.max(np.abs(co),axis=(1,2))/margin
    s[s==0]=1
    v=co/s[:,None,None]
    near=np.rint(v).astype(np.int32)
    sub=np.clip(((v-near+.5)*bins).astype(np.int32),0,bins-1)
    offset=lut[(sub[:,:,0]*bins+sub[:,:,1])*bins+sub[:,:,2]]
    code=near+offset
    if code.min() < -128 or code.max() > 127:
        raise AssertionError('coordinate does not fit int8')
    return code.reshape(len(x),-1).astype(np.int8),s


def main():
    shape=(160,160,4096)
    a,b=sample_operands(shape,seed=290)
    ref=a.astype(np.float64)@b.astype(np.float64).T
    norm=np.linalg.norm(ref)
    rows=[]
    bins=16
    for normal in ((1,3,6),(2,3,6),(2,4,5),(3,4,5),(4,4,5),(4,5,6),(3,5,6),(5,6,7),(3,4,4),(3,5,5)):
        for penalty in (.0003,.0007,.001,.0015,.002,.003):
            p,lut=build(normal,penalty,bins)
            margin=127-np.max(np.abs(lut))
            u,s=encode(a,p,lut,bins,margin);v,t=encode(b,p,lut,bins,margin)
            approx=(u.astype(np.float64)@v.astype(np.float64).T)*(s[:,None]*t)
            error=np.linalg.norm(approx-ref)/norm
            rows.append(dict(normal=normal,penalty=penalty,bins=bins,margin=int(margin),error=float(error)))
        print(normal, min(rows[-6:],key=lambda r:r['error']),flush=True)
    (SOURCE_DIR/'results'/'plane-cpu.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')


if __name__=='__main__':main()
