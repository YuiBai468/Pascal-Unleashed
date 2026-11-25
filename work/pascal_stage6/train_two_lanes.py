"""Train two signed-byte codebooks on product error rather than coordinate error."""

import argparse
import json

import numpy as np

from bench import SOURCE_DIR
from rgs import sample_operands


LIMIT = 127
STEP = 1/64
GRID = np.arange(LIMIT*64+1, dtype=np.float64)*STEP
DIGITS = np.arange(-LIMIT,LIMIT+1,dtype=np.float64)


def features(code, prob, grid=GRID):
    cov = code.T@(prob[:,None]*code)
    cross = code.T@(prob*grid)
    return cov, cross


def nearest(t, cov, cross, weights, limits=(LIMIT,LIMIT)):
    m = cov*weights[:,None]*weights[None,:]
    v = cross*weights
    m += np.eye(2)*max(1e-14,np.trace(m)*1e-13)
    target = np.linalg.solve(m,v)
    out = np.empty((len(t),2),dtype=np.int8)
    choices=DIGITS[LIMIT-limits[0]:LIMIT+limits[0]+1]
    for i in range(0,len(t),256):
        s = slice(i,min(i+256,len(t)))
        dx = choices[None,:] - t[s,None]*target[0]
        center = t[s,None]*target[1]
        q1 = np.clip(np.rint(center - (m[0,1]/m[1,1])*dx),-limits[1],limits[1])
        dy = q1-center
        cost = m[0,0]*dx**2 + (2*m[0,1])*dx*dy + m[1,1]*dy**2
        ix = np.argmin(cost,axis=1)
        out[s,0] = choices[ix]
        out[s,1] = q1[np.arange(len(ix)),ix]
    return out


def optimum(a, b):
    sa,ca = a
    sb,cb = b
    return np.linalg.lstsq(sa*sb,ca*cb,rcond=1e-12)[0]


def normalized_mse(a, b, weights, true_power):
    sa,ca = a
    sb,cb = b
    variance = weights@(sa*sb)@weights
    covariance = weights@(ca*cb)
    return float(np.sqrt(max(0,variance-2*covariance+true_power**2))/true_power)


def initial(kind):
    if kind==0:
        z = np.rint(15*GRID)
        high = np.rint(z/16)
        return np.column_stack((high,np.clip(z-15*high,-127,127))).astype(np.int8),np.array([16/15,16/225])
    ratio = [0,.75,.95,.55,1.2,1.32,.9,.4,.2,.75,.95,.55][kind]
    scale = min(1,1/ratio)
    h,g = np.rint(GRID*scale),np.rint(GRID*scale*ratio)
    c0,c1 = .8,.2
    if kind in (3,7,8): c0,c1=.99,.01
    if kind in (9,10,11): c0,c1=1.25,-.25
    w = np.array([c0/scale**2,c1/(scale*ratio)**2])
    return np.column_stack((h,g)).astype(np.int8),w


def train(prob, kind, iterations, start=None, grid=GRID, limits=(LIMIT,LIMIT)):
    a,w = initial(kind) if start is None else start
    b = a.copy()
    norm = float(prob@(grid**2))
    ac = bc = features(a,prob,grid)
    err = normalized_mse(ac,bc,w,norm)
    for _ in range(iterations):
        a = nearest(grid,*bc,w,limits)
        ac = features(a,prob,grid)
        b = nearest(grid,*ac,w,limits)
        bc = features(b,prob,grid)
        w = optimum(ac,bc)
        err = normalized_mse(ac,bc,w,norm)
    return a,b,w,err


def validate(mapping, test):
    a,b,w,_=mapping
    x,y,gold=test
    encoded=[]
    for input_array, table in ((x,a),(y,b)):
        scales=np.max(abs(input_array),axis=1)/LIMIT
        coordinates=np.minimum(np.rint(abs(input_array)/scales[:,None]/STEP).astype(np.int32),len(GRID)-1)
        values=table[coordinates].astype(np.float64)*np.sign(input_array)[:,:,None]
        encoded.append((values,scales))
    (ca,sa),(cb,sb)=encoded
    out=sum(v*(ca[:,:,i]@cb[:,:,i].T) for i,v in enumerate(w))*(sa[:,None]*sb)
    return float(np.linalg.norm(out-gold)/np.linalg.norm(gold))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--iterations',type=int,default=15)
    args=parser.parse_args()
    tests=[]
    for seed in (43,20261125,1089):
        a,b=sample_operands((128,128,8192),seed=seed)
        a,b=a.astype(np.float64),b.astype(np.float64)
        tests.append((a,b,a@b.T))
    records=[]
    candidates=[]
    for sigma in (28,32,36):
        probability = np.exp(-.5*(GRID/sigma)**2)
        probability /= probability.sum()
        for kind in range(12):
            model = train(probability,kind,args.iterations)
            measured=[validate(model,test) for test in tests]
            record=dict(sigma=sigma,initial=kind,estimated=model[-1],error=measured,weights=model[2].tolist())
            records.append(record)
            candidates.append((max(measured),len(records)-1,model))
            print(record,flush=True)
    output=SOURCE_DIR/'results'
    ranked=sorted(candidates,key=lambda item:item[0])
    for i,(_,j,model) in enumerate(ranked[:6]):
        a,b,w,_=model
        np.savez(output/f'two-lane-{i}.npz',a=a,b=b,weights=w,step=STEP,meta=json.dumps(records[j]))
    (output/'trained-two-lanes.json').write_text(json.dumps(records,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
