"""Compare continuous normal shifts and the archived mixed projection code."""

import argparse
import itertools
import json

import numpy as np

from bench import SOURCE_DIR
from plane_lattice import basis,build_offsets,line
from rgs import sample_operands


NORMALS=((3,4,5),(2,3,4),(3,5,7),(5,7,8),(7,8,9),(7,9,11),
         (9,10,11),(13,14,15),(29,39,51),(4,5,6),(2,5,7),(1,1,1))


def projection(x,p):
    folded=x.reshape(len(x),-1,p.shape[1]) @ p.T
    return folded


def faces(x,normal,blend):
    result=np.max(abs(x),axis=(1,2))*blend
    for j,k in itertools.combinations(range(len(normal)),2):
        face=np.max(np.abs(x[:,:,j]*normal[k]-x[:,:,k]*normal[j]),axis=1)/(normal[j]+normal[k])
        np.maximum(result,face,out=result)
    return result


def codes(x,normal,table,scale,margin,continuous,clip_edge=False):
    values=x/scale[:,None,None]
    if continuous:
        low=np.max((-margin-values)/normal,axis=2)
        high=np.min((margin-values)/normal,axis=2)
        shift=np.minimum(np.maximum(0,low),high)
        values=values+shift[:,:,None]*normal
    bins=np.floor(values*16+8).astype(np.int32)
    key=bins[:,:,0]&15
    for k in range(1,len(normal)):
        key=(key<<4)|(bins[:,:,k]&15)
    digits=(bins>>4)+table[key]
    if not continuous:
        normal=np.asarray(normal,dtype=np.int32)
        lo=np.max(-((127+digits)//normal),axis=2)
        hi=np.min((127-digits)//normal,axis=2)
        overflow=lo>hi
        if np.any(overflow) and not clip_edge:
            raise ValueError('Missing normal translation')
        slide=np.minimum(np.maximum(0,lo),hi)
        if clip_edge and np.any(overflow):
            finite=digits[overflow]
            lower=np.max((-127-finite)/normal,axis=1)
            upper=np.min((127-finite)/normal,axis=1)
            slide[overflow]=np.rint((lower+upper)*.5).astype(np.int32)
        digits += slide[:,:,None]*normal
        if clip_edge:
            np.clip(digits,-127,127,out=digits)
    if np.any(np.abs(digits)>127):
        raise ValueError('Signed byte range exceeded')
    return digits.reshape(len(x),-1).astype(np.float64)


def evaluate(inputs,norm,dt,blend,margin,continuous,dual,clip_edge=False):
    a,b,gold,base=inputs
    _,p,table,line_p,line_table=dt
    prepared=[]
    for x in (a,b):
        a1=projection(x[:,:dual],line_p)
        a2=projection(x[:,dual:],p)
        s=np.maximum(faces(a1,(3,4),.89),faces(a2,norm,blend))/margin
        s=np.where(s>0,s,1)
        qa=codes(a1,(3,4),line_table,s,margin,continuous,clip_edge)
        qb=codes(a2,norm,table,s,margin,continuous,clip_edge)
        prepared.append((np.concatenate((qa,qb),axis=1),s))
    (qa,sa),(qb,sb)=prepared
    error=(qa@qb.T)*(sa[:,None]*sb)-gold
    return float(np.linalg.norm(error)/base)


def problem(m,k,seed):
    a,b=(x.astype(np.float32) for x in sample_operands((m,m,k),seed=seed))
    gold=a.astype(np.float64)@b.astype(np.float64).T
    return a,b,gold,float(np.linalg.norm(gold))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--screen-rows',type=int,default=56)
    p.add_argument('--edge-caps',action='store_true')
    args=p.parse_args()
    line_p,line_lut=line((3,4),.0001,16)
    if args.edge_caps:
        tests=[problem(args.screen_rows,8192,seed) for seed in (2109,43,101)]
        n,plane=basis((3,4,5))
        rows=[]
        for penalty in (.0015,.002,.0025,.003):
            dt=(n,plane,build_offsets(n,plane,penalty,16),line_p,line_lut)
            for blend in (.77,.81,.85):
                for cap in (121,124,126,128,130):
                    errors=[evaluate(t,(3,4,5),dt,blend,cap,False,1024,True) for t in tests]
                    rows.append(dict(penalty=penalty,blend=blend,cap=cap,errors=errors))
        out=SOURCE_DIR/'results'/'edge-cap-scan.json'
        out.write_text(json.dumps(rows,indent=2),encoding='utf-8')
        for row in sorted(rows,key=lambda r:np.mean(r['errors']))[:12]:
            print(row,flush=True)
        return
    screening=problem(args.screen_rows,4096,2109)
    records=[]
    out=SOURCE_DIR/'results'/'centered-mix-scan.json'
    for normal in NORMALS:
        n,plane=basis(normal)
        for penalty in (.001,.002,.004,.007):
            table=build_offsets(n,plane,penalty,16)
            cap=int(126-max(np.abs(table).max(),np.abs(line_lut).max()))
            dt=(n,plane,table,line_p,line_lut)
            for blend in (.55,.65,.75,.81,.9):
                err=evaluate(screening,normal,dt,blend,cap,True,512)
                records.append(dict(normal=normal,penalty=penalty,blend=blend,cap=cap,error=err))
        print(normal,min(records[-20:],key=lambda r:r['error']),flush=True)
        out.write_text(json.dumps(records,indent=2),encoding='utf-8')
    tests=[problem(100,8192,seed) for seed in (2196,43,101)]
    shortlist=sorted(records,key=lambda r:r['error'])[:10]
    for r in shortlist:
        n,plane=basis(r['normal'])
        table=build_offsets(n,plane,r['penalty'],16)
        dt=(n,plane,table,line_p,line_lut)
        r['validation']=[evaluate(t,r['normal'],dt,r['blend'],r['cap'],True,1024) for t in tests]
        print('validation',r,flush=True)
    n,plane=basis((3,4,5))
    dt=(n,plane,build_offsets(n,plane,.002,16),line_p,line_lut)
    baseline=[evaluate(t,(3,4,5),dt,.81,121,False,1024) for t in tests]
    print('baseline',baseline,flush=True)
    out.write_text(json.dumps(dict(records=records,baseline=baseline),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
