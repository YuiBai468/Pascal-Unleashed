"""Screen dual (nonidentical) embedding planes using the same single integer dot."""

from itertools import combinations, product
import json

import numpy as np

from bench import SOURCE_DIR
from bounded_slide_cpu import lookup_codes, required_range
from hyperplane_cpu import embedding, operands
from plane_lattice import build_offsets


def dual_bases(direction_a, direction_b):
    a, b = (np.asarray(value, dtype=np.float64) for value in (direction_a, direction_b))
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    c = a @ b
    if c > 1-1e-12:
        _, p = embedding(direction_a)
        return a, b, p, p
    _, _, axes = np.linalg.svd(np.stack((a,b)),full_matrices=True)
    shared = axes[2:].T
    u = (b-c*a)/np.sqrt((1-c*c)*c)
    v = (c*b-a)/np.sqrt((1-c*c)*c)
    pa=np.column_stack((shared,u))
    pb=np.column_stack((shared,v))
    np.testing.assert_allclose(pa.T@pb,np.eye(3),atol=1e-12)
    return a,b,pa.astype(np.float32),pb.astype(np.float32)


def quantize(projected, direction, lut, bins, beta):
    margin = 127-int(np.ceil(max(direction)/2 + np.abs(lut).max() + .5))
    scale = required_range(projected,direction,beta)/margin
    scale[scale==0]=1
    code = lookup_codes(projected/scale[:,None,None],lut,bins)
    n = np.asarray(direction)
    lo=np.max(-((127+code)//n),axis=2)
    hi=np.min((127-code)//n,axis=2)
    if np.any(lo>hi):
        raise ValueError('No feasible normal slide')
    code += np.minimum(np.maximum(0,lo),hi)[:,:,None]*n
    return code.reshape(len(projected),-1).astype(np.float64),scale


def evaluate(a,b,pa,pb,dirs,tables,bins,beta,gold,norm):
    va=a.reshape(len(a),-1,3)@pa.T
    vb=b.reshape(len(b),-1,3)@pb.T
    la,sa=quantize(va,dirs[1],tables[0],bins,beta)
    lb,sb=quantize(vb,dirs[0],tables[1],bins,beta)
    return float(np.linalg.norm((la@lb.T)*(sa[:,None]*sb)-gold)/norm)


def main():
    seeds=(2109,4129,1171)
    training=operands(64,4095,seeds[0]); check=[operands(144,4095,seed) for seed in seeds[1:]]
    normals=((3,4,5,6),(4,5,6,7),(3,5,4,6),(4,3,5,6),(3,4,6,5),(2,4,5,6),(3,4,5,5),(3,4,5,7),
             (3,4,6,7),(4,5,6,8),(3,5,6,7),(4,4,5,6),(3,5,5,6),(4,5,4,5),(2,3,4,5))
    pairs=[(normals[0],normals[0]),*combinations(normals,2)]
    rows=[]
    for dirs in pairs:
        a,b,pa,pb=dual_bases(*dirs)
        for penalty in (.002,.004,.007):
            lut_a=build_offsets(a,pb,penalty,8)
            lut_b=build_offsets(b,pa,penalty,8)
            for beta in (.82,.87,.92):
                err=evaluate(training[0],training[1],pa,pb,dirs,(lut_a,lut_b),8,beta,training[2],training[3])
                rows.append((err,dirs,penalty,beta))
    rows.sort()
    print('screen',rows[:15],flush=True)
    final=[]
    shortlist=list(dict.fromkeys(row[1] for row in rows))[:10]
    if pairs[0] not in shortlist: shortlist.append(pairs[0])
    for dirs in shortlist:
        a,b,pa,pb=dual_bases(*dirs)
        for penalty in (.002,.003,.004,.006):
            lut_a=build_offsets(a,pb,penalty,16)
            lut_b=build_offsets(b,pa,penalty,16)
            for beta in (.81,.84,.87,.9):
                errors=[evaluate(x,y,pa,pb,dirs,(lut_a,lut_b),16,beta,gold,norm) for x,y,gold,norm in check]
                final.append((float(np.mean(errors)),dirs,penalty,beta,errors))
        print(min(row for row in final if row[1]==dirs),flush=True)
    output=SOURCE_DIR/'results'/'oblique-cpu.json'
    output.write_text(json.dumps(dict(screen=rows,validated=sorted(final)),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
