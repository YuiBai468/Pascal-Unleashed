"""Quantizer sweep allowing an occasional projected error at a cube face."""

from itertools import product
import json

import numpy as np

from bench import SOURCE_DIR
from bounded_slide_cpu import lookup_codes, prepare, required_range
from hyperplane_cpu import operands


def soft_fit(projection, scales, spec):
    m, basis, lut, bins, penalty = spec
    units = projection / scales[:, None, None]
    codes = lookup_codes(units, lut, bins)
    mask = (np.abs(codes) > 127).any(axis=2)
    original = codes[mask]
    low = (-127 - original) / m
    high = (127 - original) / m
    minimum, maximum = np.ceil(np.max(low, axis=1)), np.floor(np.min(high, axis=1))
    feasible = minimum <= maximum
    ordinary = np.minimum(np.maximum(0, minimum), maximum).astype(np.int32)
    fixed = original + ordinary[:, None] * m
    if np.any(~feasible):
        targets = units[mask][~feasible]
        starting = original[~feasible]
        minimum, maximum = minimum[~feasible], maximum[~feasible]
        center = np.rint((minimum + maximum) / 2).astype(np.int32)
        errors = np.full(len(center), np.inf)
        fitted = np.zeros_like(starting)
        normal = m / np.linalg.norm(m)
        for move in (-2,-1,0,1,2):
            candidate = np.clip(starting+(center+move)[:, None]*m,-127,127)
            diff = np.sum(((candidate-targets)@basis)**2,axis=1)+penalty*(candidate@normal)**2
            improved = diff<errors
            fitted[improved] = candidate[improved]
            errors = np.minimum(errors,diff)
        fixed[~feasible] = fitted
    codes[mask] = fixed
    return codes.reshape(len(projection),-1).astype(np.float64),int((~feasible).sum())


def apply(entries, coefficients, margin):
    p, q, required, spec_p, spec_q = entries
    a, b = coefficients
    s = np.maximum(required[0],np.maximum(a*required[1],b*required[2]))/margin
    s[s==0]=1
    left,nleft=soft_fit(p,s,spec_p)
    right,nright=soft_fit(q,s,spec_q)
    return np.concatenate((left,right),axis=1),s,nleft+nright


def main():
    rows=[]
    data=[operands(160,4096,seed) for seed in (2109,4129)]
    for triples, wb in product((768,896,1024,1152,1280),(.003,.004,.006)):
        prepared=[]
        spec_p = prepare((3,4,5),.002,16)
        spec_q = prepare((3,4,5,6),wb,16)
        for a,b,gold,norm in data:
            batches=[]
            for x in (a,b):
                p=x[:,3*triples:].reshape(len(x),-1,2)@spec_p[1].T
                q=x[:,:3*triples].reshape(len(x),-1,3)@spec_q[1].T
                face=np.maximum(required_range(p,spec_p[0],0),required_range(q,spec_q[0],0))
                scales=(face,np.max(abs(p),axis=(1,2)),np.max(abs(q),axis=(1,2)))
                batches.append((p,q,scales,spec_p,spec_q))
            prepared.append((*batches,gold,norm))
        for margin,(beta2,beta3) in product((122,124,126,128,130,132),((.80,.87),(.83,.90),(.86,.91),(.85,.94))):
            errors=[]; edits=[]
            for a,b,gold,norm in prepared:
                left,sa,na=apply(a,(beta2,beta3),margin)
                right,sb,nb=apply(b,(beta2,beta3),margin)
                approx=(left@right.T)*(sa[:,None]*sb)
                errors.append(float(np.linalg.norm(approx-gold)/norm))
                edits.append(na+nb)
            rows.append(dict(triples=triples,penalty=wb,margin=margin,beta2=beta2,beta3=beta3,
                             errors=errors,mean=float(np.mean(errors)),clamped=edits))
        print(triples,wb,min(rows[-24:],key=lambda row:row['mean']),flush=True)
    (SOURCE_DIR/'results'/'soft-bound-cpu.json').write_text(json.dumps(rows,indent=2),encoding='utf8')


if __name__=='__main__':
    main()
