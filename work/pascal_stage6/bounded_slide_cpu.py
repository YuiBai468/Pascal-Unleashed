"""Refit rare overflows around a continuously translated lattice target."""

from itertools import combinations, product
import json

import numpy as np

from bench import SOURCE_DIR
from hyperplane_cpu import embedding, operands
from plane_lattice import basis, build_offsets


def lookup_codes(units, table, bins):
    cell = np.floor(units * bins + bins / 2).astype(np.int32)
    key = cell[..., 0] % bins
    for axis in range(1, cell.shape[-1]):
        key = key * bins + cell[..., axis] % bins
    return cell // bins + table[key]


def required_range(projected, normal, blend):
    peak = np.max(np.abs(projected), axis=(1, 2)) * blend
    for a,b in combinations(range(len(normal)),2):
        face = normal[b]*projected[:,:,a] - normal[a]*projected[:,:,b]
        np.maximum(peak,np.max(abs(face),axis=1)/(normal[a]+normal[b]),out=peak)
    return peak


def fit_codes(projected, scale, specification):
    direction, p, table, bins, penalty = specification
    normal = direction / np.linalg.norm(direction)
    units = projected / scale[:, None, None]
    codes = lookup_codes(units, table, bins)
    invalid = np.any(np.abs(codes) > 127, axis=2)
    fringe = units[invalid]
    if len(fringe):
        q = codes[invalid]
        lo_int = np.max(-((127 + q) // direction), axis=1)
        hi_int = np.min((127 - q) // direction, axis=1)
        shift = np.minimum(np.maximum(0, lo_int), hi_int)
        q += shift[:,None]*direction
        room = 126.499 - np.abs(table).max()
        low = np.max((-room - fringe)/direction,axis=1)
        high = np.min((room - fringe)/direction,axis=1)
        near = np.minimum(np.maximum(0,low),high)
        best = np.full(len(fringe),np.inf)
        for move in (None,0,.2,-.2,.4,-.4,.7,-.7):
            if move is None:
                candidate = q.copy()
            else:
                overlap = np.minimum(np.maximum(near+move,low),high)
                candidate = lookup_codes(fringe+overlap[:,None]*direction,table,bins)
            error = np.sum(((candidate-fringe)@p)**2,axis=1) + penalty*(candidate@normal)**2
            take = (error<best) & np.all(np.abs(candidate)<=127,axis=1)
            codes[tuple(v[take] for v in np.nonzero(invalid))] = candidate[take]
            best[take] = error[take]
        if not np.all(np.isfinite(best)):
            raise ValueError('Unrepresentable code')
    return codes.reshape(len(projected), -1).astype(np.float64)


def encode(x, triples, pair, triple, beta, margin):
    left = x[:,:3*triples].reshape(len(x),-1,3) @ triple[1].T
    right = x[:,3*triples:].reshape(len(x),-1,2) @ pair[1].T
    s = np.maximum(required_range(left,triple[0],beta[1]),required_range(right,pair[0],beta[0]))/margin
    s[s==0]=1
    return np.concatenate((fit_codes(left,s,triple),fit_codes(right,s,pair)),axis=1),s


def prepare(normal, penalty, bins):
    n,p = basis(normal) if len(normal)==3 else embedding(normal)
    return np.asarray(normal),p,build_offsets(n,p,penalty,bins),bins,penalty


def main():
    inputs = [operands(128,4096,seed) for seed in (2109,4129,31401)]
    results = []
    for penalty in (.002,.003,.004,.006):
        pair = prepare((3,4,5),.002,16)
        triple = prepare((3,4,5,6),penalty,16)
        max_margin = 126.499 - max(np.abs(c[2]).max() for c in (pair,triple))
        for triples,beta in product((896,1024,1152,1280),(.76,.8,.84,.87,.89)):
            errors=[]
            for a,b,gold,norm in inputs:
                left,sa=encode(a,triples,pair,triple,(.81,beta),max_margin)
                right,sb=encode(b,triples,pair,triple,(.81,beta),max_margin)
                approx=(left@right.T)*(sa[:,None]*sb)
                errors.append(float(np.linalg.norm(approx-gold)/norm))
            results.append(dict(triples=triples,beta=beta,penalty=penalty,margin=float(max_margin),errors=errors,
                                mean=float(np.mean(errors))))
        print(penalty,[(n,min((row for row in results if row['triples']==n),key=lambda r:r['mean'])) for n in (896,1024,1152,1280)],flush=True)
    (SOURCE_DIR/'results'/'bounded-slide-cpu.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
