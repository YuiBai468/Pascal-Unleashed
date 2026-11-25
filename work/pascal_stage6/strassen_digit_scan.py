"""Check the precision cost of reserving headroom for exact INT8 Strassen sums."""

import json
from itertools import product
from math import erfc,sqrt

import numpy as np

from bench import SOURCE_DIR
from rgs import sample_operands
from train_clipped import validate
from train_two_lanes import GRID,LIMIT,STEP,train


def initial(radix,gain,margin):
    cap=127*radix+margin
    target=np.rint(GRID*cap/LIMIT)
    h=np.rint(target/(radix+gain))
    low=np.maximum(-127,np.ceil((target-margin)/radix))
    high=np.minimum(127,np.floor((target+margin)/radix))
    h=np.clip(h,low,high)
    g=target-radix*h
    weight=(radix+gain)*np.array((radix,1/gain))*(LIMIT/cap)**2
    return np.column_stack((h,g)).astype(np.int8),weight


def main():
    tests=[]
    for seed in (2109,43,101):
        a,b=(t.astype(np.float64) for t in sample_operands((88,88,4096),seed=seed))
        ref=a@b.T
        tests.append((a,b,ref,np.sqrt(np.mean(a*a,axis=1)),np.sqrt(np.mean(b*b,axis=1)),np.linalg.norm(ref)))
    output=SOURCE_DIR/'results'/'strassen-digit-scan.json'
    all_rows=[]
    for margin,clip in product((63,31),(2.72,2.90,3.06,3.18)):
        sigma=LIMIT/clip
        probability=np.exp(-.5*(GRID/sigma)**2)*STEP/sigma*sqrt(2/np.pi)
        probability[0]*=.5
        probability[-1]=erfc((LIMIT-.5*STEP)/(sqrt(2)*sigma))
        probability/=probability.sum()
        tested=[]
        for radix,gain in product((8,10,12,14),(.3,.5)):
            model=train(probability,0,7,start=initial(radix,gain,margin),limits=(127,margin))
            tested.append((model[-1],radix,gain,model))
        estimated,radix,gain,model=min(tested,key=lambda x:x[0])
        m=dict(margin=margin,clip=clip,radix=radix,gain=gain,estimated=estimated,
               measured=[validate(model,t,clip) for t in tests],tail_density=erfc(clip/sqrt(2)))
        all_rows.append(m)
        output.write_text(json.dumps(all_rows,indent=2),encoding='utf-8')
        print(m,flush=True)


if __name__=='__main__':
    main()
