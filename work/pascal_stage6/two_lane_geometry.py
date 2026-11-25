"""Search continuous decoder weights for two signed-byte scalar quantizers."""

import argparse
import itertools
import json
from math import erfc, sqrt

import numpy as np

from bench import SOURCE_DIR
from train_two_lanes import features, normalized_mse, optimum


def nearest(x, sh, sg, share, penalty, room=11):
    wh, wg = (1-share)/sh, share/sg
    normal_h, normal_g = 1/sh, -1/sg
    a=wg*wg+penalty*normal_g*normal_g
    b=wh*wg+penalty*normal_h*normal_g
    choices=np.arange(-room,room+1)
    h=np.clip(np.rint(x[:,None]*sh)+choices,-127,127)
    g=np.clip(np.rint((wg*x[:,None]-b*h)/a),-127,127)
    projection=wh*h+wg*g-x[:,None]
    normal=normal_h*h+normal_g*g
    idx=np.argmin(projection**2+penalty*normal**2,axis=1)
    rows=np.arange(len(x))
    return np.stack((h[rows,idx],g[rows,idx]),axis=1).astype(np.int8)


def search(x,prob,clip,phases):
    power=prob@(x*x)
    options=[]
    step=x[1]-x[0]
    for sh,sg,c,penalty in itertools.product((.94,.97,.99,1.,1.005),(.8,.9,.96,1.,1.005),
                                             phases,(.000024,.000035,.000049,.000067,.00009)):
        digits=nearest(x,sh,sg,c,penalty)
        feat=features(digits,prob,x)
        w=optimum(feat,feat)
        error=normalized_mse(feat,feat,w,power)
        options.append((error,sh,sg,c,penalty,w))
    best=sorted(options,key=lambda r:r[0])[:10]
    return [dict(error=error,clip=clip,sh=sh,sg=sg,share=share,penalty=penalty,
                 weights=w.tolist(),step=float(step)) for error,sh,sg,share,penalty,w in best]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--clips',type=float,nargs='+',default=(3.2,3.32,3.45,3.6,3.8,4.0))
    p.add_argument('--bins',type=int,default=24)
    args=p.parse_args()
    x=np.arange(127*args.bins+1,dtype=np.float64)/args.bins
    rows=[]
    for clip in args.clips:
        sigma=127/clip
        prob=np.exp(-.5*(x/sigma)**2)*sqrt(2/np.pi)/sigma/args.bins
        prob[0]*=.5
        prob[-1]=erfc((127-.5/args.bins)/(sqrt(2)*sigma))
        prob/=prob.sum()
        best=search(x,prob,clip,(-.1,-.075,-.055,.045,.055,.065,.075,.085,.1))
        rows.extend(best)
        print(best[0],flush=True)
        (SOURCE_DIR/'results'/'two-lane-geometry.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
