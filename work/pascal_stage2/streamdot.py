import argparse
import json
from contextlib import ExitStack

import numpy as np

from bench import SOURCE_DIR,upload_many
from cuda import Cuda,INT,compile_ptx
from microbench import pack


def partition(values,width):
    shape=values.shape
    tail=(-shape[-1])%width
    padded=np.pad(values,[(0,0)]*(len(shape)-1)+[(0,tail)])
    return padded.reshape(*shape[:-1],-1,width)


def fixture(a,b,name):
    if name=='packed2':
        a=(a[:,0]+64*a[:,1]);p=pack(partition(a,4),8).T.copy(); q=pack(partition(b,4),8).T.copy()
    elif name=='packed3':
        a=a[:,0]+128*a[:,1]+16384*a[:,2];p=pack(partition(a,2),16).T.copy();q=pack(partition(b,2),8).T.copy()
    elif name.startswith('plain'):
        p=pack(partition(a,4),8).transpose(2,1,0).copy();q=pack(partition(b,4),8).T.copy()
    elif name=='binary_pop2':
        p=pack(partition(a>0,32),1).transpose(2,1,0).copy();q=pack(partition(b>0,32),1).T.copy()
    else:
        p=np.stack([pack(partition(a>0,32),1).transpose(2,1,0),pack(partition(a<0,32),1).transpose(2,1,0)],axis=2)
        q=np.stack([pack(partition(b>0,32),1).T,pack(partition(b<0,32),1).T],axis=1)
    return np.ascontiguousarray(p),np.ascontiguousarray(q)


def main():
    cli=argparse.ArgumentParser();cli.add_argument('--gpu',required=True);args=cli.parse_args()
    ptx=compile_ptx((SOURCE_DIR/'streamdot.cu').read_bytes())
    count,k=8192,843
    rng=np.random.default_rng(610); raw=rng.integers(-1,2,(count,3,k),dtype=np.int32)
    other=rng.integers(-1,2,(count,k),dtype=np.int32)
    records=[]
    with Cuda(args.gpu) as cuda,cuda.load(ptx) as module:
        print('GPU',cuda.name,flush=True)
        for label in ('plain2','packed2','ternary_pop2','binary_pop2','plain3','packed3'):
            c=int(label[-1]);a,b=raw[:,:c],other
            if label=='binary_pop2':a=np.where(a>0,1,-1);b=np.where(b>0,1,-1)
            p,q=fixture(a,b,label)
            expected=np.einsum('ijk,ik->ij',a,b)
            with ExitStack() as stack:
                x,y=upload_many(cuda,stack,(p,q));o=stack.enter_context(cuda.buffer(expected.nbytes))
                fn,info=cuda.kernel(module,label)
                op=cuda.grid_invocation(fn,(count//128,1,1),(128,1,1),(x,y,o,INT(p.shape[0]),INT(k)))
                op();np.testing.assert_array_equal(cuda.read(o,expected.shape,np.int32),expected)
                ms=cuda.measure(op,7,9); median=float(np.median(ms)); rate=2*count*c*k/median/1e9
                print(label,round(median,5),'ms, effective TOPS',round(rate,3),flush=True)
                records.append(dict(name=label,outputs=c,k=k,rows=count,timings_ms=ms,
                                    median_ms=median,effective_TOPS=rate,resources=info))
    (SOURCE_DIR/'results'/'streamdot.json').write_text(json.dumps(records,indent=2),encoding='utf-8')


if __name__=='__main__':main()
