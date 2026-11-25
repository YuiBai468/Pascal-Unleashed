import argparse
import json
from contextlib import ExitStack

import numpy as np

from bench import SOURCE_DIR,upload_many
from cuda import Cublas,Cuda,INT,compile_ptx
from research import measure


def encode(a):
    return np.ascontiguousarray(np.stack([np.packbits(test,axis=1,bitorder='little').view(np.uint32)
                                          for test in (a>0,a<0)],axis=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);p.add_argument('--size',type=int,default=4096);args=p.parse_args()
    n=args.size;shape=(n,)*3
    if n<32 or n%32:raise ValueError('Use a positive multiple of 32')
    rng=np.random.default_rng(87);a=rng.integers(-1,2,(n,n),dtype=np.int8);b=rng.integers(-1,2,(n,n),dtype=np.int8)
    ptx=compile_ptx((SOURCE_DIR/'lowbit_tile.cu').read_bytes());records=[]
    with Cuda(args.gpu) as cuda,cuda.load(ptx) as module,Cublas() as blas:
        for mode in ('ternary','binary'):
            if mode=='binary':a=np.where(a>0,1,-1).astype(np.int8);b=np.where(b>0,1,-1).astype(np.int8)
            expected=a[:32].astype(np.int64)@b[:32].astype(np.int64).T
            with ExitStack() as stack:
                ah,bh=upload_many(cuda,stack,(a,b));ac,bc=upload_many(cuda,stack,(encode(a),encode(b)))
                out=stack.enter_context(cuda.buffer(4*n*n))
                ops=[('cuBLAS INT8',blas.invocation('int8',shape,ah,bh,out,algorithm=1),{})]
                for name,tile in (('ternary32',32),('ternary64',64)) if mode=='ternary' else (('binary64',64),):
                    kernel,info=cuda.kernel(module,name)
                    op=cuda.invocation(kernel,shape,(tile,tile),(ac,bc,out,INT(n),INT(n//32),INT(1)),block=(16,16,1))
                    ops.append((name,op,info))
                for name,op,info in ops:
                    op();np.testing.assert_array_equal(cuda.read(out,(n,n),np.int32)[:32,:32],expected)
                    times=measure(cuda,op,shape,7)
                    print(mode,name,times['median_ms'],times['equivalent_TFLOPS'],flush=True)
                    records.append(dict(mode=mode,kernel=name,performance=times,resources=info))
    (SOURCE_DIR/'results'/'lowbit.json').write_text(json.dumps(records,indent=2),encoding='utf-8')


if __name__=='__main__':main()
