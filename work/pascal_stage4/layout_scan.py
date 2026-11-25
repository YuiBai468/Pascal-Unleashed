import argparse
import ctypes as ct
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from bench import SOURCE_DIR, upload_many
from cuda import Cublas, Cuda, DEV_PTR, INT


def int8_gemm(blas, m, n, k, a, b, c, *, ld_a=None, ld_b=None, ld_c=None,
              transpose_a=False, transpose_b=True, algorithm=1, beta=0):
    """Return a row-major output operation. Transposed flags describe the cuBLAS arguments."""
    ld_a=k if ld_a is None else ld_a
    ld_b=k if ld_b is None else ld_b
    ld_c=n if ld_c is None else ld_c
    one,carry=INT(1),INT(beta)
    return lambda:blas.gemm_ex(blas.handle,int(transpose_b),int(transpose_a),n,m,k,ct.byref(one),
                               b,3,ld_b,a,3,ld_a,ct.byref(carry),c,10,ld_c,72,algorithm)


def time(cuda, operation, samples):
    ms=cuda.measure(operation,2,samples)
    return dict(median_ms=float(np.median(ms)),samples_ms=ms)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',required=True)
    parser.add_argument('--k',type=int,nargs='+',default=(5440,5632,6144,6656,6912,7168,7296,7424,7520))
    parser.add_argument('--pads',type=int,nargs='+',default=(0,16,64,128))
    parser.add_argument('--algorithms',type=int,nargs='+',default=(-1,1))
    parser.add_argument('--samples',type=int,default=3)
    parser.add_argument('--output',type=Path,default=SOURCE_DIR/'results'/'layout-scan.json')
    args=parser.parse_args()
    n=4096; stride=max(args.k)+max(args.pads)
    rng=np.random.default_rng(137)
    data=[rng.integers(-32,33,(n,stride),dtype=np.int8) for _ in range(2)]
    records=[]; unsupported=[]
    with Cuda(args.gpu) as cuda,Cublas() as blas,ExitStack() as stack:
        a,b=upload_many(cuda,stack,data)
        c=stack.enter_context(cuda.buffer(4*n*n))
        for k in args.k:
            candidates=[]
            for tb,ta in ((True,False),(False,False),(True,True),(False,True)):
                for pad in args.pads if (tb and not ta) else (0,):
                    stride_a=k+pad if not ta else n
                    stride_b=k+pad if tb else n
                    for algorithm in args.algorithms:
                        op=int8_gemm(blas,n,n,k,a,b,c,ld_a=stride_a,ld_b=stride_b,
                                     transpose_a=ta,transpose_b=tb,algorithm=algorithm)
                        request=dict(k=k,pad=pad,ta=ta,tb=tb,algorithm=algorithm)
                        try:
                            result=request|time(cuda,op,args.samples)
                        except RuntimeError as problem:
                            unsupported.append(request|dict(error=str(problem)))
                            continue
                        candidates.append((result,op))
            ranked=sorted(candidates,key=lambda x:x[0]['median_ms'])
            for row,op in ranked[:4]:
                op()
                sample=cuda.read(c,(12,n),np.int32)[:,:12]
                A=data[0].reshape(-1)[:n*(n if row['ta'] else k+row['pad']) if not row['ta'] else k*n]
                B=data[1].reshape(-1)[:n*(k+row['pad']) if row['tb'] else k*n]
                A=(A.reshape(k,n)[:,:12].T if row['ta'] else A.reshape(n,k+row['pad'])[:12,:k])
                B=(B.reshape(n,k+row['pad'])[:12,:k] if row['tb'] else B.reshape(k,n)[:,:12].T)
                np.testing.assert_array_equal(sample,A.astype(np.int64)@B.astype(np.int64).T)
                row['checked']=True
            records.extend(row for row,_ in ranked)
            if ranked:print('K',k,'best',ranked[0][0],flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(device=cuda.name,blas=blas.version,records=records,unsupported=unsupported),indent=2),encoding='utf-8')


if __name__=='__main__':main()
