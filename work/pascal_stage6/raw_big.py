"""Measure the actual INT8 bandwidth of the large-shape NVIDIA DP4A path."""

import argparse
from contextlib import ExitStack
from itertools import product
import json
from statistics import median

import numpy as np

from bench import SOURCE_DIR
from cuda import Cublas,Cuda,DEV_PTR,UINT,SIZE,api
from layout_scan import int8_gemm


def trial(cuda,blas,m,n,k,algo):
    with ExitStack() as stack:
        a,b,c=(stack.enter_context(cuda.buffer(v)) for v in (m*k,n*k,4*m*n))
        fill=api(cuda.driver,'cuMemsetD32_v2',DEV_PTR,UINT,SIZE)
        fill(a,0x01010101,m*k//4)
        fill(b,0x01010101,n*k//4)
        core=int8_gemm(blas,m,n,k,a,b,c,ld_b=n,transpose_b=False,algorithm=algo)
        for _ in range(5):
            core()
        actual=cuda.read(c,(3,n),np.int32)
        if not np.all(actual==k):
            raise AssertionError('Integer product incorrect')
        samples=cuda.measure(core,2,9)
        ms=float(median(samples))
        return dict(m=m,n=n,k=k,algo=algo,ms=ms,tops=2*m*n*k/ms/1e9,samples=samples)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',required=True)
    parser.add_argument('--side',nargs='+',type=int,default=(8192,12288,16384))
    parser.add_argument('--depth',nargs='+',type=int,default=(8192,16384))
    parser.add_argument('--shape',nargs=3,type=int,action='append',metavar=('M','N','K'))
    parser.add_argument('--output',default='raw-big.json')
    args=parser.parse_args()
    records=[]
    with Cuda(args.gpu) as cuda,Cublas() as blas:
        shapes=args.shape or [(n,n,k) for n,k in product(args.side,args.depth)]
        for m,n,k in shapes:
            for algo in (1,-1):
                r=trial(cuda,blas,m,n,k,algo)
                records.append(r)
                print(r,flush=True)
                (SOURCE_DIR/'results'/args.output).write_text(json.dumps(records,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
