"""Check which uneven partitions preserve full-size cuBLAS throughput."""

import argparse
from contextlib import ExitStack
import json
import statistics

import numpy as np

from bench import SOURCE_DIR, upload_many
from cuda import Cublas, Cuda
from layout_scan import int8_gemm
from overlap_gpu import shift


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',required=True)
    args=parser.parse_args()
    n,k=4096,5760
    rng=np.random.default_rng(317)
    l=rng.integers(-30,31,(n,k),dtype=np.int8)
    r=rng.integers(-30,31,(k,n),dtype=np.int8)
    rows=[]
    with Cuda(args.gpu) as cuda,Cublas() as blas,ExitStack() as stack:
        a,b=upload_many(cuda,stack,(l,r))
        out=stack.enter_context(cuda.buffer(4*n*n))
        left=int8_gemm(blas,n,n,k,a,b,out,ld_b=n,transpose_b=False,algorithm=1)
        for _ in range(70):
            left()
        for axis in ('row','col'):
            for split in ((4096,),(3584,512),(3328,768),(3072,1024),(2816,1280),(2560,1536),(2304,1792),
                          (2048,2048),(2048,1536,512),(3072,640,384),(3456,640),(3200,896),(2944,1152),
                          (2048,1024,1024),(1536,1536,1024)):
                offset=0
                steps=[]
                for width in split:
                    if axis=='row':
                        step=int8_gemm(blas,width,n,k,shift(a,offset*k),b,shift(out,4*offset*n),
                                       ld_b=n,transpose_b=False,algorithm=1)
                    else:
                        step=int8_gemm(blas,n,width,k,a,shift(b,offset),shift(out,4*offset),ld_b=n,ld_c=n,
                                       transpose_b=False,algorithm=1)
                    offset+=width
                    steps.append(step)

                def run():
                    for stage in steps:stage()

                times=cuda.measure(run,3,5)
                rows.append(dict(axis=axis,split=split,ms=statistics.median(times),samples=times))
                print(rows[-1],flush=True)
        (SOURCE_DIR/'results'/'raw-split.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
