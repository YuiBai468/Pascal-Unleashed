"""Time both K block transforms on P40 with GPU-resident prepared inputs."""

import argparse
from contextlib import ExitStack
import ctypes as ct
import json
from statistics import median

import numpy as np

from bench import SOURCE_DIR
from cuda import Cublas,Cuda,DEV_PTR,FLOAT,INT,PTR,SIZE,UINT,api,library_path


def trial(cuda,blas,mm,rows,depth,width):
    batches=depth//width
    size=rows*width
    alpha,beta=FLOAT(1),FLOAT(0)
    q=np.eye(width,dtype=np.float32)
    q[0,1]=.125
    dual=np.linalg.inv(q.T)
    with ExitStack() as stack:
        def mem(data):
            return stack.enter_context(cuda.buffer(data))

        lhs,rhs=mem(4*rows*depth),mem(4*rows*depth)
        out_a,out_b=mem(4*rows*depth),mem(4*rows*depth)
        qa,qb=(mem(np.broadcast_to(a,(batches,width,width)).copy()) for a in (q,dual))
        fill=api(cuda.driver,'cuMemsetD32_v2',DEV_PTR,UINT,SIZE)
        for source in (lhs,rhs):
            fill(source,0x3f800000,rows*depth)

        def complete():
            for transform,source,target in ((qa,lhs,out_a),(qb,rhs,out_b)):
                mm(blas.handle,0,0,width,rows,width,ct.byref(alpha),transform,width,width*width,
                   source,width,size,ct.byref(beta),target,width,size,batches)

        complete()
        for out,transform in ((out_a,q),(out_b,dual)):
            np.testing.assert_array_equal(cuda.read(out,(1,width),np.float32)[0],transform.sum(axis=0))
            last=DEV_PTR(out.value+4*(rows*depth-width))
            np.testing.assert_array_equal(cuda.read(last,(1,width),np.float32)[0],transform.sum(axis=0))
        samples=cuda.measure(complete,2,9)
        elapsed=float(median(samples))
        return dict(rows_per_operand=rows,depth=depth,width=width,batches=batches,median_ms=elapsed,
                    fp32_tflops=4*rows*depth*width/elapsed/1e9,samples_ms=samples)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gpu',required=True)
    p.add_argument('--rows',type=int,default=8192)
    p.add_argument('--depth',type=int,default=16384)
    p.add_argument('--widths',type=int,nargs='+',default=(32,64,128))
    args=p.parse_args()
    if any(w<2 or args.depth%w for w in args.widths):
        p.error('Use transform widths of at least two that divide the depth')
    results=[]
    i64=ct.c_int64
    with Cuda(args.gpu) as cuda,Cublas() as blas:
        dll=ct.WinDLL(str(library_path()/'cublas64_12.dll'))
        mm=api(dll,'cublasSgemmStridedBatched',PTR,INT,INT,INT,INT,INT,PTR,DEV_PTR,INT,i64,
               DEV_PTR,INT,i64,PTR,DEV_PTR,INT,i64,INT)
        for width in args.widths:
            report=trial(cuda,blas,mm,args.rows,args.depth,width)
            results.append(report)
            print(report,flush=True)
            path=SOURCE_DIR/'results'/'dual-transform-gpu-probe.json'
            path.write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
