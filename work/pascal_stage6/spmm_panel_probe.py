"""Check whether sparse panels remain resident in the P40's L2 cache."""

import argparse
from contextlib import ExitStack
import ctypes as ct
import json
from statistics import median

import numpy as np

from bench import SOURCE_DIR, upload_many
from clip_spmm_gpu import SparseLibrary
from cuda import Cuda, DEV_PTR, FLOAT, SIZE


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gpu',required=True)
    p.add_argument('--shape',type=int,nargs=3,default=(8192,8192,16384))
    p.add_argument('--nnz-row',type=int,default=43)
    p.add_argument('--panels',type=int,nargs='+',default=(32,64,128,256,512,1024,2048,4096,8192))
    args=p.parse_args()
    m,n,k=args.shape
    rng=np.random.default_rng(53)
    indices=np.stack([np.sort(rng.choice(k,args.nnz_row,replace=False)).astype(np.int32) for _ in range(m)])
    offsets=np.arange(m+1,dtype=np.int32)*args.nnz_row
    values=rng.integers(-5,6,size=m*args.nnz_row,dtype=np.int8).astype(np.float16)
    dense=rng.integers(-3,4,size=(k,n),dtype=np.int8).astype(np.float16)
    expected=np.sum(values[:args.nnz_row].astype(np.float32)[:,None]*dense[indices[0]].astype(np.float32),axis=0)
    records=[]
    with ExitStack() as stack:
        cuda=stack.enter_context(Cuda(args.gpu))
        cusparse=SparseLibrary(stack)
        off,idx,val,b=upload_many(cuda,stack,(offsets,indices,values,dense))
        yr=stack.enter_context(cuda.buffer(4*m*n))
        sa=cusparse.csr(m,k,m*args.nnz_row,off,idx,val)
        stack.callback(cusparse.destroy_sparse,sa)
        alpha,beta=FLOAT(1),FLOAT(0)
        for width in args.panels:
            if n%width: continue
            with ExitStack() as panel:
                calls=[]
                max_bytes=0
                for j in range(0,n,width):
                    desc_b=ct.c_void_p()
                    desc_c=ct.c_void_p()
                    cusparse.create_dense(ct.byref(desc_b),k,width,n,DEV_PTR(b.value+2*j),2,2)
                    cusparse.create_dense(ct.byref(desc_c),m,width,n,DEV_PTR(yr.value+4*j),0,2)
                    panel.callback(cusparse.destroy_dense,desc_b)
                    panel.callback(cusparse.destroy_dense,desc_c)
                    args_mm=(cusparse.handle,0,0,ct.byref(alpha),sa,desc_b,ct.byref(beta),desc_c,0,6)
                    size=SIZE()
                    cusparse.size(*args_mm,ct.byref(size))
                    max_bytes=max(max_bytes,size.value)
                    calls.append(args_mm)
                work=panel.enter_context(cuda.buffer(max(1,max_bytes)))

                def operation():
                    for mm in calls: cusparse.mm(*mm,work)

                times=cuda.measure(operation,2,9)
                np.testing.assert_array_equal(cuda.read(yr,(n,),np.float32),expected)
                record=dict(width=width,median_ms=float(median(times)),samples=times,shape=(m,n,k),nnz_per_row=args.nnz_row)
                records.append(record)
                print(record,flush=True)
    (SOURCE_DIR/'results'/'spmm-panels.json').write_text(json.dumps(records,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
