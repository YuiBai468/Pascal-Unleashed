"""Check the installed cuSPARSE generic CSR kernels with Gaussian-tail density."""

import argparse
from contextlib import ExitStack
import ctypes as ct
import json
import os
from statistics import median

import numpy as np

from bench import SOURCE_DIR, upload_many
from cuda import Cuda, DEV_PTR, FLOAT, INT, PTR, SIZE, api, library_path


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument('--gpu',required=True)
    cli.add_argument('--shape',nargs=3,type=int,default=(4096,4096,8192))
    cli.add_argument('--nnz-row',type=int,default=24)
    cli.add_argument('--layout',choices=('rr','rc','cr','cc'),default='rr')
    cli.add_argument('--output',default='spmm-probe.json')
    args=cli.parse_args()
    m,n,k=args.shape
    t=args.nnz_row
    rng=np.random.default_rng(117)
    indices=np.stack([np.sort(rng.choice(k,t,replace=False)).astype(np.int32) for _ in range(m)])
    offsets=np.arange(m+1,dtype=np.int32)*t
    dense=rng.integers(-9,10,size=(k,n),dtype=np.int8)
    values=rng.integers(-30,31,size=m*t,dtype=np.int8)
    expected=np.sum(values[:t,None].astype(np.int32)*dense[indices[0],:32],axis=0)
    i64=ct.c_int64
    results=[]
    with Cuda(args.gpu) as device,os.add_dll_directory(str(library_path())),ExitStack() as stack:
        dll=ct.WinDLL(str(library_path()/'cusparse64_12.dll'))
        h=PTR()
        api(dll,'cusparseCreate',ct.POINTER(PTR))(ct.byref(h))
        stack.callback(api(dll,'cusparseDestroy',PTR),h)
        create_sparse=api(dll,'cusparseCreateCsr',ct.POINTER(PTR),i64,i64,i64,DEV_PTR,DEV_PTR,DEV_PTR,INT,INT,INT,INT)
        create_dense=api(dll,'cusparseCreateDnMat',ct.POINTER(PTR),i64,i64,i64,DEV_PTR,INT,INT)
        get_size=api(dll,'cusparseSpMM_bufferSize',PTR,INT,INT,PTR,PTR,PTR,PTR,PTR,INT,INT,ct.POINTER(SIZE))
        gemm=api(dll,'cusparseSpMM',PTR,INT,INT,PTR,PTR,PTR,PTR,PTR,INT,INT,DEV_PTR)
        for dtype,num,out_type,name in ((np.int8,3,10,'i8'),(np.float16,2,0,'f16'),(np.float32,0,0,'f32')):
            with ExitStack() as trial:
                dense_input=dense if args.layout[0]=='r' else dense.T.copy()
                off,col,val,mat=upload_many(device,trial,(offsets,indices,values.astype(dtype),dense_input.astype(dtype)))
                raw=trial.enter_context(device.buffer(m*n*4))
                a,b,c=PTR(),PTR(),PTR()
                create_sparse(ct.byref(a),m,k,m*t,off,col,val,2,2,0,num)
                trial.callback(api(dll,'cusparseDestroySpMat',PTR),a)
                dense_args=((b,k,n,mat,num,args.layout[0]),(c,m,n,raw,out_type,args.layout[1]))
                for desc,rows,cols,data,type_id,layout in dense_args:
                    create_dense(ct.byref(desc),rows,cols,cols if layout=='r' else rows,data,type_id,2 if layout=='r' else 1)
                    trial.callback(api(dll,'cusparseDestroyDnMat',PTR),desc)
                scalar=INT if out_type==10 else FLOAT
                alpha=scalar(1)
                beta=scalar(0)
                for algo in (0,4,6,7):
                    params=(h,0,0,ct.byref(alpha),a,b,ct.byref(beta),c,out_type,algo)
                    size=SIZE()
                    result=dict(dtype=name,algo=algo,nnz=m*t,shape=(m,n,k),layout=args.layout)
                    try:
                        get_size(*params,ct.byref(size))
                        with device.buffer(max(1,size.value)) as work:
                            timings=device.measure(lambda:gemm(*params,work),2,9)
                        dtype_out=np.int32 if out_type==10 else np.float32
                        getting=device.read(raw,(1,n),dtype_out)[0,:32] if args.layout[1]=='r' else device.read(raw,(32,m),dtype_out)[:,0]
                        np.testing.assert_array_equal(getting,expected)
                        result.update(median_ms=float(median(timings)),samples_ms=timings)
                    except RuntimeError as error:
                        result['unavailable']=str(error)
                    print(result,flush=True)
                    results.append(result)
    (SOURCE_DIR/'results'/args.output).write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
