import argparse
import ctypes as ct
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from bench import SOURCE_DIR, upload_many
from cuda import Cublas,Cuda,DEV_PTR,FLOAT,INT,PTR,api,compile_ptx,library_path
from research import capacity,measure,metrics,sample


def strided(blas,size,a,b,c):
    dll=ct.WinDLL(str(library_path()/'cublas64_12.dll'))
    L=ct.c_longlong
    call=api(dll,'cublasGemmStridedBatchedEx',PTR,INT,INT,INT,INT,INT,PTR,DEV_PTR,INT,INT,L,
             DEV_PTR,INT,INT,L,PTR,DEV_PTR,INT,INT,L,INT,INT,INT)
    one,zero=INT(1),INT(0); stride=size*size
    def run():
        call(blas.handle,1,0,size,size,size,ct.byref(one),b,3,size,stride,a,3,size,stride,
             ct.byref(zero),c,10,size,stride,7,72,-1)
    return run


def ordered(steps):
    def go():
        for step in steps:step()
    return go


def run(cuda,blas,f,size,base,exact,tries):
    shape=(size,)*3; half=size//2; side=(half,)*3; slots=half*half
    a,b=sample(shape,'normal'); reference=a.astype(np.float64)@b.astype(np.float64).T
    with ExitStack() as stack:
        def mem(x):return stack.enter_context(cuda.buffer(x))
        da,db=upload_many(cuda,stack,(a,b))
        flag=mem(np.zeros(1,np.int32));null=DEV_PTR();out=mem(4*size*size)
        cap=capacity(base,63); count=3 if exact else 2
        def encode(raw):
            planes=[mem(size*size) for _ in range(count)]; scales=mem(4*size);cut=mem(4*size)
            pn=(*planes,) if exact else (*planes,null)
            job=cuda.grid_invocation(f['encode_rows'],(size,1,1),(256,1,1),
                (raw,*pn,scales,cut,null,null,null,flag,INT(size),INT(base),INT(cap),INT(0 if exact else -1),FLOAT(0)))
            return planes,scales,job
        planes_a,sa,pack_a=encode(da);planes_b,sb,pack_b=encode(db)
        packers=[pack_a,pack_b]
        direct_temp=[mem(4*size*size) for _ in range(count)]
        direct_per=[blas.invocation('int8',shape,x,y,z,algorithm=1) for x,y,z in zip(planes_a,planes_b,direct_temp)]
        direct_tmp=direct_temp if exact else direct_temp+[null]
        direct_finish=cuda.invocation(f['reconstruct'],shape,(8,32),(*direct_tmp,sa,sb,out,INT(size),INT(size),
            INT(base),INT(0 if exact else -1),INT(0 if exact else -1)),block=(32,8,1))
        ordinary=ordered((*packers,*direct_per,direct_finish));ordinary()
        expected=cuda.read(out,(size,size),np.float32)
        quality=metrics(expected,reference)
        baseline=measure(cuda,ordinary,shape,tries)
        transform=[]; subproducts=[]; batched=[]; products=[]
        for aa,bb in zip(planes_a,planes_b):
            ax,bx,products_out=mem(7*slots),mem(7*slots),mem(7*4*slots)
            products.append(products_out)
            for x,y,right in ((aa,ax,0),(bb,bx,1)):
                transform.append(cuda.grid_invocation(f['strassen_form'],((half+31)//32,(half+7)//8,7),(32,8,1),
                                                      (x,y,INT(size),INT(right),flag)))
            for i in range(7):
                subproducts.append(blas.invocation('int8',side,ax,bx,DEV_PTR(products_out.value+i*4*slots),offset=i*slots,algorithm=1))
            batched.append(strided(blas,half,ax,bx,products_out))
        raw_outputs=products if exact else products+[null]
        final=cuda.invocation(f['strassen_finish'],shape,(8,32),(*raw_outputs,sa,sb,out,
            INT(size),INT(base),INT(exact)),block=(32,8,1))
        out_records={}
        for method,main in (('7_separate',subproducts),('strided_batch',batched)):
            complete=ordered((*packers,*transform,*main,final))
            try:
                complete()
                actual=cuda.read(out,(size,size),np.float32)
            except RuntimeError as problem:
                print(method,problem,flush=True)
                continue
            assert cuda.read(flag,(1,),np.int32)[0]==0
            np.testing.assert_array_equal(actual,expected)
            stages=dict(transform=transform,main=main,reconstruct=[final],
                        prepacked=(*transform,*main,final),fp16=(*packers,*transform,*main,final))
            out_records[method]={label:measure(cuda,ordered(funcs),shape,tries) for label,funcs in stages.items()}
            print(size,base,method,'direct',baseline['median_ms'],'Strassen',out_records[method]['fp16']['median_ms'],
                  'main',out_records[method]['main']['median_ms'],'error',quality['relative_frobenius'],flush=True)
        return dict(size=size,base=base,exact=exact,cap=cap,quality=quality,baseline=baseline,strassen=out_records)


def main():
    cli=argparse.ArgumentParser();cli.add_argument('--gpu',required=True)
    cli.add_argument('--sizes',type=int,nargs='+',default=(2048,4096));cli.add_argument('--samples',type=int,default=7)
    args=cli.parse_args()
    ptx=compile_ptx(b'\n'.join((SOURCE_DIR/name).read_bytes() for name in ('stage2.cu','strassen.cu')))
    rows=[]
    with Cuda(args.gpu) as cuda,cuda.load(ptx) as module,Cublas() as blas:
        print('GPU',cuda.name,'cuBLAS',blas.version,flush=True)
        f={n:cuda.kernel(module,n)[0] for n in ('encode_rows','reconstruct','strassen_form','strassen_finish')}
        for size in args.sizes:
            if size>4096 or size%8:raise ValueError('Expected square multiple of eight up to 4096')
            for base,exact in ((16,False),(64,True)):
                rows.append(run(cuda,blas,f,size,base,exact,args.samples))
                (SOURCE_DIR/'results'/'strassen.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')


if __name__=='__main__':main()
