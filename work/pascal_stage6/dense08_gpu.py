"""Test uncapped end-to-end P40 throughput with peak-scaled dense two-byte codes."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from statistics import median

import numpy as np

from bench import SOURCE_DIR,collect,upload_many
from clip_ar2_gpu import verify
from cuda import Cublas,Cuda,DEV_PTR,FLOAT,INT,compile_ptx
from fp64_reference import verify_on_gpu
from layout_scan import int8_gemm
from rgs import sample_operands


def trial(cuda,blas,module,args,a,b,seed,lattice):
    m,k=a.shape
    n=b.shape[0]
    mode='one' if lattice else 'ar2'
    with ExitStack() as stack:
        def mem(data):
            return stack.enter_context(cuda.buffer(data))

        def launch(name,grid,block,params):
            fn,_=cuda.kernel(module,name)
            return cuda.grid_invocation(fn,grid,block,params)

        da,db=upload_many(cuda,stack,(a,b))
        status=mem(np.zeros(1,np.uint32))
        sa,sb=mem(8*m),mem(8*n)
        raw=mem(4*m*n)
        if lattice:
            table,settings=lattice
            lut=mem(np.ascontiguousarray(table).view(np.uint16).ravel())
            cap,weight=settings['cap'],settings['u']**2+settings['v']**2
            ah,bh=mem(2*m*k),mem(2*n*k)
            ag,bg,second=(DEV_PTR(0),)*3
            multiply=(int8_gemm(blas,m,n,2*k,ah,bh,raw,ld_b=n,transpose_b=False,algorithm=args.algo),)
        else:
            lut=DEV_PTR(0)
            cap,weight=127*15,15
            ah,bh,ag,bg=mem(m*k),mem(n*k),mem(m*k),mem(n*k)
            second=mem(4*m*n)
            multiply=(int8_gemm(blas,m,n,k,ah,bh,raw,ld_b=n,transpose_b=False,algorithm=args.algo),
                      int8_gemm(blas,m,n,k,ag,bg,second,ld_b=n,transpose_b=False,algorithm=args.algo))
        encode_left=launch(f'dense08_{mode}_{k}',(m,1,1),(512,1,1),(da,ah,ag,sa,status,INT(cap),lut))
        right_scales=launch('dense08_peaks',(n,1,1),(256,1,1),(db,sb,status,INT(k),INT(cap)))
        encode_right=launch(f'dense08_{mode}_right',(k//64,n//64,1),(256,1,1),
                            (db,bh,bg,sb,INT(k),INT(n),INT(cap),lut))
        finish=launch(f'dense08_{mode}_write',(n//128,m//16,1),(32,4,1),(raw,second,sa,sb,raw,INT(n),FLOAT(weight)))

        def encode():
            encode_left()
            right_scales()
            encode_right()

        def products():
            for step in multiply:
                step()

        def complete():
            encode()
            products()
            finish()

        complete()
        if cuda.read(status,(1,),np.uint32)[0]:
            raise ValueError('Non-finite FP16 input')
        quality=verify_on_gpu(cuda,blas,module,raw,da,db,a,b) if args.gpu_reference else verify(cuda,raw,a,b,args.probes)
        for _ in range(args.warmup):
            complete()
        record=collect(cuda,(m,n,k),f'dense08_{mode}',complete,args.samples,seed=seed,dataset='normal',
                       model=args.model.name if args.model else None,cap=cap,algo=args.algo,**quality)
        record['phase_ms']={name:float(median(cuda.measure(op,1,5))) for name,op in
                            (('encode',encode),('products',products),('finish',finish))}
        return record


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gpu',required=True)
    p.add_argument('--shape',nargs=3,type=int,action='append',metavar=('M','N','K'))
    p.add_argument('--seeds',nargs='+',type=int,default=(2109,))
    p.add_argument('--warmup',type=int,default=6)
    p.add_argument('--samples',type=int,default=9)
    p.add_argument('--probes',type=int,default=0)
    p.add_argument('--gpu-reference',action='store_true')
    p.add_argument('--algo',type=int,default=1)
    p.add_argument('--model',type=Path)
    p.add_argument('--output',default='dense08-gpu.json')
    args=p.parse_args()
    shapes=args.shape or [(8192,8192,16384)]
    if any(min(m,n)<128 or m%128 or n%128 or k not in (8192,16384,32768) for m,n,k in shapes):
        p.error('Use M and N divisible by 128 and K equal to 8192, 16384, or 32768')
    if args.gpu_reference and args.probes:
        p.error('The GPU FP64 reference verifies the full output')
    lattice=None
    if args.model:
        with np.load(args.model,allow_pickle=False) as model:
            table=model['table']
            settings=json.loads(str(model['metadata']))
        if table.shape!=(settings['cap']+1,2) or table.dtype!=np.int8 or table[0].any() or np.any(table==-128):
            p.error('Invalid codebook')
        lattice=(table,settings)
    files=('plane_sm61.cu','dense08.cu')+ (('fp64_reference.cu',) if args.gpu_reference else ())
    source=b'\n'.join((SOURCE_DIR/f).read_bytes() for f in files)
    ptx=compile_ptx(source)
    records=[]
    with Cuda(args.gpu) as cuda,cuda.load(ptx) as module,Cublas() as blas:
        for shape in shapes:
            for seed in args.seeds:
                a,b=sample_operands(shape,seed=seed)
                records.append(trial(cuda,blas,module,args,a,b,seed,lattice))
                (SOURCE_DIR/'results'/args.output).write_text(json.dumps(records,indent=2),encoding='utf-8')
                del a,b


if __name__=='__main__':
    main()
