"""Time a burst of complete GEMMs with distinct resident FP16 inputs."""

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path

import numpy as np

from bench import SOURCE_DIR, collect, upload_many
from cuda import Cublas, Cuda, FLOAT, INT, PTR, UINT, api, compile_ptx
from frontier import full_error
from layout_scan import int8_gemm
from plane_gpu import packed_table
from plane_lattice import build
from rgs import sample_operands
from slide_gpu import lookup_tables


def execute(cuda, blas, module, args):
    m = n = k = 4096
    shape = (m,n,k)
    physical = 3*k//2 - args.triples//2
    if args.triples:
        tables, margin = lookup_tables(16,.002,.004)
        encoder_name = f'vec_{args.triples}_{args.vector}' if args.vector else f'slide_{args.triples}_4'
    else:
        p, q = build((3,4,5),.002,16)
        tables = (p,packed_table(q))
        margin = 127 - int(np.max(np.abs(q)))
        encoder_name = 'plane_bytes_registers'
    encode = cuda.kernel(module,encoder_name)[0]
    transpose = cuda.kernel(module,'transpose_packed64')[0]
    unpack = cuda.kernel(module,'unpack_plane')[0]
    with ExitStack() as stack:
        def mem(x):
            return stack.enter_context(cuda.buffer(x))

        def event():
            return stack.enter_context(cuda.dependency_event())

        coeff = upload_many(cuda,stack,tables)
        stat = mem(np.zeros(1,np.uint32))
        side = (stat,INT(margin),FLOAT(.81),FLOAT(.87)) if args.triples else (stat,INT(k//2),INT(16),INT(margin))
        precedence = (None,)*3
        if args.priority != 'default':
            low,high = cuda.priority_range()
            precedence = {'auxiliary':(high,low,high), 'compute':(low,high,low),
                          'output':(low,low,high), 'input':(high,low,low)}[args.priority]
        prep, core, epilogue = (stack.enter_context(cuda.stream(p)) for p in precedence) if not args.serial else (None,)*3
        blas.set_stream(blas.handle,core)
        stack.callback(blas.set_stream,blas.handle,None)
        wait = api(cuda.driver,'cuStreamWaitEvent', PTR,PTR,UINT)
        begin, finish = event(), event()

        jobs=[]
        for seed in args.seeds:
            a,b = sample_operands(shape,seed=seed)
            da,db = upload_many(cuda,stack,(a,b))
            pa,pb,pbt = (mem(m*physical) for _ in range(3))
            sa,sb = mem(4*m),mem(4*n)
            int_out,out = mem(4*m*n),mem(4*m*n)
            packing = [cuda.grid_invocation(encode,(rows,1,1),(args.vector or 256,1,1),(x,y,s,*coeff,*side),prep)
                       for x,y,s,rows in ((da,pa,sa,m),(db,pb,sb,n))]
            packing.append(cuda.grid_invocation(transpose,(physical//64,n//64,1),(256,1,1),
                                               (pb,pbt,INT(n),INT(physical)),prep))
            gemm=int8_gemm(blas,m,n,physical,pa,pbt,int_out,ld_b=n,transpose_b=False,algorithm=1)
            write=cuda.grid_invocation(unpack,(n//128,m//16,1),(32,4,1),(int_out,sa,sb,out,INT(m),INT(n)),epilogue)
            jobs.append((packing,gemm,write,event(),event(),a,b,out))

        def complete():
            if not args.serial:
                cuda.record_event(begin,None)
                wait(prep,begin,0)
            for packing,gemm,write,packed,ready,*_ in jobs:
                for pack in packing:
                    pack()
                if not args.serial:
                    cuda.record_event(packed,prep)
                    wait(core,packed,0)
                gemm()
                if not args.serial:
                    cuda.record_event(ready,core)
                    wait(epilogue,ready,0)
                write()
            if not args.serial:
                cuda.record_event(finish,epilogue)
                wait(None,finish,0)

        complete()
        if cuda.read(stat,(1,),np.uint32)[0]:
            raise ValueError('Invalid input or out-of-range packed value')
        checks=[]
        for index,(*operations,a,b,out) in enumerate(jobs):
            gold = a.astype(np.float64)@b.astype(np.float64).T
            result,quality = full_error(cuda,out,m,n,gold,np.linalg.norm(gold))
            checks.append((hashlib.sha256(result).digest(),quality))
            jobs[index] = (*operations,None,None,out)
        for _ in range(args.warmup):
            complete()
        quality = [value for _,value in checks]
        batch_shape = (len(jobs)*m,n,k)
        record=collect(cuda,batch_shape,'streaming_fp16_e2e',complete,args.samples,serial=args.serial,
                       problem_shape=shape,physical_k=physical,seeds=args.seeds,triples=args.triples,vector=args.vector,
                       priorities=precedence,
                       errors=[check['full_relative_frobenius'] for check in quality])
        for job,(expected,_) in zip(jobs,checks):
            if hashlib.sha256(cuda.read(job[-1],(m,n),np.float32)).digest() != expected:
                raise AssertionError('Repeated operation changed a full output')
        return record


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--gpu', required=True)
    cli.add_argument('--seeds',nargs='+',type=int,default=[2109,43,101,20261125,17,4129])
    cli.add_argument('--triples',type=int,choices=(0,768,896,1024),default=0)
    cli.add_argument('--vector',type=int,choices=(0,256,512),default=0)
    cli.add_argument('--samples',type=int,default=9)
    cli.add_argument('--warmup',type=int,default=15)
    cli.add_argument('--serial',action='store_true')
    cli.add_argument('--priority', choices=('default','auxiliary','compute','output','input'), default='default')
    cli.add_argument('--output',type=Path,default=SOURCE_DIR/'results'/'streaming.json')
    args=cli.parse_args()
    if args.vector and not args.triples:
        cli.error('Vector packing is available for the hybrid encoder')
    source = b'\n'.join((SOURCE_DIR / f).read_bytes() for f in
                       ('plane_sm61.cu','packed_layout.cu','normal_slide.cu'))
    ptx=compile_ptx(source)
    with Cuda(args.gpu) as cuda, cuda.load(ptx) as module, Cublas() as blas:
        result=execute(cuda,blas,module,args)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
