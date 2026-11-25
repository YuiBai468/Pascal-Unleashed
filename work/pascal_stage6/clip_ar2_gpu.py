"""Benchmark the fused two-pass integer and sparse-tail pipeline on SM 6.1."""

import argparse
from contextlib import ExitStack
import json
from statistics import median

import numpy as np

from bench import SOURCE_DIR, collect, upload_many
from clip_ar2_cpu import capacity
from cuda import Cublas, Cuda, DEV_PTR, FLOAT, INT, compile_ptx
from layout_scan import int8_gemm
from rgs import sample_operands
from research import sample as stress_operands
from slide_gpu import TRANSPOSES


def verify(cuda, out, a, b, probes):
    m,n=a.shape[0],b.shape[0]
    if probes and probes<m:
        left_rows=np.linspace(0,m-1,probes,dtype=int)
        actual=np.stack([cuda.read(DEV_PTR(out.value+4*n*int(i)),(n,),np.float32) for i in left_rows])
        estimated=a[left_rows].astype(np.float64)@b.astype(np.float64).T
        return dict(relative_frobenius=float(np.linalg.norm(actual-estimated)/np.linalg.norm(estimated)),
                    certified_rows=probes,verified_elements=actual.size)
    error, gold, peak=0.0,0.0,0.0
    for i in range(0,m,512):
        rows=min(512,m-i)
        actual=cuda.read(DEV_PTR(out.value+4*n*i),(rows,n),np.float32)
        if not np.all(np.isfinite(actual)):
            raise AssertionError('Non-finite result')
        left=a[i:i+rows].astype(np.float64)
        for j in range(0,n,1536):
            right=b[j:j+1536].astype(np.float64)
            expected=left@right.T
            d=actual[:,j:j+len(right)].astype(np.float64)-expected
            error+=float(np.vdot(d,d))
            gold+=float(np.vdot(expected,expected))
            peak=max(peak,float(np.max(abs(d))))
    return dict(full_relative_frobenius=float(np.sqrt(error/max(gold,1e-200))),max_abs_error=peak,
                rmse=float(np.sqrt(error/(m*n))),certified_rows=m,verified_elements=m*n)


def execute(cuda,blas,module,args,a,b,seed,variant):
    m,k=a.shape
    n=b.shape[0]
    shape=(m,n,k)
    cap=capacity(args.base)-args.guard
    with ExitStack() as stack:
        def mem(length):
            return stack.enter_context(cuda.buffer(length))

        def launch(kernel,grid,block,parameters):
            function,info=cuda.kernel(module,kernel)
            return cuda.grid_invocation(function,grid,block,parameters)

        da,db=upload_many(cuda,stack,(a,b))
        ah,ag,at=(mem(m*k) for _ in range(3))
        bh,bg,bf,bht,bgt,bft=(mem(n*k) for _ in range(6))
        sa,sb=mem(8*m),mem(8*n)
        ca,cb=mem(2*m),mem(2*n)
        ra,rb=mem(8*m*args.capacity),mem(8*n*args.capacity)
        raw_h,raw_g,out=(mem(4*m*n) for _ in range(3))
        status=mem(np.zeros(1,np.uint32))
        extra=(status,INT(args.base),INT(cap),FLOAT(args.clip),INT(args.capacity))
        prep=[launch(f'clip_a_{k}',(m,1,1),(512,1,1),(da,ah,ag,ah,sa,ca,ra,*extra)),
              launch(f'clip_b_{k}',(n,1,1),(512,1,1),(db,bh,bg,bf,sb,cb,rb,*extra))]
        transpose,tr,tc=TRANSPOSES[args.transpose]
        for src,dest,rows in ((ah,at,m),(bh,bht,n),(bg,bgt,n),(bf,bft,n)):
            prep.append(launch(transpose,(k//tc,rows//tr,1),(tr*tc//16,1,1),(src,dest,INT(rows),INT(k))))
        gh=int8_gemm(blas,m,n,k,ah,bht,raw_h,ld_b=n,transpose_b=False,algorithm=args.algo)
        gg=int8_gemm(blas,m,n,k,ag,bgt,raw_g,ld_b=n,transpose_b=False,algorithm=args.algo)
        packet=variant.startswith('packet_')
        cached=variant.startswith('cache')
        tile=64 if packet else 32
        if variant.endswith(('cols','rows')) and (n//tile if variant.endswith('cols') else m//tile)%args.panel:
            raise ValueError('Expected complete cache panels')
        params=(raw_h,raw_g,sa,sb,ca,cb,ra,rb,bft,at,out,INT(m),INT(n),INT(args.base),
                INT(args.capacity),INT(args.panel))
        packet_jobs=[]
        if packet:
            groups=(args.capacity+3)//4
            pa,pb=mem(20*m*groups),mem(20*n*groups)
            qa,qb=mem(4*m),mem(4*n)
            for rows,src,count,dest,scale in ((m,ra,ca,pa,qa),(n,rb,cb,pb,qb)):
                packet_jobs.append(launch('packetize_tail',(rows,1,1),(128,1,1),
                                          (src,count,dest,scale,INT(args.capacity),INT(groups))))
            params=(raw_h,raw_g,sa,sb,ca,cb,pa,pb,qa,qb,bft,at,out,INT(m),INT(n),INT(args.base),
                    INT(groups),INT(args.panel))
        if cached:
            chunk=int(variant[5:])
            sa_index,sb_index=mem(2*m*(k//chunk+1)),mem(2*n*(k//chunk+1))
            order_a,order_b=mem(8*m*args.capacity),mem(8*n*args.capacity)
            for rows,src,cnt,dest,offs in ((m,ra,ca,order_a,sa_index),(n,rb,cb,order_b,sb_index)):
                packet_jobs.append(launch('group_tail',(rows,1,1),(128,1,1),
                                    (src,cnt,dest,offs,INT(k),INT(chunk),INT(args.capacity))))
            suffix=(out,INT(m),INT(n),INT(k),INT(args.base),INT(args.capacity))
            left=launch(f'cache_left_{chunk}',(m//64,n//128,1),(256,1,1),
                        (raw_h,raw_g,sa,sb,order_a,sa_index,bft,*suffix))
            right=launch(f'cache_right_{chunk}',(n//64,m//128,1),(256,1,1),
                         (raw_h,raw_g,sa,sb,order_b,sb_index,at,*suffix))

            def finish():
                left()
                right()
        else:
            finish=launch(variant if packet else f'clip_finish_{variant}',(n//tile,m//tile,1),
                          (512 if variant=='512' else 256,1,1),params)

        def gemm():
            gh()
            gg()

        def complete():
            for kernel in prep:
                kernel()
            for kernel in packet_jobs:
                kernel()
            gemm()
            finish()

        complete()
        got=int(cuda.read(status,(1,),np.uint32)[0])
        if got:
            raise RuntimeError(f'Input, digits, or residual capacity invalid: {got}')
        for x,y,raw in ((ah,bh,raw_h),(ag,bg,raw_g)):
            lhs=cuda.read(x,(12,k),np.int8).astype(np.int64)
            rhs=cuda.read(y,(12,k),np.int8).astype(np.int64)
            np.testing.assert_array_equal(lhs@rhs.T,cuda.read(raw,(12,n),np.int32)[:,:12])
        quality=verify(cuda,out,a,b,args.probes)
        counts=[cuda.read(buf,(rows,),np.uint16) for buf,rows in ((ca,m),(cb,n))]
        for _ in range(args.warmup):
            complete()
        record=collect(cuda,shape,'clipped_ar2_e2e',complete,args.samples,seed=seed,dataset=args.dataset,
                       base=args.base,clip=args.clip,cap=cap,variant=variant,transpose=args.transpose,
                       panel=args.panel,capacity=args.capacity,sparse_average=[float(np.mean(x)) for x in counts],
                       sparse_max=[int(np.max(x)) for x in counts],**quality)
        record['stage_ms']={name:float(median(cuda.measure(fn,2,5))) for name,fn in
                            [('pack_a',prep[0]),('pack_b',prep[1]),('transpose_a',prep[2]),
                             ('transpose_bh',prep[3]),('transpose_bg',prep[4]),('transpose_bf',prep[5]),
                             ('gemm',gemm),('finish',finish)]+[(f'packet_{i}',fn) for i,fn in enumerate(packet_jobs)]}
        return record


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gpu',required=True)
    p.add_argument('--shape',type=int,nargs=3,default=(4096,4096,8192))
    p.add_argument('--seeds',type=int,nargs='+',default=(2109,))
    p.add_argument('--dataset',choices=('normal','outliers','co_spike','uniform'),default='normal')
    p.add_argument('--base',type=int,default=14)
    p.add_argument('--clip',type=float,default=2.98)
    p.add_argument('--guard',type=int,default=1)
    p.add_argument('--capacity',type=int,default=512)
    p.add_argument('--variant',nargs='+',choices=('256','512','cols','rows','packet_64','packet_cols','packet_rows',
                                            'cache128','cache256'),default=('256',))
    p.add_argument('--transpose',choices=TRANSPOSES,default='128-stream')
    p.add_argument('--panel',type=int,default=2)
    p.add_argument('--algo',type=int,default=1)
    p.add_argument('--probes',type=int,default=0)
    p.add_argument('--samples',type=int,default=9)
    p.add_argument('--warmup',type=int,default=12)
    p.add_argument('--output',default='clipped-ar2-gpu.json')
    args=p.parse_args()
    m,n,k=args.shape
    if min(m,n)<128 or m%128 or n%128 or k not in (8192,16384,32768,65536):
        p.error('M and N must be 128 aligned and K must be one of 8192, 16384, 32768 or 65536')
    source=b'\n'.join((SOURCE_DIR/f).read_bytes() for f in ('plane_sm61.cu','packed_layout.cu','clip_ar2.cu',
                                                          'clip_packets.cu','clip_cached.cu'))
    ptx=compile_ptx(source)
    results=[]
    with Cuda(args.gpu) as cuda,cuda.load(ptx) as module,Cublas() as blas:
        for seed in args.seeds:
            if args.dataset=='normal':
                a,b=sample_operands(tuple(args.shape),seed=seed)
            else:
                a,b=stress_operands(tuple(args.shape),args.dataset,seed)
            for variant in args.variant:
                results.append(execute(cuda,blas,module,args,a,b,seed,variant))
                (SOURCE_DIR/'results'/args.output).write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
