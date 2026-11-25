import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from bench import SOURCE_DIR, collect, upload_many
from cuda import Cublas, Cuda, INT, compile_ptx
from end_to_end import ENCODER_KERNELS, gpu_encoder
from research import sample as stress_operands
from rgs import sample_operands
from try_20t import integer_lattice, sample_expected


def verify_lattice():
    d, cap, h, g = integer_lattice(5, 4)
    q = np.arange(-cap, cap+1)
    h, g = h.astype(np.int64), g.astype(np.int64)
    r = -4*h + 5*g
    np.testing.assert_array_equal(5*h + 4*g, q)
    assert d == 41 and cap == 1035 and abs(r).max() == 20
    assert 8192*128**2 < 2**31
    # Verify all 4,289,041 products, including both ends of the range.
    np.testing.assert_array_equal(d*(h[:,None]*h + g[:,None]*g), q[:,None]*q + r[:,None]*r)


def verify_constant_rows(cuda, kernels):
    rows,k,hi=4,64,16
    values=np.zeros((rows,k),np.float16)
    values[1,hi:]=1.5
    values[2,hi:]=-0.5
    values[3,0]=np.inf
    with ExitStack() as stack:
        def mem(size):return stack.enter_context(cuda.buffer(size))
        inp=mem(values); packed=mem(rows*(k+hi)); scale=mem(rows*4); offset=mem(rows*4); sums=mem(rows*4)
        flag=mem(np.zeros(1,np.uint32))
        op=cuda.grid_invocation(kernels['pack_shared_affine'],(rows,1,1),(256,1,1),
                                (inp,packed,scale,offset,sums,flag,INT(k//4),INT(hi//4)))
        op()
        assert cuda.read(flag,(1,),np.uint32)[0]==1
        np.testing.assert_array_equal(cuda.read(offset,(rows,),np.float32),[0,1.5,-0.5,0])
        np.testing.assert_array_equal(cuda.read(packed,(rows,k+hi),np.int8),np.zeros((rows,k+hi),np.int8))


def full_error(cuda, output, m, n, ref, denom):
    actual = cuda.read(output, (m, n), np.float32)
    squared = 0.0
    max_abs = 0.0
    for start in range(0, m, 256):
        delta = actual[start:start+256].astype(np.float64) - ref[start:start+256]
        squared += np.sum(delta*delta)
        max_abs = max(max_abs, float(np.max(np.abs(delta))))
    if not np.all(np.isfinite(actual)):
        raise AssertionError('Non-finite output')
    return actual, dict(full_relative_frobenius=float(np.sqrt(squared)/denom),
                                 max_abs_error=max_abs,rmse=float(np.sqrt(squared/(m*n))))


def run(cuda, blas, kernels, resident, ref, denom, shape, split, method, algo, samples, seed, warm, dataset):
    m,n,k=shape
    hi=split
    lo=k-hi
    if k%4 or n%4 or hi%4 or not 0<hi<k<=4096:
        raise ValueError('K and split must be multiples of four, K≤4096, N must be aligned')
    nominal=shape
    with ExitStack() as stack:
        def mem(argument):
            return stack.enter_context(cuda.buffer(argument))

        raw_a,raw_b=resident
        result=mem(4*m*n)
        stages=[]
        if method=='split-ar2':
            group=(hi+31)//32*32
            high_enc,fields,flag,_=gpu_encoder(cuda,kernels,stack,(m,n,hi),resident,group,input_stride=k)
            ah,ag,bh,bg,sa,sb,p,q=fields
            al,bl,sla,slb=mem(m*lo),mem(n*lo),mem(4*m),mem(4*n)
            stages.append(high_enc)
            for raw,packed,scale,rows in ((raw_a,al,sla,m),(raw_b,bl,slb,n)):
                stages.append(cuda.grid_invocation(kernels['pack_one_round'],(rows,1,1),(256,1,1),
                              (raw,packed,scale,flag,INT(k//4),INT(hi//4),INT(lo//4))))
            dot=mem(4*m*n)
            compute=[blas.invocation('int8',(m,n,hi),ah,bh,p,algorithm=algo),
                     blas.invocation('int8',(m,n,hi),ag,bg,q,algorithm=algo),
                     blas.invocation('int8',(m,n,lo),al,bl,dot,algorithm=algo)]
            finish=cuda.invocation(kernels['finish_20t'],nominal,(8,32),
                (p,q,dot,sa,sb,sla,slb,result,INT(m),INT(n),INT(2)),block=(32,8,1))

            def expected_sample():
                head=[cuda.read(h,(32,hi),np.int8) for h in (ah,ag,bh,bg)]
                tail=[cuda.read(h,(32,lo),np.int8) for h in (al,bl)]
                head_scales=(16*cuda.read(sa,(32,),np.float32),cuda.read(sb,(32,),np.float32))
                tail_scales=tuple(cuda.read(h,(32,),np.float32) for h in (sla,slb))
                return sample_expected(head,head_scales,tail,tail_scales,32)
        else:
            affine=method=='shared-affine'
            name='pack_shared_affine' if affine else 'pack_shared_symmetric'
            a,sa,za,suma=[mem(x) for x in (m*(k+hi),4*m,4*m,4*m)]
            b,sb,zb,sumb=[mem(x) for x in (n*(k+hi),4*n,4*n,4*n)]
            flag=mem(np.zeros(1,np.uint32));dot=mem(4*m*n)
            for raw,packed,scales,offsets,sums,rows in ((raw_a,a,sa,za,suma,m),(raw_b,b,sb,zb,sumb,n)):
                stages.append(cuda.grid_invocation(kernels[name],(rows,1,1),(256,1,1),
                    (raw,packed,scales,offsets,sums,flag,INT(k//4),INT(hi//4))))
            compute=[blas.invocation('int8',(m,n,k+hi),a,b,dot,algorithm=algo)]
            finish=cuda.invocation(kernels['unpack_shared'],nominal,(8,32),
                (dot,sa,sb,za,zb,suma,sumb,result,INT(m),INT(n),INT(lo),INT(affine)),block=(32,8,1))

            def expected_sample():
                qa,qb=[cuda.read(h,(32,k+hi),np.int8).astype(np.int64) for h in (a,b)]
                z=qa@qb.T
                np.testing.assert_array_equal(cuda.read(dot,(32,n),np.int32)[:,:32],z)
                left,right=[cuda.read(h,(32,),np.float32) for h in (sa,sb)]
                correction=0.0
                if affine:
                    da,db=[cuda.read(h,(32,),np.float32) for h in (za,zb)]
                    qa_sum,qb_sum=[cuda.read(h,(32,),np.float32) for h in (suma,sumb)]
                    np.testing.assert_array_equal(qa_sum,qa[:,hi:k].sum(1).astype(np.float32)*left)
                    np.testing.assert_array_equal(qb_sum,qb[:,hi:k].sum(1).astype(np.float32)*right)
                    bias_b=(lo*db.astype(np.float64)+qb_sum).astype(np.float32)
                    correction=(da[:,None].astype(np.float64)*bias_b+(db*qa_sum[:,None])).astype(np.float32)
                return (z.astype(np.float32).astype(np.float64)*(left[:,None]*right)+correction).astype(np.float32)

        def gemm():
            for op in compute:op()
            finish()

        def complete():
            for op in stages:op()
            gemm()

        complete()
        if cuda.read(flag,(1,),np.uint32)[0]:
            raise ValueError('Non-finite source input')
        actual,quality=full_error(cuda,result,m,n,ref,denom)
        np.testing.assert_array_max_ulp(actual[:32,:32],expected_sample(),maxulp=1)
        for _ in range(warm):
            complete()
        cuda.sync()
        row=collect(cuda,nominal,'fp16_end_to_end',complete,samples,algo=algo,method_variant=method,
                    high_k=hi,seed=seed,dataset=dataset,**quality)
        np.testing.assert_array_equal(cuda.read(result,(m,n),np.float32),actual)
        print(f"    whole-matrix relative error {100*quality['full_relative_frobenius']:.6f}%",flush=True)
        row['prepacked_main_and_epilogue_ms']=cuda.measure(gemm,3,5)
        return row


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument('--gpu',required=True)
    cli.add_argument('--methods',nargs='+',choices=('split-ar2','shared-symmetric','shared-affine'),
                     default=('split-ar2','shared-symmetric','shared-affine'))
    cli.add_argument('--high-k',nargs='+',type=int,default=(768,1024,1280,1536,1792,2048,2560))
    cli.add_argument('--algos',nargs='+',type=int,default=(-1,))
    cli.add_argument('--seeds',nargs='+',type=int,default=(2109,))
    cli.add_argument('--samples',type=int,default=7)
    cli.add_argument('--warm',type=int,default=12)
    cli.add_argument('--dataset',choices=('normal','uniform','outliers','span','cancellation'),default='normal')
    cli.add_argument('--output',type=Path,default=SOURCE_DIR/'results'/'frontier.json')
    args=cli.parse_args()
    verify_lattice()
    shape=(4096,)*3
    source=b'\n'.join((SOURCE_DIR/name).read_bytes() for name in ('encode_sm61.cu','try_20t.cu','shared_scale.cu'))
    ptx=compile_ptx(source)
    names=(*ENCODER_KERNELS,'pack_one_round','finish_20t','pack_shared_affine','pack_shared_symmetric','unpack_shared')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    results=[]
    with Cuda(args.gpu) as cuda,cuda.load(ptx) as module,Cublas() as blas:
        print(f'{cuda.name} / cuBLAS {blas.version}',flush=True)
        kernels={name:cuda.kernel(module,name)[0] for name in names}
        verify_constant_rows(cuda,kernels)
        for seed in args.seeds:
            a,b=sample_operands(shape,seed=seed) if args.dataset=='normal' else stress_operands(shape,args.dataset,seed)
            ref=a.astype(np.float64)@b.astype(np.float64).T
            denom=np.linalg.norm(ref)
            with ExitStack() as parent:
                data=upload_many(cuda,parent,(a,b))
                for method in args.methods:
                    for split in args.high_k:
                        for algo in args.algos:
                            results.append(run(cuda,blas,kernels,data,ref,denom,shape,split,method,algo,args.samples,
                                               seed,args.warm,args.dataset))
                            args.output.write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
