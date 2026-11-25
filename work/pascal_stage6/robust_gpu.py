"""Benchmark dynamic INT8 GEMM with exact FP32 corrections for exceptional K pairs."""

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path

import numpy as np

from bench import SOURCE_DIR, collect, upload_many
from cuda import Cublas, Cuda, FLOAT, INT, compile_ptx
from layout_scan import int8_gemm
from plane_gpu import packed_table
from plane_lattice import build, line
from research import sample as stress_operands
from rgs import sample_operands


def full_reference_error(actual, a, b):
    delta_sq = gold_sq = largest = 0.0
    right = b.astype(np.float64).T
    for start in range(0, len(a), 256):
        gold = a[start:start+256].astype(np.float64) @ right
        delta = actual[start:start+256] - gold
        delta_sq += float(np.sum(delta * delta))
        gold_sq += float(np.sum(gold * gold))
        largest = max(largest, float(np.max(np.abs(delta))))
    return dict(full_relative_frobenius=float(np.sqrt(delta_sq / max(gold_sq, 1e-200))),
                max_abs_error=largest, rmse=float(np.sqrt(delta_sq / actual.size)))


def get_inputs(shape, kind, seed):
    if kind == 'isolated':
        a, b = sample_operands(shape, seed=seed)
        m,n,k=shape
        rng=np.random.default_rng(seed+17)
        spots=rng.choice(k//2,8,replace=False)
        a[rng.integers(0,m,4),2*spots[:4]] = np.float16(256)
        b[rng.integers(0,n,4),2*spots[4:]+1] = np.float16(-256)
        return a,b
    if kind == 'normal':
        return sample_operands(shape,seed=seed)
    return stress_operands(shape,kind,seed)


def cpu_detector(a,b,extra_exponent):
    peaks=[]
    for matrix in (a,b):
        raw = matrix.view(np.uint16).reshape(len(matrix),-1,2)&0x7fff
        peaks.append(np.max(raw,axis=(0,2)).astype(np.int32)>>10)
    cuts = [int(np.partition(p,(len(p)-1)//2)[(len(p)-1)//2])+extra_exponent for p in peaks]
    groups = np.flatnonzero((peaks[0]>=cuts[0])|(peaks[1]>=cuts[1]))
    return groups,cuts


def run(cpu_a,cpu_b,shape,cuda,blas,module,args,seed,kind):
    m,n,k=shape
    pairs,physical=k//2,3*k//2+args.dual//2
    p,lut=build((3,4,5),.002,16)
    margin=127-int(np.max(np.abs(lut)))
    with ExitStack() as stack:
        def mem(length):
            return stack.enter_context(cuda.buffer(length))

        da,db,coeff,lookup=upload_many(cuda,stack,(cpu_a,cpu_b,p,packed_table(lut)))
        aa,bb,pbt=mem(m*physical),mem(n*physical),mem(n*physical)
        sa,sb,check=mem(4*m),mem(4*n),mem(np.zeros(1,np.uint32))
        peaks_a,peaks_b,mask=mem(4*pairs),mem(4*pairs),mem(pairs)
        groups,info=mem(4*args.capacity),mem(12)
        extra_a,extra_b=mem(8*m*args.capacity),mem(8*n*args.capacity)
        raw,out=mem(4*m*n),mem(4*m*n)

        def launch(name,grid,block,params):
            return cuda.grid_invocation(cuda.kernel(module,name)[0],grid,block,params)

        mark=[launch('clear_pair_peaks',((pairs+255)//256,1,1),(256,1,1),(peaks_a,peaks_b,INT(pairs)))]
        for x,y,rows in ((da,peaks_a,m),(db,peaks_b,n)):
            mark.append(launch('reduce_pair_peaks',(pairs//64,(rows+255)//256,1),(256,1,1),
                               (x,y,INT(rows),INT(pairs))))
        mark.append(launch('identify_exceptional_pairs',(1,1,1),(256,1,1),
                           (peaks_a,peaks_b,mask,groups,info,INT(pairs),INT(args.capacity),INT(args.extra_exponent))))
        gather=launch('gather_pairs',((max(m,n)+127)//128,args.capacity,2),(128,1,1),
                      (da,db,extra_a,extra_b,groups,info,INT(m),INT(n),INT(pairs),INT(args.capacity)))
        packing_kernel=f'masked_plane_{k}'
        packing_params=(coeff,lookup,check,INT(margin),mask)
        if args.slide:
            _, second_table = line((3,4),.0001,16)
            line_lut = mem(packed_table(second_table))
            limit = 127-max(math.ceil(3+np.abs(lut).max()), math.ceil(2.5+np.abs(second_table).max()))
            packing_kernel=f'masked_line_{args.dual}'
            packing_params=(coeff,lookup,line_lut,check,INT(limit),FLOAT(.81),FLOAT(.89),mask)
        preparation=[launch(packing_kernel,(rows,1,1),(k//16,1,1),(x,y,s,*packing_params))
                     for x,y,s,rows in ((da,aa,sa,m),(db,bb,sb,n))]
        preparation.append(launch('transpose_packed64',(physical//64,n//64,1),(256,1,1),
                                  (bb,pbt,INT(n),INT(physical))))
        gemm=int8_gemm(blas,m,n,physical,aa,pbt,raw,ld_b=n,transpose_b=False,algorithm=1)
        finish=launch('unpack_robust',(n//128,m//16,1),(32,4,1),
                      (raw,sa,sb,extra_a,extra_b,info,out,INT(m),INT(n),INT(args.capacity)))

        def complete():
            for stage in mark:
                stage()
            gather()
            for stage in preparation:
                stage()
            gemm()
            finish()

        complete()
        if cuda.read(check,(1,),np.uint32)[0]:
            raise ValueError('Invalid FP16 value or out-of-range packed code')
        selected, thresholds=cpu_detector(cpu_a,cpu_b,args.extra_exponent)
        np.testing.assert_array_equal(cuda.read(info,(3,),np.int32),[len(selected),*thresholds])
        enabled=len(selected)<=args.capacity
        used=selected if enabled else selected[:0]
        if len(used):
            np.testing.assert_array_equal(cuda.read(groups,(len(used),),np.int32),used)
        al,bl=[cuda.read(x,(32,physical),np.int8).astype(np.int64) for x in (aa,bb)]
        sl,sr=[cuda.read(x,(32,),np.float32) for x in (sa,sb)]
        base=al@bl.T
        np.testing.assert_array_equal(cuda.read(raw,(32,n),np.int32)[:,:32],base)
        actual=cuda.read(out,(m,n),np.float32)
        if not np.all(np.isfinite(actual)):
            raise AssertionError('Nonfinite output')
        expected=base.astype(np.float32)*(sl[:,None]*sr)
        for group in used:
            for col in (2*group,2*group+1):
                product=cpu_a[:32,col,None].astype(np.float64)*cpu_b[None,:32,col].astype(np.float64)
                expected=(expected+product).astype(np.float32)
        np.testing.assert_array_equal(actual[:32,:32],expected)
        quality=full_reference_error(actual,cpu_a,cpu_b)
        for _ in range(args.warmup):
            complete()
        record=collect(cuda,shape,'robust_fp16_e2e',complete,args.samples,seed=seed,dataset=kind,
                       physical_k=physical,capacity=args.capacity,selected=selected.tolist(),fallback=not enabled,
                       slide=args.slide,dual=args.dual,**quality)
        np.testing.assert_array_equal(actual,cuda.read(out,(m,n),np.float32))
        record['detection_ms']=cuda.measure(lambda:[stage() for stage in mark],3,5)
        record['finish_ms']=cuda.measure(finish,3,5)
        return record


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',required=True)
    parser.add_argument('--shape',type=int,nargs=3,default=(4096,4096,4096))
    parser.add_argument('--datasets',nargs='+',choices=('normal','uniform','outliers','co_spike','isolated'),
                        default=('normal','outliers','co_spike','isolated'))
    parser.add_argument('--seeds',type=int,nargs='+',default=(2109,))
    parser.add_argument('--capacity',type=int,default=16)
    parser.add_argument('--slide',action='store_true')
    parser.add_argument('--dual',type=int,choices=(0,512,768,1024),default=0)
    parser.add_argument('--extra-exponent',type=int,default=3)
    parser.add_argument('--warmup',type=int,default=40)
    parser.add_argument('--samples',type=int,default=11)
    parser.add_argument('--output',type=Path,default=SOURCE_DIR/'results'/'robust.json')
    args=parser.parse_args()
    m,n,k=args.shape
    if m<32 or m%16 or n<128 or n%128 or k not in (4096,8192) or args.capacity<1 or args.extra_exponent<1:
        parser.error('Expected 16-aligned M, 128-aligned N, K in {4096,8192} and a positive detection configuration')
    if (args.slide and k!=8192) or (args.dual and not args.slide):
        parser.error('Sliding normal packing requires K=8192, and dual encoding requires --slide')
    files=('plane_sm61.cu','packed_layout.cu','robust_plane.cu')
    if args.slide:
        files+=('normal_slide.cu','slide_line_large.cu')
    ptx=compile_ptx(b'\n'.join((SOURCE_DIR/f).read_bytes() for f in files))
    results=[]
    with Cuda(args.gpu) as cuda,cuda.load(ptx) as module,Cublas() as blas:
        for dataset in args.datasets:
            for seed in args.seeds:
                a,b=get_inputs(tuple(args.shape),dataset,seed)
                results.append(run(a,b,tuple(args.shape),cuda,blas,module,args,seed,dataset))
                args.output.write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
