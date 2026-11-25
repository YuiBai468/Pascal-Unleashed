"""Two dense INT8 passes with cuSPARSE FP16 tail correction into FP32 results."""

import argparse
from contextlib import ExitStack
import ctypes as ct
import json
import os
from pathlib import Path
from statistics import median

import numpy as np

from bench import SOURCE_DIR, collect, upload_many
from clip_ar2_cpu import capacity
from clip_ar2_gpu import verify
from cuda import Cublas, Cuda, DEV_PTR, FLOAT, INT, PTR, SIZE, UINT, api, compile_ptx, library_path
from fp64_reference import verify_on_gpu
from layout_scan import int8_gemm
from research import sample as stress_operands
from rgs import sample_operands
from slide_gpu import TRANSPOSES


DIRECTIONAL_TILES={'tail_directional_bytes':64,'tail_directional128_bytes':128}
BYTE_TILES=('tail_tile_bytes','tail_tile_bytes_u4','tail_tile_bytes_16','tail_tile_bytes_u8',*DIRECTIONAL_TILES)


class SparseLibrary:
    def __init__(self,stack):
        i64=ct.c_int64
        path=library_path()
        stack.enter_context(os.add_dll_directory(str(path)))
        dll=ct.WinDLL(str(path/'cusparse64_12.dll'))
        self.handle=PTR()
        api(dll,'cusparseCreate',ct.POINTER(PTR))(ct.byref(self.handle))
        stack.callback(api(dll,'cusparseDestroy',PTR),self.handle)
        self.create_sparse=api(dll,'cusparseCreateCsr',ct.POINTER(PTR),i64,i64,i64,DEV_PTR,DEV_PTR,DEV_PTR,
                               INT,INT,INT,INT)
        self.destroy_sparse=api(dll,'cusparseDestroySpMat',PTR)
        self.create_dense=api(dll,'cusparseCreateDnMat',ct.POINTER(PTR),i64,i64,i64,DEV_PTR,INT,INT)
        self.destroy_dense=api(dll,'cusparseDestroyDnMat',PTR)
        self.size=api(dll,'cusparseSpMM_bufferSize',PTR,INT,INT,PTR,PTR,PTR,PTR,PTR,INT,INT,ct.POINTER(SIZE))
        self.mm=api(dll,'cusparseSpMM',PTR,INT,INT,PTR,PTR,PTR,PTR,PTR,INT,INT,DEV_PTR)
        self.set_stream=api(dll,'cusparseSetStream',PTR,PTR)

    def dense(self,stack,rows,cols,data,typ,layout):
        desc=PTR()
        self.create_dense(ct.byref(desc),rows,cols,cols if layout=='r' else rows,data,typ,2 if layout=='r' else 1)
        stack.callback(self.destroy_dense,desc)
        return desc

    def csr(self,rows,cols,nnz,off,idx,val):
        desc=PTR()
        self.create_sparse(ct.byref(desc),rows,cols,nnz,off,idx,val,2,2,0,2)
        return desc


def trial(cuda,blas,lib,module,args,a,b,seed,model):
    m,k=a.shape
    n=b.shape[0]
    shape=(m,n,k)
    cap=127*args.base if args.fold else capacity(args.base)-1
    with ExitStack() as allocation:
        def mem(length):
            return allocation.enter_context(cuda.buffer(length))

        def launch(name,grid,block,params,stream=None):
            return cuda.grid_invocation(cuda.kernel(module,name)[0],grid,block,params,stream)

        da,db=upload_many(cuda,allocation,(a,b))
        ah,ag=mem(m*k),mem(m*k)
        bht,bgt=mem(n*k),mem(n*k)
        bh,bg=(bht,bgt) if args.fused_right else (mem(n*k),mem(n*k))
        sa,sb=mem(8*m),mem(8*n)
        ca,cb=mem(2*m),mem(2*n)
        ra,rb=mem(8*m*args.capacity),mem(8*n*args.capacity)
        raw_h,raw_g,out=(mem(4*m*n) for _ in range(3))
        status=mem(np.zeros(1,np.uint32))
        total=mem(8)
        extra=(status,INT(args.base),INT(cap),FLOAT(args.clip),INT(args.capacity))
        operands=((m,da,ah,ag,sa,ca,ra),(n,db,bh,bg,sb,cb,rb))
        preparation=[]
        if args.hot_columns:
            energies,hot_count,hot_indices=mem(8*k),mem(4),mem(4*16)
            keep=mem(2*k)
            hot_a,hot_b=mem(4*16*m),mem(4*16*n)
            preparation.append(launch('sample_column_energy',(k//128,1,1),(128,1,1),
                                      (da,db,energies,INT(m),INT(n),INT(k))))
            preparation.append(launch('select_hot_columns',(1,1,1),(256,1,1),
                                      (energies,hot_count,hot_indices,INT(k))))
            preparation.append(launch('hot_column_mask',(k//512,1,1),(256,1,1),
                                      (hot_count,hot_indices,keep,INT(k))))
            for src,dst,rows in ((da,hot_a,m),(db,hot_b,n)):
                preparation.append(launch('gather_hot_columns',((rows+255)//256,16,1),(256,1,1),
                                          (src,hot_count,hot_indices,dst,INT(rows),INT(k))))
        if args.bucket_width:
            groups=k//args.bucket_width
            factors=[mem(rows*2*groups*4) for rows in (m,n)]
            one=FLOAT(1)

            def bucket_correction():
                blas.sgemm(blas.handle,1,0,n,m,2*groups,ct.byref(one),factors[1],2*groups,
                           factors[0],2*groups,ct.byref(one),out,n)

        if model is None:
            weights=(FLOAT(args.base-1),FLOAT(args.base))
            preparation=[launch(f'clip_a_{k}',(rows,1,1),(512,1,1),(src,h,g,h,scale,count,tail,*extra))
                         for rows,src,h,g,scale,count,tail in operands]
        else:
            wa,wg=model[2]
            weights=(FLOAT(wa/wg),FLOAT(wg*(cap/127)**2))
            for side,((rows,src,h,g,scale,count,tail),table) in enumerate(zip(operands,model[:2])):
                lookup=mem(np.ascontiguousarray(table).view(np.uint16).ravel())
                if args.hot_columns:
                    name='clip_hot_right' if side else 'clip_hot_left'
                    preparation.append(launch(f'{name}_{k}',(rows,1,1),(512,1,1),
                        (src,h,g,scale,count,tail,status,INT(cap),FLOAT(args.clip),INT(args.capacity),
                         lookup,INT(len(table)-1),keep)))
                    if side:
                        right_lookup,right_digits=lookup,len(table)-1
                elif args.bucket_width:
                    preparation.append(launch('bucket_rows',(rows,1,1),(256,1,1),
                        (src,factors[side],scale,status,INT(k),INT(args.bucket_width),INT(cap),FLOAT(args.clip),INT(side))))
                    name='bucket_right' if side else 'bucket_left'
                    preparation.append(launch(f'{name}_{k}',(rows,1,1),(512,1,1),
                        (src,h,g,scale,count,tail,status,INT(cap),INT(args.capacity),lookup,INT(len(table)-1),
                         factors[side],INT(args.bucket_width),INT(side))))
                    if side:
                        right_lookup,right_digits=lookup,len(table)-1
                elif args.left_only and side==1:
                    right_lookup,right_digits=lookup,len(table)-1
                    preparation.append(launch('dense08_peaks',(n,1,1),(256,1,1),
                                              (db,sb,status,INT(k),INT(cap))))
                elif args.fused_right and side==1:
                    right_lookup,right_digits=lookup,len(table)-1
                    preparation.append(launch(f'clip_tails_{k}',(rows,1,1),(512,1,1),
                                              (src,scale,count,tail,*extra)))
                else:
                    preparation.append(launch(f'clip_table_{k}',(rows,1,1),(512,1,1),
                                              (src,h,g,h,scale,count,tail,*extra,lookup,INT(len(table)-1))))
        metadata=dict(base=args.base,clip=args.clip,cap=cap,fold=args.fold,seed=seed,dataset=args.dataset,
                      model=args.model.name if args.model else None)
        trans,tr,tc=TRANSPOSES[args.transpose]
        if not args.fused_right:
            for src,dst in ((bh,bht),(bg,bgt)):
                preparation.append(launch(trans,(k//tc,n//tr,1),(tr*tc//16,1,1),(src,dst,INT(n),INT(k))))
        right_inputs=[]
        gain_a=cap/127
        gain_b=gain_a if args.left_only else gain_a*args.coarse_clip/args.clip
        if args.tail_tile in BYTE_TILES:
            packed_width=DIRECTIONAL_TILES.get(args.tail_tile,64) if args.packed_tail else 0
            for side,(dense,scales,rows,gain) in enumerate(((db,sb,n,gain_b),(da,sa,m,gain_a))):
                if args.left_only and side==1:
                    break
                materialized=mem(k*rows)
                if args.bucket_width:
                    name='bucket_transpose_left' if side else 'bucket_transpose_right'
                    preparation.append(launch(name,(k//64,rows//64,1),(256,1,1),
                        (dense,bht,bgt,materialized,scales,right_lookup,factors[1-side],INT(rows),INT(k),
                         INT(right_digits),FLOAT(right_digits/cap),FLOAT(1/gain),INT(args.bucket_width))))
                elif side==0 and args.fused_right:
                    name='transpose_trained_hot' if args.hot_columns else 'transpose_trained_64'
                    mask=(keep,) if args.hot_columns else ()
                    preparation.append(launch(name,(k//64,rows//64,1),(256,1,1),
                                              (dense,bht,bgt,materialized,scales,right_lookup,INT(rows),INT(k),
                                               INT(right_digits),FLOAT(right_digits/cap),FLOAT(1/gain),INT(packed_width),*mask)))
                else:
                    name='transpose_coarse_hot' if args.hot_columns else 'transpose_coarse_64'
                    mask=(keep,) if args.hot_columns else ()
                    preparation.append(launch(name,(k//64,rows//64,1),(256,1,1),
                                              (dense,materialized,scales,INT(rows),INT(k),FLOAT(1/gain),INT(packed_width),*mask)))
                right_inputs.append(materialized)
        else:
            a_core=mem(2*m*k)
            preparation.append(launch('half_core',(m*k//512,1,1),(256,1,1),(da,a_core,sa,INT(m),INT(k),INT(cap))))
            for dense,rows,layout in ((db,n,args.left_dense),(a_core,m,args.right_dense)):
                if layout=='r':
                    materialized=mem(2*k*rows)
                    preparation.append(launch('transpose_half_64',(k//64,rows//64,1),(256,1,1),
                                              (dense,materialized,INT(rows),INT(k))))
                else:
                    materialized=dense
                right_inputs.append(materialized)
        gh=int8_gemm(blas,m,n,k,ah,bht,raw_h,ld_b=n,transpose_b=False,algorithm=args.algo)
        gg=int8_gemm(blas,m,n,k,ag,bgt,raw_g,ld_b=n,transpose_b=False,algorithm=args.algo)
        if args.tail_tile:
            if args.parallel_tail:
                ready=allocation.enter_context(cuda.dependency_event())
                corrected=allocation.enter_context(cuda.dependency_event())
                side=allocation.enter_context(cuda.stream())
                wait=api(cuda.driver,'cuStreamWaitEvent',PTR,PTR,UINT)
                merge_tails=launch('merge_tail_bytes',(n//128,m//16,1),(32,4,1),
                                   (raw_h,raw_g,sa,sb,out,INT(n),*weights))
                correction=launch('tail_correction_bytes',(n//64,m//64,1),(256,1,1),
                                   (raw_h,raw_g,sa,sb,ca,cb,ra,rb,right_inputs[1],right_inputs[0],out,
                                    INT(args.capacity),INT(m),INT(n),*weights,INT(args.tile_panel),
                                    FLOAT(gain_a),FLOAT(gain_b),INT(k if args.packed_tail else 0)),side)
            if args.tail_tile in DIRECTIONAL_TILES:
                width=DIRECTIONAL_TILES[args.tail_tile]
                other=4096//width
                prefix='tail_direction128' if width==128 else 'tail_direction'
                coeff_a,coeff_b=(ra,),(rb,)
                if width==128:
                    coeff_a,coeff_b=(ra,DEV_PTR(0)),(rb,DEV_PTR(0))
                if args.tail_packet_bits:
                    prefix=f'{prefix}_q{args.tail_packet_bits}'
                    if args.tail_streaming:
                        prefix+='_stream'
                    packer='pack_tail_one_byte' if args.tail_packet_bits==8 else 'pack_tail_two_bytes'
                    data=[]
                    residuals=((m,ra,ca),)
                    if not args.left_only:
                        residuals+=((n,rb,cb),)
                    for rows,source,counts in residuals:
                        packed=mem(rows*((args.capacity+3)//4)*16)
                        scales=mem(rows*4)
                        data.append((packed,scales))
                        preparation.append(launch(packer,(rows,1,1),(32,1,1),
                                                  (source,counts,packed,scales,INT(args.capacity))))
                    coeff_a=data[0]
                    if not args.left_only:
                        coeff_b=data[1]
                shape_args=(INT(n),INT(k),INT(args.capacity))
                grid_b=(n//other,m//width,1)
                partial=mem(2*m*n) if args.compact_tail else out
                if not args.left_only:
                    name_b='tail_direction128_compact_b' if args.compact_tail else f'{prefix}_b'
                    params_b=(sa,cb,*coeff_b,right_inputs[1],partial,*shape_args,FLOAT(gain_a))
                    finish_b=launch(name_b,grid_b,(256,1,1),params_b)
                name_a='tail_direction128_hot_a' if args.hot_columns else f'{prefix}_a'
                extra_finish=(hot_count,hot_a,hot_b) if args.hot_columns else ()
                if args.compact_tail:
                    name_a='tail_direction128_compact_a'
                    extra_finish=(partial,coeff_b[1],FLOAT(gain_a))
                if args.left_only:
                    name_a='tail_direction128_left_q8_a' if args.tail_packet_bits==8 else 'tail_direction128_left_a'
                finish_a=launch(name_a,(n//width,m//other,1),(256,1,1),
                                (raw_h,raw_g,sa,sb,ca,*coeff_a,right_inputs[0],out,INT(m),*shape_args,
                                 *weights,FLOAT(gain_b),*extra_finish))
                if args.overlap_direction:
                    ready=allocation.enter_context(cuda.dependency_event())
                    corrected=allocation.enter_context(cuda.dependency_event())
                    priority=cuda.priority_range()[1] if args.overlap_direction=='priority' else None
                    side=allocation.enter_context(cuda.stream(priority=priority))
                    wait=api(cuda.driver,'cuStreamWaitEvent',PTR,PTR,UINT)
                    correction=launch(name_b,grid_b,(256,1,1),params_b,side)

                def finish_tile():
                    if not args.left_only:
                        finish_b()
                    finish_a()
            else:
                threads={'tail_tile_4':128,'tail_tile_bytes_16':512,'tail_tile_bytes_u8':512}.get(args.tail_tile,256)
                finish_tile=launch(args.tail_tile,(n//64,m//64,1),(threads,1,1),
                                   (raw_h,raw_g,sa,sb,ca,cb,ra,rb,right_inputs[1],right_inputs[0],out,
                                    INT(args.capacity),INT(m),INT(n),*weights,INT(args.tile_panel),
                                    FLOAT(gain_a),FLOAT(gain_b),INT(k if args.packed_tail else 0)))

            def encode_inputs():
                for step in preparation:
                    step()

            def dense_products():
                gh()
                gg()

            def tile_pipeline():
                encode_inputs()
                if args.parallel_tail:
                    cuda.record_event(ready,None)
                    wait(side,ready,0)
                    correction()
                    cuda.record_event(corrected,side)
                    dense_products()
                    wait(None,corrected,0)
                    merge_tails()
                elif args.overlap_direction:
                    cuda.record_event(ready,None)
                    wait(side,ready,0)
                    correction()
                    cuda.record_event(corrected,side)
                    dense_products()
                    wait(None,corrected,0)
                    finish_a()
                else:
                    dense_products()
                    finish_tile()
                    if args.bucket_width:
                        bucket_correction()

            tile_pipeline()
            if cuda.read(status,(1,),np.uint32)[0]:
                raise RuntimeError('Input, integer range, or residual capacity error')
            quality=verify_on_gpu(cuda,blas,module,out,da,db,a,b) if args.gpu_reference else verify(cuda,out,a,b,args.probes)
            for _ in range(args.warmup):
                tile_pipeline()
            method='tail_bytes_parallel' if args.parallel_tail else args.tail_tile
            if args.bucket_width:
                method='tail_bytes_buckets'
            row=collect(cuda,shape,method,tile_pipeline,args.samples,tile_panel=args.tile_panel,
                        packed_tail=args.packed_tail,fused_right=args.fused_right,
                        **metadata,**quality)
            if args.tail_tile in BYTE_TILES and not args.left_only:
                row['coarse_clip']=args.coarse_clip
            if args.overlap_direction:
                row['overlap_direction']=args.overlap_direction
            if args.tail_packet_bits:
                row['tail_packet_bits']=args.tail_packet_bits
            if args.tail_streaming:
                row['tail_streaming']=True
            if args.compact_tail:
                row['compact_tail']=True
            if args.left_only:
                row['left_only']=True
                row['right_scale']='peak'
            if args.hot_columns:
                hot=int(cuda.read(hot_count,(1,),np.int32)[0])
                row['hot_columns']=sorted(cuda.read(hot_indices,(hot,),np.int32).tolist()) if hot else []
            row['phase_ms']={name:float(median(cuda.measure(op,1,5))) for name,op in
                             (('encode',encode_inputs),('two_int8_products',dense_products),('finish',finish_tile))}
            if args.bucket_width:
                row['bucket_width']=args.bucket_width
                row['tail_nnz']=[int(cuda.read(count,(rows,),np.uint16).sum()) for count,rows in ((ca,m),(cb,n))]
                row['phase_ms']['low_rank']=float(median(cuda.measure(bucket_correction,1,5)))
            return row

        right_arrays=[lib.dense(allocation,k,rows,ptr,2,layout) for rows,ptr,layout in
                      zip((n,m),right_inputs,(args.left_dense,args.right_dense))]
        overlap=args.overlap
        flip=args.separate or overlap
        out_a=mem(4*m*n) if overlap else out
        out_b=mem(4*m*n) if flip else out
        left_out=lib.dense(allocation,m,n,out_a,0,'r')
        right_out=lib.dense(allocation,n,m,out_b,0,'r' if flip else 'c')
        sparse=[]
        for side,(rows,cnt,tail) in enumerate(((m,ca,ra),(n,cb,rb))):
            off,idx,val=mem(4*(rows+1)),mem(4*rows*args.capacity),mem(2*rows*args.capacity)
            off_total=DEV_PTR(total.value+4*side)
            preparation.extend((launch('tail_offsets',(1,1,1),(256,1,1),(cnt,off,off_total,INT(rows))),
                                launch('compact_tail',(rows,1,1),(128,1,1),(tail,off,idx,val,INT(args.capacity)))))
            sparse.append((rows,off,idx,val))
        finish=launch('ar2_base_finish',(n//128,m//16,1),(32,4,1),(raw_h,raw_g,sa,sb,out,INT(m),INT(n),*weights))
        merge=launch('add_transposed',(m//32,n//32,1),(256,1,1),(out_b,out,INT(m),INT(n))) if flip else None
        if overlap:
            combine=launch('combine_sparse_ar2',(n//32,m//32,1),(256,1,1),
                           (raw_h,raw_g,out_a,out_b,sa,sb,out,INT(m),INT(n),*weights))
            first=allocation.enter_context(cuda.dependency_event())
            dense_done=allocation.enter_context(cuda.dependency_event())
            sparse_done=allocation.enter_context(cuda.dependency_event())
            ds=allocation.enter_context(cuda.stream())
            ss=allocation.enter_context(cuda.stream())
            allocation.callback(blas.set_stream,blas.handle,None)
            allocation.callback(lib.set_stream,lib.handle,None)
            blas.set_stream(blas.handle,ds)
            lib.set_stream(lib.handle,ss)
            wait=api(cuda.driver,'cuStreamWaitEvent',PTR,PTR,UINT)
        one=FLOAT(1)
        factors=(FLOAT(0 if overlap else 1),FLOAT(0 if flip else 1))

        def get_params(side,nnz):
            rows,*indices=sparse[side]
            matrix=lib.csr(rows,k,nnz,*indices)
            out_desc=(left_out,right_out)[side]
            pars=(lib.handle,0,0,ct.byref(one),matrix,right_arrays[side],ct.byref(factors[side]),out_desc,0,args.sparse_algo)
            return matrix,pars

        for operation in preparation:
            operation()
        error_flag=int(cuda.read(status,(1,),np.uint32)[0])
        if error_flag:
            raise RuntimeError(f'Input, integer range, or residual capacity error {error_flag}')
        initial_nnz=tuple(int(x) for x in cuda.read(total,(2,),np.int32))
        work=[]
        for side,count in enumerate(initial_nnz):
            size=SIZE()
            matrix,pars=get_params(side,count)
            try:
                lib.size(*pars,ct.byref(size))
                work.append(mem(max(size.value,1)))
            finally:
                lib.destroy_sparse(matrix)

        def corrections(current):
            if current!=initial_nnz:
                raise ValueError('Changed inputs require refreshing the workspace')
            for side,nnz in enumerate(current):
                if nnz:
                    matrix,pars=get_params(side,nnz)
                    try:
                        lib.mm(*pars,work[side])
                    finally:
                        lib.destroy_sparse(matrix)
                elif overlap:
                    fill=api(cuda.driver,'cuMemsetD32Async',DEV_PTR,UINT,SIZE,PTR)
                    fill((out_a,out_b)[side],0,m*n,ss)
                elif side==1 and flip:
                    return
            if merge and not overlap:
                merge()

        def complete():
            for operation in preparation:
                operation()
            current=tuple(int(x) for x in cuda.read(total,(2,),np.int32))
            if overlap:
                cuda.record_event(first,None)
                wait(ds,first,0)
                wait(ss,first,0)
                gh()
                gg()
                cuda.record_event(dense_done,ds)
                corrections(current)
                cuda.record_event(sparse_done,ss)
                wait(None,dense_done,0)
                wait(None,sparse_done,0)
                combine()
            else:
                gh()
                gg()
                finish()
                corrections(current)

        complete()
        quality=verify(cuda,out,a,b,args.probes)
        for _ in range(args.warmup):
            complete()
        row=collect(cuda,shape,'clipped_ar2_cusparse_e2e',complete,args.samples,tail_nnz=initial_nnz,
                    dense_order=(args.left_dense,args.right_dense),separate=flip,overlap=overlap,
                    sparse_algo=args.sparse_algo,**metadata,**quality)
        if not overlap:
            row['phase_ms']={name:float(median(cuda.measure(fn,1,3))) for name,fn in
                             (('pack_and_csr',lambda:[f() for f in preparation]),('gemms',lambda:(gh(),gg())),
                              ('merge',finish),('tails',lambda:(finish(),corrections(initial_nnz))))}
        return row


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gpu',required=True)
    p.add_argument('--shape',nargs=3,type=int,default=(4096,4096,8192))
    p.add_argument('--seeds',nargs='+',type=int,default=(2109,))
    p.add_argument('--dataset',choices=('normal','outliers','co_spike','uniform'),default='normal')
    p.add_argument('--clip',type=float,default=3.01)
    p.add_argument('--base',type=int,default=14)
    p.add_argument('--fold',action='store_true')
    p.add_argument('--model',type=Path,help='Use the cutoff, basis and digits from a trained codebook')
    p.add_argument('--clip-override',type=float,help='Use a different cutoff with the selected codebook')
    p.add_argument('--capacity',type=int,default=512)
    p.add_argument('--transpose',choices=TRANSPOSES,default='128-stream')
    p.add_argument('--algo',type=int,default=1)
    p.add_argument('--sparse-algo',type=int,default=6)
    p.add_argument('--left-dense',choices=('r','c'),default='r')
    p.add_argument('--right-dense',choices=('r','c'),default='r')
    p.add_argument('--separate',action='store_true')
    p.add_argument('--overlap',action='store_true')
    p.add_argument('--tail-tile',choices=('tail_tile_4','tail_tile_8','tail_tile_panel_8',*BYTE_TILES))
    p.add_argument('--tile-panel',type=int,default=2)
    p.add_argument('--coarse-clip',type=float,default=3.9)
    p.add_argument('--packed-tail',action='store_true')
    p.add_argument('--fused-right',action='store_true')
    p.add_argument('--parallel-tail',action='store_true')
    p.add_argument('--overlap-direction',choices=('default','priority'))
    p.add_argument('--tail-packet-bits',type=int,choices=(8,16))
    p.add_argument('--tail-streaming',action='store_true')
    p.add_argument('--compact-tail',action='store_true',help='Store the right-tail correction in normalized FP16')
    p.add_argument('--left-only',action='store_true',help='Clip only the left operand and scale the right by its peak')
    p.add_argument('--hot-columns',action='store_true',help='Restore strongly concentrated K columns in FP32')
    p.add_argument('--bucket-width',type=int,choices=(64,128,256))
    p.add_argument('--bucket-clip',type=float)
    p.add_argument('--warmup',type=int,default=12)
    p.add_argument('--samples',type=int,default=11)
    p.add_argument('--probes',type=int,default=0)
    p.add_argument('--gpu-reference',action='store_true')
    p.add_argument('--output',default='clipped-cusparse.json')
    args=p.parse_args()
    m,n,k=args.shape
    model=None
    if args.model:
        with np.load(args.model,allow_pickle=False) as data:
            settings=json.loads(str(data['metadata']))
            model=(data['a'],data['b'],data['weights'])
        for table in model[:2]:
            if (table.shape!=(settings['bins']*127+1,2) or table.dtype!=np.int8
                    or np.any(table==-128) or table[0].any()):
                p.error('Invalid codebook digits or size')
        if model[2].shape!=(2,) or not np.all(np.isfinite(model[2])) or model[2][1]==0:
            p.error('Invalid codebook weights')
        args.base,args.clip,args.fold=(settings[key] for key in ('base','clip','fold'))
    if args.clip_override is not None:
        if model is None:
            p.error('A cutoff override requires a trained codebook')
        args.clip=args.clip_override
    if args.bucket_clip is not None:
        if not args.bucket_width:
            p.error('A bucket clip override requires a bucket width')
        if args.clip_override is not None:
            p.error('Choose one cutoff override')
        args.clip=args.bucket_clip
    if not 2 <= args.base <= 128 or not np.isfinite(args.clip) or args.clip <= 0:
        p.error('Use a base from 2 through 128 and a finite positive clip')
    if min(m,n)<128 or m%128 or n%128 or k not in (8192,16384,32768,65536):
        p.error('Unsupported matrix alignment or K')
    if not 1<=args.capacity<=min(k,65535):
        p.error('Tail capacity must be positive and fit the row count representation')
    if args.tail_tile and (args.left_dense!='r' or args.right_dense!='r' or args.overlap or args.separate):
        p.error('The tiled correction requires row-major operands and a single stream')
    if args.tail_tile in ('tail_tile_panel_8',*BYTE_TILES) and (args.tile_panel<1 or (n//64)%args.tile_panel):
        p.error('The tile panel must divide the output tile columns')
    if not np.isfinite(args.coarse_clip) or args.coarse_clip<=0:
        p.error('Use a finite positive coarse clip')
    if args.packed_tail and args.tail_tile not in BYTE_TILES:
        p.error('Panel-major storage requires the INT8 correction tile')
    if args.fused_right and (model is None or args.tail_tile not in BYTE_TILES):
        p.error('Fused right packing requires the trained codebook and the INT8 correction tile')
    if args.parallel_tail and args.tail_tile!='tail_tile_bytes':
        p.error('Parallel correction requires the INT8 correction tile')
    if args.tail_tile in DIRECTIONAL_TILES and not args.packed_tail:
        p.error('Directional correction requires packed tails')
    if args.overlap_direction and (args.tail_tile not in DIRECTIONAL_TILES or args.bucket_width):
        p.error('Directional overlap requires directional correction without bucket shifts')
    if args.tail_packet_bits and args.tail_tile!='tail_directional128_bytes':
        p.error('Tail packets require 128-wide directional correction')
    if args.tail_streaming and args.tail_packet_bits!=16:
        p.error('Streaming stores require two-byte tail packets')
    if args.compact_tail and (args.tail_packet_bits!=16 or args.hot_columns or args.bucket_width or args.tail_streaming):
        p.error('Compact tails require the standard two-byte directional path')
    if args.left_only and (not args.fused_right or not args.tail_packet_bits or args.compact_tail or args.overlap_direction
                          or args.hot_columns or args.bucket_width or args.tail_streaming):
        p.error('Left-only tails require fused right packing and a single directional packet correction')
    if args.hot_columns and (args.tail_packet_bits!=16 or not args.fused_right or args.bucket_width or args.tail_streaming):
        p.error('Hot-column restoration requires two-byte tail packets and fused right encoding')
    if args.bucket_width and (not args.fused_right or not args.packed_tail or args.parallel_tail or k>32768):
        p.error('Bucket shifts require fused right encoding, packed tails, and K up to 32768')
    if args.bucket_width and args.tail_tile=='tail_directional128_bytes':
        p.error('Bucket shifts use 64-wide packed tails')
    if args.gpu_reference and (args.probes or not args.tail_tile):
        p.error('The GPU FP64 reference verifies the full output of tiled correction')
    files=('plane_sm61.cu','packed_layout.cu','clip_ar2.cu','clip_csr.cu','clip_tail_tile.cu')
    if args.bucket_width:
        files+=('bucket_shift.cu',)
    if args.gpu_reference:
        files+=('fp64_reference.cu',)
    if args.hot_columns:
        files+=('hot_columns.cu',)
    if args.left_only:
        files+=('dense08.cu',)
    ptx=compile_ptx(b'\n'.join((SOURCE_DIR/f).read_bytes() for f in files))
    results=[]
    with ExitStack() as stack:
        cuda=stack.enter_context(Cuda(args.gpu))
        module=stack.enter_context(cuda.load(ptx))
        blas=stack.enter_context(Cublas())
        library=SparseLibrary(stack)
        for seed in args.seeds:
            if args.dataset=='normal':
                a,b=sample_operands(tuple(args.shape),seed=seed)
            else:
                a,b=stress_operands(tuple(args.shape),args.dataset,seed)
            results.append(trial(cuda,blas,library,module,args,a,b,seed,model))
            (SOURCE_DIR/'results'/args.output).write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
