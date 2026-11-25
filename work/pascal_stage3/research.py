import argparse
import json
import os
import subprocess
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from bench import SOURCE_DIR, parse_shape, upload_many
from cuda import Cublas, Cuda, DEV_PTR, FLOAT, INT, compile_ptx


@dataclass(frozen=True)
class Spec:
    name: str
    base: int
    level_a: int = -1
    level_b: int = -1
    clip_a: float = 0
    clip_b: float = 0
    lowbit: str = ''


SPECS = (
    Spec('AR2-16',16), Spec('exact-32',32,0,0), Spec('exact-64',64,0,0), Spec('exact-128',128,0,0),
    Spec('LRC-16-unclipped',38,16,16), Spec('OP-16',38,16,16,3.7,3.7),
    Spec('OP-8',30,8,8,3.05,3.05), Spec('OP-4',26,4,4,2.5,2.5),
    Spec('OP-3x16',26,3,16,2.5,2.55), Spec('OP-3',22,3,3,2.30,2.35),
    Spec('OP-3-POPC',22,3,3,2.30,2.35,'ternary64'),
    Spec('LRC-2-POPC',18,2,2,0,0,'binary64'), Spec('OP-2-POPC',18,2,2,1.85,1.85,'binary64'),
)


def capacity(base, digit=127):
    q = np.arange(-base*(digit+2),base*(digit+2)+1,dtype=np.int64)
    h = np.rint(q/base).astype(np.int64)
    g = q-(base-1)*h
    return int(np.min(np.abs(q[(np.abs(h)>digit)|(np.abs(g)>digit)]))-1)


def sample(shape, kind, seed=17):
    m,n,k=shape
    rng=np.random.default_rng(seed)
    a=rng.standard_normal((m,k),dtype=np.float32)
    b=rng.standard_normal((n,k),dtype=np.float32)
    if kind=='uniform':
        a=rng.uniform(-1,1,a.shape); b=rng.uniform(-1,1,b.shape)
    if kind in ('outliers','co_spike'):
        indices=np.array([3,k//3,k*2//3]); a[:,indices]*=32
        b[:,indices if kind=='co_spike' else indices+1]*=32
    if kind=='span':
        a*=np.repeat(np.exp2(rng.integers(-8,9,size=(k+31)//32)),32)[:k]
        b*=np.repeat(np.exp2(rng.integers(-8,9,size=(k+31)//32)),32)[:k]
    if kind=='cancellation':
        a[:,k//2:]=a[:,:k//2]+0.001*a[:,k//2:]
        b[:,k//2:]=-b[:,:k//2]+0.001*b[:,k//2:]
    if kind=='finite_range':
        a=np.clip(a*np.exp2(rng.integers(-24,16,a.shape)),-65504,65504)
        b=np.clip(b*np.exp2(rng.integers(-24,16,b.shape)),-65504,65504)
        a[0,:5]=[65504,-65504,2**-24,-2**-24,0]
        b[0,:5]=[-65504,65504,-2**-24,2**-24,0]
    return a.astype(np.float16),b.astype(np.float16)


def metrics(value, reference):
    diff = value.astype(np.float64)-reference
    error=np.abs(diff)
    reference_norm=float(np.linalg.norm(reference))
    with np.errstate(over='ignore',invalid='ignore'):
        v16=value.astype(np.float16); ref16=reference.astype(np.float16)
    finite=np.isfinite(v16)&np.isfinite(ref16)
    vb=v16.view(np.uint16); rb=ref16.view(np.uint16)
    def ordered(half):
        v=half.astype(np.int32)
        return np.where(v>=32768,32768-(v&32767),32768+v)
    ulp=np.abs(ordered(vb[finite])-ordered(rb[finite]))
    return dict(relative_frobenius=float(np.linalg.norm(diff)/max(reference_norm,1e-100)),
                reference_frobenius=reference_norm,error_frobenius=float(np.linalg.norm(diff)),
                max_abs=float(error.max()),rmse=float(np.sqrt(np.mean(diff*diff))),
                abs_percentiles=np.percentile(error,[50,95,99]).tolist(),
                fp16_ulp_percentiles=np.percentile(ulp,[50,95,99,100]).tolist() if ulp.size else [],
                fp16_finite_count=int(finite.sum()),fp16_bitmatch=float(np.mean(vb==rb)),
                fp32_nan=int(np.isnan(value).sum()),fp32_inf=int(np.isinf(value).sum()),
                fp16_inf=int(np.isinf(v16).sum()),reference_fp16_inf=int(np.isinf(ref16).sum()))


def bind(cuda, fn, params, grid, block):
    return cuda.grid_invocation(fn,grid,block,params)


def summarize(ms, shape):
    median=float(np.median(ms))
    return dict(median_ms=median,p95_ms=float(np.percentile(ms,95)),samples_ms=list(ms),
                equivalent_TFLOPS=2*np.prod(shape).item()/median/1e9)


def measure(cuda,fn,shape,samples):
    return summarize(cuda.measure(fn,3 if shape[0]>=2048 else 5,samples),shape)


def run_case(cuda,blas,kernels,a,b,spec,reference,samples,kind):
    m,k=a.shape; n=b.shape[0]; shape=(m,n,k)
    if k%4 or n%4 or k>4096:
        raise ValueError('Kernel requires K and N to be multiples of four and K <= 4096')
    cap=1904 if spec.name=='AR2-16' else capacity(spec.base)
    with ExitStack() as stack:
        def mem(size): return stack.enter_context(cuda.buffer(size))
        da,db=upload_many(cuda,stack,(a,b)); flag=mem(np.zeros(1,np.int32))
        result=mem(4*m*n); p=mem(4*m*n); q=mem(4*m*n)
        r=mem(4*m*n) if spec.level_a>=0 else DEV_PTR()

        def encoder(raw,rows,levels,clip):
            h,g=mem(rows*k),mem(rows*k)
            rem=mem(rows*k) if levels>=0 else DEV_PTR()
            scales,cut=mem(4*rows),mem(4*rows)
            ix,dx,ct=(mem(2*rows*k),mem(4*rows*k),mem(4*rows)) if clip else (DEV_PTR(),)*3
            params=(raw,h,g,rem,scales,cut,ix,dx,ct,flag,INT(k),INT(spec.base),INT(cap),INT(levels),FLOAT(clip))
            launch=bind(cuda,kernels['encode_rows'],params,(rows,1,1),(256,1,1))
            return launch,(h,g,rem,scales,cut,ix,dx,ct)

        ea,A=encoder(da,m,spec.level_a,spec.clip_a)
        eb,B=encoder(db,n,spec.level_b,spec.clip_b)
        main_steps=[blas.invocation('int8',shape,A[0],B[0],p,algorithm=1),
                    blas.invocation('int8',shape,A[1],B[1],q,algorithm=1)]
        pack_a=[ea];pack_b=[eb]
        if spec.lowbit:
            if k%32 or m!=n:raise ValueError('POPC experiment requires K divisible by 32 and M=N')
            masks_a,masks_b=mem(m*k//4),mem(n*k//4)
            for data,mask,rows,list_of_ops in ((A,masks_a,m,pack_a),(B,masks_b,n,pack_b)):
                list_of_ops.append(bind(cuda,kernels['masks_from_int8'],(data[2],mask,INT(k),INT(rows)),
                                         ((k//32+7)//8,rows,1),(256,1,1)))
            main_steps.append(cuda.invocation(kernels[spec.lowbit],shape,(64,64),
                (masks_a,masks_b,r,INT(m),INT(k//32),INT(4 if spec.level_a==3 else 1)),block=(16,16,1)))
        elif spec.level_a>=0:
            main_steps.append(blas.invocation('int8',shape,A[2],B[2],r,algorithm=1))
        finish=cuda.invocation(kernels['reconstruct'],shape,(8,32),(p,q,r,A[3],B[3],result,
            INT(m),INT(n),INT(spec.base),INT(spec.level_a),INT(spec.level_b)),block=(32,8,1))
        trans_pre=[]; post=[finish]
        if spec.clip_a:
            bt=mem(4*k*n)
            trans_pre.append(bind(cuda,kernels['transpose_half'],(db,B[4],bt,INT(n),INT(k),INT(0)),
                                  ((k+31)//32,(n+31)//32,1),(32,8,1)))
            post.append(cuda.invocation(kernels['sparse_fp32'],shape,(8,32),(*A[5:],bt,result,
                            INT(m),INT(n),INT(k),INT(1)),block=(32,8,1)))
        if spec.clip_b:
            at,prec=mem(4*k*m),mem(4*m*n)
            trans_pre.append(bind(cuda,kernels['transpose_half'],(da,A[4],at,INT(m),INT(k),INT(bool(spec.clip_a))),
                                  ((k+31)//32,(m+31)//32,1),(32,8,1)))
            post.append(cuda.invocation(kernels['sparse_fp32'],(n,m,k),(8,32),(*B[5:],at,prec,
                            INT(n),INT(m),INT(k),INT(0)),block=(32,8,1)))
            post.append(bind(cuda,kernels['add_transpose'],(prec,result,INT(m),INT(n)),
                                     ((m+31)//32,(n+31)//32,1),(32,8,1)))

        def ordered(steps):
            def invoke():
                for step in steps: step()
            return invoke

        complete=ordered((*pack_a,*pack_b,*trans_pre,*main_steps,*post))
        complete()
        if cuda.read(flag,(1,),np.int32)[0]: raise ValueError('Nonfinite input to stage 2 encoder')
        actual=cuda.read(result,(m,n),np.float32)
        limit=min(8,m,n)
        def plane(handle): return cuda.read(handle,(limit,k),np.int8).astype(np.int64)
        ah,ag,bh,bg=map(plane,(A[0],A[1],B[0],B[1]))
        z=spec.base*((spec.base-1)*(ah@bh.T)+ag@bg.T)
        if spec.level_a>=0:
            low=plane(A[2])@plane(B[2]).T
            weight=1 if spec.level_a==0 else spec.base**2/(4*spec.level_a*spec.level_b)
            z=z-(spec.base-1)*weight*low
        sa=cuda.read(A[3],(limit,),np.float32); sb=cuda.read(B[3],(limit,),np.float32)
        reconstructed=z*(sa[:,None]*sb)
        if spec.clip_a or spec.clip_b:
            ac=a[:limit].astype(np.float64); bc=b[:limit].astype(np.float64)
            ca=cuda.read(A[4],(limit,),np.float32)[:,None]; cb=cuda.read(B[4],(limit,),np.float32)[:,None]
            va=np.clip(ac,-ca,ca) if spec.clip_a else ac
            vb=np.clip(bc,-cb,cb) if spec.clip_b else bc
            reconstructed+=(ac-va)@bc.T+va@(bc-vb).T
        np.testing.assert_allclose(actual[:limit,:limit],reconstructed,rtol=1e-4,
                                   atol=max(float(np.max(np.abs(reconstructed)))*2e-6,1e-20))
        timings={mode:measure(cuda,ordered(steps),shape,samples) for mode,steps in (
            ('A_main',main_steps),('B_prepacked',(*main_steps,*post)),('C_fp16', (*pack_a,*pack_b,*trans_pre,*main_steps,*post)),
            ('D_fixed_B',(*pack_a,*(trans_pre[1:] if spec.clip_a else trans_pre),*main_steps,*post)))}
        transfers=[]
        for _ in range(3):
            start=perf_counter()
            cuda.upload(da,a.ctypes.data,a.nbytes);cuda.upload(db,b.ctypes.data,b.nbytes)
            complete();cuda.read(result,(m,n),np.float32)
            transfers.append((perf_counter()-start)*1000)
        timings['E_host_roundtrip']=summarize(transfers,shape)
        counts=[float(np.mean(cuda.read(info[7],(size,),np.int32))/k) if clip else 0
                for info,size,clip in ((A,m,spec.clip_a),(B,n,spec.clip_b))]
        quality=metrics(actual,reference)
        print(f'{shape} {kind:14} {spec.name:18} {timings["C_fp16"]["equivalent_TFLOPS"]:6.2f} T/s '
              f'{100*quality["relative_frobenius"]:.6f}% sparse {counts}',flush=True)
        return dict(shape=shape,data=kind,spec=spec.__dict__,cap=cap,sparse_density=counts,
                    quality=quality,performance=timings)


def launch_monitor(gpu,path):
    file=path.open('w',encoding='utf-8')
    proc=subprocess.Popen(['nvidia-smi','-i',gpu,'--query-gpu=clocks.sm,temperature.gpu,power.draw,pstate,utilization.gpu,clocks_event_reasons.active',
                           '--format=csv','-lms','200'],stdout=file,stderr=subprocess.STDOUT,
                           creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    return proc,file


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',required=True)
    parser.add_argument('--shapes',nargs='+',default=('1024','2048','4096','256x4096x2048','32x4096x4096'))
    parser.add_argument('--samples',type=int,default=7)
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--cases',nargs='+',default=('normal',))
    parser.add_argument('--specs',nargs='+',default=tuple(s.name for s in SPECS if not s.lowbit))
    parser.add_argument('--output',type=Path,default=SOURCE_DIR/'results'/'research.json')
    parser.add_argument('--model-data',type=Path)
    args=parser.parse_args()
    args.output.parent.mkdir(exist_ok=True,parents=True)
    source=b'\n'.join((SOURCE_DIR/name).read_bytes() for name in ('stage2.cu','lowbit_tile.cu'))
    ptx=compile_ptx(source)
    (args.output.parent/'stage2.ptx').write_bytes(ptx.value)
    names=('encode_rows','reconstruct','transpose_half','sparse_fp32','add_transpose','sparse_scc',
           'masks_from_int8','ternary64','binary64')
    specs=[s for s in SPECS if s.name in args.specs]
    rows=[]
    monitor,log=launch_monitor(args.gpu,args.output.with_suffix('.telemetry.csv'))
    try:
        with Cuda(args.gpu) as cuda,cuda.load(ptx) as module,Cublas() as blas:
            print('GPU',cuda.name,'cuBLAS',blas.version,flush=True)
            kernels={name:cuda.kernel(module,name)[0] for name in names}
            def evaluate(a,b,kind):
                ref=a.astype(np.float64)@b.astype(np.float64).T
                for spec in specs:
                    rows.append(run_case(cuda,blas,kernels,a,b,spec,ref,args.samples,kind))
                    args.output.write_text(json.dumps(rows,indent=2),encoding='utf-8')
            if args.model_data:
                with np.load(args.model_data) as tensors:
                    for name in ('inference','gradient'):
                        evaluate(tensors[name+'_a'],tensors[name+'_b'],name)
            else:
                for shape in map(parse_shape,args.shapes):
                    for kind in args.cases:
                        evaluate(*sample(shape,kind,args.seed),kind)
    finally:
        monitor.terminate();monitor.wait();log.close()


if __name__=='__main__':main()
