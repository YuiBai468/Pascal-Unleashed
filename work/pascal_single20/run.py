import argparse
from contextlib import ExitStack
import csv
import json
from pathlib import Path

import numpy as np

from cuda import Cublas, Cuda, INT, compile_ptx


ROOT=Path(__file__).resolve().parent


def sample(shape,seed,distribution):
    m,n,k=shape
    rng=np.random.default_rng(seed)
    a=rng.standard_normal((m,k),dtype=np.float32)
    bt=rng.standard_normal((n,k),dtype=np.float32)
    if distribution=='outliers':
        columns=np.array([3,k//3,2*k//3]);a[:,columns]*=32;bt[:,columns+1]*=32
    if distribution=='scale_span':
        for x in (a,bt):
            x*=np.repeat(np.exp2(rng.integers(-6,7,size=(k+31)//32)),32)[:k]
    if distribution=='zero_constant':
        a[0]=0;bt[0]=0;a[1]=1.25;bt[1]=-0.125
    return a.astype(np.float16),bt.astype(np.float16)


def ordered(*steps):
    def call():
        for step in steps:step()
    return call


def measure(cuda,job,shape,samples):
    times=cuda.measure(job,3,samples)
    median=float(np.median(times));work=2*shape[0]*shape[1]*shape[2]
    return dict(median_ms=median,p95_ms=float(np.percentile(times,95)),samples_ms=times,
                effective_TFLOPS=work/median/1e9)


def quality(actual,ref):
    diff=actual.astype(np.float64)-ref
    norm=float(np.linalg.norm(ref)); err=float(np.linalg.norm(diff))
    return dict(relative_error=err/norm if norm else (0 if err==0 else None),
                reference_frobenius=norm,error_frobenius=err,max_abs=float(np.abs(diff).max()),
                rmse=err/np.sqrt(diff.size),checked_elements=diff.size,
                nan=int(np.isnan(actual).sum()),infinity=int(np.isinf(actual).sum()))


def run_case(cuda,blas,kernels,shape,seed,distribution,samples,include_baseline):
    m,n,k=shape
    a,bt=sample(shape,seed,distribution)
    af,bf=a.astype(np.float64),bt.astype(np.float64)
    reference=af@bf.T
    records=[]
    with ExitStack() as stack:
        def mem(arg):return stack.enter_context(cuda.buffer(arg))
        da,db=mem(a),mem(np.ascontiguousarray(bt.T))
        db_row_major=mem(bt)
        transposed=mem(bt.nbytes);out=mem(4*m*n)
        qa,qb=mem(m*k),mem(n*k)
        sa,sb=mem(4*m),mem(4*n)
        ma,mb=mem(4*m),mem(4*n)
        u,v=mem(4*m),mem(4*n)
        nonfinite=mem(np.zeros(1,np.uint32))
        trans=cuda.grid_invocation(kernels['transpose_fp16'],((n+31)//32,(k+31)//32,1),
                                  (32,8,1),(db,transposed,INT(k),INT(n)))
        multiply=blas.invocation('int8',shape,qa,qb,out,algorithm=1)
        for label in ('symmetric','affine'):
            encoder=kernels[label+'_encode']
            ae=cuda.grid_invocation(encoder,(m,1,1),(256,1,1),(da,qa,sa,ma,u,nonfinite,INT(k//4)))
            finish=cuda.invocation(kernels['reconstruct_q8'],shape,(8,32),
                                   (out,sa,sb,ma,mb,u,v,INT(m),INT(n),INT(k),INT(label=='affine')),
                                   block=(32,8,1))
            for layout in ('ordinary_row_major_B','pretransposed_raw_B'):
                source=transposed if layout=='ordinary_row_major_B' else db_row_major
                be=cuda.grid_invocation(encoder,(n,1,1),(256,1,1),(source,qb,sb,mb,v,nonfinite,INT(k//4)))
                steps=(trans,ae,be,multiply,finish) if layout=='ordinary_row_major_B' else (ae,be,multiply,finish)
                operation=ordered(*steps)
                for _ in range(12):operation()
                cuda.sync()
                result=cuda.read(out,(m,n),np.float32)
                if cuda.read(nonfinite,(1,),np.uint32)[0]:raise ValueError('NaN/Inf in FP16 input')
                # Check the GPU epilogue against the actual encoded operands.
                rows=min(16,m);cols=min(16,n)
                ai=cuda.read(qa,(rows,k),np.int8).astype(np.int64)
                bi=cuda.read(qb,(cols,k),np.int8).astype(np.int64)
                ab=ai@bi.T
                comb=cuda.read(sa,(rows,),np.float32).astype(np.float64)[:,None]*cuda.read(sb,(cols,),np.float32)
                if label=='affine':
                    np.testing.assert_array_equal(cuda.read(u,(rows,),np.int32),ai.sum(1))
                    np.testing.assert_array_equal(cuda.read(v,(cols,),np.int32),bi.sum(1))
                    center=ai.sum(1)[:,None]*bi.sum(1)/k
                    means=cuda.read(ma,(rows,),np.float32).astype(np.float64)[:,None]*cuda.read(mb,(cols,),np.float32)
                    expected=(ab-center)*comb+k*means
                else:expected=ab*comb
                np.testing.assert_allclose(result[:rows,:cols],expected,rtol=4e-6,
                                           atol=max(float(np.max(np.abs(expected)))*1e-6,1e-10))
                timing=measure(cuda,operation,shape,samples)
                q=quality(result,reference)
                rel=q['relative_error']
                record=dict(shape=shape,dataset=distribution,seed=seed,method=label,layout=layout,
                            full_output_quality=q,device_fp16_to_fp32=timing,
                            pass_joint=bool(rel is not None and rel<.016 and timing['effective_TFLOPS']>=20))
                print(shape,seed,distribution,label,layout,
                      f'{timing["effective_TFLOPS"]:.3f} T/s {timing["median_ms"]:.4f} ms '
                      f'P95 {timing["p95_ms"]:.4f} ms error {100*rel:.6f}% joint {record["pass_joint"]}',flush=True)
                records.append(record)
        if include_baseline:
            sg=blas.invocation('sgemm',shape,mem(a.astype(np.float32)),mem(bt.astype(np.float32)),out)
            timing=measure(cuda,sg,shape,samples)
            print('resident FP32 SGEMM',timing['effective_TFLOPS'],timing['median_ms'],flush=True)
            records.append(dict(shape=shape,dataset=distribution,seed=seed,method='cuBLAS SGEMM (FP32 resident)',
                                resident_fp32_gemm=timing))
    return records


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument('--gpu',required=True)
    cli.add_argument('--shape',nargs=3,type=int,default=(4096,4096,4096))
    cli.add_argument('--seeds',type=int,nargs='+',default=(17,29,101))
    cli.add_argument('--samples',type=int,default=9)
    cli.add_argument('--data',nargs='+',choices=('normal','outliers','scale_span','zero_constant'),default=('normal',))
    cli.add_argument('--output',type=Path,default=ROOT/'results'/'single-q8.json')
    args=cli.parse_args();shape=tuple(args.shape)
    if min(shape)<2 or shape[1]%4 or shape[2]%4 or shape[2]>4096:
        raise ValueError('Requires M >= 2, N/K multiples of four and K <= 4096')
    ptx=compile_ptx((ROOT/'single.cu').read_bytes())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    (args.output.parent/'single-q8.ptx').write_bytes(ptx.value)
    with Cuda(args.gpu) as cuda,cuda.load(ptx) as mod,Cublas() as blas:
        names=('symmetric_encode','affine_encode','reconstruct_q8','transpose_fp16')
        kernels={name:cuda.kernel(mod,name)[0] for name in names}
        print('GPU',cuda.name,'cuBLAS',blas.version,flush=True)
        result=dict(gpu=cuda.name,cublas=blas.version,rows=[])
        for kind in args.data:
            for index,seed in enumerate(args.seeds):
                result['rows']+=run_case(cuda,blas,kernels,shape,seed,kind,args.samples,kind=='normal' and index==0)
                args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    with args.output.with_suffix('.csv').open('w',newline='',encoding='utf-8') as f:
        csv_writer=csv.writer(f)
        csv_writer.writerow(('shape','dataset','seed','method','layout','ms','p95_ms','TFLOPS','error_percent','joint_pass'))
        for r in result['rows']:
            if 'pass_joint' not in r:continue
            t=r['device_fp16_to_fp32'];q=r['full_output_quality']
            csv_writer.writerow((r['shape'],r['dataset'],r['seed'],r['method'],r['layout'],t['median_ms'],t['p95_ms'],
                                 t['effective_TFLOPS'],100*q['relative_error'],r['pass_joint']))


if __name__=='__main__':main()
