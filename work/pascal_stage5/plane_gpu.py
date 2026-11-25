import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np

from bench import SOURCE_DIR, collect, upload_many
from cuda import Cublas, Cuda, INT, compile_ptx
from frontier import full_error
from layout_scan import int8_gemm
from plane_lattice import build, line
from research import sample as stress_operands
from rgs import sample_operands


MIXED = {f'plane_mixed_{count}': count for count in (512, 1024, 1152, 1280, 1408, 1536, 1664, 2048)}
EPILOGUES = {'tiled': 'unpack_plane', 'linear': 'unpack_plane_linear', 'stream': 'unpack_plane_stream'}


def packed_table(lut):
    return np.asarray(sum((lut[:, j] + 8) << (4 * j) for j in range(lut.shape[1])), dtype=np.uint16)


def gpu_trial(cuda, blas, kernels, a,b,p,lut,bins,algo,layout,samples,seed,dataset,encoder,line_spec,tile,epilogue,warmup):
    m,k=a.shape;n=b.shape[0]; nominal=(m,n,k)
    dual = MIXED.get(encoder, 0)
    effective_k = 3*k//2 + dual//2
    margin=127-int(np.max(np.abs(lut)))
    if np.max(np.abs(lut)) > 7 or k % 8 or n % 4:
        raise ValueError('Unsupported operand dimensions or lookup offsets')
    lut_data = packed_table(lut)
    if encoder == 'plane_bytes_8192' and k != 8192:
        raise ValueError('This packer requires K=8192')
    if encoder != 'plane_bytes_8192' and k != 4096:
        raise ValueError('The register-based packers require K=4096')
    if dual:
        q, offsets = line_spec
        if np.max(np.abs(offsets)) > 7:
            raise ValueError('Line offsets do not fit their packed lookup table')
        margin = min(margin, 127 - int(np.max(np.abs(offsets))))
        p = np.concatenate((p.ravel(), q.ravel()))
        lut_data = np.concatenate((lut_data, packed_table(offsets)))
    gold=a.astype(np.float64)@b.astype(np.float64).T
    norm=max(np.linalg.norm(gold),1e-100)
    with ExitStack() as stack:
        def mem(x):return stack.enter_context(cuda.buffer(x))
        da,db,coef,lookup=upload_many(cuda,stack,(a,b,p,lut_data))
        pa,pb=mem(m*effective_k),mem(n*effective_k)
        sa,sb=mem(4*m),mem(4*n);stat=mem(np.zeros(1,np.uint32));raw,out=mem(4*m*n),mem(4*m*n)
        encode=[]
        for x,y,s,r in ((da,pa,sa,m),(db,pb,sb,n)):
            threads = 512 if encoder == 'plane_bytes_8192' else 256
            encode.append(cuda.grid_invocation(kernels[encoder],(r,1,1),(threads,1,1),
                          (x,y,s,coef,lookup,stat,INT(k//2),INT(bins),INT(margin))))
        if layout=='nn':
            pbt=mem(n*effective_k)
            if tile == 64 and (n % 64 or effective_k % 64):
                raise ValueError('The 64x64 transpose requires both sides divisible by 64')
            transpose = 'transpose_packed64' if tile==64 else 'transpose_packed'
            encode.append(cuda.grid_invocation(kernels[transpose],((effective_k+tile-1)//tile,(n+tile-1)//tile,1),
                          (256,1,1),(pb,pbt,INT(n),INT(effective_k))))
            g=int8_gemm(blas,m,n,effective_k,pa,pbt,raw,ld_b=n,transpose_b=False,algorithm=algo)
        else:
            g=blas.invocation('int8',(m,n,effective_k),pa,pb,raw,algorithm=algo)
        finish_args = (raw, sa, sb, out, INT(m), INT(n))
        if epilogue == 'tiled':
            finish = cuda.invocation(kernels[EPILOGUES[epilogue]], nominal, (16,128),
                                     finish_args, block=(32,4,1))
        else:
            if n % 1024:
                raise ValueError('The linear epilogue requires N divisible by 1024')
            finish = cuda.invocation(kernels[EPILOGUES[epilogue]], nominal, (1,1024),
                                     finish_args, block=(256,1,1))

        def packed_gemm():g();finish()

        def complete():
            for operation in encode:operation()
            packed_gemm()

        complete()
        if cuda.read(stat,(1,),np.uint32)[0]:raise ValueError('Nonfinite input or packing overflow')
        actual,quality=full_error(cuda,out,m,n,gold,norm)
        aa,bb=[cuda.read(x,(32,effective_k),np.int8).astype(np.int64) for x in (pa,pb)]
        dot=aa@bb.T
        np.testing.assert_array_equal(cuda.read(raw,(32,n),np.int32)[:,:32],dot)
        left,right=[cuda.read(x,(32,),np.float32) for x in (sa,sb)]
        expected=dot.astype(np.float32)*(left[:,None]*right)
        np.testing.assert_array_equal(expected,actual[:32,:32])
        for _ in range(warmup):complete()
        cuda.sync()
        record=collect(cuda,nominal,'plane_fp16_e2e',complete,samples,algo=algo,layout=layout,
                       seed=seed,dataset=dataset,physical_k=effective_k,dual=dual,
                       encoder=encoder,transpose_tile=tile,epilogue=epilogue,warmup=warmup,**quality)
        np.testing.assert_array_equal(actual,cuda.read(out,(m,n),np.float32))
        record['preparation_stages_ms']=[cuda.measure(fn,3,5) for fn in encode]
        record['prepacked_ms']=cuda.measure(packed_gemm,3,5)
        record['prepacked_main_ms']=cuda.measure(g,3,5)
        return record


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument('--gpu',required=True)
    cli.add_argument('--normal',nargs=3,type=int,default=(3,4,5))
    cli.add_argument('--penalty',type=float,default=.002)
    cli.add_argument('--bins',type=int,default=16)
    cli.add_argument('--layout',choices=('nn','tn'),default='nn')
    cli.add_argument('--algos',nargs='+',type=int,default=(1,))
    cli.add_argument('--seeds',nargs='+',type=int,default=(2109,))
    options=('plane_pack','plane_fast','plane_shared','plane_projected','plane_shared_projected',
             'plane_bytes_registers','plane_bytes_shared','plane_bytes_cached','plane_bytes_8192',*MIXED)
    cli.add_argument('--encoders',nargs='+',choices=options,default=('plane_bytes_registers',))
    cli.add_argument('--samples',type=int,default=7)
    cli.add_argument('--warmup',type=int,default=60)
    cli.add_argument('--line-normal',nargs=2,type=int,default=(3,4))
    cli.add_argument('--line-penalty',type=float,default=.0001)
    cli.add_argument('--transpose-tile',type=int,nargs='+',choices=(32,64),default=(64,))
    cli.add_argument('--epilogue',choices=EPILOGUES,default='tiled')
    cli.add_argument('--dataset',choices=('normal','uniform','outliers'),default='normal')
    cli.add_argument('--shape', type=int, nargs=3, default=(4096,4096,4096), metavar=('M','N','K'))
    cli.add_argument('--output',type=Path,default=SOURCE_DIR/'results'/'plane-gpu.json')
    args=cli.parse_args()
    if min(args.shape) < 32:
        raise ValueError('The independent integer comparison requires at least 32 rows and columns')
    if args.samples < 1 or args.warmup < 0:
        raise ValueError('Expected at least one timing sample and a nonnegative warm-up count')
    if not 2<=args.bins<=32:raise ValueError('bins must be between 2 and 32')
    if args.bins!=16 and any(x!='plane_pack' for x in args.encoders):
        raise ValueError('The optimized encoders require 16 bins per axis')
    p,lut=build(args.normal,args.penalty,args.bins)
    line_spec = line(args.line_normal, args.line_penalty, args.bins)
    source=b'\n'.join((SOURCE_DIR/x).read_bytes() for x in ('plane_sm61.cu','packed_layout.cu'))
    ptx=compile_ptx(source);records=[]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with Cuda(args.gpu) as cuda,cuda.load(ptx) as module,Cublas() as blas:
        print(cuda.name,'cuBLAS',blas.version,'normal',args.normal,'penalty',args.penalty,'LUT',args.bins,flush=True)
        entries={n:cuda.kernel(module,n) for n in (*options,*EPILOGUES.values(),'transpose_packed','transpose_packed64')}
        kernels={n:value[0] for n,value in entries.items()}
        print('pack_resources', {n:entries[n][1] for n in args.encoders},flush=True)
        for seed in args.seeds:
            shape=tuple(args.shape)
            a,b=sample_operands(shape,seed=seed) if args.dataset=='normal' else stress_operands(shape,args.dataset,seed)
            for algo in args.algos:
                for encoder in args.encoders:
                    for tile in args.transpose_tile:
                        row=gpu_trial(cuda,blas,kernels,a,b,p,lut,args.bins,algo,args.layout,
                                      args.samples,seed,args.dataset,encoder,line_spec,tile,args.epilogue,args.warmup)
                        row.update(normal=args.normal,penalty=args.penalty,bins=args.bins,resources=entries[encoder][1])
                        if encoder in MIXED:
                            row.update(line_normal=args.line_normal, line_penalty=args.line_penalty)
                        records.append(row)
                        args.output.write_text(json.dumps(records,indent=2),encoding='utf-8')


if __name__=='__main__':main()
