import argparse
import json
from contextlib import ExitStack

import numpy as np

from bench import SOURCE_DIR,grouped_invocation,upload_many
from cuda import Cublas,Cuda,INT,compile_ptx
from research import measure
from rgs import Encoding,encode,grouped_reference,sample_operands


def main():
    cli=argparse.ArgumentParser();cli.add_argument('--gpu',required=True);cli.add_argument('--size',type=int,default=4096)
    args=cli.parse_args();n=args.size
    if n<=0 or n>4096 or n%32:raise ValueError('Expected a positive square multiple of 32 up to 4096')
    shape=(n,)*3; a,b=sample_operands(shape); encoded=encode(a,b,n)
    sample=Encoding(encoded.a[:64],encoded.b[:64],encoded.alpha[:,:64],encoded.beta[:,:64],n)
    expected=grouped_reference(sample,n)
    sources=b'\n'.join((SOURCE_DIR/x).read_bytes() for x in ('rgs_sm61.cu','fused.cu'))
    ptx=compile_ptx(sources);items=[]
    with Cuda(args.gpu) as cuda,cuda.load(ptx) as module,Cublas() as blas,ExitStack() as stack:
        print('GPU',cuda.name,'cuBLAS',blas.version,flush=True)
        vals=upload_many(cuda,stack,(encoded.a,encoded.b,encoded.alpha,encoded.beta))
        out=stack.enter_context(cuda.buffer(4*n*n))
        for label,tile in (('rgs64x64_k64',(64,64)),('full32x64k32',(32,64)),('full64x64k32',(64,64)),
                           ('full64x64k64',(64,64)),('full64x128k32',(64,128))):
            fn,info=cuda.kernel(module,label)
            if label.startswith('rgs'):
                op=grouped_invocation(cuda,fn,label,shape,encoded,out,stack)
            else:
                op=cuda.invocation(fn,shape,tile,(*vals,out,INT(n),INT(n),INT(n//4)),block=(16,16,1))
            op(); np.testing.assert_array_equal(cuda.read(out,(n,n),np.float32)[:64,:64],expected)
            result=measure(cuda,op,shape,9)
            print(label,result['median_ms'],result['equivalent_TFLOPS'],info,flush=True)
            items.append(dict(name=label,performance=result,resources=info))
        with ExitStack() as refs:
            lt,rt=upload_many(cuda,refs,(a.astype(np.float32),b.astype(np.float32)))
            result=measure(cuda,blas.invocation('sgemm',shape,lt,rt,out),shape,9)
            print('cuBLAS SGEMM',result,flush=True)
            items.append(dict(name='cuBLAS SGEMM',performance=result))
        with ExitStack() as ints:
            x,y,u,v=upload_many(cuda,ints,[np.ascontiguousarray(data[:,:,i]) for data in (encoded.a,encoded.b) for i in range(2)])
            q=ints.enter_context(cuda.buffer(4*n*n))
            operations=[blas.invocation('int8',shape,p,r,s,algorithm=1) for p,r,s in ((x,u,out),(y,v,q))]
            def both():
                for op in operations:op()
            result=measure(cuda,both,shape,9)
            print('cuBLAS dual main',result,flush=True)
            items.append(dict(name='cuBLAS dual main',performance=result))
            for alpha,beta in ((15,0),(1,15),(1,1)):
                operations[0](); initial=cuda.read(out,(n,n),np.int32)[:8,:8]
                try:
                    blas.invocation('int8',shape,y,v,out,int_alpha=alpha,int_beta=beta,algorithm=1)()
                    observed=cuda.read(out,(n,n),np.int32)[:8,:8]
                    operations[1](); ref=cuda.read(q,(n,n),np.int32)[:8,:8]
                    np.testing.assert_array_equal(observed,alpha*ref+beta*initial)
                    diagnostic=f'alpha={alpha} beta={beta}: supported and matched reference'
                except RuntimeError as issue:
                    diagnostic=f'alpha={alpha} beta={beta}: {issue}'
                print(diagnostic,flush=True)
                items.append(dict(name='cuBLAS scalar probe',diagnostic=diagnostic))
    (SOURCE_DIR/'results'/'fused.json').write_text(json.dumps(items,indent=2),encoding='utf-8')


if __name__=='__main__':main()
