import argparse
import json
from contextlib import ExitStack

import numpy as np

from bench import SOURCE_DIR,upload_many
from cuda import Cuda,INT,compile_ptx
from research import measure


def main():
    cli=argparse.ArgumentParser();cli.add_argument('--gpu',required=True);cli.add_argument('--size',type=int,default=2048)
    args=cli.parse_args(); size=args.size
    rng=np.random.default_rng(59); shape=(size,)*3
    dense=rng.integers(-4000,4001,(size,size),dtype=np.int16)
    ptx=compile_ptx((SOURCE_DIR/'stage2.cu').read_bytes());results=[]
    with Cuda(args.gpu) as cuda,cuda.load(ptx) as module:
        print('GPU',cuda.name,flush=True)
        for density in (.0005,.0045,.023,.0304,.1238):
            length=max(1,int(np.rint(size*density)))
            ix=np.zeros((size,size),np.uint16)
            for row in ix:row[:length]=rng.choice(size,length,replace=False)
            signs=rng.choice(np.array([-1,1],np.int8),size=(size,size))
            ct=np.full(size,length,np.int32)
            result={}
            for integer in (True,False):
                with ExitStack() as stack:
                    values=signs if integer else signs.astype(np.float32)/8
                    matrix=dense if integer else dense.astype(np.float32)/1024
                    fields=upload_many(cuda,stack,(ix,values,ct,matrix))
                    out=stack.enter_context(cuda.buffer(4*size*size))
                    name='sparse_scc' if integer else 'sparse_fp32'
                    fn,_=cuda.kernel(module,name)
                    params=(*fields,out,INT(size),INT(size),INT(size))
                    if not integer:params=(*params,INT(0))
                    operation=cuda.invocation(fn,shape,(8,32),params,block=(32,8,1))
                    operation()
                    actual=cuda.read(out,(size,size),np.int32 if integer else np.float32)
                    expected=np.einsum('ij,ijk->ik',values[:16,:length].astype(np.float64),matrix[ix[:16,:length],:16])
                    np.testing.assert_array_equal(actual[:16,:16],expected)
                    timing=measure(cuda,operation,shape,9)
                    result[name]=timing
                    print(name,'density',round(length/size,6),'time(ms)',round(timing['median_ms'],5),flush=True)
            results.append(dict(shape=shape,density=length/size,timings=result))
    (SOURCE_DIR/'results'/'sparse.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':main()
