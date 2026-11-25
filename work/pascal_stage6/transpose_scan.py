"""Time alternative byte transposes and verify every transposed element."""

import argparse
from contextlib import ExitStack
import json
import statistics

import numpy as np

from bench import SOURCE_DIR
from cuda import Cuda, INT, compile_ptx
from slide_gpu import TRANSPOSES


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',required=True)
    parser.add_argument('--dims',nargs=2,type=int,default=(8192,12928))
    args=parser.parse_args()
    rows,cols=args.dims
    rng=np.random.default_rng(435)
    source=rng.integers(-128,128,(rows,cols),dtype=np.int8)
    ptx=compile_ptx((SOURCE_DIR/'packed_layout.cu').read_bytes())
    records=[]
    with Cuda(args.gpu) as cuda,cuda.load(ptx) as module,ExitStack() as stack:
        a=stack.enter_context(cuda.buffer(source))
        b=stack.enter_context(cuda.buffer(source.nbytes))
        for key,(name,tr,tc) in TRANSPOSES.items():
            if rows%tr or cols%tc:continue
            kernel,resources=cuda.kernel(module,name)
            op=cuda.grid_invocation(kernel,(cols//tc,rows//tr,1),(tr*tc//16,1,1),(a,b,INT(rows),INT(cols)))
            op()
            np.testing.assert_array_equal(cuda.read(b,(cols,rows),np.int8),source.T)
            for _ in range(110):op()
            times=cuda.measure(op,4,17)
            r=dict(rows=rows,cols=cols,tile=key,median=statistics.median(times),samples=times,resources=resources)
            records.append(r)
            print(key,r['median'],flush=True)
    output=SOURCE_DIR/'results'/f'transpose-{rows}-{cols}.json'
    output.write_text(json.dumps(records,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
