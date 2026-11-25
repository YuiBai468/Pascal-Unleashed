"""Measure leading-dimension padding for the cuBLAS NN INT8 path."""

import argparse
from contextlib import ExitStack
from itertools import product
import json
import statistics

import numpy as np

from bench import SOURCE_DIR, upload_many
from cuda import Cublas, Cuda
from layout_scan import int8_gemm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--k', type=int, nargs='+', default=(5632, 5760, 6144))
    parser.add_argument('--samples', type=int, default=4)
    args = parser.parse_args()
    n = 4096
    rng = np.random.default_rng(311)
    a_host = rng.integers(-31, 32, n * (max(args.k) + 256), dtype=np.int8)
    b_host = rng.integers(-31, 32, max(args.k) * (n + 256), dtype=np.int8)
    rows = []
    with Cuda(args.gpu) as cuda, Cublas() as blas, ExitStack() as stack:
        a, b = upload_many(cuda, stack, (a_host, b_host))
        c = stack.enter_context(cuda.buffer(4 * n * (n + 256)))
        for k in args.k:
            ranked = []
            for pa,pb,pc in product((0,16,32,64,128), (0,16,32,64,128), (0,32,64,128)):
                da, db, dc = k + pa, n + pb, n + pc
                operation = int8_gemm(blas, n,n,k,a,b,c,ld_a=da,ld_b=db,ld_c=dc,transpose_b=False,algorithm=1)
                samples = cuda.measure(operation,2,args.samples)
                ranked.append((dict(k=k,pad_a=pa,pad_b=pb,pad_c=pc,ms=statistics.median(samples),samples=samples),operation))
            ranked.sort(key=lambda item:item[0]['ms'])
            for record, operation in ranked[:8]:
                pa,pb,pc = (record[f'pad_{p}'] for p in 'abc')
                sa, sb, sc = k+pa, n+pb, n+pc
                A = a_host[:n*sa].reshape(n,sa)[:32,:k].astype(np.int64)
                B = b_host[:k*sb].reshape(k,sb)[:,:32].astype(np.int64)
                operation()
                np.testing.assert_array_equal(A@B, cuda.read(c,(32,sc),np.int32)[:,:32])
                record['verified'] = True
            rows.extend(r for r,_ in ranked)
            print(k, [r for r,_ in ranked[:4]], flush=True)
            (SOURCE_DIR/'results'/'padded-layout.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
