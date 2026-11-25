"""Probe which cuBLAS accumulation/output formats retain INT8 throughput."""

import argparse
from contextlib import ExitStack
import ctypes as ct
import json
import statistics

import numpy as np

from bench import SOURCE_DIR, upload_many
from cuda import Cublas, Cuda, DEV_PTR, FLOAT, INT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--k', nargs='+', type=int, default=(5632, 5760, 6144))
    args = parser.parse_args()
    n = 4096
    rng = np.random.default_rng(307)
    results = []
    cases = [(72, 10, INT), (68, 0, FLOAT), (69, 0, FLOAT), (72, 0, INT)]
    with Cuda(args.gpu) as cuda, Cublas() as blas:
        for k in args.k:
            left = rng.integers(-32, 33, (n,k), dtype=np.int8)
            right = rng.integers(-32, 33, (k,n), dtype=np.int8)
            gold = left[:32].astype(np.int64) @ right[:,:32].astype(np.int64)
            with ExitStack() as stack:
                a,b = upload_many(cuda, stack, (left,right))
                c = stack.enter_context(cuda.buffer(4*n*n))
                for compute, output, scalar in cases:
                    for algo in (-1,1):
                        row = dict(compute=compute, output=output, k=k, algo=algo)
                        one,zero=scalar(1),scalar(0)
                        def gemm():
                            blas.gemm_ex(blas.handle,0,0,n,n,k,ct.byref(one),b,3,n,a,3,k,
                                         ct.byref(zero),c,output,n,compute,algo)

                        try:
                            times = cuda.measure(gemm, 2, 5)
                            arr = cuda.read(c, (32,n), np.float32 if output == 0 else np.int32)[:,:32]
                            np.testing.assert_array_equal(arr,gold)
                            row.update(ms=statistics.median(times),samples=times)
                        except RuntimeError as error:
                            row['error'] = str(error)
                        results.append(row)
                        print(row,flush=True)
    (SOURCE_DIR/'results'/'raw-formats.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
