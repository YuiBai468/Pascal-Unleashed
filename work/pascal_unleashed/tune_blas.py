import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from bench import SOURCE_DIR, collect, parse_shape, upload_many
from cuda import Cublas, Cuda
from rgs import encode, sample_operands


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--shape', default='4096')
    parser.add_argument('--output', type=Path, default=SOURCE_DIR / 'results' / 'tune-blas.json')
    args = parser.parse_args()
    shape = parse_shape(args.shape)
    m, n, k = shape
    if k % 4:
        raise ValueError('The tuning shape must have a multiple-of-four K')
    a, b = sample_operands(shape)
    encoded = encode(a, b, min(4096, (k + 31) // 32 * 32))
    a8, b8 = [np.ascontiguousarray(matrix[:, :, 0]).view(np.int8) for matrix in (encoded.a, encoded.b)]
    expected = a8[:16].astype(np.int32) @ b8[:16].astype(np.int32).T
    rows = []
    with Cuda(args.gpu) as cuda, Cublas() as blas, ExitStack() as buffers:
        da, db = upload_many(cuda, buffers, (a8, b8))
        out = buffers.enter_context(cuda.buffer(4 * m * n))
        for algo in (-1, *range(24)):
            operation = blas.invocation('int8', shape, da, db, out, algorithm=algo)
            try:
                row = collect(cuda, shape, 'int8_cublas', operation, 3, algorithm=algo)
                np.testing.assert_array_equal(cuda.read(out, (m, n), np.int32)[:16, :16], expected)
                rows.append(row)
            except RuntimeError as failure:
                print(f'algo {algo}: {failure}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
