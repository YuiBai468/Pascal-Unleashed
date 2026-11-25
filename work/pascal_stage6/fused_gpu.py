"""Measure DP4A kernels that directly write scaled FP32 results."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np

from bench import collect, SOURCE_DIR, upload_many
from cuda import Cuda, INT, compile_ptx
from frontier import full_error
from plane_gpu import packed_table
from plane_lattice import build
from rgs import sample_operands


KERNELS = {
    'fused_128_64_64_u2': (128, 64),
    'fused_64_128_64_u2': (64, 128),
    'fused_64_64_32_u2': (64, 64),
    'fused_64_64_64_u2': (64, 64),
    'fused_128_64_32_u2': (128, 64),
    'fused_128_64_64_u4': (128, 64),
    'fused_128_64_64_u2_linear': (128, 64),
    'pipe_128_64_32': (128, 64),
    'pipe_128_64_64': (128, 64),
    'pipe_128_128_32': (128, 128),
    'pipe_64_128_32': (64, 128),
    'pipe_64_64_32': (64, 64),
    'micro_128_64_8x4_32': (128,64,256),
    'micro_128_64_8x4_64': (128,64,256),
    'micro_64_128_4x8_64': (64,128,256),
    'micro_128_128_8x4_64': (128,128,512),
    'micro_128_128_4x8_64': (128,128,512),
    'micro_128_128_8x8_64': (128,128,256),
    'micro_128_128_8x8_32': (128,128,256),
}


def run(args):
    shape = (4096, 4096, 4096)
    m, n, k = shape
    physical_k = 3 * k // 2
    p, table = build(args.normal, args.penalty, 16)
    margin = 127 - int(np.abs(table).max())
    source = b'\n'.join((SOURCE_DIR / f).read_bytes() for f in ('plane_sm61.cu', 'fused_gemm.cu', 'micro_gemm.cu'))
    ptx = compile_ptx(source)
    all_results = []
    with Cuda(args.gpu) as cuda, cuda.load(ptx) as module:
        encode = cuda.kernel(module, 'plane_bytes_registers')[0]
        for seed in args.seeds:
            a, b = sample_operands(shape, seed=seed)
            gold = a.astype(np.float64) @ b.astype(np.float64).T
            norm = np.linalg.norm(gold)
            with ExitStack() as stack:
                def mem(length):
                    return stack.enter_context(cuda.buffer(length))

                da, db, basis, lookup = upload_many(cuda, stack, (a, b, p, packed_table(table)))
                pa, pb = mem(m * physical_k), mem(n * physical_k)
                sa, sb = mem(4 * m), mem(4 * n)
                stat = mem(np.zeros(1, np.uint32))
                out = mem(4 * m * n)
                packers = [cuda.grid_invocation(encode, (rows, 1, 1), (256, 1, 1),
                           (x, y, s, basis, lookup, stat, INT(k // 2), INT(16), INT(margin)))
                           for x, y, s, rows in ((da, pa, sa, m), (db, pb, sb, n))]
                for packer in packers:
                    packer()
                if cuda.read(stat, (1,), np.uint32)[0]:
                    raise ValueError('Nonfinite input or packing overflow')
                left, right = [cuda.read(x, (32, physical_k), np.int8).astype(np.int64) for x in (pa, pb)]
                sleft, sright = [cuda.read(x, (32,), np.float32) for x in (sa, sb)]
                dot = (left @ right.T).astype(np.float32) * (sleft[:, None] * sright)

                for name in args.kernels:
                    tile = KERNELS[name][:2]
                    fn, resources = cuda.kernel(module, name)
                    threads = KERNELS[name][2] if len(KERNELS[name])==3 else tile[0] * tile[1] // 64
                    gemm = cuda.invocation(fn, shape, tile, (pa, pb, sa, sb, out, INT(n), INT(physical_k)),
                                           block=(threads, 1, 1))
                    def complete():
                        for packer in packers:
                            packer()
                        gemm()

                    gemm()
                    result, quality = full_error(cuda, out, m, n, gold, norm)
                    np.testing.assert_array_equal(dot, result[:32, :32])
                    for _ in range(args.warmup):
                        complete()
                    record = collect(cuda, shape, name, complete, args.samples, seed=seed, **quality)
                    np.testing.assert_array_equal(result, cuda.read(out, (m, n), np.float32))
                    record.update(physical_k=physical_k, resources=resources, prepacked_ms=cuda.measure(gemm, 3, 5))
                    all_results.append(record)
                    args.output.write_text(json.dumps(all_results, indent=2), encoding='utf-8')


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--gpu', required=True)
    cli.add_argument('--kernels', nargs='+', choices=KERNELS, default=list(KERNELS))
    cli.add_argument('--seeds', nargs='+', type=int, default=(2109,))
    cli.add_argument('--normal', nargs=3, type=int, default=(3, 4, 5))
    cli.add_argument('--penalty', type=float, default=.002)
    cli.add_argument('--samples', type=int, default=7)
    cli.add_argument('--warmup', type=int, default=60)
    cli.add_argument('--output', type=Path, default=SOURCE_DIR / 'results' / 'fused-gemm.json')
    run(cli.parse_args())


if __name__ == '__main__':
    main()
