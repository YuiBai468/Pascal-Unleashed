"""Benchmark mixed-width INT8 encodings with normal-direction translations."""

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path

import numpy as np

from bench import SOURCE_DIR, collect, upload_many
from cuda import Cublas, Cuda, FLOAT, INT, compile_ptx
from frontier import full_error
from hyperplane_cpu import embedding
from layout_scan import int8_gemm
from plane_gpu import packed_table
from plane_lattice import build, build_offsets, line
from research import sample as stress_operands
from rgs import sample_operands


SHAPE = (4096,) * 3
KERNEL_SOURCE = ('plane_sm61.cu', 'packed_layout.cu', 'normal_slide.cu', 'slide_line_large.cu')
TRANSPOSES = {'64': ('transpose_packed64',64,64), '64-stream': ('transpose_64_64_stream',64,64),
              '128x64': ('transpose_128_64',128,64), '64x128': ('transpose_64_128',64,128),
              '128': ('transpose_128_128',128,128), '128-stream': ('transpose_128_128_stream',128,128)}


def lookup_tables(bins, pair_penalty, triple_penalty):
    pair_basis, pair_lut = build((3, 4, 5), pair_penalty, 16)
    direction, triple_basis = embedding((3, 4, 5, 6))
    triple_lut = build_offsets(direction, triple_basis, triple_penalty, bins)
    room = max(math.ceil(2.5 + np.abs(pair_lut).max() + .5),
               math.ceil(3 + np.abs(triple_lut).max() + .5))
    return (pair_basis, triple_basis, packed_table(pair_lut), packed_table(triple_lut)), 127 - room


def benchmark(cuda, blas, module, args, inputs, tables, margin, triples, dual, seed):
    a, b, target, denominator = inputs
    m, k = a.shape
    n = b.shape[0]
    shape = (m, n, k)
    physical_k = 3*k//2 - triples//2 + dual//2
    kernel = f'slide_{triples}_{args.bins.bit_length() - 1 if triples else 4}'
    if k==8192:
        kernel = f'slide_line_{dual}' if dual else 'slide_8192'
    if args.profile:
        kernel = f'profile_{triples}_{args.profile}'
    if args.vector:
        kernel = f'vec_{triples}_{args.vector}'
    encode, resources = cuda.kernel(module, kernel)
    transpose_name, tile_rows, tile_k = TRANSPOSES[args.transpose]
    transpose = cuda.kernel(module, transpose_name)[0]
    epilogues={'tiled':('unpack_plane',(16,128),(32,4,1)),
               'linear':('unpack_plane_linear',(1,1024),(256,1,1)),
               'stream':('unpack_plane_stream',(1,1024),(256,1,1))}
    epilogue_name, epilogue_tile, epilogue_threads=epilogues[args.epilogue]
    reconstruct = cuda.kernel(module, epilogue_name)[0]
    if n % tile_rows or physical_k % tile_k:
        raise ValueError('Transpose tile does not divide the packed matrix')
    if args.epilogue!='tiled' and n%1024:
        raise ValueError('The linear epilogue requires N divisible by 1024')
    with ExitStack() as stack:
        def mem(length):
            return stack.enter_context(cuda.buffer(length))

        da, db, *params = upload_many(cuda, stack, (a, b, *tables))
        pa, pbt = (mem(m * physical_k) for _ in range(2))
        sa, sb = mem(4 * m), mem(4 * n)
        raw = mem(4 * m * n)
        out = raw if args.inplace else mem(4 * m * n)
        status = mem(np.zeros(1, dtype=np.uint32))
        common = (*params, status, INT(margin), FLOAT(args.pair_blend), FLOAT(args.line_blend if dual else args.triple_blend))
        threads = 512 if k==8192 else args.vector or 256
        preparation = [cuda.grid_invocation(encode, (m, 1, 1), (threads, 1, 1), (da, pa, sa, *common))]
        if args.direct:
            stats = cuda.kernel(module, f'stat_{triples}_4')[0]
            right_pack = cuda.kernel(module, f'right_{triples}_4')[0]
            preparation.append(cuda.grid_invocation(stats, (n, 1, 1), (256, 1, 1), (db, pbt, sb, *common)))
            source_groups = triples + (k - 3 * triples) // 2
            preparation.append(cuda.grid_invocation(right_pack, (source_groups // 32, n // 32, 1), (256, 1, 1),
                                                    (db, pbt, sb, *params, status, INT(n))))
        else:
            pb = mem(n * physical_k)
            preparation.append(cuda.grid_invocation(encode, (n, 1, 1), (threads, 1, 1), (db, pb, sb, *common)))
            preparation.append(cuda.grid_invocation(transpose, (physical_k // tile_k, n // tile_rows, 1),
                                                    (tile_rows * tile_k // 16, 1, 1),
                                                    (pb, pbt, INT(n), INT(physical_k))))
        gemm = int8_gemm(blas, m, n, physical_k, pa, pbt, raw, ld_b=n, transpose_b=False, algorithm=args.algorithm)
        finish = cuda.invocation(reconstruct, shape, epilogue_tile, (raw, sa, sb, out, INT(m), INT(n)), block=epilogue_threads)

        def complete():
            for stage in preparation:
                stage()
            gemm()
            finish()

        complete()
        flag = cuda.read(status, (1,), np.uint32)[0]
        if flag:
            raise ValueError(f'Input or integer range check failed ({flag})')
        actual, quality = full_error(cuda, out, m, n, target, denominator)
        left = cuda.read(pa, (32, physical_k), np.int8).astype(np.int64)
        if args.direct:
            right = cuda.read(pbt, (physical_k, n), np.int8)[:, :32].T.astype(np.int64)
        else:
            right = cuda.read(pb, (32, physical_k), np.int8).astype(np.int64)
        sl, sr = [cuda.read(x, (32,), np.float32) for x in (sa, sb)]
        dot = left @ right.T
        if not args.inplace:
            np.testing.assert_array_equal(dot, cuda.read(raw, (32, n), np.int32)[:, :32])
        np.testing.assert_array_equal(dot.astype(np.float32) * (sl[:, None] * sr), actual[:32, :32])
        for _ in range(args.warmup):
            complete()
        record = collect(cuda, shape, 'normal_slide_fp16_e2e', complete, args.samples,
                         seed=seed, dataset=args.dataset, triples=triples, dual=dual, physical_k=physical_k,
                         margin=margin, bins=args.bins, pair_blend=args.pair_blend, triple_blend=args.triple_blend,
                         pair_penalty=args.pair_penalty, triple_penalty=args.triple_penalty,
                         profile=args.profile, direct=args.direct, inplace=args.inplace, vector=args.vector,
                         transpose=args.transpose, epilogue=args.epilogue, **quality)
        np.testing.assert_array_equal(cuda.read(out, (m, n), np.float32), actual)
        record['resources'] = resources
        record['preparation_ms'] = [cuda.measure(stage, 3, 5) for stage in preparation]
        record['gemm_ms'] = cuda.measure(gemm, 3, 5)
        if not args.inplace:
            record['finish_ms'] = cuda.measure(finish, 3, 5)
        return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--shape', type=int, nargs=3, default=SHAPE)
    parser.add_argument('--dual', type=int, nargs='+', choices=(0,1024,1280,1536,1792,2048), default=(0,))
    parser.add_argument('--line-blend', type=float, default=.89)
    parser.add_argument('--triples', type=int, nargs='+', choices=(0, 768, 896, 1024, 1152, 1280), default=(768, 896, 1024))
    parser.add_argument('--bins', type=int, choices=(8, 16), default=16)
    parser.add_argument('--pair-blend', type=float, default=.81)
    parser.add_argument('--triple-blend', type=float, default=.87)
    parser.add_argument('--pair-penalty', type=float, default=.002)
    parser.add_argument('--triple-penalty', type=float, default=.004)
    parser.add_argument('--samples', type=int, default=9)
    parser.add_argument('--warmup', type=int, default=60)
    parser.add_argument('--algorithm', type=int, default=1)
    parser.add_argument('--profile', type=int, choices=(0, 1, 2), default=0)
    parser.add_argument('--direct', action='store_true')
    parser.add_argument('--inplace', action='store_true')
    parser.add_argument('--transpose',choices=TRANSPOSES,default='64')
    parser.add_argument('--epilogue',choices=('tiled','linear','stream'),default='tiled')
    parser.add_argument('--vector', type=int, choices=(0, 128, 256, 512), default=0)
    parser.add_argument('--seeds', type=int, nargs='+', default=(2109,))
    parser.add_argument('--dataset', choices=('normal', 'uniform', 'outliers', 'co_spike'), default='normal')
    parser.add_argument('--output', type=Path, default=SOURCE_DIR / 'results' / 'slide-gpu.json')
    args = parser.parse_args()
    m,n,k=args.shape
    if m<32 or n<32 or n%64 or k not in (4096,8192):
        parser.error('Expected M and a 64-aligned N of at least 32, and K in {4096,8192}')
    if k==8192 and (set(args.triples)!={0} or args.vector or args.profile or args.direct or args.bins!=16):
        parser.error('K=8192 uses the 16-bin pair or pair/line encoder and requires --triples 0')
    if k!=8192 and set(args.dual)!={0}:
        parser.error('The sliding line encoder requires K=8192')
    if args.profile and (args.bins != 16 or set(args.triples) - {0, 768}):
        parser.error('The diagnostic variants use 16 bins and support 0 or 768 triples')
    if args.direct and (args.bins != 16 or 0 in args.triples or args.profile):
        parser.error('Direct right packing requires the regular 16-bin path with nonzero triples')
    vector_counts = {128: {768}, 256: {768, 896, 1024, 1152, 1280}, 512: {768, 896, 1024}}
    if args.vector and (args.direct or args.profile or args.bins != 16 or set(args.triples) - vector_counts[args.vector]):
        parser.error('Unavailable combination for vectorized encoding')
    tables, margin = lookup_tables(args.bins, args.pair_penalty, args.triple_penalty)
    line_spec = None
    if any(args.dual):
        _, line_lut = line((3,4),.0001,16)
        pair_basis,pair_lut=build((3,4,5),args.pair_penalty,16)
        line_spec=(pair_basis,packed_table(pair_lut),packed_table(line_lut))
        line_margin=127-max(math.ceil(2.5+np.abs(pair_lut).max()+.5),math.ceil(2+np.abs(line_lut).max()+.5))
    ptx = compile_ptx(b'\n'.join((SOURCE_DIR / src).read_bytes() for src in KERNEL_SOURCE))
    results = []
    with Cuda(args.gpu) as cuda, cuda.load(ptx) as module, Cublas() as blas:
        for seed in args.seeds:
            if args.dataset == 'normal':
                a, b = sample_operands(tuple(args.shape), seed=seed)
            else:
                a, b = stress_operands(tuple(args.shape), args.dataset, seed)
            target = a.astype(np.float64) @ b.astype(np.float64).T
            inputs = (a, b, target, max(float(np.linalg.norm(target)), 1e-100))
            for triples in args.triples:
                for dual in args.dual:
                    packed, limit = (line_spec,line_margin) if dual else (tables,margin)
                    results.append(benchmark(cuda, blas, module, args, inputs, packed, limit, triples, dual, seed))
                    args.output.write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
