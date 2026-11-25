"""Pipeline row slices while timing every resident-input preparation stage."""

import argparse
from contextlib import ExitStack
import ctypes as ct
import json
from pathlib import Path

import numpy as np

from bench import SOURCE_DIR, collect, upload_many
from cuda import Cublas, Cuda, DEV_PTR, FLOAT, INT, PTR, UINT, api, compile_ptx
from frontier import full_error
from layout_scan import int8_gemm
from plane_gpu import packed_table
from plane_lattice import build
from rgs import sample_operands
from slide_gpu import TRANSPOSES, lookup_tables


def shift(buffer, byte_offset):
    return DEV_PTR(buffer.value + byte_offset)


def trial(cuda, blas, kernels, operands, quantizer, seed, partition, args):
    a, b, gold, norm = operands
    m, k = a.shape
    n = b.shape[0]
    physical = 3 * k // 2 - args.triples // 2
    if sum(partition)!=m or min(partition)<128 or any(rows%128 for rows in partition):
        raise ValueError('Expected a complete partition into slices divisible by 128')
    shape = (m, n, k)
    tables, margin = quantizer
    split, direct = not args.serial, args.direct

    with ExitStack() as stack:
        def mem(spec):
            return stack.enter_context(cuda.buffer(spec))

        def event():
            return stack.enter_context(cuda.dependency_event())

        da, db, *coeff = upload_many(cuda, stack, (a, b, *tables))
        pa, pbt = mem(m * physical), mem(n * physical)
        sa, sb = mem(4 * m), mem(4 * n)
        check = mem(np.zeros(1, dtype=np.uint32))
        raw, out = mem(4 * m * n), mem(4 * m * n)
        priorities = (None,)*3
        if args.priority != 'default':
            low,high = cuda.priority_range()
            priorities = (low,low,high) if args.priority=='output' else (high,low,high)
        prep, math, output = (stack.enter_context(cuda.stream(p)) for p in priorities) if split else (None,) * 3
        blas.set_stream(blas.handle, math)
        stack.callback(blas.set_stream, blas.handle, None)
        wait = api(cuda.driver, 'cuStreamWaitEvent', PTR, PTR, UINT)
        start, right_ready, finished = event(), event(), event()

        def packing(inp, packed, scale, length, stream):
            name = f'vec_{args.triples}_{args.vector}' if args.triples else 'plane_bytes_registers'
            suffix = (check,INT(margin),FLOAT(.81),FLOAT(.87)) if args.triples else (check,INT(k//2),INT(16),INT(margin))
            return cuda.grid_invocation(kernels[name], (length, 1, 1), (args.vector or 256, 1, 1),
                                        (inp, packed, scale, *coeff, *suffix), stream)

        if direct:
            coef, lookup = coeff
            factor = mem(4 * n)
            prepare_right = cuda.grid_invocation(kernels['projected_row_scales'], (n, 1, 1), (256, 1, 1),
                              (db, coef, sb, factor, check, INT(k // 2), INT(margin)), prep)
            transpose = cuda.grid_invocation(kernels['pack_transposed_right'], (k // 64, n // 32, 1),
                              (256, 1, 1), (db, pbt, coef, lookup, factor, INT(n), INT(k // 2)), prep)
        else:
            pb = mem(n * physical)
            prepare_right = packing(db, pb, sb, n, prep)
            transpose_name, tile_rows, tile_k = TRANSPOSES[args.transpose]
            transpose = cuda.grid_invocation(kernels[transpose_name], (physical//tile_k,n//tile_rows,1),
                              (tile_rows*tile_k//16,1,1), (pb,pbt,INT(n),INT(physical)),prep)
        operations = []
        cursor = 0
        for extent in partition:
            if args.axis=='row':
                pas, sas = shift(pa, cursor*physical), shift(sa, cursor*4)
                raw_part, out_part = shift(raw,cursor*n*4), shift(out,cursor*n*4)
                pack = packing(shift(da,cursor*k*2),pas,sas,extent,prep)
                work = int8_gemm(blas,extent,n,physical,pas,pbt,raw_part,ld_b=n,
                                 transpose_b=False,algorithm=args.algorithm)
                finish = cuda.grid_invocation(kernels['unpack_plane'],(n//128,extent//16,1),(32,4,1),
                                    (raw_part,sas,sb,out_part,INT(extent),INT(n)),output)
            else:
                pack = None
                raw_part,out_part = shift(raw,cursor*4),shift(out,cursor*4)
                work = int8_gemm(blas,m,extent,physical,pa,shift(pbt,cursor),raw_part,ld_b=n,ld_c=n,
                                 transpose_b=False,algorithm=args.algorithm)
                finish = cuda.grid_invocation(kernels['unpack_plane_slice'],(extent//128,m//16,1),(32,4,1),
                                    (raw_part,sa,shift(sb,cursor*4),out_part,INT(m),INT(extent),INT(n)),output)
            operations.append((pack, work, finish, event(), event()))
            cursor += extent
        prepare_left = packing(da,pa,sa,m,prep) if args.axis=='col' else None

        def complete():
            if split:
                cuda.record_event(start, None)
                wait(prep, start, 0)
            prepare_right()
            transpose()
            if prepare_left:
                prepare_left()
            if split:
                cuda.record_event(right_ready, prep)
                wait(math, right_ready, 0)
            for pack, work, finish, input_ready, result_ready in operations:
                if pack:
                    pack()
                if split and pack:
                    cuda.record_event(input_ready, prep)
                    wait(math, input_ready, 0)
                work()
                if split:
                    cuda.record_event(result_ready, math)
                    wait(output, result_ready, 0)
                finish()
            if split:
                cuda.record_event(finished, output)
                wait(None, finished, 0)

        complete()
        if cuda.read(check, (1,), np.uint32)[0]:
            raise ValueError('Invalid inputs or integer encoding overflow')
        actual, quality = full_error(cuda, out, m, n, gold, norm)
        al = cuda.read(pa, (32, physical), np.int8).astype(np.int64)
        if direct:
            bl = cuda.read(pbt, (physical, n), np.int8)[:, :32].T.astype(np.int64)
        else:
            bl = cuda.read(pb, (32, physical), np.int8).astype(np.int64)
        sl, sr = [cuda.read(buf, (32,), np.float32) for buf in (sa, sb)]
        dot = al @ bl.T
        np.testing.assert_array_equal(dot, cuda.read(raw, (32, n), np.int32)[:, :32])
        np.testing.assert_array_equal(dot.astype(np.float32) * (sl[:, None] * sr), actual[:32, :32])
        for _ in range(args.warmup):
            complete()
        record = collect(cuda, shape, 'overlap_fp16_e2e', complete, args.samples, seed=seed,
                         chunks=len(partition), partition=partition, split=split, direct=direct,
                         algo=args.algorithm, physical_k=physical, triples=args.triples,
                         priority=args.priority, transpose=args.transpose, axis=args.axis, **quality)
        np.testing.assert_array_equal(actual, cuda.read(out, (m, n), np.float32))
        return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--chunks', type=int, nargs='+', default=(1, 2, 4, 8, 16))
    parser.add_argument('--rows',type=int,nargs='+',action='append',help='One unequal 4096-row partition. Can be repeated.')
    parser.add_argument('--axis',choices=('row','col'),default='row')
    parser.add_argument('--triples',type=int,choices=(0,768),default=0)
    parser.add_argument('--vector',type=int,choices=(0,256,512),default=0)
    parser.add_argument('--seeds', type=int, nargs='+', default=(2109,))
    parser.add_argument('--samples', type=int, default=9)
    parser.add_argument('--warmup', type=int, default=55)
    parser.add_argument('--algorithm', type=int, default=1)
    parser.add_argument('--serial', action='store_true')
    parser.add_argument('--direct', action='store_true')
    parser.add_argument('--priority',choices=('default','output','auxiliary'),default='default')
    parser.add_argument('--transpose',choices=TRANSPOSES,default='64')
    parser.add_argument('--output', type=Path, default=SOURCE_DIR / 'results' / 'overlap.json')
    args = parser.parse_args()
    if args.triples and (args.direct or not args.vector) or (args.vector and not args.triples):
        parser.error('Hybrid row packing requires a vectorized transpose-free source encoder')
    shape = (4096,) * 3
    p, lut = build((3, 4, 5), .002, 16)
    quantizer = lookup_tables(16,.002,.004) if args.triples else ((p, packed_table(lut)),127-int(np.abs(lut).max()))
    source = b'\n'.join((SOURCE_DIR / name).read_bytes() for name in
                       ('plane_sm61.cu','packed_layout.cu','direct_right.cu','normal_slide.cu'))
    ptx = compile_ptx(source)
    results = []
    with Cuda(args.gpu) as cuda, cuda.load(ptx) as module, Cublas() as blas:
        kernels = {name: cuda.kernel(module, name)[0] for name in
                   ('plane_bytes_registers','unpack_plane','unpack_plane_slice','vec_768_256','vec_768_512',
                    'projected_row_scales','pack_transposed_right',*(value[0] for value in TRANSPOSES.values()))}
        for seed in args.seeds:
            a, b = sample_operands(shape, seed=seed)
            gold = a.astype(np.float64) @ b.astype(np.float64).T
            operands = (a, b, gold, np.linalg.norm(gold))
            partitions=args.rows or [(4096//chunks,)*chunks for chunks in args.chunks]
            for partition in partitions:
                results.append(trial(cuda,blas,kernels,operands,quantizer,seed,partition,args))
                args.output.write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
