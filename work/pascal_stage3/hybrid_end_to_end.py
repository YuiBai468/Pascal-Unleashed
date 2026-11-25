import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from bench import SOURCE_DIR, collect, upload_many
from cuda import Cublas, Cuda, INT, compile_ptx
from end_to_end import ENCODER_KERNELS, gpu_encoder
from rgs import sample_operands
from try_20t import errors, sample_expected


def evaluate(cuda, blas, kernels, a, b, high_k, samples, kind):
    m, n, k = a.shape[0], b.shape[0], a.shape[1]
    low_k = k - high_k
    if k % 4 or high_k % 4 or not 0 < low_k <= 4096:
        raise ValueError('Segments must be multiples of four and one-round K must fit the shared tile')
    group = (high_k + 31) // 32 * 32
    with ExitStack() as buffers:
        def buffer(size):
            return buffers.enter_context(cuda.buffer(size))

        raw = upload_many(cuda, buffers, (a, b))
        high_prepare, fields, invalid, _ = gpu_encoder(cuda, kernels, buffers, (m, n, high_k), raw,
                                                      group, input_stride=k)
        ah, ag, bh, bg, alpha, beta, first, second = fields
        a8, b8, sa, sb = buffer(m*low_k), buffer(n*low_k), buffer(4*m), buffer(4*n)
        legacy = kernels['pack_one_round']
        packing = [cuda.grid_invocation(legacy, (rows, 1, 1), (256, 1, 1),
                                       (source, encoded, scale, invalid, INT(k//4), INT(high_k//4), INT(low_k//4)))
                   for source, encoded, scale, rows in ((raw[0], a8, sa, m), (raw[1], b8, sb, n))]
        low, output = buffer(4*m*n), buffer(4*m*n)
        shape_high, shape_low = (m, n, high_k), (m, n, low_k)
        ops = (blas.invocation('int8', shape_high, ah, bh, first),
               blas.invocation('int8', shape_high, ag, bg, second),
               blas.invocation('int8', shape_low, a8, b8, low),
               cuda.invocation(kernels['finish_20t'], (m, n, k), (8, 32),
                               (first, second, low, alpha, beta, sa, sb, output, INT(m), INT(n), INT(2)),
                               block=(32, 8, 1)))

        def gemm():
            for step in ops:
                step()

        def complete():
            high_prepare()
            for step in packing:
                step()
            gemm()

        complete()
        if cuda.read(invalid, (1,), np.uint32)[0]:
            raise ValueError('Hybrid encoder received NaN or infinity')
        sample = min(48, m, n)
        high = [cuda.read(value, (sample, high_k), np.int8) for value in (ah, ag, bh, bg)]
        scales_high = 16*cuda.read(alpha, (sample,), np.float32), cuda.read(beta, (sample,), np.float32)
        low_planes = [cuda.read(value, (sample, low_k), np.int8) for value in (a8, b8)]
        scales_low = cuda.read(sa, (sample,), np.float32), cuda.read(sb, (sample,), np.float32)
        expected = sample_expected(high, scales_high, low_planes, scales_low, sample)
        actual = cuda.read(output, (m, n), np.float32)[:sample, :sample]
        np.testing.assert_array_equal(actual, expected)
        if kind == 'exact_int8_suffix':
            for plane, source, scale in zip(low_planes, (a, b), scales_low):
                np.testing.assert_array_equal(plane.astype(np.float32)*scale[:, None], source[:sample, high_k:])
        reference = a[:sample].astype(np.float64) @ b[:sample].astype(np.float64).T
        stats = dict(high_k=high_k, dataset=kind, sample_error=errors(expected, reference))
        return [collect(cuda, (m, n, k), operation, function, samples, **stats)
                for operation, function in (('hybrid_prepacked', gemm), ('hybrid_fp16_end_to_end', complete))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--size', type=int, default=4096)
    parser.add_argument('--high-k', nargs='+', type=int, default=(512, 768, 1024, 2048))
    parser.add_argument('--samples', type=int, default=7)
    parser.add_argument('--near-grid-noise', type=float, default=0.0)
    parser.add_argument('--output', type=Path, default=SOURCE_DIR / 'results' / 'hybrid-end-to-end.json')
    args = parser.parse_args()
    source = b'\n'.join((SOURCE_DIR / file).read_bytes() for file in ('encode_sm61.cu', 'try_20t.cu'))
    ptx = compile_ptx(source)
    results = []
    with Cuda(args.gpu) as cuda, cuda.load(ptx) as module, Cublas() as blas:
        names = (*ENCODER_KERNELS, 'pack_one_round', 'finish_20t')
        kernels = {name: cuda.kernel(module, name)[0] for name in names}
        for high_k in args.high_k:
            if not 0 < high_k < args.size:
                raise ValueError('high K must be inside K')
            a, b = sample_operands((args.size,)*3, seed=2109)
            results.extend(evaluate(cuda, blas, kernels, a, b, high_k, args.samples, 'normal'))
            rng = np.random.default_rng(2110)
            for matrix in (a, b):
                suffix = matrix[:, high_k:]
                suffix[:] = rng.integers(-127, 128, size=suffix.shape, dtype=np.int16).astype(np.float16) / 64
                suffix[:, 0] = np.float16(127 / 64)
            results.extend(evaluate(cuda, blas, kernels, a, b, high_k, args.samples, 'exact_int8_suffix'))
            if args.near_grid_noise > 0:
                for matrix in (a, b):
                    suffix = matrix[:, high_k+1:]
                    jitter = rng.standard_normal(suffix.shape, dtype=np.float32) * args.near_grid_noise
                    suffix[:] = np.clip(suffix, -126/64, 126/64) + jitter
                results.extend(evaluate(cuda, blas, kernels, a, b, high_k, args.samples, 'near_int8_suffix'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
