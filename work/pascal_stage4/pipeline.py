import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter

import numpy as np

from bench import SOURCE_DIR, collect, grouped_invocation, parse_shape, upload_many
from cuda import Cublas, Cuda, INT, compile_ptx
from rgs import encode, grouped_reference, sample_operands


def prepare(cuda, buffers, encoded, m, n, combined):
    planes = [np.ascontiguousarray(matrix[:, :, index]).view(np.int8)
              for matrix in (encoded.a, encoded.b) for index in range(2)]
    operands = upload_many(cuda, buffers, (*planes, encoded.alpha, encoded.beta))
    high = buffers.enter_context(cuda.buffer(4 * m * n))
    mixed = high if combined else buffers.enter_context(cuda.buffer(4 * m * n))
    return (*operands, high, mixed)


def pipeline(cuda, blas, finish, shape, group, inputs, output, combined, *, algorithm=-1):
    m, n, k = shape
    ah, ag, bh, bg, alpha, beta, high, mixed = inputs
    padded_k = (k + 3) // 4 * 4
    steps = []
    for index, offset in enumerate(range(0, padded_k, group)):
        partial = (m, n, min(group, padded_k - offset))
        steps.append(blas.invocation('int8', partial, ah, bh, high, lda=padded_k, offset=offset,
                                     algorithm=algorithm))
        steps.append(blas.invocation('int8', partial, ag, bg, mixed, lda=padded_k, offset=offset,
                                     int_beta=15 if combined else 0, algorithm=algorithm))
        parameters = (high, mixed, alpha, beta, output, INT(m), INT(n), INT(index), INT(combined))
        steps.append(cuda.invocation(finish, partial, (8, 32), parameters, block=(32, 8, 1)))

    def run():
        for step in steps:
            step()

    return run


def validates(cuda, blas, finish, combined):
    shape = (64, 68, 155)
    m, n, k = shape
    a, b = sample_operands(shape, 'reciprocal')
    for group in (32, 64, 256):
        encoded = encode(a, b, group)
        expected = grouped_reference(encoded, k)
        with ExitStack() as buffers:
            out = buffers.enter_context(cuda.buffer(4 * m * n))
            inputs = prepare(cuda, buffers, encoded, m, n, combined)
            pipeline(cuda, blas, finish, shape, group, inputs, out, combined)()
            actual = cuda.read(out, (m, n), np.float32)
            np.testing.assert_array_equal(actual, expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--shapes', nargs='+', default=('2048', '4096'))
    parser.add_argument('--groups', nargs='+', type=int, default=(512, 1024, 2048, 4096))
    parser.add_argument('--samples', type=int, default=7)
    parser.add_argument('--output', type=Path, default=SOURCE_DIR / 'results' / 'pipeline.json')
    args = parser.parse_args()
    ptx = compile_ptx((SOURCE_DIR / 'rgs_sm61.cu').read_bytes())
    rows = []
    with Cuda(args.gpu) as cuda, cuda.load(ptx) as module, Cublas() as blas:
        print(f'Device {cuda.name}, cuBLAS {blas.version}', flush=True)
        finish, _ = cuda.kernel(module, 'rgs_finish')
        custom, _ = cuda.kernel(module, 'rgs64x64_k64')
        modes = []
        for combined in (False, True):
            try:
                validates(cuda, blas, finish, combined)
                modes.append(combined)
                print(f'cuBLAS pipeline verified with combined={combined}, including non-four-multiple K', flush=True)
            except (RuntimeError, AssertionError) as error:
                print(f'cuBLAS pipeline combined={combined} unavailable: {error}', flush=True)

        for shape in map(parse_shape, args.shapes):
            m, n, k = shape
            a, b = sample_operands(shape)
            a_sample, b_sample = a[:64].astype(np.float64), b[:64].astype(np.float64)
            exact = a_sample @ b_sample.T
            for group in args.groups:
                if group > (k + 31) // 32 * 32:
                    continue
                start = perf_counter()
                encoded = encode(a, b, group)
                cpu_encode_ms = 1000 * (perf_counter() - start)
                sub_a = np.ascontiguousarray(encoded.a[:64])
                sub_b = np.ascontiguousarray(encoded.b[:64])
                sub_alpha = np.ascontiguousarray(encoded.alpha[:, :64])
                sub_beta = np.ascontiguousarray(encoded.beta[:, :64])
                expected = grouped_reference(type(encoded)(sub_a, sub_b, sub_alpha, sub_beta, group), k)
                error = float(np.linalg.norm(expected - exact) / np.linalg.norm(exact))
                with cuda.buffer(4 * m * n) as out:
                    with ExitStack() as buffers:
                        label = 'rgs64x64_k64'
                        operation = grouped_invocation(cuda, custom, label, shape, encoded, out, buffers)
                        operation()
                        np.testing.assert_array_equal(cuda.read(out, (m, n), np.float32)[:64, :64], expected)
                        rows.append(collect(cuda, shape, 'rgs_fused_k64', operation, args.samples,
                                            group=group, cpu_encode_ms=cpu_encode_ms, sample_error=error))
                    for combined in modes:
                        with ExitStack() as buffers:
                            operands = prepare(cuda, buffers, encoded, m, n, combined)
                            operation = pipeline(cuda, blas, finish, shape, group, operands, out, combined)
                            try:
                                operation()
                                np.testing.assert_array_equal(cuda.read(out, (m, n), np.float32)[:64, :64], expected)
                                label = 'rgs_blas_inplace' if combined else 'rgs_blas_two_buffers'
                                rows.append(collect(cuda, shape, label, operation, args.samples,
                                                    group=group, cpu_encode_ms=cpu_encode_ms, sample_error=error))
                            except RuntimeError as failure:
                                print(f'pipeline unavailable, combined={combined}, shape={shape}, group={group}: {failure}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
