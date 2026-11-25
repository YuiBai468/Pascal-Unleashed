import argparse
import csv
import json
import statistics
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter

import numpy as np

from cuda import Cublas, Cuda, INT, compile_ptx
from rgs import GROUPS, channels, encode, grouped_reference, sample_operands


SOURCE_DIR = Path(__file__).resolve().parent
TILES = {
    'ar2_gemm': (32, 32),
    'rgs32x32_k32': (32, 32),
    'rgs64x32_k32': (64, 32),
    'rgs32x64_k32': (32, 64),
    'rgs64x64_k32': (64, 64),
    'rgs64x64_k64': (64, 64),
    'int8_64x64_k32': (64, 64),
}
RGS_KERNELS = tuple(name for name in TILES if name.startswith('rgs'))


def upload_many(cuda, stack, arrays):
    return [stack.enter_context(cuda.buffer(array)) for array in arrays]


def grouped_invocation(cuda, function, name, shape, encoded, output, stack, single=False):
    m, n, k = shape
    args = upload_many(cuda, stack, (encoded.a, encoded.b, encoded.alpha, encoded.beta))
    packed_group = (k + 3) // 4 if single else encoded.group // 4
    return cuda.invocation(function, shape, TILES[name], (*args, output, INT(m), INT(n),
                                                         INT((k + 3) // 4), INT(packed_group)))


def original_invocation(cuda, function, shape, encoded, output, stack):
    m, n, k = shape
    planes = [np.ascontiguousarray(matrix[:, :, plane]) for matrix in (encoded.a, encoded.b) for plane in range(2)]
    bases = np.full(len(encoded.alpha), 16, dtype=np.int32)
    args = upload_many(cuda, stack, (*planes, encoded.alpha, encoded.beta, bases))
    return cuda.invocation(function, shape, TILES['ar2_gemm'], (*args, output, INT(m), INT(n), INT((k + 3) // 4)))


def validate(cuda, kernels, blas):
    shapes = ((67, 51, 159), (5, 7, 3), (35, 39, 545))
    checks = 0
    for shape, kind in zip(shapes, ('reciprocal', 'normal', 'common_spike')):
        m, n, k = shape
        a, b = sample_operands(shape, kind)
        if k > 32:
            a[0, :] = 0
            b[0, :] = 0
        with cuda.buffer(4 * m * n) as output:
            for group in GROUPS:
                encoded = encode(a, b, group)
                expected = grouped_reference(encoded, k)
                for name in RGS_KERNELS:
                    with ExitStack() as buffers:
                        operation = grouped_invocation(cuda, kernels[name], name, shape, encoded, output, buffers)
                        operation()
                        actual = cuda.read(output, (m, n), np.float32)
                    if not np.array_equal(actual, expected):
                        raise AssertionError((shape, group, name, float(np.max(np.abs(actual - expected)))))
                    checks += actual.size
                if group == 32:
                    with ExitStack() as buffers:
                        original_invocation(cuda, kernels['ar2_gemm'], shape, encoded, output, buffers)()
                        actual = cuda.read(output, (m, n), np.float32)
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
                    checks += actual.size
            with ExitStack() as buffers:
                name = 'int8_64x64_k32'
                grouped_invocation(cuda, kernels[name], name, shape, encoded, output, buffers, single=True)()
                actual = cuda.read(output, (m, n), np.int32)
            ah, _ = channels(encoded.a, k)
            bh, _ = channels(encoded.b, k)
            np.testing.assert_array_equal(actual, ah @ bh.T)
            checks += actual.size

    shape = (64, 68, 64)
    a, b = sample_operands(shape)
    expected = a.astype(np.float64) @ b.astype(np.float64).T
    with ExitStack() as buffers:
        out = buffers.enter_context(cuda.buffer(4 * shape[0] * shape[1]))
        for method in ('sgemm', 'fp16_fp32', 'fp16_fp16', 'int8'):
            dtype = {'sgemm': np.float32, 'fp16_fp32': np.float16, 'fp16_fp16': np.float16, 'int8': np.int8}[method]
            da, db = upload_many(cuda, buffers, (a.astype(dtype), b.astype(dtype)))
            try:
                blas.invocation(method, shape, da, db, out)()
            except RuntimeError as error:
                print(f'cuBLAS {method} unavailable: {error}', flush=True)
                continue
            output_type = {'fp16_fp16': np.float16, 'int8': np.int32}.get(method, np.float32)
            actual = cuda.read(out, shape[:2], output_type)
            if method == 'int8':
                np.testing.assert_array_equal(actual, a.astype(dtype).astype(np.int32) @ b.astype(dtype).astype(np.int32).T)
            elif method == 'fp16_fp16':
                assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 0.02
            else:
                np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-6)
    print(f'CUDA numerical validation: {checks:,} matching outputs across variants, groups and tail shapes', flush=True)


def collect(cuda, shape, method, operation, samples, **metadata):
    k = shape[2]
    repetitions = 10 if k <= 512 else 5 if k <= 2048 else 3
    times = cuda.measure(operation, repetitions, samples)
    median_ms = statistics.median(times)
    operations = 2 * shape[0] * shape[1] * shape[2]
    equivalent = operations / median_ms / 1e9
    row = dict(zip(('m', 'n', 'k'), shape)) | dict(method=method, median_ms=median_ms,
                min_ms=min(times), max_ms=max(times), effective_tflops=equivalent, samples_ms=times) | metadata
    if method.startswith(('rgs', 'ar2', 'tune')):
        row['int8_tops'] = equivalent * 2
    elif method.startswith('int8'):
        row['int8_tops'] = equivalent
    print(f'{shape} {method:25} {median_ms:8.4f} ms  {equivalent:6.2f} T/s  {metadata}', flush=True)
    return row


def tune(cuda, kernels, shape, samples):
    a, b = sample_operands(shape)
    m, n, _ = shape
    encoded = encode(a, b, 256)
    rows = []
    with cuda.buffer(4 * m * n) as out, ExitStack() as buffers:
        for name in RGS_KERNELS:
            operation = grouped_invocation(cuda, kernels[name], name, shape, encoded, out, buffers)
            rows.append(collect(cuda, shape, 'tune_' + name, operation, samples, group=256, kernel=name))
    compatible = [row for row in rows if row['kernel'].endswith('k32')]
    return min(compatible, key=lambda row: row['median_ms'])['kernel'], rows


def benchmark(cuda, blas, kernels, shape, selected, samples):
    a, b = sample_operands(shape)
    m, n, k = shape
    rows = []
    with cuda.buffer(4 * m * n) as out:
        for method in ('sgemm', 'fp16_fp32', 'fp16_fp16'):
            with ExitStack() as buffers:
                dtype = np.float32 if method == 'sgemm' else np.float16
                da, db = upload_many(cuda, buffers, (a.astype(dtype, copy=False), b.astype(dtype, copy=False)))
                operation = blas.invocation(method, shape, da, db, out)
                try:
                    rows.append(collect(cuda, shape, 'cublas_' + method, operation, samples))
                except RuntimeError as error:
                    print(f'{method} unavailable for {shape}: {error}', flush=True)
        for group in GROUPS:
            start = perf_counter()
            encoded = encode(a, b, group)
            host_pack_ms = 1000 * (perf_counter() - start)
            with ExitStack() as buffers:
                operation = grouped_invocation(cuda, kernels[selected], selected, shape, encoded, out, buffers)
                rows.append(collect(cuda, shape, 'rgs_gemm', operation, samples, group=group,
                                    kernel=selected, cpu_encode_ms=host_pack_ms))
            if group != 32:
                continue
            with ExitStack() as buffers:
                operation = original_invocation(cuda, kernels['ar2_gemm'], shape, encoded, out, buffers)
                rows.append(collect(cuda, shape, 'ar2_original', operation, samples, group=group, kernel='ar2_gemm'))
            with ExitStack() as buffers:
                name = 'int8_64x64_k32'
                operation = grouped_invocation(cuda, kernels[name], name, shape, encoded, out, buffers, single=True)
                rows.append(collect(cuda, shape, 'int8_custom', operation, samples, kernel=name))
            if k % 4 == 0:
                with ExitStack() as buffers:
                    a8 = np.ascontiguousarray(encoded.a[:, :, 0]).view(np.int8)
                    b8 = np.ascontiguousarray(encoded.b[:, :, 0]).view(np.int8)
                    da, db = upload_many(cuda, buffers, (a8, b8))
                    operation = blas.invocation('int8', shape, da, db, out)
                    try:
                        rows.append(collect(cuda, shape, 'int8_cublas', operation, samples))
                    except RuntimeError as error:
                        print(f'cuBLAS INT8 unavailable for {shape}: {error}', flush=True)
    return rows


def accuracy():
    shape = (128, 128, 512)
    rows = []
    for kind in ('normal', 'reciprocal', 'common_spike'):
        runs = []
        for seed in (42, 43, 44, 45):
            a, b = sample_operands(shape, kind, seed=seed)
            ref = a.astype(np.float64) @ b.astype(np.float64).T
            run = {}
            for group in GROUPS:
                for reciprocal in (False, True):
                    c = grouped_reference(encode(a, b, group, reciprocal), shape[2])
                    key = (group, reciprocal)
                    error = c - ref
                    run[key] = (float(np.linalg.norm(error) / np.linalg.norm(ref)), float(np.max(np.abs(error))))
            runs.append(run)
        for group, reciprocal in runs[0]:
            rel, abs_error = np.median([run[group, reciprocal] for run in runs], axis=0)
            row = dict(distribution=kind, group=group, reciprocal=reciprocal,
                       relative_frobenius=float(rel), max_absolute=float(abs_error))
            rows.append(row)
            print(f'accuracy {kind:12} group={group:3} RGS={int(reciprocal)} rel={100*rel:.5f}%', flush=True)
    return rows


def parse_shape(text):
    values = tuple(int(x) for x in text.lower().split('x'))
    if len(values) not in (1, 3) or min(values) < 1:
        raise ValueError('Use a square size or MxNxK')
    return values * 3 if len(values) == 1 else values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', required=True, help='UUID from nvidia-smi -L')
    parser.add_argument('--shapes', nargs='*', default=('512', '1024', '2048', '4096'))
    parser.add_argument('--tune-size', type=int, default=2048)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--output', type=Path, default=SOURCE_DIR / 'results')
    args = parser.parse_args()
    source = b'\n'.join((SOURCE_DIR / name).read_bytes() for name in ('ar2_sm61.cu', 'rgs_sm61.cu'))
    ptx = compile_ptx(source)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'pascal_unleashed.ptx').write_bytes(ptx.value)
    with Cuda(args.gpu) as cuda, cuda.load(ptx) as module, Cublas() as blas:
        print(f'Device {cuda.name}, cuBLAS {blas.version}', flush=True)
        metadata = {name: cuda.kernel(module, name) for name in TILES}
        kernels = {name: item[0] for name, item in metadata.items()}
        info = {name: item[1] for name, item in metadata.items()}
        print('JIT resources:', json.dumps(info), flush=True)
        validate(cuda, kernels, blas)
        if args.validate_only:
            return
        selected, records = tune(cuda, kernels, (args.tune_size,) * 3, args.samples)
        print('Selected 32-K tile:', selected, flush=True)
        for shape in map(parse_shape, args.shapes):
            records.extend(benchmark(cuda, blas, kernels, shape, selected, args.samples))
        report = dict(device=cuda.name, cublas_version=blas.version, selected_kernel=selected,
                      kernel_resources=info, performance=records, accuracy=accuracy())
    (args.output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (args.output / 'performance.csv').open('w', newline='', encoding='utf-8') as file:
        fields = sorted(set().union(*(r.keys() for r in records)))
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


if __name__ == '__main__':
    main()
