import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from bench import SOURCE_DIR, collect, upload_many
from cuda import Cublas, Cuda, INT, compile_ptx
from end_to_end import ENCODER_KERNELS, gpu_encoder, verify_sample
from pipeline import pipeline
from rgs import LIMIT, block_view, encode_channels, median_exponent, sample_operands


def prepare_weight(weights):
    blocks = block_view(weights)
    maxima = np.max(np.abs(blocks), axis=2) / LIMIT
    logs = median_exponent(maxima)
    shift = -np.rint(logs - logs[0]).astype(np.int32)
    reciprocal = np.exp2(shift).astype(np.float32)
    beta = np.max(maxima * reciprocal, axis=1)
    beta[beta == 0] = 1
    encoded = encode_channels(blocks, beta[:, None] / reciprocal, weights.shape[1])
    return (np.ascontiguousarray(encoded[:, :, 0]), np.ascontiguousarray(encoded[:, :, 1]), beta, shift)


def cached_benchmark(cuda, blas, kernels, size, kind, samples, algorithm, warm_launches):
    shape = (size,)*3
    _, b = sample_operands(shape, kind, seed=42)
    array_bh, array_bg, beta, shift = prepare_weight(b)
    results = []
    print(f'{kind}: cached group {size}, shifts {int(shift.min())} to {int(shift.max())}', flush=True)
    with ExitStack() as matrix_buffers:
        bh, bg, sb, shifts = upload_many(cuda, matrix_buffers, (array_bh, array_bg, beta, shift))
        output = matrix_buffers.enter_context(cuda.buffer(4*size**2))
        for seed in (43, 44):
            a, _ = sample_operands(shape, kind, seed=seed)
            with ExitStack() as buffers:
                def memory(size_or_array):
                    return buffers.enter_context(cuda.buffer(size_or_array))

                da, db = upload_many(cuda, buffers, (a, b))
                ah, ag = memory(size**2), memory(size**2)
                sa, invalid = memory(4*size), memory(np.zeros(1, dtype=np.uint32))
                first, second = memory(4*size**2), memory(4*size**2)
                prep = cuda.grid_invocation(kernels['pack_activation'], (size, 1, 1), (256, 1, 1),
                                            (da, shifts, ah, ag, sa, invalid, INT(size//4)))
                operands = ah, ag, bh, bg, sa, sb, first, second
                gemm = pipeline(cuda, blas, kernels['rgs_finish'], shape, size, operands, output, False,
                                algorithm=algorithm)

                def cached():
                    prep()
                    gemm()

                cached()
                if seed == 43:
                    for _ in range(warm_launches):
                        cached()
                    cuda.sync()
                if cuda.read(invalid, (1,), np.uint32)[0]:
                    raise ValueError('Cached-weight path received non-finite activation data')
                error = verify_sample(cuda, output, shape, a, b, size, operands)
                results.append(collect(cuda, shape, 'rgs_cached_weight', cached, samples, seed=seed,
                                       data=kind, sample_error=error, algorithm=algorithm))
                results.append(collect(cuda, shape, 'rgs_both_prepacked', gemm, samples, seed=seed,
                                       data=kind, sample_error=error, algorithm=algorithm))
                if seed == 43:
                    prepare, dyn, flag, _ = gpu_encoder(cuda, kernels, buffers, shape, (da, db), size)
                    dyn_gemm = pipeline(cuda, blas, kernels['rgs_finish'], shape, size, dyn, output, False,
                                        algorithm=algorithm)

                    def dynamic():
                        prepare()
                        dyn_gemm()

                    dynamic()
                    if cuda.read(flag, (1,), np.uint32)[0]:
                        raise ValueError('Dynamic path received non-finite data')
                    error = verify_sample(cuda, output, shape, a, b, size, dyn)
                    results.append(collect(cuda, shape, 'rgs_both_dynamic', dynamic, samples, seed=seed,
                                           data=kind, sample_error=error, algorithm=algorithm))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--size', type=int, default=4096)
    parser.add_argument('--samples', type=int, default=9)
    parser.add_argument('--algorithm', type=int, default=1)
    parser.add_argument('--warm-launches', type=int, default=24)
    parser.add_argument('--output', type=Path, default=SOURCE_DIR / 'results' / 'cache-weight.json')
    args = parser.parse_args()
    if not 32 <= args.size <= 4096 or args.size % 32:
        raise ValueError('This one-group prototype requires a square multiple of 32 up to 4096')
    source = b'\n'.join((SOURCE_DIR / name).read_bytes()
                        for name in ('rgs_sm61.cu', 'encode_sm61.cu', 'cache_weight.cu'))
    ptx = compile_ptx(source)
    with Cuda(args.gpu) as cuda, cuda.load(ptx) as module, Cublas() as blas:
        print(f'Device {cuda.name}, cuBLAS {blas.version}', flush=True)
        kernels = {name: cuda.kernel(module, name)[0] for name in (*ENCODER_KERNELS, 'rgs_finish', 'pack_activation')}
        records = []
        for kind in ('normal', 'reciprocal'):
            records.extend(cached_benchmark(cuda, blas, kernels, args.size, kind, args.samples,
                                            args.algorithm, args.warm_launches))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
