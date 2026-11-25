import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from bench import SOURCE_DIR, collect, parse_shape, upload_many
from cuda import Cublas, Cuda, INT, compile_ptx
from pipeline import pipeline
from rgs import Encoding, grouped_reference, sample_operands


ENCODER_KERNELS = ('block_maxima', 'estimate_differences', 'select_shifts', 'group_scales',
                   'encode_planes', 'encode_planes_aligned')


def gpu_encoder(cuda, kernels, buffers, shape, inputs, group, *, input_stride=None):
    m, n, k = shape
    stride = k if input_stride is None else input_stride
    a, b = inputs
    physical = (k + 31) // 32
    packed = (k + 3) // 4
    group_count = (k + group - 1) // group
    blocks_per_group = group // 32
    group_shift = blocks_per_group.bit_length() - 1 if blocks_per_group & (blocks_per_group - 1) == 0 else -1

    def memory(size):
        return buffers.enter_context(cuda.buffer(size))

    maxima_a, maxima_b = memory(2 * m * physical), memory(2 * n * physical)
    differences, shifts = memory(4 * physical), memory(4 * physical)
    alpha, beta = memory(4 * m * group_count), memory(4 * n * group_count)
    ah, ag = memory(4 * m * packed), memory(4 * m * packed)
    bh, bg = memory(4 * n * packed), memory(4 * n * packed)
    invalid = buffers.enter_context(cuda.buffer(np.zeros(1, dtype=np.uint32)))
    scratch = memory(4 * m * n), memory(4 * m * n)

    def launch(name, grid, args):
        return cuda.grid_invocation(kernels[name], grid, (256, 1, 1), args)

    steps = []
    for raw, maxima, rows in ((a, maxima_a, m), (b, maxima_b, n)):
        steps.append(launch('block_maxima', (physical, (rows + 7) // 8, 1),
                            (raw, maxima, invalid, INT(rows), INT(k), INT(stride))))
    steps.append(launch('estimate_differences', (physical, 1, 1),
                        (maxima_a, maxima_b, differences, INT(m), INT(n))))
    steps.append(launch('select_shifts', ((physical + 255) // 256, 1, 1),
                        (differences, shifts, INT(physical), INT(blocks_per_group))))
    for raw, maxima, scale, high, mixed, rows, side in (
            (a, maxima_a, alpha, ah, ag, m, 1), (b, maxima_b, beta, bh, bg, n, -1)):
        steps.append(launch('group_scales', ((rows + 255) // 256, group_count, 1),
                            (maxima, shifts, scale, INT(rows), INT(physical), INT(blocks_per_group), INT(side))))
        name = 'encode_planes_aligned' if k % 4 == stride % 4 == 0 else 'encode_planes'
        steps.append(launch(name, ((packed + 255) // 256, rows, 1),
                            (raw, shifts, scale, high, mixed, INT(rows), INT(k), INT(packed),
                             INT(blocks_per_group), INT(side), INT(group_shift), INT(stride))))

    def prepare():
        for step in steps:
            step()

    return prepare, (ah, ag, bh, bg, alpha, beta, *scratch), invalid, shifts


def verify_sample(cuda, output, shape, a, b, group, inputs):
    m, n, k = shape
    packed = (k + 3) // 4
    groups = (k + group - 1) // group
    ah, ag, bh, bg, alpha, beta, _, _ = inputs
    rows, cols = min(m, 32), min(n, 32)
    ha, ga = [cuda.read(value, (rows, packed), np.int32) for value in (ah, ag)]
    hb, gb = [cuda.read(value, (cols, packed), np.int32) for value in (bh, bg)]
    sa = cuda.read(alpha, (groups, m), np.float32)[:, :rows]
    sb = cuda.read(beta, (groups, n), np.float32)[:, :cols]
    candidate = Encoding(np.stack((ha, ga), axis=2), np.stack((hb, gb), axis=2), sa, sb, group)
    expected = grouped_reference(candidate, k)
    actual = cuda.read(output, (m, n), np.float32)[:rows, :cols]
    np.testing.assert_array_equal(actual, expected)
    reference = a[:rows].astype(np.float64) @ b[:cols].astype(np.float64).T
    return float(np.linalg.norm(expected - reference) / max(np.linalg.norm(reference), 1e-100))


def validate(cuda, kernels, blas):
    for shape, kind in (((67, 68, 157), 'reciprocal'), ((64, 68, 160), 'reciprocal'),
                        ((16, 12, 7), 'normal'), ((12, 8, 1), 'zero')):
        m, n, k = shape
        a, b = sample_operands(shape, kind)
        if kind == 'zero':
            a.fill(0)
            b.fill(0)
        else:
            a[0, -1] = np.float16(2 ** -24)
            b[0, -1] = np.finfo(np.float16).max
        for group in (32, 128):
            with ExitStack() as buffers:
                raw = upload_many(cuda, buffers, (a, b))
                out = buffers.enter_context(cuda.buffer(4 * m * n))
                prepare, encoded, invalid, shifts = gpu_encoder(cuda, kernels, buffers, shape, raw, group)
                prepare()
                pipeline(cuda, blas, kernels['rgs_finish'], shape, group, encoded, out, False)()
                assert cuda.read(invalid, (1,), np.uint32)[0] == 0
                error = verify_sample(cuda, out, shape, a, b, group, encoded)
                assert error < 0.02
                if kind == 'reciprocal' and group == 128:
                    assert cuda.read(shifts, ((k + 31) // 32,), np.int32).tolist()[:4] == [0, 9, 5, 11]
                print(f'GPU encode correctness {shape}, group {group}: sampled relative error {error:.7f}', flush=True)
    shape = (4, 4, 4)
    a, b = sample_operands(shape)
    a[0, 0], b[1, 1] = np.nan, np.inf
    with ExitStack() as buffers:
        raw = upload_many(cuda, buffers, (a, b))
        prepare, _, invalid, _ = gpu_encoder(cuda, kernels, buffers, shape, raw, 32)
        prepare()
        assert cuda.read(invalid, (1,), np.uint32)[0] == 1
    print('GPU encoder flagged non-finite input', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--shapes', nargs='+', default=('2048', '4096', '8192'))
    parser.add_argument('--groups', nargs='+', type=int, default=(512, 1024, 2048, 4096, 8192))
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--output', type=Path, default=SOURCE_DIR / 'results' / 'end-to-end.json')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    source = b'\n'.join((SOURCE_DIR / filename).read_bytes() for filename in ('rgs_sm61.cu', 'encode_sm61.cu'))
    ptx = compile_ptx(source)
    rows = []
    with Cuda(args.gpu) as cuda, cuda.load(ptx) as module, Cublas() as blas:
        print(f'Device {cuda.name}, cuBLAS {blas.version}', flush=True)
        kernels = {name: cuda.kernel(module, name)[0] for name in (*ENCODER_KERNELS, 'rgs_finish')}
        validate(cuda, kernels, blas)
        if args.validate_only:
            return
        for shape in map(parse_shape, args.shapes):
            m, n, k = shape
            a, b = sample_operands(shape)
            with ExitStack() as matrix_buffers:
                out = matrix_buffers.enter_context(cuda.buffer(4 * m * n))
                with ExitStack() as baseline_buffers:
                    fp32 = upload_many(cuda, baseline_buffers, (a.astype(np.float32), b.astype(np.float32)))
                    rows.append(collect(cuda, shape, 'cublas_sgemm', blas.invocation('sgemm', shape, *fp32, out), args.samples))
                raw = upload_many(cuda, matrix_buffers, (a, b))
                for group in args.groups:
                    if group > (k + 31) // 32 * 32:
                        continue
                    with ExitStack() as buffers:
                        prepare, encoded, invalid, _ = gpu_encoder(cuda, kernels, buffers, shape, raw, group)
                        run_gemm = pipeline(cuda, blas, kernels['rgs_finish'], shape, group, encoded, out, False)

                        def run():
                            prepare()
                            run_gemm()

                        run()
                        if cuda.read(invalid, (1,), np.uint32)[0]:
                            raise ValueError('RGS encoder received a NaN or infinity')
                        error = verify_sample(cuda, out, shape, a, b, group, encoded)
                        rows.append(collect(cuda, shape, 'rgs_blas_prepacked', run_gemm, args.samples,
                                            group=group, sample_error=error))
                        rows.append(collect(cuda, shape, 'rgs_fp16_device_end_to_end', run, args.samples,
                                            group=group, sample_error=error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
