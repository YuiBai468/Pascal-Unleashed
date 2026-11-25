import argparse
import json
import math
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from bench import SOURCE_DIR, collect, parse_shape, upload_many
from cuda import Cublas, Cuda, DEV_PTR, INT, compile_ptx
from rgs import encode, sample_operands


def row_scale(values, limit):
    ceiling = np.max(np.abs(values.astype(np.float32)), axis=1, keepdims=True)
    scale = ceiling / limit
    scale[scale == 0] = 1
    return scale


def integer_lattice(u, v):
    d = u*u + v*v
    z = np.arange(-128*(u+v), 128*(u+v)+1, dtype=np.int32)
    congruence = (u * pow(v, -1, d)) % d
    r = (congruence*z + d//2) % d - d//2
    x, y = (u*z - v*r) // d, (v*z + u*r) // d
    assert np.array_equal(u*x + v*y, z) and np.array_equal(-v*x + u*y, r)
    bad = (np.abs(x) > 127) | (np.abs(y) > 127)
    cap = int(np.min(np.abs(z[bad])) - 1)
    valid = np.abs(z) <= cap
    return d, cap, x[valid].astype(np.int8), y[valid].astype(np.int8)


def lattice_encode(values, basis):
    _, cap, x, y = basis
    scale = row_scale(values, cap)
    q = np.rint(values.astype(np.float32) / scale).astype(np.int32) + cap
    return np.concatenate((x[q], y[q]), axis=1), scale[:, 0]


def errors(approximation, reference):
    difference = np.asarray(approximation, dtype=np.float64) - reference
    return float(np.linalg.norm(difference) / max(np.linalg.norm(reference), 1e-100))


def tune_basis():
    sets = []
    for seed in (5613, 9010, 8221):
        a, b = sample_operands((48, 48, 512), seed=seed)
        sets.append((a, b, a.astype(np.float64) @ b.astype(np.float64).T))
    candidates = []
    for u in range(2, 12):
        for v in range(1, u):
            if math.gcd(u, v) != 1 or (u+v) % 2 == 0:
                continue
            basis = integer_lattice(u, v)
            measures = []
            for a, b, reference in sets:
                oa, sa = lattice_encode(a, basis)
                ob, sb = lattice_encode(b, basis)
                approximation = (oa.astype(np.int64) @ ob.astype(np.int64).T) * ((basis[0]*sa)[:, None] * sb)
                measures.append(errors(approximation, reference))
            candidates.append((float(np.median(measures)), u, v, basis))
    error, u, v, basis = min(candidates, key=lambda result: result[0])
    print(f'CPU basis selection: ({u}, {v}), cap {basis[1]}, median relative error {error:.7f}', flush=True)
    return u, v, basis


def int8_encode(values):
    floats = values.astype(np.float32)
    scale = row_scale(values, 127)
    dyadic = np.exp2(np.ceil(np.log2(scale)))
    q = np.rint(floats/dyadic)
    exact = np.all(q*dyadic == floats, axis=1, keepdims=True)
    chosen = np.where(exact, dyadic, scale)
    data = np.rint(floats/chosen).astype(np.int8)
    return data, chosen[:, 0], float(np.mean(exact))


def ar2_encode(a, b):
    k = a.shape[1]
    encoded = encode(a, b, (k + 31)//32*32)
    planes = [np.ascontiguousarray(mat[:, :, lane]).view(np.int8)[:, :k].copy() for mat in (encoded.a, encoded.b)
              for lane in range(2)]
    return (*planes, 16 * encoded.alpha[0], encoded.beta[0])


def dot_int(a, b, rows):
    return a[:rows].astype(np.int64) @ b[:rows].astype(np.int64).T


def sample_expected(terms, scales, low_terms, low_scales, rows):
    if len(terms) == 2:
        z = dot_int(*terms, rows)
    else:
        ah, ag, bh, bg = terms
        z = 15*dot_int(ah, bh, rows) + dot_int(ag, bg, rows)
    sa, sb = scales
    unit = sa[:rows, None] * sb[None, :rows]
    prior = 0.0
    if low_terms:
        la, lb = low_scales
        prior = dot_int(*low_terms, rows).astype(np.float32) * (la[:rows, None] * lb[None, :rows])
    return (z.astype(np.float32).astype(np.float64)*unit + prior).astype(np.float32)


def benchmark(cuda, blas, finish, shape, label, variants, a, b, samples, *, algorithm=-1, **metadata):
    m, n, k = shape
    terms, scales, low_terms, low_scales = variants
    sample = min(m, n, 48)
    expected = sample_expected(*variants, sample)
    reference = a[:sample].astype(np.float64) @ b[:sample].astype(np.float64).T
    with ExitStack() as buffers:
        matrices = upload_many(cuda, buffers, terms)
        alpha, beta = upload_many(cuda, buffers, scales)
        first, output = [buffers.enter_context(cuda.buffer(4*m*n)) for _ in range(2)]
        ar2 = len(terms) == 4
        second = buffers.enter_context(cuda.buffer(4*m*n)) if ar2 else DEV_PTR(0)
        low, low_alpha, low_beta = (DEV_PTR(0),)*3
        main_shape = (m, n, terms[0].shape[1])
        steps = []
        if ar2:
            ah, ag, bh, bg = matrices
            steps.append(blas.invocation('int8', main_shape, ah, bh, first, algorithm=algorithm))
            steps.append(blas.invocation('int8', main_shape, ag, bg, second, algorithm=algorithm))
        else:
            steps.append(blas.invocation('int8', main_shape, *matrices, first, algorithm=algorithm))
        if low_terms:
            a8, b8 = upload_many(cuda, buffers, low_terms)
            low_alpha, low_beta = upload_many(cuda, buffers, low_scales)
            low = buffers.enter_context(cuda.buffer(4*m*n))
            steps.append(blas.invocation('int8', (m, n, low_terms[0].shape[1]), a8, b8, low))
        args = (first, second, low, alpha, beta, low_alpha, low_beta, output, INT(m), INT(n), INT(ar2))
        steps.append(cuda.invocation(finish, shape, (8, 32), args, block=(32, 8, 1)))

        def run():
            for step in steps:
                step()

        run()
        np.testing.assert_array_equal(cuda.read(output, shape[:2], np.float32)[:sample, :sample], expected)
        result = collect(cuda, shape, label, run, samples, sample_error=errors(expected, reference),
                         algorithm=algorithm, **metadata)
        work_ratio = (main_shape[2] * (2 if ar2 else 1) + (low_terms[0].shape[1] if low_terms else 0)) / k
        result['int8_tops'] = result['effective_tflops'] * work_ratio
        result['int8_work_ratio'] = work_ratio
        result['step_median_ms'] = [float(np.median(cuda.measure(step, 3, 3))) for step in steps]
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--shapes', nargs='+', default=('4096',))
    parser.add_argument('--samples', type=int, default=7)
    parser.add_argument('--high-k', type=int, nargs='+')
    parser.add_argument('--packed-only', action='store_true')
    parser.add_argument('--packed-algorithms', type=int, nargs='+', default=(-1,))
    parser.add_argument('--output', type=Path, default=SOURCE_DIR / 'results' / 'try-20t.json')
    args = parser.parse_args()
    u, v, basis = tune_basis()
    ptx = compile_ptx((SOURCE_DIR / 'try_20t.cu').read_bytes())
    results = []
    with Cuda(args.gpu) as cuda, cuda.load(ptx) as module, Cublas() as blas:
        finish, _ = cuda.kernel(module, 'finish_20t')
        for shape in map(parse_shape, args.shapes):
            m, n, k = shape
            if n % 4 or k % 16 or k > 8192:
                raise ValueError('Choose N divisible by 4 and K divisible by 16 up to 8192')
            a, b = sample_operands(shape, seed=2109)
            if not args.packed_only:
                *terms, aa, bb = ar2_encode(a, b)
                results.append(benchmark(cuda, blas, finish, shape, 'rgs_two_cublas', (terms, (aa, bb), (), ()),
                                         a, b, args.samples))
            qa, sa = lattice_encode(a, basis)
            qb, sb = lattice_encode(b, basis)
            for algorithm in args.packed_algorithms:
                results.append(benchmark(cuda, blas, finish, shape, 'packed_two_channels',
                                         ((qa, qb), (basis[0]*sa, sb), (), ()), a, b, args.samples,
                                         basis=[u, v], algorithm=algorithm))
            if args.packed_only:
                continue
            for high_k in args.high_k or (k//4, k//2, 3*k//4):
                if high_k % 4 or not 0 < high_k < k:
                    raise ValueError('Choose high K divisible by four and inside the original K extent')
                *terms, aa, bb = ar2_encode(a[:, :high_k], b[:, :high_k])
                la8, la, a_exact = int8_encode(a[:, high_k:])
                lb8, lb, b_exact = int8_encode(b[:, high_k:])
                variants = (terms, (aa, bb), (la8, lb8), (la, lb))
                results.append(benchmark(cuda, blas, finish, shape, 'hybrid_normal', variants, a, b, args.samples,
                                         high_fraction=high_k/k, exact_rows_a=a_exact, exact_rows_b=b_exact))

            high_k = k // 4
            rng = np.random.default_rng(2110)
            a[:, high_k:] = rng.integers(-127, 128, size=(m, k-high_k), dtype=np.int16).astype(np.float16) / 64
            b[:, high_k:] = rng.integers(-127, 128, size=(n, k-high_k), dtype=np.int16).astype(np.float16) / 64
            *terms, aa, bb = ar2_encode(a[:, :high_k], b[:, :high_k])
            la8, la, a_exact = int8_encode(a[:, high_k:])
            lb8, lb, b_exact = int8_encode(b[:, high_k:])
            assert a_exact == b_exact == 1.0
            results.append(benchmark(cuda, blas, finish, shape, 'hybrid_75pct_exact_int8',
                                     (terms, (aa, bb), (la8, lb8), (la, lb)), a, b, args.samples,
                                     high_fraction=0.25, exact_rows_a=a_exact, exact_rows_b=b_exact))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
