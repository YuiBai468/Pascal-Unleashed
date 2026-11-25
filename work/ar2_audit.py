import numpy as np


BASES = (8, 12, 16, 24, 32)


def split(values, base):
    high = (values + base // 2) // base
    return high, values - base * high


def safe_signed_limit(base):
    values = np.arange(-128 * base, 128 * base + 1)
    high, residue = split(values, base)
    sums = high + residue
    outside = (high < -128) | (high > 127) | (sums < -128) | (sums > 127)
    return int(np.min(np.abs(values[outside])) - 1)


LIMITS = {base: safe_signed_limit(base) for base in BASES}


def encode(values, base, axis):
    scale = np.max(np.abs(values), axis=axis, keepdims=True) / LIMITS[base]
    scale[scale == 0] = 1
    integer = np.rint(values / scale).astype(np.int64)
    high, residue = split(integer, base)
    for code in (high, high + residue):
        assert code.min() >= -128 and code.max() <= 127
    scaled_integer = integer * scale
    return high, residue, scale, scaled_integer, values - scaled_integer


def select_encoding(a, b, bases):
    candidates = []
    for base in bases:
        ea = encode(a, base, axis=1)
        eb = encode(b, base, axis=0)
        _, ra, sa, aq, da = ea
        _, rb, sb, _, db = eb
        norm = np.linalg.norm
        bound = (
            (base - 1) * norm(ra * sa) * norm(rb * sb)
            + norm(da) * norm(b)
            + norm(aq) * norm(db)
        )
        candidates.append((bound, base, ea, eb))
    return min(candidates, key=lambda candidate: candidate[0])[1:]


def block_products(a, b, bases):
    base, (ha, ra, sa, aq, da), (hb, rb, sb, _, db) = select_encoding(a, b, bases)
    first = ha @ hb
    second = (ha + ra) @ (hb + rb)
    scales = sa * sb
    approx_int = (base * base - base) * first + base * second
    ai, bi = base * ha + ra, base * hb + rb
    assert np.array_equal(approx_int - ai @ bi, (base - 1) * (ra @ rb))
    raw = approx_int * scales
    k = a.shape[1]

    mean_a, mean_b = ra.mean(axis=1, keepdims=True), rb.mean(axis=0, keepdims=True)
    mean_product = k * mean_a * mean_b
    mean_corrected = raw - (base - 1) * scales * mean_product

    sign_a, sign_b = np.where(ra >= 0, 1, -1), np.where(rb >= 0, 1, -1)
    sca = sign_a - sign_a.mean(axis=1, keepdims=True)
    scb = sign_b - sign_b.mean(axis=0, keepdims=True)
    va = np.sum(sca * sca, axis=1, keepdims=True)
    vb = np.sum(scb * scb, axis=0, keepdims=True)
    ca = np.sum(ra * sca, axis=1, keepdims=True)
    cb = np.sum(rb * scb, axis=0, keepdims=True)
    alpha = np.divide(ca, va, out=np.zeros_like(ca), where=va != 0)
    beta = np.divide(cb, vb, out=np.zeros_like(cb), where=vb != 0)
    centered_sign_product = sign_a @ sign_b - k * (
        sign_a.mean(axis=1, keepdims=True) * sign_b.mean(axis=0, keepdims=True)
    )
    two_level_product = mean_product + alpha * beta * centered_sign_product
    bit_corrected = raw - (base - 1) * scales * two_level_product

    def outer_norm(x, y):
        return np.linalg.norm(x, axis=1)[:, None] * np.linalg.norm(y, axis=0)[None, :]

    radius = (base - 1) * scales * outer_norm(ra, rb) + outer_norm(da, b) + outer_norm(aq, db)
    centered_radius = (base - 1) * scales * outer_norm(ra - mean_a, rb - mean_b)
    centered_radius += outer_norm(da, b) + outer_norm(aq, db)
    return raw, mean_corrected, bit_corrected, radius, centered_radius, base


def hadamard(size):
    transform = np.ones((1, 1))
    while transform.shape[0] < size:
        transform = np.block([[transform, transform], [transform, -transform]])
    return transform


def multiply(a, b, bases, k_block, center=False, rotate=False):
    a, b = a.astype(np.float64), b.astype(np.float64)
    variants = [np.zeros((a.shape[0], b.shape[1])) for _ in range(5)]
    chosen = []
    rotation_rng = np.random.default_rng(1367)
    for start in range(0, a.shape[1], k_block):
        aa, bb = a[:, start:start + k_block], b[start:start + k_block]
        k = aa.shape[1]
        adjustment = 0
        factor = 1
        if rotate and k == k_block and not (k & (k - 1)):
            signs = rotation_rng.choice((-1, 1), size=k)
            transform = signs[:, None] * hadamard(k)
            aa, bb = aa @ transform, transform.T @ bb
            factor = 1 / k
        if center:
            mean_a, mean_b = aa.mean(axis=1, keepdims=True), bb.mean(axis=0, keepdims=True)
            adjustment = aa.sum(axis=1, keepdims=True) * mean_b
            adjustment += mean_a * bb.sum(axis=0, keepdims=True) - k * mean_a * mean_b
            aa, bb = aa - mean_a, bb - mean_b
        *products, base = block_products(aa, bb, bases)
        chosen.append(base)
        for index, product in enumerate(products):
            variants[index] += factor * (product + adjustment if index < 3 else product)
    return variants, chosen


def data(rng, kind):
    a = rng.standard_normal((64, 128))
    b = rng.standard_normal((128, 64))
    if kind == 'outliers':
        for matrix in (a, b):
            matrix.reshape(-1)[rng.choice(matrix.size, 24, replace=False)] *= 256
    elif kind == 'exponents':
        a *= np.exp2(rng.integers(-8, 9, size=a.shape))
        b *= np.exp2(rng.integers(-8, 9, size=b.shape))
    elif kind == 'positive':
        a = rng.uniform(0.7, 1.3, a.shape)
        b = rng.uniform(0.7, 1.3, b.shape)
    elif kind == 'cancellation':
        a[:, 1::2] = a[:, ::2]
        b[1::2] = -b[::2] + 2**-11 * rng.standard_normal((64, 64))
    return a.astype(np.float16), b.astype(np.float16)


def stats():
    rng = np.random.default_rng(42)
    print('signed limits', LIMITS)
    print('relative Frobenius errors as percentages, median of eight draws')
    print('case                 whole16   K32-b16   K32-adapt        DC       sign     recenter   hadamard')
    for kind in ('normal', 'outliers', 'exponents', 'positive', 'cancellation'):
        rows, match, certified, magnitudes, chosen = [], [], [], [], []
        for _ in range(8):
            a, b = data(rng, kind)
            ref = a.astype(np.float64) @ b.astype(np.float64)
            whole, _ = multiply(a, b, (16,), 128)
            blocks, _ = multiply(a, b, (16,), 32)
            adapt, bases = multiply(a, b, BASES, 32)
            recentered, _ = multiply(a, b, BASES, 32, center=True)
            rotated, _ = multiply(a, b, BASES, 32, rotate=True)
            results = (whole[0], blocks[0], *adapt[:3], recentered[0], rotated[0])
            rows.append([100 * np.linalg.norm(result - ref) / np.linalg.norm(ref) for result in results])
            chosen.extend(bases)
            magnitudes.append((np.linalg.norm(ref), np.linalg.norm(adapt[0] - ref)))
            # Compare FP16 bit patterns, including signed zero. A small guard covers
            # the FP64 evaluation error for these synthetic, bounded test data.
            with np.errstate(over='ignore'):
                half = ref.astype(np.float16)
            finite = np.isfinite(half)
            ref_half = half.view(np.uint16)
            with np.errstate(over='ignore'):
                match.append([np.mean(result[finite].astype(np.float16).view(np.uint16) == ref_half[finite]) for result in results])
            checks = []
            for result, radius in ((adapt[0], adapt[3]), (adapt[1], adapt[4])):
                guard = 1e-10 * (1 + np.abs(result) + radius)
                with np.errstate(over='ignore'):
                    lower = (result - radius - guard).astype(np.float16).view(np.uint16)
                    upper = (result + radius + guard).astype(np.float16).view(np.uint16)
                confirmed = (lower == upper) & finite
                assert np.all(lower[confirmed] == ref_half[confirmed])
                checks.append(np.mean(confirmed[finite]))
            certified.append(checks)
        med = np.median(rows, axis=0)
        print(f'{kind:18s}' + ' '.join(f'{v:10.5f}' for v in med))
        print('  bases', {base: chosen.count(base) for base in BASES},
              'fp16 match %', np.round(100 * np.median(match, axis=0), 2),
              'certified %', np.round(100 * np.median(certified, axis=0), 2),
              'reference / absolute Frobenius', np.round(np.median(magnitudes, axis=0), 6))


def counterexamples():
    a, b = np.array([1, 3]), np.array([9, -3])
    h1, r1 = split(a, 16)
    h2, r2 = split(b, 16)
    approximate = 240 * (h1 @ h2) + 16 * ((h1 + r1) @ (h2 + r2))
    assert a @ b == 0 and approximate == -240
    print('cancellation counterexample', a, b, 'exact', a @ b, 'approx', approximate)


if __name__ == '__main__':
    stats()
    counterexamples()
