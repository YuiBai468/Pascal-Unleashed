"""Build the release module and embedded codebooks using the archived encoders."""

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
STAGE = ROOT.parent / 'work' / 'pascal_stage6'
sys.path.insert(0, str(STAGE))
from cuda import compile_ptx
from plane_gpu import packed_table
from plane_lattice import build, line


def emit(header, name, data):
    header.write(f'static const unsigned char {name}[] = {{\n')
    for start in range(0, len(data), 32):
        header.write(','.join(str(v) for v in data[start:start+32]) + ',\n')
    header.write('};\n')


def main():
    target = Path(sys.argv[1])
    target.mkdir(parents=True, exist_ok=True)
    names = ('plane_sm61.cu', 'packed_layout.cu', 'normal_slide.cu', 'slide_line_large.cu',
             'clip_ar2.cu', 'clip_csr.cu', 'clip_tail_tile.cu', 'dense08.cu')
    parts = [(STAGE / name).read_bytes() for name in names]
    parts.extend((ROOT.parent / 'work' / 'anima_p40' / name).read_bytes()
                 for name in ('anima_gemm.cu', 'model_tail.cu'))
    parts.append((ROOT / 'native' / 'runtime.cu').read_bytes())
    source = b'\n'.join(parts)
    (target / 'pascal_sm61.ptx').write_bytes(bytes(compile_ptx(source)))
    basis, pair_lut = build((3, 4, 5), .0025, 16)
    _, line_lut = line((3, 4), .0001, 16)
    with (target / 'codebooks.h').open('w', encoding='ascii') as header:
        for name, data in [('basis', basis), ('pair_lut', packed_table(pair_lut)), ('line_lut', packed_table(line_lut))]:
            emit(header, name, np.ascontiguousarray(data).tobytes())
        for name, filename, cutoff in [('middle', 'asymmetric-stage10-3p45.npz', 3.45),
                                       ('precise', 'folded-lane-0.npz', 3.02)]:
            with np.load(STAGE / 'results' / filename, allow_pickle=False) as book:
                config = json.loads(str(book['metadata']))
                cap = 127 * config['base']
                a, b = book['weights']
                for side in ('a','b'):
                    emit(header, name + '_' + side, book[side].tobytes())
                header.write(f'static const Book {name} = {{{cap}, {len(book["a"])-1}, {cutoff:.8f}f, '
                             f'{a/b:.12g}f, {b*(cap/127)**2:.12g}f}};\n')


if __name__ == '__main__':
    main()
