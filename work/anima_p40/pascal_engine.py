"""Run the saved Pascal Unleashed encoders on Anima's CUDA linear layers."""

from contextlib import ExitStack
import ctypes as ct
from pathlib import Path
import sys
import weakref

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
STAGE = ROOT.parent / 'pascal_stage6'
sys.path.insert(0, str(STAGE))
from cuda import Cublas, Cuda, DEV_PTR, FLOAT, INT, PTR, compile_ptx
from plane_gpu import packed_table
from plane_lattice import build, line


def pointer(tensor):
    return DEV_PTR(0 if tensor is None else tensor.data_ptr())


class Engine:
    def __init__(self, gpu):
        self.stack = ExitStack()
        self.cuda = self.stack.enter_context(Cuda(gpu))
        self.blas = self.stack.enter_context(Cublas())
        basis, plane = build((3, 4, 5), .002, 16)
        _, scalar = line((3, 4), .0001, 16)
        self.margin = 127 - max(int(np.ceil(3 + np.abs(plane).max())),
                                int(np.ceil(2.5 + np.abs(scalar).max())))
        self.tables = tuple(self.stack.enter_context(self.cuda.buffer(np.ascontiguousarray(table)))
                            for table in (basis, packed_table(plane), packed_table(scalar)))
        sources = [STAGE / name for name in ('plane_sm61.cu', 'normal_slide.cu', 'slide_line_large.cu')]
        sources.append(ROOT / 'anima_gemm.cu')
        self.module = self.stack.enter_context(self.cuda.load(compile_ptx(b'\n'.join(p.read_bytes() for p in sources))))
        self.pack = self.cuda.kernel(self.module, 'anima_pack_rows')[0]
        self.finish = self.cuda.kernel(self.module, 'anima_reconstruct')[0]
        self.one, self.zero = INT(1), INT(0)
        self.mode = 'fp32'
        self.weights = {}
        self.reset()

    def reset(self):
        self.bad = torch.zeros(1, device='cuda:0', dtype=torch.int32)
        self.calls = self.fallback_calls = self.ops = self.fallback_ops = 0
        self.weight_packs = 0
        self.probes = {}

    def _geometry(self, k):
        pairs = (k + 1) // 2
        ratio = .125 if self.mode == 'turbo' else 23 / 32
        dual = round(pairs * ratio)
        physical = 3 * pairs + dual
        return dual, (physical + 3) // 4 * 4

    def _encode(self, values, dual, width, stream):
        rows, k = values.shape
        packed = torch.empty((rows, width), device=values.device, dtype=torch.int8)
        scale = torch.empty(rows, device=values.device, dtype=torch.float32)
        args = (pointer(values), pointer(packed), pointer(scale), INT(k), INT(dual), INT(width),
                *self.tables, pointer(self.bad), INT(self.margin))
        self.cuda.grid_invocation(self.pack, (rows, 1, 1), (256, 1, 1), args, stream=stream)()
        return packed, scale

    def linear(self, original, values, weight, bias=None):
        if not (values.is_cuda and weight.is_cuda and values.dtype == weight.dtype == torch.float32):
            return original(values, weight, bias)
        k, n = weight.shape[1], weight.shape[0]
        m = values.numel() // k
        ops = 2 * m * n * k
        if m < 4 or n % 4 or not m or not n:
            self.fallback_calls += 1
            self.fallback_ops += ops
            return original(values, weight, bias)

        stream = PTR(torch.cuda.current_stream(values.device).cuda_stream)
        self.blas.set_stream(self.blas.handle, stream)
        dual, width = self._geometry(k)
        matrix = values.reshape(m, k).contiguous()
        left, left_scale = self._encode(matrix, dual, width, stream)
        key = (self.mode, weight.data_ptr(), n, k)
        entry = self.weights.get(key)
        if entry is None or entry[0]() is not weight:
            right, right_scale = self._encode(weight.contiguous(), dual, width, stream)
            entry = (weakref.ref(weight), right.T.contiguous(), right_scale)
            self.weights[key] = entry
            self.weight_packs += 1
        _, right, right_scale = entry
        raw = torch.empty((m, n), device=values.device, dtype=torch.int32)
        result = torch.empty_like(raw, dtype=torch.float32)
        self.blas.gemm_ex(self.blas.handle, 0, 0, n, m, width, ct.byref(self.one), pointer(right), 3, n,
                          pointer(left), 3, width, ct.byref(self.zero), pointer(raw), 10, n, 72, 1)
        shift = bias.contiguous() if bias is not None else None
        args = (*map(pointer, (raw, left_scale, right_scale, shift, result)), INT(m), INT(n))
        self.cuda.grid_invocation(self.finish, ((n+127)//128, (m+15)//16, 1), (32, 4, 1), args, stream=stream)()
        self.calls += 1
        self.ops += ops

        shape = (m, n, k)
        if shape not in self.probes:
            rows = torch.arange(16, device=values.device) * (m-1) // 15 if m >= 16 else slice(None)
            reference = original(matrix[rows], weight, bias)
            difference = (result[rows] - reference).norm().item()
            denominator = reference.norm().item()
            self.probes[shape] = difference / max(denominator, 1e-30)
        return result.reshape(*values.shape[:-1], n)

    def report(self):
        flags = self.bad.item()
        if flags:
            raise RuntimeError(f'Pascal FP16 input or signed-byte encoding flagged {flags}')
        return {'mode': self.mode, 'int8_linear_calls': self.calls, 'int8_linear_flops': self.ops,
                'fp32_linear_calls': self.fallback_calls, 'fp32_linear_flops': self.fallback_ops,
                'weight_packs': self.weight_packs, 'margin': self.margin,
                'linear_error_probes': [{'shape': s, 'relative_frobenius': e} for s,e in sorted(self.probes.items())]}

    def self_check(self, stream):
        # At K=8192 the dynamic program must produce the archived code bytes.
        with torch.inference_mode():
            generator = torch.Generator(device='cuda:0').manual_seed(781)
            sample = torch.randn((32, 8192), generator=generator, device='cuda:0')
            original_input = sample.half()
            for mode in ('turbo', 'balanced'):
                self.mode = mode
                dual, width = self._geometry(8192)
                code, scale = self._encode(sample, dual, width, stream)
                expected = torch.empty_like(code)
                expected_scales = torch.empty_like(scale)
                function = self.cuda.kernel(self.module, f'slide_line_{2*dual}')[0]
                args = (*map(pointer, (original_input, expected, expected_scales)), *self.tables,
                        pointer(self.bad), INT(self.margin), FLOAT(.81), FLOAT(.89))
                self.cuda.grid_invocation(function, (32, 1, 1), (512, 1, 1), args, stream=stream)()
                torch.testing.assert_close(code, expected, rtol=0, atol=0)
                torch.testing.assert_close(scale, expected_scales, rtol=0, atol=0)
                print(f'{mode}: runtime K encoder matches archived K=8192 bytes and scales', flush=True)
            test = sample[:16, :96].contiguous()
            test_weight = sample[:, :96].contiguous()
            result = self.linear(torch.nn.functional.linear, test, test_weight)
            error = (result - test @ test_weight.T).norm() / (test @ test_weight.T).norm()
            if not error.item() < .01 or self.bad.item():
                raise RuntimeError(f'Pascal trial GEMM failed (relative error {error.item():.6%})')
            self.weights.clear()
            self.reset()
