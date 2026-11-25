import argparse
import ctypes as ct
import importlib.util
import os
import re
from pathlib import Path

import numpy as np

from ar2_audit import LIMITS, split


def api(library, name, *types):
    function = getattr(library, name)
    function.argtypes = types
    function.restype = ct.c_int

    def call(*args):
        status = function(*args)
        if status != 0:
            raise RuntimeError(f'{name} failed with status {status}')

    return call


def compile_kernel():
    library = Path(importlib.util.find_spec('torch').origin).parent / 'lib'
    with os.add_dll_directory(str(library)):
        builtins_path, = library.glob('nvrtc-builtins64_*.dll')
        _builtins = ct.WinDLL(str(builtins_path))
        nvrtc = ct.CDLL(str(library / 'nvrtc64_120_0.dll'))
        ptr, string, integer = ct.c_void_p, ct.c_char_p, ct.c_int
        create = api(nvrtc, 'nvrtcCreateProgram', ct.POINTER(ptr), string, string, integer, ptr, ptr)
        compile_program = api(nvrtc, 'nvrtcCompileProgram', ptr, integer, ct.POINTER(string))
        log_size = api(nvrtc, 'nvrtcGetProgramLogSize', ptr, ct.POINTER(ct.c_size_t))
        get_log = api(nvrtc, 'nvrtcGetProgramLog', ptr, ptr)
        ptx_size = api(nvrtc, 'nvrtcGetPTXSize', ptr, ct.POINTER(ct.c_size_t))
        get_ptx = api(nvrtc, 'nvrtcGetPTX', ptr, ptr)
        destroy = api(nvrtc, 'nvrtcDestroyProgram', ct.POINTER(ptr))
        program = ptr()
        source_file = Path(__file__).with_name('ar2_sm61.cu')
        create(ct.byref(program), source_file.read_bytes(), source_file.name.encode(), 0, None, None)
        try:
            options = (string * 2)(b'--gpu-architecture=compute_61', b'--std=c++14')
            try:
                compile_program(program, len(options), options)
            except RuntimeError:
                size = ct.c_size_t()
                log_size(program, ct.byref(size))
                message = ct.create_string_buffer(size.value)
                get_log(program, message)
                print(message.value.decode(errors='replace'))
                raise
            size = ct.c_size_t()
            ptx_size(program, ct.byref(size))
            ptx = ct.create_string_buffer(size.value)
            get_ptx(program, ptx)
            count = len(re.findall(rb'\bdp4a\.s32\.s32\b', ptx.value))
            assert count == 128, count
            print(f'NVRTC: compute_61 PTX, {count} static signed DP4A instructions', flush=True)
            return ptx
        finally:
            destroy(ct.byref(program))


def pack(values):
    packed_width = (values.shape[1] + 3) // 4
    padded = np.zeros((values.shape[0], packed_width * 4), dtype=np.int8)
    padded[:, :values.shape[1]] = values
    return padded.view(np.int32)


def test_case(rng, m, n, k):
    bases = np.array([tuple(LIMITS)[i % len(LIMITS)] for i in range((k + 31) // 32)], dtype=np.int32)
    a_sum = np.empty((m, k), dtype=np.int8)
    b_sum = np.empty((k, n), dtype=np.int8)
    a_high, b_high = np.empty_like(a_sum), np.empty_like(b_sum)
    a_scales = np.exp2(rng.integers(-9, -3, size=(len(bases), m))).astype(np.float32)
    b_scales = np.exp2(rng.integers(-9, -3, size=(len(bases), n))).astype(np.float32)
    expected = np.zeros((m, n), dtype=np.float32)
    for step, base in enumerate(bases):
        start = step * 32
        end = min(k, start + 32)
        lim = LIMITS[base]
        aa = rng.integers(-lim, lim + 1, size=(m, end - start))
        bb = rng.integers(-lim, lim + 1, size=(end - start, n))
        ha, ra = split(aa, base)
        hb, rb = split(bb, base)
        a_high[:, start:end], b_high[start:end] = ha, hb
        a_sum[:, start:end], b_sum[start:end] = ha + ra, hb + rb
        exact_integer = aa @ bb + (base - 1) * (ra @ rb)
        scales = a_scales[step, :, None] * b_scales[step, None, :]
        product = exact_integer.astype(np.float32).astype(np.float64) * scales
        expected = (expected.astype(np.float64) + product).astype(np.float32)
    return (pack(a_high), pack(a_sum), pack(b_high.T), pack(b_sum.T), a_scales, b_scales, bases), expected


def validate(ptx):
    driver = ct.WinDLL('nvcuda.dll')
    ptr, dp, integer, uint, size = ct.c_void_p, ct.c_uint64, ct.c_int, ct.c_uint, ct.c_size_t
    api(driver, 'cuInit', uint)(0)
    device = integer()
    api(driver, 'cuDeviceGet', ct.POINTER(integer), integer)(ct.byref(device), 0)
    device_name = ct.create_string_buffer(128)
    api(driver, 'cuDeviceGetName', ptr, integer, integer)(device_name, len(device_name), device)
    print('CUDA device:', device_name.value.decode(), flush=True)
    attribute = api(driver, 'cuDeviceGetAttribute', ct.POINTER(integer), integer, integer)
    major, minor = integer(), integer()
    attribute(ct.byref(major), 75, device)
    attribute(ct.byref(minor), 76, device)
    if (major.value, minor.value) != (6, 1):
        raise RuntimeError('The selected GPU must have compute capability 6.1')
    context = ptr()
    api(driver, 'cuDevicePrimaryCtxRetain', ct.POINTER(ptr), integer)(ct.byref(context), device)
    set_context = api(driver, 'cuCtxSetCurrent', ptr)
    set_context(context)
    module = ptr()
    api(driver, 'cuModuleLoadData', ct.POINTER(ptr), ptr)(ct.byref(module), ptx)
    function = ptr()
    api(driver, 'cuModuleGetFunction', ct.POINTER(ptr), ptr, ct.c_char_p)(ct.byref(function), module, b'ar2_gemm')
    allocate = api(driver, 'cuMemAlloc_v2', ct.POINTER(dp), size)
    upload = api(driver, 'cuMemcpyHtoD_v2', dp, ptr, size)
    download = api(driver, 'cuMemcpyDtoH_v2', ptr, dp, size)
    free = api(driver, 'cuMemFree_v2', dp)
    launch = api(driver, 'cuLaunchKernel', ptr, uint, uint, uint, uint, uint, uint, uint, ptr, ptr, ptr)
    rng = np.random.default_rng(42)
    try:
        for m, n, k in ((64, 64, 128), (67, 51, 159), (5, 7, 3), (1, 1, 1)):
            arrays, expected = test_case(rng, m, n, k)
            actual = np.empty_like(expected)
            handles = []
            try:
                for index, array in enumerate((*arrays, actual)):
                    handle = dp()
                    allocate(ct.byref(handle), array.nbytes)
                    handles.append(handle)
                    if index < len(arrays):
                        upload(handle, array.ctypes.data, array.nbytes)
                values = (*handles, integer(m), integer(n), integer((k + 3) // 4))
                params = (ptr * len(values))(*(ct.cast(ct.byref(value), ptr) for value in values))
                launch(function, (n + 31) // 32, (m + 31) // 32, 1, 8, 16, 1, 0, None, params, None)
                download(actual.ctypes.data, handles[-1], actual.nbytes)
                assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32)), np.max(np.abs(actual - expected))
                print(f'M={m}, N={n}, K={k}: {actual.size} FP32 outputs, bitwise match', flush=True)
            finally:
                for handle in handles:
                    free(handle)
    finally:
        api(driver, 'cuModuleUnload', ptr)(module)
        set_context(None)
        api(driver, 'cuDevicePrimaryCtxRelease_v2', integer)(device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='GPU UUID from nvidia-smi -L. Omit to compile without running.')
    args = parser.parse_args()
    ptx = compile_kernel()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        validate(ptx)
