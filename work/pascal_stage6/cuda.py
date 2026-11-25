import ctypes as ct
import importlib.util
import os
import re
from contextlib import contextmanager
from pathlib import Path

import numpy as np


PTR = ct.c_void_p
DEV_PTR = ct.c_uint64
INT = ct.c_int
UINT = ct.c_uint
SIZE = ct.c_size_t
FLOAT = ct.c_float
STRING = ct.c_char_p


def api(library, name, *types):
    function = getattr(library, name)
    function.argtypes = types
    function.restype = INT

    def call(*args):
        status = function(*args)
        if status:
            raise RuntimeError(f'{name} returned {status}')

    return call


def library_path():
    override = os.getenv('PASCAL_CUDA_DLLS')
    if override:
        return Path(override)
    torch = importlib.util.find_spec('torch')
    if not torch or not torch.origin:
        raise RuntimeError('Set PASCAL_CUDA_DLLS to the directory containing the CUDA 12 DLLs')
    return Path(torch.origin).parent / 'lib'


def compile_ptx(source, *, architecture='compute_61', output='PTX'):
    path = library_path()
    with os.add_dll_directory(str(path)):
        builtins_path, = path.glob('nvrtc-builtins64_*.dll')
        builtins = ct.WinDLL(str(builtins_path))
        nvrtc = ct.CDLL(str(path / 'nvrtc64_120_0.dll'))
        create = api(nvrtc, 'nvrtcCreateProgram', ct.POINTER(PTR), STRING, STRING, INT, PTR, PTR)
        compiler = api(nvrtc, 'nvrtcCompileProgram', PTR, INT, ct.POINTER(STRING))
        destroy = api(nvrtc, 'nvrtcDestroyProgram', ct.POINTER(PTR))
        program = PTR()
        create(ct.byref(program), source, b'pascal_unleashed.cu', 0, None, None)
        try:
            options = (STRING * 2)(f'--gpu-architecture={architecture}'.encode(), b'--std=c++14')
            try:
                compiler(program, len(options), options)
            except RuntimeError:
                size = SIZE()
                api(nvrtc, 'nvrtcGetProgramLogSize', PTR, ct.POINTER(SIZE))(program, ct.byref(size))
                buffer = ct.create_string_buffer(size.value)
                api(nvrtc, 'nvrtcGetProgramLog', PTR, PTR)(program, buffer)
                raise RuntimeError(buffer.value.decode(errors='replace')) from None
            size = SIZE()
            api(nvrtc, f'nvrtcGet{output}Size', PTR, ct.POINTER(SIZE))(program, ct.byref(size))
            buffer = ct.create_string_buffer(size.value)
            api(nvrtc, f'nvrtcGet{output}', PTR, PTR)(program, buffer)
            if output == 'PTX':
                count = len(re.findall(rb'\bdp4a\.s32\.s32\b', buffer.value))
                print(f'NVRTC: {architecture} PTX, {count} static signed DP4A instructions', flush=True)
            return buffer
        finally:
            destroy(ct.byref(program))


class Cuda:
    def __init__(self, gpu):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        self.driver = d = ct.WinDLL('nvcuda.dll')
        api(d, 'cuInit', UINT)(0)
        self.device = INT()
        api(d, 'cuDeviceGet', ct.POINTER(INT), INT)(ct.byref(self.device), 0)
        label = ct.create_string_buffer(128)
        api(d, 'cuDeviceGetName', PTR, INT, INT)(label, len(label), self.device)
        self.name = label.value.decode()
        attribute = api(d, 'cuDeviceGetAttribute', ct.POINTER(INT), INT, INT)
        major, minor = INT(), INT()
        attribute(ct.byref(major), 75, self.device)
        attribute(ct.byref(minor), 76, self.device)
        if (major.value, minor.value) != (6, 1):
            raise RuntimeError(f'{self.name} must have compute capability 6.1')
        self.context = PTR()
        api(d, 'cuDevicePrimaryCtxRetain', ct.POINTER(PTR), INT)(ct.byref(self.context), self.device)
        api(d, 'cuCtxSetCurrent', PTR)(self.context)
        self.allocate = api(d, 'cuMemAlloc_v2', ct.POINTER(DEV_PTR), SIZE)
        self.free = api(d, 'cuMemFree_v2', DEV_PTR)
        self.upload = api(d, 'cuMemcpyHtoD_v2', DEV_PTR, PTR, SIZE)
        self.download = api(d, 'cuMemcpyDtoH_v2', PTR, DEV_PTR, SIZE)
        self.sync = api(d, 'cuCtxSynchronize')
        self.launch = api(d, 'cuLaunchKernel', PTR, UINT, UINT, UINT, UINT, UINT, UINT, UINT, PTR, PTR, PTR)
        self.create_event = api(d, 'cuEventCreate', ct.POINTER(PTR), UINT)
        self.record_event = api(d, 'cuEventRecord', PTR, PTR)
        self.wait_event = api(d, 'cuEventSynchronize', PTR)
        self.elapsed = api(d, 'cuEventElapsedTime', ct.POINTER(FLOAT), PTR, PTR)
        self.destroy_event = api(d, 'cuEventDestroy_v2', PTR)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        api(self.driver, 'cuCtxSetCurrent', PTR)(None)
        api(self.driver, 'cuDevicePrimaryCtxRelease_v2', INT)(self.device)

    @contextmanager
    def load(self, ptx):
        module = PTR()
        api(self.driver, 'cuModuleLoadData', ct.POINTER(PTR), PTR)(ct.byref(module), ptx)
        try:
            yield module
        finally:
            api(self.driver, 'cuModuleUnload', PTR)(module)

    def kernel(self, module, name):
        function = PTR()
        api(self.driver, 'cuModuleGetFunction', ct.POINTER(PTR), PTR, STRING)(ct.byref(function), module, name.encode())
        getter = api(self.driver, 'cuFuncGetAttribute', ct.POINTER(INT), INT, PTR)
        info = {}
        for label, field in (('shared_bytes', 1), ('local_bytes', 3), ('registers', 4), ('binary_version', 6)):
            value = INT()
            getter(ct.byref(value), field, function)
            info[label] = value.value
        return function, info

    @contextmanager
    def buffer(self, host_array):
        handle = DEV_PTR()
        length = host_array if isinstance(host_array, int) else host_array.nbytes
        self.allocate(ct.byref(handle), length)
        try:
            if not isinstance(host_array, int):
                if not host_array.flags.c_contiguous:
                    raise ValueError('Upload requires a contiguous buffer')
                self.upload(handle, host_array.ctypes.data, length)
            yield handle
        finally:
            self.free(handle)

    def read(self, handle, shape, dtype):
        result = np.empty(shape, dtype=dtype)
        self.download(result.ctypes.data, handle, result.nbytes)
        return result

    def invocation(self, function, shape, tile, args, block=(8, 16, 1)):
        m, n, _ = shape
        tm, tn = tile
        launch_grid = ((n + tn - 1) // tn, (m + tm - 1) // tm, 1)
        return self.grid_invocation(function, launch_grid, block, args)

    def grid_invocation(self, function, launch_grid, block, args, stream=None):
        # A captured pointer must own each scalar across repeated launches.
        params = (PTR * len(args))(*(ct.cast(ct.pointer(value), PTR) for value in args))
        return lambda: self.launch(function, *launch_grid, *block, 0, stream, params, None)

    @contextmanager
    def stream(self, priority=None):
        handle = PTR()
        if priority is None:
            api(self.driver, 'cuStreamCreate', ct.POINTER(PTR), UINT)(ct.byref(handle), 1)
        else:
            api(self.driver, 'cuStreamCreateWithPriority', ct.POINTER(PTR), UINT, INT)(ct.byref(handle), 1, priority)
        try:
            yield handle
        finally:
            api(self.driver, 'cuStreamDestroy_v2', PTR)(handle)

    def priority_range(self):
        low, high = INT(), INT()
        api(self.driver, 'cuCtxGetStreamPriorityRange', ct.POINTER(INT), ct.POINTER(INT))(ct.byref(low), ct.byref(high))
        return low.value, high.value

    @contextmanager
    def dependency_event(self):
        event = PTR()
        self.create_event(ct.byref(event), 2)
        try:
            yield event
        finally:
            self.destroy_event(event)

    def measure(self, operation, repeats, batches):
        for _ in range(3):
            operation()
        self.sync()
        start, stop = PTR(), PTR()
        self.create_event(ct.byref(start), 0)
        self.create_event(ct.byref(stop), 0)
        try:
            times = []
            for _ in range(batches):
                self.record_event(start, None)
                for _ in range(repeats):
                    operation()
                self.record_event(stop, None)
                self.wait_event(stop)
                duration = FLOAT()
                self.elapsed(ct.byref(duration), start, stop)
                times.append(duration.value / repeats)
            return times
        finally:
            self.destroy_event(start)
            self.destroy_event(stop)


class Cublas:
    def __init__(self):
        path = library_path()
        self.dll_directory = os.add_dll_directory(str(path))
        dll = ct.WinDLL(str(path / 'cublas64_12.dll'))
        self.handle = PTR()
        api(dll, 'cublasCreate_v2', ct.POINTER(PTR))(ct.byref(self.handle))
        self.destroy = api(dll, 'cublasDestroy_v2', PTR)
        self.set_stream = api(dll, 'cublasSetStream_v2', PTR, PTR)
        version = INT()
        api(dll, 'cublasGetVersion_v2', PTR, ct.POINTER(INT))(self.handle, ct.byref(version))
        self.version = version.value
        self.sgemm = api(dll, 'cublasSgemm_v2', PTR, INT, INT, INT, INT, INT,
                         PTR, DEV_PTR, INT, DEV_PTR, INT, PTR, DEV_PTR, INT)
        self.gemm_ex = api(dll, 'cublasGemmEx', PTR, INT, INT, INT, INT, INT, PTR, DEV_PTR, INT, INT,
                           DEV_PTR, INT, INT, PTR, DEV_PTR, INT, INT, INT, INT)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.destroy(self.handle)
        self.dll_directory.close()

    def invocation(self, method, shape, a, b, c, *, lda=None, offset=0, int_alpha=1, int_beta=0, algorithm=-1):
        m, n, k = shape
        # The stored rows of B-transposed are columns to cuBLAS. Compute C-transposed.
        if method == 'sgemm':
            alpha, beta = FLOAT(1), FLOAT(0)
            return lambda: self.sgemm(self.handle, 1, 0, n, m, k, ct.byref(alpha), b, k, a, k,
                                      ct.byref(beta), c, n)
        if method == 'fp16_fp32':
            alpha, beta, input_type, output_type, compute_type = FLOAT(1), FLOAT(0), 2, 0, 68
        elif method == 'fp16_fp16':
            alpha, beta = ct.c_uint16(0x3c00), ct.c_uint16(0)
            input_type, output_type, compute_type = 2, 2, 64
        elif method == 'int8':
            alpha, beta, input_type, output_type, compute_type = INT(int_alpha), INT(int_beta), 3, 10, 72
        else:
            raise ValueError(method)
        leading = k if lda is None else lda
        da, db = DEV_PTR(a.value + offset), DEV_PTR(b.value + offset)
        return lambda: self.gemm_ex(self.handle, 1, 0, n, m, k, ct.byref(alpha), db, input_type,
                                    leading, da, input_type, leading, ct.byref(beta), c, output_type, n, compute_type, algorithm)
