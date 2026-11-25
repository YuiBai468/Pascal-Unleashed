"""Pascal Unleashed's Windows CUDA client. PyTorch and NumPy are optional."""

from contextlib import contextmanager
import ctypes as c
import os
from pathlib import Path


__version__ = '1.0.0'
MODES = {'auto': 0, 'fast': 1, 'balanced': 2, 'precise': 3, 'native': 4, 'fp16': 5}
F16, F32 = 2, 4
FALLBACK, OUTLIER_GUARD, CAST_FP32, CACHE_WEIGHTS = 1, 2, 4, 8


class Error(RuntimeError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class Matrix(c.Structure):
    _fields_ = [(key, c.c_uint64) for key in ('data', 'row', 'col', 'batch')] + [
        ('type', c.c_int32), ('reserved', c.c_uint32)]


class Gemm(c.Structure):
    _fields_ = [(key, Matrix) for key in ('a','b','c')] + [
        (key, c.c_int32) for key in ('m','n','k','batches')] + [('alpha', c.c_float), ('beta', c.c_float),
        ('mode', c.c_int32), ('flags', c.c_uint32), ('tolerance', c.c_float), ('reserved', c.c_uint32),
        ('stream', c.c_uint64), ('weight_version', c.c_uint64)]


class Result(c.Structure):
    _fields_ = [('mode', c.c_int32), ('reason', c.c_int32), ('milliseconds', c.c_double), ('tflops', c.c_double)] + [
        (key, c.c_uint64) for key in ('accelerated','native','fallback','logical_ops','workspace_bytes')]

    def as_dict(self):
        return {name: getattr(self, name) for name, _ in self._fields_}


class Benchmark(c.Structure):
    _fields_ = [('run', Result), ('error', c.c_double), ('max_absolute', c.c_double),
                ('multiplier', c.c_double), ('native_ms', c.c_double), ('peak_device_bytes', c.c_uint64)]

    def as_dict(self):
        return self.run.as_dict() | {name: getattr(self, name) for name, _ in self._fields_[1:]}


class Device(c.Structure):
    _fields_ = [('name', c.c_char * 128), ('major', c.c_int32), ('minor', c.c_int32),
                ('driver', c.c_int32), ('compatible', c.c_int32), ('memory', c.c_uint64)]


def load():
    dll = Path(os.environ.get('PASCAL_UNLEASHED_DLL') or Path(__file__).with_name('PascalUnleashed.dll')).resolve()
    lib = c.CDLL(str(dll))
    signatures = {
        'pu_last_error': (c.c_char_p,), 'pu_version': (c.c_char_p,), 'pu_devices': (c.c_int,),
        'pu_device': (c.c_int, c.c_int, c.POINTER(Device)),
        'pu_open': (c.c_int, c.c_int, c.c_wchar_p, c.POINTER(c.c_void_p)),
        'pu_close': (None, c.c_void_p), 'pu_clear_cache': (None, c.c_void_p),
        'pu_gemm': (c.c_int, c.c_void_p, c.POINTER(Gemm), c.POINTER(Result)),
        'pu_alloc': (c.c_int, c.c_void_p, c.c_uint64, c.POINTER(c.c_uint64)),
        'pu_free': (c.c_int, c.c_void_p, c.c_uint64),
        'pu_upload': (c.c_int, c.c_void_p, c.c_uint64, c.c_void_p, c.c_uint64),
        'pu_download': (c.c_int, c.c_void_p, c.c_void_p, c.c_uint64, c.c_uint64),
        'pu_benchmark': (c.c_int, c.c_void_p, *([c.c_int] * 5), c.c_uint, c.c_int, c.c_uint, c.POINTER(Benchmark)),
    }
    for name, (result, *args) in signatures.items():
        function = getattr(lib,name)
        function.argtypes = args
        function.restype = result
    return lib


def check(lib, result):
    if result:
        detail = lib.pu_last_error().decode(errors='replace') or f'Pascal Unleashed error {result}'
        raise Error(result, detail)


def devices():
    lib = load()
    values=[]
    for ordinal in range(lib.pu_devices()):
        device = Device()
        check(lib,lib.pu_device(ordinal,c.byref(device)))
        row = {name: getattr(device,name) for name, _ in Device._fields_}
        row['name'] = row['name'].decode(errors='replace')
        values.append(row | {'ordinal': ordinal})
    return values


class Context:
    def __init__(self, device=0, cublas_dir=None):
        self.lib = load()
        self.handle = c.c_void_p()
        directory = str(Path(cublas_dir).resolve()) if cublas_dir else None
        check(self.lib,self.lib.pu_open(device,directory,c.byref(self.handle)))

    def __enter__(self):
        return self

    def __exit__(self,*exc):
        self.close()

    def close(self):
        if self.handle:
            self.lib.pu_close(self.handle)
            self.handle = c.c_void_p()

    def gemm(self, call):
        result = Result()
        check(self.lib,self.lib.pu_gemm(self.handle,c.byref(call),c.byref(result)))
        return result

    def clear_cache(self):
        self.lib.pu_clear_cache(self.handle)

    @contextmanager
    def memory(self, length):
        address = c.c_uint64()
        check(self.lib,self.lib.pu_alloc(self.handle,length,c.byref(address)))
        try:
            yield address.value
        finally:
            check(self.lib,self.lib.pu_free(self.handle,address))

    def upload(self, destination, host, length):
        check(self.lib,self.lib.pu_upload(self.handle,destination,host,length))

    def download(self, host, source, length):
        check(self.lib,self.lib.pu_download(self.handle,host,source,length))

    def benchmark(self, m, n, k, mode='auto', samples=3, seed=2109, dataset=0, flags=FALLBACK | OUTLIER_GUARD):
        result = Benchmark()
        check(self.lib,self.lib.pu_benchmark(self.handle,m,n,k,MODES[mode],samples,seed,dataset,flags,c.byref(result)))
        return result

    def numpy_gemm(self, a, b, *, mode='auto', flags=FALLBACK | OUTLIER_GUARD, tolerance=0):
        """Copy two NumPy matrices to CUDA and return A @ B and kernel statistics."""
        import numpy as np

        a=np.ascontiguousarray(a)
        b=np.ascontiguousarray(b.T)
        if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[1] or any(x.dtype not in (np.float16,np.float32) for x in (a,b)):
            raise ValueError('Expected compatible 2D FP16 or FP32 arrays')
        m,k=a.shape
        n=b.shape[0]
        result=np.empty((m,n),dtype=np.float32)
        with self.memory(max(1,a.nbytes)) as da,self.memory(max(1,b.nbytes)) as db,self.memory(max(1,result.nbytes)) as dc:
            if a.nbytes: self.upload(da,a.ctypes.data,a.nbytes)
            if b.nbytes: self.upload(db,b.ctypes.data,b.nbytes)
            stats=self.gemm(Gemm(Matrix(da,k,1,0,a.itemsize),Matrix(db,k,1,0,b.itemsize),Matrix(dc,n,1,0,F32),
                                 m,n,k,1,1.,0.,MODES[mode],flags,tolerance))
            if result.nbytes: self.download(result.ctypes.data,dc,result.nbytes)
        return result,stats
