"""Opt-in PyTorch GEMM and process-local inference linear interception."""

import atexit
from contextlib import contextmanager
import itertools
import json
import os
from pathlib import Path
import threading
import time
import weakref

import torch
import torch.nn.functional as functional

from . import (CACHE_WEIGHTS, CAST_FP32, FALLBACK, OUTLIER_GUARD, F16, F32,
               MODES, Context, Error, Gemm, Matrix, __version__)


class Engine:
    def __init__(self, mode='auto', *, fallback=True, guard=True, cache=False, cast_fp32=False):
        self.mode=mode
        self.flags=(FALLBACK if fallback else 0) | (OUTLIER_GUARD if guard else 0) | (
            CACHE_WEIGHTS if cache else 0) | (CAST_FP32 if cast_fp32 else 0)
        self.contexts={}
        self.versions={}
        self.tokens=itertools.count(1)
        self.lock=threading.RLock()
        self.metrics=dict(version=__version__,pid=os.getpid(),loaded=True,requested=mode,actual=None,
                          accelerated=0,native=0,fallback=0,logical_ops=0,milliseconds=0., tflops=None)
        location=os.environ.get('PU_TELEMETRY')
        self.file=Path(location) if location else None
        self.last_write=0.
        self.flush(force=True)

    def flush(self, force=False):
        if not self.file or (not force and time.monotonic()-self.last_write<1):
            return
        pending=self.file.with_suffix('.tmp')
        try:
            pending.write_text(json.dumps(self.metrics),encoding='utf-8')
            pending.replace(self.file)
            self.last_write=time.monotonic()
        except OSError:
            pass

    def close(self):
        with self.lock:
            self.flush(force=True)
            for ctx in self.contexts.values():
                ctx.close()
            self.contexts.clear()
            self.versions.clear()

    def clear(self):
        with self.lock:
            for ctx in self.contexts.values():
                ctx.clear_cache()
            self.versions.clear()

    def _context(self, device):
        index=device.index if device.index is not None else torch.cuda.current_device()
        if index not in self.contexts:
            runtime=Path(torch.__file__).resolve().parent/'lib'
            directory=os.environ.get('PASCAL_CUDA_DLLS') or (runtime if (runtime/'cublas64_12.dll').is_file() else None)
            with torch.cuda.device(index):
                torch.cuda.current_stream(index)
                self.contexts[index]=Context(index,directory)
        return self.contexts[index]

    def _weight_version(self, weight):
        if not self.flags & CACHE_WEIGHTS:
            return 0
        try:
            version=weight._version
        except RuntimeError:
            # Inference tensors have no version counter. Caching them needs an explicit immutable-weight opt-in.
            if os.environ.get('PU_IMMUTABLE_WEIGHTS')!='1':
                return 0
            version=None
        saved=self.versions.get(id(weight))
        if saved is not None and saved[0]() is weight and saved[1]==version:
            return saved[2]
        token=next(self.tokens)
        self.versions[id(weight)]=(weakref.ref(weight),version,token)
        return token

    @staticmethod
    def _descriptor(tensor, batch=0):
        return Matrix(tensor.data_ptr(),tensor.stride(-2),tensor.stride(-1),batch,
                      F16 if tensor.dtype==torch.float16 else F32)

    def _run(self, a, b, out, weights=None, tolerance=0):
        m,k=a.shape[-2:]
        n=b.shape[-2]
        batches=a.shape[0] if a.ndim==3 else 1
        if not batches or not m or not n or not k:
            out.zero_()
            return out
        with self.lock,torch.cuda.device(a.device):
            ctx=self._context(a.device)
            tag=self._weight_version(weights) if weights is not None else 0
            ad=self._descriptor(a,a.stride(0) if a.ndim==3 else 0)
            bd=self._descriptor(b,b.stride(0) if b.ndim==3 else 0)
            cd=self._descriptor(out,out.stride(0) if out.ndim==3 else 0)
            stream=torch.cuda.current_stream(a.device).cuda_stream
            result=ctx.gemm(Gemm(ad,bd,cd,m,n,k,batches,1.,0.,MODES[self.mode],self.flags,tolerance,0,stream,tag))
            info=result.as_dict()
            for key in ('accelerated','native','fallback','logical_ops','milliseconds'):
                self.metrics[key]+=info[key]
            self.metrics.update(actual=next(name for name, index in MODES.items() if index==result.mode),
                                tflops=result.tflops,reason=result.reason,workspace_bytes=result.workspace_bytes)
            self.flush()
        return out

    def native(self, original, *args, **kwargs):
        result=original(*args,**kwargs)
        with self.lock:
            self.metrics['native']+=1
            self.metrics['fallback']+=1
            self.metrics.update(actual='native',tflops=None)
            self.flush()
        return result

    def matmul(self, a, b, *, tolerance=0):
        compatible=(a.ndim>=2 and b.ndim>=2 and a.is_cuda and b.device==a.device and a.dtype==b.dtype and
                    a.dtype in (torch.float16,torch.float32) and a.shape[-1]==b.shape[-2] and
                    not (torch.is_grad_enabled() and (a.requires_grad or b.requires_grad)))
        if not compatible:
            return self.native(torch.matmul,a,b)
        batch=torch.broadcast_shapes(a.shape[:-2],b.shape[:-2])
        m,k=a.shape[-2:]
        n=b.shape[-1]
        output=torch.empty((*batch,m,n),device=a.device,dtype=a.dtype)
        try:
            if batch:
                self._run(a.expand(*batch,m,k).flatten(0,len(batch)-1),
                          b.expand(*batch,k,n).transpose(-1,-2).flatten(0,len(batch)-1),
                          output.flatten(0,len(batch)-1),tolerance=tolerance)
            else:
                self._run(a,b.T,output,tolerance=tolerance)
        except (Error,OSError):
            if not self.flags & FALLBACK:
                raise
            return self.native(torch.matmul,a,b)
        return output

    def linear(self, original, values, weight, bias=None):
        if (not values.is_cuda or values.device!=weight.device or values.dtype!=weight.dtype or
            values.dtype not in (torch.float16,torch.float32) or weight.ndim!=2 or
            values.shape[-1]!=weight.shape[-1] or weight.shape[-1]==0 or
            (torch.is_grad_enabled() and (values.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad)))):
            return self.native(original,values,weight,bias)
        if values.dtype==torch.float32 and not self.flags&CAST_FP32:
            return self.native(original,values,weight,bias)
        matrix=values.reshape(-1,values.shape[-1])
        out=torch.empty((matrix.shape[0],weight.shape[0]),device=values.device,dtype=values.dtype)
        try:
            self._run(matrix,weight,out,weight)
        except (Error,OSError):
            if not self.flags & FALLBACK:
                raise
            return self.native(original,values,weight,bias)
        if bias is not None:
            out=out+bias
        return out.reshape(*values.shape[:-1],weight.shape[0])


@contextmanager
def patch_linear(engine):
    original=functional.linear
    functional.linear=lambda input,weight,bias=None:engine.linear(original,input,weight,bias)
    try:
        yield engine
    finally:
        functional.linear=original


def activate_from_environment():
    if os.environ.get('PU_ENABLED')!='1':
        return None
    mode=os.environ.get('PU_MODE','auto')
    if mode=='native':
        return None
    engine=Engine(mode,fallback=os.environ.get('PU_FALLBACK','1')=='1',guard=os.environ.get('PU_GUARD','1')=='1',
                  cache=os.environ.get('PU_CACHE','0')=='1',cast_fp32=os.environ.get('PU_CAST_FP32','0')=='1')
    patch=patch_linear(engine)
    patch.__enter__()
    atexit.register(engine.close)
    atexit.register(patch.__exit__,None,None,None)
    return engine
