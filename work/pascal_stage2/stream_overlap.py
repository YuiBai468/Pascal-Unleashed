"""Time independent GEMMs on one P40, with a common device-side event gate."""

import argparse
import ctypes as ct
from contextlib import ExitStack, contextmanager
import json
from pathlib import Path

import numpy as np

from bench import SOURCE_DIR, upload_many
from cuda import Cublas, Cuda, FLOAT, PTR, UINT, api, library_path


@contextmanager
def stream(cuda):
    handle = PTR()
    api(cuda.driver, 'cuStreamCreate', ct.POINTER(PTR), UINT)(ct.byref(handle), 1)
    try:
        yield handle
    finally:
        api(cuda.driver, 'cuStreamDestroy_v2', PTR)(handle)


@contextmanager
def event(cuda):
    handle = PTR()
    cuda.create_event(ct.byref(handle), 0)
    try:
        yield handle
    finally:
        cuda.destroy_event(handle)


def measure(cuda, control, event_start, event_stop, lanes):
    wait = api(cuda.driver, 'cuStreamWaitEvent', PTR, PTR, UINT)
    cuda.record_event(event_start, control)
    for handle, _, _ in lanes:
        wait(handle, event_start, 0)
    for handle, done, operations in lanes:
        for operation in operations:
            operation()
        cuda.record_event(done, handle)
    for _, done, _ in lanes:
        wait(control, done, 0)
    cuda.record_event(event_stop, control)
    cuda.wait_event(event_stop)
    elapsed = FLOAT()
    cuda.elapsed(ct.byref(elapsed), event_start, event_stop)
    return float(elapsed.value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--output', type=Path, default=SOURCE_DIR/'results'/'stream-overlap.json')
    args = parser.parse_args()
    n, fp_k = 4096, 128
    shape, fp_shape = (n,n,n), (n,n,fp_k)
    rng = np.random.default_rng(79)
    h, i, g, j = [rng.integers(-100, 101, (n,n), dtype=np.int8) for _ in range(4)]
    f, r = [rng.standard_normal((n,fp_k), dtype=np.float32) for _ in range(2)]
    with Cuda(args.gpu) as cuda, Cublas() as primary, Cublas() as secondary, ExitStack() as stack:
        streams = [stack.enter_context(stream(cuda)) for _ in range(3)]
        s0,s1,control = streams
        e0,e1,start,stop = [stack.enter_context(event(cuda)) for _ in range(4)]
        setter = api(ct.WinDLL(str(library_path()/'cublas64_12.dll')), 'cublasSetStream_v2', PTR, PTR)
        setter(primary.handle,s0)
        setter(secondary.handle,s1)
        dh,di,dg,dj,df,dr = upload_many(cuda,stack,(h,i,g,j,f,r))
        p,q,c = [stack.enter_context(cuda.buffer(4*n*n)) for _ in range(3)]
        first = primary.invocation('int8',shape,dh,di,p,algorithm=1)
        second_serial = primary.invocation('int8',shape,dg,dj,q,algorithm=1)
        second_parallel = secondary.invocation('int8',shape,dg,dj,q,algorithm=1)
        f_serial = primary.invocation('sgemm',fp_shape,df,dr,c)
        f_parallel = secondary.invocation('sgemm',fp_shape,df,dr,c)
        modes = {
            'INT8 single': [(s0,e0,(first,))],
            'INT8 dual serial': [(s0,e0,(first,second_serial))],
            'INT8 dual two streams': [(s0,e0,(first,)), (s1,e1,(second_parallel,))],
            'FP32 K128 single': [(s0,e0,(f_serial,))],
            'INT8 + FP32 serial': [(s0,e0,(first,f_serial))],
            'INT8 + FP32 two streams': [(s0,e0,(first,)),(s1,e1,(f_parallel,))],
            'INT8 dual + FP32 serial': [(s0,e0,(first,second_serial,f_serial))],
            'INT8 dual + FP32 two streams': [(s0,e0,(first,second_serial)),(s1,e1,(f_parallel,))],
        }
        for _ in range(15):
            measure(cuda,control,start,stop,modes['INT8 dual serial'])
        timings = {name: [] for name in modes}
        for turn in range(11):
            order = list(modes) if turn % 2 == 0 else list(reversed(modes))
            for name in order:
                timings[name].append(measure(cuda,control,start,stop,modes[name]))
        for data_left,data_right,buf in ((h,i,p),(g,j,q)):
            expected = data_left[:8].astype(np.int64) @ data_right[:8].astype(np.int64).T
            np.testing.assert_array_equal(cuda.read(buf,(n,n),np.int32)[:8,:8],expected)
        np.testing.assert_allclose(cuda.read(c,(n,n),np.float32)[:8,:8],f[:8].astype(np.float64)@r[:8].astype(np.float64).T,
                                   rtol=1e-5,atol=1e-5)
        result = dict(device=cuda.name,blas=primary.version,shape=shape,fp32_shape=fp_shape,
                      modes={name:dict(median_ms=float(np.median(t)),p95_ms=float(np.percentile(t,95)),samples_ms=t)
                             for name,t in timings.items()})
        print(cuda.name,primary.version,'verified independent INT8 and FP32 outputs',flush=True)
        for label,row in result['modes'].items():
            print(label,round(row['median_ms'],4),round(row['p95_ms'],4),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
