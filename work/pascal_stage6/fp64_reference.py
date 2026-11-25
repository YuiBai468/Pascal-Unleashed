"""Verify every FP32 output against FP64 cuBLAS outside the timing interval."""

import ctypes as ct
from contextlib import ExitStack

import numpy as np

from cuda import DEV_PTR, INT, PTR, SIZE, api, library_path


def verify_on_gpu(cuda,blas,module,out,da,db,a,b):
    m,k=a.shape
    n=b.shape[0]
    rows_per_call=512
    blocks=512
    alpha,beta=ct.c_double(1),ct.c_double(0)
    dgemm=api(ct.WinDLL(str(library_path()/'cublas64_12.dll')),'cublasDgemm_v2',
              PTR,INT,INT,INT,INT,INT,PTR,DEV_PTR,INT,DEV_PTR,INT,PTR,DEV_PTR,INT)
    cast=cuda.kernel(module,'cast_reference_input')[0]
    compare=cuda.kernel(module,'compare_reference_output')[0]
    totals=np.zeros(3)
    with ExitStack() as storage:
        def reserve(elements):
            return storage.enter_context(cuda.buffer(elements*8))

        av,bv,target,partials=map(reserve,(m*k,n*k,rows_per_call*n,3*blocks))
        for source,expanded,rows in ((da,av,m),(db,bv,n)):
            cuda.grid_invocation(cast,(4096,1,1),(256,1,1),(source,expanded,SIZE(rows*k)))()
        for offset in range(0,m,rows_per_call):
            rows=min(rows_per_call,m-offset)
            left=DEV_PTR(av.value+8*offset*k)
            actual=DEV_PTR(out.value+4*offset*n)
            dgemm(blas.handle,1,0,n,rows,k,ct.byref(alpha),bv,k,left,k,ct.byref(beta),target,n)
            cuda.grid_invocation(compare,(blocks,1,1),(256,1,1),(actual,target,partials,SIZE(rows*n)))()
            stats=cuda.read(partials,(blocks,3),np.float64)
            totals[:2]+=stats[:,:2].sum(axis=0)
            totals[2]=max(totals[2],float(stats[:,2].max()))
            if offset==0:
                cpu_ref=a[:2].astype(np.float64)@b[:64].astype(np.float64).T
                np.testing.assert_allclose(cuda.read(target,(2,n),np.float64)[:,:64],cpu_ref,rtol=2e-13,atol=2e-10)
    if not np.all(np.isfinite(totals)):
        raise AssertionError('Nonfinite data in the FP64 output comparison')
    return dict(full_relative_frobenius=float(np.sqrt(totals[0]/max(totals[1],1e-200))),
                max_abs_error=float(totals[2]),rmse=float(np.sqrt(totals[0]/(m*n))),certified_rows=m,
                verified_elements=m*n,reference='FP64 cuBLAS with CPU FP64 cross-check')
