"""Acceptance checks for the native SDK, release Python package and CUDA stream use."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np

from pascal_unleashed import Context, Error, F16, F32, FALLBACK, OUTLIER_GUARD, Gemm, Matrix, MODES, devices


def verify(device):
    rng=np.random.default_rng(2109)
    results={}
    with Context(device) as ctx:
        a=rng.standard_normal((256,386)).astype('f2')
        b=rng.standard_normal((386,193)).astype('f2')
        gold=a.astype('f8')@b.astype('f8')
        for mode in ('fast','balanced','precise','native','auto'):
            result,run=ctx.numpy_gemm(a,b,mode=mode)
            error=float(np.linalg.norm(result-gold)/np.linalg.norm(gold))
            assert error<.015,(mode,error)
            assert run.accelerated==(mode in ('fast','balanced','precise'))
            results[mode]={'error':error,'run':run.as_dict()}

        output,check=ctx.numpy_gemm(a,b,mode='fast',tolerance=.001)
        assert check.fallback==1 and check.reason==2
        np.testing.assert_allclose(output,gold,rtol=2e-3,atol=2e-4)
        results['tolerance_requires_native']=check.as_dict()

        spike_a,spike_b=a.copy(),b.copy()
        for col in (3,128,257):
            spike_a[:,col]*=32
            spike_b[col,:]*=32
        _,check=ctx.numpy_gemm(spike_a,spike_b,mode='precise')
        assert check.fallback==1 and check.reason==3
        results['aligned_outliers']=check.as_dict()

        special=a.copy()
        special[1,4]=np.nan
        special[3,6]=np.inf
        with np.errstate(invalid='ignore'):
            target=special.astype('f4')@b.astype('f4')
        for mode in ('fast','balanced','precise'):
            output,check=ctx.numpy_gemm(special,b,mode=mode)
            assert check.fallback==1 and check.reason==3
            np.testing.assert_array_equal(np.isnan(output),np.isnan(target))
            np.testing.assert_array_equal(np.isposinf(output),np.isposinf(target))
            np.testing.assert_array_equal(np.isneginf(output),np.isneginf(target))
        results['nonfinite_fallback']='passed on all three paths'

        zeros,check=ctx.numpy_gemm(np.zeros_like(a),b,mode='balanced')
        assert not np.count_nonzero(zeros)
        repeated=np.concatenate((a[:,:193],a[:,:193]),axis=1)
        opposing=np.concatenate((b[:193,:],-b[:193,:]),axis=0)
        zeros,check=ctx.numpy_gemm(repeated,opposing,mode='precise',tolerance=.001)
        assert check.fallback and np.abs(zeros).max()<=.001
        results['zeros_and_cancellation']=float(np.abs(zeros).max())

        # Column-major and padded matrices, batch broadcast B, and alpha/beta.
        lhs=np.ascontiguousarray(rng.standard_normal((2,32,19)).astype('f2').transpose(0,2,1))
        rhs=np.ascontiguousarray(rng.standard_normal((19,29)).astype('f2'))
        store=np.ones((2,32,37),'f4')
        with ExitStack() as memory:
            addresses=[]
            for arr in (lhs,rhs,store):
                p=memory.enter_context(ctx.memory(arr.nbytes))
                ctx.upload(p,arr.ctypes.data,arr.nbytes)
                addresses.append(p)
            args=Gemm(Matrix(addresses[0],1,32,32*19,F16),Matrix(addresses[1],1,29,0,F16),
                      Matrix(addresses[2],37,1,32*37,F32),32,29,19,2,1.5,.25,MODES['native'],FALLBACK)
            check=ctx.gemm(args)
            assert check.native==2
            ctx.download(store.ctypes.data,addresses[2],store.nbytes)
            reference=1.5*(lhs.transpose(0,2,1).astype('f4')@rhs.astype('f4'))+.25
            np.testing.assert_allclose(store[:,:,:29],reference,rtol=1e-5,atol=1e-5)
            np.testing.assert_array_equal(store[:,:,29:],1.)
        results['strides_broadcast_batched_scalars']='passed'

        for count in (3,8):
            repeated_outputs=[ctx.numpy_gemm(a,b,mode='balanced')[0] for _ in range(count)]
            for other in repeated_outputs[1:]: np.testing.assert_array_equal(other,repeated_outputs[0])
        results['repeated_calls']='11 calls with identical results'
        try:
            with ctx.memory(1<<40): raise AssertionError('Unexpected oversized allocation')
        except Error as error:
            assert error.status==3,error
        again,_=ctx.numpy_gemm(a[:2,:],b[:,:2],mode='native')
        np.testing.assert_allclose(again,gold[:2,:2],rtol=1e-5,atol=1e-5)
        results['out_of_memory_recovery']='passed'

        try: ctx.numpy_gemm(a[:2],b,mode='fast',flags=0)
        except Error as error: assert error.status==4,error
        else: raise AssertionError('Expected disabled fallback error')
        try:
            # Unsupported K takes native before launching a DP4A accumulation.
            ctx.gemm(Gemm(Matrix(2,131072,1,0,F16),Matrix(4,131072,1,0,F16),Matrix(8,64,1,0,F32),
                          64,64,131072,1,1.,0.,MODES['fast'],0))
        except Error as error: assert error.status==4,error
        else: raise AssertionError('Expected large-K protection')
        results['disabled_fallback_and_int32_guard']='passed'

    import torch
    from pascal_unleashed.torch import Engine
    with torch.cuda.device(device),torch.inference_mode():
        stream=torch.cuda.Stream(device=device)
        engine=Engine('balanced',cache=True)
        try:
            with torch.cuda.stream(stream):
                x=torch.from_numpy(a).cuda(device)
                w=torch.from_numpy(b.T.copy()).cuda(device)
                out=engine.linear(torch.nn.functional.linear,x,w)
                ref=(x.float()@w.float().T).half()
                scaled_error=((out.float()-ref.float()).norm()/ref.float().norm()).item()
                assert scaled_error<.004 and engine.metrics['accelerated']>=1
                batch=x[:32,:].reshape(2,16,386)
                assert engine.matmul(batch,w.T).shape==(2,16,193)
            stream.synchronize()
            results['pytorch_nondefault_stream']={'error':scaled_error,'accelerated':engine.metrics['accelerated']}
        finally: engine.close()
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',type=int,default=0)
    options=parser.parse_args()
    output={'device':devices()[options.device],'tests':verify(options.device)}
    options.output.parent.mkdir(parents=True,exist_ok=True)
    options.output.write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
