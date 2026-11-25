import argparse
import json
from contextlib import ExitStack

import numpy as np

from bench import SOURCE_DIR
from cuda import Cuda, INT, compile_ptx


def pack(x, bits):
    out = np.zeros(x.shape[:-1], np.uint32)
    for i in range(x.shape[-1]):
        out |= (x[..., i].astype(np.uint32) & ((1 << bits) - 1)) << (i * bits)
    return out.view(np.int32)


def fixture(mode, count, loops):
    rng = np.random.default_rng(802)
    bound = 1 if mode.startswith(('packed', 'ternary')) else 32
    x = rng.integers(-bound, bound+1, (count, 4, 4), dtype=np.int32)
    output = np.zeros((count, 4), np.int32)
    pairs = ((0,1),(2,3),(0,3),(2,1))
    if mode in ('dense_dp4a','ternary_dp4a','dense_dp2a'):
        inner = 2 if mode == 'dense_dp2a' else 4
        words = pack(x, 8)
        if inner == 2:
            words[:, ::2] = pack(x[:, ::2, :2], 16)
        for i,(a,b) in enumerate(pairs):
            output[:,i] = (x[:,a,:inner]*x[:,b,:inner]).sum(1)*7*loops
        return words, output, 7*4*inner
    if mode in ('packed_dp4a','packed_dp2a'):
        two = mode.endswith('4a')
        inner, levels, chunk, base = (4,2,7,64) if two else (2,3,30,128)
        a = rng.integers(-1,2,(count,2,inner,levels),dtype=np.int32)
        b = rng.integers(-1,2,(count,2,inner),dtype=np.int32)
        words = np.empty((count,4),np.int32)
        words[:,::2] = pack(a @ (base**np.arange(levels)), 8 if two else 16)
        words[:,1::2] = pack(b,8)
        output[:,:levels] = (a.sum(1)*b.sum(1)[...,None]).sum(1)*chunk*loops
        return words, output, inner*levels*chunk*4
    if mode == 'dense_fma':
        values = x[:,:,0].copy().astype(np.float32)/32
        order = ((0,1),(2,3),(2,1),(0,3))
        expected = np.column_stack([values[:,a]*values[:,b]*(28*loops) for a,b in order])
        return values.copy().view(np.int32), expected.astype(np.float32).view(np.int32), 112
    x = rng.integers(-1,2,(count,2,32),dtype=np.int32)
    if mode == 'binary_pop':
        y = rng.integers(0,2,(count,4,32),dtype=np.int32)*2-1
        output[:,0] = (y[:,0]*y[:,1]).sum(1)*28*loops
        output[:,1] = (y[:,2]*y[:,3]).sum(1)*28*loops
        return pack(y>0,1), output//28, 2*32
    masks = np.stack((x[:,0]>0,x[:,0]<0,x[:,1]>0,x[:,1]<0),axis=1)
    output[:,0] = (x[:,0]*x[:,1]).sum(1)*28*loops
    return pack(masks,1),output//28,32


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu', required=True)
    args=parser.parse_args()
    count,loops=16384,64
    ptx=compile_ptx((SOURCE_DIR/'micro.cu').read_bytes())
    names=('dense_dp4a','dense_dp2a','dense_fma','ternary_dp4a','binary_pop','ternary_pop','packed_dp4a','packed_dp2a')
    rows=[]
    with Cuda(args.gpu) as cuda,cuda.load(ptx) as module:
        print('GPU',cuda.name,flush=True)
        for name in names:
            words,parts,work=fixture(name,count*loops,1)
            if name == 'dense_fma':
                expected = parts.view(np.float32).reshape(loops,count,4).sum(0).view(np.int32)
            else:
                expected = parts.reshape(loops,count,4).sum(0,dtype=np.int32)
            with ExitStack() as buffers:
                inp=buffers.enter_context(cuda.buffer(words))
                out=buffers.enter_context(cuda.buffer(expected.nbytes))
                kernel,resources=cuda.kernel(module,name)
                launch=cuda.grid_invocation(kernel,(count//128,1,1),(128,1,1),(inp,out,INT(loops)))
                launch()
                np.testing.assert_array_equal(cuda.read(out,expected.shape,np.int32),expected)
                ms=cuda.measure(launch,5,9)
                rate=2*work*count*loops/np.median(ms)/1e9
                rows.append(dict(kernel=name,median_ms=float(np.median(ms)),p95_ms=float(np.percentile(ms,95)),
                                 equivalent_Tops=rate,timings_ms=ms,resources=resources))
                print(name, f'{np.median(ms):.4f} ms', f'{rate:.2f} effective TOPS', resources,flush=True)
    (SOURCE_DIR/'results').mkdir(exist_ok=True)
    (SOURCE_DIR/'results'/'micro.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
