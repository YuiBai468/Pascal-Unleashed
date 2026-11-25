"""Run the current three Pascal configurations on denoiser linear layers."""

from contextlib import ExitStack
import ctypes as ct
import json
import weakref

import numpy as np
import torch

from pascal_engine import (
    ROOT, STAGE, Cublas, Cuda, FLOAT, INT, PTR, build, compile_ptx, line, packed_table, pointer,
)


def pad_to(n, alignment):
    return (n+alignment-1)//alignment*alignment


class FourEngine:
    def __init__(self,gpu,mode):
        self.mode=mode
        self.stack=ExitStack()
        self.cuda=self.stack.enter_context(Cuda(gpu))
        self.blas=self.stack.enter_context(Cublas())
        sources=('plane_sm61.cu','normal_slide.cu','slide_line_large.cu','clip_ar2.cu','clip_csr.cu','clip_tail_tile.cu')
        parts=[(STAGE/name).read_bytes() for name in sources]
        parts.extend((ROOT/name).read_bytes() for name in ('anima_gemm.cu','model_tail.cu'))
        self.module=self.stack.enter_context(self.cuda.load(compile_ptx(b'\n'.join(parts))))
        self.kernels={}
        self.one,self.zero=INT(1),INT(0)
        self.device=torch.device('cuda:0')
        self.cache_limit=6*2**30
        if mode=='fast':
            basis,plane=build((3,4,5),.0025,16)
            _,scalar=line((3,4),.0001,16)
            self.tables=[self.stack.enter_context(self.cuda.buffer(np.ascontiguousarray(t)))
                         for t in (basis,packed_table(plane),packed_table(scalar))]
            self.margin=128
        else:
            filename='asymmetric-stage10-3p45.npz' if mode=='middle' else 'folded-lane-0.npz'
            with np.load(STAGE/'results'/filename,allow_pickle=False) as book:
                settings=json.loads(str(book['metadata']))
                self.cap=127*settings['base']
                self.cutoff=3.45 if mode=='middle' else 3.02
                wh,wg=book['weights']
                self.ratio=float(wh/wg)
                self.weight=float(wg*(self.cap/127)**2)
                self.tables=[self.stack.enter_context(self.cuda.buffer(np.ascontiguousarray(book[side]).view(np.uint16).ravel()))
                             for side in ('a','b')]
                self.digits=len(book['a'])-1
            self.gain_a=self.cap/127
            self.gain_b=self.gain_a if mode=='middle' else self.gain_a*3.9/self.cutoff
        self.reset()

    def reset(self):
        self.cache={}
        self.cache_bytes=0
        self.weight_packs=0
        self.calls=self.ops=self.fallback_calls=self.fallback_ops=0
        self.shapes={}
        self.bad=torch.zeros(1,device=self.device,dtype=torch.int32)

    def empty(self,shape,dtype):
        return torch.empty(shape,device=self.device,dtype=dtype)

    def launch(self,name,grid,block,args,stream):
        if name not in self.kernels:
            self.kernels[name]=self.cuda.kernel(self.module,name)[0]
        self.cuda.grid_invocation(self.kernels[name],grid,block,args,stream=stream)()

    def matmul(self,left,right,raw,m,n,k,stream):
        self.blas.set_stream(self.blas.handle,stream)
        self.blas.gemm_ex(self.blas.handle,0,0,n,m,k,ct.byref(self.one),pointer(right),3,n,
                         pointer(left),3,k,ct.byref(self.zero),pointer(raw),10,n,72,1)

    def remember(self,weight,build):
        key=(weight.data_ptr(),tuple(weight.shape))
        entry=self.cache.get(key)
        if entry is not None:
            ref,values,size=entry
            if ref() is weight:
                return values
            del self.cache[key]
            self.cache_bytes-=size
        values=build()
        self.weight_packs+=1
        size=sum(t.numel()*t.element_size() for t in values)
        # The denoiser revisits weights in the same order every step. Keep the
        # admitted weights so a model larger than the cache still gets hits.
        if self.cache_bytes+size<=self.cache_limit:
            self.cache[key]=(weakref.ref(weight),values,size)
            self.cache_bytes+=size
        return values

    def fast_encode(self,values,k,dual,width,stream):
        rows=values.shape[0]
        result=self.empty((rows,width),torch.int8)
        scale=self.empty((rows,),torch.float32)
        self.launch('anima_pack_rows_edge',(rows,1,1),(256,1,1),
                    (pointer(values),pointer(result),pointer(scale),INT(k),INT(dual),INT(width),
                     *self.tables,pointer(self.bad),INT(self.margin)),stream)
        return result,scale

    def fast(self,values,weight,stream):
        m,k=values.shape
        n=weight.shape[0]
        pairs=(k+1)//2
        dual=round(pairs*2560/8192)
        width=pad_to(3*pairs+dual,4)
        left,sa=self.fast_encode(values,k,dual,width,stream)

        def pack_weight():
            right,sb=self.fast_encode(weight.contiguous(),k,dual,width,stream)
            return right.T.contiguous(),sb

        right,sb=self.remember(weight,pack_weight)
        raw=self.empty((m,n),torch.int32)
        result=self.empty((m,n),torch.float32)
        self.matmul(left,right,raw,m,n,width,stream)
        self.launch('anima_reconstruct',((n+127)//128,(m+15)//16,1),(32,4,1),
                    (*map(pointer,(raw,sa,sb,None,result)),INT(m),INT(n)),stream)
        return result

    def half_inputs(self,values,rows,k,stride,stream):
        result=self.empty((rows,stride),torch.int16)
        self.launch('model_half_inputs',((stride+255)//256,rows,1),(256,1,1),
                    (pointer(values),pointer(result),INT(values.shape[0]),INT(k),INT(stride),pointer(self.bad)),stream)
        return result

    def tail_packet(self,rows,k,capacity,half,scale,stream,side,high=None,mixed=None):
        clipped=side==0 or self.mode=='precise'
        cutoff=self.cutoff if clipped else 0.
        count=self.empty((rows,),torch.int16) if clipped else None
        tails=self.empty((rows,capacity,2),torch.int32) if clipped else None
        name='model_pack_left' if side==0 else 'model_measure_right'
        self.launch(name,(rows,1,1),(256,1,1),
                    (*map(pointer,(half,high,mixed,scale,count,tails,self.bad)),INT(k),INT(half.shape[1]),
                     INT(self.cap),FLOAT(cutoff),INT(capacity),self.tables[side],INT(self.digits)),stream)
        if not clipped:
            return None,None,None
        packets=self.empty((rows,capacity//4,4),torch.int32)
        weights=self.empty((rows,),torch.float32)
        kernel='pack_tail_one_byte' if self.mode=='middle' else 'pack_tail_two_bytes'
        self.launch(kernel,(rows,1,1),(32,1,1),(*map(pointer,(tails,count,packets,weights)),INT(capacity)),stream)
        return count,packets,weights

    def two_lanes(self,values,weight,stream):
        m0,k=values.shape
        n0=weight.shape[0]
        m,n,stride=pad_to(m0,128),pad_to(n0,128),pad_to(k,64)
        # An RMS threshold limits the number of active tails by K / cutoff^2.
        capacity=pad_to(int(k/(self.cutoff*self.cutoff))+5,4)
        left=self.half_inputs(values,m,k,stride,stream)
        ah,ag=[self.empty((m,stride),torch.int8) for _ in range(2)]
        sa=self.empty((m,2),torch.float32)
        ca,ra,ra_scales=self.tail_packet(m,k,capacity,left,sa,stream,0,ah,ag)

        def pack_weight():
            right=self.half_inputs(weight.contiguous(),n,k,stride,stream)
            sb=self.empty((n,2),torch.float32)
            cb,rb,rb_scales=self.tail_packet(n,k,capacity,right,sb,stream,1)
            bh,bg,b_coarse=[self.empty((n,stride),torch.int8) for _ in range(3)]
            self.launch('transpose_trained_64',(stride//64,n//64,1),(256,1,1),
                        (*map(pointer,(right,bh,bg,b_coarse,sb)),self.tables[1],INT(n),INT(stride),
                         INT(self.digits),FLOAT(self.digits/self.cap),FLOAT(1/self.gain_b),INT(128)),stream)
            return (bh,bg,b_coarse,sb,cb,rb,rb_scales) if self.mode=='precise' else (bh,bg,b_coarse,sb)

        packed=self.remember(weight,pack_weight)
        bh,bg,b_coarse,sb=packed[:4]
        if self.mode=='precise':
            cb,rb,rb_scales=packed[4:]
            a_coarse=self.empty((m,stride),torch.int8)
            self.launch('transpose_coarse_64',(stride//64,m//64,1),(256,1,1),
                        (*map(pointer,(left,a_coarse,sa)),INT(m),INT(stride),FLOAT(1/self.gain_a),INT(128)),stream)
        raw_h,raw_g=[self.empty((m,n),torch.int32) for _ in range(2)]
        result=self.empty((m,n),torch.float32)
        self.matmul(ah,bh,raw_h,m,n,stride,stream)
        self.matmul(ag,bg,raw_g,m,n,stride,stream)
        common=(*map(pointer,(raw_h,raw_g,sa,sb,ca,ra,ra_scales,b_coarse,result)),INT(m),INT(n),INT(stride),
                INT(capacity),FLOAT(self.ratio),FLOAT(self.weight),FLOAT(self.gain_b))
        if self.mode=='middle':
            self.launch('tail_direction128_left_q8_a',(n//128,m//32,1),(256,1,1),common,stream)
        else:
            partial=self.empty((m,n),torch.int16)
            self.launch('tail_direction128_compact_b',(n//32,m//128,1),(256,1,1),
                        (*map(pointer,(sa,cb,rb,rb_scales,a_coarse,partial)),INT(n),INT(stride),INT(capacity),
                         FLOAT(self.gain_a)),stream)
            self.launch('tail_direction128_compact_a',(n//128,m//32,1),(256,1,1),
                        (*common,pointer(partial),pointer(rb_scales),FLOAT(self.gain_a)),stream)
        return result[:m0,:n0]

    def linear(self,original,values,weight,bias=None):
        if not (values.is_cuda and weight.is_cuda and values.dtype==weight.dtype==torch.float32 and weight.ndim==2):
            return original(values,weight,bias)
        n,k=weight.shape
        if not k:
            return original(values,weight,bias)
        m=values.numel()//k
        ops=2*m*n*k
        if m<16 or n<64 or k<32 or (self.mode=='fast' and n%4):
            self.fallback_calls+=1
            self.fallback_ops+=ops
            return original(values,weight,bias)
        stream=PTR(torch.cuda.current_stream(values.device).cuda_stream)
        matrix=values.reshape(m,k).contiguous()
        result=self.fast(matrix,weight,stream) if self.mode=='fast' else self.two_lanes(matrix,weight,stream)
        self.calls+=1
        self.ops+=ops
        shape=(m,n,k)
        self.shapes[shape]=self.shapes.get(shape,0)+1
        if bias is not None:
            result=result+bias
        return result.reshape(*values.shape[:-1],n)

    def report(self):
        if flag:=self.bad.item():
            raise RuntimeError(f'Invalid FP16 conversion or tail packing (flags {flag})')
        return dict(mode=self.mode,int8_linear_calls=self.calls,int8_linear_flops=self.ops,
                    fp32_linear_calls=self.fallback_calls,fp32_linear_flops=self.fallback_ops,
                    weight_packs=self.weight_packs,weight_cache_bytes=self.cache_bytes,
                    shapes=[dict(shape=shape,calls=count) for shape,count in sorted(self.shapes.items())])

    def self_check(self):
        with torch.inference_mode():
            rng=torch.Generator(device=self.device).manual_seed(781)
            actual=torch.randn((96,320),generator=rng,device=self.device)
            weights=torch.randn((112,320),generator=rng,device=self.device)
            fp32=torch.nn.functional.linear
            ref=fp32(actual.half().float(),weights.half().float())
            test=self.linear(fp32,actual,weights)
            error=((test-ref).norm()/ref.norm()).item()
            if error>=.01 or self.bad.item():
                raise RuntimeError(f'{self.mode} CUDA linear check failed, relative error {error:.6%}')
            if self.mode=='fast':
                big=torch.randn((16,8192),generator=rng,device=self.device)
                stream=PTR(torch.cuda.current_stream(self.device).cuda_stream)
                encoded,scale=self.fast_encode(big,8192,1280,13568,stream)
                expected=torch.empty_like(encoded)
                expected_scale=torch.empty_like(scale)
                half=big.half()
                self.launch('edge_line_2560',(16,1,1),(512,1,1),
                            (*map(pointer,(half,expected,expected_scale)),*self.tables,pointer(self.bad),
                             INT(128),FLOAT(.81),FLOAT(.89)),stream)
                torch.testing.assert_close(encoded,expected,rtol=0,atol=0)
                torch.testing.assert_close(scale,expected_scale,rtol=0,atol=0)
            print(f'{self.mode}: dynamic CUDA linear checked against FP16 inputs with FP32 accumulation ({error:.4%})',flush=True)
            self.reset()
