extern "C" __global__ void model_half_inputs(const float* input,unsigned short* out,
                                              int rows,int width,int stride,unsigned* bad) {
    const int row=blockIdx.y,col=blockIdx.x*blockDim.x+threadIdx.x;
    if (col>=stride) return;
    unsigned short value=0;
    if (rows>row && width>col) {
        const float x=input[static_cast<size_t>(row)*width+col];
        asm("cvt.rn.f16.f32 %0,%1;" : "=h"(value) : "f"(x));
        if ((value&0x7fffu)>=0x7c00u) {
            atomicOr(bad,1u);
            value=0;
        }
    }
    out[static_cast<size_t>(row)*stride+col]=value;
}

template<bool Pack>
__device__ void model_rows(const unsigned int* input,unsigned short* high,unsigned short* mixed,
                           float2* scales,unsigned short* counts,Residual* tails,unsigned* bad,
                           int k,int stride,int cap,float clip,int capacity,
                           const unsigned short* lookup,int digits) {
    const int tid=threadIdx.x,lane=tid%32,warp=tid/32,row=blockIdx.x;
    const size_t start=static_cast<size_t>(row)*(stride/2);
    __shared__ float sums[8],limit,inverse;
    __shared__ unsigned int peaks[8];
    __shared__ int active;
    float energy=0;
    unsigned int peak=0;
    for (int j=tid;j<stride/2;j+=256) {
        const unsigned int bits=input[start+j];
        if (clip>0) {
            const float x=decode_plane_half(static_cast<unsigned short>(bits));
            const float y=decode_plane_half(static_cast<unsigned short>(bits>>16));
            energy=fmaf(x,x,fmaf(y,y,energy));
        } else {
            peak=max(peak,max(bits&0x7fffu,(bits>>16)&0x7fffu));
        }
    }
    for (int step=16;step;step>>=1) {
        energy+=__shfl_down_sync(0xffffffff,energy,step);
        peak=max(peak,__shfl_down_sync(0xffffffff,peak,step));
    }
    if (!lane) {
        sums[warp]=energy;
        peaks[warp]=peak;
    }
    __syncthreads();
    if (!tid) {
        float total=0;
        peak=0;
        #pragma unroll
        for (int i=0;i<8;++i) {
            total+=sums[i];
            peak=max(peak,peaks[i]);
        }
        limit=clip>0?clip*sqrtf(total/k):decode_plane_half(static_cast<unsigned short>(peak));
        const float scale=limit>0?limit/cap:1.f;
        inverse=limit>0?digits/limit:0.f;
        scales[row]=make_float2(scale,__frcp_rn(scale));
        active=0;
    }
    __syncthreads();
    if (Pack || clip>0) {
        for (int j=tid;j<stride/2;j+=256) {
            const unsigned int bits=input[start+j];
            unsigned int a=0,b=0;
            #pragma unroll
            for (int i=0;i<2;++i) {
                const float x=decode_plane_half(static_cast<unsigned short>(bits>>(16*i)));
                if (clip>0 && fabsf(x)>limit) {
                    const int at=atomicAdd(&active,1);
                    if (at<capacity) tails[static_cast<size_t>(row)*capacity+at]=Residual{2*j+i,x-copysignf(limit,x)};
                    else atomicOr(bad,4u);
                }
                if (Pack) {
                    const int key=__float2int_rn(fminf(digits,fabsf(x)*inverse));
                    const unsigned int word=__ldg(lookup+key);
                    const int sign=x<0?-1:1;
                    a|=(255u&(sign*static_cast<signed char>(word)))<<(8*i);
                    b|=(255u&(sign*static_cast<signed char>(word>>8)))<<(8*i);
                }
            }
            if (Pack) {
                high[start+j]=static_cast<unsigned short>(a);
                mixed[start+j]=static_cast<unsigned short>(b);
            }
        }
    }
    __syncthreads();
    if (!tid && counts) counts[row]=static_cast<unsigned short>(min(active,capacity));
}

#define MODEL_ROWS(name, pack) extern "C" __global__ void name( \
    const unsigned int* input,unsigned short* high,unsigned short* mixed,float2* scales, \
    unsigned short* counts,Residual* tails,unsigned* bad,int k,int stride,int cap,float clip,int capacity, \
    const unsigned short* lookup,int digits) { \
    model_rows<pack>(input,high,mixed,scales,counts,tails,bad,k,stride,cap,clip,capacity,lookup,digits); }
MODEL_ROWS(model_pack_left,true)
MODEL_ROWS(model_measure_right,false)
