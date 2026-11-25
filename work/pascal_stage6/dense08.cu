// Peak-based encoding for the dense two-byte paths. Raw FP16 magnitude bits are monotonic.
__device__ __forceinline__ unsigned int peak_half2(unsigned int value) {
    return max(value&0x7fffu,(value>>16)&0x7fffu);
}

template<int Threads>
__device__ float2 dense_row_scale(unsigned int peak,int cap,unsigned int* bad) {
    __shared__ unsigned int warps[Threads/32];
    __shared__ float2 scale;
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    for (int d=16;d;d>>=1) peak=max(peak,__shfl_down_sync(0xffffffff,peak,d));
    if (!lane) warps[warp]=peak;
    __syncthreads();
    if (!warp) {
        peak=lane<Threads/32?warps[lane]:0;
        for (int d=Threads/64;d;d>>=1) peak=max(peak,__shfl_down_sync(0xffffffff,peak,d));
        if (!lane) {
            if (peak>=0x7c00u) atomicOr(bad,1u);
            float value=peak<0x7c00u?decode_plane_half(static_cast<unsigned short>(peak)):0.f;
            const float step=value>0.f?value/cap:1.f;
            scale=make_float2(step,value>0.f?__frcp_rn(step):0.f);
        }
    }
    __syncthreads();
    return scale;
}

template<bool OneProduct>
__device__ __forceinline__ unsigned int dense_digits(unsigned short input,float inv,int cap,
                                                     const unsigned short* lut) {
    const float v=decode_plane_half(input);
    if (OneProduct) {
        const int index=__float2int_rn(fminf(cap,fabsf(v)*inv));
        const unsigned int digits=__ldg(lut+index);
        if (!(input&0x8000u)) return digits;
        return ((0u-digits)&255u)|(((0u-(digits>>8))&255u)<<8);
    }
    const int q=max(-cap,min(cap,__float2int_rn(v*inv)));
    int h=__float2int_rn(q*(1.0f/15.0f));
    int g=q-14*h;
    const int shift=(g>127)-(g<-127);
    h+=shift;
    g-=14*shift;
    return (h&255u)|((g&255u)<<8);
}

extern "C" __global__ void dense08_peaks(const unsigned int* src,float2* scales,unsigned int* bad,int k,int cap) {
    unsigned int peak=0;
    const int stride=k/2;
    const size_t row=static_cast<size_t>(blockIdx.x)*stride;
    for (int i=threadIdx.x;i<stride;i+=256) peak=max(peak,peak_half2(src[row+i]));
    const float2 scale=dense_row_scale<256>(peak,cap,bad);
    if (!threadIdx.x) scales[blockIdx.x]=scale;
}

template<int K, bool OneProduct>
__device__ void dense08_pack_left(const unsigned int* input,unsigned short* left,unsigned short* other,
                                  float2* scales,unsigned int* bad,int cap,const unsigned short* lut) {
    constexpr int threads=512, stride=K/2, count=stride/threads;
    const size_t row=static_cast<size_t>(blockIdx.x)*stride;
    const int tid=threadIdx.x;
    unsigned int values[count];
    unsigned int peak=0;
    #pragma unroll
    for (int i=0;i<count;++i) {
        const unsigned int bits=input[row+tid+i*threads];
        values[i]=bits;
        peak=max(peak,peak_half2(bits));
    }
    const float2 scale=dense_row_scale<threads>(peak,cap,bad);
    if (!tid) scales[blockIdx.x]=scale;
    #pragma unroll
    for (int i=0;i<count;++i) {
        const unsigned int bits=values[i];
        const unsigned int a=dense_digits<OneProduct>(static_cast<unsigned short>(bits),scale.y,cap,lut);
        const unsigned int b=dense_digits<OneProduct>(static_cast<unsigned short>(bits>>16),scale.y,cap,lut);
        const size_t at=row+tid+i*threads;
        if (OneProduct) {
            reinterpret_cast<unsigned int*>(left)[at]=a|(b<<16);
        } else {
            left[at]=static_cast<unsigned short>((a&255u)|((b&255u)<<8));
            other[at]=static_cast<unsigned short>((a>>8)|(b&0xff00u));
        }
    }
}

template<bool OneProduct>
__device__ void dense08_pack_right(const unsigned short* input,signed char* left,signed char* other,
                                    const float2* scales,int k,int n,int cap,const unsigned short* lut) {
    __shared__ unsigned short digits[64][66];
    const int row0=blockIdx.y*64,k0=blockIdx.x*64;
    const int lane=threadIdx.x&63,group=threadIdx.x>>6;
    #pragma unroll
    for (int i=group;i<64;i+=4) {
        const int r=row0+i;
        const float inv=__ldg(&scales[r].y);
        const unsigned short value=input[static_cast<size_t>(r)*k+k0+lane];
        digits[i][lane]=static_cast<unsigned short>(dense_digits<OneProduct>(value,inv,cap,lut));
    }
    __syncthreads();
    #pragma unroll
    for (int i=group;i<64;i+=4) {
        const unsigned short value=digits[lane][i];
        const size_t at=static_cast<size_t>(k0+i)*n;
        const int r=row0+lane;
        if (OneProduct) {
            left[2*at+r]=static_cast<signed char>(value);
            left[2*at+n+r]=static_cast<signed char>(value>>8);
        } else {
            left[at+r]=static_cast<signed char>(value);
            other[at+r]=static_cast<signed char>(value>>8);
        }
    }
}

template<bool OneProduct>
__device__ void dense08_write(const int* high,const int* other,const float2* a,const float2* b,
                              float* output,int n,float weight) {
    const int col=(blockIdx.x*32+threadIdx.x)*4;
    const float4 sb=make_float4(__ldg(&b[col].x),__ldg(&b[col+1].x),__ldg(&b[col+2].x),__ldg(&b[col+3].x));
    #pragma unroll
    for (int i=0;i<4;++i) {
        const int row=blockIdx.y*16+threadIdx.y+4*i;
        const size_t at=static_cast<size_t>(row)*n+col;
        const float scale=__ldg(&a[row].x)*weight;
        const int4 h=*reinterpret_cast<const int4*>(high+at);
        float4 v=make_float4(h.x,h.y,h.z,h.w);
        if (!OneProduct) {
            const int4 g=*reinterpret_cast<const int4*>(other+at);
            v=make_float4(fmaf(14.f,v.x,g.x),fmaf(14.f,v.y,g.y),fmaf(14.f,v.z,g.z),fmaf(14.f,v.w,g.w));
        }
        *reinterpret_cast<float4*>(output+at)=make_float4((scale*sb.x)*v.x,(scale*sb.y)*v.y,
                                                         (scale*sb.z)*v.z,(scale*sb.w)*v.w);
    }
}

#define DENSE_LEFT(name,k,one) extern "C" __global__ __launch_bounds__(512) void name( \
    const unsigned int* in,unsigned short* high,unsigned short* other,float2* scales,unsigned int* bad,int cap, \
    const unsigned short* table) { dense08_pack_left<k,one>(in,high,other,scales,bad,cap,table); }
DENSE_LEFT(dense08_ar2_8192,8192,false)
DENSE_LEFT(dense08_ar2_16384,16384,false)
DENSE_LEFT(dense08_ar2_32768,32768,false)
DENSE_LEFT(dense08_one_8192,8192,true)
DENSE_LEFT(dense08_one_16384,16384,true)
DENSE_LEFT(dense08_one_32768,32768,true)

#define DENSE_RIGHT(name,one) extern "C" __global__ void name( \
    const unsigned short* in,signed char* high,signed char* other,const float2* scales, \
    int k,int n,int cap,const unsigned short* table) { dense08_pack_right<one>(in,high,other,scales,k,n,cap,table); }
DENSE_RIGHT(dense08_ar2_right,false)
DENSE_RIGHT(dense08_one_right,true)

#define DENSE_WRITE(name,one) extern "C" __global__ void name( \
    const int* high,const int* other,const float2* a,const float2* b,float* out,int n,float weight) { \
    dense08_write<one>(high,other,a,b,out,n,weight); }
DENSE_WRITE(dense08_ar2_write,false)
DENSE_WRITE(dense08_one_write,true)
