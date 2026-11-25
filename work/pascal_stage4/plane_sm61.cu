__device__ __forceinline__ float decode_plane_half(unsigned short raw) {
    float x;
    asm("cvt.f32.f16 %0, %1;" : "=f"(x) : "h"(raw));
    return x;
}

__device__ __forceinline__ float3 project_pair(unsigned int data,const float* p){
    const float x=decode_plane_half(static_cast<unsigned short>(data));
    const float y=decode_plane_half(static_cast<unsigned short>(data>>16));
    return make_float3(fmaf(x,p[0],y*p[1]),fmaf(x,p[2],y*p[3]),fmaf(x,p[4],y*p[5]));
}

__device__ __forceinline__ int lookup_bin(float x,int base,int bins){
    return max(0,min(bins-1,__float2int_rd((x-static_cast<float>(base)+.5f)*bins)));
}

extern "C" __global__ void plane_pack(const unsigned int* input,unsigned int* packed,float* scales,
                                       const float* basis,const unsigned short* lut,
                                       unsigned int* invalid,int pairs,int bins,int margin){
    __shared__ unsigned int raw[2048];
    __shared__ float warp_peak[8], inverse;
    const int row=blockIdx.x,t=threadIdx.x;
    float peak=0;
    for(int i=t;i<pairs;i+=256){
        unsigned int v=input[row*pairs+i];
        if((v&0x7fff)>=0x7c00){atomicOr(invalid,1u);v&=0xffff0000u;}
        if(((v>>16)&0x7fff)>=0x7c00){atomicOr(invalid,1u);v&=0x0000ffffu;}
        raw[i]=v;
        const float3 z=project_pair(v,basis);
        peak=fmaxf(peak,fmaxf(fabsf(z.x),fmaxf(fabsf(z.y),fabsf(z.z))));
    }
    for(int d=16;d>0;d>>=1)peak=fmaxf(peak,__shfl_down_sync(0xffffffff,peak,d));
    if(t%32==0)warp_peak[t/32]=peak;
    __syncthreads();
    if(t<32){
        peak=t<8?warp_peak[t]:0;
        for(int d=4;d>0;d>>=1)peak=fmaxf(peak,__shfl_down_sync(0xffffffff,peak,d));
        if(t==0){
            const float scale=peak>0?peak/margin:1.0f;
            scales[row]=scale;
            inverse=__frcp_rn(scale);
        }
    }
    __syncthreads();
    const int groups=pairs/4;
    for(int g=t;g<groups;g+=256){
        unsigned int bits0=0,bits1=0,bits2=0;
        #pragma unroll
        for(int j=0;j<4;++j){
            const float3 pos=project_pair(raw[4*g+j],basis);
            const float x=pos.x*inverse,y=pos.y*inverse,z=pos.z*inverse;
            const int a=__float2int_rn(x), b=__float2int_rn(y), c=__float2int_rn(z);
            const int i=lookup_bin(x,a,bins),k=lookup_bin(y,b,bins), l=lookup_bin(z,c,bins);
            const unsigned int shift=__ldg(lut+(i*bins+k)*bins+l);
            const int p0=a+static_cast<int>(shift&15u)-8;
            const int p1=b+static_cast<int>((shift>>4)&15u)-8;
            const int p2=c+static_cast<int>((shift>>8)&15u)-8;
            if(p0<-128||p0>127||p1<-128||p1>127||p2<-128||p2>127)atomicOr(invalid,2u);
            bits0|=(static_cast<unsigned int>(p0)&255u)<<(8*j);
            bits1|=(static_cast<unsigned int>(p1)&255u)<<(8*j);
            bits2|=(static_cast<unsigned int>(p2)&255u)<<(8*j);
        }
        const int offset=3*(row*groups+g);
        packed[offset]=bits0; packed[offset+1]=bits1; packed[offset+2]=bits2;
    }
}

extern "C" __global__ void unpack_plane(const int* dot,const float* a,const float* b,
                                         float* output,int m,int n){
    const int j=4*(blockIdx.x*32+threadIdx.x);
    if(j>=n)return;
    const float4 right=*reinterpret_cast<const float4*>(b+j);
    #pragma unroll
    for(int s=0;s<4;++s){
        const int i=blockIdx.y*16+threadIdx.y+4*s;
        if(i<m){
            const float v=a[i];
            const int pos=i*n+j;
            const int4 z=*reinterpret_cast<const int4*>(dot+pos);
            *reinterpret_cast<float4*>(output+pos)=make_float4(z.x*(v*right.x),z.y*(v*right.y),
                                                               z.z*(v*right.z),z.w*(v*right.w));
        }
    }
}

template<bool Stream>
__device__ void unpack_plane_row(const int4* dot, const float* a, const float4* b,
                                  float4* out, int n) {
    const int j = blockIdx.x * 256 + threadIdx.x;
    const int index = blockIdx.y * (n / 4) + j;
    const float scale = __ldg(a + blockIdx.y);
    const float4 right = __ldg(b + j);
    int4 v;
    if (Stream) {
        asm("ld.global.cs.v4.s32 {%0,%1,%2,%3}, [%4];" :
            "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(dot + index));
    } else {
        v = dot[index];
    }
    const float4 z = make_float4(v.x * (scale * right.x), v.y * (scale * right.y),
                                 v.z * (scale * right.z), v.w * (scale * right.w));
    if (Stream) {
        asm volatile("st.global.cs.v4.f32 [%0], {%1,%2,%3,%4};" ::
            "l"(out + index), "f"(z.x), "f"(z.y), "f"(z.z), "f"(z.w) : "memory");
    } else {
        out[index] = z;
    }
}

#define PLANE_EPILOGUE(name, stream) extern "C" __global__ void name( \
    const int4* dot,const float* a,const float4* b,float4* out,int m,int n) { \
    unpack_plane_row<stream>(dot, a, b, out, n); }
PLANE_EPILOGUE(unpack_plane_linear, false)
PLANE_EPILOGUE(unpack_plane_stream, true)

template<bool SharedTable, bool Projected>
__device__ void pack_plane_binned16(const unsigned int* input,unsigned int* packed,float* scales,
                                    const float* basis,const unsigned short* lut,unsigned int* invalid,
                                    int pairs,int margin) {
    __shared__ unsigned short table[SharedTable?4096:1];
    __shared__ unsigned int raw[Projected?1:2048];
    __shared__ float xs[Projected?2048:1],ys[Projected?2048:1],zs[Projected?2048:1];
    __shared__ float warp_peak[8],factor;
    const int row=blockIdx.x,t=threadIdx.x;
    if(SharedTable)for(int i=t;i<4096;i+=256)table[i]=__ldg(lut+i);
    float peak=0;
    for(int i=t;i<pairs;i+=256){
        unsigned int v=input[row*pairs+i];
        if((v&0x7fff)>=0x7c00){atomicOr(invalid,1u);v&=0xffff0000u;}
        if(((v>>16)&0x7fff)>=0x7c00){atomicOr(invalid,1u);v&=0x0000ffffu;}
        const float3 p=project_pair(v,basis);
        if(Projected){ xs[i]=p.x;ys[i]=p.y;zs[i]=p.z; }
        else raw[i]=v;
        peak=fmaxf(peak,fmaxf(fabsf(p.x),fmaxf(fabsf(p.y),fabsf(p.z))));
    }
    for(int d=16;d>0;d>>=1)peak=fmaxf(peak,__shfl_down_sync(0xffffffff,peak,d));
    if(t%32==0)warp_peak[t/32]=peak;
    __syncthreads();
    if(t<32){
        peak=t<8?warp_peak[t]:0;
        for(int d=4;d>0;d>>=1)peak=fmaxf(peak,__shfl_down_sync(0xffffffff,peak,d));
        if(t==0){
            const float scale=peak>0?peak/margin:1.0f;
            scales[row]=scale;
            factor=16*__frcp_rn(scale);
        }
    }
    __syncthreads();
    const int groups=pairs/4;
    for(int g=t;g<groups;g+=256){
        unsigned int bits0=0,bits1=0,bits2=0;
        #pragma unroll
        for(int j=0;j<4;++j){
            const int i=4*g+j;
            const float3 p=Projected?make_float3(xs[i],ys[i],zs[i]):project_pair(raw[i],basis);
            const int x=__float2int_rd(fmaf(p.x,factor,8.0f));
            const int y=__float2int_rd(fmaf(p.y,factor,8.0f));
            const int z=__float2int_rd(fmaf(p.z,factor,8.0f));
            const int key=((x&15)<<8)|((y&15)<<4)|(z&15);
            const unsigned int shift=SharedTable?table[key]:__ldg(lut+key);
            const int p0=(x>>4)+static_cast<int>(shift&15u)-8;
            const int p1=(y>>4)+static_cast<int>((shift>>4)&15u)-8;
            const int p2=(z>>4)+static_cast<int>((shift>>8)&15u)-8;
            if(p0<-128||p0>127||p1<-128||p1>127||p2<-128||p2>127)atomicOr(invalid,2u);
            bits0|=(static_cast<unsigned int>(p0)&255u)<<(8*j);
            bits1|=(static_cast<unsigned int>(p1)&255u)<<(8*j);
            bits2|=(static_cast<unsigned int>(p2)&255u)<<(8*j);
        }
        const int offset=row*3*groups+g;
        packed[offset]=bits0; packed[offset+groups]=bits1; packed[offset+2*groups]=bits2;
    }
}

#define PLANE_OPT(name,shared_lut,projected) extern "C" __global__ void name( \
    const unsigned int* input,unsigned int* packed,float* scales,const float* basis,const unsigned short* lut, \
    unsigned int* invalid,int pairs,int bins,int margin) { \
    pack_plane_binned16<shared_lut,projected>(input,packed,scales,basis,lut,invalid,pairs,margin); }
PLANE_OPT(plane_fast,false,false)
PLANE_OPT(plane_shared,true,false)
PLANE_OPT(plane_projected,false,true)
PLANE_OPT(plane_shared_projected,true,true)

__device__ __forceinline__ void pack_line(float x, float y, float factor,
                                          const unsigned short* lut, signed char* dest,
                                          int stride) {
    const int a = __float2int_rd(fmaf(x, factor, 8.0f));
    const int b = __float2int_rd(fmaf(y, factor, 8.0f));
    const unsigned shift = __ldg(lut + (((a & 15) << 4) | (b & 15)));
    dest[0] = (a >> 4) + (shift & 15) - 8;
    dest[stride] = (b >> 4) + ((shift >> 4) & 15) - 8;
}

template<bool CacheTable, bool RegisterCoordinates, int DualPairs = 0>
__device__ void pack_plane_4096(const unsigned int* input, signed char* packed, float* scales,
                                const float* basis, const unsigned short* lut,
                                unsigned int* invalid, int margin) {
    constexpr int pairs = 2048;
    constexpr int threads = 256;
    constexpr int count = pairs / threads;
    constexpr int dual_pairs = DualPairs;
    constexpr int plane_pairs = pairs - dual_pairs;
    constexpr int width = 3 * plane_pairs + 4 * dual_pairs;
    const int row = blockIdx.x, lane = threadIdx.x;
    __shared__ unsigned short table[CacheTable ? 4096 : 1];
    __shared__ float shared_x[RegisterCoordinates ? 1 : pairs];
    __shared__ float shared_y[RegisterCoordinates ? 1 : pairs];
    __shared__ float shared_z[RegisterCoordinates ? 1 : pairs];
    __shared__ float warp_peak[8], shared_factor;
    const float x0 = basis[0], y0 = basis[1], x1 = basis[2];
    const float y1 = basis[3], x2 = basis[4], y2 = basis[5];
    const float lx = dual_pairs ? basis[6] : 0.0f;
    const float ly = dual_pairs ? basis[7] : 0.0f;
    float4 coordinates[count];
    float peak = 0;
    if (CacheTable) {
        for (int i = lane; i < 4096; i += threads) table[i] = __ldg(lut + i);
    }
    #pragma unroll
    for (int i = 0; i < count; ++i) {
        const int pair = i * threads + lane;
        unsigned v = input[row * pairs + pair];
        if (((v & 0x7fff) >= 0x7c00) || (((v >> 16) & 0x7fff) >= 0x7c00)) {
            atomicOr(invalid, 1u);
            v = 0;
        }
        const float x = decode_plane_half(static_cast<unsigned short>(v));
        const float y = decode_plane_half(static_cast<unsigned short>(v >> 16));
        const bool linear = i < dual_pairs / threads ||
            (i == dual_pairs / threads && lane < dual_pairs % threads);
        const float4 p = linear ? make_float4(x*lx,x*ly,y*lx,y*ly) :
            make_float4(fmaf(x,x0,y*y0), fmaf(x,x1,y*y1), fmaf(x,x2,y*y2), 0.0f);
        if (RegisterCoordinates) coordinates[i] = p;
        else { shared_x[pair] = p.x; shared_y[pair] = p.y; shared_z[pair] = p.z; }
        peak = fmaxf(peak, fmaxf(fmaxf(fabsf(p.x), fabsf(p.y)), fmaxf(fabsf(p.z), fabsf(p.w))));
    }
    for (int d = 16; d > 0; d >>= 1) peak = fmaxf(peak, __shfl_down_sync(0xffffffff, peak, d));
    if ((lane & 31) == 0) warp_peak[lane >> 5] = peak;
    __syncthreads();
    if (lane < 32) {
        peak = lane < 8 ? warp_peak[lane] : 0;
        for (int d = 4; d > 0; d >>= 1) peak = fmaxf(peak, __shfl_down_sync(0xffffffff, peak, d));
        if (lane == 0) {
            const float scale = peak > 0 ? peak / margin : 1.0f;
            scales[row] = scale;
            shared_factor = 16 * __frcp_rn(scale);
        }
    }
    __syncthreads();
    const float factor = shared_factor;
    signed char* output = packed + row * width;
    #pragma unroll
    for (int i = 0; i < count; ++i) {
        const int pair = i * threads + lane;
        const float4 p = RegisterCoordinates ? coordinates[i] :
            make_float4(shared_x[pair], shared_y[pair], shared_z[pair], 0.0f);
        const bool linear = i < dual_pairs / threads ||
            (i == dual_pairs / threads && lane < dual_pairs % threads);
        if (linear) {
            pack_line(p.x, p.y, factor, lut + 4096, output + pair, dual_pairs);
            pack_line(p.z, p.w, factor, lut + 4096, output + pair + 2 * dual_pairs, dual_pairs);
            continue;
        }
        const int x = __float2int_rd(fmaf(p.x, factor, 8.0f));
        const int y = __float2int_rd(fmaf(p.y, factor, 8.0f));
        const int z = __float2int_rd(fmaf(p.z, factor, 8.0f));
        const int key = ((x & 15) << 8) | ((y & 15) << 4) | (z & 15);
        const unsigned shift = CacheTable ? table[key] : __ldg(lut + key);
        const int index = 4 * dual_pairs + pair - dual_pairs;
        output[index] = (x >> 4) + (shift & 15) - 8;
        output[index + plane_pairs] = (y >> 4) + ((shift >> 4) & 15) - 8;
        output[index + 2 * plane_pairs] = (z >> 4) + ((shift >> 8) & 15) - 8;
    }
}

#define PLANE_BYTES(name, cache, registers) extern "C" __global__ void name( \
    const unsigned int* input, signed char* packed, float* scales, const float* basis, \
    const unsigned short* lut, unsigned int* invalid, int pairs, int bins, int margin) { \
    pack_plane_4096<cache, registers>(input, packed, scales, basis, lut, invalid, margin); }
PLANE_BYTES(plane_bytes_registers, false, true)
PLANE_BYTES(plane_bytes_shared, false, false)
PLANE_BYTES(plane_bytes_cached, true, true)

#define PLANE_MIX(name, dual_pairs) extern "C" __global__ void name( \
    const unsigned int* input, signed char* packed, float* scales, const float* basis, \
    const unsigned short* lut, unsigned int* invalid, int pairs, int bins, int margin) { \
    pack_plane_4096<false, true, dual_pairs>(input, packed, scales, basis, lut, invalid, margin); }
PLANE_MIX(plane_mixed_512, 256)
PLANE_MIX(plane_mixed_1024, 512)
PLANE_MIX(plane_mixed_1152, 576)
PLANE_MIX(plane_mixed_1280, 640)
PLANE_MIX(plane_mixed_1408, 704)
PLANE_MIX(plane_mixed_1536, 768)
PLANE_MIX(plane_mixed_1664, 832)
PLANE_MIX(plane_mixed_2048, 1024)
