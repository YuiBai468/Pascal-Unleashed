__device__ __forceinline__ float anima_half_input(float x, unsigned int* bad) {
    unsigned short bits;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(bits) : "f"(x));
    if ((bits&0x7fffu)>=0x7c00u) {
        atomicOr(bad,1u);
        return 0.0f;
    }
    return decode_plane_half(bits);
}

__device__ __forceinline__ float4 anima_project(float x,float y,bool scalar,float3 u,float3 v) {
    if (scalar) return make_float4(.8f*x,-.6f*x,.8f*y,-.6f*y);
    return make_float4(fmaf(x,u.x,y*v.x),fmaf(x,u.y,y*v.y),fmaf(x,u.z,y*v.z),0.0f);
}

// The same projections, signed-byte translations, lookup tables and blends as
// slide_line_8192. K and the proportion of scalar coordinates are runtime inputs.
template<bool ClipEdge>
__device__ void anima_pack_rows_impl(
        const float* input,signed char* output,float* scales,int k,int dual,int width,
        const float* basis,const unsigned short* plane_lut,const unsigned short* line_lut,
        unsigned int* bad,int margin) {
    const int row=blockIdx.x, tid=threadIdx.x, lane=tid&31, warp=tid>>5;
    const int pairs=(k+1)/2, plane=pairs-dual;
    const float* src=input+static_cast<size_t>(row)*k;
    signed char* dst=output+static_cast<size_t>(row)*width;
    const float3 u=make_float3(__ldg(basis),__ldg(basis+2),__ldg(basis+4));
    const float3 v=make_float3(__ldg(basis+1),__ldg(basis+3),__ldg(basis+5));
    __shared__ float warp_peak[8], inverse;
    float peak=0.0f;
    for (int i=tid;i<pairs;i+=256) {
        const float x=anima_half_input(src[2*i],bad);
        const float y=2*i+1<k?anima_half_input(src[2*i+1],bad):0.0f;
        const bool scalar=i<dual;
        const float4 p=anima_project(x,y,scalar,u,v);
        if (scalar) {
            peak=fmaxf(peak,line_scale(make_float2(p.x,p.y),.89f));
            peak=fmaxf(peak,line_scale(make_float2(p.z,p.w),.89f));
        } else {
            peak=fmaxf(peak,cube_scale<3,4,5>(p,.81f));
        }
    }
    for (int d=16;d;d>>=1) peak=fmaxf(peak,__shfl_down_sync(0xffffffff,peak,d));
    if (!lane) warp_peak[warp]=peak;
    __syncthreads();
    if (!warp) {
        peak=lane<8?warp_peak[lane]:0.0f;
        for (int d=4;d;d>>=1) peak=fmaxf(peak,__shfl_down_sync(0xffffffff,peak,d));
        if (!lane) {
            const float scale=peak>0.0f?peak/margin:1.0f;
            scales[row]=scale;
            inverse=16.0f*__frcp_rn(scale);
        }
    }
    __syncthreads();
    for (int i=tid;i<pairs;i+=256) {
        const float x=anima_half_input(src[2*i],bad);
        const float y=2*i+1<k?anima_half_input(src[2*i+1],bad):0.0f;
        const bool scalar=i<dual;
        const float4 p=anima_project(x,y,scalar,u,v);
        if (scalar) {
            write_sliding_line<ClipEdge>(make_float2(p.x,p.y),inverse,line_lut,bad,dst+i,dual);
            write_sliding_line<ClipEdge>(make_float2(p.z,p.w),inverse,line_lut,bad,dst+i+2*dual,dual);
        } else {
            const int4 q=encode_sliding<3,4,5,0,4,true,ClipEdge>(p,inverse,plane_lut,bad);
            signed char* out=dst+i+3*dual;
            out[0]=q.x;
            out[plane]=q.y;
            out[2*plane]=q.z;
        }
    }
    const int physical_k=3*pairs+dual;
    for (int i=physical_k+tid;i<width;i+=256) dst[i]=0;
}

#define ANIMA_PACK(name,clip_edge) extern "C" __global__ __launch_bounds__(256) void name( \
    const float* input,signed char* output,float* scales,int k,int dual,int width, \
    const float* basis,const unsigned short* plane_lut,const unsigned short* line_lut, \
    unsigned int* bad,int margin) { \
    anima_pack_rows_impl<clip_edge>(input,output,scales,k,dual,width,basis,plane_lut,line_lut,bad,margin); }
ANIMA_PACK(anima_pack_rows,false)
ANIMA_PACK(anima_pack_rows_edge,true)

extern "C" __global__ void anima_reconstruct(const int* raw,const float* lhs,const float* rhs,
                                              const float* bias,float* out,int m,int n) {
    const int first=(blockIdx.x*32+threadIdx.x)*4;
    if (first>=n) return;
    const float4 sc=*reinterpret_cast<const float4*>(rhs+first);
    const float4 shift=bias?*reinterpret_cast<const float4*>(bias+first):make_float4(0.f,0.f,0.f,0.f);
    #pragma unroll
    for (int i=0;i<4;++i) {
        const int row=blockIdx.y*16+threadIdx.y+4*i;
        if (row>=m) continue;
        const size_t at=static_cast<size_t>(row)*n+first;
        const float a=__ldg(lhs+row);
        const int4 dot=*reinterpret_cast<const int4*>(raw+at);
        *reinterpret_cast<float4*>(out+at)=make_float4(
            fmaf(a*sc.x,dot.x,shift.x),fmaf(a*sc.y,dot.y,shift.y),
            fmaf(a*sc.z,dot.z,shift.z),fmaf(a*sc.w,dot.w,shift.w));
    }
}
