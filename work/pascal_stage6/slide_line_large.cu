__device__ __forceinline__ float line_scale(float2 p, float blend) {
    return fmaxf(blend*fmaxf(fabsf(p.x),fabsf(p.y)),fabsf(fmaf(4,p.x,-3*p.y))*(1.0f/7));
}

template<bool ClipEdge = false>
__device__ __forceinline__ int2 encode_line(float2 p, float factor, const unsigned short* lut, unsigned* invalid) {
    const int a=__float2int_rd(fmaf(p.x,factor,8.0f));
    const int b=__float2int_rd(fmaf(p.y,factor,8.0f));
    const unsigned int offsets=__ldg(lut+((a&15)<<4)+(b&15));
    int x=(a>>4)+static_cast<int>(offsets&15)-8;
    int y=(b>>4)+static_cast<int>((offsets>>4)&15)-8;
    if (static_cast<unsigned>(x+127)>254 || static_cast<unsigned>(y+127)>254) {
        constexpr float epsilon=.0001f;
        const float lower=fmaxf((-127-x)*(1.0f/3),(-127-y)*.25f);
        const float upper=fminf((127-x)*(1.0f/3),(127-y)*.25f);
        const int lo=__float2int_ru(lower-epsilon),hi=__float2int_rd(upper+epsilon);
        int shift=min(max(0,lo),hi);
        if (lo>hi) {
            if (ClipEdge) shift=__float2int_rn((lower+upper)*.5f);
            else atomicOr(invalid,2u);
        }
        x+=3*shift;
        y+=4*shift;
        if (ClipEdge) {
            x=max(-127,min(127,x));
            y=max(-127,min(127,y));
        }
    }
    return make_int2(x,y);
}

template<bool ClipEdge = false>
__device__ __forceinline__ void write_sliding_line(float2 p, float factor, const unsigned short* lut,
                                                  unsigned* invalid, signed char* output, int stride) {
    const int2 q=encode_line<ClipEdge>(p,factor,lut,invalid);
    output[0]=q.x;
    output[stride]=q.y;
}

template<int DualPairs, bool Masked = false, bool ClipEdge = false, bool ProjectedScale = false>
__device__ void pack_slide_line_8192(const unsigned int* input, signed char* packed, float* scales,
                                   const float* basis, const unsigned short* plane_lut,
                                   const unsigned short* line_lut, unsigned* invalid,
                                   int margin, float pair_blend, float line_blend,
                                   const unsigned char* mask = nullptr) {
    constexpr int pairs=4096, threads=512, count=pairs/threads;
    constexpr int plane=pairs-DualPairs, width=4*DualPairs+3*plane;
    __shared__ float peak_warp[16], factor;
    const int row=blockIdx.x, lane=threadIdx.x;
    const float u0=__ldg(basis), v0=__ldg(basis+1), u1=__ldg(basis+2);
    const float v1=__ldg(basis+3),u2=__ldg(basis+4),v2=__ldg(basis+5);
    float4 embed[count];
    float peak=0;
    #pragma unroll
    for (int i=0;i<count;++i) {
        const int spot=i*threads+lane;
        unsigned data=input[row*pairs+spot];
        if ((data&0x7fff)>=0x7c00 || ((data>>16)&0x7fff)>=0x7c00) {
            atomicOr(invalid,1u);
            data=0;
        }
        if (Masked && __ldg(mask+spot)) data=0;
        const float x=decode_plane_half(static_cast<unsigned short>(data));
        const float y=decode_plane_half(static_cast<unsigned short>(data>>16));
        const bool line=i<DualPairs/threads || (i==DualPairs/threads && lane<DualPairs%threads);
        if (line) {
            const float4 p=make_float4(.8f*x,-.6f*x,.8f*y,-.6f*y);
            embed[i]=p;
            if (ProjectedScale) {
                const float support=fmaxf(.8f*line_blend,5.0f/7.0f);
                peak=fmaxf(peak,support*fmaxf(fabsf(x),fabsf(y)));
            } else {
                peak=fmaxf(peak,fmaxf(line_scale(make_float2(p.x,p.y),line_blend),
                                     line_scale(make_float2(p.z,p.w),line_blend)));
            }
        } else {
            const float4 p=make_float4(fmaf(x,u0,y*v0),fmaf(x,u1,y*v1),fmaf(x,u2,y*v2),0);
            embed[i]=p;
            if (ProjectedScale) {
                // Project the three pairs of cube faces onto the fixed (3,4,5) plane first.
                const float face01=(5.0f/7.0f)*fabsf(x);
                const float face02=fabsf(fmaf(.5f,x,.53033008589f*y));
                const float face12=fabsf(fmaf(-1.0f/3.0f,x,.62853936105f*y));
                const float corner=pair_blend*fmaxf(fabsf(p.x),fmaxf(fabsf(p.y),fabsf(p.z)));
                peak=fmaxf(peak,fmaxf(corner,fmaxf(face01,fmaxf(face02,face12))));
            } else peak=fmaxf(peak,cube_scale<3,4,5>(p,pair_blend));
        }
    }
    for (int i=16;i;i>>=1) peak=fmaxf(peak,__shfl_down_sync(0xffffffff,peak,i));
    if ((lane&31)==0) peak_warp[lane>>5]=peak;
    __syncthreads();
    if (lane<32) {
        peak=lane<16?peak_warp[lane]:0;
        for (int i=8;i;i>>=1) peak=fmaxf(peak,__shfl_down_sync(0xffffffff,peak,i));
        if (lane==0) {
            const float scale=peak>0?peak/margin:1.0f;
            scales[row]=scale;
            factor=16*__frcp_rn(scale);
        }
    }
    __syncthreads();
    signed char* out=packed+row*width;
    #pragma unroll
    for (int i=0;i<count;++i) {
        const int spot=i*threads+lane;
        const float4 p=embed[i];
        const bool line=i<DualPairs/threads || (i==DualPairs/threads && lane<DualPairs%threads);
        if (line) {
            write_sliding_line<ClipEdge>(make_float2(p.x,p.y),factor,line_lut,invalid,out+spot,DualPairs);
            write_sliding_line<ClipEdge>(make_float2(p.z,p.w),factor,line_lut,invalid,out+spot+2*DualPairs,DualPairs);
        } else {
            const int4 q=encode_sliding<3,4,5,0,4,true,ClipEdge>(p,factor,plane_lut,invalid);
            signed char* t=out+spot+3*DualPairs;
            t[0]=q.x; t[plane]=q.y; t[2*plane]=q.z;
        }
    }
}

#define SLIDE_LINE(dual) extern "C" __global__ __launch_bounds__(512) void slide_line_##dual( \
    const unsigned int* in, signed char* out, float* scales, const float* basis, \
    const unsigned short* plane_lut, const unsigned short* line_lut, unsigned* invalid, \
    int margin, float pair_blend, float line_blend) { \
    pack_slide_line_8192<dual/2>(in,out,scales,basis,plane_lut,line_lut,invalid,margin,pair_blend,line_blend); }
SLIDE_LINE(768)
SLIDE_LINE(1024)
SLIDE_LINE(1280)
SLIDE_LINE(1536)
SLIDE_LINE(1792)
SLIDE_LINE(2048)
SLIDE_LINE(2304)
SLIDE_LINE(2432)
SLIDE_LINE(2560)
SLIDE_LINE(4096)
SLIDE_LINE(4608)
SLIDE_LINE(5120)
SLIDE_LINE(5632)
SLIDE_LINE(5888)
SLIDE_LINE(6144)
SLIDE_LINE(6656)
SLIDE_LINE(7168)
SLIDE_LINE(7680)
SLIDE_LINE(8192)

#define EDGE_LINE(dual) extern "C" __global__ __launch_bounds__(512) void edge_line_##dual( \
    const unsigned int* in, signed char* out, float* scales, const float* basis, \
    const unsigned short* plane_lut, const unsigned short* line_lut, unsigned* invalid, \
    int margin, float pair_blend, float line_blend) { \
    pack_slide_line_8192<dual/2,false,true>(in,out,scales,basis,plane_lut,line_lut,invalid,margin,pair_blend,line_blend); }
EDGE_LINE(768)
EDGE_LINE(1024)
EDGE_LINE(1280)
EDGE_LINE(1536)
EDGE_LINE(1792)
EDGE_LINE(2048)
EDGE_LINE(2304)
EDGE_LINE(2432)
EDGE_LINE(2560)
EDGE_LINE(2816)
EDGE_LINE(3072)
EDGE_LINE(3328)
EDGE_LINE(3584)

#define FACE_LINE(dual) extern "C" __global__ __launch_bounds__(512) void face_line_##dual( \
    const unsigned int* in, signed char* out, float* scales, const float* basis, \
    const unsigned short* plane_lut, const unsigned short* line_lut, unsigned* invalid, \
    int margin, float pair_blend, float line_blend) { \
    pack_slide_line_8192<dual/2,false,true,true>(in,out,scales,basis,plane_lut,line_lut,invalid,margin,pair_blend,line_blend); }
FACE_LINE(2048)
FACE_LINE(2304)
FACE_LINE(2432)
FACE_LINE(2560)
FACE_LINE(2816)
FACE_LINE(3072)
FACE_LINE(3328)
FACE_LINE(3584)

#define MASKED_LINE(dual) extern "C" __global__ __launch_bounds__(512) void masked_line_##dual( \
    const unsigned int* in, signed char* out, float* scales, const float* basis, \
    const unsigned short* plane_lut, const unsigned short* line_lut, unsigned* invalid, \
    int margin, float pair_blend, float line_blend, const unsigned char* mask) { \
    pack_slide_line_8192<dual/2,true>(in,out,scales,basis,plane_lut,line_lut,invalid,margin,pair_blend,line_blend,mask); }
MASKED_LINE(0)
MASKED_LINE(512)
MASKED_LINE(768)
MASKED_LINE(1024)
