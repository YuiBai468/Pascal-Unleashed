__device__ __forceinline__ float4 project_mixed_pair(unsigned int bits,const float* basis) {
    const float x=decode_plane_half(static_cast<unsigned short>(bits));
    const float y=decode_plane_half(static_cast<unsigned short>(bits>>16));
    return make_float4(fmaf(x,__ldg(basis),y*__ldg(basis+1)),
                       fmaf(x,__ldg(basis+2),y*__ldg(basis+3)),
                       fmaf(x,__ldg(basis+4),y*__ldg(basis+5)),0);
}

extern "C" __global__ void mixed_right_scales(const unsigned int* input,float* scales,float* factors,
                                               const float* basis,unsigned int* invalid,int dual_pairs,
                                               int margin,float plane_blend,float line_blend) {
    constexpr int pairs=4096, threads=256;
    __shared__ float warp_peaks[threads/32];
    const int row=blockIdx.x, tid=threadIdx.x;
    float peak=0;
    for (int i=tid;i<pairs;i+=threads) {
        unsigned int bits=input[row*pairs+i];
        if ((bits&0x7fff)>=0x7c00 || ((bits>>16)&0x7fff)>=0x7c00) {
            atomicOr(invalid,1u);
            bits=0;
        }
        if (i<dual_pairs) {
            const float x=decode_plane_half(static_cast<unsigned short>(bits));
            const float y=decode_plane_half(static_cast<unsigned short>(bits>>16));
            peak=fmaxf(peak,line_scale(make_float2(.8f*x,-.6f*x),line_blend));
            peak=fmaxf(peak,line_scale(make_float2(.8f*y,-.6f*y),line_blend));
        } else {
            peak=fmaxf(peak,cube_scale<3,4,5>(project_mixed_pair(bits,basis),plane_blend));
        }
    }
    for (int i=16;i;i>>=1) peak=fmaxf(peak,__shfl_down_sync(0xffffffff,peak,i));
    if (!(tid%32)) warp_peaks[tid/32]=peak;
    __syncthreads();
    if (tid<32) {
        peak=tid<threads/32?warp_peaks[tid]:0;
        for (int i=threads/64;i;i>>=1) peak=fmaxf(peak,__shfl_down_sync(0xffffffff,peak,i));
        if (!tid) {
            const float scale=peak>0?peak/margin:1.0f;
            scales[row]=scale;
            factors[row]=16*__frcp_rn(scale);
        }
    }
}

extern "C" __global__ void mixed_right_pack(const unsigned int* input,signed char* output,const float* factors,
                                             const float* basis,const unsigned short* plane_lut,
                                             const unsigned short* line_lut,unsigned* invalid,int n,int dual_pairs) {
    constexpr int pairs=4096;
    __shared__ unsigned int tile[32][33];
    const int row0=blockIdx.y*32,pair0=blockIdx.x*32;
    const int lane=threadIdx.x%32, warp=threadIdx.x/32;
    const bool line=pair0<dual_pairs;
    #pragma unroll
    for (int step=0;step<32;step+=8) {
        const int row=row0+step+warp;
        const unsigned int bits=input[row*pairs+pair0+lane];
        const float factor=__ldg(factors+row);
        unsigned int digits;
        if (line) {
            const float x=decode_plane_half(static_cast<unsigned short>(bits));
            const float y=decode_plane_half(static_cast<unsigned short>(bits>>16));
            const int2 a=encode_line(make_float2(.8f*x,-.6f*x),factor,line_lut,invalid);
            const int2 b=encode_line(make_float2(.8f*y,-.6f*y),factor,line_lut,invalid);
            digits=(a.x&255u)|((a.y&255u)<<8)|((b.x&255u)<<16)|((b.y&255u)<<24);
        } else {
            const int4 q=encode_sliding<3,4,5,0,4>(project_mixed_pair(bits,basis),factor,plane_lut,invalid);
            digits=(q.x&255u)|((q.y&255u)<<8)|((q.z&255u)<<16);
        }
        tile[step+warp][lane]=digits;
    }
    __syncthreads();
    const int slab=(line?dual_pairs:pairs-dual_pairs)*n;
    #pragma unroll
    for (int step=0;step<32;step+=8) {
        const int pos=pair0+step+warp+(line?0:3*dual_pairs);
        const int at=pos*n+row0+lane;
        const unsigned int digits=tile[lane][step+warp];
        output[at]=static_cast<signed char>(digits);
        output[at+slab]=static_cast<signed char>(digits>>8);
        output[at+2*slab]=static_cast<signed char>(digits>>16);
        if (line) output[at+3*slab]=static_cast<signed char>(digits>>24);
    }
}
