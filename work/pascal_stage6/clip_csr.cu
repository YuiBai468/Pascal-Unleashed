extern "C" __global__ void tail_offsets(const unsigned short* counts,int* starts,int* nnz,int n) {
    __shared__ int sums[8];
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5,t=threadIdx.x;
    const int stride=(n+255)/256;
    const int lo=min(t*stride,n),hi=min(lo+stride,n);
    int running=0;
    for (int i=lo;i<hi;++i) running+=counts[i];
    const int local=running;
    for (int i=1;i<32;i<<=1) {
        const int previous=__shfl_up_sync(0xffffffff,running,i);
        if (lane>=i) running+=previous;
    }
    if (lane==31) sums[warp]=running;
    __syncthreads();
    if (!warp) {
        int subtotal=lane<8?sums[lane]:0;
        for (int i=1;i<8;i<<=1) {
            const int previous=__shfl_up_sync(0xffffffff,subtotal,i);
            if (lane>=i) subtotal+=previous;
        }
        if (lane<8) sums[lane]=subtotal;
        if (lane==7) {
            starts[n]=subtotal;
            nnz[0]=subtotal;
        }
    }
    __syncthreads();
    running+= (warp?sums[warp-1]:0)-local;
    for (int i=lo;i<hi;++i) {
        starts[i]=running;
        running+=counts[i];
    }
}

__device__ __forceinline__ unsigned short encode_half(float x) {
    unsigned short y;
    asm("cvt.rn.f16.f32 %0,%1;" : "=h"(y) : "f"(x));
    return y;
}

extern "C" __global__ void compact_tail(const Residual* tail,const int* starts,int* index,
                                       unsigned short* value,int capacity) {
    const int row=blockIdx.x,tid=threadIdx.x;
    const int lo=__ldg(starts+row),count=__ldg(starts+row+1)-lo;
    for (int i=tid;i<count;i+=128) {
        const Residual r=tail[row*capacity+i];
        index[lo+i]=r.k;
        value[lo+i]=encode_half(r.value);
    }
}

extern "C" __global__ void half_core(const unsigned int* source,unsigned int* result,
                                     const float2* scales,int rows,int k,int cap) {
    const int pair=blockIdx.x*blockDim.x+threadIdx.x;
    const int row=pair/(k/2);
    if (row>=rows) return;
    const unsigned int p=source[pair];
    const float t=__ldg(&scales[row].x)*cap;
    const float x=decode_plane_half(static_cast<unsigned short>(p));
    const float y=decode_plane_half(static_cast<unsigned short>(p>>16));
    const unsigned int a=encode_half(fminf(fmaxf(x,-t),t));
    const unsigned int b=encode_half(fminf(fmaxf(y,-t),t));
    result[pair]=a|(b<<16);
}

extern "C" __global__ void transpose_half_64(const unsigned short* in,unsigned short* out,int rows,int cols) {
    __shared__ unsigned short tiles[64][66];
    const int row0=64*blockIdx.y,col0=64*blockIdx.x;
    const int tx=threadIdx.x&63,ty=threadIdx.x>>6;
    #pragma unroll
    for (int i=0;i<64;i+=4) tiles[i+ty][tx]=in[static_cast<size_t>(row0+i+ty)*cols+col0+tx];
    __syncthreads();
    #pragma unroll
    for (int i=0;i<64;i+=4) out[static_cast<size_t>(col0+i+ty)*rows+row0+tx]=tiles[tx][i+ty];
}

extern "C" __global__ void ar2_base_finish(const int* h,const int* g,const float2* a,const float2* b,
                                           float* out,int m,int n,float weight_ratio,float weight_g) {
    const int j=4*(blockIdx.x*32+threadIdx.x);
    if (j>=n) return;
    const float a0=__ldg(&b[j].x),a1=__ldg(&b[j+1].x),a2=__ldg(&b[j+2].x),a3=__ldg(&b[j+3].x);
    #pragma unroll
    for (int s=0;s<4;++s) {
        const int i=blockIdx.y*16+threadIdx.y+4*s;
        if (i>=m) continue;
        const float mul=__ldg(&a[i].x)*weight_g;
        const size_t pos=static_cast<size_t>(i)*n+j;
        const int4 u=*reinterpret_cast<const int4*>(h+pos),v=*reinterpret_cast<const int4*>(g+pos);
        *reinterpret_cast<float4*>(out+pos)=make_float4(
            (mul*a0)*fmaf(weight_ratio,u.x,v.x),(mul*a1)*fmaf(weight_ratio,u.y,v.y),
            (mul*a2)*fmaf(weight_ratio,u.z,v.z),(mul*a3)*fmaf(weight_ratio,u.w,v.w));
    }
}

extern "C" __global__ void add_transposed(const float* source,float* target,int m,int n) {
    __shared__ float tmp[32][33];
    const int i=blockIdx.x*32,j=blockIdx.y*32;
    const int x=threadIdx.x&31,y=threadIdx.x>>5;
    #pragma unroll
    for (int p=0;p<32;p+=8) tmp[p+y][x]=source[static_cast<size_t>(j+p+y)*m+i+x];
    __syncthreads();
    #pragma unroll
    for (int p=0;p<32;p+=8) target[static_cast<size_t>(i+p+y)*n+j+x]+=tmp[x][p+y];
}

extern "C" __global__ void combine_sparse_ar2(const int* h,const int* g,const float* ca,const float* cb,
                                              const float2* sa,const float2* sb,float* out,int m,int n,
                                              float weight_ratio,float weight_g) {
    __shared__ float transposed[32][33];
    const int row0=blockIdx.y*32,col0=blockIdx.x*32;
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    #pragma unroll
    for (int s=0;s<32;s+=8) transposed[warp+s][lane]=cb[static_cast<size_t>(col0+warp+s)*m+row0+lane];
    __syncthreads();
    const int j=col0+lane;
    const float b=__ldg(&sb[j].x)*weight_g;
    #pragma unroll
    for (int s=0;s<32;s+=8) {
        const int i=row0+warp+s;
        const size_t at=static_cast<size_t>(i)*n+j;
        const float a=__ldg(&sa[i].x);
        out[at]=(a*b)*fmaf(weight_ratio,h[at],g[at])+ca[at]+transposed[lane][warp+s];
    }
}
