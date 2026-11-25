__device__ __forceinline__ float to_float(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0,%1;" : "=f"(result) : "h"(bits));
    return result;
}

template<bool Affine> __device__ void quantize(const uint2* input, unsigned int* codes,
                                               float* scales, float* means, int* sums,
                                               unsigned int* invalid, int packed_k) {
    __shared__ uint2 row_data[1024];
    __shared__ float minima[8], maxima[8], x_sums[8], scale, inverse, midpoint;
    __shared__ int q_sums[8];
    const int row=blockIdx.x, tid=threadIdx.x, lane=tid%32, warp=tid/32;
    float low=65504, high=-65504, x_sum=0;
    for (int p=tid;p<packed_k;p+=256) {
        uint2 v=input[row*packed_k+p];
        #pragma unroll
        for (int j=0;j<4;++j) {
            unsigned int& word=(j<2?v.x:v.y);
            const int shift=16*(j%2);
            unsigned short bits=word>>shift;
            if ((bits&0x7fff)>=0x7c00) {
                atomicOr(invalid,1u);
                word&=~(0xffffu<<shift);
                bits=0;
            }
            const float x=to_float(bits);
            if (Affine) {
                low=fminf(low,x);
                high=fmaxf(high,x);
                x_sum+=x;
            } else high=fmaxf(high,fabsf(x));
        }
        row_data[p]=v;
    }
    for (int stride=16;stride>0;stride/=2) {
        high=fmaxf(high,__shfl_down_sync(0xffffffff,high,stride));
        if (Affine) {
            low=fminf(low,__shfl_down_sync(0xffffffff,low,stride));
            x_sum+=__shfl_down_sync(0xffffffff,x_sum,stride);
        }
    }
    if (lane==0) {maxima[warp]=high; minima[warp]=low; x_sums[warp]=x_sum;}
    __syncthreads();
    if (warp==0) {
        high=lane<8?maxima[lane]:-65504;
        low=lane<8?minima[lane]:65504;
        x_sum=lane<8?x_sums[lane]:0;
        for (int stride=4;stride>0;stride/=2) {
            high=fmaxf(high,__shfl_down_sync(0xffffffff,high,stride));
            if (Affine) {
                low=fminf(low,__shfl_down_sync(0xffffffff,low,stride));
                x_sum+=__shfl_down_sync(0xffffffff,x_sum,stride);
            }
        }
        if (tid==0) {
            if (Affine) {
                scale=high==low?1.0f:(high-low)*(1.0f/255.0f);
                midpoint=high==low?low:0.5f*(high+low+scale);
                means[row]=x_sum/(4*packed_k);
            } else {
                scale=high==0?1.0f:high*(1.0f/127.0f);
                midpoint=0;
            }
            scales[row]=scale;
            inverse=__frcp_rn(scale);
        }
    }
    __syncthreads();
    int q_sum=0;
    for (int p=tid;p<packed_k;p+=256) {
        const uint2 v=row_data[p];
        unsigned int packed=0;
        #pragma unroll
        for (int j=0;j<4;++j) {
            const unsigned short bits=(j<2?v.x:v.y)>>(16*(j%2));
            const int q=min(127,max(Affine?-128:-127,__float2int_rn((to_float(bits)-midpoint)*inverse)));
            packed|=(static_cast<unsigned int>(q)&255)<<(8*j);
            if (Affine) q_sum+=q;
        }
        codes[row*packed_k+p]=packed;
    }
    if (Affine) {
        for (int stride=16;stride>0;stride/=2) q_sum+=__shfl_down_sync(0xffffffff,q_sum,stride);
        if(lane==0) q_sums[warp]=q_sum;
        __syncthreads();
        if(warp==0) {
            q_sum=lane<8?q_sums[lane]:0;
            for (int stride=4;stride>0;stride/=2) q_sum+=__shfl_down_sync(0xffffffff,q_sum,stride);
            if(tid==0) sums[row]=q_sum;
        }
    }
}

#define ENCODE(name, mode) \
extern "C" __global__ void name(const uint2* input, unsigned int* codes, float* scale, \
                    float* mean, int* sum, unsigned int* invalid, int packed_k) { \
    quantize<mode>(input,codes,scale,mean,sum,invalid,packed_k); \
}
ENCODE(symmetric_encode,false)
ENCODE(affine_encode,true)

extern "C" __global__ void reconstruct_q8(int* output, const float* as, const float* bs,
                                           const float* am, const float* bm,
                                           const int* aq, const int* bq,
                                           int m, int n, int k, int affine) {
    const int j=blockIdx.x*blockDim.x+threadIdx.x, i=blockIdx.y*blockDim.y+threadIdx.y;
    if(i>=m||j>=n)return;
    const int loc=i*n+j;
    const float scale=as[i]*bs[j], dot=static_cast<float>(output[loc]);
    float result=dot*scale;
    if (affine) {
        const float centered=fmaf(-static_cast<float>(aq[i])/k,static_cast<float>(bq[j]),dot);
        result=fmaf(centered,scale,(static_cast<float>(k)*am[i])*bm[j]);
    }
    // Store FP32 bits in the existing INT32 allocation after the GEMM.
    output[loc]=__float_as_int(result);
}

extern "C" __global__ void transpose_fp16(const unsigned short* a, unsigned short* b, int rows, int cols) {
    __shared__ unsigned short tile[32][33];
    const int x=threadIdx.x,y=threadIdx.y;
    int i=blockIdx.y*32+y,j=blockIdx.x*32+x;
    #pragma unroll
    for(int p=0;p<32;p+=8) if(i+p<rows&&j<cols) tile[y+p][x]=a[(i+p)*cols+j];
    __syncthreads();
    i=blockIdx.x*32+y;j=blockIdx.y*32+x;
    #pragma unroll
    for(int p=0;p<32;p+=8) if(i+p<cols&&j<rows) b[(i+p)*rows+j]=tile[x][y+p];
}
