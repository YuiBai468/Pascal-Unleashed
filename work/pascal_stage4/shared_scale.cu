// A low-precision channel and both coordinates of a high-precision lattice
// use the same row scale. Their dot products can share one INT32 accumulator.
constexpr float sqrt41 = 6.403124237432849f;
constexpr int lattice_cap = 1035;

template<bool Affine>
__device__ void pack_shared(const uint2* input, unsigned int* output, float* scales,
                            float* offsets, float* sums, unsigned int* invalid,
                            int k_words, int high_words) {
    __shared__ uint2 values[1024];
    __shared__ float warp_hi[8], warp_lo_min[8], warp_lo_max[8];
    __shared__ float inverse, center, step;
    __shared__ int warp_sum[8];
    const int row=blockIdx.x, lane=threadIdx.x, row_base=row*k_words;
    const float infinity=__int_as_float(0x7f800000);
    float high_peak=0, lo_min=infinity, lo_max=-infinity;
    for(int p=lane;p<k_words;p+=256){
        const uint2 data=input[row_base+p];
        values[p]=data;
        #pragma unroll
        for(int j=0;j<4;++j){
            const unsigned short bits=(j<2?data.x:data.y)>>(16*(j%2));
            const bool good=(bits&0x7fff)<0x7c00;
            if(!good)atomicOr(invalid,1u);
            const float x=good?half_value(bits):0.0f;
            if(p<high_words) high_peak=fmaxf(high_peak,fabsf(x));
            else {lo_min=fminf(lo_min,x);lo_max=fmaxf(lo_max,x);}
        }
    }
    for(int d=16;d>0;d>>=1){
        high_peak=fmaxf(high_peak,__shfl_down_sync(0xffffffff,high_peak,d));
        lo_min=fminf(lo_min,__shfl_down_sync(0xffffffff,lo_min,d));
        lo_max=fmaxf(lo_max,__shfl_down_sync(0xffffffff,lo_max,d));
    }
    if(lane%32==0) {warp_hi[lane/32]=high_peak;warp_lo_min[lane/32]=lo_min;warp_lo_max[lane/32]=lo_max;}
    __syncthreads();
    if(lane<32){
        high_peak=lane<8?warp_hi[lane]:0;
        lo_min=lane<8?warp_lo_min[lane]:infinity;
        lo_max=lane<8?warp_lo_max[lane]:-infinity;
        for(int d=4;d>0;d>>=1){
            high_peak=fmaxf(high_peak,__shfl_down_sync(0xffffffff,high_peak,d));
            lo_min=fminf(lo_min,__shfl_down_sync(0xffffffff,lo_min,d));
            lo_max=fmaxf(lo_max,__shfl_down_sync(0xffffffff,lo_max,d));
        }
        if(lane==0){
            const float lo=Affine?(lo_max-lo_min)/255.0f:fmaxf(fabsf(lo_min),fabsf(lo_max))/127.0f;
            const float span=fmaxf(lo,high_peak*(sqrt41/lattice_cap));
            step=span>0?span:1.0f;
            center=Affine?(lo_min==lo_max?lo_min:0.5f*(lo_min+lo_max+step)):0.0f;
            inverse=__frcp_rn(step);
            scales[row]=step;
            if(Affine)offsets[row]=center;
        }
    }
    __syncthreads();
    const int packed_stride=k_words+high_words;
    int local_sum=0;
    for(int p=lane;p<k_words;p+=256){
        const uint2 data=values[p]; unsigned int first=0,second=0;
        if(p<high_words){
            #pragma unroll
            for(int j=0;j<4;++j){
                const unsigned short bits=(j<2?data.x:data.y)>>(16*(j%2));
                const float x=(bits&0x7fff)<0x7c00?half_value(bits):0.0f;
                const int q=max(-lattice_cap,min(lattice_cap,__float2int_rn(x*(sqrt41*inverse))));
                const int mod=(32*q+20)%41;
                const int r=(mod<0?mod+41:mod)-20;
                const int h=(5*q-4*r)/41, g=(4*q+5*r)/41;
                first|=(static_cast<unsigned int>(h)&255u)<<(8*j);
                second|=(static_cast<unsigned int>(g)&255u)<<(8*j);
            }
            output[row*packed_stride+k_words+p]=second;
        } else {
            #pragma unroll
            for(int j=0;j<4;++j){
                const unsigned short bits=(j<2?data.x:data.y)>>(16*(j%2));
                const float x=(bits&0x7fff)<0x7c00?half_value(bits):0.0f;
                const int q=max(Affine?-128:-127,min(127,__float2int_rn((x-center)*inverse)));
                first|=(static_cast<unsigned int>(q)&255u)<<(8*j);
                if(Affine)local_sum+=q;
            }
        }
        output[row*packed_stride+p]=first;
    }
    if(Affine){
        for(int d=16;d>0;d>>=1)local_sum+=__shfl_down_sync(0xffffffff,local_sum,d);
        if(lane%32==0)warp_sum[lane/32]=local_sum;
        __syncthreads();
        if(lane<32){
            local_sum=lane<8?warp_sum[lane]:0;
            for(int d=4;d>0;d>>=1)local_sum+=__shfl_down_sync(0xffffffff,local_sum,d);
            if(lane==0)sums[row]=step*local_sum;
        }
    }
}

#define SHARED_ENTRY(name,affine) \
extern "C" __global__ void name(const uint2* a,unsigned int* out,float* scales,float* offsets,float* sums, \
                               unsigned int* invalid,int k,int hi){pack_shared<affine>(a,out,scales,offsets,sums,invalid,k,hi);}
SHARED_ENTRY(pack_shared_symmetric,false)
SHARED_ENTRY(pack_shared_affine,true)

extern "C" __global__ void unpack_shared(const int* dot,const float* a,const float* b,
                                         const float* za,const float* zb,const float* qa,const float* qb,
                                         float* output,int m,int n,int low_k,int affine){
    const int i=blockIdx.y*blockDim.y+threadIdx.y,j=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=m||j>=n)return;
    const int p=i*n+j;
    const float correction=affine?fmaf(za[i],fmaf(static_cast<float>(low_k),zb[j],qb[j]),zb[j]*qa[i]):0.0f;
    output[p]=fmaf(static_cast<float>(dot[p]),a[i]*b[j],correction);
}
