__device__ __forceinline__ float hfloat(unsigned short x) {
    float out;
    asm("cvt.f32.f16 %0,%1;" : "=f"(out):"h"(x));
    return out;
}

extern "C" __global__ void encode_rows(const uint2* input, int* h, int* g, int* remainder,
                                      float* scales, float* cutoffs, unsigned short* indices,
                                      float* deltas, int* counts, int* invalid,
                                      int k, int base, int cap, int levels, float clip) {
    __shared__ uint2 values[1024];
    __shared__ float warp_max[8], warp_sum[8], scale, cutoff;
    __shared__ int nonzero;
    const int row=blockIdx.x, t=threadIdx.x, packed=k/4;
    float maximum=0,squares=0;
    for(int p=t;p<packed;p+=256) {
        const uint2 v=input[row*packed+p]; values[p]=v;
        #pragma unroll
        for(int j=0;j<4;++j) {
            const unsigned short bits=(j<2?v.x:v.y)>>(16*(j%2));
            const bool finite=(bits&0x7fff)<0x7c00;
            if(!finite) atomicOr(invalid,1);
            const float x=finite?hfloat(bits):0;
            maximum=fmaxf(maximum,fabsf(x));
            if(clip>0) squares=fmaf(x,x,squares);
        }
    }
    for(int offset=16;offset>0;offset/=2) {
        maximum=fmaxf(maximum,__shfl_down_sync(0xffffffff,maximum,offset));
        if(clip>0) squares+=__shfl_down_sync(0xffffffff,squares,offset);
    }
    if(t%32==0) {warp_max[t/32]=maximum;warp_sum[t/32]=squares;}
    __syncthreads();
    if(t<32) {
        maximum=t<8?warp_max[t]:0; squares=t<8?warp_sum[t]:0;
        for(int offset=4;offset>0;offset/=2) {
            maximum=fmaxf(maximum,__shfl_down_sync(0xffffffff,maximum,offset));
            if(clip>0) squares+=__shfl_down_sync(0xffffffff,squares,offset);
        }
        if(t==0) {
            cutoff=clip>0?fminf(maximum,clip*sqrtf(squares/k)):maximum;
            scale=cutoff>0?cutoff/cap:1;
            scales[row]=scale; cutoffs[row]=cutoff; nonzero=0;
        }
    }
    __syncthreads();
    const float inv=__frcp_rn(scale);
    for(int p=t;p<packed;p+=256) {
        const uint2 v=values[p]; unsigned int ph=0,pg=0,pr=0;
        #pragma unroll
        for(int j=0;j<4;++j) {
            const unsigned short bits=(j<2?v.x:v.y)>>(16*(j%2));
            const float x=(bits&0x7fff)<0x7c00?hfloat(bits):0;
            const float bounded=clip>0?fminf(cutoff,fmaxf(-cutoff,x)):x;
            if(clip>0 && x!=bounded) {
                const int index=row*k+atomicAdd(&nonzero,1);
                indices[index]=4*p+j; deltas[index]=x-bounded;
            }
            const int q=max(-cap,min(cap,__float2int_rn(bounded*inv)));
            const int hi=__float2int_rn(__fdiv_rn(static_cast<float>(q),static_cast<float>(base)));
            const int r=q-base*hi, mix=hi+r;
            int low=r;
            if(levels>0) {
                const int bin=min(levels-1,((2*r+base)*levels)/(2*base));
                low=2*bin+1-levels;
            }
            ph|=(static_cast<unsigned int>(hi)&255)<<(8*j);
            pg|=(static_cast<unsigned int>(mix)&255)<<(8*j);
            pr|=(static_cast<unsigned int>(low)&255)<<(8*j);
        }
        h[row*packed+p]=ph;g[row*packed+p]=pg;
        if(levels>=0) remainder[row*packed+p]=pr;
    }
    if(clip>0) {
        __syncthreads();
        if(t==0) counts[row]=nonzero;
    }
}

extern "C" __global__ void reconstruct(const int* p,const int* q,const int* r,
                                       const float* a,const float* b,float* output,
                                       int m,int n,int base,int levels_a,int levels_b) {
    const int i=blockIdx.y*blockDim.y+threadIdx.y,j=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=m||j>=n)return;
    const int index=i*n+j;
    const long long z=static_cast<long long>(base)*((base-1LL)*p[index]+q[index]);
    float adjusted;
    if(levels_a==0) adjusted=static_cast<float>(z-(base-1LL)*r[index]);
    else if(levels_a>0) {
        const float weight=static_cast<float>((base-1)*base*base)/(4.0f*levels_a*levels_b);
        adjusted=fmaf(-weight,static_cast<float>(r[index]),static_cast<float>(z));
    } else adjusted=static_cast<float>(z);
    output[index]=adjusted*(a[i]*b[j]);
}

extern "C" __global__ void transpose_half(const unsigned short* in,const float* cutoff,
                                          float* out,int rows,int k,int clip) {
    __shared__ float tile[32][33];
    const int t=threadIdx.x,s=threadIdx.y;
    int row=blockIdx.y*32+s,col=blockIdx.x*32+t;
    #pragma unroll
    for(int d=0;d<32;d+=8) if(row+d<rows && col<k) {
        float x=hfloat(in[(row+d)*k+col]);
        if(clip)x=fminf(cutoff[row+d],fmaxf(-cutoff[row+d],x));
        tile[s+d][t]=x;
    }
    __syncthreads();
    row=blockIdx.y*32+t; col=blockIdx.x*32+s;
    #pragma unroll
    for(int d=0;d<32;d+=8) if(row<rows && col+d<k) out[(col+d)*rows+row]=tile[t][s+d];
}

extern "C" __global__ void sparse_fp32(const unsigned short* indices,const float* values,
                                       const int* counts,const float* dense,float* output,
                                       int rows,int cols,int k,int add) {
    const int row=blockIdx.y*blockDim.y+threadIdx.y, col=blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=rows||col>=cols)return;
    float total=0;
    for(int item=0;item<counts[row];++item) {
        const int pos=row*k+item;
        total=fmaf(values[pos], dense[indices[pos]*cols+col], total);
    }
    if(add) output[row*cols+col]+=total;
    else output[row*cols+col]=total;
}

extern "C" __global__ void sparse_scc(const unsigned short* indices,const signed char* signs,
                                      const int* counts,const short* dense,int* output,
                                      int rows,int cols,int k) {
    const int row=blockIdx.y*blockDim.y+threadIdx.y, col=blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=rows||col>=cols)return;
    int total=0;
    for(int item=0;item<counts[row];++item) {
        const int pos=row*k+item;
        int value=dense[indices[pos]*cols+col];
        total+=signs[pos]>0?value:-value;
    }
    output[row*cols+col]=total;
}

extern "C" __global__ void add_transpose(const float* transpose,float* output,int m,int n) {
    __shared__ float tile[32][33];
    const int t=threadIdx.x,s=threadIdx.y;
    int row=blockIdx.y*32+s,col=blockIdx.x*32+t;
    #pragma unroll
    for(int d=0;d<32;d+=8) if(row+d<n && col<m) tile[s+d][t]=transpose[(row+d)*m+col];
    __syncthreads();
    row=blockIdx.y*32+t; col=blockIdx.x*32+s;
    #pragma unroll
    for(int d=0;d<32;d+=8) if(row<n && col+d<m) output[(col+d)*n+row]+=tile[t][s+d];
}
