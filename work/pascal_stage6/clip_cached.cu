extern "C" __global__ void group_tail(const Residual* tail,const unsigned short* count,
                                     Residual* ordered,unsigned short* offsets,int k,int chunk,int stride) {
    const int row=blockIdx.x,tid=threadIdx.x;
    const int groups=k/chunk,n=__ldg(count+row);
    const Residual* input=tail+row*stride;
    __shared__ int pos[512];
    for (int g=tid;g<groups;g+=128) pos[g]=0;
    __syncthreads();
    for (int i=tid;i<n;i+=128) atomicAdd(pos+input[i].k/chunk,1);
    __syncthreads();
    if (!tid) {
        unsigned short* out=offsets+row*(groups+1);
        int sum=0;
        for (int g=0;g<groups;++g) {
            out[g]=sum;
            const int last=pos[g];
            pos[g]=sum;
            sum+=last;
        }
        out[groups]=sum;
    }
    __syncthreads();
    for (int i=tid;i<n;i+=128) {
        const Residual r=input[i];
        ordered[row*stride+atomicAdd(pos+r.k/chunk,1)]=r;
    }
}

template<int Chunk,bool Right>
__device__ void cached_clip(const int* h,const int* g,const float2* sa,const float2* sb,
                              const Residual* tail,const unsigned short* offsets,
                              const unsigned int* dense,float* out,int m,int n,int k,int base,int stride) {
    const int rows=Right?n:m, cols=Right?m:n;
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5,tid=threadIdx.x;
    const int first_row=blockIdx.x*64,first_col=blockIdx.y*128,col=first_col+4*lane;
    const int parts=k/Chunk;
    union Work { unsigned data[Chunk][32]; float rotated[64][129]; };
    __shared__ Work shared;
    __shared__ unsigned int needed[Chunk/32];
    __shared__ int starts[64],ends[64];
    float4 answer[8];
    #pragma unroll
    for (int i=0;i<8;++i) answer[i]=make_float4(0,0,0,0);
    for (int block=0;block<parts;++block) {
        if (tid<Chunk/32) needed[tid]=0;
        if (tid<64) {
            const int row=first_row+tid;
            starts[tid]=__ldg(offsets+row*(parts+1)+block);
            ends[tid]=__ldg(offsets+row*(parts+1)+block+1);
        }
        __syncthreads();
        if (tid<64) {
            const Residual* ptr=tail+(first_row+tid)*stride;
            for (int p=starts[tid];p<ends[tid];++p) {
                const int q=__ldg(&ptr[p].k)%Chunk;
                atomicOr(needed+(q/32),1u<<(q%32));
            }
        }
        __syncthreads();
        for (int p=warp;p<Chunk/32;p+=8) {
            unsigned int bits=needed[p];
            while (bits) {
                const int z=p*32+__ffs(bits)-1;
                shared.data[z][lane]=__ldg(dense+static_cast<size_t>(block*Chunk+z)*(cols/4)+col/4);
                bits&=bits-1;
            }
        }
        __syncthreads();
        #pragma unroll 2
        for (int i=0;i<8;++i) {
            const int y=8*i+warp;
            const Residual* ptr=tail+(first_row+y)*stride;
            float4 sum=answer[i];
            for (int p=starts[y];p<ends[y];++p) {
                const int z=__ldg(&ptr[p].k)%Chunk;
                const float r=__ldg(&ptr[p].value);
                const unsigned int q=shared.data[z][lane];
                sum.x=fmaf(r,static_cast<signed char>(q),sum.x);
                sum.y=fmaf(r,static_cast<signed char>(q>>8),sum.y);
                sum.z=fmaf(r,static_cast<signed char>(q>>16),sum.z);
                sum.w=fmaf(r,static_cast<signed char>(q>>24),sum.w);
            }
            answer[i]=sum;
        }
        __syncthreads();
    }
    if (Right) {
        const float a0=__ldg(&sa[col].x)*base,a1=__ldg(&sa[col+1].x)*base;
        const float a2=__ldg(&sa[col+2].x)*base,a3=__ldg(&sa[col+3].x)*base;
        #pragma unroll
        for (int i=0;i<8;++i) {
            const int row=8*i+warp;
            shared.rotated[row][4*lane]=answer[i].x*a0;
            shared.rotated[row][4*lane+1]=answer[i].y*a1;
            shared.rotated[row][4*lane+2]=answer[i].z*a2;
            shared.rotated[row][4*lane+3]=answer[i].w*a3;
        }
        __syncthreads();
        #pragma unroll 4
        for (int row=warp;row<128;row+=8) {
            float2* dst=reinterpret_cast<float2*>(out+static_cast<size_t>(first_col+row)*n+first_row+2*lane);
            float2 now=*dst;
            now.x+=shared.rotated[2*lane][row];
            now.y+=shared.rotated[2*lane+1][row];
            *dst=now;
        }
    } else {
        const float2 b0=__ldg(sb+col),b1=__ldg(sb+col+1),b2=__ldg(sb+col+2),b3=__ldg(sb+col+3);
        #pragma unroll
        for (int i=0;i<8;++i) {
            const int row=first_row+8*i+warp;
            const size_t at=static_cast<size_t>(row)*n+col;
            const float a=__ldg(&sa[row].x)*base;
            const int4 hi=*reinterpret_cast<const int4*>(h+at),gm=*reinterpret_cast<const int4*>(g+at);
            const float v=static_cast<float>(base-1);
            *reinterpret_cast<float4*>(out+at)=make_float4(
                    fmaf(answer[i].x,b0.y,(a*b0.x)*fmaf(v,hi.x,gm.x)),
                    fmaf(answer[i].y,b1.y,(a*b1.x)*fmaf(v,hi.y,gm.y)),
                    fmaf(answer[i].z,b2.y,(a*b2.x)*fmaf(v,hi.z,gm.z)),
                    fmaf(answer[i].w,b3.y,(a*b3.x)*fmaf(v,hi.w,gm.w)));
        }
    }
}

#define CACHE_CLIP(name,chunk,right) extern "C" __global__ void name( \
    const int*h,const int*g,const float2*sa,const float2*sb,const Residual*tail,const unsigned short*offsets, \
    const unsigned int*data,float*out,int m,int n,int k,int base,int stride) { \
    cached_clip<chunk,right>(h,g,sa,sb,tail,offsets,data,out,m,n,k,base,stride); }
CACHE_CLIP(cache_left_128,128,false)
CACHE_CLIP(cache_right_128,128,true)
CACHE_CLIP(cache_left_256,256,false)
CACHE_CLIP(cache_right_256,256,true)
