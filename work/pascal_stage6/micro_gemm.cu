template<int M, int N, int A, int B, int K, int Threads, bool Left>
__device__ __forceinline__ void load_micro_tile(const int* source, int start, int depth, int stride,
                                               int4* values) {
    constexpr int extent=Left?M:N;
    constexpr int vectors=extent*K/4;
    #pragma unroll
    for(int i=0;i<(vectors+Threads-1)/Threads;++i) {
        const int index=threadIdx.x+i*Threads;
        if(index<vectors) {
            const int row=index/(K/4),word=4*(index%(K/4));
            values[i]=__ldg(reinterpret_cast<const int4*>(source+(start+row)*stride+depth+word));
        }
    }
}

template<int Rows, int Micro, int K, int Threads>
__device__ __forceinline__ void store_micro_tile(int (&tile)[K][Rows], const int4* data) {
    constexpr int vectors=Rows*K/4;
    #pragma unroll
    for(int i=0;i<(vectors+Threads-1)/Threads;++i) {
        const int index=threadIdx.x+i*Threads;
        if(index<vectors) {
            const int row=index/(K/4),depth=4*(index%(K/4));
            const int location=(row/Micro)*4+(row&3)+(row%Micro/4)*(Rows/(Micro/4));
            const int4 value=data[i];
            tile[depth][location]=value.x;
            tile[depth+1][location]=value.y;
            tile[depth+2][location]=value.z;
            tile[depth+3][location]=value.w;
        }
    }
}

template<int M, int N, int A, int B, int K, int Unroll>
__device__ void micro_gemm(const int* left, const int* right, const float* as, const float* bs,
                            float* output, int n, int words) {
    constexpr int lanes=M*N/(A*B), col_groups=N/B;
    __shared__ __align__(16) int at[2][K][M], bt[2][K][N];
    const int first_row=blockIdx.y*M, first_col=blockIdx.x*N;
    const int row_group=threadIdx.x/col_groups, col_group=threadIdx.x%col_groups;
    int sums[A][B]={};
    int4 av[(M*K/4+lanes-1)/lanes],bv[(N*K/4+lanes-1)/lanes];
    load_micro_tile<M,N,A,B,K,lanes,true>(left,first_row,0,words,av);
    load_micro_tile<M,N,A,B,K,lanes,false>(right,first_col,0,words,bv);
    store_micro_tile<M,A,K,lanes>(at[0],av);
    store_micro_tile<N,B,K,lanes>(bt[0],bv);
    __syncthreads();
    for(int base=0;base<words;base+=K) {
        const int next=base+K, active=(base/K)&1;
        if(next<words) {
            load_micro_tile<M,N,A,B,K,lanes,true>(left,first_row,next,words,av);
            load_micro_tile<M,N,A,B,K,lanes,false>(right,first_col,next,words,bv);
        }
        #pragma unroll Unroll
        for(int depth=0;depth<K;++depth) {
            int x[A],y[B];
            #pragma unroll
            for(int i=0;i<A/4;++i) {
                const int4 v=*reinterpret_cast<const int4*>(at[active][depth]+4*row_group+i*(M/(A/4)));
                x[4*i]=v.x; x[4*i+1]=v.y; x[4*i+2]=v.z; x[4*i+3]=v.w;
            }
            #pragma unroll
            for(int i=0;i<B/4;++i) {
                const int4 v=*reinterpret_cast<const int4*>(bt[active][depth]+4*col_group+i*(N/(B/4)));
                y[4*i]=v.x; y[4*i+1]=v.y; y[4*i+2]=v.z; y[4*i+3]=v.w;
            }
            #pragma unroll
            for(int i=0;i<A;++i) {
                #pragma unroll
                for(int j=0;j<B;++j) sums[i][j]=__dp4a(x[i],y[j],sums[i][j]);
            }
        }
        if(next<words) {
            store_micro_tile<M,A,K,lanes>(at[active^1],av);
            store_micro_tile<N,B,K,lanes>(bt[active^1],bv);
            __syncthreads();
        }
    }
    const int col=first_col+B*col_group;
    float4 sc[B/4];
    #pragma unroll
    for(int j=0;j<B/4;++j) sc[j]=__ldg(reinterpret_cast<const float4*>(bs+col+4*j));
    #pragma unroll
    for(int i=0;i<A;++i) {
        const int row=first_row+A*row_group+i;
        const float s=__ldg(as+row);
        float4* dest=reinterpret_cast<float4*>(output+row*n+col);
        #pragma unroll
        for(int j=0;j<B/4;++j) {
            const float4 f=sc[j];
            dest[j]=make_float4(sums[i][4*j]*(s*f.x),sums[i][4*j+1]*(s*f.y),
                                sums[i][4*j+2]*(s*f.z),sums[i][4*j+3]*(s*f.w));
        }
    }
}

#define MICRO(name,m,n,a,b,k,u,ctas) extern "C" __global__ __launch_bounds__(m*n/(a*b),ctas) \
void name(const int* l,const int* r,const float* as,const float* bs,float* out,int cols,int depth) { \
    micro_gemm<m,n,a,b,k,u>(l,r,as,bs,out,cols,depth/4); }
MICRO(micro_128_64_8x4_32,128,64,8,4,8,2,3)
MICRO(micro_128_64_8x4_64,128,64,8,4,16,2,3)
MICRO(micro_64_128_4x8_64,64,128,4,8,16,2,3)
MICRO(micro_128_128_8x4_64,128,128,8,4,16,4,2)
MICRO(micro_128_128_4x8_64,128,128,4,8,16,4,2)
MICRO(micro_128_128_8x8_64,128,128,8,8,16,1,2)
MICRO(micro_128_128_8x8_32,128,128,8,8,8,1,3)
