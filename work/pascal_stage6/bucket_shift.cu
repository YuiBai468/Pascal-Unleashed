// A common bucket indicator basis makes row-specific shifts an exact low-rank correction.
extern "C" __global__ void bucket_rows(const unsigned int* input,float* factors,float2* scales,
                                       unsigned* invalid,int k,int width,int cap,float clip,int right) {
    __shared__ float lows[512],highs[512],sums[512],warp_energy[8],limit;
    const int tid=threadIdx.x,lane=tid%32,warp=tid/32;
    const int groups=k/width,row=blockIdx.x;
    const size_t initial=static_cast<size_t>(row)*(k/2);
    float energy=0;
    for (int group=warp;group<groups;group+=8) {
        float lo=0,hi=0,sum=0;
        for (int col=lane;col<width/2;col+=32) {
            unsigned int bits=input[initial+group*(width/2)+col];
            if ((bits&0x7fff)>=0x7c00 || ((bits>>16)&0x7fff)>=0x7c00) {
                atomicOr(invalid,1u);
                bits=0;
            }
            const float x=decode_plane_half(static_cast<unsigned short>(bits));
            const float y=decode_plane_half(static_cast<unsigned short>(bits>>16));
            lo=fminf(lo,fminf(x,y));
            hi=fmaxf(hi,fmaxf(x,y));
            sum+=x+y;
            energy=fmaf(x,x,fmaf(y,y,energy));
        }
        for (int d=16;d;d>>=1) {
            lo=fminf(lo,__shfl_down_sync(0xffffffff,lo,d));
            hi=fmaxf(hi,__shfl_down_sync(0xffffffff,hi,d));
            sum+=__shfl_down_sync(0xffffffff,sum,d);
        }
        if (!lane) {
            lows[group]=lo;
            highs[group]=hi;
            sums[group]=sum;
        }
    }
    for (int d=16;d;d>>=1) energy+=__shfl_down_sync(0xffffffff,energy,d);
    if (!lane) warp_energy[warp]=energy;
    __syncthreads();
    if (!tid) {
        float total=0;
        for (int i=0;i<8;++i) total+=warp_energy[i];
        const float step=total>0?clip*sqrtf(total/k)/cap:1.0f;
        limit=step*cap;
        scales[row]=make_float2(step,__frcp_rn(step));
    }
    __syncthreads();
    float* first=factors+static_cast<size_t>(row)*(2*groups);
    float* last=first+groups;
    for (int group=tid;group<groups;group+=256) {
        const float hi=highs[group],lo=lows[group];
        const float lower=hi-limit,upper=lo+limit;
        const float shift=lower<=upper?fminf(fmaxf(0,lower),upper):(hi+lo)*.5f;
        first[group]=right?sums[group]:shift;
        last[group]=right?shift:fmaf(-width,shift,sums[group]);
    }
}

#define BUCKET_PACK(name,k,only_tails) extern "C" __global__ __launch_bounds__(512) void name( \
    const unsigned int* input,unsigned short* h,unsigned short* g,float2* scales, \
    unsigned short* counts,Residual* tail,unsigned int* bad,int cap,int capacity, \
    const unsigned short* table,int digits,const float* factors,int width,int shift_lane) { \
    pack_clipped<k,false,!only_tails,only_tails,true>(input,h,g,nullptr,scales,counts,tail,bad, \
        0,cap,0.f,capacity,table,digits,factors,width,shift_lane); }
BUCKET_PACK(bucket_left_8192,8192,false)
BUCKET_PACK(bucket_left_16384,16384,false)
BUCKET_PACK(bucket_left_32768,32768,false)
BUCKET_PACK(bucket_right_8192,8192,true)
BUCKET_PACK(bucket_right_16384,16384,true)
BUCKET_PACK(bucket_right_32768,32768,true)

template<bool Encode>
__device__ void bucket_transpose(const unsigned short* source,signed char* high,signed char* mixed,
                                 signed char* coarse,const float2* scales,const unsigned short* lookup,
                                 const float* factors,int rows,int cols,int max_digit,float lookup_gain,
                                 float inverse_gain,int width) {
    __shared__ signed char h[Encode?64:1][Encode?68:1],g[Encode?64:1][Encode?68:1],q[64][68];
    const int row0=64*blockIdx.y,col0=64*blockIdx.x,groups=cols/width;
    const int x=threadIdx.x%64,y=threadIdx.x/64;
    #pragma unroll
    for (int i=y;i<64;i+=4) {
        const int row=row0+i;
        const float inverse_step=__ldg(&scales[row].y);
        const float shift=__ldg(factors+row*2*groups+(Encode?groups:0)+col0/width);
        const float v=decode_plane_half(source[static_cast<size_t>(row)*cols+col0+x])-shift;
        if (Encode) {
            const float t=fabsf(v)*(inverse_step*lookup_gain);
            const unsigned short digits=__ldg(lookup+__float2int_rn(fminf(max_digit,t)));
            const int sign=v<0?-1:1;
            h[i][x]=static_cast<signed char>(digits)*sign;
            g[i][x]=static_cast<signed char>(digits>>8)*sign;
        }
        q[i][x]=static_cast<signed char>(__float2int_rn(fminf(127,fmaxf(-127,v*(inverse_step*inverse_gain)))));
    }
    __syncthreads();
    #pragma unroll
    for (int i=y;i<64;i+=4) {
        const int k=col0+i;
        if (Encode) {
            const size_t at=static_cast<size_t>(k)*rows+row0+x;
            high[at]=h[x][i];
            mixed[at]=g[x][i];
        }
        coarse[(static_cast<size_t>(blockIdx.y)*cols+k)*64+x]=q[x][i];
    }
}

#define BUCKET_TRANSPOSE(name,encode) extern "C" __global__ void name( \
    const unsigned short* source,signed char* high,signed char* mixed,signed char* coarse, \
    const float2* scales,const unsigned short* lookup,const float* factors,int rows,int cols, \
    int max_digit,float lookup_gain,float inverse_gain,int width) { \
    bucket_transpose<encode>(source,high,mixed,coarse,scales,lookup,factors,rows,cols, \
                             max_digit,lookup_gain,inverse_gain,width); }
BUCKET_TRANSPOSE(bucket_transpose_right,true)
BUCKET_TRANSPOSE(bucket_transpose_left,false)
