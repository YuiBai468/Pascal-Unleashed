struct Residual { int k; float value; };

template<int K, bool Full, bool Table=false, bool OnlyTails=false, bool Shifted=false>
__device__ void pack_clipped(const unsigned int* input, unsigned short* hi, unsigned short* mixed,
                             unsigned short* full, float2* scales, unsigned short* counts, Residual* tail,
                             unsigned int* bad, int base, int cap, float clip, int capacity,
                             const unsigned short* codes=nullptr, int max_digit=0,
                             const float* shift_data=nullptr,int bucket_width=0,int shift_lane=0,
                             const unsigned int* keep=nullptr) {
    constexpr int threads=512, half_count=K/2, tiles=half_count/threads;
    constexpr int mask_count=(2*tiles+31)/32;
    __shared__ float energy_warp[16], peak_warp[16];
    __shared__ int prefix_warp[16];
    __shared__ float step, mul, max_scale, max_mul, limit, lookup_mul;
    const int lane=threadIdx.x&31, warp=threadIdx.x>>5, tid=threadIdx.x, row=blockIdx.x;
    const float base_inv=__frcp_rn(static_cast<float>(base));
    unsigned int regs[tiles];
    float energy=0, peak=0;
    #pragma unroll
    for (int tile=0;tile<tiles;++tile) {
        const int col=tile*threads+tid;
        unsigned int bits=input[row*half_count+col];
        if ((bits&0x7fff)>=0x7c00 || ((bits>>16)&0x7fff)>=0x7c00) {
            atomicOr(bad,1u);
            bits=0;
        }
        if (keep) bits&=__ldg(keep+col);
        regs[tile]=bits;
        const float x=decode_plane_half(static_cast<unsigned short>(bits));
        const float y=decode_plane_half(static_cast<unsigned short>(bits>>16));
        if (!Shifted) energy=fmaf(x,x,fmaf(y,y,energy));
        if (Full) peak=fmaxf(peak,fmaxf(fabsf(x),fabsf(y)));
    }
    if (Shifted) {
        if (!tid) {
            const float2 row_scale=scales[row];
            limit=row_scale.x*cap;
            lookup_mul=row_scale.y*max_digit/cap;
        }
    } else {
    for (int i=16;i;i>>=1) {
        energy+=__shfl_down_sync(0xffffffff,energy,i);
        if (Full) peak=fmaxf(peak,__shfl_down_sync(0xffffffff,peak,i));
    }
    if (!lane) {
        energy_warp[warp]=energy;
        if (Full) peak_warp[warp]=peak;
    }
    __syncthreads();
    if (!warp) {
        energy=lane<16?energy_warp[lane]:0;
        if (Full) peak=lane<16?peak_warp[lane]:0;
        for (int i=8;i;i>>=1) {
            energy+=__shfl_down_sync(0xffffffff,energy,i);
            if (Full) peak=fmaxf(peak,__shfl_down_sync(0xffffffff,peak,i));
        }
        if (!lane) {
            limit=clip*sqrtf(energy*(1.0f/K));
            step=limit>0?limit/cap:1.0f;
            mul=__frcp_rn(step);
            if (Table) lookup_mul=limit>0?max_digit/limit:0;
            max_scale=Full&&peak>0?peak*(1.0f/127):1;
            max_mul=__frcp_rn(max_scale);
            scales[row]=make_float2(step,Full?max_scale:mul);
        }
    }
    }
    __syncthreads();
    const int groups=Shifted?K/bucket_width:0;
    const float* shifts=Shifted?shift_data+(2*row+shift_lane)*groups:nullptr;
    unsigned int active[mask_count]={};
    #pragma unroll
    for (int tile=0;tile<tiles;++tile) {
        unsigned int hb=0,gb=0,fb=0;
        #pragma unroll
        for (int i=0;i<2;++i) {
            float x=decode_plane_half(static_cast<unsigned short>(regs[tile]>>(i*16)));
            if (Shifted) x-=__ldg(shifts+((tile*threads+tid)*2+i)/bucket_width);
            if (fabsf(x)>limit) active[(2*tile+i)/32]|=1u<<((2*tile+i)%32);
            if (OnlyTails) continue;
            int h,g;
            if (Table) {
                const int digit=__float2int_rn(fminf(max_digit,fabsf(x)*lookup_mul));
                const unsigned short value=__ldg(codes+digit);
                const int sign=x<0?-1:1;
                h=sign*static_cast<signed char>(value);
                g=sign*static_cast<signed char>(value>>8);
            } else {
                const int q=max(-cap,min(cap,__float2int_rn(x*mul)));
                h=__float2int_rn(q*base_inv);
                g=q-(base-1)*h;
                const int adjustment=(g>127)-(g<-127);
                h+=adjustment;
                g-=(base-1)*adjustment;
                if (h<-127 || h>127 || g<-127 || g>127) atomicOr(bad,2u);
            }
            hb|=(h&255u)<<(i*8);
            gb|=(g&255u)<<(i*8);
            if (Full) fb|=(__float2int_rn(x*max_mul)&255u)<<(i*8);
        }
        const int index=row*half_count+tile*threads+tid;
        if (!OnlyTails) {
            hi[index]=static_cast<unsigned short>(hb);
            mixed[index]=static_cast<unsigned short>(gb);
            if (Full) full[index]=static_cast<unsigned short>(fb);
        }
    }
    int count=0;
    #pragma unroll
    for (int i=0;i<mask_count;++i) count+=__popc(active[i]);
    int prefix=count;
    for (int i=1;i<32;i<<=1) {
        const int previous=__shfl_up_sync(0xffffffff,prefix,i);
        if (lane>=i) prefix+=previous;
    }
    if (lane==31) prefix_warp[warp]=prefix;
    __syncthreads();
    if (!warp) {
        int subtotal=lane<16?prefix_warp[lane]:0;
        for (int i=1;i<16;i<<=1) {
            const int previous=__shfl_up_sync(0xffffffff,subtotal,i);
            if (lane>=i) subtotal+=previous;
        }
        if (lane<16) prefix_warp[lane]=subtotal;
        if (lane==15) {
            if (subtotal>capacity) atomicOr(bad,4u);
            counts[row]=static_cast<unsigned short>(min(subtotal,capacity));
        }
    }
    __syncthreads();
    int cursor=prefix-count+(warp?prefix_warp[warp-1]:0);
    #pragma unroll
    for (int word=0;word<mask_count;++word) {
        unsigned int bits=active[word];
        while (bits && cursor<capacity) {
            const int cell=__ffs(bits)-1+32*word;
            const int col=(cell/2*threads+tid)*2+(cell&1);
            float x=decode_plane_half(static_cast<unsigned short>(regs[cell/2]>>((cell&1)*16)));
            if (Shifted) x-=__ldg(shifts+col/bucket_width);
            tail[row*capacity+cursor]=Residual{col, x-copysignf(limit,x)};
            ++cursor;
            bits&=bits-1;
        }
    }
}

#define CLIP_PACK(name,k,full) extern "C" __global__ __launch_bounds__(512) void name(\
    const unsigned int* in,unsigned short* h,unsigned short* g,unsigned short* f,float2* scale, \
    unsigned short* count,Residual* tail,unsigned int* bad,int base,int cap,float clip,int capacity) { \
    pack_clipped<k,full>(in,h,g,f,scale,count,tail,bad,base,cap,clip,capacity); }
CLIP_PACK(clip_a_8192,8192,false)
CLIP_PACK(clip_b_8192,8192,true)
CLIP_PACK(clip_a_16384,16384,false)
CLIP_PACK(clip_b_16384,16384,true)
CLIP_PACK(clip_a_32768,32768,false)
CLIP_PACK(clip_b_32768,32768,true)
CLIP_PACK(clip_a_65536,65536,false)
CLIP_PACK(clip_b_65536,65536,true)

#define CLIP_TABLE(name,k) extern "C" __global__ __launch_bounds__(512) void name(\
    const unsigned int* in,unsigned short* h,unsigned short* g,unsigned short* f,float2* scale, \
    unsigned short* count,Residual* tail,unsigned int* bad,int base,int cap,float clip,int capacity, \
    const unsigned short* codes,int max_digit) { \
    pack_clipped<k,false,true>(in,h,g,f,scale,count,tail,bad,base,cap,clip,capacity,codes,max_digit); }
CLIP_TABLE(clip_table_8192,8192)
CLIP_TABLE(clip_table_16384,16384)
CLIP_TABLE(clip_table_32768,32768)
CLIP_TABLE(clip_table_65536,65536)

#define CLIP_TAILS(name,k) extern "C" __global__ __launch_bounds__(512) void name(\
    const unsigned int* in,float2* scale,unsigned short* count,Residual* tail, \
    unsigned int* bad,int base,int cap,float clip,int capacity) { \
    pack_clipped<k,false,false,true>(in,nullptr,nullptr,nullptr,scale,count,tail,bad,base,cap,clip,capacity); }
CLIP_TAILS(clip_tails_8192,8192)
CLIP_TAILS(clip_tails_16384,16384)
CLIP_TAILS(clip_tails_32768,32768)
CLIP_TAILS(clip_tails_65536,65536)

#define CLIP_MASKED(name,k,only_tails) extern "C" __global__ __launch_bounds__(512) void name( \
    const unsigned int* in,unsigned short* h,unsigned short* g,float2* scale,unsigned short* count, \
    Residual* tail,unsigned int* bad,int cap,float clip,int capacity,const unsigned short* table, \
    int digits,const unsigned int* keep) { \
    pack_clipped<k,false,!only_tails,only_tails>(in,h,g,nullptr,scale,count,tail,bad, \
        1,cap,clip,capacity,table,digits,nullptr,0,0,keep); }
CLIP_MASKED(clip_hot_left_8192,8192,false)
CLIP_MASKED(clip_hot_left_16384,16384,false)
CLIP_MASKED(clip_hot_left_32768,32768,false)
CLIP_MASKED(clip_hot_left_65536,65536,false)
CLIP_MASKED(clip_hot_right_8192,8192,true)
CLIP_MASKED(clip_hot_right_16384,16384,true)
CLIP_MASKED(clip_hot_right_32768,32768,true)
CLIP_MASKED(clip_hot_right_65536,65536,true)

// Thread layout rotates inside a CTA. Each sparse gather then reads 32 adjacent elements.
template<int Iterations, int Order>
__device__ void restore_clip(const int* h,const int* g,const float2* scale_a,const float2* scale_b,
                              const unsigned short* count_a,const unsigned short* count_b,
                              const Residual* tails_a,const Residual* tails_b,const signed char* full_b,
                              const signed char* high_a,float* out,int m,int n,int base,int capacity,int panel) {
    __shared__ float result[32][33];
    const int lane=threadIdx.x&31, warp=threadIdx.x>>5;
    int tile_x=blockIdx.x,tile_y=blockIdx.y;
    if (Order==1) {
        const int ix=blockIdx.x+gridDim.x*blockIdx.y;
        tile_x=(ix/((m/32)*panel))*panel+ix%panel;
        tile_y=(ix/panel)%(m/32);
    } else if (Order==2) {
        const int ix=blockIdx.x+gridDim.x*blockIdx.y;
        tile_y=(ix/((n/32)*panel))*panel+ix%panel;
        tile_x=(ix/panel)%(n/32);
    }
    const int i0=32*tile_y,j0=32*tile_x;
    const int j=j0+lane;
    const float2 cb=__ldg(scale_b+j);
    #pragma unroll
    for (int step=0;step<Iterations;++step) {
        const int t=step*(32/Iterations)+warp,i=i0+t;
        const int size=__ldg(count_a+i);
        const Residual* sparse=tails_a+i*capacity;
        float partial=0;
        for (int p=0;p<size;++p) {
            const int k=__ldg(&sparse[p].k);
            const float x=__ldg(&sparse[p].value);
            partial=fmaf(x,static_cast<float>(__ldg(full_b+static_cast<size_t>(k)*n+j)),partial);
        }
        const size_t index=static_cast<size_t>(i)*n+j;
        const float w=__ldg(&scale_a[i].x)*cb.x*base;
        result[t][lane]=fmaf(partial,cb.y,w*fmaf(static_cast<float>(base-1),h[index],g[index]));
    }
    __syncthreads();
    const int i=i0+lane;
    const float a_gain=__ldg(&scale_a[i].x)*base;
    #pragma unroll
    for (int step=0;step<Iterations;++step) {
        const int t=step*(32/Iterations)+warp,col=j0+t;
        const int size=__ldg(count_b+col);
        const Residual* sparse=tails_b+col*capacity;
        float partial=0;
        for (int p=0;p<size;++p) {
            const int k=__ldg(&sparse[p].k);
            const float y=__ldg(&sparse[p].value);
            partial=fmaf(y,static_cast<float>(__ldg(high_a+static_cast<size_t>(k)*m+i)),partial);
        }
        result[lane][t]=fmaf(partial,a_gain,result[lane][t]);
    }
    __syncthreads();
    #pragma unroll
    for (int step=0;step<Iterations;++step) {
        const int t=step*(32/Iterations)+warp;
        out[static_cast<size_t>(i0+t)*n+j]=result[t][lane];
    }
}

#define CLIP_FINISH(name,iterations,order) extern "C" __global__ void name( \
    const int* h,const int* g,const float2* a,const float2* b,const unsigned short* ca,const unsigned short* cb, \
    const Residual* ra,const Residual* rb,const signed char* fb,const signed char* ha,float* out,int m,int n,int base, \
    int capacity,int panel) {restore_clip<iterations,order>(h,g,a,b,ca,cb,ra,rb,fb,ha,out,m,n,base,capacity,panel);}
CLIP_FINISH(clip_finish_256,4,0)
CLIP_FINISH(clip_finish_512,2,0)
CLIP_FINISH(clip_finish_cols,4,1)
CLIP_FINISH(clip_finish_rows,4,2)
