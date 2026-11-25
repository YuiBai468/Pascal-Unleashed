__device__ void transpose_trained(const unsigned short* source,signed char* high,signed char* mixed,
                                                signed char* coarse,const float2* scales,const unsigned short* lookup,
                                                int rows,int cols,int max_digit,float lookup_gain,float inverse_gain,
                                                int packed,const unsigned short* keep) {
    __shared__ signed char h[64][68],g[64][68],q[64][68];
    const int row0=64*blockIdx.y,col0=64*blockIdx.x;
    const int x=threadIdx.x&63,y=threadIdx.x>>6;
    #pragma unroll
    for (int i=0;i<64;i+=4) {
        const int row=row0+i+y;
        const float inverse_step=__ldg(&scales[row].y);
        unsigned short bits=source[static_cast<size_t>(row)*cols+col0+x];
        if (keep) bits&=__ldg(keep+col0+x);
        const float v=decode_plane_half(bits);
        const float t=fabsf(v)*(inverse_step*lookup_gain);
        const unsigned short digits=__ldg(lookup+__float2int_rn(fminf(max_digit,t)));
        const int sign=v<0?-1:1;
        h[i+y][x]=static_cast<signed char>(digits)*sign;
        g[i+y][x]=static_cast<signed char>(digits>>8)*sign;
        q[i+y][x]=static_cast<signed char>(__float2int_rn(fminf(127,fmaxf(-127,v*(inverse_step*inverse_gain)))));
    }
    __syncthreads();
    #pragma unroll
    for (int i=0;i<64;i+=4) {
        const int k=col0+i+y;
        const size_t normal=static_cast<size_t>(k)*rows+row0+x;
        high[normal]=h[x][i+y];
        mixed[normal]=g[x][i+y];
        const bool wide=packed==128;
        const size_t tail=packed?(static_cast<size_t>(blockIdx.y>>wide)*cols+k)*packed+(wide?(blockIdx.y&1)*64:0)+x:normal;
        coarse[tail]=q[x][i+y];
    }
}

extern "C" __global__ void transpose_trained_64(const unsigned short* source,signed char* high,signed char* mixed,
                                                signed char* coarse,const float2* scales,const unsigned short* lookup,
                                                int rows,int cols,int max_digit,float lookup_gain,float inverse_gain,int packed) {
    transpose_trained(source,high,mixed,coarse,scales,lookup,rows,cols,max_digit,lookup_gain,inverse_gain,packed,nullptr);
}

extern "C" __global__ void transpose_trained_hot(const unsigned short* source,signed char* high,signed char* mixed,
                                                 signed char* coarse,const float2* scales,const unsigned short* lookup,
                                                 int rows,int cols,int max_digit,float lookup_gain,float inverse_gain,int packed,
                                                 const unsigned short* keep) {
    transpose_trained(source,high,mixed,coarse,scales,lookup,rows,cols,max_digit,lookup_gain,inverse_gain,packed,keep);
}

__device__ void transpose_coarse(const unsigned short* source,signed char* target,const float2* scales,int rows,
                                 int cols,float inverse_gain,int packed,const unsigned short* keep) {
    __shared__ signed char tile[64][68];
    const int row0=64*blockIdx.y,col0=64*blockIdx.x;
    const int x=threadIdx.x&63,y=threadIdx.x>>6;
    #pragma unroll
    for (int i=0;i<64;i+=4) {
        const int row=row0+i+y;
        const float inverse_step=__ldg(&scales[row].y)*inverse_gain;
        unsigned short bits=source[static_cast<size_t>(row)*cols+col0+x];
        if (keep) bits&=__ldg(keep+col0+x);
        const float v=decode_plane_half(bits);
        tile[i+y][x]=static_cast<signed char>(__float2int_rn(fminf(127,fmaxf(-127,v*inverse_step))));
    }
    __syncthreads();
    if (packed) {
        const bool wide=packed==128;
        #pragma unroll
        for (int i=0;i<64;i+=4) {
            const size_t at=(static_cast<size_t>(blockIdx.y>>wide)*cols+col0+i+y)*packed+(wide?(blockIdx.y&1)*64:0)+x;
            target[at]=tile[x][i+y];
        }
    } else {
        #pragma unroll
        for (int i=0;i<64;i+=4) target[static_cast<size_t>(col0+i+y)*rows+row0+x]=tile[x][i+y];
    }
}

extern "C" __global__ void transpose_coarse_64(const unsigned short* source,signed char* target,const float2* scales,
                                               int rows,int cols,float inverse_gain,int packed) {
    transpose_coarse(source,target,scales,rows,cols,inverse_gain,packed,nullptr);
}

extern "C" __global__ void transpose_coarse_hot(const unsigned short* source,signed char* target,const float2* scales,
                                                int rows,int cols,float inverse_gain,int packed,const unsigned short* keep) {
    transpose_coarse(source,target,scales,rows,cols,inverse_gain,packed,keep);
}
template <typename Dense>
__device__ __forceinline__ float2 decode_tail_pair(Dense packed);

template <>
__device__ __forceinline__ float2 decode_tail_pair(unsigned int packed) {
    return make_float2(decode_plane_half(static_cast<unsigned short>(packed)),
                       decode_plane_half(static_cast<unsigned short>(packed>>16)));
}

template <>
__device__ __forceinline__ float2 decode_tail_pair(unsigned short packed) {
    return make_float2(static_cast<signed char>(packed),static_cast<signed char>(packed>>8));
}

// One CTA reconstructs an output tile and applies both clipping corrections.
template <typename Dense>
__device__ __forceinline__ float2 tail_dot(const Residual* tails, int count, const Dense* dense, int ld) {
    float2 even=make_float2(0.0f,0.0f), odd=even;
    int p=0;
    for (;p+1<count;p+=2) {
        const Residual u=tails[p], v=tails[p+1];
        const auto a=decode_tail_pair(__ldg(dense+static_cast<size_t>(u.k)*ld));
        const auto b=decode_tail_pair(__ldg(dense+static_cast<size_t>(v.k)*ld));
        even.x=fmaf(u.value,a.x,even.x);
        even.y=fmaf(u.value,a.y,even.y);
        odd.x=fmaf(v.value,b.x,odd.x);
        odd.y=fmaf(v.value,b.y,odd.y);
    }
    if (p<count) {
        const Residual u=tails[p];
        const auto a=decode_tail_pair(__ldg(dense+static_cast<size_t>(u.k)*ld));
        even.x=fmaf(u.value,a.x,even.x);
        even.y=fmaf(u.value,a.y,even.y);
    }
    return make_float2(even.x+odd.x,even.y+odd.y);
}

template<int Unroll, typename Dense>
__device__ __forceinline__ float2 tail_dot_unrolled(const Residual* tails,int count,const Dense* dense,int ld) {
    float2 partial[Unroll]={};
    int p=0;
    for (;p+Unroll<=count;p+=Unroll) {
        int2 indices[Unroll];
        Dense values[Unroll];
        #pragma unroll
        for (int i=0;i<Unroll;++i) indices[i]=__ldg(reinterpret_cast<const int2*>(tails+p+i));
        #pragma unroll
        for (int i=0;i<Unroll;++i) values[i]=__ldg(dense+static_cast<size_t>(indices[i].x)*ld);
        #pragma unroll
        for (int i=0;i<Unroll;++i) {
            const float2 q=decode_tail_pair(values[i]);
            const float w=__int_as_float(indices[i].y);
            partial[i].x=fmaf(w,q.x,partial[i].x);
            partial[i].y=fmaf(w,q.y,partial[i].y);
        }
    }
    for (;p<count;++p) {
        const int2 t=__ldg(reinterpret_cast<const int2*>(tails+p));
        const float2 q=decode_tail_pair(__ldg(dense+static_cast<size_t>(t.x)*ld));
        partial[0].x=fmaf(__int_as_float(t.y),q.x,partial[0].x);
        partial[0].y=fmaf(__int_as_float(t.y),q.y,partial[0].y);
    }
    #pragma unroll
    for (int i=1;i<Unroll;++i) {
        partial[0].x+=partial[i].x;
        partial[0].y+=partial[i].y;
    }
    return partial[0];
}

template <int Warps, bool Panel, bool CorrectionOnly = false, int Unroll = 2, typename Dense>
__device__ void tail_tile(const int* high,const int* mixed,const float2* sa,const float2* sb,
                          const unsigned short* count_a,const unsigned short* count_b,
                          const Residual* tail_a,const Residual* tail_b,const Dense* dense_a,
                          const Dense* dense_b,float* out,int capacity,int m,int n,
                          float weight_ratio,float weight_g,int panel,float gain_a,float gain_b,int packed_k) {
    __shared__ float tile[64][66];
    constexpr bool bytes=sizeof(Dense)==2;
    const int lane=threadIdx.x&31, warp=threadIdx.x>>5;
    int tile_x=blockIdx.x,tile_y=blockIdx.y;
    if (Panel) {
        const int tile_index=blockIdx.x+gridDim.x*blockIdx.y;
        tile_x=(tile_index/(panel*(m/64)))*panel+tile_index%panel;
        tile_y=(tile_index/panel)%(m/64);
    }
    const int row=tile_y*64,col=tile_x*64;
    const int x=col+2*lane,y=row+2*lane;
    const Dense* left=packed_k?dense_a+static_cast<size_t>(tile_y)*packed_k*32+lane:dense_a+y/2;
    const Dense* right=packed_k?dense_b+static_cast<size_t>(tile_x)*packed_k*32+lane:dense_b+x/2;
    const int left_ld=packed_k?32:m/2,right_ld=packed_k?32:n/2;
    const float bx=__ldg(&sb[x].x),bx1=__ldg(&sb[x+1].x);
    const float sb0=bx*weight_g,sb1=bx1*weight_g;
    const float correction_b0=bytes?bx*gain_b:1.0f,correction_b1=bytes?bx1*gain_b:1.0f;
    #pragma unroll 2
    for (int i=warp;i<64;i+=Warps) {
        const int r=row+i;
        const int count=__ldg(count_a+r);
        const auto correction=Unroll==2?tail_dot(tail_a+r*capacity,count,right,right_ld):
                                         tail_dot_unrolled<Unroll>(tail_a+r*capacity,count,right,right_ld);
        float first=correction_b0*correction.x, second=correction_b1*correction.y;
        if (!CorrectionOnly) {
            const size_t at=static_cast<size_t>(r)*n+x;
            const int2 h=*reinterpret_cast<const int2*>(high+at),g=*reinterpret_cast<const int2*>(mixed+at);
            const float a=__ldg(&sa[r].x);
            first=fmaf(a*sb0,fmaf(weight_ratio,h.x,g.x),first);
            second=fmaf(a*sb1,fmaf(weight_ratio,h.y,g.y),second);
        }
        tile[i][2*lane]=first;
        tile[i][2*lane+1]=second;
    }
    __syncthreads();
    const float correction_a0=bytes?__ldg(&sa[y].x)*gain_a:1.0f;
    const float correction_a1=bytes?__ldg(&sa[y+1].x)*gain_a:1.0f;
    #pragma unroll 2
    for (int j=warp;j<64;j+=Warps) {
        const int c=col+j;
        const int count=__ldg(count_b+c);
        const auto correction=Unroll==2?tail_dot(tail_b+c*capacity,count,left,left_ld):
                                         tail_dot_unrolled<Unroll>(tail_b+c*capacity,count,left,left_ld);
        tile[2*lane][j]=fmaf(correction.x,correction_a0,tile[2*lane][j]);
        tile[2*lane+1][j]=fmaf(correction.y,correction_a1,tile[2*lane+1][j]);
    }
    __syncthreads();
    #pragma unroll 2
    for (int i=warp;i<64;i+=Warps) {
        const size_t at=static_cast<size_t>(row+i)*n+x;
        *reinterpret_cast<float2*>(out+at)=make_float2(tile[i][2*lane],tile[i][2*lane+1]);
    }
}

#define TAIL_TILE(name, warps, panel, dense, correction_only, unroll) extern "C" __global__ void name( \
    const int* h,const int* g,const float2* sa,const float2* sb,const unsigned short* ca, \
    const unsigned short* cb,const Residual* ra,const Residual* rb,const dense* a,const dense* b,float* out, \
    int capacity,int m,int n,float weight_ratio,float weight_g,int panel_width,float gain_a,float gain_b,int packed_k) { \
    tail_tile<warps,panel,correction_only,unroll>(h,g,sa,sb,ca,cb,ra,rb,a,b,out,capacity,m,n,weight_ratio,weight_g,panel_width,gain_a,gain_b, \
                          packed_k); \
}

TAIL_TILE(tail_tile_4,4,false,unsigned int,false,2)
TAIL_TILE(tail_tile_8,8,false,unsigned int,false,2)
TAIL_TILE(tail_tile_panel_8,8,true,unsigned int,false,2)
TAIL_TILE(tail_tile_bytes,8,true,unsigned short,false,2)
TAIL_TILE(tail_tile_bytes_u4,8,true,unsigned short,false,4)
TAIL_TILE(tail_tile_bytes_16,16,true,unsigned short,false,4)
TAIL_TILE(tail_tile_bytes_u8,16,true,unsigned short,false,8)
TAIL_TILE(tail_correction_bytes,8,true,unsigned short,true,2)

extern "C" __global__ void tail_direction_b(const float2* sa,const unsigned short* counts,
                                             const Residual* tails,const unsigned short* dense,
                                             float* output,int n,int k,int capacity,float gain) {
    __shared__ float transposed[64][66];
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    const int row=64*blockIdx.y,col=64*blockIdx.x;
    const unsigned short* panel=dense+static_cast<size_t>(blockIdx.y)*k*32+lane;
    const float a0=__ldg(&sa[row+2*lane].x)*gain,a1=__ldg(&sa[row+2*lane+1].x)*gain;
    for (int j=warp;j<64;j+=8) {
        const int c=col+j;
        const float2 v=tail_dot_unrolled<4>(tails+c*capacity,__ldg(counts+c),panel,32);
        transposed[2*lane][j]=a0*v.x;
        transposed[2*lane+1][j]=a1*v.y;
    }
    __syncthreads();
    for (int i=warp;i<64;i+=8) {
        const size_t at=static_cast<size_t>(row+i)*n+col+2*lane;
        *reinterpret_cast<float2*>(output+at)=make_float2(transposed[i][2*lane],transposed[i][2*lane+1]);
    }
}

extern "C" __global__ void tail_direction_a(const int* high,const int* mixed,const float2* sa,const float2* sb,
                                             const unsigned short* counts,const Residual* tails,
                                             const unsigned short* dense,float* output,int m,int n,int k,int capacity,
                                             float ratio,float weight,float gain) {
    const int index=blockIdx.x+gridDim.x*blockIdx.y;
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    const int tile_x=index/(m/64),row=64*(index%(m/64)),col=64*tile_x+2*lane;
    const unsigned short* panel=dense+static_cast<size_t>(tile_x)*k*32+lane;
    const float b0=__ldg(&sb[col].x),b1=__ldg(&sb[col+1].x);
    for (int i=warp;i<64;i+=8) {
        const int r=row+i;
        const float2 v=tail_dot_unrolled<4>(tails+r*capacity,__ldg(counts+r),panel,32);
        const size_t at=static_cast<size_t>(r)*n+col;
        const int2 h=*reinterpret_cast<const int2*>(high+at),g=*reinterpret_cast<const int2*>(mixed+at);
        const float a=__ldg(&sa[r].x)*weight;
        float2* dst=reinterpret_cast<float2*>(output+at);
        const float2 previous=*dst;
        *dst=make_float2(fmaf(a*b0,fmaf(ratio,h.x,g.x),fmaf(b0*gain,v.x,previous.x)),
                         fmaf(a*b1,fmaf(ratio,h.y,g.y),fmaf(b1*gain,v.y,previous.y)));
    }
}

__device__ __forceinline__ float4 tail_dot_bytes4(const Residual* tails,int count,const unsigned int* dense) {
    float4 partial[4]={};
    int p=0;
    for (;p+4<=count;p+=4) {
        int2 indices[4];
        unsigned int values[4];
        #pragma unroll
        for (int i=0;i<4;++i) indices[i]=__ldg(reinterpret_cast<const int2*>(tails+p+i));
        #pragma unroll
        for (int i=0;i<4;++i) values[i]=__ldg(dense+indices[i].x*32);
        #pragma unroll
        for (int i=0;i<4;++i) {
            const unsigned int q=values[i];
            const float v=__int_as_float(indices[i].y);
            partial[i].x=fmaf(v,static_cast<signed char>(q),partial[i].x);
            partial[i].y=fmaf(v,static_cast<signed char>(q>>8),partial[i].y);
            partial[i].z=fmaf(v,static_cast<signed char>(q>>16),partial[i].z);
            partial[i].w=fmaf(v,static_cast<signed char>(q>>24),partial[i].w);
        }
    }
    for (;p<count;++p) {
        const int2 t=__ldg(reinterpret_cast<const int2*>(tails+p));
        const unsigned int q=__ldg(dense+t.x*32);
        const float v=__int_as_float(t.y);
        partial[0].x=fmaf(v,static_cast<signed char>(q),partial[0].x);
        partial[0].y=fmaf(v,static_cast<signed char>(q>>8),partial[0].y);
        partial[0].z=fmaf(v,static_cast<signed char>(q>>16),partial[0].z);
        partial[0].w=fmaf(v,static_cast<signed char>(q>>24),partial[0].w);
    }
    #pragma unroll
    for (int i=1;i<4;++i) {
        partial[0].x+=partial[i].x;
        partial[0].y+=partial[i].y;
        partial[0].z+=partial[i].z;
        partial[0].w+=partial[i].w;
    }
    return partial[0];
}

template<int Digits>
__device__ void pack_tail_digits(const Residual* residual,const unsigned short* counts,uint4* packets,float* scales,
                                 int capacity) {
    constexpr int limit=Digits==1?127:32512;
    const int row=blockIdx.x,lane=threadIdx.x,stride=(capacity+3)/4;
    const int count=__ldg(counts+row);
    const Residual* in=residual+row*capacity;
    float peak=0;
    for (int i=lane;i<count;i+=32) peak=fmaxf(peak,fabsf(in[i].value));
    for(int d=16;d;d>>=1) peak=fmaxf(peak,__shfl_xor_sync(0xffffffff,peak,d));
    const float step=peak>0?peak/limit:1.f;
    if (!lane) scales[row]=step;
    const float inverse=__frcp_rn(step);
    for (int i=lane;4*i<count;i+=32) {
        uint4 word=make_uint4(0,0,0,0);
        #pragma unroll
        for (int j=0;j<4;++j) {
            const int slot=4*i+j;
            const Residual r=slot<count?in[slot]:Residual{0,0};
            if (j<2) word.x|=static_cast<unsigned int>(r.k)<<(16*j);
            else word.y|=static_cast<unsigned int>(r.k)<<(16*(j-2));
            const int q=max(-limit,min(limit,__float2int_rn(r.value*inverse)));
            const int hi=Digits==1?q:__float2int_rn(q*(1.0f/255));
            const int lo=Digits==1?0:q-255*hi;
            word.z|=(hi&255u)<<(8*j);
            word.w|=(lo&255u)<<(8*j);
        }
        packets[row*stride+i]=word;
    }
}

#define TAIL_PACKET_PACK(name, digits) extern "C" __global__ void name( \
    const Residual* values,const unsigned short* counts,uint4* packets,float* scales,int capacity) { \
    pack_tail_digits<digits>(values,counts,packets,scales,capacity); }
TAIL_PACKET_PACK(pack_tail_one_byte,1)
TAIL_PACKET_PACK(pack_tail_two_bytes,2)

template<int Digits>
__device__ __forceinline__ float4 tail_dot_packets4(const uint4* packets,int count,const unsigned int* dense,float scale) {
    int4 hi={},lo={};
    for (int i=0;i<(count+3)/4;++i) {
        const uint4 p=__ldg(packets+i);
        const unsigned int u0=__ldg(dense+(p.x&65535u)*32),u1=__ldg(dense+(p.x>>16)*32);
        const unsigned int u2=__ldg(dense+(p.y&65535u)*32),u3=__ldg(dense+(p.y>>16)*32);
        const unsigned int l01=__byte_perm(u0,u1,0x5140),l23=__byte_perm(u2,u3,0x5140);
        const unsigned int h01=__byte_perm(u0,u1,0x7362),h23=__byte_perm(u2,u3,0x7362);
        const int x=__byte_perm(l01,l23,0x5410),y=__byte_perm(l01,l23,0x7632);
        const int z=__byte_perm(h01,h23,0x5410),w=__byte_perm(h01,h23,0x7632);
        const int high=p.z,low=p.w;
        hi.x=__dp4a(high,x,hi.x); hi.y=__dp4a(high,y,hi.y);
        hi.z=__dp4a(high,z,hi.z); hi.w=__dp4a(high,w,hi.w);
        if (Digits==2) {
            lo.x=__dp4a(low,x,lo.x); lo.y=__dp4a(low,y,lo.y);
            lo.z=__dp4a(low,z,lo.z); lo.w=__dp4a(low,w,lo.w);
        }
    }
    if (Digits==1) return make_float4(scale*hi.x,scale*hi.y,scale*hi.z,scale*hi.w);
    return make_float4(scale*fmaf(255.f,hi.x,lo.x),scale*fmaf(255.f,hi.y,lo.y),
                       scale*fmaf(255.f,hi.z,lo.z),scale*fmaf(255.f,hi.w,lo.w));
}

template<int Digits>
__device__ __forceinline__ float4 tail_panel_dot(const void* tails,const float* scales,int capacity,int row,
                                                int count,const unsigned int* panel) {
    if (!Digits) return tail_dot_bytes4(static_cast<const Residual*>(tails)+row*capacity,count,panel);
    return tail_dot_packets4<Digits>(static_cast<const uint4*>(tails)+row*((capacity+3)/4),count,panel,__ldg(scales+row));
}

template<int Digits, bool Stream, bool Compact = false>
__device__ void tail_wide_b(const float2* sa,const unsigned short* counts,const void* tails,
                             const float* residual_scales,const unsigned int* dense,
                             void* output,int n,int k,int capacity,float gain) {
    __shared__ float transposed[128][33];
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    const int row=128*blockIdx.y,col=32*blockIdx.x;
    const unsigned int* panel=dense+static_cast<size_t>(blockIdx.y)*k*32+lane;
    const float a0=Compact?1.0f:__ldg(&sa[row+4*lane].x)*gain;
    const float a1=Compact?1.0f:__ldg(&sa[row+4*lane+1].x)*gain;
    const float a2=Compact?1.0f:__ldg(&sa[row+4*lane+2].x)*gain;
    const float a3=Compact?1.0f:__ldg(&sa[row+4*lane+3].x)*gain;
    for (int j=warp;j<32;j+=8) {
        const int c=col+j;
        const float4 v=Compact?tail_dot_packets4<2>(static_cast<const uint4*>(tails)+c*((capacity+3)/4),
                                                  __ldg(counts+c),panel,1.f/(32512.f*256.f)):
                               tail_panel_dot<Digits>(tails,residual_scales,capacity,c,__ldg(counts+c),panel);
        transposed[lane][j]=a0*v.x;
        transposed[32+lane][j]=a1*v.y;
        transposed[64+lane][j]=a2*v.z;
        transposed[96+lane][j]=a3*v.w;
    }
    __syncthreads();
    for (int i=warp;i<128;i+=8) {
        const size_t at=static_cast<size_t>(row+i)*n+col+lane;
        const float value=transposed[(i%4)*32+i/4][lane];
        if (Compact) {
            static_cast<unsigned short*>(output)[at]=encode_half(value);
        } else {
            float* dst=static_cast<float*>(output)+at;
            if (Stream) asm volatile("st.global.cs.f32 [%0],%1;" :: "l"(dst),"f"(value) : "memory");
            else *dst=value;
        }
    }
}

template<int Digits, bool Stream, bool Hot = false, bool Compact = false, bool OneSided = false>
__device__ void tail_wide_a(const int* high,const int* mixed,const float2* sa,const float2* sb,
                             const unsigned short* counts,const void* tails,const float* residual_scales,
                             const unsigned int* dense,float* output,int m,int n,int k,int capacity,
                             float ratio,float weight,float gain,const int* hot_count=nullptr,
                             const float* hot_a=nullptr,const float* hot_b=nullptr,
                             const unsigned short* compact=nullptr,const float* compact_scales=nullptr,
                             float compact_gain=0.f) {
    const int index=blockIdx.x+gridDim.x*blockIdx.y;
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    const int tile_x=index/(m/32),row=32*(index%(m/32)),col=128*tile_x+4*lane;
    const unsigned int* panel=dense+static_cast<size_t>(tile_x)*k*32+lane;
    const float4 b=make_float4(__ldg(&sb[col].x),__ldg(&sb[col+1].x),__ldg(&sb[col+2].x),__ldg(&sb[col+3].x));
    const int hot=Hot?__ldg(hot_count):0;
    float4 rhs;
    if (Compact) rhs=__ldg(reinterpret_cast<const float4*>(compact_scales+col));
    for (int i=warp;i<32;i+=8) {
        const int r=row+i;
        const float4 v=tail_panel_dot<Digits>(tails,residual_scales,capacity,r,__ldg(counts+r),panel);
        const size_t at=static_cast<size_t>(r)*n+col;
        const float a=__ldg(&sa[r].x)*weight;
        float4* dst=reinterpret_cast<float4*>(output+at);
        int4 h,g;
        float4 p;
        if (Stream) {
            asm("ld.global.cs.v4.s32 {%0,%1,%2,%3},[%4];"
                : "=r"(h.x),"=r"(h.y),"=r"(h.z),"=r"(h.w) : "l"(high+at));
            asm("ld.global.cs.v4.s32 {%0,%1,%2,%3},[%4];"
                : "=r"(g.x),"=r"(g.y),"=r"(g.z),"=r"(g.w) : "l"(mixed+at));
            asm("ld.global.cs.v4.f32 {%0,%1,%2,%3},[%4];"
                : "=f"(p.x),"=f"(p.y),"=f"(p.z),"=f"(p.w) : "l"(dst));
        } else {
            h=*reinterpret_cast<const int4*>(high+at);
            g=*reinterpret_cast<const int4*>(mixed+at);
            if (OneSided) p=make_float4(0,0,0,0);
            else if (Compact) {
                const uint2 u=*reinterpret_cast<const uint2*>(compact+at);
                const float s=__ldg(&sa[r].x)*(compact_gain*32512.f*256.f);
                p=make_float4(decode_plane_half(static_cast<unsigned short>(u.x))*(rhs.x*s),
                              decode_plane_half(static_cast<unsigned short>(u.x>>16))*(rhs.y*s),
                              decode_plane_half(static_cast<unsigned short>(u.y))*(rhs.z*s),
                              decode_plane_half(static_cast<unsigned short>(u.y>>16))*(rhs.w*s));
            } else p=*dst;
        }
        float4 value=make_float4(fmaf(a*b.x,fmaf(ratio,h.x,g.x),fmaf(b.x*gain,v.x,p.x)),
                                 fmaf(a*b.y,fmaf(ratio,h.y,g.y),fmaf(b.y*gain,v.y,p.y)),
                                 fmaf(a*b.z,fmaf(ratio,h.z,g.z),fmaf(b.z*gain,v.z,p.z)),
                                 fmaf(a*b.w,fmaf(ratio,h.w,g.w),fmaf(b.w*gain,v.w,p.w)));
        if (Hot) {
            for (int j=0;j<hot;++j) {
                const float x=__ldg(hot_a+j*m+r);
                const float4 y=__ldg(reinterpret_cast<const float4*>(hot_b+j*n+col));
                value.x=fmaf(x,y.x,value.x);
                value.y=fmaf(x,y.y,value.y);
                value.z=fmaf(x,y.z,value.z);
                value.w=fmaf(x,y.w,value.w);
            }
        }
        if (Stream) {
            asm volatile("st.global.cs.v4.f32 [%0],{%1,%2,%3,%4};" ::
                         "l"(dst),"f"(value.x),"f"(value.y),"f"(value.z),"f"(value.w) : "memory");
        } else *dst=value;
    }
}

#define TAIL_WIDE(name, digits, stream) \
extern "C" __global__ void name##_b(const float2* sa,const unsigned short* counts,const void* tails, \
                                      const float* scales,const unsigned int* dense,float* output,int n,int k, \
                                      int capacity,float gain) { \
    tail_wide_b<digits,stream>(sa,counts,tails,scales,dense,output,n,k,capacity,gain); } \
extern "C" __global__ void name##_a(const int* high,const int* mixed,const float2* sa,const float2* sb, \
                                      const unsigned short* counts,const void* tails,const float* scales, \
                                      const unsigned int* dense,float* output,int m,int n,int k,int capacity, \
                                      float ratio,float weight,float gain) { \
    tail_wide_a<digits,stream>(high,mixed,sa,sb,counts,tails,scales,dense,output,m,n,k,capacity,ratio,weight,gain); }
TAIL_WIDE(tail_direction128,0,false)
TAIL_WIDE(tail_direction128_q8,1,false)
TAIL_WIDE(tail_direction128_q16,2,false)
TAIL_WIDE(tail_direction128_q16_stream,2,true)

// Normalizing by 256 times the largest residual bounds the FP16 intermediate
// by count / 2, even when the original operands are near the FP16 endpoints.
extern "C" __global__ void tail_direction128_compact_b(const float2* sa,const unsigned short* counts,
    const void* tails,const float* scales,const unsigned int* dense,unsigned short* output,int n,int k,
    int capacity,float gain) {
    tail_wide_b<2,false,true>(sa,counts,tails,scales,dense,output,n,k,capacity,gain);
}

extern "C" __global__ void tail_direction128_compact_a(const int* high,const int* mixed,const float2* sa,
    const float2* sb,const unsigned short* counts,const void* tails,const float* scales,const unsigned int* dense,
    float* output,int m,int n,int k,int capacity,float ratio,float weight,float gain,const unsigned short* compact,
    const float* compact_scales,float compact_gain) {
    tail_wide_a<2,false,false,true>(high,mixed,sa,sb,counts,tails,scales,dense,output,m,n,k,capacity,ratio,weight,gain,
                                 nullptr,nullptr,nullptr,compact,compact_scales,compact_gain);
}

extern "C" __global__ void tail_direction128_left_a(const int* high,const int* mixed,const float2* sa,
    const float2* sb,const unsigned short* counts,const void* tails,const float* scales,const unsigned int* dense,
    float* output,int m,int n,int k,int capacity,float ratio,float weight,float gain) {
    tail_wide_a<2,false,false,false,true>(high,mixed,sa,sb,counts,tails,scales,dense,output,m,n,k,capacity,ratio,weight,gain);
}

extern "C" __global__ void tail_direction128_left_q8_a(const int* high,const int* mixed,const float2* sa,
    const float2* sb,const unsigned short* counts,const void* tails,const float* scales,const unsigned int* dense,
    float* output,int m,int n,int k,int capacity,float ratio,float weight,float gain) {
    tail_wide_a<1,false,false,false,true>(high,mixed,sa,sb,counts,tails,scales,dense,output,m,n,k,capacity,ratio,weight,gain);
}

extern "C" __global__ void tail_direction128_hot_a(const int* high,const int* mixed,const float2* sa,const float2* sb,
                                                    const unsigned short* counts,const void* tails,const float* scales,
                                                    const unsigned int* dense,float* output,int m,int n,int k,int capacity,
                                                    float ratio,float weight,float gain,const int* count,
                                                    const float* hot_a,const float* hot_b) {
    tail_wide_a<2,false,true>(high,mixed,sa,sb,counts,tails,scales,dense,output,m,n,k,capacity,ratio,weight,gain,count,hot_a,hot_b);
}

extern "C" __global__ void merge_tail_bytes(const int* high,const int* mixed,const float2* sa,const float2* sb,
                                             float* output,int n,float weight_ratio,float weight_g) {
    const int col=(blockIdx.x*32+threadIdx.x)*4;
    const float4 b=make_float4(__ldg(&sb[col].x),__ldg(&sb[col+1].x),__ldg(&sb[col+2].x),__ldg(&sb[col+3].x));
    #pragma unroll
    for (int i=0;i<4;++i) {
        const int row=blockIdx.y*16+threadIdx.y+4*i;
        const size_t at=static_cast<size_t>(row)*n+col;
        const float scale=__ldg(&sa[row].x)*weight_g;
        const int4 h=*reinterpret_cast<const int4*>(high+at),g=*reinterpret_cast<const int4*>(mixed+at);
        float4* dst=reinterpret_cast<float4*>(output+at);
        const float4 c=*dst;
        *dst=make_float4(fmaf(scale*b.x,fmaf(weight_ratio,h.x,g.x),c.x),
                         fmaf(scale*b.y,fmaf(weight_ratio,h.y,g.y),c.y),
                         fmaf(scale*b.z,fmaf(weight_ratio,h.z,g.z),c.z),
                         fmaf(scale*b.w,fmaf(weight_ratio,h.w,g.w),c.w));
    }
}
