struct FourResiduals {
    int leading[4];
    unsigned int digit;
};

extern "C" __global__ void packetize_tail(const Residual* src,const unsigned short* count,
                                         FourResiduals* out,float* scale,int stride,int limit) {
    const int row=blockIdx.x,tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    const int n=__ldg(count+row);
    const Residual* in=src+row*stride;
    __shared__ float maxima[4],inverse;
    float peak=0;
    for (int d=tid;d<n;d+=128) peak=fmaxf(peak,fabsf(in[d].value));
    for (int i=16;i;i>>=1) peak=fmaxf(peak,__shfl_down_sync(0xffffffff,peak,i));
    if (!lane) maxima[warp]=peak;
    __syncthreads();
    if (!tid) {
        const float bound=fmaxf(fmaxf(maxima[0],maxima[1]),fmaxf(maxima[2],maxima[3]));
        const float s=bound>0?bound*(1.0f/127):1;
        scale[row]=s;
        inverse=__frcp_rn(s);
    }
    __syncthreads();
    FourResiduals* packed=out+row*limit;
    for (int p=tid;4*p<n;p+=128) {
        FourResiduals value;
        unsigned int bits=0;
        #pragma unroll
        for (int i=0;i<4;++i) {
            const int slot=4*p+i;
            const Residual r=slot<n?in[slot]:Residual{0,0};
            value.leading[i]=r.k;
            bits|=(__float2int_rn(r.value*inverse)&255u)<<(8*i);
        }
        value.digit=bits;
        packed[p]=value;
    }
}

template<int Order>
__device__ void restore_packets(const int* h,const int* g,const float2* scale_a,const float2* scale_b,
                         const unsigned short* count_a,const unsigned short* count_b,
                         const FourResiduals* ra,const FourResiduals* rb,const float* rs_a,const float* rs_b,
                         const unsigned short* full_b,const unsigned short* high_a,float* out,
                         int m,int n,int base,int stride,int panel) {
    __shared__ float tile[64][66];
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    int bx=blockIdx.x,by=blockIdx.y;
    if (Order==1) {
        const int linear=blockIdx.x+gridDim.x*blockIdx.y;
        bx=linear/((m/64)*panel)*panel+linear%panel;
        by=(linear/panel)%(m/64);
    } else if (Order==2) {
        const int linear=blockIdx.x+gridDim.x*blockIdx.y;
        by=linear/((n/64)*panel)*panel+linear%panel;
        bx=(linear/panel)%(n/64);
    }
    const int ix=64*by,jx=64*bx;
    const int j=jx+2*lane;
    const float2 b0=__ldg(scale_b+j),b1=__ldg(scale_b+j+1);
    #pragma unroll
    for (int y=warp;y<64;y+=8) {
        const int row=ix+y;
        const int groups=(__ldg(count_a+row)+3)/4;
        const FourResiduals* samples=ra+row*stride;
        int q0=0,q1=0;
        for (int p=0;p<groups;++p) {
            const int k0=__ldg(&samples[p].leading[0]);
            const int k1=__ldg(&samples[p].leading[1]);
            const int k2=__ldg(&samples[p].leading[2]);
            const int k3=__ldg(&samples[p].leading[3]);
            const unsigned int code=__ldg(&samples[p].digit);
            const unsigned int v0=__ldg(full_b+static_cast<size_t>(k0)*(n/2)+j/2);
            const unsigned int v1=__ldg(full_b+static_cast<size_t>(k1)*(n/2)+j/2);
            const unsigned int v2=__ldg(full_b+static_cast<size_t>(k2)*(n/2)+j/2);
            const unsigned int v3=__ldg(full_b+static_cast<size_t>(k3)*(n/2)+j/2);
            const unsigned int u=__byte_perm(v0,v1,0x5140),v=__byte_perm(v2,v3,0x5140);
            q0=__dp4a(static_cast<int>(code),static_cast<int>(__byte_perm(u,v,0x5410)),q0);
            q1=__dp4a(static_cast<int>(code),static_cast<int>(__byte_perm(u,v,0x7632)),q1);
        }
        const float a=__ldg(&scale_a[row].x)*base;
        const float residual=__ldg(rs_a+row);
        const size_t at=static_cast<size_t>(row)*n+j;
        const int2 hh=*reinterpret_cast<const int2*>(h+at),gg=*reinterpret_cast<const int2*>(g+at);
        tile[y][2*lane]=fmaf(residual*b0.y,q0,(a*b0.x)*fmaf(static_cast<float>(base-1),hh.x,gg.x));
        tile[y][2*lane+1]=fmaf(residual*b1.y,q1,(a*b1.x)*fmaf(static_cast<float>(base-1),hh.y,gg.y));
    }
    __syncthreads();
    const int i=ix+2*lane;
    const float a0=__ldg(&scale_a[i].x)*base,a1=__ldg(&scale_a[i+1].x)*base;
    #pragma unroll
    for (int x=warp;x<64;x+=8) {
        const int col=jx+x;
        const int groups=(__ldg(count_b+col)+3)/4;
        const FourResiduals* samples=rb+col*stride;
        int q0=0,q1=0;
        for (int p=0;p<groups;++p) {
            const int k0=__ldg(&samples[p].leading[0]);
            const int k1=__ldg(&samples[p].leading[1]);
            const int k2=__ldg(&samples[p].leading[2]);
            const int k3=__ldg(&samples[p].leading[3]);
            const unsigned int code=__ldg(&samples[p].digit);
            const unsigned int v0=__ldg(high_a+static_cast<size_t>(k0)*(m/2)+i/2);
            const unsigned int v1=__ldg(high_a+static_cast<size_t>(k1)*(m/2)+i/2);
            const unsigned int v2=__ldg(high_a+static_cast<size_t>(k2)*(m/2)+i/2);
            const unsigned int v3=__ldg(high_a+static_cast<size_t>(k3)*(m/2)+i/2);
            const unsigned int u=__byte_perm(v0,v1,0x5140),v=__byte_perm(v2,v3,0x5140);
            q0=__dp4a(static_cast<int>(code),static_cast<int>(__byte_perm(u,v,0x5410)),q0);
            q1=__dp4a(static_cast<int>(code),static_cast<int>(__byte_perm(u,v,0x7632)),q1);
        }
        const float residual=__ldg(rs_b+col);
        tile[2*lane][x]=fmaf(residual*a0,q0,tile[2*lane][x]);
        tile[2*lane+1][x]=fmaf(residual*a1,q1,tile[2*lane+1][x]);
    }
    __syncthreads();
    #pragma unroll
    for (int y=warp;y<64;y+=8) {
        const size_t at=static_cast<size_t>(ix+y)*n+j;
        *reinterpret_cast<float2*>(out+at)=make_float2(tile[y][2*lane],tile[y][2*lane+1]);
    }
}

#define PACKET_FINISH(name,order) extern "C" __global__ void name( \
    const int*h,const int*g,const float2*sa,const float2*sb,const unsigned short*ca,const unsigned short*cb, \
    const FourResiduals*ra,const FourResiduals*rb,const float*as,const float*bs,const unsigned short*fb, \
    const unsigned short*ha,float*out,int m,int n,int base,int stride,int panel) { \
    restore_packets<order>(h,g,sa,sb,ca,cb,ra,rb,as,bs,fb,ha,out,m,n,base,stride,panel); }
PACKET_FINISH(packet_64,0)
PACKET_FINISH(packet_cols,1)
PACKET_FINISH(packet_rows,2)
