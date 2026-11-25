// Find exceptional K groups at runtime, then add their exact FP32 contributions.
extern "C" __global__ void clear_pair_peaks(unsigned* a, unsigned* b, int pairs) {
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < pairs) { a[i]=0; b[i]=0; }
}

extern "C" __global__ void reduce_pair_peaks(const unsigned* input, unsigned* output, int rows, int pairs) {
    __shared__ unsigned peaks[4][64];
    const int lane = threadIdx.x & 63;
    const int group = threadIdx.x >> 6;
    const int pair = blockIdx.x*64+lane;
    const int start = blockIdx.y*256;
    unsigned peak=0;
    for (int r = start+group; r < min(rows,start+256); r += 4) {
        const unsigned v = __ldg(input + r*pairs + pair);
        peak = max(peak,max(v&0x7fff,(v>>16)&0x7fff));
    }
    peaks[group][lane] = peak;
    __syncthreads();
    if (group==0) {
        peak = max(max(peaks[0][lane],peaks[1][lane]),max(peaks[2][lane],peaks[3][lane]));
        atomicMax(output+pair,peak);
    }
}

extern "C" __global__ void identify_exceptional_pairs(const unsigned* a, const unsigned* b,
                                                      unsigned char* mask, int* indices,
                                                      int* info, int pairs, int capacity,
                                                      int extra_exponent) {
    __shared__ int hist_a[32], hist_b[32];
    __shared__ int count, bin_a, bin_b;
    const int lane=threadIdx.x;
    if (lane<32) { hist_a[lane]=0; hist_b[lane]=0; }
    if (lane==0) count=0;
    __syncthreads();
    for (int p=lane; p<pairs; p+=256) {
        atomicAdd(hist_a+(a[p]>>10),1);
        atomicAdd(hist_b+(b[p]>>10),1);
    }
    __syncthreads();
    if (lane==0) {
        int ma=0, mb=0, ca=0, cb=0;
        while (ma<31 && (ca+=hist_a[ma]) < (pairs+1)/2) ++ma;
        while (mb<31 && (cb+=hist_b[mb]) < (pairs+1)/2) ++mb;
        bin_a=ma+extra_exponent;
        bin_b=mb+extra_exponent;
    }
    __syncthreads();
    for (int p=lane; p<pairs; p+=256) {
        const bool selected = (a[p]>>10)>=bin_a || (b[p]>>10)>=bin_b;
        mask[p]=selected;
        if (selected) {
            const int i=atomicAdd(&count,1);
            if (i<capacity) indices[i]=p;
        }
    }
    __syncthreads();
    if (count>capacity) for (int p=lane; p<pairs; p+=256) mask[p]=0;
    if (lane==0) {
        if (count<=capacity) {
            for (int i=1;i<count;++i) {
                const int group=indices[i];
                int j=i;
                while (j && indices[j-1]>group) { indices[j]=indices[j-1]; --j; }
                indices[j]=group;
            }
        }
        info[0]=count; info[1]=bin_a; info[2]=bin_b;
    }
}

extern "C" __global__ void gather_pairs(const unsigned* a, const unsigned* b, float2* out_a, float2* out_b,
                                         const int* indices, const int* info, int m, int n, int pairs, int capacity) {
    const int row=blockIdx.x*128+threadIdx.x;
    const int i=blockIdx.y;
    const int count=info[0];
    if (count>capacity || i>=count) return;
    const bool right=blockIdx.z!=0;
    const int rows=right ? n : m;
    if (row>=rows) return;
    const unsigned* operand=right ? b : a;
    float2* target=right ? out_b : out_a;
    const unsigned v=__ldg(operand+row*pairs+indices[i]);
    target[i*rows+row]=make_float2(decode_plane_half(static_cast<unsigned short>(v)),
                                  decode_plane_half(static_cast<unsigned short>(v>>16)));
}

extern "C" __global__ void unpack_robust(const int* dot, const float* a, const float* b,
                                          const float2* extra_a, const float2* extra_b, const int* info,
                                          float* output, int m, int n, int capacity) {
    const int j=4*(blockIdx.x*32+threadIdx.x);
    if (j>=n) return;
    const float4 right=*reinterpret_cast<const float4*>(b+j);
    const int selected=info[0] <= capacity ? info[0] : 0;
    float4 values[4];
    #pragma unroll
    for (int s=0;s<4;++s) {
        const int i=blockIdx.y*16+threadIdx.y+4*s;
        const int index=i*n+j;
        const float scale=a[i];
        const int4 z=*reinterpret_cast<const int4*>(dot+index);
        values[s]=make_float4(z.x*(scale*right.x),z.y*(scale*right.y),z.z*(scale*right.z),z.w*(scale*right.w));
    }
    for (int t=0;t<selected;++t) {
        const float2* b0=extra_b+t*n+j;
        const float2 b1=__ldg(b0), b2=__ldg(b0+1), b3=__ldg(b0+2), b4=__ldg(b0+3);
        #pragma unroll
        for (int s=0;s<4;++s) {
            const int i=blockIdx.y*16+threadIdx.y+4*s;
            const float2 u=__ldg(extra_a+t*m+i);
            float4 v=values[s];
            v.x=fmaf(u.y,b1.y,fmaf(u.x,b1.x,v.x));
            v.y=fmaf(u.y,b2.y,fmaf(u.x,b2.x,v.y));
            v.z=fmaf(u.y,b3.y,fmaf(u.x,b3.x,v.z));
            v.w=fmaf(u.y,b4.y,fmaf(u.x,b4.x,v.w));
            values[s]=v;
        }
    }
    #pragma unroll
    for (int s=0;s<4;++s) {
        const int i=blockIdx.y*16+threadIdx.y+4*s;
        *reinterpret_cast<float4*>(output+i*n+j)=values[s];
    }
}
