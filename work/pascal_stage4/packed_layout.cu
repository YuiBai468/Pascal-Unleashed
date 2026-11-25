// Transpose a tile of INT8 in registers after coalesced word transfers.
extern "C" __global__ void transpose_packed(const unsigned int* in, unsigned int* out,
                                             int rows, int columns) {
    __shared__ unsigned int tile[32][9];
    const int tid=threadIdx.x;
    const int ir=blockIdx.y*32+tid/8, iw=blockIdx.x*8+tid%8;
    const int input_words=columns/4;
    tile[tid/8][tid%8]=(ir<rows&&iw<input_words)?in[ir*input_words+iw]:0u;
    __syncthreads();
    const int lane_k=tid/8, group=tid%8;
    const int j=blockIdx.x*32+lane_k, w=blockIdx.y*8+group;
    if(j<columns&&w<rows/4){
        const int shift=8*(lane_k%4),source_word=lane_k/4;
        unsigned int value=0;
        #pragma unroll
        for(int i=0;i<4;++i) value|=((tile[4*group+i][source_word]>>shift)&255u)<<(8*i);
        out[j*(rows/4)+w]=value;
    }
}

extern "C" __global__ void transpose_packed64(const unsigned int* in, unsigned int* out,
                                               int rows, int columns) {
    __shared__ unsigned int tile[4][16][17];
    const int group = threadIdx.x >> 4;
    const int word = threadIdx.x & 15;
    const int row = 64 * blockIdx.y + 4 * group;
    const int col = 16 * blockIdx.x + word;
    const int stride = columns / 4;
    const unsigned* source = in + row * stride + col;
    const unsigned a = source[0], b = source[stride];
    const unsigned c = source[2 * stride], d = source[3 * stride];
    const unsigned lo0 = __byte_perm(a, b, 0x5140), lo1 = __byte_perm(c, d, 0x5140);
    const unsigned hi0 = __byte_perm(a, b, 0x7362), hi1 = __byte_perm(c, d, 0x7362);
    tile[0][group][word] = __byte_perm(lo0, lo1, 0x5410);
    tile[1][group][word] = __byte_perm(lo0, lo1, 0x7632);
    tile[2][group][word] = __byte_perm(hi0, hi1, 0x5410);
    tile[3][group][word] = __byte_perm(hi0, hi1, 0x7632);
    __syncthreads();
    const int target_stride = rows / 4;
    unsigned* target = out + (64 * blockIdx.x + 4 * group) * target_stride + 16 * blockIdx.y + word;
    #pragma unroll
    for (int j = 0; j < 4; ++j) target[j * target_stride] = tile[j][word][group];
}

extern "C" __global__ void unpack_affine_wide(const int* dot,const float* a,const float* b,
                                               const float* za,const float* zb,const float* qa,const float* qb,
                                               float* output,int m,int n,int low_k) {
    const int base_j=4*(blockIdx.x*32+threadIdx.x);
    if(base_j>=n)return;
    const float4 bs=*reinterpret_cast<const float4*>(b+base_j);
    const float4 bz=*reinterpret_cast<const float4*>(zb+base_j);
    const float4 bq=*reinterpret_cast<const float4*>(qb+base_j);
    const float bs4[]={bs.x,bs.y,bs.z,bs.w}, bz4[]={bz.x,bz.y,bz.z,bz.w};
    const float bq4[]={bq.x,bq.y,bq.z,bq.w};
    float prior[4];
    #pragma unroll
    for(int i=0;i<4;++i) prior[i]=fmaf(static_cast<float>(low_k),bz4[i],bq4[i]);
    #pragma unroll
    for(int j=0;j<4;++j){
        const int row=blockIdx.y*16+threadIdx.y+4*j;
        if(row<m){
            const int offset=row*n+base_j;
            const int4 z=*reinterpret_cast<const int4*>(dot+offset);
            const int v[]={z.x,z.y,z.z,z.w};
            const float as=a[row], az=za[row], aq=qa[row];
            float o[4];
            #pragma unroll
            for(int i=0;i<4;++i){
                const float fix=fmaf(az,prior[i],bz4[i]*aq);
                o[i]=fmaf(static_cast<float>(v[i]),as*bs4[i],fix);
            }
            *reinterpret_cast<float4*>(output+offset)=make_float4(o[0],o[1],o[2],o[3]);
        }
    }
}
