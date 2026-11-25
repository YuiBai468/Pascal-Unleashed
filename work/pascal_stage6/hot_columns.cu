constexpr int hot_limit=16;

extern "C" __global__ void sample_column_energy(const unsigned short* a,const unsigned short* b,float2* energies,
                                                 int m,int n,int k) {
    const int col=blockIdx.x*blockDim.x+threadIdx.x;
    if (col>=k) return;
    float sa=0,sb=0;
    for (int sample=0;sample<64;++sample) {
        const float x=decode_plane_half(a[static_cast<size_t>(sample*m/64)*k+col]);
        const float y=decode_plane_half(b[static_cast<size_t>(sample*n/64)*k+col]);
        sa=fmaf(x,x,sa);
        sb=fmaf(y,y,sb);
    }
    energies[col]=make_float2(sa,sb);
}

extern "C" __global__ void select_hot_columns(const float2* energies,int* count,int* indices,int k) {
    __shared__ float2 totals[8],threshold;
    __shared__ int found;
    const int tid=threadIdx.x,lane=tid%32,warp=tid/32;
    float x=0,y=0;
    for (int col=tid;col<k;col+=256) {
        const float2 e=energies[col];
        x+=e.x;
        y+=e.y;
    }
    for (int d=16;d;d>>=1) {
        x+=__shfl_down_sync(0xffffffff,x,d);
        y+=__shfl_down_sync(0xffffffff,y,d);
    }
    if (!lane) totals[warp]=make_float2(x,y);
    __syncthreads();
    if (!tid) {
        x=y=0;
        for (int i=0;i<8;++i) {
            x+=totals[i].x;
            y+=totals[i].y;
        }
        threshold=make_float2(x*(16.f/k),y*(16.f/k));
        found=0;
    }
    __syncthreads();
    for (int col=tid;col<k;col+=256) {
        const float2 e=energies[col];
        if (e.x>threshold.x || e.y>threshold.y) {
            const int pos=atomicAdd(&found,1);
            if (pos<hot_limit) indices[pos]=col;
        }
    }
    __syncthreads();
    if (!tid) {
        const int size=found<=hot_limit?found:0;
        for (int i=1;i<size;++i) {
            const int col=indices[i];
            int j=i;
            while (j>0 && indices[j-1]>col) {
                indices[j]=indices[j-1];
                --j;
            }
            indices[j]=col;
        }
        *count=size;
    }
}

extern "C" __global__ void hot_column_mask(const int* count,const int* indices,unsigned int* keep,int k) {
    const int pair=blockIdx.x*blockDim.x+threadIdx.x;
    if (pair>=k/2) return;
    const int size=__ldg(count);
    unsigned int bits=0xffffffffu;
    for (int i=0;i<size;++i) {
        const int col=__ldg(indices+i);
        if (col/2==pair) bits&=col&1?0x0000ffffu:0xffff0000u;
    }
    keep[pair]=bits;
}

extern "C" __global__ void gather_hot_columns(const unsigned short* source,const int* count,const int* indices,
                                               float* dense,int rows,int k) {
    const int row=blockIdx.x*blockDim.x+threadIdx.x,position=blockIdx.y;
    if (row>=rows || position>=__ldg(count)) return;
    const int column=__ldg(indices+position);
    dense[static_cast<size_t>(position)*rows+row]=decode_plane_half(source[static_cast<size_t>(row)*k+column]);
}
