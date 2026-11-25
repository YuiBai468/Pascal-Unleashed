__device__ float pu_read(const void* p, unsigned long long at, int type) {
    return type==2 ? decode_plane_half(static_cast<const unsigned short*>(p)[at])
                   : static_cast<const float*>(p)[at];
}

__device__ void pu_write(void* p, unsigned long long at, int type, float x) {
    if (type==2) static_cast<unsigned short*>(p)[at]=encode_half(x);
    else static_cast<float*>(p)[at]=x;
}

extern "C" __global__ void pu_gather(const void* in,void* out,int rows,int cols,int dst_rows,int dst_cols,
                                      unsigned long long rs,unsigned long long cs,int in_type,int out_type,unsigned* bad) {
    for (size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<(size_t)dst_rows*dst_cols;i+=gridDim.x*blockDim.x) {
        int row=i/dst_cols,col=i%dst_cols;
        float x=(row<rows && col<cols)?pu_read(in,row*rs+col*cs,in_type):0.f;
        if (bad && (!isfinite(x) || fabsf(x)>65504.f)) atomicOr(bad,1u);
        pu_write(out,i,out_type,x);
    }
}

extern "C" __global__ void pu_scatter(const void* in,void* out,int rows,int cols,int pitch,
                                       unsigned long long rs,unsigned long long cs,int in_type,int out_type,float alpha,float beta) {
    for (size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<(size_t)rows*cols;i+=gridDim.x*blockDim.x) {
        size_t row=i/cols,col=i%cols,dst=row*rs+col*cs;
        float x=in ? alpha*pu_read(in,row*pitch+col,in_type) : 0.f;
        if (beta!=0.f) x=fmaf(beta,pu_read(out,dst,out_type),x);
        pu_write(out,dst,out_type,x);
    }
}

extern "C" __global__ void pu_transpose(const signed char* in,signed char* out,int rows,int cols) {
    __shared__ signed char tile[32][36];
    int x=threadIdx.x%32,y=threadIdx.x/32;
    int row=blockIdx.y*32,col=blockIdx.x*32;
    for (int j=y;j<32;j+=8) if (row+j<rows && col+x<cols)
        tile[j][x]=in[(size_t)(row+j)*cols+col+x];
    __syncthreads();
    for (int j=y;j<32;j+=8) if (col+j<cols && row+x<rows)
        out[(size_t)(col+j)*rows+row+x]=tile[x][j];
}

extern "C" __global__ void pu_check_half(const unsigned short* in,size_t count,unsigned* bad) {
    for (size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x)
        if ((in[i]&0x7fff)>=0x7c00u) atomicOr(bad,1u);
}

extern "C" __global__ void pu_risk(const unsigned short* in,int k,int pitch,unsigned* marks,unsigned side,unsigned* bad) {
    __shared__ float warp_sums[8],cut;
    const unsigned short* row=in+(size_t)blockIdx.x*pitch;
    int t=threadIdx.x;
    float sum=0;
    for (int i=t;i<k;i+=256) {
        float x=decode_plane_half(row[i]);
        if (!isfinite(x)) { atomicOr(bad,1u); x=0.f; }
        sum=fmaf(x,x,sum);
    }
    for (int d=16;d;d>>=1) sum+=__shfl_down_sync(0xffffffff,sum,d);
    if (!(t%32)) warp_sums[t/32]=sum;
    __syncthreads();
    if (!t) {
        sum=0.f;
        for (int i=0;i<8;++i) sum+=warp_sums[i];
        cut=8.f*sqrtf(sum/k);
    }
    __syncthreads();
    for (int i=t;i<k;i+=256) if (fabsf(decode_plane_half(row[i]))>cut) {
        unsigned previous=atomicOr(marks+i,side);
        if (previous & (3u^side)) atomicOr(bad,8u);
    }
}

__device__ unsigned pu_hash(unsigned x) {
    x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; return x^(x>>16);
}

extern "C" __global__ void pu_generate(unsigned short* out,int rows,int k,unsigned seed,int dataset,int side) {
    for (size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<(size_t)rows*k;i+=gridDim.x*blockDim.x) {
        int col=i%k, key=i;
        if (dataset==2 && col>=k/2 && k%2==0) key-=k/2;
        float u=(1.f+(pu_hash(key^seed)>>8))*(1.f/16777216.f);
        float v=(pu_hash(key^seed^0xa4309e41u)>>8)*(1.f/16777216.f);
        float x=sqrtf(-2.f*logf(u))*cosf(6.283185307f*v);
        if (dataset==1 && (col==3 || col==k/3 || col==2*k/3)) x*=32.f;
        if (dataset==2 && col>=k/2 && side && k%2==0) x=-x;
        if (dataset==3) x=0.f;
        out[i]=encode_half(x);
    }
}

extern "C" __global__ void pu_compare(const float* ref,const void* output,int type,size_t elements,double* stats) {
    __shared__ double sums[3][8];
    double err=0,energy=0,highest=0;
    for (size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<elements;i+=gridDim.x*blockDim.x) {
        double value=ref[i], diff=static_cast<double>(pu_read(output,i,type))-value;
        err+=diff*diff; energy+=value*value; highest=fmax(highest,fabs(diff));
    }
    for (int d=16;d;d>>=1) {
        err+=__shfl_down_sync(0xffffffff,err,d);
        energy+=__shfl_down_sync(0xffffffff,energy,d);
        highest=fmax(highest,__shfl_down_sync(0xffffffff,highest,d));
    }
    int t=threadIdx.x;
    if (!(t%32)) { sums[0][t/32]=err; sums[1][t/32]=energy; sums[2][t/32]=highest; }
    __syncthreads();
    if (!t) {
        for (int i=1;i<8;++i) {
            sums[0][0]+=sums[0][i]; sums[1][0]+=sums[1][i]; sums[2][0]=fmax(sums[2][0],sums[2][i]);
        }
        stats[3*blockIdx.x]=sums[0][0]; stats[3*blockIdx.x+1]=sums[1][0]; stats[3*blockIdx.x+2]=sums[2][0];
    }
}
