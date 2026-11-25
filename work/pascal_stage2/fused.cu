template<int Rows,int Cols,int PK>
__device__ void full_group(const int2* a,const int2* b,const float* sa,const float* sb,float* out,
                           int m,int n,int packed) {
    constexpr int M=16*Rows,N=16*Cols;
    constexpr int pad=32/PK;
    __shared__ int ah[PK][M+pad],ag[PK][M+pad],bh[PK][N+pad],bg[PK][N+pad];
    int p[Rows][Cols]={},q[Rows][Cols]={};
    const int row0=blockIdx.y*M,col0=blockIdx.x*N;
    const int ty=threadIdx.y,tx=threadIdx.x,tid=ty*16+tx;
    for(int first=0;first<packed;first+=PK) {
        #pragma unroll
        for(int i=tid;i<M*PK;i+=256) {
            const int r=i/PK,k=i%PK;
            const int2 v=row0+r<m&&first+k<packed?a[(row0+r)*packed+first+k]:int2{};
            ah[k][r]=v.x; ag[k][r]=v.y;
        }
        #pragma unroll
        for(int i=tid;i<N*PK;i+=256) {
            const int c=i/PK,k=i%PK;
            const int2 v=col0+c<n&&first+k<packed?b[(col0+c)*packed+first+k]:int2{};
            bh[k][c]=v.x; bg[k][c]=v.y;
        }
        __syncthreads();
        #pragma unroll 4
        for(int k=0;k<PK;++k) {
            int xa[Rows],xb[Rows];
            #pragma unroll
            for(int i=0;i<Rows;++i){xa[i]=ah[k][ty+i*16];xb[i]=ag[k][ty+i*16];}
            #pragma unroll
            for(int j=0;j<Cols;++j) {
                int ya=bh[k][tx+j*16],yb=bg[k][tx+j*16];
                #pragma unroll
                for(int i=0;i<Rows;++i){p[i][j]=__dp4a(xa[i],ya,p[i][j]);q[i][j]=__dp4a(xb[i],yb,q[i][j]);}
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int i=0;i<Rows;++i) {
        int row=row0+ty+16*i;
        if(row>=m)continue;
        #pragma unroll
        for(int j=0;j<Cols;++j) {
            int col=col0+tx+16*j;
            if(col<n)out[row*n+col]=static_cast<float>(15*p[i][j]+q[i][j])*((16.0f*sa[row])*sb[col]);
        }
    }
}

#define NEW(name, rows, cols, pk, occupancy) \
extern "C" __global__ __launch_bounds__(256,occupancy) void name(const int2* a,const int2* b, \
const float* sa,const float* sb,float* out,int m,int n,int packed) {full_group<rows,cols,pk>(a,b,sa,sb,out,m,n,packed);}
NEW(full32x64k32,2,4,8,4)
NEW(full64x64k32,4,4,8,3)
NEW(full64x64k64,4,4,16,3)
NEW(full64x128k32,4,8,8,2)
