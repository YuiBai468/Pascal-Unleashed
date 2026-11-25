template<int Rows,int Cols,bool Binary>
__device__ void product(const uint2* a,const uint2* b,int* out,int n,int words,int multiplier){
    constexpr int M=16*Rows,N=16*Cols,PK=16;
    __shared__ unsigned int ap[PK][M+2],an[PK][M+2],bp[PK][N+2],bn[PK][N+2];
    const int tx=threadIdx.x,ty=threadIdx.y,t=16*ty+tx;
    const int r0=blockIdx.y*M,c0=blockIdx.x*N;
    int total[Rows][Cols]={};
    for(int start=0;start<words;start+=PK){
        #pragma unroll
        for(int x=t;x<PK*M;x+=256){
            int r=x/PK,p=x%PK;
            uint2 v=(r0+r<n && start+p<words)?a[(r0+r)*words+start+p]:uint2{};
            ap[p][r]=v.x;if(!Binary)an[p][r]=v.y;
        }
        #pragma unroll
        for(int x=t;x<PK*N;x+=256){
            int c=x/PK,p=x%PK;
            uint2 v=(c0+c<n && start+p<words)?b[(c0+c)*words+start+p]:uint2{};
            bp[p][c]=v.x;if(!Binary)bn[p][c]=v.y;
        }
        __syncthreads();
        #pragma unroll 4
        for(int p=0;p<PK;++p){
            unsigned int x[Rows],y[Rows];
            #pragma unroll
            for(int i=0;i<Rows;++i){x[i]=ap[p][ty+16*i];if(!Binary)y[i]=an[p][ty+16*i];}
            #pragma unroll
            for(int j=0;j<Cols;++j){
                unsigned int u=bp[p][tx+16*j],v=Binary?0:bn[p][tx+16*j];
                #pragma unroll
                for(int i=0;i<Rows;++i){
                    if(Binary)total[i][j]+=__popc(x[i]^u);
                    else total[i][j]+=__popc((x[i]&u)|(y[i]&v))-__popc((x[i]&v)|(y[i]&u));
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int i=0;i<Rows;++i){
        int row=r0+ty+16*i;
        #pragma unroll
        for(int j=0;j<Cols;++j){
            int col=c0+tx+16*j;
            if(row<n&&col<n)out[row*n+col]=multiplier*(Binary?32*words-2*total[i][j]:total[i][j]);
        }
    }
}
#define LOW(name,r,c,bi) extern "C" __global__ __launch_bounds__(256,3) void name(const uint2* a,const uint2* b,int* out,int n,int w,int mul){product<r,c,bi>(a,b,out,n,w,mul);}
LOW(ternary32,2,2,false)
LOW(ternary64,4,4,false)
LOW(binary64,4,4,true)

extern "C" __global__ void masks_from_int8(const signed char* a,uint2* masks,int k,int rows){
    const int warps=blockDim.x/32;
    const int word=blockIdx.x*warps+threadIdx.x/32,row=blockIdx.y,lane=threadIdx.x%32;
    if(32*word>=k || row>=rows)return;
    const int v=a[row*k+32*word+lane];
    const uint2 x=make_uint2(__ballot_sync(0xffffffff,v>0),__ballot_sync(0xffffffff,v<0));
    if(lane==0)masks[row*(k/32)+word]=x;
}
