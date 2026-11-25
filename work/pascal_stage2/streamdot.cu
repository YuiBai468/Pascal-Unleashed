template<int Channels,int Kind> __device__ void dots(const int* a,const int* b,int* out,int words,int k) {
    const int t=blockIdx.x*blockDim.x+threadIdx.x, threads=gridDim.x*blockDim.x;
    int result[Channels]={},z=0;
    for(int i=0;i<words;++i) {
        if(Kind==0) {
            const int v=b[i*threads+t];
            #pragma unroll
            for(int j=0;j<Channels;++j)result[j]=__dp4a(a[(i*Channels+j)*threads+t],v,result[j]);
        } else if(Kind==1 || Kind==2) {
            const int u=a[i*threads+t],v=b[i*threads+t];
            constexpr int chunk=Kind==1?7:30, shift=Kind==1?6:7;
            if(Kind==1)z=__dp4a(u,v,z);else z=__dp2a_lo(u,v,z);
            if((i+1)%chunk==0 || i+1==words) {
                #pragma unroll
                for(int j=0;j<Channels-1;++j) {
                    const int digit=((z+(1<<(shift-1)))&((1<<shift)-1))-(1<<(shift-1));
                    result[j]+=digit; z=(z-digit)>>shift;
                }
                result[Channels-1]+=z;z=0;
            }
        } else {
            const unsigned int v=b[(Kind==3?2*i:i)*threads+t];
            const unsigned int w=Kind==3?b[(2*i+1)*threads+t]:0;
            #pragma unroll
            for(int j=0;j<Channels;++j) {
                if(Kind==3) {
                    const unsigned int u=a[(2*(i*Channels+j))*threads+t],x=a[(2*(i*Channels+j)+1)*threads+t];
                    result[j]+=__popc((u&v)|(x&w))-__popc((u&w)|(x&v));
                } else {
                    const unsigned int u=a[(i*Channels+j)*threads+t];
                    result[j]+=32-2*__popc(u^v);
                }
            }
        }
    }
    #pragma unroll
    for(int j=0;j<Channels;++j)out[t*Channels+j]=result[j]-(Kind==4?32*words-k:0);
}
#define STREAM(name,c,m) extern "C" __global__ void name(const int* a,const int* b,int* o,int w,int k){dots<c,m>(a,b,o,w,k);}
STREAM(plain2,2,0)
STREAM(plain3,3,0)
STREAM(packed2,2,1)
STREAM(packed3,3,2)
STREAM(ternary_pop2,2,3)
STREAM(binary_pop2,2,4)
