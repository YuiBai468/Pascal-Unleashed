extern "C" __global__ void strassen_form(const signed char* in,signed char* out,int n,int right,int* invalid) {
    const int h=n/2, r=blockIdx.y*blockDim.y+threadIdx.y, c=blockIdx.x*blockDim.x+threadIdx.x;
    if(r>=h||c>=h)return;
    const int e11=in[r*n+c],e12=in[r*n+c+h],e21=in[(r+h)*n+c],e22=in[(r+h)*n+c+h];
    int v=0;
    switch(blockIdx.z) {
      case 0:v=e11+e22;break;
      case 1:v=right?e11:e21+e22;break;
      case 2:v=right?e21-e22:e11;break;
      case 3:v=right?e12-e11:e22;break;
      case 4:v=right?e22:e11+e12;break;
      case 5:v=right?e11+e21:e21-e11;break;
      case 6:v=right?e12+e22:e12-e22;break;
    }
    if(v<-128||v>127)atomicOr(invalid,1);
    out[(blockIdx.z*h+r)*h+c]=v<-128||v>127?0:v;
}

__device__ __forceinline__ int combine(const int* p,int offset,int plane,int quadrant) {
    if(quadrant==0)return p[offset]+p[offset+3*plane]-p[offset+4*plane]+p[offset+6*plane];
    if(quadrant==1)return p[offset+2*plane]+p[offset+4*plane];
    if(quadrant==2)return p[offset+plane]+p[offset+3*plane];
    return p[offset]-p[offset+plane]+p[offset+2*plane]+p[offset+5*plane];
}

extern "C" __global__ void strassen_finish(const int* p,const int* q,const int* r,
                                          const float* a,const float* b,float* output,
                                          int n,int base,int exact) {
    const int i=blockIdx.y*blockDim.y+threadIdx.y, j=blockIdx.x*blockDim.x+threadIdx.x,h=n/2;
    if(i>=n||j>=n)return;
    const int off=(i%h)*h+j%h, tile=2*(i>=h)+(j>=h), plane=h*h;
    long long z=base*((base-1LL)*combine(p,off,plane,tile)+combine(q,off,plane,tile));
    if(exact)z-=(base-1LL)*combine(r,off,plane,tile);
    output[i*n+j]=static_cast<float>(z)*(a[i]*b[j]);
}
