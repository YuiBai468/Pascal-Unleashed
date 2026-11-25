__device__ __forceinline__ int centered(int x, int base) {
    return ((x + base/2) & (base-1)) - base/2;
}

// Four resident packed operands per thread. Inline PTX preserves each measured instruction.
template<int Mode> __device__ void run_micro(const int* words, int* out, int segments) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    int z0=0, z1=0, z2=0, z3=0;
    float f0=0, f1=0, f2=0, f3=0;
    for (int s=0;s<segments;++s) {
        const int i = 4*(s*gridDim.x*blockDim.x+t);
        const int a = words[i], b = words[i+1], c = words[i+2], d = words[i+3];
        if (Mode == 0 || Mode == 1 || Mode == 2 || Mode == 3 || Mode == 8) {
            int p0=0,p1=0,p2=0,p3=0;
            #pragma unroll
            for(int j=0;j<(Mode == 3 ? 30 : 7);++j) {
                if (Mode == 0 || Mode == 2 || Mode == 8) {
                    asm volatile("dp4a.s32.s32 %0,%1,%2,%0;" : "+r"(p0) : "r"(a),"r"(b));
                    asm volatile("dp4a.s32.s32 %0,%1,%2,%0;" : "+r"(p1) : "r"(c),"r"(d));
                    asm volatile("dp4a.s32.s32 %0,%1,%2,%0;" : "+r"(p2) : "r"(a),"r"(d));
                    asm volatile("dp4a.s32.s32 %0,%1,%2,%0;" : "+r"(p3) : "r"(c),"r"(b));
                } else {
                    asm volatile("dp2a.lo.s32.s32 %0,%1,%2,%0;" : "+r"(p0) : "r"(a),"r"(b));
                    asm volatile("dp2a.lo.s32.s32 %0,%1,%2,%0;" : "+r"(p1) : "r"(c),"r"(d));
                    asm volatile("dp2a.lo.s32.s32 %0,%1,%2,%0;" : "+r"(p2) : "r"(a),"r"(d));
                    asm volatile("dp2a.lo.s32.s32 %0,%1,%2,%0;" : "+r"(p3) : "r"(c),"r"(b));
                }
            }
            if (Mode == 2 || Mode == 3) {
                const int p[4]={p0,p1,p2,p3};
                #pragma unroll
                for(int j=0;j<4;++j) {
                    const int base = Mode==2 ? 64 : 128;
                    const int low = centered(p[j],base);
                    const int remaining = (p[j]-low) >> (Mode==2 ? 6 : 7);
                    const int mid=Mode==3 ? centered(remaining,128) : remaining;
                    z0+=low;
                    z1+=mid;
                    if(Mode==3) z2+=(remaining-mid)>>7;
                }
            } else { z0+=p0; z1+=p1; z2+=p2; z3+=p3; }
        } else {
            #pragma unroll
            for(int j=0;j<(Mode==4 ? 28 : 1);++j) {
                if (Mode==4) {
                    const float u=__int_as_float(a),v=__int_as_float(b),w=__int_as_float(c),x=__int_as_float(d);
                    asm volatile("fma.rn.f32 %0,%1,%2,%0;" : "+f"(f0) : "f"(u),"f"(v));
                    asm volatile("fma.rn.f32 %0,%1,%2,%0;" : "+f"(f1) : "f"(w),"f"(x));
                    asm volatile("fma.rn.f32 %0,%1,%2,%0;" : "+f"(f2) : "f"(w),"f"(v));
                    asm volatile("fma.rn.f32 %0,%1,%2,%0;" : "+f"(f3) : "f"(u),"f"(x));
                } else if (Mode==5) {
                    int p;
                    asm volatile("popc.b32 %0,%1;" : "=r"(p) : "r"(a^b)); z0+=32-2*p;
                    asm volatile("popc.b32 %0,%1;" : "=r"(p) : "r"(c^d)); z1+=32-2*p;
                } else {
                    int hit, miss;
                    asm volatile("popc.b32 %0,%1;" : "=r"(hit) : "r"((a&c)|(b&d)));
                    asm volatile("popc.b32 %0,%1;" : "=r"(miss) : "r"((a&d)|(b&c)));
                    z0+=hit-miss;
                }
            }
        }
    }
    out[4*t]=Mode==4?__float_as_int(f0):z0;
    out[4*t+1]=Mode==4?__float_as_int(f1):z1;
    out[4*t+2]=Mode==4?__float_as_int(f2):z2;
    out[4*t+3]=Mode==4?__float_as_int(f3):z3;
}
#define ENTRY(name,mode) extern "C" __global__ void name(const int* a,int* b,int segments){run_micro<mode>(a,b,segments);}
ENTRY(dense_dp4a,0)
ENTRY(dense_dp2a,1)
ENTRY(packed_dp4a,2)
ENTRY(packed_dp2a,3)
ENTRY(dense_fma,4)
ENTRY(binary_pop,5)
ENTRY(ternary_pop,6)
ENTRY(ternary_dp4a,8)
