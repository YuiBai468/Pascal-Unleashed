// Quantize with the projected cube's bounds, then translate overflowing codes
// along an integer normal. Such translations preserve the planar projection.
template<int A, int B, int C, int D = 0>
__device__ __forceinline__ float cube_scale(float4 p, float blend) {
    float maximum = fmaxf(fabsf(p.x), fmaxf(fabsf(p.y), fmaxf(fabsf(p.z), fabsf(p.w))));
    float peak = blend * maximum;
    peak = fmaxf(peak, fabsf(fmaf(B, p.x, -A*p.y)) * (1.0f / (A+B)));
    peak = fmaxf(peak, fabsf(fmaf(C, p.x, -A*p.z)) * (1.0f / (A+C)));
    peak = fmaxf(peak, fabsf(fmaf(C, p.y, -B*p.z)) * (1.0f / (B+C)));
    if (D) {
        peak = fmaxf(peak, fabsf(fmaf(D, p.x, -A*p.w)) * (1.0f / (A+D)));
        peak = fmaxf(peak, fabsf(fmaf(D, p.y, -B*p.w)) * (1.0f / (B+D)));
        peak = fmaxf(peak, fabsf(fmaf(D, p.z, -C*p.w)) * (1.0f / (C+D)));
    }
    return peak;
}

template<int A, int B, int C, int D>
__device__ __noinline__ int4 fit_signed_bytes(int4 q, unsigned int* invalid) {
    constexpr float epsilon = .0001f;
    float lower = fmaxf((-127 - q.x) * (1.0f / A),
                        fmaxf((-127 - q.y) * (1.0f / B), (-127 - q.z) * (1.0f / C)));
    float upper = fminf((127 - q.x) * (1.0f / A),
                        fminf((127 - q.y) * (1.0f / B), (127 - q.z) * (1.0f / C)));
    if (D) {
        constexpr float reciprocal = 1.0f / (D ? D : 1);
        lower = fmaxf(lower, (-127 - q.w) * reciprocal);
        upper = fminf(upper, (127 - q.w) * reciprocal);
    }
    const int first = __float2int_ru(lower - epsilon);
    const int last = __float2int_rd(upper + epsilon);
    if (first > last) atomicOr(invalid, 2u);
    const int slide = min(max(0, first), last);
    q.x += slide * A;
    q.y += slide * B;
    q.z += slide * C;
    q.w += slide * D;
    return q;
}

template<int A, int B, int C, int D, int BinLog, bool Slide = true>
__device__ __forceinline__ int4 encode_sliding(float4 p, float factor, const unsigned short* lut,
                                              unsigned int* invalid) {
    constexpr int mask = (1 << BinLog) - 1;
    constexpr float half_bins = (1 << BinLog) / 2;
    const int a = __float2int_rd(fmaf(p.x, factor, half_bins));
    const int b = __float2int_rd(fmaf(p.y, factor, half_bins));
    const int c = __float2int_rd(fmaf(p.z, factor, half_bins));
    const int d = D ? __float2int_rd(fmaf(p.w, factor, half_bins)) : 0;
    int key = (((a & mask) << BinLog) | (b & mask));
    key = (key << BinLog) | (c & mask);
    if (D) key = (key << BinLog) | (d & mask);
    const unsigned int offset = __ldg(lut + key);
    int4 q = make_int4((a >> BinLog) + static_cast<int>(offset & 15) - 8,
                       (b >> BinLog) + static_cast<int>((offset >> 4) & 15) - 8,
                       (c >> BinLog) + static_cast<int>((offset >> 8) & 15) - 8,
                       D ? (d >> BinLog) + static_cast<int>((offset >> 12) & 15) - 8 : 0);
    if (Slide && (static_cast<unsigned>(q.x + 127) > 254 || static_cast<unsigned>(q.y + 127) > 254 ||
        static_cast<unsigned>(q.z + 127) > 254 || static_cast<unsigned>(q.w + 127) > 254))
        q = fit_signed_bytes<A,B,C,D>(q, invalid);
    return q;
}

template<bool Aligned = false>
__device__ __forceinline__ float4 triple_project(const unsigned short* input, float3 p0, float3 p1,
                                                float3 p2, float3 p3, unsigned int* invalid) {
    unsigned int a, b, c;
    if (Aligned) {
        const unsigned long long addr = reinterpret_cast<unsigned long long>(input);
        const unsigned* first = reinterpret_cast<const unsigned*>(addr & ~3ull);
        const unsigned u = __ldg(first), v = __ldg(first+1);
        const bool odd = addr & 2;
        a = odd ? u >> 16 : u & 65535;
        b = odd ? v & 65535 : u >> 16;
        c = odd ? v >> 16 : v & 65535;
    } else {
        a = input[0]; b = input[1]; c = input[2];
    }
    if ((a & 0x7fff) >= 0x7c00 || (b & 0x7fff) >= 0x7c00 || (c & 0x7fff) >= 0x7c00) {
        atomicOr(invalid, 1u);
        return make_float4(0,0,0,0);
    }
    const float x = decode_plane_half(a), y = decode_plane_half(b), z = decode_plane_half(c);
    return make_float4(fmaf(x,p0.x, fmaf(y,p0.y, z*p0.z)),
                       fmaf(x,p1.x, fmaf(y,p1.y, z*p1.z)),
                       fmaf(x,p2.x, fmaf(y,p2.y, z*p2.z)),
                       fmaf(x,p3.x, fmaf(y,p3.y, z*p3.z)));
}

template<int Triples, int BinLog, int Tune = 0, bool StatsOnly = false,
         int Threads = 256, bool VectorCode = false, int OriginalK = 4096>
__device__ void pack_sliding_row(const unsigned short* input, signed char* codes, float* scales,
                                 const float* pair_basis, const float* triple_basis,
                                 const unsigned short* pair_lut, const unsigned short* triple_lut,
                                 unsigned int* invalid, int margin, float pair_blend, float triple_blend) {
    constexpr int pairs = (OriginalK - 3 * Triples) / 2;
    constexpr int width = 4 * Triples + 3 * pairs;
    constexpr int triple_groups = (Triples + Threads - 1) / Threads;
    constexpr int pair_groups = (pairs + Threads - 1) / Threads;
    constexpr int warps = Threads / 32;
    __shared__ float warp_peak[warps], row_inverse;
    const int row = blockIdx.x, lane = threadIdx.x;
    const unsigned short* source = input + row * OriginalK;
    float4 pair_coords[pair_groups];
    float4 triple_coords[triple_groups ? triple_groups : 1];
    const float u0 = __ldg(pair_basis + 0), v0 = __ldg(pair_basis + 1);
    const float u1 = __ldg(pair_basis + 2), v1 = __ldg(pair_basis + 3);
    const float u2 = __ldg(pair_basis + 4), v2 = __ldg(pair_basis + 5);
    float3 t0, t1, t2, t3;
    if (Triples) {
        t0 = make_float3(__ldg(triple_basis+0), __ldg(triple_basis+1), __ldg(triple_basis+2));
        t1 = make_float3(__ldg(triple_basis+3), __ldg(triple_basis+4), __ldg(triple_basis+5));
        t2 = make_float3(__ldg(triple_basis+6), __ldg(triple_basis+7), __ldg(triple_basis+8));
        t3 = make_float3(__ldg(triple_basis+9), __ldg(triple_basis+10), __ldg(triple_basis+11));
    }
    float peak = 0.0f;

    #pragma unroll
    for (int group = 0; group < triple_groups; ++group) {
        const int i = group * Threads + lane;
        if (i < Triples) {
            triple_coords[group] = triple_project<VectorCode>(source + 3*i, t0, t1, t2, t3, invalid);
            const float4 p = triple_coords[group];
            peak = fmaxf(peak, (Tune & 1) ? fmaxf(fmaxf(fabsf(p.x), fabsf(p.y)), fmaxf(fabsf(p.z), fabsf(p.w))) :
                           cube_scale<3,4,5,6>(p, triple_blend));
        }
    }
    #pragma unroll
    for (int group = 0; group < pair_groups; ++group) {
        const int i = group * Threads + lane;
        if (i < pairs) {
            unsigned v = reinterpret_cast<const unsigned*>(source + 3*Triples)[i];
            if ((v & 0x7fff) >= 0x7c00 || ((v >> 16) & 0x7fff) >= 0x7c00) {
                atomicOr(invalid, 1u);
                v = 0;
            }
            const float x = decode_plane_half(static_cast<unsigned short>(v));
            const float y = decode_plane_half(static_cast<unsigned short>(v >> 16));
            const float4 p = make_float4(fmaf(x,u0,y*v0), fmaf(x,u1,y*v1), fmaf(x,u2,y*v2), 0.0f);
            pair_coords[group] = p;
            peak = fmaxf(peak, (Tune & 1) ? fmaxf(fabsf(p.x), fmaxf(fabsf(p.y), fabsf(p.z))) :
                           cube_scale<3,4,5>(pair_coords[group], pair_blend));
        }
    }

    for (int distance = 16; distance; distance >>= 1) {
        peak = fmaxf(peak, __shfl_down_sync(0xffffffff, peak, distance));
    }
    if ((lane & 31) == 0) warp_peak[lane >> 5] = peak;
    __syncthreads();
    if (lane < 32) {
        peak = lane < warps ? warp_peak[lane] : 0.0f;
        for (int distance = warps / 2; distance; distance >>= 1) {
            peak = fmaxf(peak, __shfl_down_sync(0xffffffff, peak, distance));
        }
        if (lane == 0) {
            const float scale = peak > 0 ? peak / margin : 1.0f;
            scales[row] = scale;
            row_inverse = __frcp_rn(scale);
        }
    }
    __syncthreads();
    if (StatsOnly) return;
    signed char* result = codes + row * width;
    const float pair_factor = 16 * row_inverse;
    const float triple_factor = (1 << BinLog) * row_inverse;
    #pragma unroll
    for (int group = 0; group < triple_groups; ++group) {
        const int i = group * Threads + lane;
        if (i < Triples) {
            const int4 q = encode_sliding<3,4,5,6,BinLog,!(Tune & 2)>(triple_coords[group], triple_factor, triple_lut, invalid);
            if (VectorCode) {
                reinterpret_cast<unsigned*>(result)[i] = (q.x&255u) | ((q.y&255u)<<8)
                                                        | ((q.z&255u)<<16) | ((q.w&255u)<<24);
            } else {
                result[i] = q.x;
                result[i + Triples] = q.y;
                result[i + 2 * Triples] = q.z;
                result[i + 3 * Triples] = q.w;
            }
        }
    }
    #pragma unroll
    for (int group = 0; group < pair_groups; ++group) {
        const int i = group * Threads + lane;
        if (i < pairs) {
            const int4 q = encode_sliding<3,4,5,0,4,!(Tune & 2)>(pair_coords[group], pair_factor, pair_lut, invalid);
            const int dest = i + 4 * Triples;
            result[dest] = q.x;
            result[dest + pairs] = q.y;
            result[dest + 2 * pairs] = q.z;
        }
    }
}

#define SLIDING(triples, bins) extern "C" __global__ void slide_##triples##_##bins( \
    const unsigned short* input, signed char* codes, float* scales, const float* pair_basis, \
    const float* triple_basis, const unsigned short* pair_lut, const unsigned short* triple_lut, \
    unsigned int* invalid, int margin, float pair_blend, float triple_blend) { \
    pack_sliding_row<triples,bins>(input, codes, scales, pair_basis, triple_basis, pair_lut, triple_lut, \
                                 invalid, margin, pair_blend, triple_blend); \
}
SLIDING(0,4)
SLIDING(768,3)
SLIDING(768,4)
SLIDING(896,3)
SLIDING(896,4)
SLIDING(1024,3)
SLIDING(1024,4)
SLIDING(1152,3)
SLIDING(1152,4)
SLIDING(1280,3)
SLIDING(1280,4)

#define COMPARE(triples, tune) extern "C" __global__ void profile_##triples##_##tune( \
    const unsigned short* input, signed char* codes, float* scales, const float* pair_basis, \
    const float* triple_basis, const unsigned short* pair_lut, const unsigned short* triple_lut, \
    unsigned int* invalid, int margin, float pair_blend, float triple_blend) { \
    pack_sliding_row<triples,4,tune>(input, codes, scales, pair_basis, triple_basis, pair_lut, triple_lut, \
                                  invalid, margin, pair_blend, triple_blend); \
}
COMPARE(0,1)
COMPARE(0,2)
COMPARE(768,1)
COMPARE(768,2)

template<int Triples, int BinLog, int TileRows>
__device__ void slide_right_transposed(const unsigned short* input, signed char* output, const float* scales,
                                       const float* pair_basis, const float* triple_basis,
                                       const unsigned short* pair_lut, const unsigned short* triple_lut,
                                       unsigned int* invalid, int rows) {
    constexpr int pairs = (4096 - 3 * Triples) / 2;
    __shared__ unsigned int tile[TileRows][33];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row0 = blockIdx.y * TileRows;
    const bool triple = blockIdx.x < Triples / 32;
    const int group0 = (triple ? blockIdx.x : blockIdx.x - Triples / 32) * 32;
    const float u0 = __ldg(pair_basis + 0), v0 = __ldg(pair_basis + 1);
    const float u1 = __ldg(pair_basis + 2), v1 = __ldg(pair_basis + 3);
    const float u2 = __ldg(pair_basis + 4), v2 = __ldg(pair_basis + 5);
    const float3 t0 = make_float3(__ldg(triple_basis+0), __ldg(triple_basis+1), __ldg(triple_basis+2));
    const float3 t1 = make_float3(__ldg(triple_basis+3), __ldg(triple_basis+4), __ldg(triple_basis+5));
    const float3 t2 = make_float3(__ldg(triple_basis+6), __ldg(triple_basis+7), __ldg(triple_basis+8));
    const float3 t3 = make_float3(__ldg(triple_basis+9), __ldg(triple_basis+10), __ldg(triple_basis+11));

    #pragma unroll
    for (int i = 0; i < TileRows / 8; ++i) {
        const int row = i * 8 + warp;
        const unsigned short* source = input + (row0 + row) * 4096;
        const int group = group0 + lane;
        const float inv = __frcp_rn(__ldg(scales + row0 + row));
        int4 q;
        if (triple) {
            const float4 p = triple_project(source + 3 * group, t0, t1, t2, t3, invalid);
            q = encode_sliding<3,4,5,6,BinLog>(p, (1<<BinLog) * inv, triple_lut, invalid);
        } else {
            const unsigned value = reinterpret_cast<const unsigned*>(source + 3*Triples)[group];
            const float x = decode_plane_half(static_cast<unsigned short>(value));
            const float y = decode_plane_half(static_cast<unsigned short>(value >> 16));
            const float4 p = make_float4(fmaf(x,u0,y*v0),fmaf(x,u1,y*v1),fmaf(x,u2,y*v2),0);
            q = encode_sliding<3,4,5,0,4>(p, 16 * inv, pair_lut, invalid);
        }
        tile[row][lane] = (q.x & 255u) | ((q.y & 255u) << 8) | ((q.z & 255u) << 16) | ((q.w & 255u) << 24);
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < TileRows / 8; ++i) {
        const int column = i * 8 + warp;
        const int q_row = column & 31;
        const int q_col = (column & 32) + lane;
        const unsigned value = tile[q_col][q_row];
        const int step = (triple ? Triples : pairs) * rows;
        const int origin = (triple ? 0 : 4 * Triples) + group0 + q_row;
        signed char* q = output + origin * rows + row0 + q_col;
        q[0] = static_cast<signed char>(value);
        q[step] = static_cast<signed char>(value >> 8);
        q[2 * step] = static_cast<signed char>(value >> 16);
        if (triple) q[3 * step] = static_cast<signed char>(value >> 24);
    }
}

#define RIGHT(triples, bins, tile) \
extern "C" __global__ void stat_##triples##_##bins( \
    const unsigned short* input, signed char* codes, float* scales, const float* pair_basis, \
    const float* triple_basis, const unsigned short* pair_lut, const unsigned short* triple_lut, \
    unsigned int* invalid, int margin, float pair_blend, float triple_blend) { \
    pack_sliding_row<triples,bins,0,true>(input, codes, scales, pair_basis, triple_basis, pair_lut, triple_lut, \
                                        invalid, margin, pair_blend, triple_blend); } \
extern "C" __global__ void right_##triples##_##bins( \
    const unsigned short* input, signed char* output, const float* scales, const float* pair_basis, \
    const float* triple_basis, const unsigned short* pair_lut, const unsigned short* triple_lut, \
    unsigned int* invalid, int rows) { \
    slide_right_transposed<triples,bins,tile>(input, output, scales, pair_basis, triple_basis, pair_lut, triple_lut, \
                                            invalid, rows); }

RIGHT(768,4,32)
RIGHT(896,4,32)
RIGHT(1024,4,32)
RIGHT(1152,4,32)
RIGHT(1280,4,32)

#define VECTOR(triples, threads) extern "C" __global__ __launch_bounds__(threads) \
void vec_##triples##_##threads( \
    const unsigned short* input, signed char* codes, float* scales, const float* pair_basis, \
    const float* triple_basis, const unsigned short* pair_lut, const unsigned short* triple_lut, \
    unsigned int* invalid, int margin, float pair_blend, float triple_blend) { \
    pack_sliding_row<triples,4,0,false,threads,true>(input, codes, scales, pair_basis, triple_basis, \
                                   pair_lut, triple_lut, invalid, margin, pair_blend, triple_blend); \
}
VECTOR(768,128)
VECTOR(768,256)
VECTOR(768,512)
VECTOR(896,256)
VECTOR(896,512)
VECTOR(1024,256)
VECTOR(1024,512)
VECTOR(1152,256)
VECTOR(1280,256)

extern "C" __global__ __launch_bounds__(512) void slide_8192(
    const unsigned short* input, signed char* codes, float* scales, const float* pair_basis,
    const float* triple_basis, const unsigned short* pair_lut, const unsigned short* triple_lut,
    unsigned int* invalid, int margin, float pair_blend, float triple_blend) {
    pack_sliding_row<0,4,0,false,512,false,8192>(input, codes, scales, pair_basis, triple_basis,
                        pair_lut, triple_lut, invalid, margin, pair_blend, triple_blend);
}
