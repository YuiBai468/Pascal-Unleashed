constexpr int sample_size = 256;
constexpr int fixed_limit = 1904;

__device__ __forceinline__ float decode_half(unsigned short bits) {
    float value;
    asm("cvt.f32.f16 %0, %1;" : "=f"(value) : "h"(bits));
    return value;
}

__device__ __forceinline__ float power2(int exponent) {
    // The difference between two finite binary16 exponent ratios bounds the shifts by 40.
    return __int_as_float((exponent + 127) << 23);
}

extern "C" __global__ void block_maxima(const unsigned short* input, unsigned short* maxima,
                                       unsigned int* invalid, int rows, int k, int stride) {
    const int row = blockIdx.y * 8 + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int col = blockIdx.x * 32 + lane;
    unsigned int maximum = 0;
    if (row < rows && col < k) {
        maximum = input[row * stride + col] & 0x7fff;
        if (maximum >= 0x7c00) {
            atomicOr(invalid, 1u);
            maximum = 0;
        }
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        maximum = max(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
    }
    if (row < rows && lane == 0) maxima[blockIdx.x * rows + row] = maximum;
}

__device__ float median_log(const unsigned short* values) {
    int first = 0, end = sample_size;
    while (first < end) {
        const int middle = (first + end) / 2;
        if (values[middle] == 0) first = middle + 1;
        else end = middle;
    }
    if (first == sample_size) return 0;
    const int count = sample_size - first;
    const float low = log2f(decode_half(values[first + (count - 1) / 2]));
    const float high = log2f(decode_half(values[first + count / 2]));
    return 0.5f * (low + high);
}

extern "C" __global__ void estimate_differences(const unsigned short* a, const unsigned short* b,
                                               float* differences, int m, int n) {
    __shared__ unsigned short left[sample_size];
    __shared__ unsigned short right[sample_size];
    const int tid = threadIdx.x;
    const int row = m < sample_size ? tid : tid * m / sample_size;
    const int col = n < sample_size ? tid : tid * n / sample_size;
    left[tid] = row < m ? a[blockIdx.x * m + row] : 0;
    right[tid] = col < n ? b[blockIdx.x * n + col] : 0;
    __syncthreads();
    for (int size = 2; size <= sample_size; size *= 2) {
        for (int stride = size / 2; stride > 0; stride /= 2) {
            if ((tid & stride) == 0) {
                const int partner = tid | stride;
                const bool ascending = (tid & size) == 0;
                const unsigned short a0 = left[tid], a1 = left[partner];
                const unsigned short b0 = right[tid], b1 = right[partner];
                left[tid] = ascending ? min(a0, a1) : max(a0, a1);
                left[partner] = ascending ? max(a0, a1) : min(a0, a1);
                right[tid] = ascending ? min(b0, b1) : max(b0, b1);
                right[partner] = ascending ? max(b0, b1) : min(b0, b1);
            }
            __syncthreads();
        }
    }
    if (tid == 0) differences[blockIdx.x] = median_log(left) - median_log(right);
}

extern "C" __global__ void select_shifts(const float* differences, int* shifts,
                                        int physical_blocks, int blocks_per_group) {
    const int block = blockIdx.x * blockDim.x + threadIdx.x;
    if (block < physical_blocks) {
        const int first = block / blocks_per_group * blocks_per_group;
        shifts[block] = blocks_per_group == 1 ? 0 : __float2int_rn(0.5f * (differences[block] - differences[first]));
    }
}

extern "C" __global__ void group_scales(const unsigned short* maxima, const int* shifts, float* scales,
                                       int rows, int physical_blocks, int blocks_per_group, int side) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const int start = blockIdx.y * blocks_per_group;
    const int end = min(physical_blocks, start + blocks_per_group);
    float maximum = 0;
    for (int block = start; block < end; ++block) {
        const float value = decode_half(maxima[block * rows + row]);
        maximum = fmaxf(maximum, value * power2(-side * shifts[block]));
    }
    scales[blockIdx.y * rows + row] = maximum == 0 ? 1 : maximum / fixed_limit;
}

template<bool Aligned>
__device__ void pack_planes(const unsigned short* input, const int* shifts,
                            const float* scales, unsigned int* high, unsigned int* mixed,
                            int rows, int k, int packed_k, int blocks_per_group, int side, int group_shift,
                            int stride) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (p >= packed_k) return;
    const int physical = p / 8;
    const int group = group_shift >= 0 ? physical >> group_shift : physical / blocks_per_group;
    const float scale = scales[group * rows + row] * power2(side * shifts[physical]);
    const float inverse = __frcp_rn(scale);
    uint2 vector = {};
    if (Aligned) vector = reinterpret_cast<const uint2*>(input)[row * (stride / 4) + p];
    unsigned int h32 = 0, g32 = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int col = 4 * p + i;
        const unsigned short bits = Aligned ? (i < 2 ? vector.x : vector.y) >> (16 * (i % 2))
                                            : col < k ? input[row * stride + col] : 0;
        const float value = (bits & 0x7fff) < 0x7c00 ? decode_half(bits) : 0.0f;
        const int q = min(fixed_limit, max(-fixed_limit, __float2int_rn(value * inverse)));
        const int h = __float2int_rn(q * (1.0f / 16));
        const int g = q - 15 * h;
        h32 |= (static_cast<unsigned int>(h) & 255) << (8 * i);
        g32 |= (static_cast<unsigned int>(g) & 255) << (8 * i);
    }
    high[row * packed_k + p] = h32;
    mixed[row * packed_k + p] = g32;
}

#define PACK_ENTRY(name, aligned) \
extern "C" __global__ void name(const unsigned short* input, const int* shifts, \
                                const float* scales, unsigned int* high, unsigned int* mixed, \
                                int rows, int k, int packed_k, int blocks_per_group, int side, int group_shift, int stride) { \
    pack_planes<aligned>(input, shifts, scales, high, mixed, rows, k, packed_k, blocks_per_group, side, group_shift, stride); \
}

PACK_ENTRY(encode_planes, false)
PACK_ENTRY(encode_planes_aligned, true)
