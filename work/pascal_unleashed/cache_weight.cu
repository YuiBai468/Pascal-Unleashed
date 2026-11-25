__device__ __forceinline__ float cached_half_to_float(unsigned short bits) {
    float value;
    asm("cvt.f32.f16 %0, %1;" : "=f"(value) : "h"(bits));
    return value;
}

__device__ __forceinline__ float cached_power2(int exponent) {
    return __int_as_float((exponent + 127) << 23);
}

extern "C" __global__ void pack_activation(const uint2* input, const int* shifts,
                                            unsigned int* high, unsigned int* mixed,
                                            float* alpha, unsigned int* invalid, int packed_k) {
    __shared__ uint2 values[1024];
    __shared__ float warp_max[8];
    __shared__ float inverse;
    const int row = blockIdx.x, tid = threadIdx.x;
    float maximum = 0;
    for (int i = tid; i < packed_k; i += 256) {
        const uint2 data = input[row * packed_k + i];
        values[i] = data;
        const unsigned int a = data.x & 0x7fff, b = (data.x >> 16) & 0x7fff;
        const unsigned int c = data.y & 0x7fff, d = (data.y >> 16) & 0x7fff;
        const unsigned int local = max(max(a, b), max(c, d));
        if (local >= 0x7c00) atomicOr(invalid, 1u);
        else maximum = fmaxf(maximum, cached_half_to_float(local) * cached_power2(-shifts[i / 8]));
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
    }
    if (tid % 32 == 0) warp_max[tid / 32] = maximum;
    __syncthreads();
    if (tid < 32) {
        maximum = tid < 8 ? warp_max[tid] : 0;
        for (int offset = 4; offset > 0; offset /= 2) {
            maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
        }
        if (tid == 0) {
            const float scale = maximum ? maximum * (1.0f / 1904.0f) : 1.0f;
            alpha[row] = scale;
            inverse = __frcp_rn(scale);
        }
    }
    __syncthreads();
    for (int i = tid; i < packed_k; i += 256) {
        const uint2 data = values[i];
        const float multiplier = inverse * cached_power2(-shifts[i / 8]);
        unsigned int first = 0, second = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned short bits = (j < 2 ? data.x : data.y) >> (16 * (j % 2));
            const float value = (bits & 0x7fff) < 0x7c00 ? cached_half_to_float(bits) : 0;
            const int q = min(1904, max(-1904, __float2int_rn(value * multiplier)));
            const int h = __float2int_rn(q * (1.0f / 16.0f));
            first |= (static_cast<unsigned int>(h) & 255) << (8 * j);
            second |= (static_cast<unsigned int>(q - 15*h) & 255) << (8 * j);
        }
        high[row * packed_k + i] = first;
        mixed[row * packed_k + i] = second;
    }
}
