extern "C" __global__ void finish_20t(const int* first, const int* second, const int* low,
                                     const float* alpha, const float* beta,
                                     const float* low_alpha, const float* low_beta,
                                     float* output, int m, int n, int ar2) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= n) return;
    const int index = row * n + col;
    const int z = ar2 ? 15 * first[index] + second[index] : first[index];
    const float scale = (ar2 == 2 ? 16.0f*alpha[row] : alpha[row]) * beta[col];
    const float prior = low ? static_cast<float>(low[index]) * (low_alpha[row] * low_beta[col]) : 0.0f;
    output[index] = fmaf(static_cast<float>(z), scale, prior);
}

__device__ __forceinline__ float half_value(unsigned short bits) {
    float value;
    asm("cvt.f32.f16 %0, %1;" : "=f"(value) : "h"(bits));
    return value;
}

extern "C" __global__ void pack_one_round(const uint2* input, unsigned int* quantized,
                                           float* scales, unsigned int* invalid,
                                           int stride, int begin, int length) {
    __shared__ uint2 values[1024];
    __shared__ unsigned int warp_max[8];
    __shared__ float inverse;
    const int row = blockIdx.x, tid = threadIdx.x;
    unsigned int maximum = 0;
    for (int i = tid; i < length; i += 256) {
        uint2 data = input[row*stride + begin + i];
        const unsigned int a = data.x & 0x7fff, b = (data.x >> 16) & 0x7fff;
        const unsigned int c = data.y & 0x7fff, d = (data.y >> 16) & 0x7fff;
        const unsigned int peak = max(max(a, b), max(c, d));
        if (peak >= 0x7c00) {
            atomicOr(invalid, 1u);
            if (a >= 0x7c00) data.x &= 0xffff0000u;
            if (b >= 0x7c00) data.x &= 0x0000ffffu;
            if (c >= 0x7c00) data.y &= 0xffff0000u;
            if (d >= 0x7c00) data.y &= 0x0000ffffu;
            maximum = max(maximum, max(max(a < 0x7c00 ? a : 0, b < 0x7c00 ? b : 0),
                                       max(c < 0x7c00 ? c : 0, d < 0x7c00 ? d : 0)));
        } else maximum = max(maximum, peak);
        values[i] = data;
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        maximum = max(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
    }
    if (tid % 32 == 0) warp_max[tid / 32] = maximum;
    __syncthreads();
    if (tid < 32) {
        maximum = tid < 8 ? warp_max[tid] : 0;
        for (int offset = 4; offset > 0; offset /= 2) {
            maximum = max(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
        }
        if (tid == 0) {
            const float scale = maximum ? half_value(maximum) * (1.0f / 127.0f) : 1.0f;
            scales[row] = scale;
            inverse = __frcp_rn(scale);
        }
    }
    __syncthreads();
    for (int i = tid; i < length; i += 256) {
        const uint2 data = values[i];
        unsigned int packed = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned short bits = (j < 2 ? data.x : data.y) >> (16 * (j % 2));
            const int q = min(127, max(-127, __float2int_rn(half_value(bits) * inverse)));
            packed |= (static_cast<unsigned int>(q) & 255) << (8*j);
        }
        quantized[row*length+i] = packed;
    }
}
