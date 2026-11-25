// Measure B row scales, then write quantized B directly in cuBLAS's K-by-N layout.
extern "C" __global__ void projected_row_scales(const unsigned int* input, const float* basis,
                                                 float* scales, float* factors,
                                                 unsigned int* invalid, int pairs, int margin) {
    __shared__ float maxima[8];
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    float peak = 0;
    for (int pair = lane; pair < pairs; pair += 256) {
        unsigned int v = input[row * pairs + pair];
        if ((v & 0x7fff) >= 0x7c00 || ((v >> 16) & 0x7fff) >= 0x7c00) {
            atomicOr(invalid, 1u);
            v = 0;
        }
        const float3 q = project_pair(v, basis);
        peak = fmaxf(peak, fmaxf(fabsf(q.x), fmaxf(fabsf(q.y), fabsf(q.z))));
    }
    for (int offset = 16; offset; offset >>= 1) {
        peak = fmaxf(peak, __shfl_down_sync(0xffffffff, peak, offset));
    }
    if ((lane & 31) == 0) maxima[lane >> 5] = peak;
    __syncthreads();
    if (lane < 32) {
        peak = lane < 8 ? maxima[lane] : 0;
        for (int offset = 4; offset; offset >>= 1) {
            peak = fmaxf(peak, __shfl_down_sync(0xffffffff, peak, offset));
        }
        if (lane == 0) {
            const float scale = peak > 0 ? peak / margin : 1.0f;
            scales[row] = scale;
            factors[row] = 16 * __frcp_rn(scale);
        }
    }
}

extern "C" __global__ void pack_transposed_right(const unsigned int* in, signed char* out,
                                                  const float* basis, const unsigned short* lookup,
                                                  const float* factors, int rows, int pairs) {
    __shared__ unsigned int tile[32][33];
    const int tid = threadIdx.x;
    const int row0 = blockIdx.y * 32;
    const int pair0 = blockIdx.x * 32;
    const int col = tid & 31;
    #pragma unroll
    for (int group = 0; group < 4; ++group) {
        const int row = group * 8 + (tid >> 5);
        const float3 q = project_pair(in[(row0 + row) * pairs + pair0 + col], basis);
        const float factor = factors[row0 + row];
        const int x = __float2int_rd(fmaf(q.x, factor, 8.0f));
        const int y = __float2int_rd(fmaf(q.y, factor, 8.0f));
        const int z = __float2int_rd(fmaf(q.z, factor, 8.0f));
        const int key = ((x & 15) << 8) | ((y & 15) << 4) | (z & 15);
        const unsigned int v = __ldg(lookup + key);
        const unsigned int a = (x >> 4) + (v & 15) - 8;
        const unsigned int b = (y >> 4) + ((v >> 4) & 15) - 8;
        const unsigned int c = (z >> 4) + ((v >> 8) & 15) - 8;
        tile[row][col] = (a & 255) | ((b & 255) << 8) | ((c & 255) << 16);
    }
    __syncthreads();
    #pragma unroll
    for (int group = 0; group < 4; ++group) {
        const int pair = group * 8 + (tid >> 5);
        const unsigned int v = tile[col][pair];
        const int dest = (pair0 + pair) * rows + row0 + col;
        out[dest] = static_cast<signed char>(v);
        out[dest + pairs * rows] = static_cast<signed char>(v >> 8);
        out[dest + 2 * pairs * rows] = static_cast<signed char>(v >> 16);
    }
}
