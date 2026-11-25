constexpr int threads_x = 8;
constexpr int threads_y = 16;
constexpr int threads = threads_x * threads_y;

template<int Rows, int Cols, int PackedTile, bool Single>
__device__ __forceinline__ void grouped_gemm(
    const int2* a, const int2* b, const float* alpha, const float* beta,
    void* output, int m, int n, int packed_k, int packed_group) {
    constexpr int tile_m = Rows * threads_y;
    constexpr int tile_n = Cols * threads_x;
    constexpr int a_pitch = tile_m + 4;
    constexpr int b_pitch = tile_n + 4;
    __shared__ int a_high[PackedTile][a_pitch];
    __shared__ int a_mixed[Single ? 1 : PackedTile][Single ? 1 : a_pitch];
    __shared__ int b_high[PackedTile][b_pitch];
    __shared__ int b_mixed[Single ? 1 : PackedTile][Single ? 1 : b_pitch];

    const int tile_row = blockIdx.y * tile_m;
    const int tile_col = blockIdx.x * tile_n;
    const int lane_row = threadIdx.y;
    const int lane_col = threadIdx.x;
    const int tid = lane_row * threads_x + lane_col;
    float result[Rows][Cols] = {};

    for (int begin = 0, group = 0; begin < packed_k; begin += packed_group, ++group) {
        int high[Rows][Cols] = {};
        int mixed[Rows][Cols] = {};
        const int end = min(begin + packed_group, packed_k);
        for (int start = begin; start < end; start += PackedTile) {
            #pragma unroll
            for (int i = tid; i < tile_m * PackedTile; i += threads) {
                const int row = i / PackedTile;
                const int p = i % PackedTile;
                int2 value = {};
                if (tile_row + row < m && start + p < end) {
                    value = a[(tile_row + row) * packed_k + start + p];
                }
                a_high[p][row] = value.x;
                if (!Single) a_mixed[p][row] = value.y;
            }
            #pragma unroll
            for (int i = tid; i < tile_n * PackedTile; i += threads) {
                const int col = i / PackedTile;
                const int p = i % PackedTile;
                int2 value = {};
                if (tile_col + col < n && start + p < end) {
                    value = b[(tile_col + col) * packed_k + start + p];
                }
                b_high[p][col] = value.x;
                if (!Single) b_mixed[p][col] = value.y;
            }
            __syncthreads();

            #pragma unroll
            for (int p = 0; p < PackedTile; ++p) {
                int ah[Rows], am[Rows], bh[Cols], bm[Cols];
                #pragma unroll
                for (int i = 0; i < Rows; ++i) {
                    const int row = lane_row + i * threads_y;
                    ah[i] = a_high[p][row];
                    if (!Single) am[i] = a_mixed[p][row];
                }
                #pragma unroll
                for (int j = 0; j < Cols; ++j) {
                    const int col = lane_col + j * threads_x;
                    bh[j] = b_high[p][col];
                    if (!Single) bm[j] = b_mixed[p][col];
                }
                #pragma unroll
                for (int i = 0; i < Rows; ++i) {
                    #pragma unroll
                    for (int j = 0; j < Cols; ++j) {
                        high[i][j] = __dp4a(ah[i], bh[j], high[i][j]);
                        if (!Single) mixed[i][j] = __dp4a(am[i], bm[j], mixed[i][j]);
                    }
                }
            }
            __syncthreads();
        }

        #pragma unroll
        for (int i = 0; i < Rows; ++i) {
            const int row = tile_row + lane_row + i * threads_y;
            if (row < m) {
                const float sa = Single ? 0.0f : 16.0f * alpha[group * m + row];
                #pragma unroll
                for (int j = 0; j < Cols; ++j) {
                    const int col = tile_col + lane_col + j * threads_x;
                    if (col < n) {
                        if (Single) {
                            static_cast<int*>(output)[row * n + col] = high[i][j];
                        } else {
                            const int z = 15 * high[i][j] + mixed[i][j];
                            const float scale = sa * beta[group * n + col];
                            result[i][j] = fmaf(static_cast<float>(z), scale, result[i][j]);
                        }
                    }
                }
            }
        }
    }
    if (!Single) {
        #pragma unroll
        for (int i = 0; i < Rows; ++i) {
            const int row = tile_row + lane_row + i * threads_y;
            #pragma unroll
            for (int j = 0; j < Cols; ++j) {
                const int col = tile_col + lane_col + j * threads_x;
                if (row < m && col < n) static_cast<float*>(output)[row * n + col] = result[i][j];
            }
        }
    }
}

#define GPU_ENTRY(name, rows, cols, tile_k, single) \
extern "C" __global__ __launch_bounds__(128) \
void name(const int2* a, const int2* b, const float* alpha, const float* beta, \
          void* output, int m, int n, int packed_k, int packed_group) { \
    grouped_gemm<rows, cols, tile_k, single>(a, b, alpha, beta, output, m, n, packed_k, packed_group); \
}

GPU_ENTRY(rgs32x32_k32, 2, 4, 8, false)
GPU_ENTRY(rgs64x32_k32, 4, 4, 8, false)
GPU_ENTRY(rgs32x64_k32, 2, 8, 8, false)
GPU_ENTRY(rgs64x64_k32, 4, 8, 8, false)
GPU_ENTRY(rgs64x64_k64, 4, 8, 16, false)
GPU_ENTRY(int8_64x64_k32, 4, 8, 8, true)

extern "C" __global__ void rgs_finish(const int* high, const int* mixed,
                                      const float* alpha, const float* beta,
                                      float* output, int m, int n,
                                      int group, int combined) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < m && col < n) {
        const int index = row * n + col;
        const int z = combined ? mixed[index] : 15 * high[index] + mixed[index];
        const float scale = (16.0f * alpha[group * m + row]) * beta[group * n + col];
        const float prior = group == 0 ? 0.0f : output[index];
        output[index] = fmaf(static_cast<float>(z), scale, prior);
    }
}
