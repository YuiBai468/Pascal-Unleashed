constexpr int Tile = 32;
constexpr int PackedK = 8;

extern "C" __global__ __launch_bounds__(128)
void ar2_gemm(const int* a_high, const int* a_sum, const int* b_high, const int* b_sum,
              const float* a_scales, const float* b_scales, const int* bases,
              float* output, int m, int n, int packed_k) {
    __shared__ int a0[Tile][PackedK];
    __shared__ int a1[Tile][PackedK];
    __shared__ int b0[PackedK][Tile];
    __shared__ int b1[PackedK][Tile];

    const int tile_row = blockIdx.y * Tile;
    const int tile_col = blockIdx.x * Tile;
    const int thread = threadIdx.y * blockDim.x + threadIdx.x;
    float result[2][4] = {};

    for (int start = 0; start < packed_k; start += PackedK) {
        for (int i = thread; i < Tile * PackedK; i += 128) {
            const int line = i / PackedK;
            const int offset = i % PackedK;
            const int k = start + offset;
            const int row = tile_row + line;
            const int col = tile_col + line;
            const bool a_valid = (row < m && k < packed_k);
            const bool b_valid = (col < n && k < packed_k);
            a0[line][offset] = a_valid ? a_high[row * packed_k + k] : 0;
            a1[line][offset] = a_valid ? a_sum[row * packed_k + k] : 0;
            b0[offset][line] = b_valid ? b_high[col * packed_k + k] : 0;
            b1[offset][line] = b_valid ? b_sum[col * packed_k + k] : 0;
        }
        __syncthreads();

        int high[2][4] = {};
        int mixed[2][4] = {};
        #pragma unroll
        for (int p = 0; p < PackedK; ++p) {
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int row = threadIdx.y + i * 16;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int col = threadIdx.x + j * 8;
                    high[i][j] = __dp4a(a0[row][p], b0[p][col], high[i][j]);
                    mixed[i][j] = __dp4a(a1[row][p], b1[p][col], mixed[i][j]);
                }
            }
        }
        __syncthreads();

        const int k_block = start / PackedK;
        const int base = bases[k_block];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int row = tile_row + threadIdx.y + i * 16;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int col = tile_col + threadIdx.x + j * 8;
                if (row < m && col < n) {
                    const int sum = base * ((base - 1) * high[i][j] + mixed[i][j]);
                    const float scale = a_scales[k_block * m + row] * b_scales[k_block * n + col];
                    result[i][j] = fmaf(static_cast<float>(sum), scale, result[i][j]);
                }
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int row = tile_row + threadIdx.y + i * 16;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int col = tile_col + threadIdx.x + j * 8;
            if (row < m && col < n) {
                output[row * n + col] = result[i][j];
            }
        }
    }
}
