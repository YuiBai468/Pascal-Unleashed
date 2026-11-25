// Both packed operand matrices store contiguous rows of K. Finish in FP32.
template<int RowGroups, int ColGroups, int KWords, int Unroll, bool SeparateHalves>
__device__ void gemm_packed_fused(const int* lhs, const int* rhs,
                                   const float* row_scale, const float* col_scale,
                                   float* output, int n, int k_words) {
    constexpr int rows = RowGroups * 8;
    constexpr int cols = ColGroups * 8;
    constexpr int threads = RowGroups * ColGroups;
    __shared__ __align__(16) int left_tile[KWords][rows];
    __shared__ __align__(16) int right_tile[KWords][cols];
    const int tid = threadIdx.x;
    const int row_group = tid / ColGroups;
    const int col_group = tid % ColGroups;
    const int first_row = blockIdx.y * rows;
    const int first_col = blockIdx.x * cols;
    int accumulator[8][8] = {};

    for (int base = 0; base < k_words; base += KWords) {
        #pragma unroll
        for (int x = 0; x < rows * KWords / threads; ++x) {
            const int index = tid + x * threads;
            const int row = index / KWords;
            const int depth = index % KWords;
            const int slot = SeparateHalves ? (row / 8 * 4 + row % 4 + ((row & 4) ? rows / 2 : 0)) : row;
            left_tile[depth][slot] = __ldg(lhs + (first_row + row) * k_words + base + depth);
        }
        #pragma unroll
        for (int x = 0; x < cols * KWords / threads; ++x) {
            const int index = tid + x * threads;
            const int col = index / KWords;
            const int depth = index % KWords;
            const int slot = SeparateHalves ? (col / 8 * 4 + col % 4 + ((col & 4) ? cols / 2 : 0)) : col;
            right_tile[depth][slot] = __ldg(rhs + (first_col + col) * k_words + base + depth);
        }
        __syncthreads();

        #pragma unroll Unroll
        for (int depth = 0; depth < KWords; ++depth) {
            const int a0 = SeparateHalves ? row_group * 4 : row_group * 8;
            const int a1 = a0 + (SeparateHalves ? rows / 2 : 4);
            const int b0 = SeparateHalves ? col_group * 4 : col_group * 8;
            const int b1 = b0 + (SeparateHalves ? cols / 2 : 4);
            const int4 al = *reinterpret_cast<int4*>(left_tile[depth] + a0);
            const int4 ah = *reinterpret_cast<int4*>(left_tile[depth] + a1);
            const int4 bl = *reinterpret_cast<int4*>(right_tile[depth] + b0);
            const int4 bh = *reinterpret_cast<int4*>(right_tile[depth] + b1);
            const int a[8] = {al.x, al.y, al.z, al.w, ah.x, ah.y, ah.z, ah.w};
            const int b[8] = {bl.x, bl.y, bl.z, bl.w, bh.x, bh.y, bh.z, bh.w};
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) accumulator[i][j] = __dp4a(a[i], b[j], accumulator[i][j]);
            }
        }
        __syncthreads();
    }

    const int col = first_col + 8 * col_group;
    const float4 s0 = __ldg(reinterpret_cast<const float4*>(col_scale + col));
    const float4 s1 = __ldg(reinterpret_cast<const float4*>(col_scale + col + 4));
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = first_row + 8 * row_group + i;
        const float scale = __ldg(row_scale + row);
        float4* dest = reinterpret_cast<float4*>(output + row * n + col);
        dest[0] = make_float4(accumulator[i][0] * (scale * s0.x), accumulator[i][1] * (scale * s0.y),
                              accumulator[i][2] * (scale * s0.z), accumulator[i][3] * (scale * s0.w));
        dest[1] = make_float4(accumulator[i][4] * (scale * s1.x), accumulator[i][5] * (scale * s1.y),
                              accumulator[i][6] * (scale * s1.z), accumulator[i][7] * (scale * s1.w));
    }
}

#define KERNEL(name, row_groups, col_groups, k_words, unroll, separated) \
extern "C" __global__ __launch_bounds__(row_groups * col_groups) \
void name(const int* left, const int* right, const float* a, const float* b, float* out, int n, int k) { \
    gemm_packed_fused<row_groups, col_groups, k_words, unroll, separated>(left, right, a, b, out, n, k / 4); \
}
KERNEL(fused_128_64_64_u2,16,8,16,2,true)
KERNEL(fused_64_128_64_u2,8,16,16,2,true)
KERNEL(fused_64_64_32_u2,8,8,8,2,true)
KERNEL(fused_64_64_64_u2,8,8,16,2,true)
KERNEL(fused_128_64_32_u2,16,8,8,2,true)
KERNEL(fused_128_64_64_u4,16,8,16,4,true)
KERNEL(fused_128_64_64_u2_linear,16,8,16,2,false)

template<int Groups, int ColGroups, int KWords>
__device__ __forceinline__ void fetch_vectors(const int* source, int first, int depth, int stride,
                                              int4 (&vectors)[8 * KWords / (4 * ColGroups)]) {
    constexpr int rows = 8 * Groups, lanes = Groups * ColGroups;
    constexpr int per_row = KWords / 4;
    #pragma unroll
    for (int i = 0; i < rows * per_row / lanes; ++i) {
        const int vector = threadIdx.x + i * lanes;
        const int row = vector / per_row;
        const int col = (vector % per_row) * 4;
        vectors[i] = __ldg(reinterpret_cast<const int4*>(source + (first + row) * stride + depth + col));
    }
}

template<int RowCount, int Threads, int KWords, int Count>
__device__ __forceinline__ void stage_vectors(int (&tile)[KWords][RowCount], const int4 (&values)[Count]) {
    constexpr int per_row = KWords / 4;
    #pragma unroll
    for (int i = 0; i < Count; ++i) {
        const int loc = threadIdx.x + i * Threads;
        const int row = loc / per_row;
        const int k = (loc % per_row) * 4;
        const int slot = row / 8 * 4 + row % 4 + ((row & 4) ? RowCount / 2 : 0);
        tile[k][slot] = values[i].x;
        tile[k+1][slot] = values[i].y;
        tile[k+2][slot] = values[i].z;
        tile[k+3][slot] = values[i].w;
    }
}

template<int R, int C, int KWords, int Unroll>
__device__ void pipeline_dot_out(const int* a, const int* b, const float* sa, const float* sb,
                                 float* out, int n, int k_words) {
    constexpr int rows = 8*R, cols = 8*C, threads = R*C;
    __shared__ __align__(16) int at[2][KWords][rows], bt[2][KWords][cols];
    const int row0 = blockIdx.y * rows, col0 = blockIdx.x * cols;
    const int r = threadIdx.x/C, c = threadIdx.x%C;
    int accumulation[8][8] = {};
    int4 next_a[rows * KWords / 4 / threads];
    int4 next_b[cols * KWords / 4 / threads];
    fetch_vectors<R,C,KWords>(a, row0, 0, k_words, next_a);
    fetch_vectors<C,R,KWords>(b, col0, 0, k_words, next_b);
    stage_vectors<rows,threads,KWords>(at[0], next_a);
    stage_vectors<cols,threads,KWords>(bt[0], next_b);
    __syncthreads();
    for (int stage = 0; stage < k_words; stage += KWords) {
        const int active = (stage / KWords) & 1;
        const int next = stage + KWords;
        if (next < k_words) {
            fetch_vectors<R,C,KWords>(a, row0, next, k_words, next_a);
            fetch_vectors<C,R,KWords>(b, col0, next, k_words, next_b);
        }
        #pragma unroll Unroll
        for (int depth = 0; depth < KWords; ++depth) {
            const int4 av0 = *reinterpret_cast<const int4*>(at[active][depth] + 4*r);
            const int4 av1 = *reinterpret_cast<const int4*>(at[active][depth] + 4*r+rows/2);
            const int4 bv0 = *reinterpret_cast<const int4*>(bt[active][depth] + 4*c);
            const int4 bv1 = *reinterpret_cast<const int4*>(bt[active][depth] + 4*c+cols/2);
            const int av[8] = {av0.x,av0.y,av0.z,av0.w,av1.x,av1.y,av1.z,av1.w};
            const int bv[8] = {bv0.x,bv0.y,bv0.z,bv0.w,bv1.x,bv1.y,bv1.z,bv1.w};
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) accumulation[i][j] = __dp4a(av[i],bv[j],accumulation[i][j]);
            }
        }
        if (next < k_words) {
            stage_vectors<rows,threads,KWords>(at[active ^ 1], next_a);
            stage_vectors<cols,threads,KWords>(bt[active ^ 1], next_b);
            __syncthreads();
        }
    }
    const int col = col0 + 8*c;
    const float4 scale0 = __ldg(reinterpret_cast<const float4*>(sb+col));
    const float4 scale1 = __ldg(reinterpret_cast<const float4*>(sb+col+4));
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = row0 + 8*r + i;
        const float scale = __ldg(sa + row);
        float4* dst = reinterpret_cast<float4*>(out + row*n + col);
        dst[0] = make_float4(accumulation[i][0]*(scale*scale0.x),accumulation[i][1]*(scale*scale0.y),
                             accumulation[i][2]*(scale*scale0.z),accumulation[i][3]*(scale*scale0.w));
        dst[1] = make_float4(accumulation[i][4]*(scale*scale1.x),accumulation[i][5]*(scale*scale1.y),
                             accumulation[i][6]*(scale*scale1.z),accumulation[i][7]*(scale*scale1.w));
    }
}

#define PIPE(name, r, c, depth, unroll) extern "C" __global__ __launch_bounds__(r*c) \
void name(const int* left, const int* right, const float* a, const float* b, float* out, int n, int k) { \
    pipeline_dot_out<r,c,depth,unroll>(left,right,a,b,out,n,k/4); }
PIPE(pipe_128_64_32,16,8,8,4)
PIPE(pipe_128_64_64,16,8,16,4)
PIPE(pipe_128_128_32,16,16,8,4)
PIPE(pipe_64_128_32,8,16,8,4)
PIPE(pipe_64_64_32,8,8,8,4)
