extern "C" __global__ void cast_reference_input(const unsigned short* in,double* out,size_t count) {
    const size_t first=blockIdx.x*blockDim.x+threadIdx.x;
    const size_t stride=gridDim.x*blockDim.x;
    for (size_t i=first;i<count;i+=stride) out[i]=decode_plane_half(in[i]);
}

extern "C" __global__ void compare_reference_output(const float* actual,const double* expected,
                                                     double* results,size_t count) {
    __shared__ double squared_warp[8],reference_warp[8],maximum_warp[8];
    const int warp=threadIdx.x/32,lane=threadIdx.x%32;
    const size_t first=blockIdx.x*blockDim.x+threadIdx.x;
    const size_t stride=gridDim.x*blockDim.x;
    double squared=0,reference=0,maximum=0;
    for (size_t i=first;i<count;i+=stride) {
        const double target=expected[i];
        const double delta=static_cast<double>(actual[i])-target;
        squared=fma(delta,delta,squared);
        reference=fma(target,target,reference);
        maximum=fmax(maximum,fabs(delta));
    }
    for (int offset=16;offset;offset>>=1) {
        squared+=__shfl_down_sync(0xffffffff,squared,offset);
        reference+=__shfl_down_sync(0xffffffff,reference,offset);
        maximum=fmax(maximum,__shfl_down_sync(0xffffffff,maximum,offset));
    }
    if (!lane) {
        squared_warp[warp]=squared;
        reference_warp[warp]=reference;
        maximum_warp[warp]=maximum;
    }
    __syncthreads();
    if (!warp) {
        squared=lane<8?squared_warp[lane]:0;
        reference=lane<8?reference_warp[lane]:0;
        maximum=lane<8?maximum_warp[lane]:0;
        for (int offset=4;offset;offset>>=1) {
            squared+=__shfl_down_sync(0xffffffff,squared,offset);
            reference+=__shfl_down_sync(0xffffffff,reference,offset);
            maximum=fmax(maximum,__shfl_down_sync(0xffffffff,maximum,offset));
        }
        if (!lane) {
            results[blockIdx.x*3]=squared;
            results[blockIdx.x*3+1]=reference;
            results[blockIdx.x*3+2]=maximum;
        }
    }
}
