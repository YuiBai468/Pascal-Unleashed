#ifndef PASCAL_UNLEASHED_H
#define PASCAL_UNLEASHED_H
#include <stdint.h>
#include <stddef.h>
#if defined(PU_BUILD)
#define PU_API __declspec(dllexport)
#else
#define PU_API __declspec(dllimport)
#endif
#ifdef __cplusplus
extern "C" {
#endif

typedef struct PuContext PuContext;
typedef enum { PU_AUTO, PU_FAST, PU_BALANCED, PU_PRECISE, PU_NATIVE, PU_FP16 } PuMode;
typedef enum { PU_F16 = 2, PU_F32 = 4 } PuType;
typedef enum { PU_OK, PU_INVALID, PU_UNAVAILABLE, PU_MEMORY, PU_UNSUPPORTED, PU_CUDA } PuStatus;
enum { PU_FALLBACK=1, PU_OUTLIER_GUARD=2, PU_CAST_FP32=4, PU_CACHE_WEIGHTS=8 };
enum { PU_REASON_NONE, PU_REASON_SHAPE, PU_REASON_TOLERANCE, PU_REASON_VALUES,
       PU_REASON_MEMORY, PU_REASON_DEVICE, PU_REASON_DISABLED, PU_REASON_COST };

/* All strides are in elements. An input batch stride of zero broadcasts it.
   B has logical shape N by K (a row-major K by N B uses row=1, col=N).
   CUDA device addresses and stream belong to the context. The call completes before return. */
typedef struct { uint64_t data, row, col, batch; int32_t type; uint32_t reserved; } PuMatrix;
typedef struct { PuMatrix a,b,c; int32_t m,n,k,batches; float alpha,beta;
                 int32_t mode; uint32_t flags; float tolerance; uint32_t reserved;
                 uint64_t stream, weight_version; } PuGemm;
typedef struct { int32_t mode,reason; double milliseconds,tflops;
                 uint64_t accelerated,native,fallback,logical_ops,workspace_bytes; } PuResult;
typedef struct { char name[128]; int32_t major,minor,driver,compatible; uint64_t memory; } PuDevice;
typedef struct { PuResult run; double error,max_absolute,multiplier,native_ms;
                 uint64_t peak_device_bytes; } PuBenchmark;

PU_API const char *pu_version(void);
PU_API const char *pu_last_error(void);
PU_API int pu_devices(void);
PU_API int pu_device(int ordinal, PuDevice *info);
PU_API int pu_open(int ordinal, const wchar_t *cublas_dir, PuContext **out);
PU_API void pu_close(PuContext *ctx);
PU_API int pu_alloc(PuContext *ctx,uint64_t bytes,uint64_t *device);
PU_API int pu_free(PuContext *ctx,uint64_t device);
PU_API int pu_upload(PuContext *ctx,uint64_t device,const void *host,uint64_t bytes);
PU_API int pu_download(PuContext *ctx,void *host,uint64_t device,uint64_t bytes);
PU_API int pu_gemm(PuContext *ctx,const PuGemm *call,PuResult *result);
PU_API void pu_clear_cache(PuContext *ctx);
PU_API int pu_benchmark(PuContext *ctx,int m,int n,int k,int mode,int samples,
                       unsigned seed,int dataset,uint32_t flags,PuBenchmark *result);

#ifdef __cplusplus
}
#endif
#endif
