#define WIN32_LEAN_AND_MEAN
#define PU_BUILD
#include <windows.h>
#include <math.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "pascal_unleashed.h"

typedef uint64_t D;
typedef void *H;
typedef struct { int cap,digits; float clip,ratio,weight; } Book;
#include "codebooks.h"

static _Thread_local char message[384];
static int fail(int status,const char *format,...) {
    va_list args;
    va_start(args,format);
    vsnprintf(message,sizeof message,format,args);
    va_end(args);
    return status;
}
const char *pu_last_error(void) { return message; }
const char *pu_version(void) { return "1.0.0"; }

#define DRIVER(X) \
 X(cuInit,(unsigned)) X(cuDeviceGetCount,(int*)) X(cuDeviceGet,(int*,int)) \
 X(cuDeviceGetName,(char*,int,int)) X(cuDeviceGetAttribute,(int*,int,int)) \
 X(cuDeviceTotalMem_v2,(size_t*,int)) X(cuDriverGetVersion,(int*)) \
 X(cuCtxGetCurrent,(H*)) X(cuCtxGetDevice,(int*)) X(cuCtxPushCurrent_v2,(H)) X(cuCtxPopCurrent_v2,(H*)) \
 X(cuDevicePrimaryCtxRetain,(H*,int)) X(cuDevicePrimaryCtxRelease_v2,(int)) \
 X(cuModuleLoadData,(H*,const void*)) X(cuModuleUnload,(H)) X(cuModuleGetFunction,(H*,H,const char*)) \
 X(cuMemAlloc_v2,(D*,size_t)) X(cuMemFree_v2,(D)) X(cuMemsetD32Async,(D,unsigned,size_t,H)) \
 X(cuMemcpyHtoD_v2,(D,const void*,size_t)) X(cuMemcpyDtoH_v2,(void*,D,size_t)) \
 X(cuMemcpyDtoHAsync_v2,(void*,D,size_t,H)) X(cuMemAllocHost_v2,(void**,size_t)) X(cuMemFreeHost,(void*)) \
 X(cuLaunchKernel,(H,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,H,void**,void**)) \
 X(cuStreamSynchronize,(H)) X(cuEventCreate,(H*,unsigned)) X(cuEventRecord,(H,H)) \
 X(cuEventElapsedTime,(float*,H,H)) X(cuEventDestroy_v2,(H)) X(cuMemGetInfo_v2,(size_t*,size_t*));
#define DECL(name,args) static int (WINAPI *name) args;
DRIVER(DECL)
#undef DECL

static INIT_ONCE driver_once=INIT_ONCE_STATIC_INIT;
static BOOL CALLBACK init_driver(PINIT_ONCE once, PVOID param,PVOID *context) {
    (void)once; (void)param;
    HMODULE dll=LoadLibraryExW(L"nvcuda.dll",NULL,LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!dll) return TRUE;
#define LOAD(name,args) name=(void*)GetProcAddress(dll,#name); if (!name) return TRUE;
    DRIVER(LOAD)
#undef LOAD
    if (!cuInit(0)) *context=dll;
    return TRUE;
}
static int ready(void) {
    void *ok=NULL;
    InitOnceExecuteOnce(&driver_once,init_driver,NULL,&ok);
    return ok!=NULL;
}
int pu_devices(void) {
    int count=0;
    if (ready()) cuDeviceGetCount(&count);
    return count;
}
int pu_device(int ordinal,PuDevice *info) {
    int dev;
    if (!info) return fail(PU_INVALID,"Device output is required");
    if (!ready() || cuDeviceGet(&dev,ordinal)) return fail(PU_UNAVAILABLE,"CUDA device %d is unavailable",ordinal);
    memset(info,0,sizeof *info);
    cuDeviceGetName(info->name,sizeof info->name,dev);
    cuDeviceGetAttribute(&info->major,75,dev);
    cuDeviceGetAttribute(&info->minor,76,dev);
    cuDeviceTotalMem_v2(&info->memory,dev);
    cuDriverGetVersion(&info->driver);
    info->compatible=info->major==6 && info->minor==1;
    return PU_OK;
}

typedef int (WINAPI *Ex)(H,int,int,int,int,int,const void*,D,int,int,D,int,int,const void*,D,int,int,int,int);
typedef int (WINAPI *SetStream)(H,H);
typedef int (WINAPI *Destroy)(H);
typedef struct { const char *name; H function; } Kernel;
typedef struct { D identity,data; uint64_t version,size,row,col; int n,k,mode,type; } Cache;
struct PuContext {
    SRWLOCK lock;
    int device,retained,compatible;
    H context,module,blas,start,end;
    HMODULE blas_dll;
    Ex gemm;
    SetStream set_stream;
    Destroy destroy;
    D tables[7],workspace;
    uint64_t workspace_bytes,cache_bytes;
    unsigned *flag;
    Kernel kernels[64];
    unsigned kernel_count,cache_count;
    Cache cache[1536];
};

static HMODULE load_blas(const wchar_t *dir) {
    wchar_t path[32768];
    if (dir && dir[0]) {
        if (!GetFullPathNameW(dir,32000,path,NULL)) return NULL;
        if (wcslen(path)>32000) return NULL;
        wcscat(path,L"\\cublas64_12.dll");
        return LoadLibraryExW(path,NULL,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    }
    HMODULE loaded=NULL;
    if (GetModuleHandleExW(0,L"cublas64_12.dll",&loaded)) return loaded;
    if (GetEnvironmentVariableW(L"PASCAL_CUDA_DLLS",path,32000)>0) {
        loaded=load_blas(path);
        if (loaded) return loaded;
    }
    if (GetEnvironmentVariableW(L"CUDA_PATH_V12_6",path,32000) || GetEnvironmentVariableW(L"CUDA_PATH",path,32000)) {
        if (wcslen(path)>32000) return NULL;
        wcscat(path,L"\\bin");
        loaded=load_blas(path);
        if (loaded) return loaded;
    }
    return LoadLibraryExW(L"cublas64_12.dll",NULL,LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
}

static int load_module(PuContext *ctx) {
    HMODULE own=NULL;
    wchar_t path[32768];
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            (LPCWSTR)(uintptr_t)pu_version,&own)) return PU_UNAVAILABLE;
    DWORD size=GetModuleFileNameW(own,path,32000);
    if (!size || size>=32000) return PU_UNAVAILABLE;
    wchar_t *last=wcsrchr(path,L'\\');
    if (!last) return PU_UNAVAILABLE;
    wcscpy(last+1,L"pascal_sm61.ptx");
    FILE *file=_wfopen(path,L"rb");
    if (!file) return fail(PU_UNAVAILABLE,"pascal_sm61.ptx is missing beside PascalUnleashed.dll");
    fseek(file,0,SEEK_END);
    long bytes=ftell(file);
    rewind(file);
    if (bytes<=0 || bytes>32*1024*1024) { fclose(file); return fail(PU_UNAVAILABLE,"Invalid CUDA module size"); }
    char *ptx=calloc((size_t)bytes+1,1);
    if (!ptx) { fclose(file); return PU_MEMORY; }
    int rc=fread(ptx,1,bytes,file)==(size_t)bytes?cuModuleLoadData(&ctx->module,ptx):1;
    free(ptx);
    fclose(file);
    return rc?fail(PU_UNAVAILABLE,"CUDA module load failed (%d)",rc):PU_OK;
}

static int enter(PuContext *ctx) {
    AcquireSRWLockExclusive(&ctx->lock);
    int status=cuCtxPushCurrent_v2(ctx->context);
    if (status) ReleaseSRWLockExclusive(&ctx->lock);
    return status?fail(PU_CUDA,"CUDA context entry failed (%d)",status):PU_OK;
}
static void leave(PuContext *ctx) {
    H ignored;
    cuCtxPopCurrent_v2(&ignored);
    ReleaseSRWLockExclusive(&ctx->lock);
}
static void clear_weights(PuContext *ctx) {
    for (unsigned i=0;i<ctx->cache_count;++i) cuMemFree_v2(ctx->cache[i].data);
    ctx->cache_count=0;
    ctx->cache_bytes=0;
}
void pu_clear_cache(PuContext *ctx) {
    if (ctx && !enter(ctx)) { clear_weights(ctx); leave(ctx); }
}
void pu_close(PuContext *ctx) {
    if (!ctx) return;
    if (ctx->context && !enter(ctx)) {
        if (ctx->blas) ctx->destroy(ctx->blas);
        if (ctx->module) cuModuleUnload(ctx->module);
        if (ctx->workspace) cuMemFree_v2(ctx->workspace);
        clear_weights(ctx);
        for (int i=0;i<7;++i) if (ctx->tables[i]) cuMemFree_v2(ctx->tables[i]);
        if (ctx->flag) cuMemFreeHost(ctx->flag);
        if (ctx->start) cuEventDestroy_v2(ctx->start);
        if (ctx->end) cuEventDestroy_v2(ctx->end);
        leave(ctx);
    }
    if (ctx->retained) cuDevicePrimaryCtxRelease_v2(ctx->device);
    if (ctx->blas_dll) FreeLibrary(ctx->blas_dll);
    free(ctx);
}
int pu_open(int ordinal,const wchar_t *dir,PuContext **out) {
    if (!out) return fail(PU_INVALID,"Context output is required");
    *out=NULL;
    PuDevice info;
    int rc=pu_device(ordinal,&info);
    if (rc) return rc;
    if (info.major<6) return fail(PU_UNAVAILABLE,"The CUDA module requires compute capability 6.0 or later");
    PuContext *ctx=calloc(1,sizeof *ctx);
    if (!ctx) return fail(PU_MEMORY,"Cannot allocate context");
    InitializeSRWLock(&ctx->lock);
    ctx->compatible=info.compatible;
    cuDeviceGet(&ctx->device,ordinal);
    int active=-1;
    cuCtxGetCurrent(&ctx->context);
    if (ctx->context) cuCtxGetDevice(&active);
    if (active!=ctx->device) {
        ctx->context=NULL;
        if (cuDevicePrimaryCtxRetain(&ctx->context,ctx->device)) goto cuda_failure;
        ctx->retained=1;
    }
    ctx->blas_dll=load_blas(dir);
    if (!ctx->blas_dll) { pu_close(ctx); return fail(PU_UNAVAILABLE,"CUDA 12 cuBLAS is missing. Select its bin directory or use an existing CUDA 12 PyTorch environment."); }
    int (WINAPI *create)(H*)=(void*)GetProcAddress(ctx->blas_dll,"cublasCreate_v2");
    ctx->gemm=(void*)GetProcAddress(ctx->blas_dll,"cublasGemmEx");
    ctx->set_stream=(void*)GetProcAddress(ctx->blas_dll,"cublasSetStream_v2");
    ctx->destroy=(void*)GetProcAddress(ctx->blas_dll,"cublasDestroy_v2");
    if (!create || !ctx->gemm || !ctx->set_stream || !ctx->destroy || enter(ctx)) goto cuda_failure;
    rc=load_module(ctx);
    if (!rc && (create(&ctx->blas) || cuMemAllocHost_v2((void**)&ctx->flag,sizeof(unsigned)) ||
                cuEventCreate(&ctx->start,0) || cuEventCreate(&ctx->end,0))) rc=fail(PU_CUDA,"CUDA workspace initialization failed");
    const void *data[]={basis,pair_lut,line_lut,middle_a,middle_b,precise_a,precise_b};
    const size_t sizes[]={sizeof basis,sizeof pair_lut,sizeof line_lut,sizeof middle_a,sizeof middle_b,sizeof precise_a,sizeof precise_b};
    if (!rc && ctx->compatible) for (int i=0;i<7;++i) {
        if (cuMemAlloc_v2(&ctx->tables[i],sizes[i]) || cuMemcpyHtoD_v2(ctx->tables[i],data[i],sizes[i])) {
            rc=fail(PU_MEMORY,"Cannot allocate CUDA codebooks");
            break;
        }
    }
    leave(ctx);
    if (rc) { pu_close(ctx); return rc; }
    *out=ctx;
    return PU_OK;
cuda_failure:
    pu_close(ctx);
    return fail(PU_CUDA,"CUDA context or cuBLAS initialization failed");
}

int pu_alloc(PuContext *ctx,uint64_t bytes,D *out) {
    if (!ctx || !out || !bytes) return fail(PU_INVALID,"Invalid allocation");
    int rc=enter(ctx);
    if (rc) return rc;
    rc=cuMemAlloc_v2(out,bytes);
    leave(ctx);
    return rc?fail(rc==2?PU_MEMORY:PU_CUDA,"CUDA allocation failed (%d)",rc):PU_OK;
}
int pu_free(PuContext *ctx,D p) {
    if (!ctx) return PU_INVALID;
    if (!p) return PU_OK;
    int rc=enter(ctx);
    if (rc) return rc;
    rc=cuMemFree_v2(p);
    leave(ctx);
    return rc?fail(PU_CUDA,"CUDA free failed (%d)",rc):PU_OK;
}
int pu_upload(PuContext *ctx,D device,const void *host,uint64_t bytes) {
    if (!ctx || !device || !host) return PU_INVALID;
    int rc=enter(ctx);
    if (rc) return rc;
    rc=cuMemcpyHtoD_v2(device,host,bytes);
    leave(ctx);
    return rc?fail(PU_CUDA,"CUDA upload failed (%d)",rc):PU_OK;
}
int pu_download(PuContext *ctx,void *host,D device,uint64_t bytes) {
    if (!ctx || !device || !host) return PU_INVALID;
    int rc=enter(ctx);
    if (rc) return rc;
    rc=cuMemcpyDtoH_v2(host,device,bytes);
    leave(ctx);
    return rc?fail(PU_CUDA,"CUDA download failed (%d)",rc):PU_OK;
}

static int launch(PuContext *ctx,H stream,const char *name,unsigned x,unsigned y,unsigned tx,unsigned ty,void **args) {
    for (unsigned i=0;i<ctx->kernel_count;++i) {
        if (!strcmp(ctx->kernels[i].name,name))
            return cuLaunchKernel(ctx->kernels[i].function,x,y,1,tx,ty,1,0,stream,args,NULL);
    }
    H kernel;
    int rc=cuModuleGetFunction(&kernel,ctx->module,name);
    if (rc) return rc;
    if (ctx->kernel_count<64) ctx->kernels[ctx->kernel_count++]=(Kernel){name,kernel};
    return cuLaunchKernel(kernel,x,y,1,tx,ty,1,0,stream,args,NULL);
}
static int round_to(int x,int align) { return (x+align-1)/align*align; }
static unsigned grid(uint64_t len) { uint64_t n=(len+255)/256; return n>4096?4096:(unsigned)n; }
typedef struct { D base; uint64_t used; } Arena;
static D block(Arena *arena,uint64_t bytes) {
    D ptr=arena->base+arena->used;
    arena->used+=(bytes+255)&~(uint64_t)255;
    return ptr;
}
static int reserve(PuContext *ctx,uint64_t bytes) {
    if (bytes<=ctx->workspace_bytes) return 0;
    if (ctx->workspace) cuMemFree_v2(ctx->workspace);
    ctx->workspace=0;
    ctx->workspace_bytes=0;
    int rc=cuMemAlloc_v2(&ctx->workspace,bytes);
    if (rc==2 && ctx->cache_count) { clear_weights(ctx); rc=cuMemAlloc_v2(&ctx->workspace,bytes); }
    if (!rc) ctx->workspace_bytes=bytes;
    return rc;
}

#define RUN(name,x,y,bx,by,...) do { rc=launch(ctx,stream,name,x,y,bx,by,(void*[]){__VA_ARGS__}); if(rc) return rc; } while(0)
static int contiguous(PuMatrix source,int rows,int cols,int out_r,int out_c,int type) {
    return source.type==type && source.col==1 && source.row==(uint64_t)cols && rows==out_r && cols==out_c;
}
static int gather(PuContext *ctx,H stream,PuMatrix src,D dst,int r,int k,int pr,int pk,int type,D bad) {
    int rc;
    RUN("pu_gather",grid((uint64_t)pr*pk),1,256,1,&src.data,&dst,&r,&k,&pr,&pk,&src.row,&src.col,&src.type,&type,&bad);
    return 0;
}
static int scatter(PuContext *ctx,H stream,const PuGemm *q,D src,int pitch,int type,float alpha,float beta) {
    int rc;
    RUN("pu_scatter",grid((uint64_t)q->m*q->n),1,256,1,&src,(void*)&q->c.data,(void*)&q->m,(void*)&q->n,&pitch,
        (void*)&q->c.row,(void*)&q->c.col,&type,(void*)&q->c.type,&alpha,&beta);
    return 0;
}
#undef RUN
#define RUN(name,x,y,bx,by,...) do { rc=launch(ctx,stream,name,x,y,bx,by,(void*[]){__VA_ARGS__}); if(rc) return rc; } while(0)

static int native(PuContext *ctx,H stream,const PuGemm *q,int mode) {
    if (!q->k || q->alpha==0.f) return scatter(ctx,stream,q,0,q->n,PU_F32,0.f,q->beta);
    int rc,m=q->m,n=q->n,k=q->k;
    int it=mode==PU_FP16?PU_F16:PU_F32,ot=it;
    /* FP16 input, FP32 accumulation is the native fallback when both inputs are FP16. */
    if (mode==PU_NATIVE && q->a.type==PU_F16 && q->b.type==PU_F16) it=PU_F16;
    int ca=contiguous(q->a,m,k,m,k,it),cb=contiguous(q->b,n,k,n,k,it);
    int direct=contiguous(q->c,m,n,m,n,ot) && mode==PU_NATIVE;
    Arena plan={0,0};
    D a=ca?0:block(&plan,(uint64_t)m*k*it), b=cb?0:block(&plan,(uint64_t)n*k*it);
    D out=direct?0:block(&plan,(uint64_t)m*n*ot);
    rc=reserve(ctx,plan.used);
    if (rc) return rc;
    a=ca?q->a.data:ctx->workspace+a;
    b=cb?q->b.data:ctx->workspace+b;
    out=direct?q->c.data:ctx->workspace+out;
    if (!ca && (rc=gather(ctx,stream,q->a,a,m,k,m,k,it,0))) return rc;
    if (!cb && (rc=gather(ctx,stream,q->b,b,n,k,n,k,it,0))) return rc;
    float alpha=direct?q->alpha:1.f,beta=direct?q->beta:0.f;
    unsigned short one=0x3c00,zero=0;
    rc=ctx->gemm(ctx->blas,1,0,n,m,k,mode==PU_FP16?(void*)&one:(void*)&alpha,b,it==2?2:0,k,a,it==2?2:0,k,
                mode==PU_FP16?(void*)&zero:(void*)&beta,out,ot==2?2:0,n,mode==PU_FP16?64:68,-1);
    if (rc) return 1000+rc;
    return direct?0:scatter(ctx,stream,q,out,n,ot,q->alpha,q->beta);
}

static int igemm(PuContext *ctx,int m,int n,int k,D a,D b,D out) {
    int one=1,zero=0;
    int rc=ctx->gemm(ctx->blas,0,0,n,m,k,&one,b,3,n,a,3,k,&zero,out,10,n,72,1);
    return rc?1000+rc:0;
}

static Cache *cache_find(PuContext *ctx,const PuGemm *q,int mode) {
    if (!(q->flags&PU_CACHE_WEIGHTS) || !q->weight_version) return NULL;
    for (unsigned i=0;i<ctx->cache_count;++i) {
        Cache *c=ctx->cache+i;
        if (c->identity==q->b.data && c->version==q->weight_version && c->n==q->n && c->k==q->k &&
            c->mode==mode && c->type==q->b.type && c->row==q->b.row && c->col==q->b.col) return c;
    }
    return NULL;
}
static Cache *cache_add(PuContext *ctx,const PuGemm *q,int mode,uint64_t size) {
    if (!(q->flags&PU_CACHE_WEIGHTS) || !q->weight_version || ctx->cache_count>=1536 ||
        size+ctx->cache_bytes>6ULL*1024*1024*1024) return NULL;
    size_t available=0,total=0;
    if (cuMemGetInfo_v2(&available,&total) || available<size+512ULL*1024*1024) return NULL;
    D data;
    if (cuMemAlloc_v2(&data,size)) return NULL;
    Cache *c=ctx->cache+ctx->cache_count++;
    *c=(Cache){q->b.data,data,q->weight_version,size,q->b.row,q->b.col,q->n,q->k,mode,q->b.type};
    ctx->cache_bytes+=size;
    return c;
}
static void cache_forget(PuContext *ctx,Cache *entry) {
    if (!entry) return;
    cuMemFree_v2(entry->data);
    ctx->cache_bytes-=entry->size;
    *entry=ctx->cache[--ctx->cache_count];
}

static int fast(PuContext *ctx,H stream,const PuGemm *q,D *result,int *pitch) {
    int rc,m=q->m,n=round_to(q->n,4),k=q->k,pairs=(k+1)/2;
    int dual=(int)nearbyint(pairs*(2560.0/8192.0)), width=round_to(3*pairs+dual,4),margin=128;
    int archived=k==8192, type=archived?PU_F16:PU_F32;
    int direct=contiguous(q->c,m,q->n,m,n,PU_F32) && q->alpha==1.f && q->beta==0.f;
    int ca=contiguous(q->a,m,k,m,k,type),cb=contiguous(q->b,q->n,k,n,k,type);
    uint64_t ml=(uint64_t)m*width,nl=(uint64_t)n*width;
    Arena side={0,0},plan={0,0};
    D right=block(&side,nl),sb=block(&side,4ULL*n);
    D a=ca?0:block(&plan,(uint64_t)m*k*type),b=cb?0:block(&plan,(uint64_t)n*k*type);
    D left=block(&plan,ml),sa=block(&plan,4ULL*m),row_right=block(&plan,nl);
    D raw=direct?0:block(&plan,4ULL*m*n),bad=block(&plan,4);
    D backup=block(&plan,side.used);
    if ((rc=reserve(ctx,plan.used))) return rc;
    /* A resize can evict the cache on memory pressure. */
    Cache *cached=cache_find(ctx,q,PU_FAST);
    Cache *slot=cached?cached:cache_add(ctx,q,PU_FAST,side.used);
    D base=ctx->workspace,rbase=slot?slot->data:base+backup,none=0;
    a=ca?q->a.data:base+a; b=cb?q->b.data:base+b;
    left+=base; sa+=base; row_right+=base; raw=direct?q->c.data:base+raw; bad+=base; right+=rbase; sb+=rbase;
    if ((rc=cuMemsetD32Async(bad,0,1,stream))) return rc;
    if (!ca && (rc=gather(ctx,stream,q->a,a,m,k,m,k,type,bad))) return rc;
    if (!cb && !cached && (rc=gather(ctx,stream,q->b,b,q->n,k,n,k,type,bad))) return rc;
    float blend=.81f,line_blend=.89f;
    if (archived) {
        RUN("edge_line_2560",m,1,512,1,&a,&left,&sa,&ctx->tables[0],&ctx->tables[1],&ctx->tables[2],&bad,&margin,&blend,&line_blend);
        if (!cached) RUN("edge_line_2560",n,1,512,1,&b,&row_right,&sb,&ctx->tables[0],&ctx->tables[1],&ctx->tables[2],&bad,&margin,&blend,&line_blend);
    } else {
        RUN("anima_pack_rows_edge",m,1,256,1,&a,&left,&sa,&k,&dual,&width,&ctx->tables[0],&ctx->tables[1],&ctx->tables[2],&bad,&margin);
        if (!cached) RUN("anima_pack_rows_edge",n,1,256,1,&b,&row_right,&sb,&k,&dual,&width,&ctx->tables[0],&ctx->tables[1],&ctx->tables[2],&bad,&margin);
    }
    if (!cached) {
        if (!(width%128) && !(n%128)) RUN("transpose_128_128_stream",width/128,n/128,1024,1,&row_right,&right,&n,&width);
        else RUN("pu_transpose",(width+31)/32,(n+31)/32,256,1,&row_right,&right,&n,&width);
    }
    if ((rc=igemm(ctx,m,n,width,left,right,raw))) return rc;
    if (!(n%1024)) RUN("unpack_plane_stream",n/1024,m,256,1,&raw,&sa,&sb,&raw,&m,&n);
    else RUN("anima_reconstruct",(n+127)/128,(m+15)/16,32,4,&raw,&sa,&sb,&none,&raw,&m,&n);
    if ((rc=cuMemcpyDtoHAsync_v2(ctx->flag,bad,4,stream)) || (rc=cuStreamSynchronize(stream))) return rc;
    if (*ctx->flag) { cache_forget(ctx,slot); return -1; }
    *result=raw; *pitch=n;
    return 0;
}

static int lanes(PuContext *ctx,H stream,const PuGemm *q,int mode,D *result,int *pitch) {
    int rc,m=round_to(q->m,128),n=round_to(q->n,128),k=round_to(q->k,64),real_k=q->k;
    int high_precision=mode==PU_PRECISE,table=high_precision?5:3;
    Book book=high_precision?precise:middle;
    int capacity=round_to((int)(real_k/(book.clip*book.clip))+5,4);
    uint64_t mk=(uint64_t)m*k,nk=(uint64_t)n*k,mn=(uint64_t)m*n;
    int direct=contiguous(q->c,q->m,q->n,m,n,PU_F32) && q->alpha==1.f && q->beta==0.f;
    int ca=contiguous(q->a,q->m,real_k,m,k,PU_F16),cb=contiguous(q->b,q->n,real_k,n,k,PU_F16);
    Arena side={0,0},work={0,0};
    D bh=block(&side,nk),bg=block(&side,nk),bc=block(&side,nk),sb=block(&side,8ULL*n);
    D cb_count=0,rb=0,rbs=0;
    if (high_precision) { cb_count=block(&side,2ULL*n); rb=block(&side,4ULL*n*capacity); rbs=block(&side,4ULL*n); }
    D a=ca?0:block(&work,2*mk),b=cb?0:block(&work,2*nk);
    D ah=block(&work,mk),ag=block(&work,mk),sa=block(&work,8ULL*m),ac=0;
    D ca_count=block(&work,2ULL*m),ra=block(&work,4ULL*m*capacity),ras=block(&work,4ULL*m);
    D tails_a=block(&work,8ULL*m*capacity),tails_b=high_precision?block(&work,8ULL*n*capacity):0;
    D h=direct?0:block(&work,4*mn),g=block(&work,4*mn),partial=0;
    D bad=block(&work,4),marks=0, backup=block(&work,side.used);
    if (high_precision) { ac=block(&work,mk); partial=block(&work,2*mn); }
    int guard=high_precision && (q->flags&PU_OUTLIER_GUARD);
    if (guard) marks=block(&work,4ULL*k);
    if ((rc=reserve(ctx,work.used))) return rc;
    Cache *cached=cache_find(ctx,q,mode);
    Cache *slot=cached?cached:cache_add(ctx,q,mode,side.used);
    D base=ctx->workspace,rbase=slot?slot->data:base+backup,none=0;
    a=ca?q->a.data:base+a; b=cb?q->b.data:base+b;
    bh+=rbase; bg+=rbase; bc+=rbase; sb+=rbase;
    ah+=base; ag+=base; sa+=base; ca_count+=base; ra+=base; ras+=base; tails_a+=base;
    h=direct?q->c.data:base+h; g+=base; bad+=base;
    if (high_precision) { cb_count+=rbase; rb+=rbase; rbs+=rbase; tails_b+=base; ac+=base; partial+=base; }
    if ((rc=cuMemsetD32Async(bad,0,1,stream))) return rc;
    if (!ca && (rc=gather(ctx,stream,q->a,a,q->m,real_k,m,k,PU_F16,bad))) return rc;
    if (!cb && (!cached || guard) && (rc=gather(ctx,stream,q->b,b,q->n,real_k,n,k,PU_F16,bad))) return rc;
    if (guard) {
        marks+=base;
        unsigned lhs=1,rhs=2;
        if ((rc=cuMemsetD32Async(marks,0,k,stream))) return rc;
        RUN("pu_risk",q->m,1,256,1,&a,&real_k,&k,&marks,&lhs,&bad);
        RUN("pu_risk",q->n,1,256,1,&b,&real_k,&k,&marks,&rhs,&bad);
        if ((rc=cuMemcpyDtoHAsync_v2(ctx->flag,bad,4,stream)) || (rc=cuStreamSynchronize(stream))) return rc;
        if (*ctx->flag) {
            if (!cached || (*ctx->flag&1u)) cache_forget(ctx,slot);
            return -1;
        }
    }
    int static_k=(k==8192 || k==16384) && k==real_k;
    if (!static_k && !guard && ca) RUN("pu_check_half",grid(mk),1,256,1,&a,&mk,&bad);
    if (!static_k && !guard && cb && !cached) RUN("pu_check_half",grid(nk),1,256,1,&b,&nk,&bad);
    int intbase=book.cap/127,packed=128;
    float gain_a=intbase,gain_b=high_precision?gain_a*3.9f/book.clip:gain_a;
    float lookup=(float)book.digits/book.cap, inv_b=1.f/gain_b,inv_a=1.f/gain_a;
    if (static_k) {
        RUN(k==8192?"clip_table_8192":"clip_table_16384",m,1,512,1,&a,&ah,&ag,&ah,&sa,&ca_count,&tails_a,&bad,
            &intbase,&book.cap,&book.clip,&capacity,&ctx->tables[table],&book.digits);
    } else {
        RUN("model_pack_left",m,1,256,1,&a,&ah,&ag,&sa,&ca_count,&tails_a,&bad,&real_k,&k,&book.cap,&book.clip,
            &capacity,&ctx->tables[table],&book.digits);
    }
    RUN(high_precision?"pack_tail_two_bytes":"pack_tail_one_byte",m,1,32,1,&tails_a,&ca_count,&ra,&ras,&capacity);
    if (!cached) {
        if (static_k && high_precision) {
            RUN(k==8192?"clip_tails_8192":"clip_tails_16384",n,1,512,1,&b,&sb,&cb_count,&tails_b,&bad,&intbase,&book.cap,&book.clip,&capacity);
        } else if (static_k) {
            RUN("dense08_peaks",n,1,256,1,&b,&sb,&bad,&k,&book.cap);
        } else {
            float cutoff=high_precision?book.clip:0.f;
            RUN("model_measure_right",n,1,256,1,&b,&none,&none,&sb,&cb_count,&tails_b,&bad,&real_k,&k,&book.cap,&cutoff,
                &capacity,&ctx->tables[table+1],&book.digits);
        }
        if (high_precision) RUN("pack_tail_two_bytes",n,1,32,1,&tails_b,&cb_count,&rb,&rbs,&capacity);
        RUN("transpose_trained_64",k/64,n/64,256,1,&b,&bh,&bg,&bc,&sb,&ctx->tables[table+1],&n,&k,&book.digits,&lookup,&inv_b,&packed);
    }
    if (high_precision) RUN("transpose_coarse_64",k/64,m/64,256,1,&a,&ac,&sa,&m,&k,&inv_a,&packed);
    if ((rc=igemm(ctx,m,n,k,ah,bh,h)) || (rc=igemm(ctx,m,n,k,ag,bg,g))) return rc;
    if (high_precision) {
        RUN("tail_direction128_compact_b",n/32,m/128,256,1,&sa,&cb_count,&rb,&rbs,&ac,&partial,&n,&k,&capacity,&gain_a);
        RUN("tail_direction128_compact_a",n/128,m/32,256,1,&h,&g,&sa,&sb,&ca_count,&ra,&ras,&bc,&h,&m,&n,&k,&capacity,
            &book.ratio,&book.weight,&gain_b,&partial,&rbs,&gain_a);
    } else {
        RUN("tail_direction128_left_q8_a",n/128,m/32,256,1,&h,&g,&sa,&sb,&ca_count,&ra,&ras,&bc,&h,&m,&n,&k,&capacity,
            &book.ratio,&book.weight,&gain_b);
    }
    if ((rc=cuMemcpyDtoHAsync_v2(ctx->flag,bad,4,stream)) || (rc=cuStreamSynchronize(stream))) return rc;
    if (*ctx->flag) { cache_forget(ctx,slot); return -1; }
    *result=h; *pitch=n;
    return 0;
}

static int descriptor(PuMatrix a,int rows,int cols,int batches,int output) {
    if (a.type!=PU_F16 && a.type!=PU_F32) return 0;
    if (rows==0 || cols==0) return 1;
    if (!a.data || a.data%a.type || a.row>UINT32_MAX || a.col>UINT32_MAX || a.batch>(1ULL<<48)) return 0;
    uint64_t span=(rows-1)*a.row+(cols-1)*a.col+1;
    if (output && !((a.col==1 && a.row>=(uint64_t)cols) || (a.row==1 && a.col>=(uint64_t)rows))) return 0;
    if (output && batches>1 && a.batch<span) return 0;
    if (a.batch && (uint64_t)(batches-1)>(UINT64_MAX/a.type-span)/a.batch) return 0;
    return ((batches-1)*a.batch+span)<=(UINT64_MAX-a.data)/a.type;
}

static int select_mode(PuContext *ctx,const PuGemm *q,int *reason) {
    int mode=q->mode;
    if (mode==PU_NATIVE || mode==PU_FP16) return mode;
    if (!ctx->compatible) { *reason=PU_REASON_DEVICE; return PU_NATIVE; }
    if (q->tolerance>0.f) { *reason=PU_REASON_TOLERANCE; return PU_NATIVE; }
    if (!q->k || q->alpha==0.f || ((q->a.type!=PU_F16 || q->b.type!=PU_F16) && !(q->flags&PU_CAST_FP32)) ||
        q->m<16 || q->n<64 || q->k<32 || q->m>65535 || q->n>65535 || q->k>32768) {
        *reason=PU_REASON_SHAPE;
        return PU_NATIVE;
    }
    if (mode!=PU_AUTO) return mode;
    if (q->m<2048 || q->n<2048 || q->k<1024) { *reason=PU_REASON_COST; return PU_NATIVE; }
    return q->k==8192 && q->m>=8192 && q->n>=8192?PU_FAST:PU_BALANCED;
}

int pu_gemm(PuContext *ctx,const PuGemm *call,PuResult *result) {
    if (!ctx || !call || !result) return fail(PU_INVALID,"Context, call and result are required");
    memset(result,0,sizeof *result);
    if (call->m<0 || call->n<0 || call->k<0 || call->m>1000000 || call->n>1000000 || call->k>1000000 ||
        call->batches<0 || call->batches>4096 || call->mode<PU_AUTO || call->mode>PU_FP16 ||
        !isfinite(call->alpha) || !isfinite(call->beta) || !isfinite(call->tolerance) || call->tolerance<0 ||
        !descriptor(call->a,call->m,call->k,call->batches,0) || !descriptor(call->b,call->n,call->k,call->batches,0) ||
        !descriptor(call->c,call->m,call->n,call->batches,1)) return fail(PU_INVALID,"Unsupported dimensions, layout, scalar or descriptor");
    if (!call->m || !call->n || !call->batches) return PU_OK;
    uint64_t ops=2ULL*call->m*call->n*call->k;
    if (ops>UINT64_MAX/call->batches) return fail(PU_INVALID,"Operation count overflow");
    int rc=enter(ctx);
    if (rc) return rc;
    H stream=(H)(uintptr_t)call->stream;
    int initial_reason=0;
    int selected=select_mode(ctx,call,&initial_reason);
    result->reason=initial_reason;
    result->mode=selected;
    if (result->reason && !(call->flags&PU_FALLBACK) && call->mode!=PU_AUTO) {
        leave(ctx);
        return fail(PU_UNSUPPORTED,"Requested mode cannot serve this operation (reason %d)",result->reason);
    }
    rc=ctx->set_stream(ctx->blas,stream);
    if (rc) rc+=1000;
    if (!rc) rc=cuEventRecord(ctx->start,stream);
    PuGemm q=*call;
    for (int batch=0;!rc && batch<q.batches;++batch) {
        int mode=selected,reason=initial_reason;
        if (mode==PU_NATIVE || mode==PU_FP16) {
            rc=native(ctx,stream,&q,mode);
        } else {
            D data=0;
            int pitch=0;
            rc=mode==PU_FAST?fast(ctx,stream,&q,&data,&pitch):lanes(ctx,stream,&q,mode,&data,&pitch);
            if ((rc==-1 || rc==2 || rc==1003 || rc==1015) && (q.flags&PU_FALLBACK)) {
                reason=rc==-1?PU_REASON_VALUES:rc==2?PU_REASON_MEMORY:PU_REASON_DEVICE;
                mode=PU_NATIVE;
                cuStreamSynchronize(stream);
                rc=native(ctx,stream,&q,mode);
            } else if (!rc && data!=q.c.data) rc=scatter(ctx,stream,&q,data,pitch,PU_F32,q.alpha,q.beta);
        }
        if (!rc) {
            result->reason=reason;
            result->mode=mode;
            if (mode==PU_NATIVE || mode==PU_FP16) ++result->native;
            else ++result->accelerated;
            if (reason) ++result->fallback;
        }
        q.a.data+=q.a.batch*q.a.type;
        q.b.data+=q.b.batch*q.b.type;
        q.c.data+=q.c.batch*q.c.type;
    }
    if (!rc) rc=cuEventRecord(ctx->end,stream);
    int sync=cuStreamSynchronize(stream);
    if (!rc) rc=sync;
    if (rc && ctx->cache_count) clear_weights(ctx);
    float duration=0;
    if (!rc) rc=cuEventElapsedTime(&duration,ctx->start,ctx->end);
    result->milliseconds=duration;
    result->logical_ops=ops*(result->native+result->accelerated);
    result->tflops=duration>0?result->logical_ops/(duration*1e9):0;
    result->workspace_bytes=ctx->workspace_bytes+ctx->cache_bytes;
    leave(ctx);
    return rc?fail(rc==2?PU_MEMORY:rc==-1?PU_UNSUPPORTED:PU_CUDA,"GEMM failed (%d)",rc):PU_OK;
}

static int cmp_double(const void *a,const void *b) {
    double x=*(const double*)a,y=*(const double*)b;
    return (x>y)-(x<y);
}

int pu_benchmark(PuContext *ctx,int m,int n,int k,int mode,int samples,unsigned seed,int dataset,uint32_t flags,PuBenchmark *out) {
    if (!ctx || !out || m<1 || n<1 || k<1 || m>65536 || n>65536 || k>65536 || samples<1 || samples>9 || dataset<0 || dataset>3)
        return fail(PU_INVALID,"Invalid benchmark dimensions, sample count or dataset");
    memset(out,0,sizeof *out);
    D buffers[5]={0};
    uint64_t ab=2ULL*m*k,bb=2ULL*n*k,mn=(uint64_t)m*n;
    int blocks=grid(mn),output_type=mode==PU_FP16?PU_F16:PU_F32;
    uint64_t cb=mn*output_type,refb=4*mn,statb=24ULL*blocks;
    uint64_t bytes[]={ab,bb,cb,refb,statb};
    int rc=PU_OK,entered=0;
    H stream=NULL;
    for (int i=0;i<5;++i) if ((rc=pu_alloc(ctx,bytes[i],buffers+i))) goto done;
    if ((rc=enter(ctx))) goto done;
    entered=1;
    int a_side=0,b_side=1;
    unsigned alternate=seed^0x4c6f7a21u;
    rc=launch(ctx,stream,"pu_generate",grid((uint64_t)m*k),1,256,1,(void*[]){buffers,&m,&k,&seed,&dataset,&a_side});
    if (!rc) rc=launch(ctx,stream,"pu_generate",grid((uint64_t)n*k),1,256,1,(void*[]){buffers+1,&n,&k,&alternate,&dataset,&b_side});
    leave(ctx); entered=0;
    if (rc) { rc=fail(PU_CUDA,"Benchmark input generation failed (%d)",rc); goto done; }
    PuGemm q={0};
    q.a=(PuMatrix){buffers[0],k,1,0,PU_F16,0}; q.b=(PuMatrix){buffers[1],k,1,0,PU_F16,0};
    q.c=(PuMatrix){buffers[3],n,1,0,PU_F32,0};
    q.m=m; q.n=n; q.k=k; q.batches=1; q.alpha=1.f; q.mode=PU_NATIVE; q.flags=flags;
    PuResult baseline;
    double native_times[9],times[9];
    if ((rc=pu_gemm(ctx,&q,&baseline))) goto done;
    for (int i=0;i<samples;++i) {
        if ((rc=pu_gemm(ctx,&q,&baseline))) goto done;
        native_times[i]=baseline.milliseconds;
    }
    q.c.data=buffers[2]; q.c.type=output_type; q.mode=mode;
    if (samples>1 && mode!=PU_FP16 && (rc=pu_gemm(ctx,&q,&out->run))) goto done;
    for (int i=0;i<samples;++i) {
        if ((rc=pu_gemm(ctx,&q,&out->run))) goto done;
        times[i]=out->run.milliseconds;
    }
    qsort(native_times,samples,sizeof(double),cmp_double);
    qsort(times,samples,sizeof(double),cmp_double);
    out->run.milliseconds=times[samples/2];
    out->run.tflops=out->run.logical_ops/(out->run.milliseconds*1e9);
    out->native_ms=native_times[samples/2]; out->multiplier=out->native_ms/out->run.milliseconds;
    if ((rc=enter(ctx))) goto done;
    entered=1;
    rc=launch(ctx,stream,"pu_compare",blocks,1,256,1,(void*[]){buffers+3,buffers+2,&output_type,&mn,buffers+4});
    leave(ctx); entered=0;
    if (rc) { rc=fail(PU_CUDA,"Comparison CUDA launch failed (%d)",rc); goto done; }
    double *stats=malloc(statb);
    if (!stats) { rc=fail(PU_MEMORY,"Comparison host memory exhausted"); goto done; }
    rc=pu_download(ctx,stats,buffers[4],statb);
    if (!rc) {
        double diff=0,energy=0,maximum=0;
        for (int i=0;i<blocks;++i) { diff+=stats[3*i]; energy+=stats[3*i+1]; maximum=fmax(maximum,stats[3*i+2]); }
        out->error=energy>0?sqrt(diff/energy):diff==0?0:INFINITY;
        out->max_absolute=maximum;
        out->peak_device_bytes=ab+bb+cb+refb+statb+out->run.workspace_bytes;
    }
    free(stats);
done:
    if (entered) leave(ctx);
    for (int i=0;i<5;++i) if (buffers[i]) pu_free(ctx,buffers[i]);
    return rc;
}
