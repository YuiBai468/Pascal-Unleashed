#include <math.h>
#include <stdio.h>
#include "pascal_unleashed.h"

int main(void) {
    PuContext *ctx=NULL;
    uint64_t a=0,b=0,c=0;
    /* Half encodings for A=[1,2;3,4] and B transposed=[5,7;6,8]. */
    const uint16_t lhs[]={0x3c00,0x4000,0x4200,0x4400};
    const uint16_t rhs[]={0x4500,0x4700,0x4600,0x4800};
    float out[4];
    PuResult result;
    int status=pu_open(0,NULL,&ctx);
    if (status) goto done;
    if ((status=pu_alloc(ctx,sizeof lhs,&a)) || (status=pu_alloc(ctx,sizeof rhs,&b)) ||
        (status=pu_alloc(ctx,sizeof out,&c))) goto done;
    if ((status=pu_upload(ctx,a,lhs,sizeof lhs)) || (status=pu_upload(ctx,b,rhs,sizeof rhs))) goto done;
    PuGemm operation={0};
    operation.a=(PuMatrix){a,2,1,0,PU_F16,0};
    operation.b=(PuMatrix){b,2,1,0,PU_F16,0};
    operation.c=(PuMatrix){c,2,1,0,PU_F32,0};
    operation.m=operation.n=operation.k=2;
    operation.batches=1;
    operation.alpha=1;
    operation.mode=PU_AUTO;
    operation.flags=PU_FALLBACK|PU_OUTLIER_GUARD;
    if ((status=pu_gemm(ctx,&operation,&result)) || (status=pu_download(ctx,out,c,sizeof out))) goto done;
    printf("[%g %g; %g %g]  mode=%d fallback=%llu\n",out[0],out[1],out[2],out[3],result.mode,(unsigned long long)result.fallback);
    if (out[0]!=19 || out[1]!=22 || out[2]!=43 || out[3]!=50) status=10;
done:
    if (status) fprintf(stderr,"Pascal Unleashed: %s (%d)\n",pu_last_error(),status);
    if (ctx) { pu_free(ctx,c); pu_free(ctx,b); pu_free(ctx,a); pu_close(ctx); }
    return status;
}
