#include <stdio.h>
#include "pascal_unleashed.h"

int main() {
    PuDevice device{};
    const int status=pu_device(0,&device);
    if (status!=PU_OK) {
        fprintf(stderr,"%s\n",pu_last_error());
        return status;
    }
    printf("%s, CC %d.%d, %.2f GiB\n",device.name,device.major,device.minor,device.memory/1073741824.0);
    return 0;
}
