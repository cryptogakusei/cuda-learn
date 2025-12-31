#include <stdio.h>
#include "cuda_runtime.h"

#define CUDACHECK(cmd) do {                         \
    cudaError_t err = cmd;                          \
    if (err != cudaSuccess) {                       \
        printf("Failed: Cuda error %s:%d '%s'\n",   \
            __FILE__,__LINE__,cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                         \
    }                                               \
} while(0)

int main() {
    CUDACHECK(cudaSetDevice(0));
    printf("Success\n");
    return 0;
}
