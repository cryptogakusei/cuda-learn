#include <iostream>
#include <cuda_runtime.h>


__global__ void copyDataCoalesced(float *in, float *out, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        out[index] = in[n];
    }
}


void initializeArray(float *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = static_cast<float>(i);
    }
}

int main() {
    const int n = 1 << 24;
    float *in, *out;

    cudaMallocManaged(&in, n * sizeof(float));
    cudaMallocManaged(&out, n * sizeof(float));

    initializeArray(in, n);

    int blockSize = 1024;
    int numBlocks = (n + blockSize - 1) / blockSize;

    int minGridSize = 40;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, copyDataCoalesced, 0, 0);

    std::cout << "Recommended block size:" << blockSize
              << ", Minimum grid size:" << minGridSize << std::endl;

    copyDataCoalesced<<<numBlocks, blockSize>>>(in, out, n);
    cudaDeviceSynchronize();

    cudaFree(in);
    cudaFree(out);
    
    return 0;

}