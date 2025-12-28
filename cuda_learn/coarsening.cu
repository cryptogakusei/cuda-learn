#include <stdio.h>

#define N (1<<24)
#define THREADS_PER_BLOCK 256

__global__ void VecAdd(float* A, float* B, float* C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

__global__ void VecAddCoarsened(float* A, float* B, float* C) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i < N) {
        C[i] = A[i] + B[i];
    } 
    if (i+1 < N) {
        C[i + 1] = A[i + 1] + B[i + 1];
    }
}

void random_init(float* data, int size) {
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}


int main() {
    float *a, *b, *c;
    float *d_a, *d_b, *d_c;

    int size = N * sizeof(float);

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    a = (float *)malloc(size); random_init(a, N);
    b = (float *)malloc(size); random_init(b, N);
    c = (float *)malloc(size);

    cudaEvent_t start, stop, startCoarsened, stopCoarsened;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&startCoarsened);
    cudaEventCreate(&stopCoarsened);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    VecAdd<<<(N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();

    VecAddCoarsened<<<(N + 2*THREADS_PER_BLOCK - 1)/(2*THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    VecAdd<<<(N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("VecAdd execution time: %f ms\n", milliseconds);

    cudaEventRecord(startCoarsened);
    VecAddCoarsened<<<(N + 2*THREADS_PER_BLOCK - 1)/(2*THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(d_a, d_b, d_c);
    cudaEventRecord(stopCoarsened);
    cudaEventSynchronize(stopCoarsened);
    float millisecondsCoarsened = 0;
    cudaEventElapsedTime(&millisecondsCoarsened, startCoarsened, stopCoarsened);
    printf("VecAddCoarsened execution time: %f ms\n", millisecondsCoarsened);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(a); free(b); free(c);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(startCoarsened);
    cudaEventDestroy(stopCoarsened);
}