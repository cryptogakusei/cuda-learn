#include <iostream>
#include <cuda.h>

#define BLOCK_DIM 1024

__global__ void scan_kernel(float* input, float* output, float* partialsums, int N) {

    unsigned int segment = blockIdx.x * blockDim.x * 2;

    __shared__ float buffer_s[2 * BLOCK_DIM];
    buffer_s[threadIdx.x] = input[segment + threadIdx.x];
    buffer_s[threadIdx.x + BLOCK_DIM] = input[segment + threadIdx.x + BLOCK_DIM];
    __syncthreads();

    // REDUCTION phase
    for (unsigned int stride = 1; stride <= BLOCK_DIM; stride *= 2) {
        unsigned int i = (threadIdx.x + 1)*2*stride - 1;
        if (i < 2*BLOCK_DIM) {
            buffer_s[i] += buffer_s[i - stride]; 
        }
        __syncthreads();
    }

    // POST_REDUCTION phase
    for (unsigned int stride = BLOCK_DIM/2; stride >= 1; stride /=2) {
        unsigned int i = (threadIdx.x + 1)*2*stride - 1;
        if (i + stride < 2*BLOCK_DIM) {
            buffer_s[i + stride] += buffer_s[i];
        }
        __syncthreads();
    }

    if (threadIdx.x == BLOCK_DIM - 1) {
        partialsums[blockIdx.x] = buffer_s[2*BLOCK_DIM - 1];
    }

    output[segment + threadIdx.x] = buffer_s[threadIdx.x];
    output[segment + BLOCK_DIM + threadIdx.x] = buffer_s[BLOCK_DIM + threadIdx.x];
}



int main() {
    const int size = 8192;
    const int bytes1 = size * sizeof(float);
    const int numBlocks = (size + 2 * BLOCK_DIM - 1) / (2 * BLOCK_DIM);
    const int bytes2 = numBlocks * sizeof(float);

    float* h_input = new float[size];
    float* h_output = new float[size];
    float* h_partialsums = new float[numBlocks];
    for (int i = 0; i < size; i++) {
        h_input[i] = 1.0f; 
    }

    // allocate memory in device
    float* d_input;
    float* d_output;
    float* d_partialsums; 
    cudaMalloc(&d_input, bytes1);
    cudaMalloc(&d_output, bytes1);
    cudaMalloc(&d_partialsums, bytes2);

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, bytes1, cudaMemcpyHostToDevice);

    // Warmup run to initialize CUDA context
    scan_kernel<<<numBlocks, BLOCK_DIM>>>(d_input, d_output, d_partialsums, size);
    cudaDeviceSynchronize();

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start time
    cudaEventRecord(start);

    // Launch the kernel
    scan_kernel<<<numBlocks, BLOCK_DIM>>>(d_input, d_output, d_partialsums, size);

    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, bytes1, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_partialsums, d_partialsums, bytes2, cudaMemcpyDeviceToHost);

    // Print the result
    for (int i = 0; i < numBlocks; i++) {
        std::cout << "Partial Sum is " << h_partialsums[i] << std::endl;
    }

    std::cout << "\nKernel execution time: " << milliseconds << " ms" << std::endl;

    // Cleanup events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Cleanup
    delete[] h_input;
    delete[] h_output;
    delete[] h_partialsums;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_partialsums);

    return 0;
}