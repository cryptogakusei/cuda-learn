#include <iostream>
#include <cuda.h>

#define BLOCK_DIM 1024

__global__ void scan_double_buffer_kernel(float* input, float* output, float* partialsums, int N) {
    __shared__ float buffer1_s[BLOCK_DIM]; // shared memory
    __shared__ float buffer2_s[BLOCK_DIM]; // shared memory

    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    float* inbuffer_s = buffer1_s;
    float* outbuffer_s = buffer2_s;

    inbuffer_s[threadIdx.x] = input[t];
    __syncthreads();

    for (unsigned int stride = 1; stride <= BLOCK_DIM/2; stride *= 2) {
        if (threadIdx.x >= stride) {
            outbuffer_s[threadIdx.x] = inbuffer_s[threadIdx.x] + inbuffer_s[threadIdx.x-stride];
        } else {
            outbuffer_s[threadIdx.x] = inbuffer_s[threadIdx.x];
        }
        __syncthreads();
        float* temp = inbuffer_s;
        inbuffer_s = outbuffer_s;
        outbuffer_s = temp;
    }

    if (threadIdx.x == BLOCK_DIM - 1) {
        partialsums[blockIdx.x] = inbuffer_s[threadIdx.x];
    }

    output[t] = inbuffer_s[threadIdx.x];    
}


int main() {
    const int size = 2048;
    const int bytes1 = size * sizeof(float);
    const int numBlocks = (size + BLOCK_DIM - 1) / BLOCK_DIM;
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

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start time
    cudaEventRecord(start);

    // Launch the kernel
    scan_double_buffer_kernel<<<numBlocks, BLOCK_DIM>>>(d_input, d_output, d_partialsums, size);

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