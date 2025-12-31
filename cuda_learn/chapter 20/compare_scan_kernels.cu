#include <iostream>
#include <cuda.h>

#define BLOCK_DIM 1024

__global__ void scan_kernel(float* input, float* output, float* partialsums, int N) {
    __shared__ float buffer_s[BLOCK_DIM]; // shared memory
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < N) {
        buffer_s[threadIdx.x] = input[t];
    } else {
        buffer_s[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for (unsigned int stride = 1; stride <= BLOCK_DIM/2; stride *= 2) {
        float temp;
        if (threadIdx.x >= stride) {
            temp = buffer_s[threadIdx.x] + buffer_s[threadIdx.x-stride];
        }
        __syncthreads(); // waiting until all reads have happened for current level
        if (threadIdx.x >= stride) {
            buffer_s[threadIdx.x] = temp;
        }
        __syncthreads(); // waiting until all writes have happened into next level
    }

    if (threadIdx.x == BLOCK_DIM - 1) {
        partialsums[blockIdx.x] = buffer_s[threadIdx.x];
    }

    if (t < N) {
        output[t] = buffer_s[threadIdx.x];
    }
}

__global__ void scan_double_buffer_kernel(float* input, float* output, float* partialsums, int N) {
    __shared__ float buffer1_s[BLOCK_DIM]; // shared memory
    __shared__ float buffer2_s[BLOCK_DIM]; // shared memory

    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    float* inbuffer_s = buffer1_s;
    float* outbuffer_s = buffer2_s;

    if (t < N) {
        inbuffer_s[threadIdx.x] = input[t];
    } else {
        inbuffer_s[threadIdx.x] = 0.0f;
    }
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

    if (t < N) {
        output[t] = inbuffer_s[threadIdx.x];
    }
}

void test_size(int size) {
    const int bytes1 = size * sizeof(float);
    const int numBlocks = (size + BLOCK_DIM - 1) / BLOCK_DIM;
    const int bytes2 = numBlocks * sizeof(float);

    float* h_input = new float[size];
    float* h_output = new float[size];
    float* h_partialsums = new float[numBlocks];

    for (int i = 0; i < size; i++) {
        h_input[i] = 1.0f;
    }

    // Allocate memory in device
    float* d_input;
    float* d_output;
    float* d_partialsums;
    cudaMalloc(&d_input, bytes1);
    cudaMalloc(&d_output, bytes1);
    cudaMalloc(&d_partialsums, bytes2);

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, bytes1, cudaMemcpyHostToDevice);

    // Warmup GPU - run both kernels once without timing
    scan_kernel<<<numBlocks, BLOCK_DIM>>>(d_input, d_output, d_partialsums, size);
    scan_double_buffer_kernel<<<numBlocks, BLOCK_DIM>>>(d_input, d_output, d_partialsums, size);
    cudaDeviceSynchronize();

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Test single buffer kernel
    cudaEventRecord(start);
    scan_kernel<<<numBlocks, BLOCK_DIM>>>(d_input, d_output, d_partialsums, size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_single = 0;
    cudaEventElapsedTime(&time_single, start, stop);

    // Test double buffer kernel
    cudaEventRecord(start);
    scan_double_buffer_kernel<<<numBlocks, BLOCK_DIM>>>(d_input, d_output, d_partialsums, size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_double = 0;
    cudaEventElapsedTime(&time_double, start, stop);

    // Verify results
    cudaMemcpy(h_partialsums, d_partialsums, bytes2, cudaMemcpyDeviceToHost);

    std::cout << "Size: " << size << " (" << numBlocks << " blocks)" << std::endl;
    std::cout << "  Single Buffer: " << time_single << " ms" << std::endl;
    std::cout << "  Double Buffer: " << time_double << " ms" << std::endl;
    std::cout << "  Speedup: " << (time_single / time_double) << "x";
    if (time_double < time_single) {
        std::cout << " (double buffer faster)";
    } else {
        std::cout << " (single buffer faster)";
    }
    std::cout << std::endl << std::endl;

    // Cleanup
    delete[] h_input;
    delete[] h_output;
    delete[] h_partialsums;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_partialsums);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    std::cout << "Comparing Single Buffer vs Double Buffer Scan Kernels" << std::endl;
    std::cout << "======================================================" << std::endl << std::endl;

    // Test various sizes
    test_size(1024);      // 1 block
    test_size(2048);      // 2 blocks
    test_size(4096);      // 4 blocks
    test_size(8192);      // 8 blocks
    test_size(16384);     // 16 blocks
    test_size(32768);     // 32 blocks
    test_size(65536);     // 64 blocks
    test_size(131072);    // 128 blocks
    test_size(262144);    // 256 blocks
    test_size(524288);    // 512 blocks
    test_size(1048576);   // 1024 blocks

    return 0;
}