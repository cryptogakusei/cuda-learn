// this is a failed code - privatization not exactly optimized and there are thread stalls in loading halos

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel without privatization: Direct global memory access
__global__ void windowSumDirect(const float *input, float *output, int n, int windowSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int halfWindow = windowSize / 2;
    if (idx < n) {
        float sum = 0.0f;
        for (int i = -halfWindow; i <= halfWindow; ++i) {
            int accessIdx = idx + i;
            if (accessIdx >= 0 && accessIdx < n) {
                sum += input[accessIdx];
            }
        }
        output[idx] = sum;
    }
}

// Kernel with privatization: Shared memory access
__global__ void windowSumPrivatized(const float *input, float *output, int n, int windowSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int halfWindow = windowSize / 2;

    extern __shared__ float sharedData[];
    int sharedIdx = threadIdx.x + halfWindow;

    if (idx < n) {
        sharedData[sharedIdx] = input[idx]; 
    }

    if (threadIdx.x < halfWindow) {
        int leftIndex = idx - halfWindow;
        if (leftIndex >= 0) {
            sharedData[threadIdx.x] = input[leftIndex];
        } else {
            sharedData[threadIdx.x] = 0.0f;
        }
    } 

    if (threadIdx.x >= blockDim.x - halfWindow) {
        int rightIndex = idx + halfWindow;
        if (rightIndex < n) {
            sharedData[sharedIdx + halfWindow] = input[rightIndex];
        } else {
            sharedData[sharedIdx + halfWindow] = 0.0f;
        }
    }

    __syncthreads();

    if (idx < n) {
        float sum = 0.0f;
        for (int i = -halfWindow; i <= halfWindow; ++i) {
            sum += sharedData[sharedIdx + i];  
        }
        output[idx] = sum;
    }
}

__global__ void windowSumPrivatizedFixed(const float *input, float *output, int n, int windowSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int halfWindow = windowSize / 2;

    extern __shared__ float sharedData[];
    
    int totalToLoad = blockDim.x + 2 * halfWindow;
    int baseIdx = blockIdx.x * blockDim.x - halfWindow;
    
    for (int i = threadIdx.x; i < totalToLoad; i += blockDim.x) {
        int globalIdx = baseIdx + i;
        if (globalIdx >= 0 && globalIdx < n) {
            sharedData[i] = input[globalIdx];
        } else {
            sharedData[i] = 0.0f;
        }
    }
    
    __syncthreads();
    
    if (idx < n) {
        float sum = 0.0f;
        int localIdx = threadIdx.x + halfWindow;
        
        for (int i = -halfWindow; i <= halfWindow; ++i) {
            sum += sharedData[localIdx + i];
        }
        output[idx] = sum;
    }
}

void initializeArray(float *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = 1.0f; // Simple initialization for demonstration
    }
}

int main() {
    int n = 1<<22; // Example array size
    int windowSize = 129; // Example window size
    float *input, *output;
    float *d_input, *d_output;

    input = (float*)malloc(n * sizeof(float));
    output = (float*)malloc(n * sizeof(float));

    // Initialize input array
    initializeArray(input, n);

    // Allocate device memory
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, input, n * sizeof(float), cudaMemcpyHostToDevice);

    // Setup execution parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    int sharedMemSize = (threadsPerBlock + windowSize) * sizeof(float);


    // Execute kernels
    windowSumDirect<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n, windowSize);
    cudaMemcpy(output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost); // Copy result back for verification

    windowSumPrivatized<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_input, d_output, n, windowSize);
    cudaMemcpy(output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost); // Copy result back for verification

    windowSumPrivatizedFixed<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_input, d_output, n, windowSize);
    cudaMemcpy(output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost); // Copy result back for verification

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    free(input);
    free(output);

    return 0;
}