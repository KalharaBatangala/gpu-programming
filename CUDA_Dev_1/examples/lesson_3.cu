#ifdef LESSON_PARALLEL_ARRAY_DOUBLING

#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>



// GPU kernel to double each element in an array
__global__ void doubleArray(int* arr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        arr[idx] *= 2;
    }
}

int main()
{
    constexpr int size = 10;
    int bytes = size * sizeof(int);

    // Allocate host memory
    int h_arr[size];
    for (int i = 0; i < size; i++) {
        h_arr[i] = i + 1; // Fill with 1, 2, 3...
    }

    // Allocate device memory
    int* d_arr;
    cudaMalloc(&d_arr, bytes);

    // Copy data from host to device
    cudaMemcpy(d_arr, h_arr, bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    doubleArray << <blocks, threads >> > (d_arr, size);

    // Copy result back to host
    cudaMemcpy(h_arr, d_arr, bytes, cudaMemcpyDeviceToHost);

    // Print results
    std::cout << "Doubled array: ";
    for (int i = 0; i < size; i++) {
        std::cout << h_arr[i] << " ";
    }
    std::cout << "\n";

    // Free device memory
    cudaFree(d_arr);

    return 0;
}



#endif