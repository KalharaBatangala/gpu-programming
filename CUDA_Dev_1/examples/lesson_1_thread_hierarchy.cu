#include <stdio.h>  // For printf
#include <cuda_runtime.h> // CUDA runtime API (optional, but good practice)
#include <iostream> 

#if LESSON_THREAD_HIERARCHY

// CUDA kernel function to print thread information
__global__ void printThreadIds() {
    // Calculate unique global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Print the thread's global ID
    printf("Hello from thread %d (block %d, thread %d)\n", idx, blockIdx.x, threadIdx.x);
}

int main() {
    // Launch kernel with 3 blocks, each having 4 threads
    printThreadIds << <3, 4 >> > ();

    // Wait for GPU to finish before CPU continues
    cudaDeviceSynchronize();
	std::cout << "Kernel launched, waiting for completion..." << std::endl;

    return 0;
}

#endif
