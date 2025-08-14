#include <iostream>
#ifdef LESSON_MAIN

// CUDA Kernel function to add 1 to each element
__global__ void helloFromGPU() {
	int threadNum = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello World from GPU thread %d!\n", threadNum);
}
int main() {
    // Launch kernel with 5 threads
    helloFromGPU << <3, 4 >> > ();

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    std::cout << "Hello from CPU!\n";
    return 0;
}

#endif
