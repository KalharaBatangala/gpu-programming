#include <iostream>

// CUDA Kernel function to add 1 to each element
__global__ void helloFromGPU() {
    printf("Hello World from GPU thread %d!\n", threadIdx.x);
}

int main() {
    // Launch kernel with 5 threads
    helloFromGPU << <1, 5 >> > ();

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    std::cout << "Hello from CPU!\n";
    return 0;
}
