#ifdef TEST

#include <iostream>

__global__ void helloFromGPU() {
	printf("\nHello from GPU");
}

int main() {
	helloFromGPU << <1, 4 >> > ();
	cudaDeviceSynchronize();

	std::cout << "\nHello from CPU!" << std::endl;

	return 0;
}

#endif