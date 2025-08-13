#include stdio.h>
#include <cuda_runtime.h>
#include <iostream>

#if LESSON_PARALLEL_ARRAY_DOUBLING

__global__ void doubleArray(int* arr, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

}