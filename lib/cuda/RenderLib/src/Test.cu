#include "Test.cuh"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <string>

__global__ void hello_kernel(int *out, int N)
{
    uint8_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        out[idx] = idx;
}



FUNC_DEF void RenderLibTestFunc()
{
    const int N = 256;
    int *d_arr = nullptr;

    CHECK_CUDA(cudaMalloc(&d_arr, N * sizeof(int)));
    int blockSize = 64;
    int gridSize = (N + blockSize - 1) / blockSize;

    hello_kernel<<<gridSize, blockSize>>>(d_arr, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<int> h_arr(N);
    CHECK_CUDA(cudaMemcpy(h_arr.data(), d_arr, N * sizeof(int), cudaMemcpyDeviceToHost));

    std::cout << "hello() sample: ";
    for (int i = 0; i < 8; ++i) std::cout << h_arr[i] << (i+1<8 ? ", " : "\n");

    cudaFree(d_arr);
}