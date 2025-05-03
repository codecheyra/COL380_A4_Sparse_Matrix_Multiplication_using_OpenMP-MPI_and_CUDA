// File: cuda_kernels.cu

#include <cuda_runtime.h>
#include "sparse_matrix.hpp"
#include <cassert>

__global__ void mulBlockKernel(
    const ll *a, const ll *b, ll *c,
    int k, int rows, int cols)
{
    int tx = threadIdx.x, ty = threadIdx.y;
    if (ty < rows && tx < cols)
    {
        ll sum = 0;
        for (int t = 0; t < k; ++t)
            sum += a[ty * k + t] * b[t * k + tx];
        // __device__ unsigned long long atomicAdd(unsigned long long* address, unsigned long long val);
        // inside mulBlockKernel:
        auto idx = ty * k + tx;
        atomicAdd(
            reinterpret_cast<unsigned long long *>(&c[idx]),
            static_cast<unsigned long long>(sum));
    }
}

void multiplyBlockGPU(
    const ll *h_a, const ll *h_b, ll *h_c,
    int k, int rows, int cols)
{
    size_t S = k * k * sizeof(ll);
    ll *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, S);
    cudaMalloc(&d_b, S);
    cudaMalloc(&d_c, S);
    cudaMemcpy(d_a, h_a, S, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, S, cudaMemcpyHostToDevice);
    cudaMemset(d_c, 0, S);

    dim3 blockDim(k, k);
    mulBlockKernel<<<1, blockDim>>>(d_a, d_b, d_c, k, rows, cols);
    cudaDeviceSynchronize();

    cudaMemcpy((void *)h_c, d_c, S, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
