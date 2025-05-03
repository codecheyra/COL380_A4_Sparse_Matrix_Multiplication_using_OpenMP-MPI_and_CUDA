#include <cuda_runtime.h>
#include <algorithm>
#include "sparse_matrix.hpp"
#include <cassert>

__global__ void mulTasksKernel(
    const ll* __restrict__ Aflat,
    const ll* __restrict__ Bflat,
    const int* __restrict__ tA,
    const int* __restrict__ tB,
    ll*       __restrict__ Cout,
    int k, int KK, int tasksCount
) {
    extern __shared__ ll smem[];
    ll* As = smem;
    ll* Bs = smem + KK;

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y, gx = gridDim.x;
    int taskIdx = by*gx + bx;
    if (taskIdx >= tasksCount) return;

    int ia = tA[taskIdx], ib = tB[taskIdx];
    const ll* a = Aflat + (size_t)ia*KK;
    const ll* b = Bflat + (size_t)ib*KK;
    ll*       c = Cout  + (size_t)taskIdx*KK;

    int tid = ty*blockDim.x + tx, nthreads = blockDim.x*blockDim.y;
    for (int idx = tid; idx < KK; idx += nthreads) {
        As[idx] = a[idx];
        Bs[idx] = b[idx];
    }
    __syncthreads();

    if (ty < k && tx < k) {
        ll sum = 0;
        #pragma unroll
        for (int t = 0; t < k; ++t) {
            sum += As[ty*k + t] * Bs[t*k + tx];
        }
        c[ty*k + tx] = sum;
    }
}

extern "C" void multiplyTasksGPU(
    const ll* h_Aflat, int nA,
    const ll* h_Bflat, int nB,
    const int* h_tA,  const int* h_tB,
    ll*       h_Cout,
    int k, int KK, int tasksCount
) {
    static ll  *d_A=nullptr, *d_B=nullptr, *d_C=nullptr;
    static int *d_tA=nullptr,*d_tB=nullptr;
    static int last_k=0, last_nA=0, last_nB=0, last_T=0;

    size_t sizeA = (size_t)nA * KK * sizeof(ll);
    size_t sizeB = (size_t)nB * KK * sizeof(ll);
    size_t sizeT = (size_t)tasksCount * sizeof(int);
    size_t sizeC = (size_t)tasksCount * KK * sizeof(ll);

    if (k!=last_k || nA!=last_nA || nB!= last_nB) {
        if (d_A) {
            cudaFree(d_A); cudaFree(d_B);
        }
        cudaMalloc(&d_A, sizeA);
        cudaMalloc(&d_B, sizeB);
        last_k = k; last_nA = nA; last_nB = nB;
    }
    if (tasksCount != last_T) {
        if (d_C) {
            cudaFree(d_tA); cudaFree(d_tB); cudaFree(d_C);
        }
        cudaMalloc(&d_tA, sizeT);
        cudaMalloc(&d_tB, sizeT);
        cudaMalloc(&d_C,  sizeC);
        last_T = tasksCount;
    }

    cudaMemcpy(d_A,  h_Aflat, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,  h_Bflat, sizeB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tA, h_tA,    sizeT, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tB, h_tB,    sizeT, cudaMemcpyHostToDevice);

    int gx = std::min(tasksCount, 1024);
    int gy = (tasksCount + gx - 1) / gx;
    dim3 grid(gx, gy), block(k,k);
    size_t sharedBytes = 2 * KK * sizeof(ll);

    mulTasksKernel<<<grid, block, sharedBytes>>>(
      d_A, d_B, d_tA, d_tB, d_C,
      k, KK, tasksCount
    );
    cudaDeviceSynchronize();

    cudaMemcpy(h_Cout, d_C, sizeC, cudaMemcpyDeviceToHost);
}
