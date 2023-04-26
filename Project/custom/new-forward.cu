#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

// Optimization 1

#define TILE_SIZE 16

__constant__ float k_const[8192];

#define cudaErrCheck()                                                      \
{                                                                           \
    cudaError_t error = cudaGetLastError();                                 \
    if(error != cudaSuccess)                                                \
    {                                                                       \
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;    \
        exit(-1);                                                           \
    }                                                                       \
}

__global__ void conv_forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    extern __shared__ float x_shared[];
    const auto MSIZE = TILE_SIZE + K - 1;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k_const[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    const int W_grid = (W_out - 1) / TILE_SIZE + 1;
    auto tx = threadIdx.x;
    auto ty = threadIdx.y;
    auto h = (blockIdx.y / W_grid) * TILE_SIZE + ty;
    auto w = (blockIdx.y % W_grid) * TILE_SIZE + tx;
    auto m = blockIdx.z;
    auto b = blockIdx.x;
    // Load tile into shared memory
    float acc = 0;
    for (int c = 0; c < C; ++c) {
        x_shared[ty * MSIZE + tx] = h < H && w < W ? x4d(b, c, h, w) : 0;
        __syncthreads();
        if (tx < TILE_SIZE && ty < TILE_SIZE && h < H_out && w < W_out) {
            for(int p = 0; p < K; p++)
                for (int q = 0; q < K; q++)
                    acc += k4d(m, c, p, q) * x_shared[(ty + p) * MSIZE + (tx + q)];
        }
        __syncthreads();
    }
    if (tx < TILE_SIZE && ty < TILE_SIZE && h < H_out && w < W_out && b < B && m < M) {
        y4d(b, m, h, w) = acc;
    }

#undef y4d
#undef x4d
#undef k4d
}
	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_y, const float *host_x, const float *host_k, float **device_y_ptr, float **device_x_ptr, float **device_k_ptr, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    cudaMalloc(device_y_ptr, B * M * H_out * W_out * sizeof(float));
    cudaMalloc(device_x_ptr, B * C * H * W * sizeof(float));
    // cudaMalloc(device_k_ptr, M * C * K * K * sizeof(float));

    cudaMemcpy(*device_x_ptr, host_x, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(*device_k_ptr, host_k, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(k_const, host_k, M * C * K * K * sizeof(float));

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

}


__host__ void GPUInterface::conv_forward_gpu(float *device_y, const float *device_x, const float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Set the kernel dimensions and call the kernel
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int W_grid = (W_out - 1) / TILE_SIZE + 1;
    int H_grid = (H_out - 1) / TILE_SIZE + 1;
    int Y = W_grid * H_grid;
    auto MSIZE = TILE_SIZE + K - 1;
    auto sharedmem_size = sizeof(float) * (MSIZE * MSIZE);

    dim3 nt(MSIZE, MSIZE);
    dim3 nb(B, Y, M);
    conv_forward_kernel<<<nb, nt, sharedmem_size>>>(device_y, device_x, device_k, B, M, C, H, W, K);
    cudaDeviceSynchronize();
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    // Copy the output back to host
    cudaMemcpy(host_output, device_output, B * M * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
