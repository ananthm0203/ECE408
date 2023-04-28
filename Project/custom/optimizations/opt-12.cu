#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <cuda_fp16.h>

// Optimization 12 (FP16)

#define TILE_SIZE 16

static __half *device_k_fp;
static __half *device_x_fp;

#define cudaErrCheck()                                                      \
{                                                                           \
    cudaError_t error = cudaGetLastError();                                 \
    if(error != cudaSuccess)                                                \
    {                                                                       \
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;    \
        exit(-1);                                                           \
    }                                                                       \
}

__global__ void float2fp16(__half *x_f16, const float* x, const int C, const int H, const int W) {
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define x_f4d(i3, i2, i1, i0) x_f16[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    const int W_grid = (W - 1) / TILE_SIZE + 1;
    auto h = (blockIdx.y / W_grid) * TILE_SIZE + threadIdx.y;
    auto w = (blockIdx.y % W_grid) * TILE_SIZE + threadIdx.x;
    auto c = blockIdx.z;
    auto b = blockIdx.x;
    if (h < H && w < W) {
        x_f4d(b, c, h, w) = __float2half(x4d(b, c, h, w));
    }
#undef x4d
#undef x_f4d
}

__global__ void conv_forward_kernel(float *y, const __half *x, const __half *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    const int W_grid = (W_out - 1) / TILE_SIZE + 1;
    auto h = (blockIdx.y / W_grid) * TILE_SIZE + threadIdx.y;
    auto w = (blockIdx.y % W_grid) * TILE_SIZE + threadIdx.x;
    auto m = blockIdx.z;
    auto b = blockIdx.x;
    if (h < H_out && w < W_out) {
        __half acc = 0;
        for(int c = 0; c < C; c++) 
            for(int p = 0; p < K; p++)
                for (int q = 0; q < K; q++) {
                    auto fp_x = x4d(b, c, h + p, w + q);
                    auto fp_k = k4d(m, c, p, q);
                    acc = __hadd(acc, __hmul(fp_x, fp_k));
                }
        y4d(b, m, h, w) = __half2float(acc);
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
    cudaMalloc(device_k_ptr, M * C * K * K * sizeof(float));

    cudaMemcpy(*device_x_ptr, host_x, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_k_ptr, host_k, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice);


    cudaMalloc(&device_x_fp, B * C * H * W * sizeof(__half));
    cudaMalloc(&device_k_fp, M * C * K * K * sizeof(__half));

    auto device_x = *device_x_ptr;
    auto device_k = *device_k_ptr;

    {
        dim3 nt(TILE_SIZE, TILE_SIZE);
        int W_grid = (W - 1) / TILE_SIZE + 1;
        int H_grid = (H - 1) / TILE_SIZE + 1;
        int Y = W_grid * H_grid;
        dim3 nb(B, Y, C);
        float2fp16<<<nb, nt>>>(device_x_fp, device_x, C, H, W);
    }

    {
        dim3 nt(TILE_SIZE, TILE_SIZE);
        int K_grid = (K - 1) / TILE_SIZE + 1;
        int Y = K_grid * K_grid;
        dim3 nb(M, Y, C);
        float2fp16<<<nb, nt>>>(device_k_fp, device_k, C, K, K);
    }

}


__host__ void GPUInterface::conv_forward_gpu(float *device_y, const float *device_x, const float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Set the kernel dimensions and call the kernel
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int W_grid = (W_out - 1) / TILE_SIZE + 1;
    int H_grid = (H_out - 1) / TILE_SIZE + 1;
    int Y = W_grid * H_grid;
    dim3 nt(TILE_SIZE, TILE_SIZE);
    dim3 nb(B, Y, M);
    conv_forward_kernel<<<nb, nt>>>(device_y, device_x_fp, device_k_fp, B, M, C, H, W, K);
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
    cudaFree(device_k_fp);
    cudaFree(device_x_fp);
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
