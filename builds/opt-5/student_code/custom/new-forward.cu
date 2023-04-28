#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

// Baseline

#define TILE_SIZE_X 16
#define TILE_SIZE_Y 16

__constant__ float const_k[8192];

#define cudaErrCheck()                                                      \
{                                                                           \
    cudaError_t error = cudaGetLastError();                                 \
    if(error != cudaSuccess)                                                \
    {                                                                       \
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;    \
        exit(-1);                                                           \
    }                                                                       \
}

__global__ void conv_forward_kernel(float * __restrict__ y, const float * __restrict__ x, const float * __restrict__ k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) const_k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    const int W_grid = (W_out - 1) / TILE_SIZE_X + 1;
    auto tx = threadIdx.x;
    auto ty = threadIdx.y;
    auto h = (blockIdx.y / W_grid) * TILE_SIZE_Y + ty;
    auto w = (blockIdx.y % W_grid) * TILE_SIZE_X + tx;
    auto m = blockIdx.z;
    auto b = blockIdx.x;
    if (h < H_out && w < W_out) {
        float acc = 0.;
        for (int c = 0; c < C; c++) {
            acc += x4d(b, c, h + 0, w + 0) * k4d(m, c, 0, 0);
            acc += x4d(b, c, h + 0, w + 1) * k4d(m, c, 0, 1);
            acc += x4d(b, c, h + 0, w + 2) * k4d(m, c, 0, 2);
            acc += x4d(b, c, h + 0, w + 3) * k4d(m, c, 0, 3);
            acc += x4d(b, c, h + 0, w + 4) * k4d(m, c, 0, 4);
            acc += x4d(b, c, h + 0, w + 5) * k4d(m, c, 0, 5);
            acc += x4d(b, c, h + 0, w + 6) * k4d(m, c, 0, 6);

            acc += x4d(b, c, h + 1, w + 0) * k4d(m, c, 1, 0);
            acc += x4d(b, c, h + 1, w + 1) * k4d(m, c, 1, 1);
            acc += x4d(b, c, h + 1, w + 2) * k4d(m, c, 1, 2);
            acc += x4d(b, c, h + 1, w + 3) * k4d(m, c, 1, 3);
            acc += x4d(b, c, h + 1, w + 4) * k4d(m, c, 1, 4);
            acc += x4d(b, c, h + 1, w + 5) * k4d(m, c, 1, 5);
            acc += x4d(b, c, h + 1, w + 6) * k4d(m, c, 1, 6);

            acc += x4d(b, c, h + 2, w + 0) * k4d(m, c, 2, 0);
            acc += x4d(b, c, h + 2, w + 1) * k4d(m, c, 2, 1);
            acc += x4d(b, c, h + 2, w + 2) * k4d(m, c, 2, 2);
            acc += x4d(b, c, h + 2, w + 3) * k4d(m, c, 2, 3);
            acc += x4d(b, c, h + 2, w + 4) * k4d(m, c, 2, 4);
            acc += x4d(b, c, h + 2, w + 5) * k4d(m, c, 2, 5);
            acc += x4d(b, c, h + 2, w + 6) * k4d(m, c, 2, 6);

            acc += x4d(b, c, h + 3, w + 0) * k4d(m, c, 3, 0);
            acc += x4d(b, c, h + 3, w + 1) * k4d(m, c, 3, 1);
            acc += x4d(b, c, h + 3, w + 2) * k4d(m, c, 3, 2);
            acc += x4d(b, c, h + 3, w + 3) * k4d(m, c, 3, 3);
            acc += x4d(b, c, h + 3, w + 4) * k4d(m, c, 3, 4);
            acc += x4d(b, c, h + 3, w + 5) * k4d(m, c, 3, 5);
            acc += x4d(b, c, h + 3, w + 6) * k4d(m, c, 3, 6);

            acc += x4d(b, c, h + 4, w + 0) * k4d(m, c, 4, 0);
            acc += x4d(b, c, h + 4, w + 1) * k4d(m, c, 4, 1);
            acc += x4d(b, c, h + 4, w + 2) * k4d(m, c, 4, 2);
            acc += x4d(b, c, h + 4, w + 3) * k4d(m, c, 4, 3);
            acc += x4d(b, c, h + 4, w + 4) * k4d(m, c, 4, 4);
            acc += x4d(b, c, h + 4, w + 5) * k4d(m, c, 4, 5);
            acc += x4d(b, c, h + 4, w + 6) * k4d(m, c, 4, 6);

            acc += x4d(b, c, h + 5, w + 0) * k4d(m, c, 5, 0);
            acc += x4d(b, c, h + 5, w + 1) * k4d(m, c, 5, 1);
            acc += x4d(b, c, h + 5, w + 2) * k4d(m, c, 5, 2);
            acc += x4d(b, c, h + 5, w + 3) * k4d(m, c, 5, 3);
            acc += x4d(b, c, h + 5, w + 4) * k4d(m, c, 5, 4);
            acc += x4d(b, c, h + 5, w + 5) * k4d(m, c, 5, 5);
            acc += x4d(b, c, h + 5, w + 6) * k4d(m, c, 5, 6);

            acc += x4d(b, c, h + 6, w + 0) * k4d(m, c, 6, 0);
            acc += x4d(b, c, h + 6, w + 1) * k4d(m, c, 6, 1);
            acc += x4d(b, c, h + 6, w + 2) * k4d(m, c, 6, 2);
            acc += x4d(b, c, h + 6, w + 3) * k4d(m, c, 6, 3);
            acc += x4d(b, c, h + 6, w + 4) * k4d(m, c, 6, 4);
            acc += x4d(b, c, h + 6, w + 5) * k4d(m, c, 6, 5);
            acc += x4d(b, c, h + 6, w + 6) * k4d(m, c, 6, 6);
        }
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
    cudaMemcpyToSymbol(const_k, host_k, M * C * K * K * sizeof(float));

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
    int W_grid = (W_out - 1) / TILE_SIZE_X + 1;
    int H_grid = (H_out - 1) / TILE_SIZE_Y + 1;
    int Y = W_grid * H_grid;

    dim3 nt(TILE_SIZE_X, TILE_SIZE_Y);
    dim3 nb(B, Y, M);
    conv_forward_kernel<<<nb, nt>>>(device_y, device_x, device_k, B, M, C, H, W, K);
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
    // cudaFree(device_mask);
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
