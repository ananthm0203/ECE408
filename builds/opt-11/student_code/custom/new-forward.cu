#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

// Optimization 11

#define TILE_SIZE 16
#define NUM_STREAMS 10
#define OVERLAP 3

__constant__ float k_const[8192];
cudaStream_t streams[NUM_STREAMS];

float *_host_x, *_host_y;

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
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k_const[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    const int W_grid = (W_out - 1) / TILE_SIZE + 1;
    auto h = (blockIdx.y / W_grid) * TILE_SIZE + threadIdx.y;
    auto w = (blockIdx.y % W_grid) * TILE_SIZE + threadIdx.x;
    auto m = blockIdx.z;
    auto b = blockIdx.x;
    if (h < H_out && w < W_out) {
        float acc = 0.;
        for(int c = 0; c < C; c++) 
            for(int p = 0; p < K; p++)
                for (int q = 0; q < K; q++)
                    acc += x4d(b, c, h + p, w + q) * k4d(m, c, p, q);
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

    // cudaMemcpy(*device_x_ptr, host_x, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(k_const, host_k, M * C * K * K * sizeof(float));

    _host_x = (float *)host_x;
    _host_y = (float *)host_y;
    
    for (int i = 0; i < NUM_STREAMS; i++)
        cudaStreamCreate(streams + i);

    // cudaFree(device_mask);

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

    int x_batch_size = (B * C * H * W) / NUM_STREAMS;
    int y_batch_size = (B * M * H_out * W_out) / NUM_STREAMS;

    dim3 nt(TILE_SIZE, TILE_SIZE);
    dim3 nb(B / NUM_STREAMS, Y, M);

    constexpr auto N_OVERLAPS = NUM_STREAMS / OVERLAP;
    constexpr auto N_LEFTOVER = NUM_STREAMS - (N_OVERLAPS * OVERLAP);

#pragma unroll
    for (int i = 0; i < N_OVERLAPS * OVERLAP; i += OVERLAP) {
#pragma unroll
        for (int j = 0; j < OVERLAP; j++) {
            int x_offset = x_batch_size * (i + j);
            cudaMemcpyAsync((float *)(device_x) + x_offset, (float *)(_host_x)  + x_offset, sizeof(float) * x_batch_size, cudaMemcpyHostToDevice, streams[i+j]);
        }
#pragma unroll
        for (int j = 0; j < OVERLAP; j++) {
            int x_offset = x_batch_size * (i + j);
            int y_offset = y_batch_size * (i + j);
            conv_forward_kernel<<<nb, nt, 0, streams[i+j]>>>((float *)(device_y) + y_offset, (float *)(device_x) + x_offset, device_k, B, M, C, H, W, K);
        }
#pragma unroll
        for (int j = 0; j < OVERLAP; j++) {
            int y_offset = y_batch_size * (i + j);
            cudaMemcpyAsync((float *)(_host_y) + y_offset, (float *)(device_y) + y_offset, sizeof(float) * y_batch_size, cudaMemcpyDeviceToHost, streams[i+j]);
        }
    }
    // Complete any leftover streams
#pragma unroll
    for (int i = 0; i < N_LEFTOVER; i++) {
        int x_offset = x_batch_size * (i + N_OVERLAPS * OVERLAP);
        cudaMemcpyAsync((float *)(device_x) + x_offset, (float *)(_host_x)  + x_offset, sizeof(float) * x_batch_size, cudaMemcpyHostToDevice, streams[i + N_OVERLAPS * OVERLAP]);
    }
#pragma unroll
    for (int i = 0; i < N_LEFTOVER; i++) {
        int x_offset = x_batch_size * (i + N_OVERLAPS * OVERLAP);
        int y_offset = y_batch_size * (i + N_OVERLAPS * OVERLAP);
        conv_forward_kernel<<<nb, nt, 0, streams[i + N_OVERLAPS * OVERLAP]>>>((float *)(device_y) + y_offset, (float *)(device_x) + x_offset, device_k, B, M, C, H, W, K);
    }
#pragma unroll
    for (int i = 0; i < N_LEFTOVER; i++) {
        int y_offset = y_batch_size * (i + N_OVERLAPS * OVERLAP);
        cudaMemcpyAsync((float *)(_host_y) + y_offset, (float *)(device_y) + y_offset, sizeof(float) * y_batch_size, cudaMemcpyDeviceToHost, streams[i + N_OVERLAPS * OVERLAP]);
    }

    cudaDeviceSynchronize();
    return;
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // const int H_out = H - K + 1;
    // const int W_out = W - K + 1;
    // // Copy the output back to host
    // cudaMemcpy(host_output, device_output, B * M * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < NUM_STREAMS; i++)
        cudaStreamDestroy(streams[i]);

    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    return;
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
