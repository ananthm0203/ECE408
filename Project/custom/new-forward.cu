#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_SIZE 16
#define CEILDIV(X, Y) (((X)-1)/(Y)+1)

#define cudaErrCheck()                                                      \
{                                                                           \
    cudaError_t error = cudaGetLastError();                                 \
    if(error != cudaSuccess)                                                \
    {                                                                       \
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;    \
        exit(-1);                                                           \
    }                                                                       \
}

// __constant__ float const_mask[15000];

__global__ void conv_forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int n, m, h, w, c, p, q;
    const int W_grid = ceil(W_out/(float)TILE_SIZE);

    n = blockIdx.x;
    m = blockIdx.y;
    int block_row_start = blockIdx.z / W_grid * TILE_SIZE; 
    int block_col_start = blockIdx.z % W_grid * TILE_SIZE;
    float acc = 0.;
    int input_tile_width = TILE_SIZE + K - 1;
    int row_o = block_row_start + threadIdx.x; 
    int col_o = block_col_start + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    extern __shared__ float shmem[];
    float* X_shared = &shmem[0];
    float* W_shared = &shmem[input_tile_width * input_tile_width];

    for(c = 0; c < C; c++){
        for(int i = tx; i < K; i += TILE_SIZE){
            for(int j = ty; j < K; j += TILE_SIZE){
                W_shared[i * K + j] = k4d(m, c, i, j);
            }
        }
        __syncthreads();

        for(int i = row_o; i < input_tile_width + block_row_start; i += TILE_SIZE){
            for(int j = col_o; j < input_tile_width + block_col_start; j += TILE_SIZE){
                if(i < H && j < W){
                    X_shared[(i - block_row_start) * input_tile_width + j - block_col_start] = x4d(n, c, i, j);
                }
            }
        }
        __syncthreads();

        for(p=0; p<K; p++){
            for (q = 0; q < K; q++){
                acc += X_shared[(threadIdx.x + p) * input_tile_width + threadIdx.y + q] * W_shared[p * K + q];
            }
        }
        __syncthreads();

        if (row_o < H_out && col_o < W_out) {
            y4d(n, m, row_o, col_o) = acc;
        }
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

    cudaMalloc((void**)device_y_ptr, B * M * H_out * W_out * sizeof(float));
    cudaMalloc((void**)device_x_ptr, B * C * H * W * sizeof(float));
    cudaMalloc((void**)device_k_ptr, M * C * K * K * sizeof(float));

    cudaMemcpy(*device_x_ptr, host_x, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_k_ptr, host_k, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice);

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
    int BLOCK_WIDTH = TILE_SIZE + K - 1;
    int W_grid = (W_out + TILE_SIZE - 1) / TILE_SIZE;
    int H_grid = (H_out + TILE_SIZE - 1) / TILE_SIZE;
    int Z = W_grid * H_grid;

    dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH, 1);
    dim3 gridDim(B, M, Z);
    size_t shared_X_size = BLOCK_WIDTH * BLOCK_WIDTH * sizeof(float);
    conv_forward_kernel<<<gridDim, blockDim, shared_X_size>>>(device_y, device_x, device_k, B, M, C, H, W, K);
    cudaDeviceSynchronize();
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    cudaMemcpy(host_output, device_output, Batch * Channel * Height * Width * sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
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
