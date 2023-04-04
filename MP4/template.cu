#include <wb.h>

#define wbCheck(stmt)                                              \
    do                                                             \
    {                                                              \
        cudaError_t err = stmt;                                    \
        if (err != cudaSuccess)                                    \
        {                                                          \
            wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err)); \
            wbLog(ERROR, "Failed to run stmt ", #stmt);            \
            return -1;                                             \
        }                                                          \
    } while (0)

//@@ Define any useful program-wide constants here
#define KSIZE 3
#define PSIZE 1
#define TILE_SIZE 8
#define MSIZE (TILE_SIZE + KSIZE - 1)

//@@ Define constant memory for device kernel here
__constant__ float K[KSIZE][KSIZE][KSIZE];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size)
{
  //@@ Insert kernel code here
  __shared__ float M[MSIZE][MSIZE][MSIZE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  int x_o = blockIdx.x * TILE_SIZE + tx;
  int y_o = blockIdx.y * TILE_SIZE + ty;
  int z_o = blockIdx.z * TILE_SIZE + tz;

  int x_i = x_o - PSIZE;
  int y_i = y_o - PSIZE;
  int z_i = z_o - PSIZE;

  // Load data into shared memory
  if ((x_i >= 0) && (x_i < x_size) &&
    (y_i >= 0) && (y_i < y_size) &&
    (z_i >= 0) && (z_i < z_size))
  {
    M[tz][ty][tx] = input[z_i * (y_size * x_size) + y_i * (x_size) + x_i];
  }
  else
  {
    M[tz][ty][tx] = 0.0f;
  }
  __syncthreads();

  // Perform computation
  if (tx < TILE_SIZE && ty < TILE_SIZE && tz < TILE_SIZE && x_o < x_size && y_o < y_size && z_o < z_size)
  {
    float tmp_sum = 0;
#pragma unroll
    for (int i = 0; i < KSIZE; i++)
    {
#pragma unroll
      for (int j = 0; j < KSIZE; j++)
      {
#pragma unroll
        for (int k = 0; k < KSIZE; k++)
        {
          tmp_sum += K[i][j][k] * M[tz + i][ty + j][tx + k];
        }
      }
    }
    output[z_o * (y_size * x_size) + y_o * (x_size) + x_o] = tmp_sum;
  }
}

int main(int argc, char *argv[])
{
    wbArg_t args;
    int z_size;
    int y_size;
    int x_size;
    int inputLength, kernelLength;
    float *hostInput;
    float *hostKernel;
    float *hostOutput;
    float *deviceInput;
    float *deviceOutput;

    args = wbArg_read(argc, argv);

    // Import data
    hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostKernel =
        (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
    hostOutput = (float *)malloc(inputLength * sizeof(float));

    // First three elements are the input dimensions
    z_size = hostInput[0];
    y_size = hostInput[1];
    x_size = hostInput[2];
    wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
    assert(z_size * y_size * x_size == inputLength - 3);
    assert(kernelLength == KSIZE * KSIZE * KSIZE);

    //@@ Allocate GPU memory here
    // Recall that inputLength is 3 elements longer than the input data
    // because the first  three elements were the dimensions
    cudaMalloc(&deviceInput, (inputLength - 3) * sizeof(float));
    cudaMalloc(&deviceOutput, (inputLength - 3) * sizeof(float));

    //@@ Copy input and kernel to GPU here
    // Recall that the first three elements of hostInput are dimensions and
    // do
    // not need to be copied to the gpu
    // copy input data
    cudaMemcpy(deviceInput, hostInput + 3, (inputLength - 3) * sizeof(float), cudaMemcpyHostToDevice);
    // copy kernel data
    cudaMemcpyToSymbol(K, hostKernel, KSIZE * KSIZE * KSIZE * sizeof(float));

    //@@ Initialize grid and block dimensions here
    dim3 nb(ceil(((float)x_size) / TILE_SIZE), ceil(((float)y_size) / TILE_SIZE), ceil(((float)z_size) / TILE_SIZE));
    dim3 nt(MSIZE, MSIZE, MSIZE);

    //@@ Launch the GPU kernel here
    conv3d<<<nb, nt>>>(deviceInput, deviceOutput, z_size, y_size, x_size);

    cudaDeviceSynchronize();

    //@@ Copy the device memory back to the host here
    // Recall that the first three elements of the output are the dimensions
    // and should not be set here (they are set below)
    cudaMemcpy(hostOutput + 3, deviceOutput, (inputLength - 3) * sizeof(float), cudaMemcpyDeviceToHost);

    // Set the output dimensions for correctness checking
    hostOutput[0] = z_size;
    hostOutput[1] = y_size;
    hostOutput[2] = x_size;
    wbSolution(args, hostOutput, inputLength);

    // Free device memory
    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    // Free host memory
    free(hostInput);
    free(hostOutput);
    return 0;
}