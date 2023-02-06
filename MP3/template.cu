
#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define TILESIZE  16
#define CEILDIV(A, B)   ((A - 1) / B + 1)

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  __shared__ float M1[TILESIZE][TILESIZE];
  __shared__ float M2[TILESIZE][TILESIZE];
  
  auto bx = blockIdx.x;
  auto tx = threadIdx.x;
  auto by = blockIdx.y;
  auto ty = threadIdx.y;

  auto row = bx * blockDim.x + tx;
  auto col = by * blockDim.y + ty;

  float tmpSum = 0;

  for(size_t i = 0; i < CEILDIV(numAColumns, TILESIZE); ++i)
  {
    auto x = i * TILESIZE + tx;
    auto y = i * TILESIZE + ty;

    // Copy data into shared memory
    if (row < numCRows && y < numAColumns)
    {
      M1[tx][ty] = A[row * numAColumns + y];
    }
    else
    {
      M1[tx][ty] = 0;
    }
    if (col < numCColumns && x < numBRows)
    {
      M2[tx][ty] = B[x * numBColumns + col];
    }
    else
    {
      M2[tx][ty] = 0;
    }

    __syncthreads();

#pragma unroll
    for (size_t k = 0; k < TILESIZE; ++k)
    {
      tmpSum += M1[tx][k] * M2[k][ty];
    }

    __syncthreads();
  }

  if (row < numCRows && col < numCColumns)
  {
    C[row * numCColumns + col] = tmpSum;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  wbTime_stop(Generic, "Importing data and creating memory on host");
  hostC = (float *)malloc(sizeof(float) * numCRows * numCColumns);

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc(&deviceA, sizeof(float) * numARows * numAColumns);
  cudaMalloc(&deviceB, sizeof(float) * numBRows * numBColumns);
  cudaMalloc(&deviceC, sizeof(float) * numCRows * numCColumns);

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, sizeof(float) * numARows * numAColumns, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, sizeof(float) * numBRows * numBColumns, cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 nt(TILESIZE, TILESIZE);
  dim3 nb(CEILDIV(numCRows, nt.x), CEILDIV(numCColumns, nt.y));

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiply <<<nb, nt>>> (deviceA, deviceB, deviceC,
    numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, sizeof(float) * numCRows * numCColumns, cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
