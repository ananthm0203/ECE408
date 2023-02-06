// MP 1
// #include <wb.h>

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  
  auto tx = threadIdx.x;
  auto bx = blockIdx.x;
  auto idx = bx * blockDim.x + tx;

  if (idx < len)
  {
    out[idx] = in1[idx] + in2[idx];
  }
}

size_t read_file(FILE* f, float** buf)
{
  size_t len;
  fscanf(f, "%lu", &len);
  *buf = (float *)malloc(sizeof(float) * len);
  float elem;
  for (size_t i = 0; i < len; ++i)
  {
    fscanf(f, "%f", &elem);
    (*buf)[i] = elem;
  }
  return len;
}

int main(int argc, char **argv) {
  // wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;

  // args = wbArg_read(argc, argv);

  // wbTime_start(Generic, "Importing data and creating memory on host");
  // hostInput1 =
  //     (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  // hostInput2 =
  //     (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  // hostOutput = (float *)malloc(inputLength * sizeof(float));
  // wbTime_stop(Generic, "Importing data and creating memory on host");

  FILE* in1 = fopen(argv[2], "r");
  FILE* in2 = fopen(argv[3], "r");

  inputLength = read_file(in1, &hostInput1);
  read_file(in2, &hostInput2);

  hostOutput = (float *)malloc(sizeof(float) * inputLength);

  fclose(in1);
  fclose(in2);

  // wbLog(TRACE, "The input length is ", inputLength);

  // wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here

  cudaMalloc(&deviceInput1, inputLength * sizeof(float));
  cudaMalloc(&deviceInput2, inputLength * sizeof(float));
  cudaMalloc(&deviceOutput, inputLength * sizeof(float));

  // wbTime_stop(GPU, "Allocating GPU memory.");

  // wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here

  cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(float), cudaMemcpyHostToDevice);

  // wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here

  dim3 nb((inputLength - 1) / 32 + 1);
  dim3 nt(32);

  // wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here

  vecAdd <<< nb, nt >>> (deviceInput1, deviceInput2, deviceOutput, inputLength);

  // cudaDeviceSynchronize();
  // wbTime_stop(Compute, "Performing CUDA computation");

  // wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here

  cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(float), cudaMemcpyDeviceToHost);

  // wbTime_stop(Copy, "Copying output memory to the CPU");

  // wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here

  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  // wbTime_stop(GPU, "Freeing GPU Memory");

  // wbSolution(args, hostOutput, inputLength);

  float* soln;

  FILE* solnf = fopen(argv[1], "r");

  read_file(solnf, &soln);

  for (int i = 0; i < inputLength; ++i)
  {
    printf("%i: %f vs. %f\n", i, hostOutput[i], soln[i]);
  }

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  free(soln);

  return 0;
}
