// Histogram Equalization

#include <wb.h>
#include <stdio.h>
#include <cmath>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 32
#define SCAN_SIZE (HISTOGRAM_LENGTH >> 1)

typedef unsigned char uchar;

__device__ float cdfmin;
__device__ float hist[HISTOGRAM_LENGTH];

__global__ void partial_hist(float *im_rgb, int *glob_hist_buf, int H, int W) {
  __shared__ int part_hist[HISTOGRAM_LENGTH];
  auto h = blockIdx.x * blockDim.x + threadIdx.x;
  auto w = blockIdx.y * blockDim.y + threadIdx.y;

  for (int i = threadIdx.x * blockDim.y + threadIdx.y; i < HISTOGRAM_LENGTH; i += blockDim.x * blockDim.y) {
    part_hist[i] = 0;
  }
  __syncthreads();

  if (w < W && h < H) {
    auto r = im_rgb[(h * W + w) * 3];
    auto g = im_rgb[(h * W + w) * 3 + 1];
    auto b = im_rgb[(h * W + w) * 3 + 2];
    auto gr = (uchar)(0.21*255*r + 0.71*255*g + 0.07*255*b);
    atomicAdd(&(part_hist[gr]), 1);
  }
  __syncthreads();

  auto g = blockIdx.x + blockIdx.y * gridDim.x;
  for (int i = threadIdx.x * blockDim.y + threadIdx.y; i < HISTOGRAM_LENGTH; i += blockDim.x * blockDim.y) {
    glob_hist_buf[g * HISTOGRAM_LENGTH + i] = part_hist[i];
  }
}

__global__ void accum_hist(int *glob_hist_buf, int n, int H, int W) {
  int i = threadIdx.x;
  int total = 0;
    for (int j = 0; j < n; ++j) {
      total += glob_hist_buf[j * HISTOGRAM_LENGTH + i];
    }
    hist[i] = ((float)total) / (H * W);
}

__global__ void hist_scan() {
  __shared__ float T[2 * SCAN_SIZE];
  auto bx = blockIdx.x;
  auto tx = threadIdx.x;
  auto idx = bx * 2 * SCAN_SIZE + tx;

  // Load data into shared memory
  T[tx] = idx < HISTOGRAM_LENGTH ? hist[idx] : 0;
  T[tx + SCAN_SIZE] = idx + SCAN_SIZE < HISTOGRAM_LENGTH ? hist[idx + SCAN_SIZE] : 0;

  // Perform hist_scan
  int idx2;
  for (int s = 1; s < 2 * SCAN_SIZE; s *= 2) {
    __syncthreads();
    idx2 = (tx + 1) * s * 2 - 1;
    if (idx2 < 2 * SCAN_SIZE && (idx2 - s) >= 0)
      T[idx2] += T[idx2 - s];
  }

  // Post reduction step
  for (int s = SCAN_SIZE / 2; s > 0; s /= 2) {
    __syncthreads();
    idx2 = (tx + 1) * s * 2 - 1;
    if (idx2 + s < 2 * SCAN_SIZE)
      T[idx2 + s] += T[idx2];
  }

  __syncthreads();
  
  // Write to input array as output
  if (idx < HISTOGRAM_LENGTH) hist[idx] = T[tx];
  if (idx + SCAN_SIZE < HISTOGRAM_LENGTH) hist[idx + SCAN_SIZE] = T[tx + SCAN_SIZE];
}

__global__ void set_cdfmin() {
  for (int i = 0; i < HISTOGRAM_LENGTH; ++i) {
    if (hist[i] != 0.) cdfmin = hist[i];
    break;
  }
}

__global__ void hist_eq(float *im_rgb, int H, int W) {
  auto h = blockIdx.x * blockDim.x + threadIdx.x;
  auto w = blockIdx.y * blockDim.y + threadIdx.y;
  if (w < W && h < H) {
    for (int c = 0; c < 3; ++c) {
      auto val = (uchar)(im_rgb[(h * W + w) * 3 + c] * 255);
      im_rgb[(h * W + w) * 3 + c] = fmin(fmax((hist[val] - cdfmin) / (1.0 - cdfmin), 0.), 1.);
    }
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;
  float *deviceRGB;
  int *local_hists;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  auto W = (imageWidth - 1) / BLOCK_SIZE + 1;
  auto H = (imageHeight - 1) / BLOCK_SIZE + 1;

  cudaMalloc(&deviceRGB, sizeof(float) * imageWidth * imageHeight * imageChannels);
  cudaMalloc(&local_hists, sizeof(int) * HISTOGRAM_LENGTH * W * H);

  cudaMemcpy(deviceRGB, hostInputImageData, sizeof(float) * imageWidth * imageHeight * imageChannels, cudaMemcpyHostToDevice);

  dim3 nb(H, W);
  dim3 nt(BLOCK_SIZE, BLOCK_SIZE);
  partial_hist<<<nb, nt>>>(deviceRGB, local_hists, imageHeight, imageWidth);
  accum_hist<<<1, HISTOGRAM_LENGTH>>>(local_hists, H * W, imageHeight, imageWidth);
  hist_scan<<<1, SCAN_SIZE>>>();
  set_cdfmin<<<1, 1>>>();
  hist_eq<<<nb, nt>>>(deviceRGB, imageHeight, imageWidth);

  cudaMemcpy(hostOutputImageData, deviceRGB, sizeof(float) * imageWidth * imageHeight * imageChannels, cudaMemcpyDeviceToHost);

  wbSolution(args, outputImage);

  cudaFree(deviceRGB);
  cudaFree(local_hists);

  return 0;
}
