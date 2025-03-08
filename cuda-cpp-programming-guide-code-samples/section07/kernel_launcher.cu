#include "kernel_launcher.cuh"

void launchKernel(int *d_data, int size) {
  kernelFunction<<<1, 1>>>(d_data, size);
}