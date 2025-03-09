#include <iostream>

#include "cuda_memory_management.cuh"

void allocateDeviceMemory(int **d_data, int size) {
  cudaError_t err = cudaMalloc(d_data, size * sizeof(int));
  if (err != cudaSuccess) {
    std::cerr << "Error allocating device memory: " << cudaGetErrorString(err)
              << std::endl;
  }
}

void freeDeviceMemory(int *d_data) { cudaFree(d_data); }

void copyHostToDevice(int *h_data, int *d_data, int size) {
  cudaError_t err =
      cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "Error copying from host to device: "
              << cudaGetErrorString(err) << std::endl;
  }
}

void copyDeviceToHost(int *d_data, int *h_data, int size) {
  cudaError_t err =
      cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    std::cerr << "Error copying from device to host: "
              << cudaGetErrorString(err) << std::endl;
  }
}