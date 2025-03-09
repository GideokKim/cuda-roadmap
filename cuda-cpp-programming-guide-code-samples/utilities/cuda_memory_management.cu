#include <iostream>

#include "cuda_memory_management.cuh"

template <typename T>
void allocateDeviceMemory(T **d_data, int size) {
  cudaError_t err = cudaMalloc(d_data, size * sizeof(T));
  if (err != cudaSuccess) {
    std::cerr << "Error allocating device memory: " << cudaGetErrorString(err)
              << std::endl;
  }
}

template <typename T>
void freeDeviceMemory(T *d_data) {
  cudaFree(d_data);
}

template <typename T>
void copyHostToDevice(T *h_data, T *d_data, int size) {
  cudaError_t err =
      cudaMemcpy(d_data, h_data, size * sizeof(T), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "Error copying from host to device: "
              << cudaGetErrorString(err) << std::endl;
  }
}

template <typename T>
void copyDeviceToHost(T *d_data, T *h_data, int size) {
  cudaError_t err =
      cudaMemcpy(h_data, d_data, size * sizeof(T), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    std::cerr << "Error copying from device to host: "
              << cudaGetErrorString(err) << std::endl;
  }
}

// 템플릿 인스턴스화
template void allocateDeviceMemory<int>(int **d_data, int size);
template void freeDeviceMemory<int>(int *d_data);
template void copyHostToDevice<int>(int *h_data, int *d_data, int size);
template void copyDeviceToHost<int>(int *d_data, int *h_data, int size);

template void allocateDeviceMemory<float>(float **d_data, int size);
template void freeDeviceMemory<float>(float *d_data);
template void copyHostToDevice<float>(float *h_data, float *d_data, int size);
template void copyDeviceToHost<float>(float *d_data, float *h_data, int size);