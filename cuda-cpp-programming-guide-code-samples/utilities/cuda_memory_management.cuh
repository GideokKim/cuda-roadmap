#ifndef CUDA_MEMORY_MANAGEMENT_H
#define CUDA_MEMORY_MANAGEMENT_H

#include <cuda_runtime.h>

// 디바이스 메모리 할당 함수
template <typename T>
void allocateDeviceMemory(T **d_data, int size);

// 디바이스 메모리 해제 함수
template <typename T>
void freeDeviceMemory(T *d_data);

// 호스트 메모리에서 디바이스 메모리로 데이터 복사
template <typename T>
void copyHostToDevice(T *h_data, T *d_data, int size);

// 디바이스 메모리에서 호스트 메모리로 데이터 복사
template <typename T>
void copyDeviceToHost(T *d_data, T *h_data, int size);

#endif  // CUDA_MEMORY_MANAGEMENT_H