#ifndef CUDA_MEMORY_MANAGEMENT_H
#define CUDA_MEMORY_MANAGEMENT_H

#include <cuda_runtime.h>

// 디바이스 메모리 할당 함수
void allocateDeviceMemory(int **d_data, int size);

// 디바이스 메모리 해제 함수
void freeDeviceMemory(int *d_data);

// 호스트 메모리에서 디바이스 메모리로 데이터 복사
void copyHostToDevice(int *h_data, int *d_data, int size);

// 디바이스 메모리에서 호스트 메모리로 데이터 복사
void copyDeviceToHost(int *d_data, int *h_data, int size);

#endif  // CUDA_MEMORY_MANAGEMENT_H