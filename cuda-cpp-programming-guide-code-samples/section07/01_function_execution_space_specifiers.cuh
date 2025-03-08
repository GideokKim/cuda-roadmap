#ifndef FUNCTION_EXECUTION_SPACE_SPECIFIERS_H
#define FUNCTION_EXECUTION_SPACE_SPECIFIERS_H

#include <cuda_runtime.h>

#include <iostream>

// __host__와 __device__ 지정자를 사용하여 호스트와 디바이스 모두에서 실행
// 가능한 함수
__host__ __device__ void exampleFunction(int *data, int size);

// __global__ 지정자를 사용하여 커널 함수 정의
__global__ void kernelFunction(int *data, int size);

// __device__ 지정자를 사용하여 디바이스 전용 함수 정의
__device__ int multiply(int a, int b);

// __host__ __device__ 지정자를 사용하여 호스트와 디바이스 모두에서 실행 가능한
// 함수
__host__ __device__ int add(int a, int b);

// __attribute__((noinline))을 사용한 예시
__attribute__((noinline)) __device__ void noInlineFunction();
__device__ __forceinline__ void forceInlineFunction();

#endif  // FUNCTION_EXECUTION_SPACE_SPECIFIERS_H