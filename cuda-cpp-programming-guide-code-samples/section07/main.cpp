#include <iostream>

#include "cuda-cpp-programming-guide-code-samples/utilities/cuda_memory_management.cuh"
#include "kernel_launcher.cuh"

int main() {
  const int size = 5;
  int h_data[size] = {1, 2, 3, 4, 5};  // 호스트 데이터
  int *d_data;                         // 디바이스 데이터 포인터

  // 디바이스 메모리 할당
  allocateDeviceMemory(&d_data, size);

  // 호스트에서 디바이스로 데이터 복사
  copyHostToDevice(h_data, d_data, size);

  // 커널 실행
  launchKernel(d_data, size);  // 커널 호출을 래핑한 함수 사용
  cudaDeviceSynchronize();     // 커널 실행 완료 대기

  // 디바이스 메모리 해제
  freeDeviceMemory(d_data);

  return 0;
}