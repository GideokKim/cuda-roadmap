#include <iostream>

#include "cuda-cpp-programming-guide-code-samples/utilities/cuda_memory_management.cuh"
#include "kernel_launcher.cuh"

int main() {
  const int size = 5;
  float h_A[size] = {1.0, 2.0, 3.0, 4.0, 5.0};       // 호스트 데이터 A
  float h_B[size] = {10.0, 20.0, 30.0, 40.0, 50.0};  // 호스트 데이터 B
  float h_C[size];                                   // 호스트 결과 데이터
  float *d_A, *d_B, *d_C;                            // 디바이스 데이터 포인터

  // 디바이스 메모리 할당
  allocateDeviceMemory(&d_A, size);
  allocateDeviceMemory(&d_B, size);
  allocateDeviceMemory(&d_C, size);

  // 호스트에서 디바이스로 데이터 복사
  copyHostToDevice(h_A, d_A, size);
  copyHostToDevice(h_B, d_B, size);

  // 커널 실행
  launchKernel(d_A, d_B, d_C, size);  // 커널 호출을 래핑한 함수 사용
  cudaDeviceSynchronize();            // 커널 실행 완료 대기

  // 디바이스에서 호스트로 결과 복사
  copyDeviceToHost(d_C, h_C, size);

  // 결과 출력
  std::cout << "Result C: ";
  for (int i = 0; i < size; i++) {
    std::cout << h_C[i] << " ";  // 결과 출력
  }
  std::cout << std::endl;

  // 디바이스 메모리 해제
  freeDeviceMemory(d_A);
  freeDeviceMemory(d_B);
  freeDeviceMemory(d_C);

  return 0;
}