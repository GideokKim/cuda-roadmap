#include "01_function_execution_space_specifiers.cuh"

// __host__와 __device__ 지정자를 사용하여 호스트와 디바이스 모두에서 실행
// 가능한 함수
__host__ __device__ void exampleFunction(int *data, int size) {
  // 데이터의 합을 계산하는 예시
  int sum = 0;
  for (int i = 0; i < size; ++i) {
    sum += data[i];
  }
  // 결과를 출력 (호스트에서만 출력 가능)
#ifdef __CUDA_ARCH__  // CUDA 아키텍처에서 실행 중일 때
                      // 디바이스에서의 처리 (필요시 추가)
#else                 // 호스트에서 실행 중일 때
  std::cout << "Sum: " << sum << std::endl;
#endif
}

// __global__ 지정자를 사용하여 커널 함수 정의
__global__ void kernelFunction(int *data, int size) {
  // 각 스레드가 데이터의 합을 계산
  exampleFunction(data, size);
}

// __device__ 지정자를 사용하여 디바이스 전용 함수 정의
__device__ int multiply(int a, int b) { return a * b; }

// __host__ __device__ 지정자를 사용하여 호스트와 디바이스 모두에서 실행 가능한
// 함수
__host__ __device__ int add(int a, int b) { return a + b; }

// __noinline__과 __forceinline__을 사용한 예시
__noinline__ __device__ void noInlineFunction() {
  // 이 함수는 인라인되지 않음
}

__forceinline__ __device__ void forceInlineFunction() {
  // 이 함수는 강제로 인라인됨
}
