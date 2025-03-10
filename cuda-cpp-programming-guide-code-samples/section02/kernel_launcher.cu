#include "01_vector_add.cuh"
#include "kernel_launcher.cuh"

void launchKernel(float *A, float *B, float *C, int size) {
  VecAdd<<<1, size>>>(A, B, C);
}