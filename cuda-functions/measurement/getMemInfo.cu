#include <cuda_runtime.h>

#include "getMemInfo.cuh"

int getMemInfo(size_t* free_mem, size_t* total_mem) {
  cudaError_t result = cudaMemGetInfo(free_mem, total_mem);

  if (result != cudaSuccess) {
    return 1;
  }

  return 0;
}
