#include <iostream>
#include <string>

#include "getMemInfo.cuh"

namespace measurement {

int printAllGpuMemInfo() {
  int num_gpus;
  cudaGetDeviceCount(&num_gpus);

  size_t free_mem, total_mem;
  for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
    cudaSetDevice(gpu_id);
    int id;
    cudaGetDevice(&id);
    std::cout << "Activated GPU ID: " << id << std::endl;
    cudaError_t result = cudaMemGetInfo(&free_mem, &total_mem);
    if (result == cudaSuccess) {
      std::cout << "Free memory: " << free_mem << " BYTE" << std::endl;
      std::cout << "Total memory: " << total_mem << " BYTE" << std::endl;
    } else {
      std::string cudaErrorString(cudaGetErrorString(result));
      std::cerr << "CUDA Error Code: " << result << std::endl;
      std::cerr << "CUDA Error Message: " << cudaErrorString << std::endl;
      return static_cast<int>(result);
    }
  }
  return 0;
}

// cudaError_t getMemInfo(size_t* free_mem, size_t* total_mem) {
//   size_t alloc_size = 40000000000;
//   int* huge_array;

//   if (cudaMallocManaged(&huge_array, alloc_size) == cudaSuccess)
//     if (cudaMemset(huge_array, 0, alloc_size) == cudaSuccess) {
//       cudaError_t result = cudaMemGetInfo(free_mem, total_mem);
//       cudaDeviceSynchronize();
//       cudaFree(huge_array);
//     } else {
//       cudaFree(huge_array);
//     }

//   return result;
// }
}  // namespace measurement
