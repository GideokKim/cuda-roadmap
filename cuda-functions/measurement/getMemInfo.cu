#include <iostream>
#include <string>

#include "getMemInfo.cuh"

namespace measurement {

void printGpuMemoryInfo() {
  std::vector<GpuMemoryInfo> gpuMemoryInfoList = getGpuMemoryInfo();

  for (size_t i = 0; i < gpuMemoryInfoList.size(); ++i) {
    std::cout << "GPU ID " << i << ": " << std::endl;
    std::cout << "  Total Memory: " << gpuMemoryInfoList[i].totalMemory
              << " BYTE" << std::endl;
    std::cout << "  Free Memory: " << gpuMemoryInfoList[i].freeMemory << " BYTE"
              << std::endl;
  }
}

std::vector<GpuMemoryInfo> getGpuMemoryInfo() {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

  std::vector<GpuMemoryInfo> memoryInfoList;

  for (int gpu_id = 0; gpu_id < deviceCount; ++gpu_id) {
    cudaSetDevice(gpu_id);
    int id;
    cudaGetDevice(&id);
    std::cout << "Activated GPU ID: " << id << std::endl;

    size_t freeMem = 0;
    size_t totalMem = 0;

    cudaError_t result = getActivatedGpuMemInfo(&freeMem, &totalMem);
    if (result == cudaSuccess) {
      GpuMemoryInfo info{freeMem, totalMem};
      memoryInfoList.push_back(info);
    } else {
      std::string cudaErrorString(cudaGetErrorString(result));
      std::cerr << "CUDA Error Code: " << result << std::endl;
      std::cerr << "CUDA Error Message: " << cudaErrorString << std::endl;
    }
  }

  return memoryInfoList;
}

cudaError_t getActivatedGpuMemInfo(size_t* free_mem, size_t* total_mem) {
  return cudaMemGetInfo(free_mem, total_mem);
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
