// TODO(GideokKim): The current build issue related to the CUDA driver API needs
// to be resolved. Comment out until resolved.
// #include <cuda.h>  // Included to use the CUDA driver API

#include <iostream>
#include <string>

#include "cuda-functions/utils/helper.cuh"
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

// TODO(GideokKim): The current build issue related to the CUDA driver API needs
// to be resolved. Comment out until resolved.
// std::vector<GpuMemoryInfo> getGpuMemoryInfo_v2() {
//   cuInit(0);

//   int deviceCount;
//   cuDeviceGetCount(&deviceCount);

//   std::vector<GpuMemoryInfo> memoryInfoList;

//   for (int i = 0; i < deviceCount; ++i) {
//     CUdevice device;
//     cuDeviceGet(&device, i);

//     size_t totalMem = 0;
//     cuDeviceTotalMem(&totalMem, device);

//     size_t freeMem = 0;
//     size_t totalMem2 = 0;
//     CUcontext context;
//     cuCtxCreate(&context, 0, device);
//     cuMemGetInfo(&freeMem, &totalMem2);
//     cuCtxDestroy(context);

//     GpuMemoryInfo info{freeMem, totalMem};

//     memoryInfoList.push_back(info);
//   }

//   return memoryInfoList;
// }

cudaError_t getActivatedGpuMemInfo(size_t* free_mem, size_t* total_mem) {
  return cudaMemGetInfo(free_mem, total_mem);
}

int64_t getAvailableGpuMemory(const GpuMemoryInfo& memInfo,
                              MemoryUsage memUsage) {
  int64_t availableMemory =
      static_cast<int64_t>(memInfo.freeMemory) -
      utils::MinSystemMemory(static_cast<int64_t>(memInfo.freeMemory));
  switch (memUsage) {
    case MemoryUsage::kHigh:
      return static_cast<int64_t>(availableMemory);
    case MemoryUsage::kMedium:
      return static_cast<int64_t>(availableMemory * 0.9);
    case MemoryUsage::kLow:
      return static_cast<int64_t>(availableMemory * 0.8);
  }
}
}  // namespace measurement
