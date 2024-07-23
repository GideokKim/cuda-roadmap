#include <iostream>

#include "getMemInfo.cuh"

int main() {
  measurement::printGpuMemoryInfo();
  std::vector<measurement::GpuMemoryInfo> gpuMemoryInfoList =
      measurement::getGpuMemoryInfo();

  for (size_t i = 0; i < gpuMemoryInfoList.size(); ++i) {
    uint64_t resultHigh = measurement::getAvailableGpuMemory(
        gpuMemoryInfoList[i], measurement::MemoryUsage::kHigh);

    uint64_t resultMedium = measurement::getAvailableGpuMemory(
        gpuMemoryInfoList[i], measurement::MemoryUsage::kMedium);

    uint64_t resultLow = measurement::getAvailableGpuMemory(
        gpuMemoryInfoList[i], measurement::MemoryUsage::kLow);
    std::cout << "GPU ID " << i << ": " << std::endl;
    std::cout << "  Total Memory:\t\t" << gpuMemoryInfoList[i].totalMemory
              << "\tBYTE" << std::endl;
    std::cout << "  Free Memory:\t\t" << gpuMemoryInfoList[i].freeMemory
              << "\tBYTE" << std::endl;
    std::cout << "  Free Memory(High):\t" << resultHigh << "\tBYTE"
              << std::endl;
    std::cout << "  Free Memory(Medium):\t" << resultMedium << "\tBYTE"
              << std::endl;
    std::cout << "  Free Memory(Low):\t" << resultLow << "\tBYTE" << std::endl;
  }
  return 0;
}
