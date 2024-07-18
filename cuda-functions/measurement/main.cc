#include <iostream>

#include "getMemInfo.cuh"

int main() {
  size_t free_mem = 0;
  size_t total_mem = 0;

  int result = getMemInfo(&free_mem, &total_mem);

  if (result) {
    std::cerr << "Error: " << result << std::endl;
  } else {
    std::cout << "Free memory: " << free_mem / (1024 * 1024) << " MB"
              << std::endl;
    std::cout << "Total memory: " << total_mem / (1024 * 1024) << " MB"
              << std::endl;
  }

  return 0;
}
