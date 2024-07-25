#include <unistd.h>

#include <iostream>
#include <stdexcept>

#include "asyncMemManager.cuh"

int main() {
  size_t size = 1024 * 1024 * 1024;  // 1 GB
  memory::AsyncMemManager memoryTest(size);

  for (size_t i = 0; i < 5; ++i) {
    sleep(3);
    std::cout << "sleep 3 seconds" << std::endl;
    memoryTest.MallocMemoryAsync();

    sleep(3);
    std::cout << "sleep 3 seconds" << std::endl;
    memoryTest.FreeMemoryAsync();

    sleep(3);
    std::cout << "sleep 3 seconds" << std::endl;
    memoryTest.SyncronizeStream();
  }

  return 0;
}
