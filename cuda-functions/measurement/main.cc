#include "getMemInfo.cuh"

int main() {
  int result = measurement::printAllGpuMemInfo();

  return 0;
}
