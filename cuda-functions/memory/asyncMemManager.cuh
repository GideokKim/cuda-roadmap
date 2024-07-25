#ifndef CUDA_FUNCTIONS_MEMORY_MANAGER_H_
#define CUDA_FUNCTIONS_MEMORY_MANAGER_H_

#include <cuda_runtime.h>

namespace memory {
class AsyncMemManager {
 public:
  AsyncMemManager() = delete;
  AsyncMemManager(size_t size);

  ~AsyncMemManager();

  void MallocMemoryAsync();
  void FreeMemoryAsync();
  void SyncronizeStream();

 private:
  size_t size_;
  char* d_ptr_;
  cudaStream_t stream_;
};
}  // namespace memory

#endif  // CUDA_FUNCTIONS_MEMORY_MANAGER_H_
