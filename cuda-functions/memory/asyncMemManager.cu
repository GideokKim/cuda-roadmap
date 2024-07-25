#include <iostream>

#include "asyncMemManager.cuh"

namespace memory {
AsyncMemManager::AsyncMemManager(size_t size)
    : size_(size), d_ptr_(nullptr), stream_(nullptr) {
  // Create a CUDA stream
  cudaError_t result = cudaStreamCreate(&stream_);
  if (result != cudaSuccess) {
    std::cerr << "Failed to create CUDA stream" << result << std::endl;
    std::string cudaErrorString(cudaGetErrorString(result));
    std::cerr << "CUDA Error Code: " << result << std::endl;
    std::cerr << "CUDA Error Message: " << cudaErrorString << std::endl;
  } else {
    std::cout << "succeded to create CUDA stream" << std::endl;
  }
}

AsyncMemManager::~AsyncMemManager() {
  if (d_ptr_ != nullptr) {
    cudaFreeAsync(d_ptr_, stream_);
  }
  if (stream_ != nullptr) {
    cudaStreamDestroy(stream_);
  }
}

void AsyncMemManager::MallocMemoryAsync() {
  if (d_ptr_ == nullptr) {
    cudaError_t result = cudaMallocAsync(&d_ptr_, size_, stream_);
    if (result != cudaSuccess) {
      std::string cudaErrorString(cudaGetErrorString(result));
      std::cerr << "CUDA Error Code: " << result << std::endl;
      std::cerr << "CUDA Error Message: " << cudaErrorString << std::endl;
    } else {
      std::cout << "Allocation complete" << std::endl;
    }
  } else {
    std::cout << "Already allocated" << std::endl;
  }
}

void AsyncMemManager::FreeMemoryAsync() {
  if (d_ptr_ != nullptr) {
    cudaError_t result = cudaFreeAsync(d_ptr_, stream_);
    if (result != cudaSuccess) {
      std::string cudaErrorString(cudaGetErrorString(result));
      std::cerr << "CUDA Error Code: " << result << std::endl;
      std::cerr << "CUDA Error Message: " << cudaErrorString << std::endl;
    } else {
      std::cout << "Free complete" << std::endl;
      d_ptr_ = nullptr;
    }
  } else {
    std::cout << "Already released" << std::endl;
  }
}

void AsyncMemManager::SyncronizeStream() { cudaStreamSynchronize(stream_); }
}  // namespace memory
