#ifndef CUDA_FUNCTIONS_MEASUREMENT_GETMEMINFO_H_
#define CUDA_FUNCTIONS_MEASUREMENT_GETMEMINFO_H_

#include <cuda_runtime.h>

namespace measurement {

int printAllGpuMemInfo();

// cudaError_t getMemInfo(size_t* free_mem, size_t* total_mem);
}  // namespace measurement

#endif  // CUDA_FUNCTIONS_MEASUREMENT_GETMEMINFO_H_
