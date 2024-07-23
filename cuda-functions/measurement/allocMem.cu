void allocMem() {
  size_t alloc_size = 7000000000;
  int *huge_array;

  if (cudaMallocManaged(&huge_array, alloc_size) == cudaSuccess)
    if (cudaMemset(huge_array, 0, alloc_size) == cudaSuccess) {
      cudaDeviceSynchronize();
      cudaFree(huge_array);
    } else {
      cudaFree(huge_array);
    }
  else
}
