#include <iostream>

#include "02_occupancy_calculator.cuh"

__global__ void MyKernel(int *d, int *a, int *b) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  d[idx] = a[idx] * b[idx];
}

void occupancyCalculator() {
  int numBlocks;  // Occupancy in terms of active blocks
  int blockSize = 32;

  // These variables are used to convert occupancy to warps
  int device;
  cudaDeviceProp prop;
  int activeWarps;
  int maxWarps;

  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);

  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, MyKernel, blockSize,
                                                0);

  activeWarps = numBlocks * blockSize / prop.warpSize;
  maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

  std::cout << "Occupancy: " << (double)activeWarps / maxWarps * 100 << "%\n";
}

void computeClusters() {
  cudaLaunchConfig_t config = {0};
  config.gridDim = 1024;
  config.blockDim = 128;  // threads_per_block = 128
  config.dynamicSmemBytes = 0;

  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim.x = 2;  // cluster_size = 2
  attribute[0].val.clusterDim.y = 1;
  attribute[0].val.clusterDim.z = 1;
  config.attrs = attribute;
  config.numAttrs = 1;

  int max_cluster_size = 0;
  cudaOccupancyMaxPotentialClusterSize(&max_cluster_size, (void *)MyKernel,
                                       &config);

  int max_active_clusters = 0;
  cudaOccupancyMaxActiveClusters(&max_active_clusters, (void *)MyKernel,
                                 &config);

  std::cout << "Max Active Clusters of size 2: " << max_active_clusters
            << std::endl;
}