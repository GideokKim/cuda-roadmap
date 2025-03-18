#ifndef OCCUPANCY_CALCULATOR_H
#define OCCUPANCY_CALCULATOR_H

#include <cuda_runtime.h>

__global__ void MyKernel(int *d, int *a, int *b);

void occupancyCalculator();

void computeClusters();

#endif  // OCCUPANCY_CALCULATOR_H