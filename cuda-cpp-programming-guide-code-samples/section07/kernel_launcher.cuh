#ifndef KERNEL_LAUNCHER_H
#define KERNEL_LAUNCHER_H

#include "01_function_execution_space_specifiers.cuh"

// 커널 실행을 위한 함수 선언
void launchKernel(int *d_data, int size);

#endif  // KERNEL_LAUNCHER_H