/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <algorithm>

namespace utils {
int64_t MinSystemMemory(int64_t available_memory) {
  // We use the following heuristic for now:
  //
  // If the available_memory is < 2GiB, we allocate 225MiB to system memory,
  // Otherwise, allocate max(300MiB, kMinSystemMemoryFraction *
  // available_memory) to system memory.
  //
  // In the future we could be more sophisticated by using a table of devices.
  int64_t min_system_memory;
  constexpr float kMinSystemMemoryFraction = 0.06;
  if (available_memory < (1LL << 31)) {
    // 225MiB
    min_system_memory = 255 * 1024 * 1024;
  } else {
    // max(300 MiB, kMinSystemMemoryFraction * available_memory)
    min_system_memory = std::max(
        int64_t{314572800},
        static_cast<int64_t>(available_memory * kMinSystemMemoryFraction));
  }
}
}  // namespace utils
