#pragma once

namespace crystal
{
  // ENUMS
  enum class DataArrangement
  {
    Striped,
    Blocked
  };

  enum class LoopType
  {
    Direct,
    Guarded,
    Flagged
  };

  // Helper functions
  template<int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD, DataArrangement ArrangementT>
  __device__ __forceinline__ int32_t ComputeBlockItem(int32_t THREAD_ITEM) {
    if constexpr(ArrangementT == DataArrangement::Striped) {
      return threadIdx.x + THREAD_ITEM * BLOCK_THREADS;
    } else { // Blocked
      return THREAD_ITEM + threadIdx.x * ITEMS_PER_THREAD;
    }
  }

} // namespace crystal