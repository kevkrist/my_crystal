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

  template<int32_t BLOCK_THREADS, int ITEMS_PER_THREAD>
  __device__ __forceinline__ size_t ComputeGlobalItem(int32_t BLOCK_ID, int32_t BLOCK_ITEM) {
    return BLOCK_ID * BLOCK_THREADS * ITEMS_PER_THREAD + BLOCK_ITEM;
  }

} // namespace crystal